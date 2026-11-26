"""Ordered embeddings and a bounded, session-owned in-memory vector cache."""

import hashlib
import json
from dataclasses import asdict
from uuid import uuid4

import numpy as np

from app.rag.cloud.client import routing
from app.rag.models import RAGError


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode("utf-8")).hexdigest()


class OpenRouterEmbedder:
    def __init__(self, config, client):
        self.config, self.client = config, client
        self.authorize = self._unbound
        self.namespace = uuid4().hex
        self.dimension = config.dimensions
        self._observed_provider = None
        self._cache, self._prepared = {}, {}
        self._fingerprint = self.fingerprint

    @staticmethod
    def _unbound():
        raise RAGError("outbound_authorization_required")

    @property
    def fingerprint(self):
        return digest({"embedding": asdict(self.config), "endpoint": "/embeddings",
                       "routing": routing(self.config.provider), "normalization": "l2-v1"})

    @property
    def metadata(self):
        return {"model": self.config.model, "provider": self.config.provider,
                "reported_provider": self._observed_provider,
                "dimension": self.dimension, "fingerprint": self.fingerprint,
                "corpus": self.namespace, "cached_vectors": len(self._cache)}

    def clear_cache(self):
        self._cache.clear()
        self._prepared.clear()

    def prepare_index(self, chunks, corpus):
        if self.fingerprint != self._fingerprint or self.namespace != corpus:
            self.clear_cache()
            self.dimension = self.config.dimensions
            self._observed_provider = None
            self._fingerprint, self.namespace = self.fingerprint, corpus
        prepared = {}
        for chunk in chunks:
            content = digest(chunk.text)
            identity = digest([self.namespace, self.fingerprint, chunk.chunk_id, content,
                               chunk.citation["chunk_config"], chunk.citation["record_sha256"]])
            # Repeated content with different provenance may reuse a vector only within this corpus.
            prepared.setdefault(content, identity)
        self._prepared = prepared
        allowed = set(prepared.values())
        self._cache = {key: entry for key, entry in self._cache.items() if key[0] in allowed}

    def _encode(self, texts, input_type):
        if not texts:
            return np.empty((0, self.dimension or 0))
        self.client.validate_input(texts)
        if any(len(t.encode("utf-8")) > self.config.max_input_bytes for t in texts):
            raise RAGError("embedding_input_limit_exceeded_rechunk_required")
        batch_size = self.config.batch_size
        self.client.require_capacity((len(texts) + batch_size - 1) // batch_size)
        vectors = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            payload = {"model": self.config.model, "input": batch, "encoding_format": "float",
                       "provider": routing(self.config.provider)}
            if self.config.dimensions is not None:
                payload["dimensions"] = self.config.dimensions
            if input_type is not None:
                payload["input_type"] = input_type
            data = self.client.post("/embeddings", payload, timeout=self.config.request_timeout,
                                    authorize=self.authorize)
            if data.get("model") != self.config.model:
                raise RAGError("embedding_model_mismatch")
            actual_provider = data.get("provider")
            if actual_provider is not None:
                if not isinstance(actual_provider, str) or not actual_provider.strip():
                    raise RAGError("embedding_provider_mismatch")
                if (self._observed_provider is not None
                        and actual_provider.casefold() != self._observed_provider.casefold()):
                    raise RAGError("embedding_provider_mismatch")
                self._observed_provider = actual_provider
            rows = data.get("data")
            if not isinstance(rows, list) or len(rows) != len(batch):
                raise RAGError("invalid_embeddings_count")
            ordered = {}
            for row in rows:
                if not isinstance(row, dict):
                    raise RAGError("invalid_embeddings")
                index, vector = row.get("index"), row.get("embedding")
                if (type(index) is not int or not 0 <= index < len(batch) or index in ordered
                        or not isinstance(vector, list) or not vector
                        or any(type(v) not in (int, float) for v in vector)):
                    raise RAGError("invalid_embeddings")
                try:
                    array = np.asarray(vector, dtype=np.float64)
                    norm = np.linalg.norm(array)
                except (ValueError, TypeError, OverflowError):
                    raise RAGError("invalid_embeddings") from None
                if not np.isfinite(array).all() or not np.isfinite(norm) or norm <= 0:
                    raise RAGError("invalid_embeddings")
                if self.dimension is None:
                    self.dimension = len(vector)
                if len(vector) != self.dimension:
                    raise RAGError("embedding_dimension_mismatch")
                ordered[index] = array / norm
            vectors.extend(ordered[i] for i in range(len(batch)))
        return np.asarray(vectors)

    def embed_documents(self, texts):
        self.authorize()  # Cached content still requires current permission.
        if len(texts) > self.client.settings.max_index_chunks:
            raise RAGError("cloud_index_chunk_budget_exceeded")
        if self.fingerprint != self._fingerprint:
            self.clear_cache()
            self.dimension = self.config.dimensions
            self._observed_provider = None
            self._fingerprint = self.fingerprint
        identities = [self._prepared.get(digest(t), digest([self.namespace, self.fingerprint, digest(t)]))
                      for t in texts]
        missing = {}
        for key, text in zip(identities, texts):
            if (key, self.dimension, self._observed_provider) not in self._cache:
                missing.setdefault(key, text)
        encoded = self._encode([self.config.passage_prefix + t for t in missing.values()],
                               self.config.document_input_type)
        self.authorize()  # Do not retain vectors produced after a revocation.
        for key, vector in zip(missing, encoded):
            self._cache[(key, self.dimension, self._observed_provider)] = {"dimension": self.dimension, "vector": vector}
        # Standalone callers cannot grow the session cache without bound either.
        if len(self._cache) > self.client.settings.max_index_chunks:
            keep = {(key, self.dimension, self._observed_provider) for key in identities}
            self._cache = {key: value for key, value in self._cache.items() if key in keep}
        return np.asarray([self._cache[(key, self.dimension, self._observed_provider)]["vector"] for key in identities])

    def embed_query(self, query):
        return self._encode([self.config.query_prefix + query], self.config.query_input_type)[0]
