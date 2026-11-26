"""R1 BM25, R2 dense cosine, R3 reciprocal rank fusion. All indexes stay local."""

import math
import re
import unicodedata
import hashlib
import json
from dataclasses import asdict
from functools import cached_property
from pathlib import Path
from collections import Counter

from app.rag.config import BM25Config, EmbeddingConfig
from app.rag.models import Chunk, ModelUnavailable, RAGError, SearchHit


def tokenize(text: str) -> list[str]:
    # Unicode normalization and case folding retain Vietnamese accents. No English-only stemmer.
    return re.findall(r"\w+", unicodedata.normalize("NFC", text).casefold(), flags=re.UNICODE)


class BM25Retriever:
    def __init__(self, chunks=(), *, k1: float = 1.5, b: float = 0.75):
        BM25Config(k1=k1, b=b)
        self.k1, self.b = k1, b
        self.index(chunks)

    def index(self, chunks):
        self.chunks = tuple(sorted({c.chunk_id: c for c in chunks}.values(), key=lambda c: c.chunk_id))
        self.terms = [Counter(tokenize(c.text)) for c in self.chunks]
        self.lengths = [sum(t.values()) for t in self.terms]
        self.average = sum(self.lengths) / len(self.lengths) if self.lengths else 0
        self.df = Counter(term for terms in self.terms for term in terms)

    def search(self, query: str, top_k: int = 5) -> tuple[SearchHit, ...]:
        if top_k <= 0:
            return ()
        hits = []
        for chunk, terms, length in zip(self.chunks, self.terms, self.lengths):
            score = 0.0
            for term in sorted(set(tokenize(query))):
                frequency = terms[term]
                if frequency == 0:
                    continue
                df = self.df[term]
                idf = math.log(1 + (len(self.chunks) - df + 0.5) / (df + 0.5))
                denominator = frequency + self.k1 * (1 - self.b + self.b * length / self.average)
                score += idf * frequency * (self.k1 + 1) / denominator
            if score > 0:
                hits.append(SearchHit(chunk, score, "BM25"))
        return tuple(sorted(hits, key=lambda h: (-h.score, h.chunk.chunk_id))[:top_k])


class LocalE5Embedder:
    """Lazy transformer loading, masked mean pooling and L2 normalization per E5 card.

    Custom models must support this pooling recipe; configure their prefixes explicitly.
    No remote code, paid endpoint, automatic download, or silent passage truncation.
    """

    def __init__(self, config: EmbeddingConfig | None = None):
        self.config = config or EmbeddingConfig()
        self._model = self._tokenizer = None

    def _load(self):
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            if self.config.device == "cuda" and not torch.cuda.is_available():
                raise ModelUnavailable("embedding_cuda_unavailable")
            options = {"local_files_only": not self.config.allow_download, "trust_remote_code": False}
            tokenizer = AutoTokenizer.from_pretrained(self.config.model, **options)
            model = AutoModel.from_pretrained(self.config.model, use_safetensors=True, **options)
            model.to(self.config.device).eval()
            self._tokenizer, self._model = tokenizer, model
        except ModelUnavailable:
            raise
        except ImportError:
            raise ModelUnavailable("embedding_dependencies_missing") from None
        except Exception:
            raise ModelUnavailable("embedding_model_unavailable_local") from None

    @cached_property
    def persistent_fingerprint(self):
        config = asdict(self.config)
        config.pop("batch_size")
        config.pop("device")
        files = {}
        root = Path(self.config.model)
        if not root.is_dir():
            raise ModelUnavailable("persistent_embedding_requires_local_model_directory")
        for path in sorted(root.glob("*")):
            if path.is_file() and path.suffix in {".json", ".model", ".safetensors"}:
                with path.open("rb") as stream:
                    files[path.name] = hashlib.file_digest(stream, "sha256").hexdigest()
        return hashlib.sha256(json.dumps([config, files, "masked-mean-l2-v1"], sort_keys=True).encode()).hexdigest()

    def _encode(self, texts: list[str], prefix: str, *, query: bool):
        self._load()
        import numpy as np
        import torch
        import torch.nn.functional as functional

        batches = []
        for offset in range(0, len(texts), self.config.batch_size):
            inputs = [prefix + t.strip() for t in texts[offset:offset + self.config.batch_size]]
            if not query:
                lengths = self._tokenizer(inputs, truncation=False, padding=False)["input_ids"]
                if any(len(ids) > self.config.max_length for ids in lengths):
                    raise RAGError("passage_exceeds_model_tokens_rechunk_required")
            options = {"max_length": self.config.max_length} if query else {}
            batch = self._tokenizer(inputs, padding=True, truncation=query,
                                    return_tensors="pt", **options).to(self.config.device)
            with torch.inference_mode():
                hidden = self._model(**batch).last_hidden_state
                mask = batch["attention_mask"]
                hidden = hidden.masked_fill(~mask[..., None].bool(), 0.0)
                pooled = hidden.sum(dim=1) / mask.sum(dim=1)[..., None]
                batches.append(functional.normalize(pooled, p=2, dim=1).cpu().numpy())
        result = np.concatenate(batches) if batches else np.empty((0, 0), dtype=np.float32)
        if len(texts) and self.config.dimensions is not None and result.shape[1] != self.config.dimensions:
            raise RAGError("embedding_dimension_mismatch")
        return result

    def embed_documents(self, texts: list[str]):
        return self._encode(texts, self.config.passage_prefix, query=False)

    def embed_query(self, query: str):
        return self._encode([query], self.config.query_prefix, query=True)[0]


class DenseRetriever:
    def __init__(self, embedder=None):
        self.embedder = embedder or LocalE5Embedder()
        self.clear()

    def clear(self):
        self.chunks = ()
        self.vectors = None

    @staticmethod
    def _normalized(values):
        import numpy as np

        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2 or not np.isfinite(values).all():
            raise RAGError("invalid_embeddings")
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        if (norms == 0).any():
            raise RAGError("invalid_embeddings")
        return values / norms

    def index(self, chunks):
        ordered = tuple(sorted({c.chunk_id: c for c in chunks}.values(), key=lambda c: c.chunk_id))
        # Clear old versions before any embedding attempt, including a failed attempt.
        self.clear()
        if ordered:
            vectors = self._normalized(self.embedder.embed_documents([c.text for c in ordered]))
            if vectors.shape[0] != len(ordered):
                raise RAGError("invalid_embeddings")
            self.chunks, self.vectors = ordered, vectors

    def search(self, query: str, top_k: int = 5) -> tuple[SearchHit, ...]:
        if not self.chunks or not query.strip() or top_k <= 0:
            return ()
        vector = self._normalized([self.embedder.embed_query(query)])
        if vector.shape[1] != self.vectors.shape[1]:
            raise RAGError("invalid_embeddings")
        scores = (self.vectors @ vector[0]).tolist()
        return tuple(sorted((SearchHit(c, float(s), "Dense") for c, s in zip(self.chunks, scores)),
                            key=lambda h: (-h.score, h.chunk.chunk_id))[:top_k])


class RRFusion:
    def __init__(self, k: float = 60):
        if type(k) not in {int, float} or not math.isfinite(k) or k <= 0:
            raise ValueError("invalid_rrf_k")
        self.k = k

    def fuse(self, rankings, top_k: int = 5) -> tuple[SearchHit, ...]:
        if top_k <= 0:
            return ()
        scores, chunks = {}, {}
        for ranking in rankings:
            seen = set()
            for hit in ranking:
                identity = hit.chunk.chunk_id
                if identity in seen:
                    continue
                seen.add(identity)
                scores[identity] = scores.get(identity, 0.0) + 1 / (self.k + len(seen))
                chunks[identity] = hit.chunk
        return tuple(SearchHit(chunks[key], scores[key], "Hybrid")
                     for key in sorted(scores, key=lambda key: (-scores[key], key))[:top_k])


class HybridRetriever:
    def __init__(self, bm25: BM25Retriever, dense: DenseRetriever, *, rrf_k: float = 60, candidates: int = 30):
        if type(candidates) is not int or candidates < 1:
            raise ValueError("invalid_candidate_count")
        self.bm25, self.dense = bm25, dense
        self.fusion, self.candidates = RRFusion(rrf_k), candidates

    def search(self, query: str, top_k: int = 5) -> tuple[SearchHit, ...]:
        count = max(top_k, self.candidates)
        return self.fusion.fuse((self.bm25.search(query, count), self.dense.search(query, count)), top_k)
