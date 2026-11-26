"""Persistent local Chroma retrieval; live service corpus controls all search visibility."""

import hashlib
import json
from pathlib import Path

from app.rag.models import RAGError, SearchHit
from app.rag.retrieval import DenseRetriever


class ChromaDenseRetriever(DenseRetriever):
    def __init__(self, embedder, path):
        super().__init__(embedder)
        self.path = Path(path)
        self._collection = None
        self.embedded_count = self.reused_count = 0

    @property
    def collection(self):
        if self._collection is None:
            import chromadb
            from chromadb.config import Settings

            self._client = chromadb.PersistentClient(path=str(self.path), settings=Settings(anonymized_telemetry=False))
            self._collection = self._client.get_or_create_collection(
                name="medical-e5-" + self.embedder.persistent_fingerprint[:32], embedding_function=None,
                metadata={"embedding_fingerprint": self.embedder.persistent_fingerprint},
                configuration={"hnsw": {"space": "cosine"}})
        return self._collection

    def index(self, chunks):
        import numpy as np

        self.clear()
        ordered = tuple(sorted({c.chunk_id: c for c in chunks}.values(), key=lambda c: c.chunk_id))
        self.embedded_count = self.reused_count = 0
        if not ordered:
            return
        # Reuse embeddings by exact text hash even when a source/heading/consent changes.
        hashes = {c.chunk_id: hashlib.sha256(c.text.encode("utf-8")).hexdigest() for c in ordered}
        cache = {}
        unique = sorted(set(hashes.values()))
        for start in range(0, len(unique), 128):
            rows = self.collection.get(where={"content_sha256": {"$in": unique[start:start + 128]}},
                                       include=["embeddings", "metadatas", "documents"])
            for meta, vector, text in zip(rows["metadatas"], rows["embeddings"], rows["documents"]):
                if hashlib.sha256(text.encode("utf-8")).hexdigest() != meta["content_sha256"]:
                    raise RAGError("persistent_vector_content_mismatch")
                normalized = self._normalized([vector])
                if normalized.shape[1] != self.embedder.config.dimensions:
                    raise RAGError("embedding_dimension_mismatch")
                cache[meta["content_sha256"]] = normalized[0]
        missing = {}
        for chunk in ordered:
            key = hashes[chunk.chunk_id]
            if key not in cache:
                missing[key] = chunk.text
        keys = list(missing)
        batch_size = self.embedder.config.batch_size
        for start in range(0, len(keys), batch_size):
            batch = keys[start:start + batch_size]
            vectors = self._normalized(self.embedder.embed_documents([missing[k] for k in batch]))
            if vectors.shape != (len(batch), self.embedder.config.dimensions):
                raise RAGError("embedding_dimension_mismatch")
            cache.update(zip(batch, vectors))
            self.embedded_count += len(batch)
        self.reused_count = len(ordered) - self.embedded_count
        for start in range(0, len(ordered), 128):
            batch = ordered[start:start + 128]
            self.collection.upsert(
                ids=[c.chunk_id for c in batch], documents=[c.text for c in batch],
                embeddings=np.asarray([cache[hashes[c.chunk_id]] for c in batch]).tolist(),
                metadatas=[{"chunk_id": c.chunk_id, "document_id": c.document_id,
                            "content_sha256": hashes[c.chunk_id], "pdf_page": c.citation.get("pdf_page", 0),
                            "source": c.citation["source"]["locator"], "title": c.citation["title"] or "",
                            "citation_json": json.dumps(c.citation, ensure_ascii=False)} for c in batch])
        self.chunks = ordered

    def search(self, query, top_k=5):
        if not self.chunks or not query.strip() or top_k <= 0:
            return ()
        vector = self._normalized([self.embedder.embed_query(query)])[0]
        current = {c.chunk_id: c for c in self.chunks}
        # Old/revoked versions may stay on disk; they are never eligible candidates.
        rows = self.collection.query(query_embeddings=[vector.tolist()],
                                     n_results=min(top_k, len(current)),
                                     where={"chunk_id": {"$in": sorted(current)}},
                                     include=["distances"])
        hits = [SearchHit(current[key], 1 - float(distance), "Dense")
                for key, distance in zip(rows["ids"][0], rows["distances"][0])]
        return tuple(sorted(hits, key=lambda h: (-h.score, h.chunk.chunk_id)))
