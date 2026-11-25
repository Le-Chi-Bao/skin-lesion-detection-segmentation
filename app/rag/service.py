"""Session-owned corpus lifecycle and a stable retrieval interface for WP6B."""

from copy import deepcopy
from threading import RLock
from time import perf_counter

from app.rag.chunking import RecursiveChunker
from app.rag.importer import DocumentImporter
from app.rag.models import ModelUnavailable, RAGError, SearchResponse
from app.rag.retrieval import BM25Retriever, DenseRetriever, HybridRetriever


class RAGService:
    def __init__(self, *, chunker=None, embedder=None, rrf_k: float = 60, candidates: int = 30):
        self.chunker = chunker or RecursiveChunker()
        self.bm25 = BM25Retriever()
        self.dense = DenseRetriever(embedder)
        self.hybrid = HybridRetriever(self.bm25, self.dense, rrf_k=rrf_k, candidates=candidates)
        self._documents, self._receipts, self._sources = {}, {}, {}
        self._suppressed = set()
        self._dense_ids = None
        self._lock = RLock()

    @property
    def document_count(self):
        return len(self._documents)

    def close(self):
        with self._lock:
            self._documents.clear()
            self._receipts.clear()
            self._sources.clear()
            self._invalidate()
            directory = getattr(self, "_demo_directory", None)
            if directory is not None:
                directory.cleanup()
                self._demo_directory = None

    def ingest(self, manifest_path, importer: DocumentImporter):
        with self._lock:
            key = str(manifest_path.absolute())
            result = importer.import_manifest(manifest_path)
            # Invalid replacement fails closed: evict all documents owned by that receipt.
            ids = {identity for identity, receipt in self._receipts.items() if receipt == key}
            ids.update(result.observed_ids)
            for identity in ids:
                self._documents.pop(identity, None)
                self._receipts.pop(identity, None)
            if result.status != "INVALID":
                self._sources[key] = (manifest_path, importer)
                for document in result.documents:
                    self._suppressed.discard(document.document_id)
                    self._documents[document.document_id] = document
                    self._receipts[document.document_id] = key
            self._invalidate()
            return result

    def remove_document(self, document_id: str):
        with self._lock:
            self._suppressed.add(document_id)
            self._documents.pop(document_id, None)
            self._receipts.pop(document_id, None)
            self._invalidate()

    def _invalidate(self):
        self.bm25.index(())
        self.dense.clear()
        self._dense_ids = None

    def _refresh(self):
        # Reverify receipts, artifacts, provenance and current rights before every query.
        # Never resurrect an older batch for a document updated through a newer receipt.
        documents = {}
        for key, (path, importer) in self._sources.items():
            result = importer.import_manifest(path)
            for doc in result.documents:
                if (self._receipts.get(doc.document_id) == key
                        and doc.document_id not in self._suppressed and importer.rights.allows(doc)):
                    documents[doc.document_id] = doc
        self._documents = documents
        chunks = tuple(c for identity in sorted(documents) for c in self.chunker.chunk(documents[identity]))
        self.bm25.index(chunks)
        identities = tuple(sorted(c.chunk_id for c in chunks))
        if identities != self._dense_ids:
            self.dense.clear()
            self._dense_ids = None
        elif self.dense.chunks:
            # Vectors remain valid when only the live rights evidence snapshot changes.
            self.dense.chunks = tuple(sorted(chunks, key=lambda chunk: chunk.chunk_id))
        return chunks, identities

    def search(self, query: str, mode: str = "Hybrid", top_k: int = 5) -> SearchResponse:
        start = perf_counter()
        with self._lock:
            if mode not in {"BM25", "Dense", "Hybrid"}:
                raise RAGError("invalid_retrieval_mode")
            if type(top_k) is not int or not 1 <= top_k <= 100:
                raise RAGError("invalid_top_k")
            chunks, identities = self._refresh()
            synthetic = any(d.synthetic for d in self._documents.values()) or (
                not self._documents and any(importer.rights.synthetic for _, importer in self._sources.values())
            )
            hits, status = (), "OK"
            if not query.strip():
                status = "EMPTY_QUERY"
            elif not chunks:
                status = "EMPTY_INDEX"
            else:
                try:
                    if mode in {"Dense", "Hybrid"} and self._dense_ids != identities:
                        self.dense.index(chunks)
                        self._dense_ids = identities
                    retriever = {"BM25": self.bm25, "Dense": self.dense, "Hybrid": self.hybrid}[mode]
                    hits = retriever.search(query, top_k)
                    # A final live rights check also covers revocation during embedding.
                    hits = tuple(h for h in hits if self._sources[self._receipts[h.chunk.document_id]][1]
                                 .rights.allows(self._documents[h.chunk.document_id]))
                    if not hits:
                        status = "NO_MATCH"
                except ModelUnavailable as exc:
                    status = str(exc)
                except RAGError as exc:
                    status = str(exc)
                except Exception:
                    status = "retrieval_failed"
            # Never return mutable references to the trusted internal corpus/citation data.
            return SearchResponse(deepcopy(hits), mode, (perf_counter() - start) * 1000, status, synthetic)
