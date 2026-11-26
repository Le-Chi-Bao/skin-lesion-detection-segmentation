"""Session-owned corpus lifecycle and a stable retrieval interface for WP6B."""

from copy import deepcopy
import re
from uuid import uuid4
from threading import RLock
from time import perf_counter

from app.rag.chunking import RecursiveChunker
from app.rag.importer import DocumentImporter
from app.rag.models import ModelUnavailable, RAGError, SearchResponse
from app.rag.config import RAGConfig
from app.rag.retrieval import BM25Retriever, DenseRetriever, HybridRetriever


class RAGService:
    def __init__(self, *, chunker=None, embedder=None, rrf_k: float = 60, candidates: int = 30,
                 bm25_k1: float = 1.5, bm25_b: float = 0.75, settings=None,
                 cloud_client=None, reranker=None, generator=None):
        self.chunker = chunker or RecursiveChunker()
        self.bm25 = BM25Retriever(k1=bm25_k1, b=bm25_b)
        self.dense = DenseRetriever(embedder)
        self.hybrid = HybridRetriever(self.bm25, self.dense, rrf_k=rrf_k, candidates=candidates)
        self._documents, self._receipts, self._sources = {}, {}, {}
        self._suppressed = set()
        self._dense_ids = None
        self._lock = RLock()
        self.settings = settings or RAGConfig()
        self.cloud_client, self.reranker, self.generator = cloud_client, reranker, generator
        self._corpus_id = uuid4().hex
        self._dense_fingerprint = None

    @property
    def document_count(self):
        return len(self._documents)

    def close(self):
        with self._lock:
            self._documents.clear()
            self._receipts.clear()
            self._sources.clear()
            self._invalidate()
            if self.cloud_client is not None:
                self.cloud_client.close()
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
        if hasattr(self.dense.embedder, "clear_cache"):
            self.dense.embedder.clear_cache()

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
        fingerprint = getattr(self.dense.embedder, "fingerprint", None)
        if identities != self._dense_ids or fingerprint != self._dense_fingerprint:
            self.dense.clear()
            self._dense_ids = None
        elif self.dense.chunks:
            # Vectors remain valid when only the live rights evidence snapshot changes.
            self.dense.chunks = tuple(sorted(chunks, key=lambda chunk: chunk.chunk_id))
        if hasattr(self.dense.embedder, "prepare_index"):
            self.dense.embedder.prepare_index(chunks, self._corpus_id)
            # Even a BM25 request must evict cloud vectors after outbound revocation.
            for doc in documents.values():
                policy = self._sources[self._receipts[doc.document_id]][1].rights
                config = self.dense.embedder.config
                if (not doc.synthetic and (not hasattr(policy, "external_evidence")
                        or not policy.external_evidence(doc, "embedding", config.model, config.provider))):
                    self.dense.embedder.clear_cache()
                    self.dense.clear()
                    self._dense_ids = None
                    break
        return chunks, identities

    def _authorize(self, action, config, chunks):
        """Reverify source integrity plus live rights before each outbound attempt and return."""
        try:
            current, _ = self._refresh()
            trusted = {c.chunk_id: c for c in current}
            for chunk in chunks:
                match = trusted.get(chunk.chunk_id)
                if match is None or match.text != chunk.text or match.document_id != chunk.document_id:
                    raise RAGError("outbound_rights_or_integrity_changed")
                document = self._documents[chunk.document_id]
                policy = self._sources[self._receipts[document.document_id]][1].rights
                if document.synthetic:
                    from app.rag.synthetic import SyntheticRights
                    if (type(policy) is not SyntheticRights or not policy.allows(document)
                            or not self.settings.cloud.synthetic_live_opt_in):
                        raise RAGError("synthetic_cloud_opt_in_required")
                elif not (hasattr(policy, "external_evidence") and policy.external_evidence(
                        document, action, config.model, config.provider)):
                    raise RAGError("outbound_processing_not_approved")
        except RAGError:
            self._invalidate()
            raise
        except Exception:
            self._invalidate()
            raise RAGError("outbound_processing_not_approved_or_revoked") from None

    @staticmethod
    def _check_query(query):
        # Defense in depth; reviewed corpus approval is the primary document privacy gate.
        if (re.search(r"[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}", query)
                or re.search(r"\d[\d\s()+.-]{7,}\d", query)
                or re.search(r"(?i)\b(patient\s*(?:name|id)|họ\s+tên|mã\s+bệnh\s+nhân)\s*:", query)):
            raise RAGError("query_possible_patient_identifier_blocked")

    def search(self, query: str, mode: str = "Hybrid", top_k: int = 5, *,
               rerank: bool | None = None, ask: bool = False) -> SearchResponse:
        start = perf_counter()
        with self._lock:
            if mode not in {"BM25", "Dense", "Hybrid"}:
                raise RAGError("invalid_retrieval_mode")
            if type(top_k) is not int or not 1 <= top_k <= 100:
                raise RAGError("invalid_top_k")
            if not isinstance(query, str) or type(ask) is not bool or rerank is not None and type(rerank) is not bool:
                raise RAGError("invalid_search_options")
            if self.cloud_client is not None:
                self.cloud_client.begin()
            rerank = self.settings.reranking.enabled if rerank is None else rerank
            chunks, identities = self._refresh()
            synthetic = any(d.synthetic for d in self._documents.values()) or (
                not self._documents and any(importer.rights.synthetic for _, importer in self._sources.values())
            )
            hits, status, answer, sources = (), "OK", "", ()
            phases = []
            if not query.strip():
                status = "EMPTY_QUERY"
            elif not chunks:
                status = "EMPTY_INDEX"
            else:
                try:
                    if ask and not self.settings.generation.enabled:
                        raise RAGError("generation_disabled_by_operator")
                    if rerank and not self.settings.reranking.enabled:
                        raise RAGError("reranking_disabled_by_operator")
                    cloud_embedding = hasattr(self.dense.embedder, "prepare_index")
                    if (cloud_embedding and mode in {"Dense", "Hybrid"}) or rerank or ask:
                        self._check_query(query)
                    if cloud_embedding and mode in {"Dense", "Hybrid"}:
                        embedder = self.dense.embedder
                        authorize = lambda: self._authorize("embedding", embedder.config, chunks)
                        authorize()
                        embedder.authorize = authorize
                        embedder.prepare_index(chunks, self._corpus_id)
                        phases.append(("embedding", embedder.config, chunks))
                    if mode in {"Dense", "Hybrid"} and self._dense_ids != identities:
                        self.dense.index(chunks)
                        self._dense_ids = identities
                        self._dense_fingerprint = getattr(self.dense.embedder, "fingerprint", None)
                    retriever = {"BM25": self.bm25, "Dense": self.dense, "Hybrid": self.hybrid}[mode]
                    count = self.settings.reranking.candidates if rerank else top_k
                    hits = retriever.search(query, count)
                    if rerank and hits:
                        if self.reranker is None:
                            raise RAGError("reranker_unavailable")
                        config = self.settings.reranking
                        candidates = tuple(h.chunk for h in hits)
                        auth = lambda: self._authorize("reranking", config, candidates)
                        auth()
                        phases.append(("reranking", config, candidates))
                        try:
                            hits = self.reranker.rerank(query, hits, authorize=auth)
                        except ModelUnavailable:
                            if not config.fallback_to_hybrid or mode != "Hybrid":
                                raise
                            status = "DEGRADED_HYBRID_WITHOUT_RERANK"
                    hits = hits[:top_k]
                    if ask:
                        if self.generator is None:
                            raise RAGError("generator_unavailable")
                        config = self.settings.generation
                        evidence = tuple(h.chunk for h in hits[:config.max_evidence])
                        auth = lambda: self._authorize("generation", config, evidence)
                        auth()
                        answer, sources, generated_status = self.generator.generate(query, hits, authorize=auth)
                        phases.append(("generation", config, evidence))
                        if generated_status != "OK":
                            status = generated_status
                    # A final live rights check also covers revocation during embedding.
                    hits = tuple(h for h in hits if self._sources[self._receipts[h.chunk.document_id]][1]
                                 .rights.allows(self._documents[h.chunk.document_id]))
                    for action, config, used in phases:
                        self._authorize(action, config, used)
                    if not hits and not ask:
                        status = "NO_MATCH"
                except ModelUnavailable as exc:
                    status = str(exc)
                except RAGError as exc:
                    status = str(exc)
                except Exception:
                    status = "retrieval_failed"
                if status not in {"OK", "NO_MATCH", "INSUFFICIENT_EVIDENCE", "DEGRADED_HYBRID_WITHOUT_RERANK"}:
                    hits, answer, sources = (), "", ()
            if ask and status in {"EMPTY_INDEX", "EMPTY_QUERY"}:
                from app.rag.cloud.generation import INSUFFICIENT
                answer, status = INSUFFICIENT, "INSUFFICIENT_EVIDENCE"
            backend = self.settings.embedding.backend + ":" + self.settings.embedding.model
            if rerank:
                backend += " · rerank:" + self.settings.reranking.model
            if ask:
                backend += " · generation:" + self.settings.generation.model
            usage = ({"requests": self.cloud_client.operation.requests, **self.cloud_client.operation.usage}
                     if self.cloud_client is not None else {"requests": 0})
            # Never return mutable references to the trusted internal corpus/citation data.
            return SearchResponse(deepcopy(hits), mode, (perf_counter() - start) * 1000, status, synthetic,
                                  answer, deepcopy(sources), backend, usage)
