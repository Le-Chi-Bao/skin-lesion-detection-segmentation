"""Operator-only configuration; the browser cannot select files, stores or rights policies."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory

from app.rag.chunking import RecursiveChunker
from app.rag.config import RAGConfig
from app.rag.importer import DocumentImporter, StoreBindings
from app.rag.models import RAGError
from app.rag.retrieval import LocalE5Embedder
from app.rag.rights import RegistryRights
from app.rag.service import RAGService
from app.rag.synthetic import SyntheticRights, write_demo


def create_service(*, demo: bool = False, config_path: Path | None = None) -> RAGService:
    if demo and config_path is not None:
        raise RAGError("demo_and_real_config_are_exclusive")
    settings = RAGConfig.load(config_path) if config_path else RAGConfig()
    if config_path and settings.paths is None:
        raise RAGError("incomplete_rag_paths")
    embeddings = LocalE5Embedder(settings.embedding)
    service = RAGService(chunker=RecursiveChunker(settings.chunking), embedder=embeddings,
                         rrf_k=settings.rrf_k, candidates=settings.candidates,
                         bm25_k1=settings.bm25.k1, bm25_b=settings.bm25.b)
    service.settings = settings
    if demo:
        # Lifetime belongs to this service/session, no shared corpus or browser input.
        service._demo_directory = TemporaryDirectory(prefix="wp6a-synthetic-")
        manifest, bindings = write_demo(Path(service._demo_directory.name))
        result = service.ingest(manifest, DocumentImporter(bindings, SyntheticRights()))
    elif config_path:
        def bound_path(key):
            return settings.paths.resolve(key, config_path)

        importer = DocumentImporter(StoreBindings(bound_path("artifacts"), bound_path("provenance")),
                                    RegistryRights(bound_path("registry"), bound_path("approvals")))
        result = service.ingest(bound_path("manifest"), importer)
    else:
        return service
    if result.status != "VERIFIED":
        service.close()
        raise RAGError(result.error_code or "batch_blocked_by_rights_or_extraction_review")
    return service


def service_from_env() -> RAGService:
    path = os.environ.get("RAG_CONFIG")
    return create_service(demo=os.environ.get("RAG_SYNTHETIC_DEMO") == "1",
                          config_path=Path(path).resolve() if path else None)
