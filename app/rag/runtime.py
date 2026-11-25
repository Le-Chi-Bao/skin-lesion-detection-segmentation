"""Operator-only configuration; the browser cannot select files, stores or rights policies."""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from app.rag.chunking import ChunkConfig, RecursiveChunker
from app.rag.importer import DocumentImporter, StoreBindings
from app.rag.models import RAGError
from app.rag.retrieval import EmbeddingConfig, LocalE5Embedder
from app.rag.rights import RegistryRights
from app.rag.service import RAGService
from app.rag.synthetic import SyntheticRights, write_demo


def create_service(*, demo: bool = False, config_path: Path | None = None) -> RAGService:
    if demo and config_path is not None:
        raise RAGError("demo_and_real_config_are_exclusive")
    settings = json.loads(config_path.read_text(encoding="utf-8")) if config_path else {}
    embeddings = LocalE5Embedder(EmbeddingConfig(**settings.get("embedding", {})))
    service = RAGService(chunker=RecursiveChunker(ChunkConfig(**settings.get("chunking", {}))),
                         embedder=embeddings, rrf_k=settings.get("rrf_k", 60),
                         candidates=settings.get("candidates", 30))
    if demo:
        # Lifetime belongs to this service/session, no shared corpus or browser input.
        service._demo_directory = TemporaryDirectory(prefix="wp6a-synthetic-")
        manifest, bindings = write_demo(Path(service._demo_directory.name))
        result = service.ingest(manifest, DocumentImporter(bindings, SyntheticRights()))
    elif config_path:
        def bound_path(key):
            value = Path(settings[key]).expanduser()
            return (config_path.parent / value).resolve()

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
