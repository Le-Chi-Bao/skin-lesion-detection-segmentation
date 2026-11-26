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


def load_environment():
    # Startup boundary only. Never override the operator's process environment.
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)


def create_service(*, demo: bool = False, config_path: Path | None = None,
                   settings: RAGConfig | None = None, transport=None) -> RAGService:
    load_environment()
    if demo and config_path is not None:
        raise RAGError("demo_and_real_config_are_exclusive")
    if settings is not None and config_path is not None:
        raise RAGError("settings_and_config_are_exclusive")
    settings = settings or (RAGConfig.load(config_path) if config_path else RAGConfig.from_dict(
        {"embedding": {"backend": "openrouter"}}))
    if config_path and settings.paths is None:
        raise RAGError("incomplete_rag_paths")
    from app.rag.cloud.client import OpenRouterClient
    from app.rag.cloud.embeddings import OpenRouterEmbedder
    from app.rag.cloud.reranking import OpenRouterReranker
    from app.rag.cloud.generation import OpenRouterGenerator

    client = OpenRouterClient(settings.cloud, transport=transport)
    embeddings = (OpenRouterEmbedder(settings.embedding, client) if settings.embedding.backend == "openrouter"
                  else LocalE5Embedder(settings.embedding))
    service = RAGService(chunker=RecursiveChunker(settings.chunking), embedder=embeddings,
                         rrf_k=settings.rrf_k, candidates=settings.candidates,
                         bm25_k1=settings.bm25.k1, bm25_b=settings.bm25.b, settings=settings,
                         cloud_client=client, reranker=OpenRouterReranker(settings.reranking, client),
                         generator=OpenRouterGenerator(settings.generation, client))
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
    load_environment()
    path = os.environ.get("RAG_CONFIG")
    return create_service(demo=os.environ.get("RAG_SYNTHETIC_DEMO") == "1",
                          config_path=Path(path).resolve() if path else None)
