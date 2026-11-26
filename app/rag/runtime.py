"""Operator-only configuration; the browser cannot select files, stores or rights policies."""

import os
from dataclasses import replace
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
    if config_path and settings.paths is None and settings.local_corpus is None:
        raise RAGError("incomplete_rag_paths")
    if settings.local_corpus is not None and (demo or config_path is None):
        raise RAGError("local_corpus_requires_operator_config")
    if settings.local_corpus is not None and (settings.embedding.backend != "local"
            or settings.reranking.enabled or settings.generation.enabled):
        raise RAGError("local_consent_does_not_allow_cloud_processing")
    if config_path and settings.embedding.backend == "local":
        model = Path(settings.embedding.model)
        if not model.is_absolute() and (config_path.parent / model).is_dir():
            settings = replace(settings, embedding=replace(settings.embedding,
                               model=str((config_path.parent / model).resolve())))
    from app.rag.cloud.client import OpenRouterClient
    from app.rag.cloud.embeddings import OpenRouterEmbedder
    from app.rag.cloud.reranking import OpenRouterReranker
    from app.rag.cloud.generation import OpenRouterGenerator

    client = OpenRouterClient(settings.cloud, transport=transport)
    embeddings = (OpenRouterEmbedder(settings.embedding, client) if settings.embedding.backend == "openrouter"
                  else LocalE5Embedder(settings.embedding))
    dense = None
    if settings.vector_store:
        from app.rag.vector_store import ChromaDenseRetriever
        if config_path is None or settings.embedding.dimensions is None:
            raise RAGError("persistent_store_requires_config_and_dimensions")
        dense = ChromaDenseRetriever(embeddings, (config_path.parent / settings.vector_store.path).resolve())
    service = RAGService(chunker=RecursiveChunker(settings.chunking), embedder=embeddings,
                         rrf_k=settings.rrf_k, candidates=settings.candidates,
                         bm25_k1=settings.bm25.k1, bm25_b=settings.bm25.b, settings=settings,
                         cloud_client=client, reranker=OpenRouterReranker(settings.reranking, client),
                         generator=OpenRouterGenerator(settings.generation, client), dense_retriever=dense)
    if settings.local_corpus:
        from app.rag.chunking import E5BudgetChunker
        service.chunker = E5BudgetChunker(settings.chunking, embeddings)
    if demo:
        # Lifetime belongs to this service/session, no shared corpus or browser input.
        service._demo_directory = TemporaryDirectory(prefix="wp6a-synthetic-")
        manifest, bindings = write_demo(Path(service._demo_directory.name))
        result = service.ingest(manifest, DocumentImporter(bindings, SyntheticRights()))
    elif settings.local_corpus:
        from app.rag.local_corpus import LocalKitImporter
        local = settings.local_corpus
        resolve = lambda value: (config_path.parent / value).resolve()
        approval = resolve(local.approval)
        importer = LocalKitImporter(resolve(local.kit), resolve(local.markdown), approval)
        result = service.ingest(approval, importer)
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
    demo = os.environ.get("RAG_SYNTHETIC_DEMO") == "1"
    demo_path = os.environ.get("RAG_DEMO_CONFIG")
    if demo_path and (not demo or path):
        raise RAGError("demo_settings_require_exclusive_synthetic_mode")
    return create_service(demo=demo, config_path=Path(path).resolve() if path else None,
                          settings=RAGConfig.load(Path(demo_path).resolve()) if demo_path else None)
