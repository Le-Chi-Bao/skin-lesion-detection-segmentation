"""Typed operator JSON settings, independent of ingestion and retrieval implementations."""

import json
import math
from dataclasses import dataclass, field, fields
from pathlib import Path

from app.rag.models import RAGError


def _integer(value, minimum, maximum=None):
    return type(value) is int and value >= minimum and (maximum is None or value <= maximum)


def _number(value):
    try:
        return type(value) in {int, float} and math.isfinite(value)
    except OverflowError:
        return False


def _text(value):
    return isinstance(value, str) and bool(value.strip()) and "\0" not in value


@dataclass(frozen=True)
class ChunkConfig:
    # Legacy name: these budgets count UTF-8 bytes, NOT model tokenizer tokens.
    max_tokens: int = 384
    overlap: int = 48
    token_metric: str = "utf8-bytes-v1"
    version: str = "recursive-blocks-1"
    strategy: str = "recursive-blocks"

    def __post_init__(self):
        if (not _integer(self.max_tokens, 4) or not _integer(self.overlap, 0)
                or self.overlap >= self.max_tokens or self.token_metric != "utf8-bytes-v1"
                or self.version != "recursive-blocks-1" or self.strategy != "recursive-blocks"):
            raise RAGError("invalid_chunk_config")


@dataclass(frozen=True)
class BM25Config:
    k1: float = 1.5
    b: float = 0.75

    def __post_init__(self):
        if not _number(self.k1) or self.k1 <= 0 or not _number(self.b) or not 0 <= self.b <= 1:
            raise RAGError("invalid_bm25_config")


@dataclass(frozen=True)
class EmbeddingConfig:
    model: str = "intfloat/multilingual-e5-base"
    allow_download: bool = False
    device: str = "cpu"
    query_prefix: str = "query: "
    passage_prefix: str = "passage: "
    max_length: int = 512
    batch_size: int = 16
    backend: str = "local"
    provider: str | None = None
    request_timeout: float = 30
    dimensions: int | None = None
    document_input_type: str | None = None
    query_input_type: str | None = None
    max_input_bytes: int = 2048

    def __post_init__(self):
        if (not _text(self.model) or self.device not in ("cpu", "cuda")
                or type(self.allow_download) is not bool
                or not isinstance(self.query_prefix, str) or "\0" in self.query_prefix
                or not isinstance(self.passage_prefix, str) or "\0" in self.passage_prefix
                or not _integer(self.max_length, 8, 512) or not _integer(self.batch_size, 1)):
            raise RAGError("invalid_embedding_config")
        _route(self.backend, self.provider, self.request_timeout)
        if (self.dimensions is not None and not _integer(self.dimensions, 1)
                or not _integer(self.max_input_bytes, 1)
                or any(v is not None and not _text(v)
                       for v in (self.document_input_type, self.query_input_type))):
            raise RAGError("invalid_embedding_config")


def _route(backend, provider, timeout):
    if (backend not in ("local", "openrouter") or provider is not None and not _text(provider)
            or not _number(timeout) or timeout <= 0):
        raise RAGError("invalid_cloud_route")


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = 2
    backoff_seconds: float = 0.5
    max_backoff_seconds: float = 4

    def __post_init__(self):
        if (not _integer(self.max_retries, 0, 5) or not _number(self.backoff_seconds)
                or not _number(self.max_backoff_seconds)
                or not 0 <= self.backoff_seconds <= self.max_backoff_seconds <= 30):
            raise RAGError("invalid_retry_policy")


@dataclass(frozen=True)
class CloudConfig:
    max_requests_per_operation: int = 12
    max_input_chars: int = 16000
    max_index_chunks: int = 128
    synthetic_live_opt_in: bool = False
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)

    def __post_init__(self):
        if (not _integer(self.max_requests_per_operation, 1, 100)
                or not _integer(self.max_input_chars, 1)
                or not _integer(self.max_index_chunks, 1)
                or type(self.synthetic_live_opt_in) is not bool
                or not isinstance(self.retry_policy, RetryPolicy)):
            raise RAGError("invalid_cloud_config")


@dataclass(frozen=True)
class RerankingConfig:
    enabled: bool = False
    model: str = "cohere/rerank-v3.5"
    provider: str | None = None
    top_n: int = 5
    candidates: int = 30
    request_timeout: float = 30
    fallback_to_hybrid: bool = False
    max_input_bytes: int = 8192

    def __post_init__(self):
        _route("openrouter", self.provider, self.request_timeout)
        if (type(self.enabled) is not bool or not _text(self.model)
                or not _integer(self.top_n, 1, 100) or not _integer(self.candidates, 1, 100)
                or self.top_n > self.candidates or not _integer(self.max_input_bytes, 1)
                or type(self.fallback_to_hybrid) is not bool):
            raise RAGError("invalid_reranking_config")


@dataclass(frozen=True)
class GenerationConfig:
    enabled: bool = False
    model: str = "openai/gpt-4o-mini"
    provider: str | None = None
    temperature: float = 0
    max_tokens: int = 512
    max_evidence: int = 5
    request_timeout: float = 30
    max_input_bytes: int = 16000

    def __post_init__(self):
        _route("openrouter", self.provider, self.request_timeout)
        if (type(self.enabled) is not bool or not _text(self.model)
                or not _number(self.temperature) or not 0 <= self.temperature <= 2
                or not _integer(self.max_tokens, 1) or not _integer(self.max_evidence, 1, 100)
                or not _integer(self.max_input_bytes, 1)):
            raise RAGError("invalid_generation_config")


@dataclass(frozen=True)
class RetrievalConfig:
    # Runtime/CLI/UI default; direct RAGService.search() retains Hybrid/5.
    mode: str = "BM25"
    top_k: int = 5

    def __post_init__(self):
        if self.mode not in ("BM25", "Dense", "Hybrid") or not _integer(self.top_k, 1, 100):
            raise RAGError("invalid_retrieval_config")


@dataclass(frozen=True)
class PathConfig:
    manifest: str
    artifacts: str
    provenance: str
    registry: str
    approvals: str

    def __post_init__(self):
        if any(not _text(getattr(self, f.name)) for f in fields(self)):
            raise RAGError("invalid_rag_paths")

    def resolve(self, name: str, config_path: Path) -> Path:
        value = Path(getattr(self, name)).expanduser()
        return (config_path.parent / value).resolve()


def _section(cls, value):
    if not isinstance(value, dict) or set(value) - {f.name for f in fields(cls)}:
        raise RAGError("invalid_rag_config_fields")
    try:
        return cls(**value)
    except TypeError:
        raise RAGError("invalid_rag_config_fields") from None


@dataclass(frozen=True)
class RAGConfig:
    paths: PathConfig | None = None
    chunking: ChunkConfig = field(default_factory=ChunkConfig)
    bm25: BM25Config = field(default_factory=BM25Config)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    rrf_k: float = 60
    candidates: int = 30
    cloud: CloudConfig = field(default_factory=CloudConfig)
    reranking: RerankingConfig = field(default_factory=RerankingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    def __post_init__(self):
        if (self.paths is not None and not isinstance(self.paths, PathConfig)):
            raise RAGError("invalid_rag_paths")
        for name, cls in (("chunking", ChunkConfig), ("bm25", BM25Config),
                          ("embedding", EmbeddingConfig), ("retrieval", RetrievalConfig),
                          ("cloud", CloudConfig), ("reranking", RerankingConfig),
                          ("generation", GenerationConfig)):
            if not isinstance(getattr(self, name), cls):
                raise RAGError("invalid_rag_config_fields")
        if not _number(self.rrf_k) or self.rrf_k <= 0:
            raise RAGError("invalid_rrf_k")
        if not _integer(self.candidates, 1):
            raise RAGError("invalid_candidate_count")

    @classmethod
    def from_dict(cls, value):
        path_keys = {f.name for f in fields(PathConfig)}
        keys = path_keys | {"chunking", "bm25", "embedding", "retrieval", "rrf_k", "candidates",
                           "cloud", "reranking", "generation"}
        if not isinstance(value, dict) or set(value) - keys:
            raise RAGError("invalid_rag_config_fields")
        present = path_keys & value.keys()
        if present and present != path_keys:
            raise RAGError("incomplete_rag_paths")
        cloud = value.get("cloud", {})
        if not isinstance(cloud, dict):
            raise RAGError("invalid_cloud_config")
        cloud = dict(cloud)
        cloud["retry_policy"] = _section(RetryPolicy, cloud.get("retry_policy", {}))
        embedding = value.get("embedding", {})
        if isinstance(embedding, dict) and embedding.get("backend") == "openrouter":
            embedding = {"model": "baai/bge-m3", "query_prefix": "", "passage_prefix": "", **embedding}
        return cls(paths=_section(PathConfig, {k: value[k] for k in path_keys}) if present else None,
                   chunking=_section(ChunkConfig, value.get("chunking", {})),
                   bm25=_section(BM25Config, value.get("bm25", {})),
                   embedding=_section(EmbeddingConfig, embedding),
                   retrieval=_section(RetrievalConfig, value.get("retrieval", {})),
                   rrf_k=value.get("rrf_k", 60), candidates=value.get("candidates", 30),
                   cloud=_section(CloudConfig, cloud),
                   reranking=_section(RerankingConfig, value.get("reranking", {})),
                   generation=_section(GenerationConfig, value.get("generation", {})))

    @classmethod
    def load(cls, path: Path):
        def unique_object(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise RAGError("duplicate_rag_config_key")
                result[key] = value
            return result

        def reject_constant(value):
            raise RAGError("non_finite_rag_config_number")

        try:
            value = json.loads(Path(path).read_text(encoding="utf-8"),
                               object_pairs_hook=unique_object, parse_constant=reject_constant)
        except (UnicodeError, json.JSONDecodeError):
            raise RAGError("invalid_rag_config_json") from None
        return cls.from_dict(value)
