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

    def __post_init__(self):
        if (not _text(self.model) or self.device not in ("cpu", "cuda")
                or type(self.allow_download) is not bool
                or not isinstance(self.query_prefix, str) or "\0" in self.query_prefix
                or not isinstance(self.passage_prefix, str) or "\0" in self.passage_prefix
                or not _integer(self.max_length, 8, 512) or not _integer(self.batch_size, 1)):
            raise RAGError("invalid_embedding_config")


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

    def __post_init__(self):
        if (self.paths is not None and not isinstance(self.paths, PathConfig)):
            raise RAGError("invalid_rag_paths")
        for name, cls in (("chunking", ChunkConfig), ("bm25", BM25Config),
                          ("embedding", EmbeddingConfig), ("retrieval", RetrievalConfig)):
            if not isinstance(getattr(self, name), cls):
                raise RAGError("invalid_rag_config_fields")
        if not _number(self.rrf_k) or self.rrf_k <= 0:
            raise RAGError("invalid_rrf_k")
        if not _integer(self.candidates, 1):
            raise RAGError("invalid_candidate_count")

    @classmethod
    def from_dict(cls, value):
        path_keys = {f.name for f in fields(PathConfig)}
        keys = path_keys | {"chunking", "bm25", "embedding", "retrieval", "rrf_k", "candidates"}
        if not isinstance(value, dict) or set(value) - keys:
            raise RAGError("invalid_rag_config_fields")
        present = path_keys & value.keys()
        if present and present != path_keys:
            raise RAGError("incomplete_rag_paths")
        return cls(paths=_section(PathConfig, {k: value[k] for k in path_keys}) if present else None,
                   chunking=_section(ChunkConfig, value.get("chunking", {})),
                   bm25=_section(BM25Config, value.get("bm25", {})),
                   embedding=_section(EmbeddingConfig, value.get("embedding", {})),
                   retrieval=_section(RetrievalConfig, value.get("retrieval", {})),
                   rrf_k=value.get("rrf_k", 60), candidates=value.get("candidates", 30))

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
