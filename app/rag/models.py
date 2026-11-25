"""Small data contracts shared by ingestion, indexing and the future WP6B adapter."""

from dataclasses import dataclass
from typing import Any


class RAGError(ValueError):
    """Safe error code; never include rejected source text in errors."""


class ModelUnavailable(RAGError):
    pass


@dataclass(frozen=True)
class Document:
    record: dict[str, Any]
    record_hash: str
    batch_id: str
    event_id: str
    provenance_hash: str
    rights: dict[str, Any]
    synthetic: bool = False

    @property
    def document_id(self) -> str:
        return self.record["document_id"]


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    text: str
    citation: dict[str, Any]


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    score: float
    mode: str


@dataclass(frozen=True)
class SearchResponse:
    hits: tuple[SearchHit, ...]
    mode: str
    elapsed_ms: float
    status: str
    synthetic: bool


@dataclass(frozen=True)
class ImportResult:
    status: str
    documents: tuple[Document, ...] = ()
    rejected: tuple[tuple[str, str], ...] = ()
    error_code: str | None = None
    # IDs are exposed only after the entire batch passes integrity verification.
    observed_ids: tuple[str, ...] = ()
