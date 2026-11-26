"""Deterministic evaluation helpers for the existing local Medical RAG pipeline."""

from __future__ import annotations

import math
from pathlib import Path

from ragas.metrics.collections import BaseMetric
from ragas.metrics.collections.base import MetricResult


def precision_recall_at_k(ranked_ids, relevant_ids, k):
    relevant = set(relevant_ids)
    if not relevant or k < 1:
        return None, None
    top = list(ranked_ids[:k])
    hits = sum(item in relevant for item in top)
    return hits / k, hits / len(relevant)


def mrr_at_k(ranked_ids, relevant_ids, k=10):
    relevant = set(relevant_ids)
    if not relevant:
        return None
    for rank, item in enumerate(ranked_ids[:k], 1):
        if item in relevant:
            return 1 / rank
    return 0.0


def ndcg_at_k(ranked_ids, relevant_ids, k=5):
    relevant = set(relevant_ids)
    if not relevant:
        return None
    dcg = sum(1 / math.log2(rank + 1) for rank, item in enumerate(ranked_ids[:k], 1)
              if item in relevant)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(k, len(relevant)) + 1))
    return dcg / ideal if ideal else None


def citation_metadata_valid(chunk):
    citation = chunk.citation
    source = citation.get("source")
    if not isinstance(source, dict):
        return False
    locator = source.get("locator")
    page = citation.get("pdf_page")
    pdf_path = citation.get("pdf_path")
    return bool(
        citation.get("document_id") == chunk.document_id
        and isinstance(page, int) and not isinstance(page, bool) and page > 0
        and isinstance(pdf_path, str) and pdf_path.strip() and Path(pdf_path).is_file()
        and isinstance(locator, str) and locator.startswith(("https://", "http://"))
        and citation.get("title")
    )


def validate_evidence_sources(excerpts, hits):
    by_id = {hit.chunk.chunk_id: hit.chunk for hit in hits}
    checks = []
    for excerpt in excerpts:
        chunk = by_id.get(excerpt.get("chunk_id"))
        exact_span = bool(chunk and excerpt.get("excerpt") and excerpt["excerpt"] in chunk.text)
        citation_consistent = bool(
            chunk and excerpt.get("citation") == chunk.citation
            and chunk.citation.get("document_id") == chunk.document_id
        )
        checks.append({
            "chunk_id": excerpt.get("chunk_id"),
            "exact_span_valid": exact_span,
            "citation_consistent": citation_consistent,
        })
    return checks


def mean_or_none(values):
    observed = [value for value in values if value is not None]
    return sum(observed) / len(observed) if observed else None


class ContextPrecisionByChunkId(BaseMetric):
    """RAGAS collection metric using reviewer-approved chunk IDs as relevance labels."""

    def __init__(self):
        super().__init__(name="context_precision")

    async def ascore(self, *, retrieved_context_ids, reference_context_ids):
        relevant = set(reference_context_ids)
        ranked = list(retrieved_context_ids)
        relevant_ranks = [rank for rank, chunk_id in enumerate(ranked, 1) if chunk_id in relevant]
        if not relevant_ranks:
            return MetricResult(value=0.0, reason="no relevant chunk retrieved")
        average_precision = sum(index / rank for index, rank in enumerate(relevant_ranks, 1)) / len(relevant_ranks)
        return MetricResult(value=average_precision)


class ContextRecallByChunkId(BaseMetric):
    """RAGAS collection metric using reviewer-approved chunk IDs as relevance labels."""

    def __init__(self):
        super().__init__(name="context_recall")

    async def ascore(self, *, retrieved_context_ids, reference_context_ids):
        relevant = set(reference_context_ids)
        if not relevant:
            return MetricResult(value=float("nan"), reason="no verified reference chunk IDs")
        retrieved = set(retrieved_context_ids)
        return MetricResult(value=len(relevant & retrieved) / len(relevant))
