import pytest

pytest.importorskip("ragas", reason="install requirements-rag-evaluation.txt for WP-RAGAS tests")

from app.rag.evaluation import (
    ContextPrecisionByChunkId,
    ContextRecallByChunkId,
    citation_metadata_valid,
    mrr_at_k,
    ndcg_at_k,
    precision_recall_at_k,
    validate_evidence_sources,
)
from app.rag.evidence_summary import summarize_evidence
from app.rag.models import Chunk, SearchHit
from scripts.rag_evaluate import _ragas_metric_rows


def test_precision_recall_use_fixed_cutoff_and_verified_relevant_ids():
    assert precision_recall_at_k(["a", "b", "c", "d", "e"], ["b", "e", "z"], 5) == (0.4, 2 / 3)
    assert precision_recall_at_k(["a"], [], 5) == (None, None)


def test_mrr_and_ndcg_apply_requested_cutoffs():
    assert mrr_at_k(["x", "y", "a"], ["a"], 2) == 0.0
    assert mrr_at_k(["x", "a"], ["a"], 10) == 0.5
    assert ndcg_at_k(["x", "a"], ["a"], 2) == 1 / 1.584962500721156
    assert ndcg_at_k(["x"], [], 5) is None


def test_citation_metadata_checks_pdf_page_document_and_source(tmp_path):
    pdf = tmp_path / "source.pdf"
    pdf.write_bytes(b"%PDF-test")
    citation = {"document_id": "doc-a", "pdf_page": 2, "pdf_path": str(pdf), "title": "Title",
                "source": {"locator": "https://example.org/article"}}
    chunk = Chunk("chunk-a", "doc-a", "An exact sentence.", citation)
    assert citation_metadata_valid(chunk)
    invalid = Chunk("chunk-b", "doc-a", "text", {**citation, "pdf_page": 0})
    assert not citation_metadata_valid(invalid)
    wrong_document = Chunk("chunk-c", "doc-b", "text", citation)
    assert not citation_metadata_valid(wrong_document)


def test_evidence_span_validation_and_out_of_scope_abstention():
    citation = {"document_id": "doc-a", "pdf_page": 1, "pdf_path": "source.pdf", "title": "Title",
                "source": {"locator": "https://example.org/article"}}
    chunk = Chunk("chunk-a", "doc-a", "The report found 12 cases.", citation)
    hit = SearchHit(chunk, 1.0, "BM25")
    answer, sources = summarize_evidence("report cases count", [hit])
    assert answer
    assert validate_evidence_sources(sources, [hit]) == [
        {"chunk_id": "chunk-a", "exact_span_valid": True, "citation_consistent": True}
    ]
    answer, sources = summarize_evidence("current insulin dosing", [hit])
    assert answer == ""
    assert sources == ()


def test_pending_ground_truth_is_not_scored():
    question = {"review_status": "pending_review", "relevant_chunk_ids": ["chunk-a"]}
    relevant = question["relevant_chunk_ids"] if question["review_status"] == "verified" else []
    assert precision_recall_at_k(["chunk-a"], relevant, 5) == (None, None)
    assert mrr_at_k(["chunk-a"], relevant, 10) is None
    assert ndcg_at_k(["chunk-a"], relevant, 5) is None


def test_metric_helpers_are_deterministic():
    ranking = ["a", "b", "c", "d", "e"]
    relevant = ["b", "d"]
    expected = (precision_recall_at_k(ranking, relevant, 5), mrr_at_k(ranking, relevant),
                ndcg_at_k(ranking, relevant))
    for _ in range(10):
        assert (precision_recall_at_k(ranking, relevant, 5), mrr_at_k(ranking, relevant),
                ndcg_at_k(ranking, relevant)) == expected


def test_ragas_collection_metrics_use_reviewed_chunk_ids_without_a_judge():
    precision = ContextPrecisionByChunkId().score(
        retrieved_context_ids=["noise", "relevant-a", "noise-2", "relevant-b"],
        reference_context_ids=["relevant-a", "relevant-b"],
    )
    recall = ContextRecallByChunkId().score(
        retrieved_context_ids=["noise", "relevant-a", "noise-2", "relevant-b"],
        reference_context_ids=["relevant-a", "relevant-b", "missing"],
    )
    assert precision.value == 0.5
    assert recall.value == 2 / 3


def test_pipeline_marks_unverified_and_judge_metrics_as_na():
    rows = [{"mode": mode, "human_review_status": "pending_review", "relevant_chunk_ids": [],
             "retrieved_context_ids": ["chunk-a"]} for mode in ("BM25", "Dense", "Hybrid")]
    metrics = _ragas_metric_rows(rows)
    assert len(metrics) == 15
    assert all(item["status"] == "N/A" and item["value"] is None for item in metrics)
    assert all(item["reason"] for item in metrics)


def test_pipeline_calculates_context_metrics_only_for_verified_rows():
    rows = [{"mode": "BM25", "human_review_status": "verified",
             "relevant_chunk_ids": ["relevant-a", "relevant-b"],
             "retrieved_context_ids": ["noise", "relevant-a", "noise-2", "relevant-b"]}]
    metrics = _ragas_metric_rows(rows)
    measured = {item["metric"]: item for item in metrics if item["mode"] == "BM25"}
    assert measured["context_precision"]["value"] == 0.5
    assert measured["context_recall"]["value"] == 1.0
    assert measured["context_precision"]["status"] == "measured"
    assert all(item["value"] is None for item in metrics if item["mode"] != "BM25")
