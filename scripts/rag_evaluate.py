"""Reproducible WP6E retrieval and extractive-summary evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.rag.config import RAGConfig
from app.rag.evaluation import (
    ContextPrecisionByChunkId,
    ContextRecallByChunkId,
    citation_metadata_valid,
    mean_or_none,
    mrr_at_k,
    ndcg_at_k,
    precision_recall_at_k,
    validate_evidence_sources,
)
from app.rag.evidence_summary import summarize_evidence
from app.rag.retrieval import LocalE5Embedder, tokenize
from app.rag.runtime import create_service, load_environment

MODES = ("BM25", "Dense", "Hybrid")
RAGAS_METRICS = (
    ("context_precision", "retrieval"),
    ("context_recall", "retrieval"),
    ("faithfulness", "extractive_summary"),
    ("answer_relevancy", "extractive_summary"),
    ("unsupported_claim_rate", "llm_generation"),
)
STOP_WORDS = {"va", "la", "nhung", "cac", "mot", "co", "trong", "cua", "den", "duoc",
              "nao", "gi", "the", "doi", "voi", "ve", "hay", "khong", "tai", "tu", "o",
              "khi", "bệnh", "viêm", "da", "cơ", "địa"}


def _write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict))
                             else value for key, value in row.items()})


def _read_questions(path):
    questions = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(questions, list) or len(questions) != 40:
        raise ValueError("WP6E input must contain exactly 40 questions")
    required = {"question_id", "query", "review_status", "relevant_document_ids", "relevant_chunk_ids"}
    if any(not isinstance(row, dict) or not required.issubset(row) for row in questions):
        raise ValueError("WP6E question schema is incomplete")
    if len({row["question_id"] for row in questions}) != len(questions):
        raise ValueError("WP6E question IDs must be unique")
    return questions


def _vector_snapshot(config_path):
    import chromadb
    from chromadb.config import Settings
    from dataclasses import replace

    settings = RAGConfig.load(config_path)
    model_path = Path(settings.embedding.model)
    if not model_path.is_absolute():
        model_path = (config_path.parent / model_path).resolve()
    embedder = LocalE5Embedder(replace(settings.embedding, model=str(model_path)))
    fingerprint = embedder.persistent_fingerprint
    vector_path = (config_path.parent / settings.vector_store.path).resolve()
    client = chromadb.PersistentClient(path=str(vector_path), settings=Settings(anonymized_telemetry=False))
    collection = client.get_collection("medical-e5-" + fingerprint[:32])
    rows = collection.get(include=["embeddings", "metadatas"])
    digest = hashlib.sha256()
    for index, chunk_id in enumerate(sorted(rows["ids"])):
        position = rows["ids"].index(chunk_id)
        digest.update(chunk_id.encode("utf-8"))
        digest.update(rows["embeddings"][position].tobytes())
        metadata = rows["metadatas"][position] or {}
        digest.update(metadata.get("content_sha256", "").encode("ascii"))
    return {"count": collection.count(), "sha256": digest.hexdigest(), "collection": collection.name,
            "embedding_fingerprint": fingerprint}


def _sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _corpus_fingerprint():
    digest = hashlib.sha256()
    paths = [ROOT / "bo_tai_lieu_y_khoa_tieng_viet_pdf_kit/metadata/documents.json"]
    paths.extend(sorted((ROOT / "data/rag/markdown").glob("*.md")))
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _memory_gib():
    if sys.platform != "win32":
        return None
    import ctypes

    class MemoryStatus(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return {"total_gib": round(status.ullTotalPhys / 1024**3, 2),
            "available_at_recording_gib": round(status.ullAvailPhys / 1024**3, 2)}


def _coverage(query, excerpts):
    terms = {token for token in tokenize(query) if len(token) > 1 and token not in STOP_WORDS}
    if not terms:
        return None
    present = {token for item in excerpts for token in tokenize(item["excerpt"])}
    return len(terms & present) / len(terms)


def _ragas_metric_rows(evaluation_rows):
    rows = []
    for mode in MODES:
        eligible = [item for item in evaluation_rows if item["mode"] == mode
                    and item["human_review_status"] == "verified" and item["relevant_chunk_ids"]]
        for metric, scope in RAGAS_METRICS:
            if metric == "context_precision" and eligible:
                scorer = ContextPrecisionByChunkId()
                values = [scorer.score(retrieved_context_ids=item["retrieved_context_ids"],
                                       reference_context_ids=item["relevant_chunk_ids"]).value
                          for item in eligible]
                value, status, reason = mean_or_none(values), "measured", "RAGAS custom metric; reviewer-approved chunk IDs"
            elif metric == "context_recall" and eligible:
                scorer = ContextRecallByChunkId()
                values = [scorer.score(retrieved_context_ids=item["retrieved_context_ids"],
                                       reference_context_ids=item["relevant_chunk_ids"]).value
                          for item in eligible]
                value, status, reason = mean_or_none(values), "measured", "RAGAS custom metric; reviewer-approved chunk IDs"
            elif metric == "context_recall":
                value, status, reason = None, "N/A", "no human-verified relevant chunk IDs"
            elif metric == "context_precision":
                value, status, reason = None, "N/A", "no human-verified relevant chunk IDs"
            elif metric == "unsupported_claim_rate":
                value, status, reason = None, "N/A", "no real LLM generation is enabled"
            else:
                value, status, reason = None, "N/A", "LLM judge not run; response is extractive Evidence Summary"
            rows.append({"mode": mode, "metric": metric, "scope": scope, "value": value,
                         "status": status, "reason": reason, "ragas_version": "0.4.4"})
    return rows


def _build_report(report):
    comparisons = report["retrieval_comparison"]
    metric_rows = {row["mode"]: row for row in comparisons}
    oos_abstention = report["evidence_summary"]["out_of_scope_summary_abstention_rate"]
    oos_abstention_text = "N/A" if oos_abstention is None else f"{oos_abstention:.2f}"
    oos_count = report["dataset_summary"]["out_of_scope_marked_count"]
    oos_abstained = round(oos_abstention * oos_count) if oos_abstention is not None else 0
    lines = [
        "# RAGAS Medical RAG Evaluation Report",
        "",
        f"Run time: {report['run_metadata']['timestamp_utc']}",
        "",
        "## 1. Mục tiêu đánh giá",
        "Đo retrieval BM25, Dense multilingual E5 Small và Hybrid RRF trên cùng bộ 40 câu WP6E; "
        "kiểm tra citation metadata và Evidence Summary extractive. Không chạy LLM generation.",
        "",
        "## 2. Dataset và tính hợp lệ của nhãn",
        f"WP6E có {report['dataset_summary']['question_count']} câu. "
        f"{report['dataset_summary']['verified_count']} verified, "
        f"{report['dataset_summary']['pending_review_count']} pending_review. "
        "Không dùng nhãn pending_review làm ground truth. File review mẫu có 15 câu gồm câu được đánh dấu "
        "ngoài phạm vi để người đánh giá xác minh.",
        "",
        "## 3. Phương pháp thực nghiệm",
        f"Corpus cố định ở {report['corpus']['documents']} tài liệu và "
        f"{report['corpus']['chunks']} chunks. Top-K=10, cùng embedding/chunking/corpus, "
        f"{report['run_metadata']['repeats']} lượt đo sau một warm-up Dense không tính thời gian. "
        f"Thiết bị {report['run_metadata']['device']}; model {report['run_metadata']['embedding_model']} "
        f"({report['run_metadata']['embedding_dimensions']} chiều). RAGAS {report['run_metadata']['ragas_version']}. "
        f"E5 fingerprint {report['run_metadata']['embedding_fingerprint']}. "
        f"Trong lượt chỉ đọc này, hash trước-sau bằng nhau: "
        f"{report['vector_integrity']['unchanged_during_evaluation']}. "
        f"So với hash pre-run đã lưu, baseline bit-for-bit được phục hồi: "
        f"{report['vector_integrity']['baseline_exactly_recovered']}.",
        "",
        "## 4. Các metric và cách tính",
        "Precision@5, Recall@5, MRR@10 và nDCG@5 dùng relevant_chunk_ids chỉ khi câu đã verified. "
        "Custom RAGAS Context Precision là average precision theo các relevant chunk ở vị trí xếp hạng; "
        "Context Recall là tỷ lệ relevant chunk IDs được truy xuất. Cả hai chỉ chạy khi có nhãn người duyệt. "
        "Citation validity kiểm tra document ID khớp, "
        "trang PDF dương, đường dẫn tồn tại, title và source locator. Evidence span validity kiểm tra "
        "trích đoạn nguyên văn có trong chunk được dẫn. Evidence coverage là proxy từ vựng, tính tỷ lệ "
        "token nội dung câu hỏi xuất hiện trong excerpt. Unsupported excerpt rate chỉ phát hiện chuỗi "
        "không khớp nguyên văn, không khẳng định tính đúng y khoa.",
        "",
        "## 5. Kết quả đo thực tế",
        "| Mode | P@5 | R@5 | MRR@10 | nDCG@5 | Latency mean (ms) | Citation hợp lệ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        row = metric_rows[mode]
        values = [row.get(key) for key in ("precision_at_5", "recall_at_5", "mrr_at_10", "ndcg_at_5")]
        display = ["N/A" if value is None else f"{value:.4f}" for value in values]
        lines.append(f"| {mode} | {' | '.join(display)} | {row['latency_mean_ms']:.2f} | "
                     f"{row['citation_valid_rate']:.4f} ({row['citation_valid_count']}/{row['citation_checked']}) |")
    lines.extend(["", "Evidence Summary extractive, các chỉ số quan sát:",
                  "| Mode | Lexical coverage | Exact span | Citation consistency | Abstention trên 2 câu gắn cờ ngoài phạm vi* |",
                  "|---|---:|---:|---:|---:|"])
    for row in report["evidence_summary"]["by_mode"]:
        values = [row["lexical_coverage_mean"], row["exact_span_valid_rate"],
                  row["citation_consistency_rate"], row["flagged_out_of_scope_abstention_rate"]]
        display = ["N/A" if value is None else f"{value:.4f}" for value in values]
        lines.append(f"| {row['mode']} | {' | '.join(display)} |")
    lines.extend(["", "## 6. So sánh ba phương pháp",
                  "Không xếp hạng chất lượng retrieval vì chưa có câu hỏi verified. Latency và citation "
                  "metadata là số đo vận hành, không chứng minh relevance hoặc độ chính xác lâm sàng.",
                  "Hai câu được gắn cờ ngoài phạm vi trong dataset vẫn pending_review; abstention rate này chỉ là "
                  "quan sát mô tả, không phải kết quả scope đã xác minh.",
                  "", "## 7. Phân tích lỗi",
                  f"Có {report['citation_summary']['invalid_metadata']} citation metadata không hợp lệ trong "
                  f"{report['citation_summary']['checked']} hit. Có "
                  f"{report['evidence_summary']['invalid_exact_spans']} excerpt không khớp nguyên văn nguồn trong "
                  f"{report['evidence_summary']['excerpt_count']} excerpt. Với {oos_count} câu đang được gắn cờ "
                  f"ngoài phạm vi, {oos_count - oos_abstained} câu trả excerpt; tỷ lệ abstention quan sát là "
                  f"{oos_abstention_text}. "
                  "Hai cờ scope vẫn pending_review. Kiểm tra này không đánh giá ngữ cảnh lâm sàng.",
                  "", "## 8. Giới hạn đánh giá tự động",
                  "40/40 nhãn hiện pending_review. Chưa có reference answer hoặc relevant chunk IDs được duyệt. "
                  "Không chạy LLM judge; OpenRouter không được gọi trong lượt này. RAGAS Faithfulness, Answer Relevancy "
                  "và unsupported claim rate được ghi N/A. Context Precision/Recall chỉ dùng custom RAGAS metric "
                  "khi có chunk IDs đã duyệt. Evidence Summary là trích xuất "
                  "theo từ khóa, không phải generation; lexical coverage và exact-span validation không thay thế review người. "
                  "Lượt chạy đầu đã upsert lại Chroma và làm đổi hash do sai khác số thực ở bước chuẩn hóa. "
                  "Event log cũ không giữ đủ 496 vector để xác nhận phục hồi bit-for-bit; hash baseline ban đầu "
                  "không được tuyên bố là đã khôi phục. Các lượt evaluation sau dùng attach vector read-only.",
                  "", "## 9. Kết luận",
                  "Pipeline chạy được trên corpus local và đo latency/citation metadata. Chưa đủ bằng chứng để kết luận "
                  "BM25, Dense hoặc Hybrid tốt hơn. Cần người đánh giá hoàn thành mẫu nhãn, sau đó chạy lại cùng pipeline.",
                  "", "## Tóm tắt cho slide bảo vệ",
                  f"- Đã chạy {report['dataset_summary']['question_count']} câu x 3 mode trên {report['corpus']['chunks']} chunks; "
                  f"citation metadata hợp lệ {report['citation_summary']['valid_metadata']}/"
                  f"{report['citation_summary']['checked']}.",
                  f"- Latency trung bình BM25/Dense/Hybrid lần lượt là "
                  f"{metric_rows['BM25']['latency_mean_ms']:.2f}/"
                  f"{metric_rows['Dense']['latency_mean_ms']:.2f}/"
                  f"{metric_rows['Hybrid']['latency_mean_ms']:.2f} ms; retrieval quality metrics N/A do chưa có nhãn verified.",
                  ""])
    return "\n".join(lines)


def _plot_results(output_dir, comparison, ragas_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    colors = {"BM25": "#26736b", "Dense": "#c05d3e", "Hybrid": "#5276a7"}
    metric_names = ["Context Precision", "Context Recall", "Faithfulness", "Answer Relevancy"]
    fig, axes = plt.subplots(1, 4, figsize=(12, 4.4))
    fig.suptitle("RAGAS metrics by retrieval method", fontsize=14, fontweight="bold")
    metric_values = {(row["mode"], row["metric"]): row["value"] for row in ragas_rows}
    for axis, metric in zip(axes, metric_names):
        axis.set_title(metric, fontsize=10)
        axis.set_ylim(0, 1)
        axis.set_xticks(range(3), MODES, rotation=20)
        axis.grid(axis="y", color="#e4e7eb", linewidth=0.8)
        for index, mode in enumerate(MODES):
            key = metric.casefold().replace(" ", "_")
            value = metric_values.get((mode, key))
            if value is None:
                axis.text(index, 0.5, "N/A", ha="center", va="center", color="#5c6570", fontsize=11)
            else:
                axis.bar(index, value, color=colors[mode], width=0.58)
                axis.text(index, min(0.97, value + 0.025), f"{value:.3f}", ha="center", fontsize=8)
        axis.spines[["top", "right"]].set_visible(False)
    fig.text(0.5, 0.02, "No verified labels or authorized LLM judge for this run",
             ha="center", color="#5c6570", fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 0.9))
    fig.savefig(output_dir / "ragas_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(MODES))
    means = [next(row["latency_mean_ms"] for row in comparison if row["mode"] == mode) for mode in MODES]
    p95 = [next(row["latency_p95_ms"] for row in comparison if row["mode"] == mode) for mode in MODES]
    axis.bar(x, means, color=[colors[mode] for mode in MODES], width=0.58, label="Mean")
    axis.scatter(x, p95, color="#20262e", marker="D", zorder=3, label="P95")
    axis.set_xticks(x, MODES)
    axis.set_ylabel("Retrieval latency (ms)")
    axis.set_title("WP6E retrieval latency after warm-up")
    axis.grid(axis="y", color="#e4e7eb", linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "latency_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    valid = [next(row["citation_valid_rate"] for row in comparison if row["mode"] == mode) for mode in MODES]
    axis.bar(x, valid, color=[colors[mode] for mode in MODES], width=0.58)
    axis.set_xticks(x, MODES)
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Valid metadata fraction")
    axis.set_title("Citation metadata validation")
    axis.grid(axis="y", color="#e4e7eb", linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    for position, row in enumerate(comparison):
        axis.text(position, min(1.02, valid[position] + 0.025),
                  f"{row['citation_valid_count']}/{row['citation_checked']}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "citation_validation.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def run(questions_path, config_path, output_dir, repeats):
    import torch
    import ragas
    from ragas import EvaluationDataset

    questions = _read_questions(questions_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    vectors_before = _vector_snapshot(config_path)
    service = create_service(config_path=config_path)
    try:
        # Attach the authorized live chunks to Chroma without rebuilding or writing vectors.
        dense = service.dense

        def attach_existing_vectors(chunks):
            dense.clear()
            dense.chunks = tuple(sorted({chunk.chunk_id: chunk for chunk in chunks}.values(),
                                        key=lambda chunk: chunk.chunk_id))
            dense.reused_count = len(dense.chunks)

        dense.index = attach_existing_vectors
        # Exclude the only deliberate warm-up from latency statistics.
        service.search("viêm da cơ địa", "Dense", 10)
        warmup_completed = True
        rows, timings = [], defaultdict(list)
        for question in questions:
            for mode in MODES:
                response = None
                for _ in range(repeats):
                    started = time.perf_counter()
                    response = service.search(question["query"], mode, 10)
                    timings[mode].append((time.perf_counter() - started) * 1000)
                hits = response.hits
                chunk_ids = [hit.chunk.chunk_id for hit in hits]
                verified = question.get("review_status") == "verified"
                relevant_chunks = question.get("relevant_chunk_ids", []) if verified else []
                if verified:
                    precision, recall = precision_recall_at_k(chunk_ids, relevant_chunks, 5)
                    mrr, ndcg = mrr_at_k(chunk_ids, relevant_chunks, 10), ndcg_at_k(chunk_ids, relevant_chunks, 5)
                else:
                    precision = recall = mrr = ndcg = None
                summary, sources = summarize_evidence(question["query"], hits, limit=5)
                excerpts = list(sources)
                evidence_checks = validate_evidence_sources(excerpts, hits)
                citation_checks = [{"chunk_id": hit.chunk.chunk_id,
                                    "metadata_valid": citation_metadata_valid(hit.chunk),
                                    "content_verified": False,
                                    "document_id": hit.chunk.document_id,
                                    "pdf_page": hit.chunk.citation.get("pdf_page"),
                                    "pdf_path": hit.chunk.citation.get("pdf_path"),
                                    "source": hit.chunk.citation.get("source", {}).get("locator")}
                                   for hit in hits]
                valid_excerpt_count = sum(check["exact_span_valid"] for check in evidence_checks)
                row = {
                    "question_id": question["question_id"],
                    "user_input": question["query"],
                    "mode": mode,
                    "status": response.status,
                    "retrieved_contexts": [hit.chunk.text for hit in hits],
                    "retrieved_context_ids": chunk_ids,
                    "retrieved_document_ids": [hit.chunk.document_id for hit in hits],
                    "response": summary,
                    "reference": question.get("reference") if verified else None,
                    "relevant_document_ids": question.get("relevant_document_ids", []) if verified else [],
                    "relevant_chunk_ids": relevant_chunks,
                    "source_page": question.get("source", {}).get("pdf_page"),
                    "human_review_status": question.get("review_status", "pending_review"),
                    "evaluation_scope": "extractive_summary",
                    "out_of_scope": bool(question.get("out_of_scope", False)),
                    "precision_at_5": precision,
                    "recall_at_5": recall,
                    "mrr_at_10": mrr,
                    "ndcg_at_5": ndcg,
                    "latency_ms": timings[mode][-1],
                    "citations": citation_checks,
                    "evidence_excerpts": excerpts,
                    "evidence_checks": evidence_checks,
                    "evidence_abstained": not bool(excerpts),
                    "evidence_coverage_lexical": _coverage(question["query"], excerpts),
                    "exact_evidence_span_rate": (valid_excerpt_count / len(excerpts) if excerpts else None),
                    "citation_consistency_rate": (sum(item["citation_consistent"] for item in evidence_checks)
                                                  / len(evidence_checks) if evidence_checks else None),
                    "unsupported_excerpt_rate": (1 - valid_excerpt_count / len(excerpts) if excerpts else None),
                }
                rows.append(row)

        metric_rows = _ragas_metric_rows(rows)
        # Validate the canonical RAGAS dataset schema even when every reference is unavailable.
        ragas_dataset = EvaluationDataset.from_list([
            {"user_input": row["user_input"], "retrieved_contexts": row["retrieved_contexts"],
             "response": row["response"], "reference": row["reference"]}
            for row in rows
        ])

        comparison = []
        for mode in MODES:
            mode_rows = [row for row in rows if row["mode"] == mode]
            citation_values = [check["metadata_valid"] for row in mode_rows for check in row["citations"]]
            measured = timings[mode]
            comparison.append({
                "mode": mode,
                "question_count": len(mode_rows),
                "verified_question_count": sum(row["human_review_status"] == "verified" for row in mode_rows),
                "precision_at_5": mean_or_none([row["precision_at_5"] for row in mode_rows]),
                "recall_at_5": mean_or_none([row["recall_at_5"] for row in mode_rows]),
                "mrr_at_10": mean_or_none([row["mrr_at_10"] for row in mode_rows]),
                "ndcg_at_5": mean_or_none([row["ndcg_at_5"] for row in mode_rows]),
                "retrieval_metric_status": "measured" if any(
                    row["human_review_status"] == "verified" and row["relevant_chunk_ids"]
                    for row in mode_rows) else "N/A_no_verified_ground_truth",
                "latency_mean_ms": statistics.mean(measured),
                "latency_median_ms": statistics.median(measured),
                "latency_p95_ms": sorted(measured)[max(0, int(0.95 * len(measured)) - 1)],
                "latency_measurements": len(measured),
                "citation_valid_count": sum(citation_values),
                "citation_checked": len(citation_values),
                "citation_valid_rate": sum(citation_values) / len(citation_values) if citation_values else None,
            })
        all_citations = [citation for row in rows for citation in row["citations"]]
        all_evidence = [check for row in rows for check in row["evidence_checks"]]
        oos_rows = [row for row in rows if row["out_of_scope"]]
        evidence_by_mode = []
        for mode in MODES:
            mode_rows = [row for row in rows if row["mode"] == mode]
            mode_checks = [check for row in mode_rows for check in row["evidence_checks"]]
            flagged_oos = [row for row in mode_rows if row["out_of_scope"]]
            evidence_by_mode.append({
                "mode": mode,
                "lexical_coverage_mean": mean_or_none([row["evidence_coverage_lexical"] for row in mode_rows]),
                "abstention_rate": (sum(row["evidence_abstained"] for row in mode_rows) / len(mode_rows)
                                     if mode_rows else None),
                "flagged_out_of_scope_abstention_rate": (
                    sum(row["evidence_abstained"] for row in flagged_oos) / len(flagged_oos)
                    if flagged_oos else None),
                "exact_span_valid_rate": (sum(item["exact_span_valid"] for item in mode_checks) / len(mode_checks)
                                          if mode_checks else None),
                "citation_consistency_rate": (sum(item["citation_consistent"] for item in mode_checks)
                                              / len(mode_checks) if mode_checks else None),
                "unsupported_excerpt_rate": (sum(not item["exact_span_valid"] for item in mode_checks)
                                             / len(mode_checks) if mode_checks else None),
                "excerpt_count": len(mode_checks),
            })
        vectors_after = _vector_snapshot(config_path)
        report = {
            "schema": "wp-ragas-1.1",
            "run_metadata": {
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "python_version": platform.python_version(),
                "ragas_version": ragas.__version__,
                "evaluation_dataset_sha256": _sha256_file(questions_path),
                "rag_config_sha256": _sha256_file(config_path),
                "corpus_fingerprint": _corpus_fingerprint(),
                "embedding_fingerprint": vectors_before["embedding_fingerprint"],
                "evaluation_dataset_samples": len(ragas_dataset),
                "embedding_model": "intfloat/multilingual-e5-small",
                "embedding_dimensions": 384,
                "embedding_device": "cpu",
                "torch_version": torch.__version__,
                "torch_threads": torch.get_num_threads(),
                "cpu": platform.processor(),
                "logical_cpu_count": os.cpu_count(),
                "memory": _memory_gib(),
                "device": "CPU",
                "top_k": 10,
                "repeats": repeats,
                "warmup": "one Dense query, excluded from measured latency",
                "judge": {"provider": None, "model": None, "status": "N/A_not_run",
                          "reason": "no approved outbound medical-data evaluation in this run"},
                "generation_enabled": False,
                "evaluation_scope": "retrieval_and_extractive_summary",
            },
            "dataset_summary": {
                "question_count": len(questions),
                "verified_count": sum(q.get("review_status") == "verified" for q in questions),
                "pending_review_count": sum(q.get("review_status") != "verified" for q in questions),
                "out_of_scope_marked_count": sum(bool(q.get("out_of_scope")) for q in questions),
            },
            "corpus": {"documents": 8, "chunks": vectors_before["count"],
                       "embedding_model": "intfloat/multilingual-e5-small", "dimensions": 384,
                       "chunking": "config/rag.local.json, unchanged"},
            "vector_integrity": {"before": vectors_before, "after": vectors_after,
                                 "unchanged_during_evaluation": vectors_before == vectors_after,
                                 "baseline_exactly_recovered": vectors_before["sha256"] ==
                                 "4390576688451c680c3e242ca880be166145376d1c70a4ff8ec74dae28bbbe10",
                                 "warmup_completed": warmup_completed},
            "retrieval_comparison": comparison,
            "ragas_metrics": metric_rows,
            "citation_summary": {"checked": len(all_citations),
                                 "valid_metadata": sum(item["metadata_valid"] for item in all_citations),
                                 "invalid_metadata": sum(not item["metadata_valid"] for item in all_citations),
                                 "content_verified": 0},
            "evidence_summary": {
                "scope": "extractive_summary",
                "excerpt_count": len(all_evidence),
                "invalid_exact_spans": sum(not item["exact_span_valid"] for item in all_evidence),
                "citation_inconsistencies": sum(not item["citation_consistent"] for item in all_evidence),
                "out_of_scope_summary_abstention_rate": (
                    sum(row["evidence_abstained"] for row in oos_rows) / len(oos_rows) if oos_rows else None),
                "out_of_scope_status": "provisional_question_metadata_pending_review",
                "unsupported_excerpt_rate": (sum(not item["exact_span_valid"] for item in all_evidence)
                                             / len(all_evidence) if all_evidence else None),
                "coverage_definition": "fraction of non-stopword query tokens present in extracted excerpts",
                "by_mode": evidence_by_mode,
            },
            "rows": rows,
        }
        _write_csv(output_dir / "per_question.csv", rows, list(rows[0]))
        _write_csv(output_dir / "retrieval_comparison.csv", comparison, list(comparison[0]))
        _write_csv(output_dir / "ragas_metrics.csv", metric_rows, list(metric_rows[0]))
        (output_dir / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        _plot_results(output_dir, comparison, metric_rows)
        report_path = ROOT / "docs" / "RAGAS_EVALUATION_REPORT.md"
        report_path.write_text(_build_report(report), encoding="utf-8")
        return report
    finally:
        service.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=ROOT / "data/rag/evaluation_questions.json")
    parser.add_argument("--config", type=Path, default=ROOT / "config/rag.local.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runtime/rag/ragas_evaluation")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1 or args.repeats > 20:
        parser.error("--repeats must be between 1 and 20")
    load_environment()
    report = run(args.questions.resolve(), args.config.resolve(), args.output_dir.resolve(), args.repeats)
    print(json.dumps({"questions": report["dataset_summary"],
                      "retrieval_comparison": report["retrieval_comparison"],
                      "citation_summary": report["citation_summary"],
                      "vector_integrity": report["vector_integrity"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
