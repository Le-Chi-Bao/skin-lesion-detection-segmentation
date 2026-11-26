"""WP6E retrieval evaluation for the existing local RAG corpus.

The runner treats only ``review_status=verified`` rows as ground truth. Rows marked
``pending_review`` are still retrieved and reported, but never enter metrics.
"""
from __future__ import annotations
import argparse, csv, json, math, statistics, time
from pathlib import Path
from collections import defaultdict
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.rag.runtime import create_service

def _metric_row(ranked, relevant, k):
    rel = set(relevant)
    top = ranked[:k]
    hits = [x for x in top if x in rel]
    precision = len(hits) / k
    recall = len(hits) / len(rel) if rel else None
    return precision, recall

def _mrr(ranked, relevant, k=10):
    rel = set(relevant)
    for i, value in enumerate(ranked[:k], 1):
        if value in rel:
            return 1 / i
    return 0.0

def _ndcg(ranked, relevant, k=5):
    rel = set(relevant)
    dcg = sum((1 / math.log2(i + 2)) for i, value in enumerate(ranked[:k]) if value in rel)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(rel))))
    return dcg / ideal if ideal else None

def _citation_check(hit):
    c = hit.chunk.citation
    required = ("document_id", "pdf_page", "pdf_path", "source")
    return {"valid": all(c.get(k) not in (None, "") for k in required),
            "document_id": c.get("document_id"), "pdf_page": c.get("pdf_page"),
            "pdf_path": c.get("pdf_path"), "source": c.get("source", {}).get("locator")}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=Path, default=ROOT / "data/rag/evaluation_questions.json")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "runtime/rag/wp6e")
    ap.add_argument("--repeats", type=int, default=1)
    args = ap.parse_args()
    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    service = create_service(config_path=ROOT / "config/rag.local.json")
    try:
        # One warm-up call loads tokenizer/model and opens Chroma. Warm-up is excluded.
        service.search("viêm da cơ địa", "Dense", 5)
        rows, timings = [], defaultdict(list)
        for item in questions:
            for mode in ("BM25", "Dense", "Hybrid"):
                for repeat in range(max(1, args.repeats)):
                    started = time.perf_counter()
                    response = service.search(item["query"], mode, 5)
                    elapsed = (time.perf_counter() - started) * 1000
                    timings[mode].append(elapsed)
                    if repeat != 0:
                        continue
                    hits = response.hits
                    ranked_chunks = [h.chunk.chunk_id for h in hits]
                    ranked_docs = [h.chunk.document_id for h in hits]
                    verified = item.get("review_status") == "verified"
                    relevant = item.get("relevant_chunk_ids", []) if verified else []
                    p, r = _metric_row(ranked_chunks, relevant, 5) if verified else (None, None)
                    rows.append({"question_id": item["question_id"], "query": item["query"], "mode": mode,
                        "review_status": item.get("review_status", "pending_review"), "status": response.status,
                        "precision_at_5": p, "recall_at_5": r,
                        "mrr_at_10": _mrr(ranked_chunks, relevant) if verified else None,
                        "ndcg_at_5": _ndcg(ranked_chunks, relevant) if verified else None,
                        "elapsed_ms": round(elapsed, 3), "ranked_chunk_ids": ranked_chunks,
                        "ranked_document_ids": ranked_docs,
                        "citations": [_citation_check(h) for h in hits],
                        "out_of_scope": item.get("out_of_scope", False)})
        metrics = []
        for mode in ("BM25", "Dense", "Hybrid"):
            eligible = [x for x in rows if x["mode"] == mode and x["review_status"] == "verified"]
            metrics.append({"mode": mode, "eligible_questions": len(eligible),
                **{key: (statistics.mean([x[key] for x in eligible]) if eligible else None)
                   for key in ("precision_at_5", "recall_at_5", "mrr_at_10", "ndcg_at_5")},
                "mean_elapsed_ms": statistics.mean(timings[mode]), "warmup_excluded": True,
                "cpu": True, "repeats": args.repeats})
        (args.output_dir / "per_query.csv").write_text("", encoding="utf-8")
        with (args.output_dir / "per_query.csv").open("w", newline="", encoding="utf-8-sig") as f:
            fields = list(rows[0]); writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); writer.writeheader()
            for row in rows: writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k,v in row.items()})
        report = {"schema": "wp6e-1", "questions": len(questions),
                  "pending_review": sum(x.get("review_status") != "verified" for x in questions),
                  "verified": sum(x.get("review_status") == "verified" for x in questions),
                  "metrics": metrics, "rows": rows,
                  "citation_summary": {"checked": sum(len(x["citations"]) for x in rows),
                    "valid": sum(sum(c["valid"] for c in x["citations"]) for x in rows)},
                  "conditions": {"device": "CPU", "warmup": "one Dense query excluded", "database": "existing ChromaDB", "embedding_recomputed": False}}
        (args.output_dir / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        _png(args.output_dir / "metrics.png", metrics)
        print(json.dumps({"questions": len(questions), "verified": report["verified"], "pending_review": report["pending_review"], "metrics": metrics}, ensure_ascii=False, indent=2))
    finally:
        service.close()

def _png(path, metrics):
    from PIL import Image, ImageDraw
    image = Image.new("RGB", (1100, 520), "white"); draw = ImageDraw.Draw(image)
    draw.text((30, 20), "WP6E RAG retrieval evaluation", fill="black")
    keys = ("precision_at_5", "recall_at_5", "mrr_at_10", "ndcg_at_5")
    for j, key in enumerate(keys):
        x = 40 + j * 260; draw.text((x, 65), key, fill="black")
        for i, row in enumerate(metrics):
            value = row[key] or 0; y = 440 - int(value * 320); draw.rectangle((x + i*65, y, x + i*65 + 40, 440), fill=(45, 105 + i*45, 180)); draw.text((x+i*65, 455), row["mode"], fill="black")
    image.save(path)

if __name__ == "__main__": main()
