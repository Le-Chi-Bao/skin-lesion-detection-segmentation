"""Operator CLI: capture already-granted local consent, index and report real retrieval."""

import argparse
import json
from pathlib import Path
from time import perf_counter

from app.rag.config import RAGConfig
from app.rag.local_corpus import consent_for
from app.rag.runtime import create_service


QUERIES = (
    "Đặc điểm lâm sàng và yếu tố liên quan đến viêm da cơ địa ở sinh viên là gì?",
    "Secukinumab cải thiện chất lượng cuộc sống của bệnh nhân vảy nến như thế nào?",
    "Những yếu tố nguy cơ nào liên quan đến mụn trứng cá?",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "index"])
    parser.add_argument("--config", type=Path, default=Path("config/rag.local.json"))
    parser.add_argument("--report", type=Path, default=Path("runtime/rag/local_acceptance.json"))
    args = parser.parse_args()
    settings = RAGConfig.load(args.config)
    if args.command == "prepare":
        local = settings.local_corpus
        resolve = lambda value: (args.config.parent / value).resolve()
        consent = consent_for(resolve(local.kit), resolve(local.markdown))
        path = resolve(local.approval)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Never overwrite a revocation or grant for different bytes without operator intervention.
        if path.exists():
            if json.loads(path.read_bytes()) != consent:
                raise ValueError("local_consent_already_exists_and_differs")
        else:
            path.write_text(json.dumps(consent, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("Local consent ready for exactly eight documents; page review unchanged.")
        return 0
    service = create_service(config_path=args.config)
    try:
        chunks, identities = service._refresh()
        start = perf_counter()
        service.dense.index(chunks)
        service._dense_ids = identities
        import numpy as np
        stored = service.dense.collection.get(ids=[c.chunk_id for c in chunks], include=["embeddings"])
        vectors = np.asarray(stored["embeddings"])
        norms = np.linalg.norm(vectors, axis=1)
        if vectors.shape != (len(chunks), 384) or not np.allclose(norms, 1, atol=1e-5):
            raise ValueError("local_vector_validation_failed")
        token_lengths = service.chunker._tokenizer(
            [settings.embedding.passage_prefix + c.text.strip() for c in chunks],
            truncation=False, padding=False, verbose=False)["input_ids"]
        report = {"documents": service.document_count, "chunks": len(chunks),
                  "active_vectors": len(service.dense.chunks), "stored_vectors": service.dense.collection.count(),
                  "embedded_texts": service.dense.embedded_count, "reused_vectors": service.dense.reused_count,
                  "dimensions": settings.embedding.dimensions,
                  "index_elapsed_ms": round((perf_counter() - start) * 1000, 2),
                  "vector_norm_min": float(norms.min()), "vector_norm_max": float(norms.max()),
                  "max_passage_tokens": max(map(len, token_lengths)),
                  "collection": service.dense.collection.name,
                  "review_claim": "No page review performed; source review statuses preserved.",
                  "per_document": {doc: sum(c.document_id == doc for c in chunks)
                                   for doc in sorted({c.document_id for c in chunks})}, "queries": []}
        for query in QUERIES:
            for mode in ("Dense", "Hybrid"):
                response = service.search(query, mode, 3)
                if response.status != "OK" or len(response.hits) != 3:
                    raise ValueError("local_acceptance_query_failed: " + response.status)
                report["queries"].append({"query": query, "mode": mode, "status": response.status,
                    "elapsed_ms": round(response.elapsed_ms, 2), "api_requests": response.usage["requests"],
                    "hits": [{"rank": rank, "document_id": h.chunk.document_id, "score": h.score,
                              "title": h.chunk.citation["title"], "pdf_page": h.chunk.citation["pdf_page"],
                              "source": h.chunk.citation["source"]["locator"], "text": h.chunk.text,
                              "chunk_id": h.chunk.chunk_id} for rank, h in enumerate(response.hits, 1)]})
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=True, indent=2))
        return 0
    finally:
        service.close()


if __name__ == "__main__":
    raise SystemExit(main())
