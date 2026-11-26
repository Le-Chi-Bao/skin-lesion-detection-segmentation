# WP6E RAG Retrieval Evaluation

The evaluation set is stored in [data/rag/evaluation_questions.json](../data/rag/evaluation_questions.json). It contains 40 Vietnamese questions across the eight local documents and two out-of-scope questions. Each row includes document and source cross-reference fields. Every row is currently `pending_review` because the corpus metadata records `needs_human_review`, so no row is treated as ground truth.

The runner [scripts/rag_evaluate.py](../scripts/rag_evaluate.py) executes BM25, Dense E5 Small, and Hybrid RRF against the same questions. It loads the existing `config/rag.local.json` service and Chroma persistence. Chroma reuses vectors by content hash. The first Dense call is a warm-up and is excluded from timing. Retrieval runs on CPU.

Outputs are written to `runtime/rag/wp6e/per_query.csv`, `runtime/rag/wp6e/results.json`, and `runtime/rag/wp6e/metrics.png`. The initial run produced 40 pending-review rows and zero eligible questions. Precision@5, Recall@5, MRR@10, and nDCG@5 are therefore `null` for all three methods. Mean retrieval time was 252.165 ms for BM25, 275.240 ms for Dense, and 284.673 ms for Hybrid in one CPU run with one measurement per question and mode. These timings include service refresh and query embedding where applicable and are environment-specific.

Citation checks inspect every returned hit for document ID, PDF page, PDF path, and source locator. The evaluator records the result per hit. It does not claim that extracted text or page alignment is human verified. The source metadata currently identifies all eight conversions as needing manual review.

To enable metrics after manual review, a reviewer must set `review_status` to `verified` and populate `relevant_chunk_ids` using the exact chunk IDs returned by the local service. The script then computes metrics only for those rows. Out-of-scope questions remain useful for inspecting retrieval behavior but do not receive relevance scores without an explicit reviewed policy.

Validation run

```text
.venv/Scripts/python.exe scripts/rag_evaluate.py --repeats 1
```

The existing test suite should be run separately. No ChromaDB data, CV code, paid API, or model files were changed.
