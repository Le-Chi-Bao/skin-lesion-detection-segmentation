# WP6E RAG Evaluation

The current RAGAS evaluation report is [RAGAS_EVALUATION_REPORT.md](RAGAS_EVALUATION_REPORT.md). Detailed outputs are generated under `runtime/rag/ragas_evaluation/`. Install the pinned evaluation dependencies separately from the demo/runtime dependencies:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-rag-evaluation.txt
.venv\Scripts\python.exe scripts\rag_evaluate.py --repeats 3
```

The runner evaluates the unchanged WP6E set with BM25, Dense E5 Small, and Hybrid RRF at Top-K 10. It validates RAGAS 0.4.4 dataset structure, uses custom RAGAS collection metrics over reviewer-approved chunk IDs when available, and computes Precision@5, Recall@5, MRR@10, and nDCG@5 only for verified questions. It runs one Dense warm-up, then repeats each measured query three times by default. Dense retrieval attaches the live, authorized chunk metadata to the existing Chroma collection in read-only mode.

Output files are `results.json`, `per_question.csv`, `retrieval_comparison.csv`, `ragas_metrics.csv`, `ragas_comparison.png`, `latency_comparison.png`, and `citation_validation.png`. `N/A` is serialized as null and is never plotted as zero.

All 40 questions remain `pending_review` until a human approves the labels. The [15-question annotation template](../data/rag/evaluation_review_template.csv) and [review guide](WP6E_HUMAN_REVIEW_GUIDE.md) cover in-scope and currently marked out-of-scope cases. Copy approved reference answers and exact relevant chunk IDs into a reviewed question JSON, then pass it with `--questions`. Never mark a pending question verified based only on retrieval or an LLM judge.

Ollama is not used. OpenRouter credentials are configured locally, but the corpus consent is `local_only`, so the current run does not send questions or excerpts to that service. Faithfulness, Answer Relevancy, and unsupported claim rate remain `N/A`; Evidence Summary is reported separately as extractive summary evaluation.

The earlier runner output under `runtime/rag/wp6e/` is retained as a historical run. Its warm-up and retrieval behavior differ from this read-only pipeline. The current report records a low-order embedding hash change caused by the first evaluation attempt re-upserting cached Chroma vectors, and the original full 496-vector hash could not be proven restored from the compacted event log. The current evaluator verifies that its own read-only measurement leaves the collection hash and 496-vector count unchanged.
