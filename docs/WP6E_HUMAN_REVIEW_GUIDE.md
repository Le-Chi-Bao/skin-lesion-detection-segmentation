# WP6E Human Review Guide

Use [evaluation_review_template.csv](../data/rag/evaluation_review_template.csv) as the annotation sheet. The 15 selected questions cover seven source documents and include two questions currently marked out of scope. Their current status remains `pending_review` until a reviewer completes and approves the row.

## Review Procedure

1. Read the original PDF page(s) and the retrieved chunk text for the question. Do not infer an answer from the question's existing source hint.
2. Record a concise reference answer supported by the source. Preserve quantities, units, population, study period, and uncertainty as printed.
3. Enter relevant document IDs and exact relevant chunk IDs. Use the chunk IDs in `runtime/rag/ragas_evaluation/results.json` or `per_question.csv`. Multiple IDs are comma-separated.
4. Record the PDF page that supports the reference. Use `in_scope` only when the corpus can answer the question. Use `out_of_scope` when the requested information is absent or asks for current guidance not established by the corpus.
5. Set `expected_abstention` to `true` only for an approved out-of-scope row. Record reviewer identity, ISO 8601 review time, and concise notes.
6. Set `human_review_status` to `verified` only after all labels and the citation have been checked. Otherwise leave the row `pending_review` and explain what remains uncertain.

Never mark a question verified based only on retrieved ranking, lexical overlap, an LLM judge, or another method's output. Do not add clinical recommendations. For out-of-scope rows, leave `reference` and relevant IDs empty and explicitly record the abstention expectation.

## Label Fields

| Field | Annotation |
|---|---|
| `reviewer_id` | Reviewer identifier used by the project. |
| `reviewed_at` | ISO 8601 timestamp with timezone. |
| `human_review_status` | `pending_review` or `verified`. |
| `scope_label` | `in_scope` or `out_of_scope`. |
| `reference` | Short evidence-backed answer, empty for approved out-of-scope questions. |
| `relevant_document_ids` | Comma-separated document IDs supporting the reference. |
| `relevant_chunk_ids` | Comma-separated exact chunk IDs supporting the reference. |
| `source_page` | PDF page number(s) supporting the reference. |
| `expected_abstention` | `true` or `false`, set after scope review. |
| `review_notes` | Ambiguities, extraction issues, or rationale for the label. |

After review, copy approved annotations into a version-controlled evaluation dataset or a separate reviewed file. The runner only counts rows whose WP6E input has `review_status=verified` and non-empty relevant chunk IDs as ground truth.
