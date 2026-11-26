"""Reranking maps provider indexes back to verified, immutable local hits."""

import math

from app.rag.cloud.client import routing
from app.rag.models import RAGError, SearchHit


class OpenRouterReranker:
    def __init__(self, config, client):
        self.config, self.client = config, client

    def rerank(self, query, hits, *, authorize):
        hits = tuple(hits)
        if not hits:
            return ()
        if len(hits) > self.config.candidates or len({h.chunk.chunk_id for h in hits}) != len(hits):
            raise RAGError("invalid_rerank_candidates")
        self.client.validate_input([query] + [h.chunk.text for h in hits], self.config.max_input_bytes)
        count = min(self.config.top_n, len(hits))
        data = self.client.post("/rerank", {"model": self.config.model, "query": query,
                                "documents": [h.chunk.text for h in hits], "top_n": count,
                                "provider": routing(self.config.provider)},
                                timeout=self.config.request_timeout, authorize=authorize)
        if data.get("model") != self.config.model:
            raise RAGError("rerank_model_mismatch")
        results = data.get("results")
        if not isinstance(results, list) or len(results) != count:
            raise RAGError("invalid_rerank_results")
        seen, ranked = set(), []
        for row in results:
            if not isinstance(row, dict):
                raise RAGError("invalid_rerank_results")
            index, score = row.get("index"), row.get("relevance_score")
            if (type(index) is not int or not 0 <= index < len(hits) or index in seen
                    or type(score) not in (int, float) or not math.isfinite(score)):
                raise RAGError("invalid_rerank_results")
            seen.add(index)
            ranked.append(SearchHit(hits[index].chunk, score, "Rerank"))
        authorize()
        return tuple(sorted(ranked, key=lambda h: (-h.score, h.chunk.chunk_id)))
