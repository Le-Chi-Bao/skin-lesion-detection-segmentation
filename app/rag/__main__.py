"""Operator CLI. Cloud demo requires explicit JSON opt-in; BM25 remains offline."""

import argparse
import json
from pathlib import Path

from app.rag.models import RAGError
from app.rag.config import RAGConfig
from app.rag.runtime import create_service


def main() -> int:
    parser = argparse.ArgumentParser(description="Medical RAG retrieval and grounded evidence")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--demo", action="store_true")
    source.add_argument("--config", type=Path)
    parser.add_argument("--mode", choices=["BM25", "Dense", "Hybrid"], default=None)
    parser.add_argument("--query", default="tổn thương da")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--settings", type=Path, help="Operator settings for --demo only")
    parser.add_argument("--rerank", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--ask", action="store_true")
    args = parser.parse_args()
    service = None
    try:
        if args.settings and not args.demo:
            raise RAGError("settings_requires_synthetic_demo")
        service = create_service(demo=args.demo, config_path=args.config,
                                 settings=RAGConfig.load(args.settings) if args.settings else None)
        defaults = service.settings.retrieval
        result = service.search(args.query, args.mode if args.mode is not None else defaults.mode,
                                args.top_k if args.top_k is not None else defaults.top_k,
                                rerank=args.rerank, ask=args.ask)
        print(json.dumps({"status": result.status, "mode": result.mode, "synthetic": result.synthetic,
                          "elapsed_ms": round(result.elapsed_ms, 2),
                          "backend": result.backend, "usage": result.usage,
                          "answer": result.answer, "sources": result.sources,
                          "hits": [{"chunk_id": h.chunk.chunk_id, "score": h.score,
                                    "text": h.chunk.text, "citation": h.chunk.citation} for h in result.hits]},
                         ensure_ascii=False, indent=2))
        return 0 if result.status in {"OK", "EMPTY_INDEX", "NO_MATCH", "EMPTY_QUERY"} else 2
    except (RAGError, OSError, ValueError, KeyError, TypeError):
        print(json.dumps({"status": "rag_configuration_or_ingestion_failed"}))
        return 2
    finally:
        if service:
            service.close()


if __name__ == "__main__":
    raise SystemExit(main())
