"""Local smoke/demo command. No generation, API keys or network requests."""

import argparse
import json
from pathlib import Path

from app.rag.models import RAGError
from app.rag.runtime import create_service


def main() -> int:
    parser = argparse.ArgumentParser(description="WP6A local retrieval (SYNTHETIC DEMO available)")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--demo", action="store_true")
    source.add_argument("--config", type=Path)
    parser.add_argument("--mode", choices=["BM25", "Dense", "Hybrid"], default="BM25")
    parser.add_argument("--query", default="tổn thương da")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    service = None
    try:
        service = create_service(demo=args.demo, config_path=args.config)
        result = service.search(args.query, args.mode, args.top_k)
        print(json.dumps({"status": result.status, "mode": result.mode, "synthetic": result.synthetic,
                          "elapsed_ms": round(result.elapsed_ms, 2),
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
