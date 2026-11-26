"""Explicit, five-request SYNTHETIC DEMO acceptance; no opt-in means no network.

Use an OpenRouter key with a small spending limit. Request/token caps bound work,
but pricing and actual account-side dollar limits belong to the operator.
"""

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

from app.rag.config import RAGConfig, RetryPolicy
from app.rag.runtime import create_service, load_environment


def run_acceptance(*, live=False, budget_confirmed=False, settings=None):
    report = {"status": "BLOCKED", "corpus": "SYNTHETIC DEMO", "requests": 0,
              "stages": {}, "usage": {}}
    if not live or not budget_confirmed:
        report["blocker"] = "explicit_live_and_operator_budget_opt_in_required"
        return report
    load_environment()
    if not os.environ.get("OPENROUTER_API_KEY"):
        report["blocker"] = "openrouter_key_missing"
        return report
    settings = settings or RAGConfig.from_dict({"embedding": {"backend": "openrouter"}})
    settings = replace(settings,
        embedding=replace(settings.embedding, backend="openrouter", batch_size=16),
        cloud=replace(settings.cloud, synthetic_live_opt_in=True, max_requests_per_operation=5,
                      max_input_chars=8000, max_index_chunks=16, retry_policy=RetryPolicy(max_retries=0)),
        reranking=replace(settings.reranking, enabled=True, candidates=5, top_n=3),
        generation=replace(settings.generation, enabled=True, max_evidence=3, max_tokens=512))
    service = None
    try:
        service = create_service(demo=True, settings=settings)
        service.cloud_client.total_request_limit = 5
        dense = service.search("source provenance evidence", "Dense", 3, rerank=False)
        report["stages"]["WP6B.1_embedding_dense"] = "PASS" if dense.status == "OK" and dense.hits else "BLOCKED"
        for key, value in dense.usage.items():
            if key != "requests":
                report["usage"][key] = report["usage"].get(key, 0) + value
        if dense.status != "OK":
            report["blocker"] = dense.status
            return report
        result = service.search("What does source provenance identify in this synthetic example?", "Hybrid", 3,
                                rerank=True, ask=True)
        for key, value in result.usage.items():
            if key != "requests":
                report["usage"][key] = report["usage"].get(key, 0) + value
        passed = result.status == "OK" and bool(result.answer and result.sources and result.hits)
        for stage in ("WP6B.1_hybrid", "WP6B.2_reranking", "WP6B.3_grounded_generation"):
            report["stages"][stage] = "PASS" if passed else "BLOCKED"
        if passed:
            report["status"] = "PASS"
        else:
            report["blocker"] = result.status
    except Exception:
        report["blocker"] = "live_acceptance_configuration_or_runtime_failed"
    finally:
        if service is not None:
            report["requests"] = service.cloud_client.total_requests
            service.close()
    return report


def main():
    parser = argparse.ArgumentParser(description="Opt-in paid OpenRouter acceptance using fixed synthetic data only")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--budget-confirmed", action="store_true",
                        help="Operator confirms a small account/key spending limit has been set")
    parser.add_argument("--settings", type=Path)
    args = parser.parse_args()
    try:
        # Missing opt-in exits before opening settings, dotenv, or an HTTP client.
        settings = RAGConfig.load(args.settings) if args.live and args.budget_confirmed and args.settings else None
        report = run_acceptance(live=args.live, budget_confirmed=args.budget_confirmed, settings=settings)
    except Exception:
        report = {"status": "BLOCKED", "requests": 0, "blocker": "invalid_operator_settings"}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
