"""Read-only auth audit; paid embedding probes require new, explicit opt-ins.

Never print account data, provider bodies, headers, vectors or credentials. No
corpus, Gradio session, reranker or generator is constructed by this command.
"""

import argparse
import json
import math
import os
from pathlib import Path

import httpx

from app.rag.cloud.client import BASE_URL, OpenRouterClient
from app.rag.cloud.embeddings import OpenRouterEmbedder
from app.rag.config import RAGConfig
from app.rag.models import RAGError
from app.rag.runtime import load_environment

SYNTHETIC_INPUT = "Synthetic embedding test"
ALLOWED_REQUESTS = frozenset({("GET", "/key"), ("GET", "/embeddings/models"), ("POST", "/embeddings")})
PARTIAL = "PARTIAL / ROOT CAUSE NOT CONFIRMED"


def _http_code(status):
    if status == 200:
        return "OK"
    if 500 <= status < 600:
        return "openrouter_server_error"
    return {401: "openrouter_unauthorized", 402: "openrouter_payment_required",
            403: "openrouter_forbidden", 404: "openrouter_model_or_endpoint_not_found",
            429: "openrouter_rate_limited"}.get(status, "openrouter_http_error")


def _numeric_usage(data):
    usage, clean = data.get("usage"), {}
    if isinstance(usage, dict):
        for name in ("prompt_tokens", "completion_tokens", "total_tokens", "search_units", "cost"):
            value = usage.get(name)
            try:
                if type(value) in (int, float) and math.isfinite(value) and value >= 0:
                    clean[name] = value
            except OverflowError:
                pass
    return clean


def _redact(value, key):
    # Defense in depth if a config model/provider string accidentally contains a key.
    if isinstance(value, str):
        return value.replace(key, "[REDACTED]") if key else value
    if isinstance(value, dict):
        return {k: _redact(v, key) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(v, key) for v in value]
    return value


def _capture_production(settings):
    """Capture the real adapter's one-input payload/header checks, using mock HTTP only."""
    class Captured(Exception):
        pass

    captured = {}
    def capture(request):
        captured["payload"] = json.loads(request.content)
        captured["wire"] = {
            "method_post": request.method == "POST",
            "official_embedding_url": str(request.url) == BASE_URL + "/embeddings",
            "bearer_matches_loaded_key": request.headers.get("Authorization") == (
                "Bearer " + os.environ.get("OPENROUTER_API_KEY", "")),
            "json_content_type": request.headers.get("Content-Type") == "application/json",
        }
        raise Captured

    client = OpenRouterClient(settings.cloud, transport=httpx.MockTransport(capture))
    try:
        captured["client"] = {"trust_env_disabled": client._http.trust_env is False,
                              "redirects_disabled": client._http.follow_redirects is False,
                              "no_request_hooks": not client._http.event_hooks["request"]}
        embedder = OpenRouterEmbedder(settings.embedding, client)
        # Only the literal authored above is authorized, and transport is a mock.
        embedder.authorize = lambda: None
        try:
            embedder.embed_documents([SYNTHETIC_INPUT])
        except Captured:
            return captured["payload"], {**captured["wire"], **captured["client"]}
        raise RAGError("diagnostic_payload_capture_failed")
    finally:
        client.close()


def _embedding_summary(data, model):
    """Independent validation for minimal float-vector responses; never returns a vector."""
    summary = {"model_requested": model, "vector_dimension": None, "usage": _numeric_usage(data)}
    rows = data.get("data")
    if (data.get("model") != model or not isinstance(rows, list) or len(rows) != 1
            or not isinstance(rows[0], dict) or type(rows[0].get("index")) is not int
            or rows[0]["index"] != 0):
        return {**summary, "code": "diagnostic_invalid_embedding_response"}
    vector = rows[0].get("embedding")
    try:
        valid = (isinstance(vector, list) and bool(vector)
                 and all(type(v) in (int, float) and math.isfinite(v) for v in vector)
                 and any(v != 0 for v in vector))
    except OverflowError:
        valid = False
    if not valid:
        return {**summary, "code": "diagnostic_invalid_embedding_response"}
    return {**summary, "code": "OK", "vector_dimension": len(vector)}


def run_diagnostics(*, settings=None, live=False, include_production=False, transport=None):
    if type(live) is not bool or type(include_production) is not bool or include_production and not live:
        raise RAGError("production_probe_requires_separate_live_opt_in")
    settings = settings or RAGConfig.from_dict({"embedding": {"backend": "openrouter"}})
    before = os.environ.get("OPENROUTER_API_KEY")
    dotenv_path = Path(__file__).resolve().parents[3] / ".env"
    dotenv_present = dotenv_path.is_file()
    load_environment()
    key = os.environ.get("OPENROUTER_API_KEY")
    key_present = bool(key and key.strip() and key != "replace_with_your_openrouter_key")
    report = {
        "status": PARTIAL, "root_cause_confirmed": False, "safe_status": "BLOCKED",
        "mode": "live" if live else "safe", "safe_requests": 0, "paid_requests": 0,
        "environment": {"project_dotenv_present": dotenv_present, "startup_loader_called": True,
                        "key_present_before_load": before is not None,
                        "key_present": key_present,
                        "existing_environment_preserved": before is None or before == key,
                        "key_source": "process_environment" if before is not None else (
                            "startup_dotenv" if key_present else "missing"),
                        "key_has_surrounding_whitespace": bool(key and key != key.strip())},
        "client": {"base_url": BASE_URL, "trust_env": False, "follow_redirects": False,
                   "retries": 0, "max_paid_requests": 2 if include_production else int(live)},
        "configuration": {"backend": settings.embedding.backend, "model": settings.embedding.model,
                          "provider": settings.embedding.provider},
        "authentication": {"http_status": None, "code": "NOT_RUN"},
        "model_catalog": {"http_status": None, "code": "NOT_RUN", "model_listed": None},
        "minimal_embedding": {"http_status": None, "code": "NOT_RUN", "reason": "new_live_opt_in_required"},
        "production_embedding": {"http_status": None, "code": "NOT_RUN",
                                 "reason": "additional_production_opt_in_required"},
    }
    if not key_present:
        report["blocker"] = "openrouter_key_missing"
        return _redact(report, key)
    if any(ord(c) < 33 or ord(c) > 126 for c in key):
        report["blocker"] = "diagnostic_key_format_invalid"
        return _redact(report, key)
    if settings.embedding.backend != "openrouter":
        report["blocker"] = "diagnostic_requires_openrouter_config"
        return _redact(report, key)

    try:
        production, wire = _capture_production(settings)
    except Exception:
        report["blocker"] = "diagnostic_production_payload_capture_failed"
        return _redact(report, key)
    report["production_wire_audit"] = wire
    report["payload_comparison"] = {
        "minimal_fields": ["input", "model"], "minimal_input_type": "string",
        "production_fields": sorted(production), "production_input_type": "list[string]",
        "production_input_count": len(production["input"]),
        "production_provider": production["provider"],
        "production_encoding_format": production["encoding_format"],
        "configured_prefix_applied": bool(settings.embedding.passage_prefix),
        "dimensions": production.get("dimensions"), "input_type": production.get("input_type"),
        "synthetic_only": True,
    }
    if not all(wire.values()):
        report["blocker"] = "diagnostic_production_wire_mismatch"
        return _redact(report, key)

    client = OpenRouterClient(settings.cloud, transport=transport)
    def send(method, path, payload=None):
        if (method, path) not in ALLOWED_REQUESTS:
            raise RAGError("diagnostic_endpoint_denied")
        paid = method == "POST"
        if paid and (not live or report["paid_requests"] >= report["client"]["max_paid_requests"]):
            raise RAGError("diagnostic_request_budget_exceeded")
        if not paid and report["safe_requests"] >= 2:
            raise RAGError("diagnostic_request_budget_exceeded")
        report["paid_requests" if paid else "safe_requests"] += 1
        try:
            response = client._http.request(method, BASE_URL + path, json=payload,
                headers={"Authorization": "Bearer " + key}, timeout=settings.embedding.request_timeout)
            result = {"http_status": response.status_code, "code": _http_code(response.status_code)}
            if response.status_code != 200:
                return result, {}
            if len(response.content) > 8 * 1024 * 1024:
                return {**result, "code": "openrouter_response_limit_exceeded"}, {}
            try:
                data = response.json()
            except (ValueError, UnicodeError):
                return {**result, "code": "openrouter_invalid_json"}, {}
            if not isinstance(data, dict) or "error" in data:
                return {**result, "code": "openrouter_invalid_response"}, {}
            return result, data
        except httpx.TimeoutException:
            return {"http_status": None, "code": "openrouter_timeout"}, {}
        except Exception:
            return {"http_status": None, "code": "openrouter_transport_failed"}, {}

    try:
        report["authentication"], auth = send("GET", "/key")
        if report["authentication"]["code"] != "OK" or not isinstance(auth.get("data"), dict):
            report["blocker"] = (report["authentication"]["code"] if report["authentication"]["code"] != "OK"
                                 else "diagnostic_invalid_key_response")
            return _redact(report, key)
        report["model_catalog"], catalog = send("GET", "/embeddings/models")
        rows = catalog.get("data")
        listed = (any(isinstance(r, dict) and r.get("id") == settings.embedding.model for r in rows)
                  if isinstance(rows, list) else None)
        report["model_catalog"]["model_listed"] = listed
        report["safe_status"] = "PASS" if report["model_catalog"]["code"] == "OK" and listed else "BLOCKED"
        if not live:
            report["blocker"] = "minimal_embedding_requires_new_operator_opt_in"
            return _redact(report, key)
        if listed is False:
            report["blocker"] = "configured_model_not_listed_no_automatic_substitution"
            return _redact(report, key)
        # A deliberately omits optional routing fields, exactly as the minimal contract.
        result, data = send("POST", "/embeddings", {"model": settings.embedding.model, "input": SYNTHETIC_INPUT})
        if result["code"] == "OK":
            result.update(_embedding_summary(data, settings.embedding.model))
        report["minimal_embedding"] = result
        if result["code"] != "OK":
            report["production_embedding"]["reason"] = "minimal_embedding_failed_stop"
            report["blocker"] = result["code"]
            return _redact(report, key)
        if not include_production:
            report["blocker"] = "production_embedding_requires_additional_operator_opt_in"
            return _redact(report, key)
        result, data = send("POST", "/embeddings", production)
        if result["code"] == "OK":
            result.update(_embedding_summary(data, settings.embedding.model))
            # Preserve the adapter's configured dimension/provider expectations.
            if (result["code"] == "OK" and settings.embedding.dimensions is not None
                    and result["vector_dimension"] != settings.embedding.dimensions):
                result["code"] = "embedding_dimension_mismatch"
            actual = data.get("provider")
            if (settings.embedding.provider is not None and actual is not None
                    and (not isinstance(actual, str)
                         or actual.casefold() != settings.embedding.provider.casefold())):
                result["code"] = "openrouter_provider_mismatch"
        report["production_embedding"] = result
        report["embedding_requests_verified"] = result["code"] == "OK"
        report["blocker"] = ("reported_401_not_reproduced_root_cause_still_unconfirmed"
                             if result["code"] == "OK" else result["code"])
        return _redact(report, key)
    finally:
        client.close()


def main():
    parser = argparse.ArgumentParser(description="Safe OpenRouter authentication/embedding differential diagnostics")
    parser.add_argument("--settings", type=Path)
    parser.add_argument("--live", action="store_true", help="New opt-in: at most ONE minimal synthetic embedding POST")
    parser.add_argument("--include-production", action="store_true",
                        help="Additional opt-in: ONE production-payload POST, only if minimal POST passes")
    args = parser.parse_args()
    if args.include_production and not args.live:
        parser.error("--include-production requires --live and an additional operator opt-in")
    try:
        settings = RAGConfig.load(args.settings) if args.settings else None
        report = run_diagnostics(settings=settings, live=args.live, include_production=args.include_production)
    except Exception:
        report = {"status": PARTIAL, "safe_status": "BLOCKED", "blocker": "diagnostic_configuration_or_runtime_failed"}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    requested_pass = (not args.live or report.get("minimal_embedding", {}).get("code") == "OK")
    if args.include_production:
        requested_pass = report.get("production_embedding", {}).get("code") == "OK"
    return 0 if report.get("safe_status") == "PASS" and requested_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
