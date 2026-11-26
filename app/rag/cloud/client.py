"""Bounded OpenRouter transport. Errors and telemetry contain codes/numbers only."""

import math
import os
import time
from dataclasses import dataclass, field

import httpx

from app.rag.config import CloudConfig
from app.rag.models import ModelUnavailable, RAGError

BASE_URL = "https://openrouter.ai/api/v1"
ENDPOINTS = frozenset({"/embeddings", "/rerank", "/chat/completions"})


@dataclass
class Operation:
    requests: int = 0
    usage: dict = field(default_factory=dict)


def routing(provider):
    route = {"allow_fallbacks": False}
    if provider is not None:
        route["only"] = [provider]
    return route


class OpenRouterClient:
    def __init__(self, settings: CloudConfig | None = None, *, transport=None, sleep=time.sleep):
        self.settings = settings or CloudConfig()
        # Never accept a key or endpoint from JSON/browser inputs.
        self._http = httpx.Client(transport=transport, follow_redirects=False, trust_env=False)
        self._sleep = sleep
        self.operation = Operation()
        self.total_requests = 0
        self.total_request_limit = None  # Optional tighter lifetime cap for live acceptance.

    def begin(self):
        self.operation = Operation()

    def close(self):
        self._http.close()

    def require_capacity(self, count):
        if (self.operation.requests + count > self.settings.max_requests_per_operation
                or self.total_request_limit is not None and self.total_requests + count > self.total_request_limit):
            raise RAGError("cloud_request_budget_exceeded")

    def validate_input(self, texts, max_bytes=None):
        if (any(not isinstance(t, str) or not t.strip() for t in texts)
                or sum(len(t) for t in texts) > self.settings.max_input_chars
                or max_bytes is not None and any(len(t.encode("utf-8")) > max_bytes for t in texts)):
            raise RAGError("cloud_input_limit_exceeded")

    def post(self, endpoint, payload, *, timeout, authorize):
        if endpoint not in ENDPOINTS:
            raise RAGError("invalid_cloud_endpoint")
        policy = self.settings.retry_policy
        for attempt in range(policy.max_retries + 1):
            # Authorization re-runs before EVERY attempt, including retries.
            authorize()
            key = os.environ.get("OPENROUTER_API_KEY")
            if not key or not key.strip() or key == "replace_with_your_openrouter_key":
                raise ModelUnavailable("openrouter_key_missing")
            self.require_capacity(1)
            self.operation.requests += 1
            self.total_requests += 1
            code, retry = "openrouter_transport_failed", False
            try:
                response = self._http.post(BASE_URL + endpoint, json=payload,
                                           headers={"Authorization": "Bearer " + key}, timeout=timeout)
                status = response.status_code
                if status == 200:
                    if len(response.content) > 8 * 1024 * 1024:
                        raise RAGError("openrouter_response_limit_exceeded")
                    try:
                        data = response.json()
                    except (ValueError, UnicodeError):
                        raise RAGError("openrouter_invalid_json") from None
                    if not isinstance(data, dict) or "error" in data:
                        raise RAGError("openrouter_invalid_response")
                    # Whitelist numeric accounting. Never retain arbitrary provider strings.
                    usage = data.get("usage", {})
                    if isinstance(usage, dict):
                        for name in ("prompt_tokens", "completion_tokens", "total_tokens", "search_units", "cost"):
                            v = usage.get(name)
                            if type(v) in (int, float) and math.isfinite(v) and v >= 0:
                                self.operation.usage[name] = self.operation.usage.get(name, 0) + v
                    return data
                code = {401: "openrouter_unauthorized", 402: "openrouter_payment_required",
                        403: "openrouter_forbidden", 404: "openrouter_model_or_endpoint_not_found",
                        429: "openrouter_rate_limited"}.get(status, "openrouter_http_error")
                if 500 <= status <= 599:
                    code = "openrouter_server_error"
                retry = status == 429 or 500 <= status <= 599
            except RAGError:
                raise
            except httpx.TimeoutException:
                code, retry = "openrouter_timeout", True
            except httpx.TransportError:
                code, retry = "openrouter_transport_failed", True
            except (httpx.HTTPError, ValueError):
                raise ModelUnavailable("openrouter_transport_failed") from None
            if not retry or attempt == policy.max_retries:
                raise ModelUnavailable(code) from None
            self._sleep(min(policy.max_backoff_seconds, policy.backoff_seconds * 2 ** attempt))
