"""Diagnostic plumbing only; all HTTP is mocked, never proof of live embedding."""

import json
import httpx
import pytest

from app.rag.cloud import diagnostics
from app.rag.config import RAGConfig

TEST_SECRET = "diagnostic-test-secret-never-print"


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", TEST_SECRET)
    monkeypatch.setattr(diagnostics, "load_environment", lambda: None)


class Router:
    def __init__(self, auth_status=200, minimal_status=200, production_status=200):
        self.auth_status, self.minimal_status, self.production_status = auth_status, minimal_status, production_status
        self.calls = []
        self.listed = True
        self.vector = [1.0, 2.0, 3.0]
        self.provider = None

    def __call__(self, request):
        self.calls.append(request)
        assert request.url.scheme == "https" and request.url.host == "openrouter.ai"
        assert request.headers["Authorization"] == "Bearer " + TEST_SECRET
        if request.url.path == "/api/v1/key":
            return httpx.Response(self.auth_status, json={"data": {
                "label": "PRIVATE_ACCOUNT_LABEL", "limit": 900, "secret": TEST_SECRET}})
        if request.url.path == "/api/v1/embeddings/models":
            return httpx.Response(200, json={"data": [{"id": "baai/bge-m3"}] if self.listed else []})
        assert request.method == "POST" and request.url.path == "/api/v1/embeddings"
        payload = json.loads(request.content)
        production = isinstance(payload["input"], list)
        status = self.production_status if production else self.minimal_status
        data = {"model": payload["model"], "data": [{"index": 0, "embedding": self.vector}],
                "usage": {"total_tokens": 4, "cost": 0.00001, "PRIVATE": TEST_SECRET}}
        if self.provider is not None:
            data["provider"] = self.provider
        # Even adversarial errors must not escape through diagnostic output.
        return httpx.Response(status, content=json.dumps(data if status == 200 else {
            "error": {"message": "PRIVATE_ERROR " + TEST_SECRET, "code": status}}))


def run(router, **kwargs):
    return diagnostics.run_diagnostics(transport=httpx.MockTransport(router), **kwargs)


def test_safe_mode_only_two_read_only_requests_and_redacted_account_data():
    router = Router()
    report = run(router)
    assert report["safe_status"] == "PASS" and report["status"] == diagnostics.PARTIAL
    assert report["root_cause_confirmed"] is False
    assert report["safe_requests"] == 2 and report["paid_requests"] == 0
    assert [r.method for r in router.calls] == ["GET", "GET"]
    assert report["authentication"] == {"http_status": 200, "code": "OK"}
    assert report["model_catalog"]["model_listed"] is True
    assert all(report["production_wire_audit"].values())
    assert report["payload_comparison"]["production_fields"] == ["encoding_format", "input", "model", "provider"]
    serialized = json.dumps(report)
    assert TEST_SECRET not in serialized and "PRIVATE_ACCOUNT" not in serialized and "limit" not in report["authentication"]
    assert "Authorization" not in serialized


@pytest.mark.parametrize("value", [None, "", "replace_with_your_openrouter_key"])
def test_missing_key_sends_nothing(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("OPENROUTER_API_KEY")
    else:
        monkeypatch.setenv("OPENROUTER_API_KEY", value)
    router = Router()
    report = run(router, live=True, include_production=True)
    assert report["blocker"] == "openrouter_key_missing" and not router.calls
    assert report["paid_requests"] == report["safe_requests"] == 0


def test_startup_loads_missing_env_but_preserves_existing_env(monkeypatch):
    def startup():
        # Mirror override=False without opening any .env in tests.
        import os
        os.environ.setdefault("OPENROUTER_API_KEY", TEST_SECRET)
    monkeypatch.setattr(diagnostics, "load_environment", startup)
    report = run(Router())
    assert report["environment"]["key_source"] == "process_environment"
    assert report["environment"]["existing_environment_preserved"]
    monkeypatch.delenv("OPENROUTER_API_KEY")
    report = run(Router())
    assert report["environment"]["key_source"] == "startup_dotenv"
    assert not report["environment"]["key_present_before_load"]


@pytest.mark.parametrize("status,code", [
    (401, "unauthorized"), (402, "payment_required"), (403, "forbidden"),
    (404, "model_or_endpoint_not_found"), (429, "rate_limited"), (503, "server_error")])
def test_auth_error_never_retries_or_runs_paid_probe(status, code):
    router = Router(auth_status=status)
    report = run(router, live=True, include_production=True)
    assert report["authentication"]["code"] == "openrouter_" + code
    assert report["safe_requests"] == len(router.calls) == 1 and report["paid_requests"] == 0
    assert report["root_cause_confirmed"] is False


@pytest.mark.parametrize("status,code", [
    (401, "unauthorized"), (402, "payment_required"), (403, "forbidden"),
    (404, "model_or_endpoint_not_found"), (429, "rate_limited"), (500, "server_error")])
def test_minimal_failure_stops_paid_tests_no_retry_and_no_unsupported_attribution(status, code, caplog):
    router = Router(minimal_status=status)
    report = run(router, live=True, include_production=True)
    assert report["minimal_embedding"] == {"http_status": status, "code": "openrouter_" + code}
    assert report["paid_requests"] == 1 and len(router.calls) == 3
    assert report["production_embedding"]["code"] == "NOT_RUN"
    assert report["production_embedding"]["reason"] == "minimal_embedding_failed_stop"
    assert report["root_cause_confirmed"] is False
    assert TEST_SECRET not in json.dumps(report) + caplog.text and "PRIVATE_ERROR" not in json.dumps(report) + caplog.text


def test_only_minimal_authorized_means_one_post():
    router = Router()
    report = run(router, live=True)
    assert report["minimal_embedding"]["http_status"] == 200
    assert report["minimal_embedding"]["vector_dimension"] == 3
    assert report["minimal_embedding"]["usage"] == {"total_tokens": 4, "cost": 0.00001}
    assert report["paid_requests"] == 1 and report["production_embedding"]["code"] == "NOT_RUN"
    assert json.loads(router.calls[-1].content) == {
        "model": "baai/bge-m3", "input": diagnostics.SYNTHETIC_INPUT}


def test_both_opt_ins_use_real_production_builder_without_auto_model_or_route_changes():
    router = Router()
    report = run(router, live=True, include_production=True)
    assert report["paid_requests"] == 2 and len(router.calls) == 4
    minimal, production = [json.loads(r.content) for r in router.calls if r.method == "POST"]
    assert minimal["model"] == production["model"] == "baai/bge-m3"
    assert production == {"model": "baai/bge-m3", "input": [diagnostics.SYNTHETIC_INPUT],
                          "encoding_format": "float", "provider": {"allow_fallbacks": False}}
    assert report["embedding_requests_verified"] and not report["root_cause_confirmed"]
    assert report["status"] == diagnostics.PARTIAL  # Historical cause is not proven by current success.


def test_minimal_pass_production_401_retains_difference_without_speculative_fix():
    router = Router(production_status=401)
    report = run(router, live=True, include_production=True)
    assert report["minimal_embedding"]["code"] == "OK"
    assert report["production_embedding"]["code"] == "openrouter_unauthorized"
    assert not report["embedding_requests_verified"] and report["paid_requests"] == 2


@pytest.mark.parametrize("vector", [[], [float("nan")], [float("inf")], [True], ["0.1"], [0, 0], [[1]]])
def test_bad_minimal_vector_does_not_trigger_second_paid_probe(vector):
    router = Router()
    router.vector = vector
    report = run(router, live=True, include_production=True)
    assert report["minimal_embedding"]["http_status"] == 200
    assert report["minimal_embedding"]["code"] == "diagnostic_invalid_embedding_response"
    assert report["paid_requests"] == 1


def test_unknown_model_no_substitution_no_post():
    router = Router()
    router.listed = False
    report = run(router, live=True, include_production=True)
    assert report["blocker"] == "configured_model_not_listed_no_automatic_substitution"
    assert not report["paid_requests"] and len(router.calls) == 2


def test_timeout_and_transport_exception_are_sanitized_and_not_retried():
    calls = []
    def timeout(request):
        calls.append(True)
        raise httpx.ReadTimeout("PRIVATE " + TEST_SECRET, request=request)
    report = run(timeout, live=True, include_production=True)
    assert report["authentication"]["code"] == "openrouter_timeout" and len(calls) == 1
    assert TEST_SECRET not in json.dumps(report)


def test_redirect_never_forwards_secret_or_runs_paid_probe():
    calls = []
    def redirect(request):
        calls.append(request)
        return httpx.Response(302, headers={"Location": "https://unapproved.example/steal"})
    report = run(redirect, live=True, include_production=True)
    assert report["authentication"]["http_status"] == 302 and len(calls) == 1
    assert report["paid_requests"] == 0


def test_old_acceptance_flags_and_feature_flags_cannot_authorize_new_paid_calls():
    config = RAGConfig.from_dict({"embedding": {"backend": "openrouter"},
        "cloud": {"synthetic_live_opt_in": True}, "reranking": {"enabled": True}, "generation": {"enabled": True}})
    router = Router()
    report = run(router, settings=config)
    assert report["paid_requests"] == 0 and all(r.method == "GET" for r in router.calls)


def test_additional_opt_in_without_minimal_opt_in_rejected_before_network():
    router = Router()
    with pytest.raises(ValueError, match="separate_live_opt_in"):
        run(router, include_production=True)
    assert not router.calls


def test_optional_production_fields_are_captured_exactly_without_live_requests():
    config = RAGConfig.from_dict({"embedding": {"backend": "openrouter", "provider": "ReviewedProvider",
        "dimensions": 3, "document_input_type": "search_document", "passage_prefix": "doc: "}})
    router = Router()
    report = run(router, settings=config)
    diff = report["payload_comparison"]
    assert diff["dimensions"] == 3 and diff["input_type"] == "search_document"
    assert diff["configured_prefix_applied"] and diff["production_provider"] == {
        "only": ["ReviewedProvider"], "allow_fallbacks": False}
    assert report["paid_requests"] == 0


def test_config_secret_accidentally_in_model_is_redacted():
    config = RAGConfig.from_dict({"embedding": {"backend": "openrouter", "model": "model/" + TEST_SECRET}})
    report = run(Router(), settings=config)
    assert TEST_SECRET not in json.dumps(report)


def test_cli_default_is_safe_and_invalid_settings_do_not_print_exception(monkeypatch, tmp_path, capsys):
    seen = []
    def fake_run(**kwargs):
        seen.append(kwargs)
        return {"safe_status": "PASS", "paid_requests": 0}
    monkeypatch.setattr(diagnostics, "run_diagnostics", fake_run)
    monkeypatch.setattr("sys.argv", ["diagnostics"])
    assert diagnostics.main() == 0
    assert not seen[0]["live"] and not seen[0]["include_production"]
    capsys.readouterr()
    path = tmp_path / "settings.json"
    path.write_text('{"api_key": "' + TEST_SECRET + '"}', encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["diagnostics", "--settings", str(path)])
    assert diagnostics.main() == 2
    assert TEST_SECRET not in capsys.readouterr().out


def test_cli_minimal_only_success_does_not_require_unauthorized_production_probe(monkeypatch, capsys):
    monkeypatch.setattr(diagnostics, "run_diagnostics", lambda **kwargs: {
        "safe_status": "PASS", "status": diagnostics.PARTIAL, "minimal_embedding": {"code": "OK"},
        "production_embedding": {"code": "NOT_RUN"}})
    monkeypatch.setattr("sys.argv", ["diagnostics", "--live"])
    assert diagnostics.main() == 0
    assert diagnostics.PARTIAL in capsys.readouterr().out
