"""Session, integrity, outbound approvals and UI integration with mocked HTTP only."""

import json
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from app.rag.cloud.client import BASE_URL
from app.rag.config import RAGConfig
from app.rag.runtime import create_service
from app.ui.rag_page import format_response, search_documents
from rag_support import approve_current, read, reviewed_batch, update_record, write


def settings(*, opt_in=True, reranking=True, generation=True, **cloud):
    return RAGConfig.from_dict({
        "embedding": {"backend": "openrouter", "provider": "ReviewedProvider", "batch_size": 16},
        "reranking": {"enabled": reranking, "provider": "ReviewedProvider", "top_n": 3},
        "generation": {"enabled": generation, "provider": "ReviewedProvider"},
        "cloud": {"synthetic_live_opt_in": opt_in, "retry_policy": {"max_retries": 0}, **cloud}})


class MockRouter:
    def __init__(self):
        self.calls = []
        self.hook = lambda path: None
        self.rerank_status = 200

    def __call__(self, request):
        payload = json.loads(request.content)
        self.calls.append((request.url.path, payload))
        path = request.url.path.rsplit("/", 1)[-1]
        self.hook(path)
        usage = {"total_tokens": 10, "cost": 0.0001}
        if path == "embeddings":
            return httpx.Response(200, json={"model": payload["model"], "data": [
                {"index": i, "embedding": [1 + len(t) % 13, 2, 3, 4]} for i, t in enumerate(payload["input"])], "usage": usage})
        if path == "rerank":
            return httpx.Response(self.rerank_status, json={"model": payload["model"], "results": [
                {"index": i, "relevance_score": 1 - i * 0.1} for i in range(payload["top_n"])], "usage": usage})
        context = json.loads(payload["messages"][1]["content"])
        item = context["evidence"][0]
        return httpx.Response(200, json={"model": payload["model"], "choices": [{"finish_reason": "stop", "message": {"content": json.dumps({
            "insufficient_evidence": False, "claims": [{"evidence_id": item["evidence_id"], "quote": item["text"]}]})}}], "usage": usage})


@pytest.fixture(autouse=True)
def isolate_secret(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only-not-a-real-key")


@pytest.fixture
def demo():
    router = MockRouter()
    service = create_service(demo=True, settings=settings(), transport=httpx.MockTransport(router))
    yield service, router
    service.close()


def grant_external(approvals, config):
    data = read(approvals)
    for grant in data["documents"]:
        grant["external_processing"] = {
            "status": "allowed", "destination": BASE_URL, "patient_identifiers": "absent_reviewed",
            "reviewer_id": "synthetic-test-reviewer", "evidence": "SYNTHETIC TEST ONLY, not a real corpus authorization",
            "reviewed_at": "2026-01-01T00:00:00Z", "review_again_at": "2027-01-01T00:00:00Z",
            "routes": [{"action": action, "model": route.model, "provider": route.provider}
                       for action, route in (("embedding", config.embedding), ("reranking", config.reranking),
                                             ("generation", config.generation))]}
    write(approvals, data)


def production_fixture(tmp_path, config):
    manifest, _, importer, registry, approvals = reviewed_batch(tmp_path)
    router = MockRouter()
    service = create_service(settings=config, transport=httpx.MockTransport(router))
    assert service.ingest(manifest, importer).status == "VERIFIED"
    return service, router, manifest, importer, registry, approvals


def test_full_pipeline_three_modes_cache_and_grounded_sources(demo):
    service, router = demo
    lexical = service.search("skin", "BM25", rerank=False)
    assert lexical.status == "OK" and lexical.usage["requests"] == 0
    dense = service.search("skin", "Dense", rerank=False)
    assert dense.status == "OK" and dense.usage["requests"] == 2
    hybrid = service.search("skin", "Hybrid", rerank=False)
    assert hybrid.status == "OK" and hybrid.usage["requests"] == 1
    assert sum(len(p["input"]) > 1 for _, p in router.calls) == 1
    result = service.search("skin", "Hybrid", 3, rerank=True, ask=True)
    assert result.status == "OK" and result.sources and "[E1]" in result.answer
    assert result.usage["requests"] == 3 and result.usage["total_tokens"] == 30
    citation = result.sources[0]
    assert citation["citation"] == next(h.chunk.citation for h in result.hits if h.chunk.chunk_id == citation["chunk_id"])
    rendered, status = format_response(result)
    assert "Câu trả lời có bằng chứng" in rendered and "Nguồn trích dẫn đã xác minh" in rendered
    assert "API requests: 3" in status and "reported cost" in status and "SYNTHETIC DEMO" in status


def test_no_model_load_in_cloud_mode(demo, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("cloud path attempted to load E5 weights")
    monkeypatch.setattr("app.rag.retrieval.LocalE5Embedder._load", forbidden)
    service, _ = demo
    assert service.search("skin", "Hybrid", rerank=False).status == "OK"


def test_missing_key_does_not_break_bm25_and_does_not_fallback(demo, monkeypatch):
    service, router = demo
    monkeypatch.delenv("OPENROUTER_API_KEY")
    for mode in ("Dense", "Hybrid"):
        response = service.search("skin", mode, rerank=False)
        assert response.status == "openrouter_key_missing" and not response.hits
    assert service.search("skin", "BM25", rerank=False).status == "OK" and not router.calls


def test_synthetic_requires_separate_opt_in():
    router = MockRouter()
    with_service = create_service(demo=True, settings=settings(opt_in=False), transport=httpx.MockTransport(router))
    try:
        assert with_service.search("skin", "Hybrid", rerank=False).status == "synthetic_cloud_opt_in_required"
        assert with_service.search("skin", "BM25", rerank=False).status == "OK"
        assert not router.calls
    finally:
        with_service.close()


def test_local_ai_rights_do_not_imply_external_permission(tmp_path):
    service, router, _, _, _, _ = production_fixture(tmp_path, settings())
    try:
        assert service.search("skin", "BM25", rerank=False).status == "OK"
        assert service.search("skin", "Hybrid", rerank=False).status == "outbound_processing_not_approved"
        assert not router.calls
    finally:
        service.close()


@pytest.mark.parametrize("change", ["status", "expiry", "reviewer", "privacy", "destination", "model", "provider", "action"])
def test_external_sidecar_fail_closed(tmp_path, change):
    config = settings()
    service, router, _, _, _, approvals = production_fixture(tmp_path, config)
    grant_external(approvals, config)
    data = read(approvals)
    for doc in data["documents"]:
        grant = doc["external_processing"]
        if change == "status": grant["status"] = "revoked"
        if change == "expiry": grant["review_again_at"] = "2026-01-02T00:00:00Z"
        if change == "reviewer": grant["reviewer_id"] = ""
        if change == "privacy": grant["patient_identifiers"] = "unknown"
        if change == "destination": grant["destination"] = "https://unapproved.example"
        if change in ("model", "provider", "action"): grant["routes"][0][change] = "unapproved"
    write(approvals, data)
    try:
        response = service.search("skin", "Dense", rerank=False)
        assert response.status == "outbound_processing_not_approved" and not router.calls
    finally:
        service.close()


def test_outbound_revocation_evicts_cache_even_on_bm25(tmp_path):
    config = settings()
    service, router, _, _, _, approvals = production_fixture(tmp_path, config)
    grant_external(approvals, config)
    try:
        assert service.search("skin", "Hybrid", rerank=False).status == "OK"
        assert service.dense.embedder.metadata["cached_vectors"] > 0
        data = read(approvals)
        data["documents"][0]["external_processing"]["status"] = "revoked"
        write(approvals, data)
        count = len(router.calls)
        assert service.search("skin", "BM25", rerank=False).status == "OK"
        assert service.dense.embedder.metadata["cached_vectors"] == 0 and not service.dense.chunks
        assert service.search("skin", "Hybrid", rerank=False).status == "outbound_processing_not_approved"
        assert len(router.calls) == count
    finally:
        service.close()


@pytest.mark.parametrize("phase", ["embeddings", "rerank", "completions"])
def test_revocation_during_any_api_stage_hides_entire_result(tmp_path, phase):
    config = settings()
    service, router, _, _, _, approvals = production_fixture(tmp_path, config)
    grant_external(approvals, config)
    def revoke(path):
        if path == phase:
            data = read(approvals)
            for doc in data["documents"]:
                doc["external_processing"]["status"] = "revoked"
            write(approvals, data)
    router.hook = revoke
    try:
        result = service.search("skin", "Hybrid", 3, rerank=True, ask=True)
        assert result.status == "outbound_processing_not_approved"
        assert not result.hits and not result.answer and not result.sources
        assert not service.dense.chunks and not service.dense.embedder._cache
    finally:
        service.close()


def test_corpus_content_model_chunk_and_session_cache_invalidation(tmp_path):
    config = settings()
    service, router, manifest, _, _, approvals = production_fixture(tmp_path, config)
    grant_external(approvals, config)
    try:
        assert service.search("skin", "Dense", rerank=False).status == "OK"
        first_ids = {c.chunk_id for c in service.dense.chunks}
        update_record(manifest, 0, lambda r: r["content"][1].update(text="SYNTHETIC DEMO edited skin evidence"))
        approve_current(manifest, approvals)
        grant_external(approvals, config)
        result = service.search("skin", "Dense", rerank=False)
        assert result.status == "OK" and {c.chunk_id for c in service.dense.chunks} != first_ids
        assert result.usage["requests"] == 2  # Only edited chunks plus query, cached others retained.
        service.dense.embedder.config = replace(config.embedding, model="approved/new-embedding")
        service.settings = replace(config, embedding=service.dense.embedder.config)
        grant_external(approvals, service.settings)
        result = service.search("skin", "Dense", rerank=False)
        assert result.status == "OK" and result.usage["requests"] == 2
        from app.rag.chunking import RecursiveChunker
        service.chunker = RecursiveChunker(replace(config.chunking, max_tokens=250, overlap=20))
        result = service.search("skin", "Dense", rerank=False)
        assert result.status == "OK" and result.usage["requests"] == 2
        other_router = MockRouter()
        other = create_service(settings=service.settings, transport=httpx.MockTransport(other_router))
        try:
            assert other.dense.embedder.metadata["cached_vectors"] == 0
            assert other._corpus_id != service._corpus_id
        finally:
            other.close()
    finally:
        service.close()


def test_rerank_failure_fallback_requires_operator_and_hybrid(demo):
    service, router = demo
    router.rerank_status = 503
    denied = service.search("skin", "Hybrid", rerank=True)
    assert denied.status == "openrouter_server_error" and not denied.hits
    service.settings = replace(service.settings, reranking=replace(service.settings.reranking, fallback_to_hybrid=True))
    service.reranker.config = service.settings.reranking
    permitted = service.search("skin", "Hybrid", rerank=True)
    assert permitted.status == "DEGRADED_HYBRID_WITHOUT_RERANK" and permitted.hits
    assert "DEGRADED" in format_response(permitted)[1]
    assert service.search("skin", "BM25", rerank=True).status == "openrouter_server_error"


def test_operator_limits_disabled_features_and_query_privacy(demo):
    service, router = demo
    for query in ("patient name: Jane", "họ tên: Nguyễn", "contact test@example.org", "phone 0901234567"):
        assert service.search(query, "Hybrid", rerank=False).status == "query_possible_patient_identifier_blocked"
    assert not router.calls
    service.settings = replace(service.settings, generation=replace(service.settings.generation, enabled=False))
    assert service.search("skin", "BM25", ask=True, rerank=False).status == "generation_disabled_by_operator"
    assert not router.calls


def test_index_cost_preflight_and_session_cleanup():
    router = MockRouter()
    service = create_service(demo=True, settings=settings(max_index_chunks=1), transport=httpx.MockTransport(router))
    response = service.search("skin", "Hybrid", rerank=False)
    assert response.status == "cloud_index_chunk_budget_exceeded" and not router.calls
    root = Path(service._demo_directory.name)
    service.close()
    assert not root.exists() and not service.dense.embedder._cache


def test_secret_env_does_not_override_existing_key(monkeypatch, tmp_path):
    from app.rag import runtime
    calls = []
    import dotenv
    monkeypatch.setattr(dotenv, "load_dotenv", lambda path, **kwargs: calls.append((path, kwargs)))
    runtime.load_environment()
    assert calls == [(Path(runtime.__file__).resolve().parents[2] / ".env", {"override": False})]


def test_ui_escapes_generated_quotes_and_sources(demo):
    service, _ = demo
    result = service.search("skin", "BM25", rerank=False)
    poisoned = replace(result, answer='<script>alert(1)</script>', sources=({"title": '<img onerror="bad">'},))
    html, status = format_response(poisoned)
    assert "<script>" not in html and "<img" not in html and "&lt;script&gt;" in html
    _, rendered, status = search_documents("skin", "Hybrid", 3, service, True, "Ask with Evidence")
    assert "Câu trả lời có bằng chứng" in rendered and "test-only-not-a-real-key" not in status + rendered


def test_operator_demo_json_wires_ui_and_runtime(tmp_path, monkeypatch):
    from dataclasses import asdict
    from app.rag import runtime
    import gradio as gr
    from app.ui.rag_page import render_rag_page
    config = settings()
    path = tmp_path / "demo.json"
    value = {key: asdict(getattr(config, key)) for key in ("embedding", "cloud", "reranking", "generation")}
    value["retrieval"] = {"mode": "Hybrid", "top_k": 3}
    write(path, value)
    monkeypatch.setenv("RAG_SYNTHETIC_DEMO", "1")
    monkeypatch.setenv("RAG_DEMO_CONFIG", str(path))
    monkeypatch.delenv("RAG_CONFIG", raising=False)
    service = runtime.service_from_env()
    try:
        assert service.settings.cloud.synthetic_live_opt_in and service.settings.generation.enabled
        # Creating a configured session never sends an API request.
        assert service.cloud_client.total_requests == 0
        with gr.Blocks() as demo:
            render_rag_page()
        try:
            components = {c["props"].get("label"): c["props"] for c in demo.config["components"]}
            assert components["Reranking"]["value"] is True
            assert components["Retrieval mode"]["value"] == "Hybrid"
        finally:
            demo.close()
    finally:
        service.close()
    monkeypatch.setenv("RAG_CONFIG", "must-not-be-opened.json")
    with pytest.raises(ValueError, match="exclusive_synthetic_mode"):
        runtime.service_from_env()


def test_live_acceptance_opt_in_and_lifetime_budget(monkeypatch):
    from app.rag import live_acceptance
    calls = []
    def blocked(*args, **kwargs):
        pytest.fail("missing opt-in must not create a service or call API")
    monkeypatch.setattr(live_acceptance, "create_service", blocked)
    assert live_acceptance.run_acceptance()["requests"] == 0
    assert live_acceptance.run_acceptance(live=True)["status"] == "BLOCKED"
    router = MockRouter()
    def mocked_service(**kwargs):
        calls.append(True)
        return create_service(**kwargs, transport=httpx.MockTransport(router))
    monkeypatch.setattr(live_acceptance, "create_service", mocked_service)
    result = live_acceptance.run_acceptance(live=True, budget_confirmed=True, settings=settings())
    assert result["status"] == "PASS" and result["requests"] == 5
    assert set(result["stages"].values()) == {"PASS"} and calls == [True]
    # This is mock acceptance plumbing, not evidence of a real API run.


@pytest.mark.parametrize("action", ["reranking", "generation"])
def test_each_stage_requires_its_own_reviewed_route(tmp_path, action):
    config = settings()
    service, router, _, _, _, approvals = production_fixture(tmp_path, config)
    grant_external(approvals, config)
    data = read(approvals)
    for doc in data["documents"]:
        grant = doc["external_processing"]
        grant["routes"] = [route for route in grant["routes"] if route["action"] != action]
    write(approvals, data)
    try:
        result = service.search("skin", "BM25", rerank=action == "reranking", ask=action == "generation")
        assert result.status == "outbound_processing_not_approved" and not router.calls
        assert not result.hits and not result.answer and not result.sources
    finally:
        service.close()


def test_identical_demo_sessions_never_share_vector_cache(demo):
    first, _ = demo
    assert first.search("skin", "Dense", rerank=False).usage["requests"] == 2
    router = MockRouter()
    second = create_service(demo=True, settings=settings(), transport=httpx.MockTransport(router))
    try:
        assert second.search("skin", "Dense", rerank=False).usage["requests"] == 2
        assert second.dense.embedder._cache is not first.dense.embedder._cache
        first.close()
        assert second.search("skin", "Dense", rerank=False).usage["requests"] == 1
    finally:
        second.close()
