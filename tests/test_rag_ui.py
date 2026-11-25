import asyncio
from dataclasses import replace

from gradio.state_holder import SessionState as GradioSessionState

from app.config import AppConfig
from app.core.temporary_files import TemporaryFiles
from app.rag.models import SearchHit
from app.rag.runtime import create_service
from app.ui.rag_page import format_response, search_documents
from app.ui.shell import build_shell


def test_gradio_sessions_own_their_rag_service(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_SYNTHETIC_DEMO", "1")
    monkeypatch.delenv("RAG_CONFIG", raising=False)
    config = AppConfig(runtime_dir=tmp_path)
    demo = build_shell(config, TemporaryFiles(tmp_path, 60))
    event = next(d for d in demo.config["dependencies"] if d["api_name"] == "rag_search")
    state_id = event["outputs"][0]
    a, b = GradioSessionState(demo), GradioSessionState(demo)

    async def search(session):
        return await demo.process_api(event["id"], ["skin", "BM25", 5, None], state=session)

    try:
        first, second = asyncio.run(search(a)), asyncio.run(search(b))
        assert "SYNTHETIC DEMO" in first["data"][1] and "BM25" in second["data"][2]
        assert a[state_id] is not b[state_id]
        target = a[state_id].search("skin", "BM25").hits[0].chunk.document_id
        a[state_id].remove_document(target)
        assert "Không có tài liệu" in asyncio.run(search(a))["data"][1]
        assert "skin lesion" in asyncio.run(search(b))["data"][1]
    finally:
        for state in (a, b):
            if state[state_id]:
                state[state_id].close()
        demo.close()


def test_ui_escapes_source_text_and_metadata():
    service = create_service(demo=True)
    try:
        response = service.search("skin", "BM25")
        chunk = response.hits[0].chunk
        chunk.citation["title"] = '<img src="https://example.org/pixel" onerror="alert(1)">'
        hit = SearchHit(replace(chunk, text="<script>alert(1)</script>"), 1, "BM25")
        rendered, status = format_response(replace(response, hits=(hit,)))
        assert "<script>" not in rendered and "<img" not in rendered
        assert "&lt;script&gt;" in rendered and "SYNTHETIC DEMO" in status
    finally:
        service.close()


def test_ui_never_echoes_rejected_document_metadata(monkeypatch):
    from app.ui import rag_page

    def denied():
        raise ValueError("DENIED PRIVATE DOCUMENT TITLE")

    monkeypatch.setattr(rag_page, "service_from_env", denied)
    state, results, status = search_documents("skin", "BM25", 5, None)
    assert state is None and not results
    assert "DENIED PRIVATE" not in status
