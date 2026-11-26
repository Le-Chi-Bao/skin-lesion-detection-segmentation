"""Evidence retrieval and grounded quotations; all untrusted text is HTML escaped."""

import html
import json
import os
from pathlib import Path

import gradio as gr

from app.rag.runtime import load_environment, service_from_env
from app.rag.config import RAGConfig, RetrievalConfig


def format_response(response) -> tuple[str, str]:
    label = " · SYNTHETIC DEMO" if response.synthetic else ""
    status = f"{response.mode}{label} · {response.elapsed_ms:.1f} ms · {response.status}"
    if response.backend:
        status += " · " + response.backend
    status += f" · API requests: {response.usage.get('requests', 0)}"
    if "cost" in response.usage:
        status += f" · reported cost: ${response.usage['cost']:.6f}"
    tokens = response.usage.get("total_tokens")
    if tokens is not None:
        status += f" · tokens: {tokens}"
    if response.usage.get("requests", 0) and "cost" not in response.usage:
        status += " · API có thể tính phí; provider chưa báo chi phí."
    if response.status.startswith("embedding_"):
        return ("", status + " — Dense chưa sẵn sàng. Cài dependencies/model local theo tài liệu WP6A; "
                "không tự tải model. Có thể chọn BM25.")
    if response.status.startswith(("openrouter_", "outbound_", "synthetic_cloud_", "cloud_")):
        return "", status + " — Cloud chưa sẵn sàng; kiểm tra key/quyền/ngân sách. Có thể chọn BM25."
    answer = ""
    if response.answer:
        sources = html.escape(json.dumps(response.sources, ensure_ascii=False, indent=2))
        answer = ("<section><h3>Câu trả lời có bằng chứng</h3>"
                  f'<pre style="white-space:pre-wrap">{html.escape(response.answer)}</pre>'
                  "<p>Nội dung do LLM lựa chọn; không phải kết luận lâm sàng.</p>"
                  f'<details><summary>Nguồn trích dẫn đã xác minh</summary><pre>{sources}</pre></details></section>')
    if not response.hits:
        return answer + "<p>Không có tài liệu phù hợp hoặc index chưa có dữ liệu đủ quyền.</p>", status
    cards = []
    for rank, hit in enumerate(response.hits, start=1):
        citation = hit.chunk.citation
        escape = lambda value: html.escape(str(value))
        badge = "<strong>SYNTHETIC DEMO</strong> · " if citation["synthetic"] else ""
        cards.append(
            f'<article style="border:1px solid #8885;border-radius:8px;padding:16px;margin:12px 0">'
            f'<h3>{rank}. {badge}{escape(citation["title"] or "Không có tiêu đề trong nguồn")}</h3>'
            f'<p>Nguồn: {escape(citation["source"]["locator"])}</p>'
            f'<p>Trang PDF: {escape(citation.get("pdf_page", "không có"))} · '
            f'Trạng thái kiểm duyệt: {escape(citation.get("human_review_status", citation["extraction_status"]))}</p>'
            f'<p>Chunk ID: <code>{escape(hit.chunk.chunk_id)}</code><br>'
            f'Score ({escape(hit.mode)}): {hit.score:.6f} · '
            f'Block [start, end): {escape(citation["block_range"])}</p>'
            f'<pre style="white-space:pre-wrap">{escape(hit.chunk.text)}</pre>'
            f'<details><summary>Citation metadata</summary><pre style="white-space:pre-wrap">'
            f'{escape(json.dumps(citation, ensure_ascii=False, indent=2))}</pre></details></article>'
        )
    return answer + "<h3>Evidence chunks — nguồn tài liệu</h3>" + "".join(cards), status


def search_documents(query, mode, top_k, service, rerank=None, interaction="Retrieve Only"):
    try:
        if service is None:
            service = service_from_env()
        response = service.search(query or "", mode, int(top_k), rerank=rerank,
                                  ask=interaction == "Ask with Evidence")
        rendered, status = format_response(response)
        return service, rendered, status
    except Exception:
        # Do not echo config paths, denied metadata or source text into the browser.
        return service, "", "Medical RAG chưa sẵn sàng: kiểm tra cấu hình, verifier và quyền dữ liệu."


def render_rag_page(session_ttl: int = 3600):
    load_environment()
    defaults = RetrievalConfig()
    settings = None
    try:
        config_path = (os.environ.get("RAG_DEMO_CONFIG") if os.environ.get("RAG_SYNTHETIC_DEMO") == "1"
                       else os.environ.get("RAG_CONFIG"))
        if config_path:
            settings = RAGConfig.load(Path(config_path))
            defaults = settings.retrieval
    except (OSError, ValueError):
        # Keep the CV shell available; search reports invalid RAG operator settings.
        pass
    route = ("Embedding local trên CPU, tìm kiếm Dense hoặc Hybrid; không cần API key. "
             if settings and settings.local_corpus else
             "Cloud embedding, reranking và chọn trích dẫn qua OpenRouter theo cấu hình operator. ")
    gr.Markdown("## Medical RAG\nTruy xuất các đoạn tài liệu và bằng chứng nguồn. " + route +
                "Không nhập tên, mã định danh hoặc thông tin cá nhân bệnh nhân.\n\n"
                "Demo tự tạo luôn mang nhãn **SYNTHETIC DEMO**; không phải nguồn y khoa thật.")
    session = gr.State(None, time_to_live=session_ttl,
                       delete_callback=lambda service: service.close() if service else None)
    question = gr.Textbox(label="Câu hỏi", placeholder="Nhập câu hỏi để tìm đoạn tài liệu liên quan", lines=2)
    with gr.Row():
        mode = gr.Radio(["BM25", "Dense", "Hybrid"], value=defaults.mode, label="Retrieval mode")
        top_k = gr.Slider(1, 100, value=defaults.top_k, step=1, label="Top-K")
    with gr.Row():
        rerank = gr.Checkbox(value=settings.reranking.enabled if settings else False,
                            label="Reranking", interactive=bool(settings and settings.reranking.enabled))
        interaction = gr.Radio(["Retrieve Only", "Ask with Evidence"], value="Retrieve Only",
                               label="Chế độ trả lời")
    button = gr.Button("Search", elem_id="rag-search", variant="primary")
    status = gr.Textbox(label="Trạng thái và thời gian tìm kiếm", interactive=False)
    results = gr.HTML(value="", label="Đoạn tài liệu liên quan")
    button.click(search_documents, [question, mode, top_k, session, rerank, interaction], [session, results, status],
                 api_name="rag_search", api_visibility="private", concurrency_limit=1)
