"""Retrieval-only Gradio view. All untrusted document strings are rendered as escaped text."""

import html
import json

import gradio as gr

from app.rag.runtime import service_from_env


def format_response(response) -> tuple[str, str]:
    label = " · SYNTHETIC DEMO" if response.synthetic else ""
    status = f"{response.mode}{label} · {response.elapsed_ms:.1f} ms · {response.status}"
    if response.status.startswith("embedding_"):
        return ("", status + " — Dense chưa sẵn sàng. Cài dependencies/model local theo tài liệu WP6A; "
                "không tự tải model. Có thể chọn BM25.")
    if not response.hits:
        return "<p>Không có tài liệu phù hợp hoặc index chưa có dữ liệu đủ quyền.</p>", status
    cards = []
    for rank, hit in enumerate(response.hits, start=1):
        citation = hit.chunk.citation
        escape = lambda value: html.escape(str(value))
        badge = "<strong>SYNTHETIC DEMO</strong> · " if citation["synthetic"] else ""
        cards.append(
            f'<article style="border:1px solid #8885;border-radius:8px;padding:16px;margin:12px 0">'
            f'<h3>{rank}. {badge}{escape(citation["title"] or "Không có tiêu đề trong nguồn")}</h3>'
            f'<p>Nguồn: {escape(citation["source"]["locator"])}</p>'
            f'<p>Chunk ID: <code>{escape(hit.chunk.chunk_id)}</code><br>'
            f'Score ({escape(hit.mode)}): {hit.score:.6f} · '
            f'Block [start, end): {escape(citation["block_range"])}</p>'
            f'<pre style="white-space:pre-wrap">{escape(hit.chunk.text)}</pre>'
            f'<details><summary>Citation metadata</summary><pre style="white-space:pre-wrap">'
            f'{escape(json.dumps(citation, ensure_ascii=False, indent=2))}</pre></details></article>'
        )
    return "".join(cards), status


def search_documents(query, mode, top_k, service):
    try:
        if service is None:
            service = service_from_env()
        response = service.search(query or "", mode, int(top_k))
        rendered, status = format_response(response)
        return service, rendered, status
    except Exception:
        # Do not echo config paths, denied metadata or source text into the browser.
        return service, "", "Medical RAG chưa sẵn sàng: kiểm tra cấu hình, verifier và quyền dữ liệu."


def render_rag_page(session_ttl: int = 3600):
    gr.Markdown("## Medical RAG\nTruy xuất các đoạn tài liệu và bằng chứng nguồn. "
                "WP6A chưa sinh câu trả lời y khoa bằng LLM.\n\n"
                "Demo tự tạo luôn mang nhãn **SYNTHETIC DEMO**; không phải nguồn y khoa thật.")
    session = gr.State(None, time_to_live=session_ttl,
                       delete_callback=lambda service: service.close() if service else None)
    question = gr.Textbox(label="Câu hỏi", placeholder="Nhập câu hỏi để tìm đoạn tài liệu liên quan", lines=2)
    with gr.Row():
        mode = gr.Radio(["BM25", "Dense", "Hybrid"], value="BM25", label="Retrieval mode")
        top_k = gr.Slider(1, 20, value=5, step=1, label="Top-K")
    button = gr.Button("Search", elem_id="rag-search", variant="primary")
    status = gr.Textbox(label="Trạng thái và thời gian tìm kiếm", interactive=False)
    results = gr.HTML(value="", label="Đoạn tài liệu liên quan")
    button.click(search_documents, [question, mode, top_k, session], [session, results, status],
                 api_name="rag_search", api_visibility="private", concurrency_limit=1)
