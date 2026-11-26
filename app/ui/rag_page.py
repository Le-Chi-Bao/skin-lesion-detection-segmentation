"""Session-isolated Medical RAG interface with sanitized Markdown rendering."""

import html
import os
from pathlib import Path
from urllib.parse import quote, urlparse

import gradio as gr

from app.rag.runtime import load_environment, service_from_env
from app.rag.config import RAGConfig, RetrievalConfig
from app.rag.models import RAGError


MODE_RETRIEVE = "Retrieve Only"
MODE_SUMMARY = "Evidence Summary"
MODE_LLM = "Ask with LLM"


def _markdown_text(value):
    text = str(value)
    for character in "\\`*_{}[]()#+-.!|>":
        text = text.replace(character, "\\" + character)
    return text.replace("\n", " ")


def _source_link(url):
    parsed = urlparse(str(url))
    if parsed.scheme == "https" and parsed.netloc:
        safe_url = quote(str(url), safe=":/?#@!$&'()*+,;=%[]")
        return f"[Mở tài liệu gốc](<{safe_url}>)"
    return "Liên kết nguồn chưa được kiểm tra"


def _response_markdown(response):
    sections = []
    if response.answer:
        heading = "Tóm lược bằng chứng" if response.sources and \
            all("excerpt" in item for item in response.sources) else "Câu trả lời có bằng chứng"
        sections.append(f"## {heading}\n\n{response.answer}")
    if not response.hits:
        if response.status == "INSUFFICIENT_EVIDENCE":
            sections.append("Chưa đủ bằng chứng trong corpus để tóm lược câu hỏi này.")
        elif response.status == "NO_MATCH":
            sections.append("Không có tài liệu phù hợp trong corpus hiện tại.")
        elif response.status in {"EMPTY_QUERY", "EMPTY_INDEX"}:
            sections.append("Nhập câu hỏi hoặc xác nhận corpus đã được nạp.")
        return "\n\n".join(sections) or "Không có kết quả phù hợp."

    cited = {item.get("chunk_id"): item for item in response.sources}
    sections.append("## Nguồn tham khảo")
    for rank, hit in enumerate(response.hits, 1):
        citation = hit.chunk.citation
        document_id = citation.get("document_id")
        if document_id != hit.chunk.document_id or not hit.chunk.chunk_id:
            continue
        source = citation.get("source") or {}
        title = _markdown_text(citation.get("title") or "Không có tiêu đề trong nguồn")
        page = citation.get("pdf_page")
        page_text = f"PDF trang {page}" if page is not None else "Không có số trang PDF"
        review = citation.get("human_review_status") or "Không có trạng thái kiểm duyệt"
        review_text = "Đã được con người kiểm duyệt" if review.casefold() in {"reviewed", "verified"} \
            else f"Cần con người xác minh ({_markdown_text(review)})"
        item = cited.get(hit.chunk.chunk_id)
        reference = f" **[{_markdown_text(item['evidence_id'])}]**" if item else ""
        sections.append(
            f"### {rank}. {title}{reference}\n\n"
            f"{page_text} · Chunk `{_markdown_text(hit.chunk.chunk_id)}` · "
            f"Score {hit.score:.4f} ({_markdown_text(hit.mode)}) · {review_text}\n\n"
            f"{_source_link(source.get('locator', ''))}\n\n"
            f"> {hit.chunk.text.replace(chr(10), chr(10) + '> ')}"
        )
    return "\n\n".join(sections)


def _original_text(response):
    sections = []
    for rank, hit in enumerate(response.hits, 1):
        citation = hit.chunk.citation
        if citation.get("document_id") != hit.chunk.document_id:
            continue
        sections.append(
            f"[{rank}] {citation.get('title', 'Không có tiêu đề')}\n"
            f"Document ID: {hit.chunk.document_id}\n"
            f"PDF page: {citation.get('pdf_page', 'không có')}\n"
            f"Chunk ID: {hit.chunk.chunk_id}\n\n{hit.chunk.text}"
        )
    return "\n\n".join(sections)


def format_response(response) -> tuple[str, str]:
    """Compatibility renderer for callers that expect a safe HTML string."""
    label = " · SYNTHETIC DEMO" if response.synthetic else ""
    status = f"{response.mode}{label} · {response.elapsed_ms:.1f} ms · {response.status}"
    if response.backend:
        status += " · " + response.backend
    status += f" · API requests: {response.usage.get('requests', 0)}"
    if "cost" in response.usage:
        status += f" · reported cost: ${response.usage['cost']:.6f}"
    if response.usage.get("requests", 0) and "cost" not in response.usage:
        status += " · API có thể tính phí; provider chưa báo chi phí."
    if response.status == "generation_disabled_by_operator":
        status += " · Generation bị tắt theo cấu hình và quyền xử lý dữ liệu."
    elif response.status == "outbound_processing_not_approved":
        status += " · Provider chưa được cấp quyền xử lý outbound cho tài liệu này."
    elif response.status == "generator_unavailable":
        status += " · Không có model generation khả dụng."
    elif response.status == "INSUFFICIENT_EVIDENCE":
        status += " · Corpus chưa có đủ bằng chứng cho câu hỏi."
    answer = f"<pre>{html.escape(response.answer)}</pre>" if response.answer else ""
    original = f"<pre>{html.escape(_original_text(response))}</pre>"
    body = ("<section><h3>Câu trả lời có bằng chứng</h3>" + answer
            + "<h3>Nguồn trích dẫn đã xác minh</h3>" + original + "</section>")
    return body, status


def _render_status(response, interaction):
    label = " · SYNTHETIC DEMO" if response.synthetic else ""
    status = f"{response.mode}{label} · {response.elapsed_ms:.1f} ms · {response.status}"
    if interaction == MODE_LLM and response.status == "generation_disabled_by_operator":
        status += " · Truy xuất thành công; LLM bị tắt theo cấu hình/quyền dữ liệu."
    elif interaction == MODE_LLM and response.status == "outbound_processing_not_approved":
        status += " · Truy xuất thành công; provider chưa có quyền outbound."
    elif response.status == "INSUFFICIENT_EVIDENCE":
        status += " · Chưa đủ bằng chứng trong corpus."
    elif response.status == "NO_MATCH":
        status += " · Không tìm thấy bằng chứng phù hợp."
    if response.backend:
        status += " · " + response.backend
    return status


def search_documents(query, mode, top_k, service, rerank=None, interaction=MODE_RETRIEVE,
                     include_original=False):
    try:
        if service is None:
            service = service_from_env()
        response = service.search(
            query or "", mode, int(top_k), rerank=rerank,
            ask=interaction in {MODE_LLM, "Ask with Evidence"},
            summarize=interaction == MODE_SUMMARY,
        )
        rendered = _response_markdown(response)
        status = _render_status(response, interaction)
        if include_original:
            return service, rendered, status, _original_text(response)
        return service, rendered, status
    except RAGError as exc:
        blocked = {
            "batch_blocked_by_rights_or_extraction_review",
            "local_kit_integrity_or_consent_failed",
            "local_indexing_consent_missing_or_changed",
        }
        status = "Corpus đang bị chặn bởi kiểm tra toàn vẹn hoặc quyền dữ liệu." \
            if str(exc) in blocked else "Medical RAG chưa sẵn sàng: kiểm tra cấu hình và quyền dữ liệu."
        if include_original:
            return service, "", status, ""
        return service, "", status
    except Exception:
        # Do not echo config paths, denied metadata, stack traces, or source text.
        if include_original:
            return service, "", "Medical RAG chưa sẵn sàng: kiểm tra cấu hình, verifier và quyền dữ liệu.", ""
        return service, "", "Medical RAG chưa sẵn sàng: kiểm tra cấu hình, verifier và quyền dữ liệu."


def _search_documents_ui(query, mode, top_k, service, rerank, interaction):
    state, rendered, status, original = search_documents(
        query, mode, top_k, service, rerank, interaction, include_original=True,
    )
    return state, rendered, status, original


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
        pass
    corpus_text = "Corpus local, embedding E5 và quyền index được cấu hình bởi operator." \
        if settings and settings.local_corpus else "Corpus và quyền xử lý được cấu hình bởi operator."
    generation_text = "LLM khả dụng theo cấu hình." if settings and settings.generation.enabled \
        else "LLM chưa khả dụng hoặc chưa được cấp quyền; Evidence Summary hoạt động cục bộ."
    gr.Markdown(
        "## Medical RAG\n" + corpus_text + " " + generation_text +
        "\n\nKhông nhập tên, mã định danh hoặc thông tin cá nhân bệnh nhân. "
        "Nguồn chưa được con người kiểm duyệt sẽ luôn được đánh dấu."
    )
    session = gr.State(None, time_to_live=session_ttl,
                       delete_callback=lambda service: service.close() if service else None)
    question = gr.Textbox(label="Câu hỏi", placeholder="Ví dụ: Những yếu tố nào liên quan đến viêm da cơ địa?",
                          lines=2, autofocus=True)
    with gr.Row():
        mode = gr.Radio(["BM25", "Dense", "Hybrid"], value=defaults.mode, label="Retrieval mode")
        top_k = gr.Slider(1, 100, value=defaults.top_k, step=1, label="Top-K")
    with gr.Row():
        interaction = gr.Radio([MODE_RETRIEVE, MODE_SUMMARY, MODE_LLM], value=MODE_RETRIEVE,
                               label="Chế độ")
        rerank = gr.Checkbox(value=settings.reranking.enabled if settings else False,
                            label="Reranking", interactive=bool(settings and settings.reranking.enabled))
    button = gr.Button("Tìm bằng chứng", elem_id="rag-search", variant="primary")
    status = gr.Textbox(label="Trạng thái", interactive=False)
    results = gr.Markdown(value="", label="Kết quả", buttons=["copy"], sanitize_html=True)
    with gr.Accordion("Văn bản gốc của các đoạn được truy xuất", open=False):
        original = gr.Code(value="", language="markdown", label="Markdown gốc", interactive=False)
    button.click(
        _search_documents_ui, [question, mode, top_k, session, rerank, interaction],
        [session, results, status, original], api_name="rag_search", api_visibility="private",
        concurrency_limit=1, show_progress="full",
    )
