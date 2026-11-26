import gradio as gr

from app.config import AppConfig
from app.core.temporary_files import TemporaryFiles
from app.cv.adapters import SegmentationAdapter
from app.cv.orchestrator import CVOrchestrator, create_orchestrator
from app.ui.cv_page import render_cv_page
from app.ui.overview import DEFERRED_MODULES, render_overview
from app.ui.png_export import PNGExports
from app.ui.rag_page import render_rag_page


def render_report_page() -> None:
    gr.Markdown(
        "## Report\n"
        "Báo cáo tổng hợp của MVP v1.0. Kết quả CV chỉ mô tả vùng định vị và mask, "
        "không suy ra loại bệnh hoặc chẩn đoán. RAG hiển thị nguồn, document ID và trang PDF."
    )
    gr.Markdown(
        "### Trạng thái bàn giao\n"
        "- Faster R-CNN E2 và SegFormer dùng checkpoint local đã đăng ký\n"
        "- Medical RAG hoạt động BM25 / Dense / Hybrid trong chế độ retrieval-only\n"
        "- Ground truth RAG chưa được con người xác minh nên metric giữ N/A\n\n"
        "Xem báo cáo đầy đủ tại `docs/FINAL_MVP_V1.md`."
    )


def build_shell(
    config: AppConfig, files: TemporaryFiles, segmenter: SegmentationAdapter | None = None, *,
    exports: PNGExports | None = None, orchestrator: CVOrchestrator | None = None,
) -> gr.Blocks:
    models = orchestrator or create_orchestrator(config, segmenter)
    availability = models.availability
    with gr.Blocks(
        title=config.app_name,
        analytics_enabled=False,
        delete_cache=(config.cleanup_interval, config.session_ttl_seconds),
    ) as demo:
        with gr.Sidebar(label="Điều hướng", width=250):
            navigation = gr.Radio(
                ("Tổng quan", "Computer Vision", "Medical RAG", "Report"), value="Tổng quan", label="Mô-đun",
            )
            gr.Markdown("\n\n".join(f"**{module}**  \nChưa triển khai" for module in DEFERRED_MODULES))

        gr.Markdown(
            f"# {config.app_name}\n"
            "> **Chỉ dùng cho nghiên cứu.** Hệ thống chưa được thẩm định để sử dụng lâm sàng "
            "và không tự động chẩn đoán bệnh da."
        )
        with gr.Column() as overview:
            render_overview(availability)
        with gr.Column(visible=False) as cv_page:
            render_cv_page(config, files, availability, models, exports)
        with gr.Column(visible=False) as rag_page:
            render_rag_page(config.session_ttl_seconds)
        with gr.Column(visible=False) as report_page:
            render_report_page()

        def navigate(section: str) -> tuple[dict, dict, dict, dict]:
            return (gr.update(visible=section == "Tổng quan"), gr.update(visible=section == "Computer Vision"),
                    gr.update(visible=section == "Medical RAG"), gr.update(visible=section == "Report"))

        navigation.input(navigate, navigation, [overview, cv_page, rag_page, report_page], api_visibility="private", queue=False)
    return demo
