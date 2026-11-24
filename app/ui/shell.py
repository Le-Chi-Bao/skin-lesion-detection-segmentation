import gradio as gr

from app.config import AppConfig
from app.core.temporary_files import TemporaryFiles
from app.cv.availability import CVAvailability
from app.cv.segformer import SegFormer
from app.ui.cv_page import render_cv_page
from app.ui.overview import DEFERRED_MODULES, render_overview


def build_shell(config: AppConfig, files: TemporaryFiles, segmenter: SegFormer | None = None) -> gr.Blocks:
    segmenter = segmenter if segmenter is not None else SegFormer(config)
    segmenter.load()
    availability = CVAvailability(segmentation=segmenter.availability)
    with gr.Blocks(
        title=config.app_name,
        analytics_enabled=False,
        delete_cache=(config.cleanup_interval, config.session_ttl_seconds),
    ) as demo:
        with gr.Sidebar(label="Điều hướng", width=250):
            navigation = gr.Radio(
                ("Tổng quan", "Computer Vision"), value="Tổng quan", label="Mô-đun",
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
            render_cv_page(config, files, availability, segmenter)

        def navigate(section: str) -> tuple[dict, dict]:
            return gr.update(visible=section == "Tổng quan"), gr.update(visible=section == "Computer Vision")

        navigation.input(navigate, navigation, [overview, cv_page], api_visibility="private", queue=False)
    return demo
