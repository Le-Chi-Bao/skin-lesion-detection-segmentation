"""Giao diện demo tải và xem ảnh; chưa thực hiện phân tích tổn thương."""

from pathlib import Path

import gradio as gr
from PIL import Image, UnidentifiedImageError


def preview_image(image_path: str | None) -> tuple[Image.Image | None, str]:
    """Đọc ảnh được chọn để hiển thị, không chạy xử lý y khoa."""
    if not image_path:
        return None, "Vui lòng chọn ảnh JPG hoặc PNG trước khi nhấn Xem ảnh."

    if Path(image_path).suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        return None, "Chỉ hỗ trợ ảnh JPG hoặc PNG."

    try:
        with Image.open(image_path) as original:
            if original.format not in {"JPEG", "PNG"}:
                return None, "Chỉ hỗ trợ ảnh JPG hoặc PNG."
            original.load()
            return original.copy(), "Đã hiển thị ảnh gốc. Chưa phân tích tổn thương."
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError):
        return None, "Không đọc được ảnh. Vui lòng chọn lại ảnh JPG hoặc PNG hợp lệ."


def build_demo() -> gr.Blocks:
    """Tạo giao diện mà không tự khởi động máy chủ khi import."""
    with gr.Blocks(title="Hỗ trợ hội chẩn da liễu", analytics_enabled=False) as demo:
        gr.Markdown(
            "# Hỗ trợ phân tích ảnh soi da và chuẩn bị hội chẩn\n"
            "**Bước 1 — Demo tải và xem ảnh. Chưa phân tích tổn thương.**\n\n"
            "Chọn ảnh JPG hoặc PNG, sau đó nhấn **Xem ảnh** để hiển thị ảnh gốc."
        )
        with gr.Row():
            image_input = gr.File(
                label="Chọn ảnh JPG hoặc PNG",
                file_types=[".jpg", ".jpeg", ".png"],
                file_count="single",
                type="filepath",
            )
            image_output = gr.Image(
                label="Ảnh gốc",
                type="pil",
                format="png",
                interactive=False,
            )
        view_button = gr.Button("Xem ảnh", variant="primary")
        status = gr.Textbox(label="Thông báo", interactive=False)
        view_button.click(
            fn=preview_image,
            inputs=image_input,
            outputs=[image_output, status],
            api_name="preview_image",
        )
    return demo


if __name__ == "__main__":
    build_demo().launch(server_name="127.0.0.1", server_port=7860, share=False)
