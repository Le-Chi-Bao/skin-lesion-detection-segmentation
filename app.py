"""Giao diện xem ảnh và phân vùng thử nghiệm; không đưa ra chẩn đoán."""

from pathlib import Path

import gradio as gr
from PIL import Image, UnidentifiedImageError

from teledermoscopy.gradio_segmentation import DEFAULT_CHECKPOINT, SegmentationController


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


def reset_results(image_path: str | None, image_state: dict):
    """State là đối tượng riêng mỗi phiên; cập nhật tại chỗ để vô hiệu hóa lượt cũ."""
    image_state["revision"] += 1
    message = ("Đã đổi ảnh. Nhấn Xem ảnh hoặc Phân vùng để tiếp tục." if image_path
               else "Đã xóa ảnh. Vui lòng chọn ảnh JPG hoặc PNG.")
    return None, None, None, "", message


def preview_and_reset(image_path: str | None, image_state: dict):
    image_state["revision"] += 1
    image, message = preview_image(image_path)
    return image, None, None, "", message


def build_demo(checkpoint_path: str | Path = DEFAULT_CHECKPOINT, *,
               controller: SegmentationController | None = None) -> gr.Blocks:
    """Tạo giao diện mà không tự khởi động máy chủ khi import."""
    controller = controller if controller is not None else SegmentationController(checkpoint_path)
    with gr.Blocks(title="Hỗ trợ hội chẩn da liễu", analytics_enabled=False) as demo:
        image_state = gr.State({"revision": 0})
        gr.Markdown(
            "# Hỗ trợ phân tích ảnh soi da và chuẩn bị hội chẩn\n"
            "**Phân vùng thử nghiệm — không phải chẩn đoán.**\n\n"
            "Chọn ảnh JPG hoặc PNG. Nhấn **Xem ảnh** để xem trước, hoặc **Phân vùng** "
            "để chạy mô hình local. Kết quả chỉ phục vụ smoke test, chưa đánh giá độ chính xác."
        )
        with gr.Row():
            image_input = gr.File(
                label="Chọn ảnh JPG hoặc PNG",
                file_types=[".jpg", ".jpeg", ".png"],
                file_count="single",
                type="filepath",
                elem_id="image-upload",
            )
            image_output = gr.Image(
                label="Ảnh gốc",
                type="pil",
                format="png",
                interactive=False,
                elem_id="original-image",
            )
        with gr.Row():
            view_button = gr.Button("Xem ảnh")
            segment_button = gr.Button("Phân vùng", variant="primary")
        gr.Markdown("## Phân vùng thử nghiệm\nMask và overlay là đầu ra mô hình, không phải kết luận bệnh.")
        with gr.Row():
            mask_output = gr.Image(label="Mask — Phân vùng thử nghiệm", type="pil", image_mode="L",
                                   format="png", interactive=False, elem_id="segmentation-mask")
            overlay_output = gr.Image(label="Overlay — Phân vùng thử nghiệm", type="pil",
                                      format="png", interactive=False, elem_id="segmentation-overlay")
        device_output = gr.Textbox(label="Thiết bị thực tế", interactive=False, elem_id="actual-device")
        status = gr.Textbox(label="Thông báo", interactive=False, elem_id="app-status")
        results = [image_output, mask_output, overlay_output, device_output, status]
        segmentation_event = segment_button.click(
            fn=controller.segment, inputs=[image_input, image_state], outputs=results,
            api_name="segment_image", concurrency_limit=1, concurrency_id="segmentation",
        )
        view_button.click(
            fn=preview_and_reset, inputs=[image_input, image_state], outputs=results,
            api_name="preview_image", queue=False, cancels=[segmentation_event],
        )
        image_input.change(
            fn=reset_results, inputs=[image_input, image_state], outputs=results,
            api_name="image_changed", queue=False, cancels=[segmentation_event],
        )
    return demo


if __name__ == "__main__":
    build_demo().launch(server_name="127.0.0.1", server_port=7860, share=False)
