import gradio as gr

from app.cv.availability import CVAvailability

DEFERRED_MODULES = ("Medical RAG", "Clinical Cases", "Consultation Reports")


def render_overview(availability: CVAvailability) -> None:
    gr.Markdown(
        "## Tổng quan\n"
        "Không gian nghiên cứu dành cho nhân viên y tế đã được đào tạo, hỗ trợ xem ảnh soi da "
        "và đặt nền tảng cho việc tra cứu tài liệu da liễu và chuẩn bị hội chẩn.\n\n"
        "| Mô-đun | Trạng thái |\n"
        "| --- | --- |\n"
        "| Computer Vision | Tải và xem ảnh |\n"
        f"| Phân đoạn · {availability.segmentation.model_id} | {availability.segmentation.label} |\n"
        f"| Phát hiện · {availability.detection.model_id} | {availability.detection.label} |\n"
        + "".join(f"| {module} | Chưa triển khai |\n" for module in DEFERRED_MODULES)
    )
