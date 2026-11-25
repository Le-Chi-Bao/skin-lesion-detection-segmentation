import gradio as gr

from app.cv.availability import CVAvailability, ModelStatus

DEFERRED_MODULES = ("Clinical Cases", "Consultation Reports")


def render_overview(availability: CVAvailability) -> None:
    cv_description = "Tải, xem và phân đoạn ảnh" if availability.segmentation.status is ModelStatus.READY else "Tải và xem ảnh"
    if availability.detection.status is ModelStatus.READY:
        cv_description += "; phát hiện tổn thương độc lập"
    gr.Markdown(
        "## Tổng quan\n"
        "Không gian nghiên cứu dành cho nhân viên y tế đã được đào tạo, hỗ trợ xem ảnh soi da "
        "và đặt nền tảng cho việc tra cứu tài liệu da liễu và chuẩn bị hội chẩn.\n\n"
        "| Mô-đun | Trạng thái |\n"
        "| --- | --- |\n"
        f"| Computer Vision | {cv_description} |\n"
        f"| Phân đoạn · {availability.segmentation.model_id} | {availability.segmentation.label} |\n"
        f"| Phát hiện · {availability.detection.model_id} | {availability.detection.label} |\n"
        "| Medical RAG | WP6A: BM25 / Dense / Hybrid; cần corpus đủ quyền hoặc SYNTHETIC DEMO |\n"
        + "".join(f"| {module} | Chưa triển khai |\n" for module in DEFERRED_MODULES)
    )
