"""Phase 1 UI: upload and inspect an image; inference is not implemented."""

from pathlib import Path

import streamlit as st
from PIL import UnidentifiedImageError

from src.data.image_io import read_rgb
from src.utils.config import load_config
from src.utils.device import get_device


def _checkpoint_message(label: str, checkpoint: Path | None) -> None:
    if checkpoint is None:
        st.info(f"{label}: chưa cấu hình checkpoint; mô hình chưa được huấn luyện.")
    elif not checkpoint.is_file():
        st.warning(f"{label}: không tìm thấy checkpoint đã cấu hình.")
    else:
        st.info(
            f"{label}: đã tìm thấy tệp checkpoint. Phase 1 chưa nạp trọng số "
            "hoặc hỗ trợ suy luận; chất lượng mô hình chưa được xác minh."
        )


def main() -> None:
    st.set_page_config(page_title="Skin Lesion Detection", layout="wide")
    config = load_config()
    st.title(config.project_name)
    st.caption("ISIC 2018 Task 1 · Phase 1 — Project Foundation")
    st.sidebar.header("Cấu hình")
    st.sidebar.write(f"Thiết bị khả dụng: {get_device(config.device)}")
    st.sidebar.write(f"Kích thước ROI: {config.image_size}")
    st.sidebar.write(f"ROI padding: {config.roi_padding:.0%} mỗi cạnh")
    st.info(
        "Phase 1 chỉ cung cấp nền tảng và xem ảnh đầu vào. "
        "Chưa có mô hình được huấn luyện hoặc kết quả dự đoán."
    )

    uploaded = st.file_uploader(
        "Tải ảnh Dermoscopy (JPG/PNG)", type=["jpg", "jpeg", "png"]
    )
    st.subheader("Ảnh đầu vào")
    if uploaded is not None:
        try:
            image = read_rgb(uploaded)
        except (UnidentifiedImageError, OSError, ValueError) as error:
            st.error(f"Không thể đọc ảnh: {error}")
        else:
            st.image(image, caption=uploaded.name, width="stretch")
            st.caption(f"Kích thước gốc: {image.shape[1]} × {image.shape[0]} pixel")
    else:
        st.caption("Tải ảnh lên để hiển thị ảnh đầu vào.")

    detection_column, segmentation_column = st.columns(2)
    with detection_column:
        st.subheader("Detection — Faster R-CNN")
        _checkpoint_message("Faster R-CNN", config.detector_checkpoint)
        st.caption("Bounding box sẽ được hiển thị sau khi triển khai suy luận.")
    with segmentation_column:
        st.subheader("Segmentation — U-Net")
        _checkpoint_message("U-Net", config.segmenter_checkpoint)
        st.caption("Mask/overlay sẽ được hiển thị sau khi triển khai suy luận.")


if __name__ == "__main__":
    main()
