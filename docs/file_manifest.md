# Phase 1 — danh sách tệp

Các tệp project mới tạo được liệt kê dưới đây. Không sửa hoặc xóa tệp PDF và
`.drawio` có sẵn. `.tools/` là môi trường kiểm chứng cục bộ được Git ignore;
không phải mã nguồn hoặc artifact cần chia sẻ.

## Root và cấu hình

- `.gitignore` — bỏ qua dữ liệu, trọng số, kết quả, runtime và thông tin nhạy cảm.
- `pyproject.toml` — package metadata, dependency, pytest và Ruff settings.
- `requirements.txt` — entrypoint cài editable project kèm dev tools.
- `README.md` — cài đặt, sử dụng, dataset thủ công và roadmap.
- `configs/base.yaml` — giá trị chung, đường dẫn, device và checkpoint tùy chọn.
- `configs/faster_rcnn.yaml` — cấu hình thí nghiệm detector dự kiến.
- `configs/unet_roi_256.yaml` — U-Net ROI 256 dự kiến.
- `configs/unet_roi_512.yaml` — U-Net ROI 512 dự kiến.
- `configs/unet_full_image_512.yaml` — đối chứng full-image 512 dự kiến.

## Source

- `src/__init__.py` — package gốc.
- `src/data/__init__.py` — data package.
- `src/data/image_io.py` — đọc RGB và binary mask.
- `src/data/geometry.py` — resize, bbox, crop, map tọa độ và restore.
- `src/utils/__init__.py` — utilities package.
- `src/utils/config.py` — YAML inheritance, validation và path resolution.
- `src/utils/device.py` — chọn thiết bị CPU/CUDA.
- `src/utils/seed.py` — seed các random generators.
- `src/models/__init__.py` — models package.
- `src/models/detection/__init__.py` — export detector contracts.
- `src/models/detection/interfaces.py` — Detection dataclass và Detector Protocol.
- `src/models/segmentation/__init__.py` — export segmenter contract.
- `src/models/segmentation/interfaces.py` — Segmenter Protocol.
- `src/pipeline/__init__.py` — pipeline package.
- `src/pipeline/spatial.py` — prepare_roi và metadata, chưa suy luận AI.
- `src/evaluation/__init__.py` — package dành cho metrics ở phase sau.

## App và commands

- `app/__init__.py` — app package.
- `app/streamlit_app.py` — upload, ảnh đầu vào và trạng thái chưa có inference.
- `scripts/__init__.py` — commands package.
- `scripts/smoke_test.py` — kiểm chứng foundation và Streamlit AppTest.
- `scripts/check_streamlit_server.py` — server HTTP smoke và tự dừng process.

## Tests

- `tests/test_image_io.py` — đọc ảnh, mask, RGB channels và lỗi tệp.
- `tests/test_geometry.py` — box nửa mở, ROI, resize và restore sát biên.
- `tests/test_config.py` — mọi cấu hình, kế thừa, path và validation.
- `tests/test_device.py` — CPU thật, CUDA selection giả lập, random seeds.
- `tests/test_app.py` — UI, upload hợp lệ/lỗi, checkpoint có/không có.
- `tests/test_smoke.py` — chạy smoke checks đầy đủ qua pytest.

## Notebook và tài liệu

- `notebooks/README.md` — hướng dẫn dùng Colab trong phạm vi Phase 1.
- `notebooks/01_foundation_smoke.ipynb` — setup, kiểm tra runtime và test.
- `docs/system_architecture.md` — luồng dữ liệu, contracts và quy ước tọa độ.
- `docs/file_manifest.md` — danh sách tệp này.
- `docs/verification.md` — kết quả thực thi và giới hạn kiểm chứng thực tế.

## Thư mục dữ liệu/artifact rỗng

- `data/raw/.gitkeep`
- `data/processed/.gitkeep`
- `data/splits/.gitkeep`
- `checkpoints/.gitkeep`
- `outputs/.gitkeep`

Các `.gitkeep` chỉ giữ cấu trúc thư mục; không có ISIC, checkpoint hoặc prediction.
