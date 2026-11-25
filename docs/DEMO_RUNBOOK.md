# CV MVP — demo cục bộ (25/11/2026)

Chỉ dùng cho nghiên cứu; chưa được thẩm định lâm sàng, không tự động chẩn đoán. Điểm detection không phải xác suất ung thư. Kết quả acceptance của máy này: [DEMO_RELEASE_ACCEPTANCE.md](DEMO_RELEASE_ACCEPTANCE.md).

## Khởi động

Chạy PowerShell tại gốc repository. Môi trường đã kiểm chứng: Windows 11, Python **3.12.5**, `.venv\Scripts\python.exe`; Torch **2.10.0+cpu**, Torchvision **0.25.0+cpu**, Gradio **6.32.0**, Pillow **12.3.0**, segmentation-models-pytorch **0.5.0**, OpenCV headless **4.13.0.92**. Dependency ứng dụng/CV nằm trong `requirements.txt`, `requirements-cv.txt`; kiểm thử dùng `requirements-dev.txt`, Playwright **1.65.0** và Chromium đã cài. Không tạo lại venv hoặc cài lại Torch trên máy demo đang hoạt động.

| Artifact bắt buộc | Phiên bản | SHA-256 |
| --- | --- | --- |
| `model_store/segformer/real_only/best.pt` | `checkpoint-v1` | `d5a92fa31296a4f7c32e87cdf84683119d07dc64410253103bfa285aacb93f87` |
| `model_store/detection/E1-final-inference.pt` | `E1-best-completed` | `eed8c28c2777005a2c6033692137f3cac8df8c68251ecb342ee4b8edf3068c9f` |

Registry: **`config/models.json` tại gốc repo**, không phải `app/config/models.json`. Hai entry trên phải active; pilot inactive không bắt buộc. Ngưỡng SegFormer **0,7**, detection **0,5**. Không thay weights, hash hoặc ngưỡng để chạy demo.

```powershell
.venv\Scripts\Activate.ps1
python main.py
```

Không cần đặt biến môi trường registry/device: tự đọc `config/models.json`, chọn model active và dùng device `auto`. Có thể override bằng `LESION_MODEL_REGISTRY`, `LESION_SEGFORMER_DEVICE`, `LESION_DETECTION_DEVICE` khi cần; `python -m app.main` vẫn hoạt động. Terminal báo registry, version, device thực tế, trạng thái từng model và lỗi kèm đường dẫn checkpoint. Startup không tải weights hoặc tự bật pilot. Xem [README](../README.md) để đăng ký phiên bản E3 đã xác minh.

Mở **http://127.0.0.1:7860** sau khi terminal báo URL. Giữ terminal mở. Kết thúc bằng **Ctrl+C**, chờ tiến trình thoát; không đóng cưỡng bức cửa sổ trong lúc phân tích. Nếu chạy đồng thời nhiều tiến trình, đặt `LESION_RUNTIME_DIR` và `LESION_PORT` riêng cho từng tiến trình.

## Kịch bản năm phút

1. **0:00–0:45:** mở ứng dụng, đọc tuyên bố nghiên cứu; chọn **Computer Vision**. Xác nhận cả hai mô hình “Sẵn sàng”.
2. **0:45–1:30:** tải ảnh demo được phép có sẵn: `model_store/wp2-verification/dermoscopy.jpg`. Original đúng ảnh **330 × 244 px**. Bấm **Phân tích**, chờ cả hai mô hình.
3. **1:30–2:30:** mở Mask, giải thích mask nhị phân. Overlay phủ mask lên cùng ảnh; đổi opacity **0%, 40%, 100%**. Thao tác này chỉ đổi hiển thị, không chạy lại mô hình.
4. **2:30–3:30:** mở Localization: **vàng** là box suy từ mask SegFormer, **đỏ** là box do Faster R-CNN dự đoán độc lập, kèm score. Không diễn giải thành chẩn đoán. Nếu mask rỗng/không có detection, trình bày trạng thái thật; không tự thêm box.
5. **3:30–4:15:** mở **Chi tiết mô hình**, chỉ hai phiên bản, hash, device CPU, thời gian từng mô hình và tổng. Lượt đo trên máy: khoảng **0,79–0,93 giây SegFormer**, **5,16–5,39 giây detector** trong năm lượt browser; đây là quan sát, không phải cam kết độ trễ.
6. **4:15–5:00:** tải **Overlay PNG** và **Localization PNG**, mở bằng trình xem ảnh, kiểm tra kích thước gốc. Đổi ảnh: kết quả cũ biến mất. Bấm **Đặt lại**: bốn thẻ trống, hai nút xuất bị vô hiệu hóa.

Ảnh demo hiện tại cho mask 1.016 pixel, box mask `[138, 89, 181, 124]`; detection xấp xỉ `[134.668, 87.285, 183.778, 126.461]`, score `0.803465`. Không dùng các số này thay cho inference thực tế. Ảnh tải xuống nằm trong thư mục Download của trình duyệt; không commit ảnh xuất.

## Khôi phục sự cố

| Sự cố | Các bước chính xác |
| --- | --- |
| Cổng 7860 bận | `Get-NetTCPConnection -LocalPort 7860 -State Listen` rồi `Get-Process -Id <OwningProcess>` để xác định tiến trình. Nếu là phiên demo của mình, Ctrl+C ở terminal đó. Nếu thuộc ứng dụng khác, giữ nguyên; đặt `$env:LESION_PORT = '7861'`, chạy lại và mở `http://127.0.0.1:7861`. |
| Thiếu SegFormer | Dừng app; kiểm tra `Test-Path 'model_store/segformer/real_only/best.pt'`. Nếu có bản sao đã duyệt, chép về đúng đường dẫn rồi xác minh hash. Nếu cần dùng downloader đã có: `.venv\Scripts\python.exe -m app.cv.checkpoint`; sau đó `.venv\Scripts\python.exe -m app.cv.checkpoint --verify-only`. Chỉ tiếp tục khi hash đúng; không ghi đè artifact đang tồn tại. |
| Thiếu Faster R-CNN | Dừng app; kiểm tra `Test-Path 'model_store/detection/E1-final-inference.pt'`. Lấy lại **đúng artifact E1 inference** từ bản sao đã xác minh, đặt đúng đường dẫn và kiểm tra hash trong bảng. Không đổi tên pilot hoặc training checkpoint thành E1 inference. Nếu chưa lấy được file, gate demo đủ hai mô hình bị chặn. |
| Checksum sai | Dừng app, dùng `Get-FileHash -Algorithm SHA256 -LiteralPath '<đường dẫn artifact>'`, đối chiếu bảng. Giữ file sai để điều tra; lấy bản đúng sang đường dẫn mới chưa tồn tại, kiểm tra hash, sau đó cấu hình entry trỏ tới bản đúng. Không sửa expected hash để chấp nhận file chưa xác minh; không tắt verification. |
| CPU hết RAM | Dừng phiên demo bằng Ctrl+C. Đóng ứng dụng nặng do mình quản lý, kiểm tra RAM trống trong Task Manager. Chạy lại **một** server bằng lệnh CPU phía trên và dùng ảnh demo 330×244. Nếu vẫn OOM, dừng demo đủ hai mô hình và ghi nhận lỗi; không giảm input/threshold của mô hình hoặc dùng fallback weights. |
| CUDA không dùng được | Máy này có RTX 3050 nhưng Torch là bản CPU; CUDA chưa kiểm chứng. Đặt lại cả `LESION_SEGFORMER_DEVICE` và `LESION_DETECTION_DEVICE` thành `cpu`, khởi động lại. Không thay bộ Torch trong buổi trình bày. |
| Browser chặn tải PNG | Cho phép download cho đúng địa chỉ loopback trong cài đặt trình duyệt, tải lại bằng nút của ứng dụng. Nếu phiên đã hết hạn/đã Reset, tải ảnh và Analyze lại trước. Không mở đường dẫn file của server hoặc nới quyền file-serving. Nếu thiếu Chromium cho QA, chạy `.venv\Scripts\python.exe -m playwright install chromium`; lỗi mạng phải báo browser test **NOT VERIFIED**, không coi là pass. |
| Mô hình khởi tạo lỗi | Đọc trạng thái từng mô hình và log terminal; chạy `.venv\Scripts\python.exe -m pip check`; xác minh hai đường dẫn/hash và registry ở gốc repo. Đặt CPU như trên, dừng rồi chạy lại app. Lỗi một mô hình có thể cho kết quả `partial`, nhưng không thay thế gate demo đủ hai mô hình. Nếu registry bị chuyển mất, khôi phục đúng bản từ Git hoặc bản sao byte-identical trước khi khởi động. |

Không xóa toàn bộ `runtime/` khi còn tiến trình sử dụng nó. Reset thu hồi export và dọn ảnh/mask của phiên; cache upload riêng của Gradio được dọn theo tuổi tệp, không nhất thiết biến mất ngay. Không dùng ảnh bệnh nhân trong demo này.

## Thay E2/E3 trong tương lai và rollback

1. Chuẩn bị artifact inference từ run đã hoàn tất theo quy trình hiện có; lưu ở **đường dẫn mới**, giữ nguyên E1. Xác minh SHA-256 và metadata trước khi đăng ký.
2. Sao chép entry detection trong `config/models.json`; điền version E2/E3, path, SHA-256, architecture, `artifact_format`, threshold, min/max size và preprocessing khớp artifact. Export chuẩn dùng `isic-fasterrcnn-inference-v1`, `research_pilot=false`.
3. Khi chủ động chuyển phiên bản, tắt entry detection cũ và bật entry mới; chỉ một entry active mỗi mô-đun. Dừng rồi khởi động lại bằng cùng `LESION_MODEL_REGISTRY`. Chạy lại real CPU/browser acceptance trước khi demo.
4. Rollback: bật lại E1, tắt E2/E3 và khởi động lại. Kiểm tra UI hiện `E1-best-completed` cùng SHA-256 đã ghi; không cần sửa UI.

Acceptance hiện tại chỉ thử selection/rollback bằng registry tạm; **không** thay active registry của demo và **không** xác nhận chất lượng E2/E3 chưa có.
