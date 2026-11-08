# Colab notebooks

`01_foundation_smoke.ipynb` chỉ phục vụ setup và kiểm tra Phase 1. Nó không chứa
training loop, không tải dataset hoặc pretrained weights. Đưa source checkout
lên Colab/Drive trước, mở notebook và nhập đường dẫn project root khi được hỏi.

Notebook dùng `sys.executable` cho pip/subprocess và `pathlib` cho đường dẫn.
Runtime cần Python >= 3.11. PyTorch/Torchvision có sẵn trên Colab sẽ được giữ nếu
đáp ứng dependency; kiểm tra import trước khi cài. T4/CUDA thực tế được kiểm tra
bằng smoke script; chế độ GPU không bắt buộc cho foundation tests.

Huấn luyện và DataLoader chỉ được thêm khi bắt đầu phase tương ứng.
