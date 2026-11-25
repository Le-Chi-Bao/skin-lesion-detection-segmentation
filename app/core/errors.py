from enum import StrEnum


class ErrorCode(StrEnum):
    MISSING_FILE = "missing_file"
    INVALID_IMAGE = "invalid_image"
    UNSUPPORTED_FORMAT = "unsupported_format"
    UPLOAD_TOO_LARGE = "upload_too_large"
    IMAGE_TOO_LARGE = "image_too_large"
    INVALID_SESSION = "invalid_session"
    MODEL_UNAVAILABLE = "model_unavailable"
    INVALID_MODEL_CONFIG = "invalid_model_config"
    CHECKPOINT_MISSING = "checkpoint_missing"
    CHECKPOINT_INCOMPATIBLE = "checkpoint_incompatible"
    CHECKPOINT_READ = "checkpoint_read"
    CHECKPOINT_WEIGHTS_ONLY = "checkpoint_weights_only"
    MODEL_DEPENDENCIES = "model_dependencies"
    MODEL_DEVICE = "model_device"
    MODEL_OUT_OF_MEMORY = "model_out_of_memory"
    INVALID_MODEL_OUTPUT = "invalid_model_output"
    INFERENCE_FAILED = "inference_failed"
    TEMPORARY_FILE = "temporary_file"
    UNSAFE_PATH = "unsafe_path"
    INVALID_CONFIG = "invalid_config"
    INTERNAL = "internal_error"


MESSAGES = {
    ErrorCode.MISSING_FILE: "Không tìm thấy ảnh. Vui lòng tải lại ảnh.",
    ErrorCode.INVALID_IMAGE: "Không đọc được ảnh. Tệp có thể bị hỏng hoặc chứa nhiều khung hình.",
    ErrorCode.UNSUPPORTED_FORMAT: "Chỉ hỗ trợ ảnh JPEG và PNG.",
    ErrorCode.UPLOAD_TOO_LARGE: "Tệp vượt quá dung lượng cho phép ({limit} MiB).",
    ErrorCode.IMAGE_TOO_LARGE: "Kích thước ảnh vượt giới hạn ({edge} px mỗi cạnh, {pixels:,} pixel).",
    ErrorCode.INVALID_SESSION: "Phiên hoặc ảnh hiện tại đã thay đổi. Vui lòng tải lại ảnh.",
    ErrorCode.MODEL_UNAVAILABLE: "Mô hình phân tích chưa sẵn sàng.",
    ErrorCode.INVALID_MODEL_CONFIG: "Cấu hình mô hình không hợp lệ.",
    ErrorCode.CHECKPOINT_MISSING: "Chưa có checkpoint SegFormer. Hãy kiểm tra đường dẫn artifact đã đăng ký và hướng dẫn trong README.",
    ErrorCode.CHECKPOINT_INCOMPATIBLE: "Checkpoint SegFormer không khớp kích thước, SHA-256 hoặc kiến trúc đã xác minh.",
    ErrorCode.CHECKPOINT_READ: "Không thể đọc checkpoint SegFormer. Hãy kiểm tra quyền truy cập tệp.",
    ErrorCode.CHECKPOINT_WEIGHTS_ONLY: "Không thể đọc checkpoint SegFormer bằng torch.load với weights_only=True.",
    ErrorCode.MODEL_DEPENDENCIES: "Thiếu thư viện SegFormer hoặc phiên bản không tương thích. Hãy kiểm tra hướng dẫn cài đặt CV.",
    ErrorCode.MODEL_DEVICE: "Không thể khởi tạo SegFormer trên thiết bị đã chọn. Hãy kiểm tra PyTorch, trình điều khiển hoặc chọn CPU.",
    ErrorCode.MODEL_OUT_OF_MEMORY: "Thiết bị hết bộ nhớ khi chạy SegFormer. Hãy giải phóng bộ nhớ hoặc chọn CPU.",
    ErrorCode.INVALID_MODEL_OUTPUT: "SegFormer trả về đầu ra sai kích thước, sai kiểu hoặc chứa NaN/Inf.",
    ErrorCode.INFERENCE_FAILED: "Suy luận SegFormer thất bại. Vui lòng thử lại.",
    ErrorCode.TEMPORARY_FILE: "Không thể xử lý tệp tạm. Vui lòng thử lại.",
    ErrorCode.UNSAFE_PATH: "Không thể truy cập tệp từ vị trí này.",
    ErrorCode.INVALID_CONFIG: "Cấu hình ứng dụng không hợp lệ.",
    ErrorCode.INTERNAL: "Đã xảy ra lỗi khi xử lý ảnh. Vui lòng thử lại.",
}


class AppError(Exception):
    def __init__(self, code: ErrorCode, *, field: str = "", **message_values: int):
        self.code = code
        self.field = field
        super().__init__(MESSAGES[code].format(**message_values))
