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
