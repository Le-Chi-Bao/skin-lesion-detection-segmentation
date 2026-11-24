import base64
import io
import logging
from html import escape
from typing import Any

import gradio as gr
from PIL import Image

from app.config import AppConfig, MIB
from app.core.errors import AppError, ErrorCode
from app.core.image_validation import uploaded_path, validate_image
from app.core.logging_config import log_error
from app.core.session import SessionState
from app.core.temporary_files import TemporaryFiles
from app.cv.availability import CVAvailability

logger = logging.getLogger(__name__)


def upload_browser_setup(max_bytes: int) -> str:
    # Remember user action order before the multipart transfer and queue requests.
    return """
    () => {
        let revision = 0;
        const uploadName = action => "wp1-upload-" + action;
        window.lesionImageActions = {
            current: () => revision,
            upload: file => file?.orig_name === uploadName(revision) ? revision : 0
        };
        document.addEventListener("click", event => {
            if (event.target.closest("#reset-image")) revision++;
        }, true);
        document.addEventListener("change", event => {
            const input = event.target;
            if (!(input instanceof HTMLInputElement) || input.type !== "file" ||
                !input.closest("#cv-controls") || !input.files?.length) return;
            const file = input.files[0];
            const action = ++revision;
            if (file.size > MAX_UPLOAD_BYTES) {
                event.stopImmediatePropagation();
                input.value = "";
                document.querySelector("button#oversize-image, #oversize-image button")?.click();
                return;
            }
            // Gradio returns this name in FileData.orig_name when this transfer completes.
            const transfer = new DataTransfer();
            transfer.items.add(new File([file], uploadName(action), {type: file.type}));
            input.files = transfer.files;
        }, true);
    }
    """.replace("MAX_UPLOAD_BYTES", str(max_bytes))


def image_panel(kind: str, src: str | None = None) -> str:
    content = f'<img src="{src}" alt="Ảnh soi da đã tải lên">' if src else "Chưa có ảnh."
    return f'<div class="lesion-view lesion-{kind}">{content}</div>'


def preview_source(image: Image.Image) -> str:
    with image.copy() as preview, io.BytesIO() as buffer:
        preview.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
        preview.save(buffer, format="JPEG", quality=90)
        return "data:image/jpeg;base64," + base64.b64encode(buffer.getbuffer()).decode("ascii")


def render_cv_page(config: AppConfig, files: TemporaryFiles, availability: CVAvailability) -> None:
    def cleanup(state: SessionState | None) -> None:
        if state is not None:
            files.close_session(state.session_id)

    state = gr.State(delete_callback=cleanup)
    action_index = gr.Number(value=0, precision=0, visible="hidden", elem_id="image-action-index")
    gr.Markdown("## Computer Vision")
    with gr.Row():
        with gr.Column(scale=1, min_width=260, elem_id="cv-controls"):
            gr.Markdown(
                "### Ảnh đầu vào\n"
                f"Chọn một ảnh JPEG hoặc PNG, tối đa {config.max_upload_mib} MiB."
            )
            upload_button = gr.UploadButton(
                "Tải ảnh lên", type="filepath", file_count="single", variant="primary", elem_id="upload-image",
            )
            preview = gr.HTML(image_panel("preview"), js_on_load=None, elem_id="lesion-preview")
            metadata = gr.Markdown("Chưa có thông tin ảnh.")
            error = gr.HTML('<div role="alert"></div>', js_on_load=None, elem_id="upload-error")
            with gr.Row():
                gr.Button("Phân tích", interactive=False, elem_id="analyze-image")
                reset_button = gr.Button("Đặt lại", elem_id="reset-image")
            oversize_button = gr.Button("Tệp vượt giới hạn", visible="hidden", elem_id="oversize-image")
            gr.Markdown(
                f"**Phân đoạn · {availability.segmentation.model_id}**  \n{availability.segmentation.label}\n\n"
                f"**Phát hiện · {availability.detection.model_id}**  \n{availability.detection.label}"
            )
        with gr.Column(scale=3, min_width=325):
            gr.Markdown("### Xem ảnh")
            with gr.Tabs(selected="original") as tabs:
                with gr.Tab("Original", id="original"):
                    original = gr.HTML(image_panel("original"), js_on_load=None, elem_id="lesion-original")
                with gr.Tab("Mask", id="mask"):
                    gr.Markdown("Chưa có mặt nạ phân đoạn.")
                with gr.Tab("Overlay", id="overlay"):
                    gr.Markdown("Chưa có ảnh chồng lớp.")
                with gr.Tab("Localization", id="localization"):
                    gr.Markdown("Chưa có kết quả định vị.")

    def clear_image(current: SessionState, message: str = "") -> tuple:
        previous = current.reset()
        if previous is not None:
            try:
                with files.operation(current.session_id):
                    files.discard(current.session_id, previous.temporary_file)
            except (AppError, OSError) as exc:
                log_error(logger, exc, current.session_id)
        return (
            current, image_panel("preview"), "Chưa có thông tin ảnh.",
            f'<div role="alert">{escape(message)}</div>', image_panel("original"), gr.Tabs(selected="original"),
        )

    def upload(payload: dict[str, Any] | None, current: SessionState | None, action: int) -> tuple:
        current = current or SessionState()
        if payload is None or not current.accept_image_action(action):
            return tuple(gr.skip() for _ in outputs)
        try:
            raw_path = payload.get("path") if isinstance(payload, dict) else None
            path = uploaded_path(raw_path if isinstance(raw_path, str) else None, files.cache_dir)
            image, details = validate_image(path, config)
            with image, files.operation(current.session_id):
                src = preview_source(image)
                stored_file = files.write(current.session_id, lambda stream: image.save(stream, format="PNG"))
                previous = current.replace_image(details, stored_file)
                if previous is not None:
                    files.discard(current.session_id, previous.temporary_file)
        except Exception as exc:
            log_error(logger, exc, current.session_id)
            message = str(exc) if isinstance(exc, AppError) else str(AppError(ErrorCode.INTERNAL))
            return clear_image(current, message)
        summary = f"**{details.width} × {details.height} px** · {details.format} · {details.size_bytes / MIB:.2f} MiB"
        return (
            current, image_panel("preview", src), summary, '<div role="alert"></div>',
            image_panel("original", src), gr.Tabs(selected="original"),
        )

    def reset(current: SessionState | None, action: int) -> tuple:
        current = current or SessionState()
        if not current.accept_image_action(action):
            return tuple(gr.skip() for _ in outputs)
        return clear_image(current)

    def reject_oversize(current: SessionState | None, action: int) -> tuple:
        current = current or SessionState()
        if not current.accept_image_action(action):
            return tuple(gr.skip() for _ in outputs)
        return clear_image(current, str(AppError(ErrorCode.UPLOAD_TOO_LARGE, limit=config.max_upload_mib)))

    outputs = [state, preview, metadata, error, original, tabs]
    events = {
        "concurrency_id": "cv-session-edits", "concurrency_limit": 1, "api_visibility": "private",
        "trigger_mode": "multiple", "preprocess": False,
    }
    # Read bounded bytes ourselves and avoid UploadButton's built-in file conversion.
    upload_button.upload(
        upload, [upload_button, state, action_index], outputs,
        js="""(file, state) => {
            const action = window.lesionImageActions?.upload(file) ?? 0;
            return [action ? file : null, state, action];
        }""", **events,
    )
    current_action_js = "(state) => [state, window.lesionImageActions?.current() ?? 0]"
    reset_button.click(reset, [state, action_index], outputs, js=current_action_js, **events)
    oversize_button.click(reject_oversize, [state, action_index], outputs, js=current_action_js, **events)
