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
from app.cv.analysis import discard_result, execute_analysis
from app.cv.availability import CVAvailability, ModelStatus, RunStatus
from app.cv.schemas import CVResult
from app.cv.segformer import SegFormer
from app.cv.visualization import (
    DEFAULT_OPACITY, display_size, draw_localization, encode_png, overlay_layer, validate_alignment,
)

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
        document.addEventListener("input", event => {
            const input = event.target;
            if (!(input instanceof HTMLInputElement) || !input.closest("#overlay-opacity")) return;
            const layer = document.querySelector("#lesion-overlay .lesion-layer");
            if (layer && Number.isFinite(input.valueAsNumber)) {
                layer.style.opacity = Math.min(100, Math.max(0, input.valueAsNumber)) / 100;
            }
        });
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


VIEWS = {
    "preview": ("Ảnh soi da xem trước", "Chưa có ảnh."),
    "original": ("Ảnh soi da đã tải lên", "Tải ảnh soi da để xem ảnh gốc."),
    "mask": ("Mặt nạ tổn thương nhị phân", "Chưa có mặt nạ phân đoạn."),
    "overlay": ("Ảnh soi da chồng mặt nạ phân đoạn", "Chưa có ảnh chồng mặt nạ."),
    "localization": ("Ảnh soi da với bounding box suy ra từ mask", "Chưa có vùng định vị từ mặt nạ."),
}


def image_panel(
    kind: str, src: str | None = None, *, layer: str | None = None, empty: str | None = None,
) -> str:
    description, placeholder = VIEWS[kind]
    content = f'<img src="{src}" alt="{description}">' if src else (empty or placeholder)
    if layer:
        content = (
            f'<div class="lesion-stack">{content}'
            f'<img class="lesion-layer" src="{layer}" alt="" style="opacity:{DEFAULT_OPACITY / 100}"></div>'
        )
    return f'<div class="lesion-view lesion-{kind}">{content}</div>'


def image_source(image: Image.Image, *, png: bool = False) -> str:
    with io.BytesIO() as buffer:
        if png:
            encode_png(image, buffer)
        else:
            image.save(buffer, format="JPEG", quality=90)
        mime = "png" if png else "jpeg"
        return f"data:image/{mime};base64," + base64.b64encode(buffer.getbuffer()).decode("ascii")


def run_status(message: str, checkpoint: str | None = None, *, alert: bool = False) -> str:
    role = "alert" if alert else "status"
    code = f"<br><code>{escape(checkpoint)}</code>" if checkpoint else ""
    return f'<div role="{role}">{escape(message)}{code}</div>'


def result_details(result: CVResult, status: str) -> str:
    segmentation = result.segmentation
    model = segmentation.model
    rows = [
        ("Trạng thái", status),
        ("Mô hình", model.model_id if model else None),
        ("Checkpoint", model.checkpoint_id if model else None),
        ("SHA-256", model.checkpoint_sha256 if model else None),
        ("Thiết bị", model.device.upper() if model and model.device else None),
        ("Ngưỡng phân đoạn", str(segmentation.threshold) if segmentation.threshold is not None else None),
        ("Kích thước ảnh gốc", f"{result.image_width} × {result.image_height} px"),
        ("Thời gian suy luận", f"{result.metadata.inference_duration_ms:.2f} ms"
         if result.metadata.inference_duration_ms is not None else None),
        ("Nguồn định vị", "Bounding box suy ra từ mask" if result.localization else None),
    ]
    return "\n\n".join(f"**{label}** · `{value}`" for label, value in rows if value is not None)


def render_result_images(files: TemporaryFiles, current: SessionState, source_file: str, result: CVResult) -> tuple:
    reference = result.segmentation.mask
    size = result.image_width, result.image_height
    session_id = current.session_id
    with (
        files.operation(session_id),
        files.open_image(session_id, source_file) as source_stream,
        files.open_mask(session_id, reference.reference) as mask_stream,
        Image.open(source_stream, formats=("PNG",)) as source,
        Image.open(mask_stream, formats=("PNG",)) as binary,
    ):
        validate_alignment(source, binary, size)
        dimensions = display_size(size)
        with (
            source.resize(dimensions, Image.Resampling.LANCZOS) as preview,
            binary.resize(dimensions, Image.Resampling.NEAREST) as display_mask,
            overlay_layer(display_mask) as layer,
        ):
            mask = image_panel("mask", image_source(display_mask, png=True))
            overlay = image_panel("overlay", image_source(preview), layer=image_source(layer, png=True))
            if result.localization is None:
                return mask, overlay, image_panel("localization", empty="Mặt nạ rỗng. Không có vùng định vị.")
            box = result.localization.box
            with draw_localization(preview, box, source_size=size) as localized:
                localization = image_panel("localization", image_source(localized, png=True))
    coordinates = (
        f'<p class="lesion-coordinates"><strong>Bounding box suy ra từ mask</strong> · '
        f'<code>[{box.x1:g}, {box.y1:g}, {box.x2:g}, {box.y2:g}]</code> px. '
        'Cạnh phải và cạnh dưới là tọa độ loại trừ.</p>'
    )
    return mask, overlay, coordinates + localization


def render_cv_page(config: AppConfig, files: TemporaryFiles, availability: CVAvailability, segmenter: SegFormer) -> None:
    ready = availability.segmentation.status is ModelStatus.READY
    idle_status = run_status(str(segmenter.error), alert=True) if segmenter.error else run_status("Chưa có ảnh để phân tích.")

    def cleanup(state: SessionState | None) -> None:
        if state is not None:
            with state.lock:
                state.reset()
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
                analyze_button = gr.Button("Phân tích", interactive=False, elem_id="analyze-image")
                reset_button = gr.Button("Đặt lại", elem_id="reset-image")
            oversize_button = gr.Button("Tệp vượt giới hạn", visible="hidden", elem_id="oversize-image")
            gr.Markdown(
                f"**Phân đoạn · {availability.segmentation.model_id}**  \n{availability.segmentation.label}\n\n"
                f"**Phát hiện · {availability.detection.model_id}**  \n{availability.detection.label}"
            )
            status = gr.HTML(idle_status, js_on_load=None, elem_id="segmentation-status")
        with gr.Column(scale=3, min_width=0, elem_id="cv-viewer"):
            gr.Markdown("### Xem ảnh")
            with gr.Tabs(selected="original", overflow_behavior="wrap") as tabs:
                with gr.Tab("Original", id="original"):
                    original = gr.HTML(image_panel("original"), js_on_load=None, elem_id="lesion-original")
                with gr.Tab("Mask", id="mask"):
                    mask = gr.HTML(image_panel("mask"), js_on_load=None, elem_id="lesion-mask")
                with gr.Tab("Overlay", id="overlay"):
                    overlay = gr.HTML(image_panel("overlay"), js_on_load=None, elem_id="lesion-overlay")
                with gr.Tab("Localization", id="localization"):
                    localization = gr.HTML(image_panel("localization"), js_on_load=None, elem_id="lesion-localization")
            opacity = gr.Slider(
                0, 100, DEFAULT_OPACITY, step=1, label="Độ mờ lớp phủ (%)", interactive=False,
                elem_id="overlay-opacity",
            )
            warnings = gr.HTML('<div role="status"></div>', js_on_load=None, elem_id="lesion-warnings")
            with gr.Accordion("Chi tiết mô hình", open=False):
                details_view = gr.Markdown("Chưa có lượt phân tích.", elem_id="lesion-model-details")

    def empty_analysis(interactive: bool, message: str | None = None, details: str = "Chưa có lượt phân tích.") -> tuple:
        return (
            gr.update(interactive=interactive), image_panel("mask"),
            message or idle_status, image_panel("localization"), image_panel("overlay"),
            gr.update(value=DEFAULT_OPACITY, interactive=False), details, '<div role="status"></div>',
        )

    def clear_image(current: SessionState, message: str = "") -> tuple:
        with current.lock:
            previous_result = current.result
            previous = current.reset()
        try:
            with files.operation(current.session_id):
                if previous is not None:
                    files.discard(current.session_id, previous.temporary_file)
                discard_result(files, current, previous_result)
        except (AppError, OSError) as exc:
            log_error(logger, exc, current.session_id)
        return (
            current, image_panel("preview"), "Chưa có thông tin ảnh.",
            f'<div role="alert">{escape(message)}</div>', image_panel("original"), gr.Tabs(selected="original"),
            *empty_analysis(False),
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
                with image.resize(display_size(image.size), Image.Resampling.LANCZOS) as display:
                    src = image_source(display)
                stored_file = files.write(current.session_id, lambda stream: image.save(stream, format="PNG"))
                with current.lock:
                    previous_result = current.result
                    previous = current.replace_image(details, stored_file)
                if previous is not None:
                    files.discard(current.session_id, previous.temporary_file)
                discard_result(files, current, previous_result)
        except Exception as exc:
            log_error(logger, exc, current.session_id)
            message = str(exc) if isinstance(exc, AppError) else str(AppError(ErrorCode.INTERNAL))
            return clear_image(current, message)
        summary = f"**{details.width} × {details.height} px** · {details.format} · {details.size_bytes / MIB:.2f} MiB"
        return (
            current, image_panel("preview", src), summary, '<div role="alert"></div>',
            image_panel("original", src), gr.Tabs(selected="original"),
            *empty_analysis(ready, run_status("Đã tải ảnh. Sẵn sàng phân tích.") if ready else idle_status),
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

    analysis_outputs = [analyze_button, mask, status, localization, overlay, opacity, details_view, warnings]
    outputs = [state, preview, metadata, error, original, tabs, *analysis_outputs]
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

    if not ready:
        return

    def begin_analysis(current: SessionState | None, action: int) -> tuple:
        if current is None:
            return tuple(gr.skip() for _ in analysis_outputs)
        with current.lock:
            if not current.matches_image_action(action) or current.image is None or current.active_run is not None:
                return tuple(gr.skip() for _ in analysis_outputs)
            previous_result = current.result
            current.begin_run()
        try:
            with files.operation(current.session_id):
                discard_result(files, current, previous_result)
        except (AppError, OSError) as exc:
            log_error(logger, exc, current.session_id)
        return empty_analysis(False, run_status("Đang phân tích ảnh bằng SegFormer…"), "Đang xử lý ảnh.")

    def run_analysis(current: SessionState | None) -> None:
        if current is not None:
            execute_analysis(current, files, segmenter)

    def show_analysis(current: SessionState | None, action: int) -> tuple:
        skip = tuple(gr.skip() for _ in (*analysis_outputs, tabs))
        if current is None:
            return skip
        with current.lock:
            result = current.result if current.matches_image_action(action) else None
            image = current.image
        if result is None:
            return skip
        if result.metadata.status is RunStatus.FAILED:
            return (*empty_analysis(
                True, run_status(str(AppError(result.metadata.error_code)), alert=True),
                result_details(result, "Phân tích thất bại"),
            ), gr.skip())

        segmentation = result.segmentation
        try:
            mask_view, overlay_view, localization_view = render_result_images(files, current, image.temporary_file, result)
        except (AppError, OSError, ValueError) as exc:
            log_error(logger, exc, current.session_id)
            with current.lock:
                if current.result is not result or not current.matches_image_action(action):
                    return skip
            return (*empty_analysis(True, run_status(str(AppError(ErrorCode.TEMPORARY_FILE)), alert=True)), gr.skip())

        with current.lock:
            if current.result is not result or not current.matches_image_action(action):
                return skip
        summary = run_status(f"Phân đoạn hoàn tất · {segmentation.model.model_id}", segmentation.model.checkpoint_id)
        caution = " ".join(segmentation.warnings)
        if "empty_mask" in result.metadata.warning_codes:
            caution = "Không có pixel tiền cảnh đạt ngưỡng phân đoạn. Kết quả này không dùng để kết luận về bệnh lý."
        return (
            gr.update(interactive=True), mask_view, summary, localization_view, overlay_view,
            gr.update(value=DEFAULT_OPACITY, interactive=True), result_details(result, "Phân đoạn hoàn tất"),
            f'<div role="status">{escape(caution)}</div>', gr.Tabs(selected="overlay"),
        )

    # Only the short start and display callbacks share the upload/Reset queue.
    # The worker holds a RunToken and never returns a Gradio UI or State update.
    start = analyze_button.click(
        begin_analysis, [state, action_index], analysis_outputs, js=current_action_js, **events,
    )
    work = start.success(
        run_analysis, state, [], concurrency_id="cv-inference", concurrency_limit=1, api_visibility="private",
    )
    work.success(show_analysis, [state, action_index], [*analysis_outputs, tabs], js=current_action_js, **events)
