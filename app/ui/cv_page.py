import base64
import io
import logging
import re
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
from app.cv.orchestrator import CVOrchestrator
from app.cv.visualization import (
    DEFAULT_OPACITY, display_size, draw_localization, draw_combined_localization, encode_png, open_aligned_images,
    open_source, overlay_layer,
)
from app.ui.png_export import EXPORT_PATH, UNAVAILABLE, PNGExports, browser_owner

logger = logging.getLogger(__name__)


def upload_browser_setup(max_bytes: int) -> str:
    # Remember user action order before the multipart transfer and queue requests.
    return """
    () => {
        let revision = 0;
        let pendingExport;
        const proof = Array.from(crypto.getRandomValues(new Uint8Array(32)),
            byte => byte.toString(16).padStart(2, "0")).join("");
        const uploadName = action => "wp1-upload-" + action;
        const opacityInput = () => document.querySelector('#overlay-opacity input[type="range"]');
        window.lesionImageActions = {
            current: () => revision,
            upload: file => file?.orig_name === uploadName(revision)
                ? [file, {index: revision, proof}] : [null, 0]
        };
        document.addEventListener("input", event => {
            const input = event.target;
            if (!(input instanceof HTMLInputElement) || !input.closest("#overlay-opacity")) return;
            const layer = document.querySelector("#lesion-overlay .lesion-layer");
            if (layer && Number.isFinite(input.valueAsNumber)) {
                layer.style.opacity = Math.min(100, Math.max(0, input.valueAsNumber)) / 100;
            }
        });
        document.addEventListener("click", async event => {
            if (event.target.closest("#reset-image")) revision++;
            if (event.target.closest("#reset-image, #analyze-image")) pendingExport?.abort();
            const button = event.target.closest("#lesion-exports button[data-kind]");
            if (!button || button.disabled || pendingExport) return;
            const actions = button.closest("[data-run]");
            const action = Number(actions.dataset.action);
            if (action !== revision) return;
            const output = actions.querySelector('[role="status"]');
            const controller = new AbortController();
            const isCurrent = () => action === revision && actions.isConnected && !controller.signal.aborted;
            pendingExport = controller;
            output.textContent = "Đang chuẩn bị PNG…";
            try {
                const response = await fetch(new URL("." + EXPORT_ENDPOINT, location.href), {
                    method: "POST",
                    headers: {"Content-Type": "application/json"},
                    signal: controller.signal,
                    body: JSON.stringify({
                        session: actions.dataset.session, run: actions.dataset.run, action, proof,
                        kind: button.dataset.kind, opacity: opacityInput()?.valueAsNumber ?? DEFAULT_PERCENT,
                    }),
                });
                if (!response.ok) {
                    const message = await response.text();
                    if (isCurrent()) output.textContent = message || "Không thể xuất PNG. Vui lòng thử lại.";
                    return;
                }
                const blob = await response.blob();
                if (!isCurrent()) return;
                const url = URL.createObjectURL(blob);
                const link = document.createElement("a");
                link.href = url;
                link.download = `segformer-${button.dataset.kind}-${actions.dataset.run.slice(0, 12)}.png`;
                document.body.append(link);
                link.click();
                link.remove();
                setTimeout(() => URL.revokeObjectURL(url), 1000);
                output.textContent = "Đã tạo PNG để tải xuống.";
            } catch (error) {
                if (isCurrent()) output.textContent = "Không thể xuất PNG. Vui lòng thử lại.";
            } finally {
                pendingExport = undefined;
            }
        }, true);
        document.addEventListener("change", event => {
            const input = event.target;
            if (!(input instanceof HTMLInputElement) || input.type !== "file" ||
                !input.closest("#cv-controls") || !input.files?.length) return;
            const file = input.files[0];
            const action = ++revision;
            pendingExport?.abort();
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
    """.replace("MAX_UPLOAD_BYTES", str(max_bytes)).replace(
        "EXPORT_ENDPOINT", f'"{EXPORT_PATH}"',
    ).replace("DEFAULT_PERCENT", str(DEFAULT_OPACITY))


VIEWS = {
    "preview": ("Ảnh soi da xem trước", "Chưa có ảnh."),
    "original": ("Ảnh soi da đã tải lên", "Tải ảnh soi da để xem ảnh gốc."),
    "mask": ("Mặt nạ tổn thương nhị phân", "Chưa có mặt nạ phân đoạn."),
    "overlay": ("Ảnh soi da chồng mặt nạ phân đoạn", "Chưa có ảnh chồng mặt nạ."),
    "localization": ("Ảnh soi da với bounding box suy ra từ mask", "Chưa có kết quả định vị."),
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
    for label, module in (("SegFormer", result.segmentation), ("Faster R-CNN", result.detection)):
        rows.append((f"{label} · trạng thái", module.status.value))
        rows.append((f"{label} · thời gian", f"{module.inference_duration_ms:.2f} ms" if module.inference_duration_ms is not None else None))
        rows.append((f"{label} · lỗi", str(AppError(module.error_code)).replace("SegFormer", label) if module.error_code else None))
        if module.model:
            identity = module.model
            rows.extend((
                (f"{label} · phiên bản", identity.model_version or identity.checkpoint_id),
                (f"{label} · định dạng", identity.artifact_format), (f"{label} · kiến trúc", identity.architecture),
            ))
            if label == "Faster R-CNN":
                rows.extend(((f"{label} · mô hình", identity.model_id), (f"{label} · SHA-256", identity.checkpoint_sha256), (f"{label} · thiết bị", identity.device)))
    rows.extend((("Số detection", str(len(result.detection.boxes))), ("Điểm detection", "Điểm của mô hình, không phải xác suất ung thư.")))
    return "\n\n".join(f"**{label}** · `{escape(str(value))}`" for label, value in rows if value is not None)


def render_result_images(files: TemporaryFiles, current: SessionState, source_file: str, result: CVResult) -> tuple:
    size = result.image_width, result.image_height
    if result.segmentation.mask is None:
        with open_source(files, current.session_id, source_file, result) as source:
            with source.resize(display_size(size), Image.Resampling.LANCZOS) as preview:
                localized = draw_combined_localization(preview, result)
                if localized is None:
                    localization = image_panel("localization", empty="Không có detection đạt ngưỡng.")
                else:
                    with localized:
                        localization = image_panel("localization", image_source(localized, png=True))
        return image_panel("mask"), image_panel("overlay"), localization_legend(result) + localization
    with open_aligned_images(files, current.session_id, source_file, result) as (source, binary):
        dimensions = display_size(size)
        with (
            source.resize(dimensions, Image.Resampling.LANCZOS) as preview,
            binary.resize(dimensions, Image.Resampling.NEAREST) as display_mask,
            overlay_layer(display_mask) as layer,
        ):
            mask = image_panel("mask", image_source(display_mask, png=True))
            overlay = image_panel("overlay", image_source(preview), layer=image_source(layer, png=True))
            if result.localization is None and not result.detection.boxes:
                return mask, overlay, image_panel("localization", empty="Mặt nạ rỗng. Không có vùng định vị.")
            box = result.localization.box if result.localization else None
            with (draw_combined_localization(preview, result) if result.detection.boxes else draw_localization(preview, box, source_size=size)) as localized:
                localization = image_panel("localization", image_source(localized, png=True))
    coordinates = (
        f'<p class="lesion-coordinates"><strong>Bounding box suy ra từ mask</strong> · '
        f'<code>[{box.x1:g}, {box.y1:g}, {box.x2:g}, {box.y2:g}]</code> px. '
        'Cạnh phải và cạnh dưới là tọa độ loại trừ.</p>'
    ) if box else ""
    return mask, overlay, coordinates + localization_legend(result) + localization


def localization_legend(result: CVResult) -> str:
    return (
        '<p><span style="color:#8a6100">SegFormer: bounding box suy ra từ mask (vàng)</span> · '
        '<span style="color:#b42332">Faster R-CNN: detection độc lập (đỏ)</span>. '
        f'{len(result.detection.boxes)} detection. Điểm detection không phải xác suất ung thư.</p>'
    )


def export_panel(current: SessionState | None = None, result: CVResult | None = None, action: int = 0) -> str:
    if current is None or result is None:
        identity = ""
        status = UNAVAILABLE
        overlay_disabled = " disabled"
    else:
        identity = f' data-session="{current.session_id.hex}" data-run="{result.run_id.hex}" data-action="{action}"'
        status = "Sẵn sàng xuất PNG." if result.localization or result.detection.boxes else "Có thể xuất Overlay. Mặt nạ rỗng nên không có ảnh định vị."
        overlay_disabled = "" if result.segmentation.mask else " disabled"
    localization_disabled = "" if result and (result.localization or result.detection.boxes) else " disabled"
    return (
        f'<div class="lesion-exports"{identity}><div class="lesion-export-actions">'
        f'<button type="button" data-kind="overlay"{overlay_disabled}>Tải Overlay PNG</button>'
        f'<button type="button" data-kind="localization"{localization_disabled}>Tải Localization PNG</button>'
        f'</div><p role="status">{status}</p></div>'
    )


def render_cv_page(
    config: AppConfig, files: TemporaryFiles, availability: CVAvailability, models: CVOrchestrator,
    exports: PNGExports | None = None,
) -> None:
    ready = any(model.status is ModelStatus.READY for model in (availability.segmentation, availability.detection))
    errors = [str(adapter.error).replace("SegFormer", label) for label, adapter in (("SegFormer", models.segmentation), ("Faster R-CNN", models.detection)) if adapter.error]
    idle_status = run_status(" ".join(errors), alert=True) if errors else run_status("Chưa có ảnh để phân tích.")

    def cleanup(state: SessionState | None) -> None:
        if state is not None:
            with state.lock:
                state.reset()
                if exports is not None:
                    exports.revoke(state.session_id)
                files.close_session(state.session_id)

    def owns(current: SessionState, request: gr.Request | None) -> bool:
        return exports is None or (current.browser_owner is not None and current.browser_owner == browser_owner(request))

    def editable(current: SessionState | None, request: gr.Request | None) -> SessionState | None:
        if current is not None:
            return current if owns(current, request) else None
        owner = browser_owner(request)
        return SessionState(browser_owner=owner) if exports is None or owner else None

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
                0, 100, DEFAULT_OPACITY, step=1, precision=0, label="Độ mờ lớp phủ (%)", interactive=False, buttons=[],
                elem_id="overlay-opacity",
            )
            export_view = gr.HTML(export_panel(), js_on_load=None, elem_id="lesion-exports")
            warnings = gr.HTML('<div role="status"></div>', js_on_load=None, elem_id="lesion-warnings")
            with gr.Accordion("Chi tiết mô hình", open=False):
                details_view = gr.Markdown("Chưa có lượt phân tích.", elem_id="lesion-model-details")

    def empty_analysis(interactive: bool, message: str | None = None, details: str = "Chưa có lượt phân tích.") -> tuple:
        return (
            gr.update(interactive=interactive), image_panel("mask"),
            message or idle_status, image_panel("localization"), image_panel("overlay"),
            gr.update(value=DEFAULT_OPACITY, interactive=False), details, '<div role="status"></div>', export_panel(),
        )

    def clear_image(current: SessionState, message: str = "") -> tuple:
        with current.lock:
            previous_result = current.result
            previous = current.reset()
            if exports is not None:
                exports.revoke(current.session_id)
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

    def accept_edit(current: SessionState, action: int) -> bool:
        with current.lock:
            if not current.accept_image_action(action):
                return False
            if exports is not None:
                exports.revoke(current.session_id)
            return True

    def upload(
        payload: dict[str, Any] | None, current: SessionState | None, action: int | dict[str, Any], request: gr.Request,
    ) -> tuple:
        current = editable(current, request)
        if not isinstance(payload, dict) or current is None:
            return tuple(gr.skip() for _ in outputs)
        proof = action.get("proof") if isinstance(action, dict) else None
        action = action.get("index") if isinstance(action, dict) else action
        if exports is not None and (
            not isinstance(proof, str) or not re.fullmatch(r"[a-f0-9]{64}", proof)
            or current.browser_proof not in (None, proof)
        ):
            return tuple(gr.skip() for _ in outputs)
        if not accept_edit(current, action):
            return tuple(gr.skip() for _ in outputs)
        try:
            raw_path = payload.get("path")
            path = uploaded_path(raw_path if isinstance(raw_path, str) else None, files.cache_dir)
            image, details = validate_image(path, config)
            with image, files.operation(current.session_id):
                with image.resize(display_size(image.size), Image.Resampling.LANCZOS) as display:
                    src = image_source(display)
                stored_file = files.write(current.session_id, lambda stream: image.save(stream, format="PNG"))
                with current.lock:
                    previous_result = current.result
                    previous = current.replace_image(details, stored_file)
                    current.browser_proof = proof
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

    def reset(current: SessionState | None, action: int, request: gr.Request) -> tuple:
        current = editable(current, request)
        if current is None or not accept_edit(current, action):
            return tuple(gr.skip() for _ in outputs)
        return clear_image(current)

    def reject_oversize(current: SessionState | None, action: int, request: gr.Request) -> tuple:
        current = editable(current, request)
        if current is None or not accept_edit(current, action):
            return tuple(gr.skip() for _ in outputs)
        return clear_image(current, str(AppError(ErrorCode.UPLOAD_TOO_LARGE, limit=config.max_upload_mib)))

    analysis_outputs = [analyze_button, mask, status, localization, overlay, opacity, details_view, warnings, export_view]
    outputs = [state, preview, metadata, error, original, tabs, *analysis_outputs]
    events = {
        "concurrency_id": "cv-session-edits", "concurrency_limit": 1, "api_visibility": "private",
        "trigger_mode": "multiple", "preprocess": False,
    }
    # Read bounded bytes ourselves and avoid UploadButton's built-in file conversion.
    upload_button.upload(
        upload, [upload_button, state, action_index], outputs,
        js="""(file, state) => {
            const [payload, action] = window.lesionImageActions?.upload(file) ?? [null, 0];
            return [payload, state, action];
        }""", **events,
    )
    current_action_js = "(state) => [state, window.lesionImageActions?.current() ?? 0]"
    reset_button.click(reset, [state, action_index], outputs, js=current_action_js, **events)
    oversize_button.click(reject_oversize, [state, action_index], outputs, js=current_action_js, **events)

    if not ready:
        return

    def begin_analysis(current: SessionState | None, action: int, request: gr.Request) -> tuple:
        if current is None or not owns(current, request):
            return tuple(gr.skip() for _ in analysis_outputs)
        with current.lock:
            if not current.matches_image_action(action) or current.image is None or current.active_run is not None:
                return tuple(gr.skip() for _ in analysis_outputs)
            previous_result = current.result
            current.begin_run()
            if exports is not None:
                exports.revoke(current.session_id)
        try:
            with files.operation(current.session_id):
                discard_result(files, current, previous_result)
        except (AppError, OSError) as exc:
            log_error(logger, exc, current.session_id)
        return empty_analysis(False, run_status("Đang phân tích ảnh bằng các mô hình sẵn sàng…"), "Đang xử lý ảnh.")

    def run_analysis(current: SessionState | None, request: gr.Request) -> None:
        if current is not None and owns(current, request):
            execute_analysis(current, files, models)

    def show_analysis(current: SessionState | None, action: int, request: gr.Request) -> tuple:
        skip = tuple(gr.skip() for _ in (*analysis_outputs, tabs))
        if current is None or not owns(current, request):
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
            if exports is not None:
                exports.issue(current, result)
        message = "Phân đoạn hoàn tất" if segmentation.mask else "Phát hiện hoàn tất"
        if result.metadata.status is RunStatus.PARTIAL:
            message += " · Thành công một phần"
        identity = segmentation.model if segmentation.mask else result.detection.model
        summary = run_status(f"{message} · {identity.model_id}", identity.checkpoint_id)
        caution = " ".join((*segmentation.warnings, *result.detection.warnings))
        for label, module in (("SegFormer", segmentation), ("Faster R-CNN", result.detection)):
            if module.error_code:
                caution += " " + str(AppError(module.error_code)).replace("SegFormer", label)
        if "empty_mask" in result.metadata.warning_codes:
            caution += " Không có pixel tiền cảnh đạt ngưỡng phân đoạn. Kết quả này không dùng để kết luận về bệnh lý."
        return (
            gr.update(interactive=True), mask_view, summary, localization_view, overlay_view,
            gr.update(value=DEFAULT_OPACITY, interactive=segmentation.mask is not None), result_details(result, message),
            f'<div role="status">{escape(caution)}</div>',
            export_panel(current, result, action) if exports is not None else export_panel(),
            gr.Tabs(selected="overlay" if segmentation.mask else "localization"),
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
