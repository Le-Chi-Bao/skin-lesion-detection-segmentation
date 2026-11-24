import gradio as gr


def clinical_theme() -> gr.themes.Soft:
    return gr.themes.Soft(
        primary_hue="teal",
        secondary_hue="slate",
        neutral_hue="slate",
        font=("system-ui", "Segoe UI", "sans-serif"),
        font_mono=("ui-monospace", "Consolas", "monospace"),
    ).set(
        body_background_fill="#f6f9fb",
        body_text_color="#17324d",
        block_background_fill="#ffffff",
        block_border_color="#d8e3e9",
        block_title_text_color="#17324d",
        color_accent_soft="#dcefed",
        button_primary_background_fill="#176962",
        button_primary_background_fill_hover="#12534e",
        button_primary_text_color="#ffffff",
    )


CSS = """
/* Keep Gradio event inputs mounted without showing internal controls. */
#image-action-index, #oversize-image { display: none; }

/* Keep the full image in view, including portrait and panorama uploads. */
.lesion-view {
    display: grid;
    place-items: center;
    padding: 0.75rem;
    border: 1px solid var(--block-border-color);
    border-radius: var(--block-radius);
    background: var(--background-fill-secondary);
    color: var(--body-text-color);
}
.lesion-preview { min-height: 8rem; }
.lesion-original { min-height: clamp(20rem, 58vh, 42rem); }
.lesion-view img { display: block; max-width: 100%; object-fit: contain; }
.lesion-preview img { max-height: 10rem; }
.lesion-original img { max-height: min(70vh, 52rem); }
"""
