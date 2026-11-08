"""Run offline foundation checks: python -m scripts.smoke_test [--skip-app]."""

import argparse
import importlib
import json

import numpy as np
import torch

from src.data.geometry import crop_mask, mask_to_bbox, restore_roi_mask
from src.pipeline.spatial import prepare_roi
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.device import get_device


def run_checks(config_path: str, include_app: bool = True) -> dict[str, object]:
    """Raise on failure; require neither ISIC files nor downloaded weights."""
    modules = (
        "src.data.image_io",
        "src.data.geometry",
        "src.models.detection",
        "src.models.segmentation",
        "src.pipeline.spatial",
        "src.evaluation",
        "app.streamlit_app",
        "torchvision",
    )
    for module in modules:
        importlib.import_module(module)
    config = load_config(config_path)
    image = np.zeros((12, 20, 3), dtype=np.uint8)
    mask = np.zeros((12, 20), dtype=bool)
    mask[3:9, 5:15] = True
    box = mask_to_bbox(mask)
    if box is None:
        raise AssertionError("Expected a box for the synthetic lesion")
    roi = prepare_roi(image, box, config.image_size, config.roi_padding)
    cropped_mask = crop_mask(mask, roi.metadata)
    restored = restore_roi_mask(cropped_mask, roi.metadata)
    if not np.array_equal(restored, mask):
        raise AssertionError("mask → box → crop → restore changed coordinates")
    if roi.image.shape != (*config.image_size, 3):
        raise AssertionError("ROI size does not match the configuration")
    device = get_device(config.device)
    if torch.zeros(1, device=device).device.type != device.type:
        raise AssertionError("Tensor allocation failed on the selected device")
    if include_app:
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file(str(PROJECT_ROOT / "app/streamlit_app.py"))
        app.run(timeout=30)
        if app.exception:
            raise AssertionError(f"Streamlit failed: {app.exception}")
        # The Streamlit entrypoint intentionally uses base.yaml, independently
        # of the experiment config selected for these spatial checks.
        if not app.title or app.title[0].value != load_config().project_name:
            raise AssertionError("Streamlit did not render the project title")
    return {
        "imports": "passed",
        "configuration": config_path,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "coordinate_round_trip": "passed (synthetic ground truth, no AI inference)",
        "streamlit_app": "passed (AppTest)" if include_app else "skipped explicitly",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--skip-app", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_checks(args.config, not args.skip_app), indent=2))


if __name__ == "__main__":
    main()
