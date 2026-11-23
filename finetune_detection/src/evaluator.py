"""Validation, inference previews and export from a verified best checkpoint."""

import os
import time
from pathlib import Path

import torch
from tqdm.auto import tqdm

from .annotations import sha256_file
from .dataset import LesionDataset, load_manifest
from .metrics import DetectionMetrics, xywh
from .model import LABELS, build_model
from .progress import validation_bar


@torch.inference_mode()
def evaluate_model(model, loader, device, threshold, description="Validation"):
    model.eval()
    metrics = DetectionMetrics(threshold)
    inference_s = 0.0
    cuda = device.type == "cuda"
    with validation_bar(len(loader.dataset), description) as bar:
        for images, targets in loader:
            inputs = [image.to(device, non_blocking=True) for image in images]
            if cuda:
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            predictions = model(inputs)
            if cuda:
                torch.cuda.synchronize(device)
            inference_s += time.perf_counter() - start
            for prediction, target, image in zip(predictions, targets, images, strict=True):
                height, width = image.shape[-2:]
                metrics.update({key: value.cpu() for key, value in prediction.items()}, target, width, height)
            bar.update(len(images))
    return {**metrics.compute(), "inference_ms_per_image": 1000 * inference_s / len(loader.dataset)}


@torch.inference_mode()
def preview_validation(checkpoint, manifest_path, device, count=4):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    manifest = load_manifest(manifest_path)
    if checkpoint["manifest_sha256"] != manifest["sha256"]:
        raise ValueError("Validation manifest differs from the checkpoint")
    config = checkpoint["config"]
    dataset = LesionDataset(manifest_path, manifest, "validation")
    model = build_model(config, pretrained=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    count = min(count, len(dataset))
    figure, axes = plt.subplots(1, count, figsize=(4 * count, 4), squeeze=False)
    for index, axis in enumerate(axes[0]):
        sample = index * len(dataset) // count
        image, target = dataset[sample]
        prediction = model([image.to(device)])[0]
        axis.imshow(image.permute(1, 2, 0))
        for box in target["boxes"].tolist():
            x, y, width, height = xywh(box)
            axis.add_patch(Rectangle((x - 0.5, y - 0.5), width, height, fill=False, edgecolor="lime", linewidth=2))
        for box, score in zip(prediction["boxes"].tolist(), prediction["scores"].tolist()):
            if score >= config["operating_score_threshold"]:
                x, y, width, height = xywh(box)
                axis.add_patch(Rectangle((x - 0.5, y - 0.5), width, height, fill=False, edgecolor="orange", linewidth=2))
                axis.text(x, y, f"{score:.2f}", color="orange", bbox={"facecolor": "black", "alpha": 0.6})
        axis.set_title(dataset.records[sample]["id"])
        axis.axis("off")
    figure.legend(handles=[Patch(color="lime", label="Ground truth"), Patch(color="orange", label="Dự đoán")],
                  loc="lower center", ncol=2)
    plt.show()
    return figure


def export_inference(best, completed, destination):
    if not completed["progress"]["completed"] or (
        best["run_id"], best["progress"]["best_epoch"], best["progress"]["best_map"]
    ) != (
        completed["run_id"], completed["progress"]["best_epoch"], completed["progress"]["best_map"]
    ) or best["progress"]["epoch"] != completed["progress"]["best_epoch"] + 1:
        raise ValueError("Export requires the verified best checkpoint of a completed run")
    config = best["config"]
    artifact = {
        "format": "isic-fasterrcnn-inference-v1",
        "model": best["model"], "labels": LABELS,
        "model_config": {key: config[key] for key in ("model", "min_size", "max_size", "trainable_backbone_layers")},
        "preprocessing": {
            "input": "RGB float32 CHW", "range": [0, 1], "resize": "torchvision GeneralizedRCNNTransform",
            "image_mean": [0.485, 0.456, 0.406], "image_std": [0.229, 0.224, 0.225],
            "box_format": "xyxy, exclusive right/bottom, in source-image pixels",
            "box_score_thresh": 0.0, "nms_threshold": 0.5, "detections_per_image": 100,
        },
        "operating_score_threshold": config["operating_score_threshold"],
        "evaluation_protocol": "COCO bbox all areas, maxDets=100, score-ranked AP; operating IoU=0.50",
        "selection": {
            "experiment": config["experiment"], "run_id": best["run_id"],
            "manifest_sha256": best["manifest_sha256"], "best_epoch": best["progress"]["best_epoch"],
            "best_map_50_95": best["progress"]["best_map"], "selection_metric": "validation map_50_95",
        },
        "environment": best["environment"],
    }
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        torch.save(artifact, temporary)
        torch.load(temporary, weights_only=True, map_location="cpu", mmap=True)
        digest = sha256_file(temporary)
        os.replace(temporary, destination)
        if sha256_file(destination) != digest:
            raise OSError(f"Export verification failed: {destination}")
    finally:
        temporary.unlink(missing_ok=True)
    tqdm.write(f"Exported inference weights to {destination}")
    return destination
