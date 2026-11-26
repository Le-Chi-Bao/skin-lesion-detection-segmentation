"""Read-only verification and sequential CPU smoke tests of local E1/E2/E3 exports."""

import argparse
import gc
import hashlib
import json
import math
import re
from pathlib import Path

from PIL import Image

from app.config import PROJECT_ROOT
from app.cv.artifacts import RegistryEntry
from app.cv.availability import ModelStatus
from app.cv.detection import FasterRCNN, predictions, validate_artifact
from app.cv.registry import ModelRegistry

EXPECTED = {"E1": (800, 1333, 9), "E2": (800, 1333, 6), "E3": (1024, 1600, 6)}


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_selection(artifact, experiment: str, run_id: str, best_epoch: int) -> None:
    selection = artifact.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("Missing selection provenance")
    if (
        selection.get("experiment") != experiment
        or not re.fullmatch(r"[0-9a-f]{32}", run_id)
        or selection.get("run_id") != run_id
        or type(selection.get("best_epoch")) is not int
        or selection["best_epoch"] != best_epoch
        or not isinstance(selection.get("manifest_sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", selection["manifest_sha256"])
        or selection.get("selection_metric") != "validation map_50_95"
    ):
        raise ValueError("Selection provenance mismatch")
    metric = selection.get("best_map_50_95")
    if type(metric) not in (float, int) or not math.isfinite(metric) or not 0 <= metric <= 1:
        raise ValueError("Invalid validation selection metric")


def verify_checkpoint(entry: RegistryEntry, experiment: str, image_path: Path | None = None) -> dict:
    """Never write weights or registry; fail closed on metadata, state or output errors."""
    import torch

    minimum, maximum, epoch = EXPECTED[experiment]
    prefix = experiment + "-"
    if not entry.path.stem.startswith(prefix):
        raise ValueError("Artifact filename does not identify the expected experiment")
    run_id = entry.path.stem[len(prefix):]
    before = sha256_file(entry.path)
    stat = entry.path.stat()
    detector = None
    try:
        if before != entry.sha256:
            raise ValueError("Registry checksum mismatch")
        if (entry.min_size, entry.max_size) != (minimum, maximum):
            raise ValueError("Experiment resize mismatch")
        artifact = torch.load(entry.path, weights_only=True, map_location="cpu")
        validate_artifact(artifact, entry, pilot_mode=False)
        validate_selection(artifact, experiment, run_id, epoch)
        for tensor in artifact["model"].values():
            if not isinstance(tensor, torch.Tensor) or not torch.isfinite(tensor).all().item():
                raise ValueError("Non-tensor or non-finite state_dict value")
        report = {
            "experiment": experiment, "version": entry.version, "path": entry.checkpoint_path,
            "sha256": before, "bytes": stat.st_size, "format": artifact["format"],
            "labels": artifact["labels"], "model_config": artifact["model_config"],
            "preprocessing": artifact["preprocessing"],
            "operating_score_threshold": artifact["operating_score_threshold"],
            "selection": artifact["selection"], "evaluation_protocol": artifact.get("evaluation_protocol"),
            "state_dict_tensors": len(artifact["model"]),
        }
        del artifact
        detector = FasterRCNN(entry, device="cpu")
        detector.load()  # Includes strict state_dict loading; no pretrained downloads.
        if detector.availability.status is not ModelStatus.READY:
            raise RuntimeError(f"Strict model load failed: {detector.error}")
        model = detector._model
        if model.training or model.transform.min_size != (minimum,) or model.transform.max_size != maximum:
            raise ValueError("Runtime eval/resize configuration mismatch")
        report.update(status="VERIFIED", device=detector.identity.device, strict_state_dict=True)
        if image_path is not None:
            raw = []
            hook = model.register_forward_hook(lambda module, args, output: raw.append(output))
            try:
                with Image.open(image_path) as source, source.convert("RGB") as image:
                    output = detector.predict(image)
                    if len(raw) != 1 or not isinstance(raw[0], list) or len(raw[0]) != 1:
                        raise ValueError("Invalid raw inference structure")
                    all_boxes = predictions(raw[0][0], image.size, 0.0)
                    if output.boxes != tuple(box for box in all_boxes if box.score >= entry.threshold):
                        raise ValueError("Confidence filtering mismatch")
                    report["inference"] = {
                        "status": "PASS", "image_size": list(image.size),
                        "image_sha256": sha256_file(image_path), "raw_predictions": len(all_boxes),
                        "raw_score_range": [min(box.score for box in all_boxes), max(box.score for box in all_boxes)] if all_boxes else [],
                        "duration_ms": output.duration_ms,
                        "boxes": [box.model_dump(mode="json") for box in output.boxes],
                        "labels": raw[0][0]["labels"].detach().cpu().tolist(),
                        "finite_in_bounds_threshold_checked": True,
                    }
            finally:
                hook.remove()
        return report
    finally:
        after = sha256_file(entry.path)
        current = entry.path.stat()
        if after != before or current.st_size != stat.st_size or current.st_mtime_ns != stat.st_mtime_ns:
            raise RuntimeError("Checkpoint changed during verification")
        if detector is not None:
            detector._model = None
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=PROJECT_ROOT / "config/models.json")
    parser.add_argument("--experiment", choices=tuple(EXPECTED), action="append")
    parser.add_argument("--image", type=Path, help="Existing permitted smoke image; never evaluates a dataset")
    parser.add_argument("--output", type=Path, help="Optional new JSON report path (never overwritten)")
    args = parser.parse_args()
    registry = ModelRegistry(args.registry)
    if registry.errors:
        parser.error("Invalid registry")
    reports = []
    for experiment in args.experiment or EXPECTED:
        matches = [entry for entry in registry.entries if entry.module == "detection"
                   and not entry.research_pilot and entry.path.stem.startswith(experiment + "-")]
        if len(matches) != 1:
            parser.error(f"Expected exactly one registered {experiment} export")
        try:
            report = verify_checkpoint(matches[0], experiment, args.image)
            report["unchanged_after_verification"] = True
        except Exception as exc:
            report = {"experiment": experiment, "status": "BLOCKED", "reason": str(exc)}
        reports.append(report)
        print(json.dumps(report, ensure_ascii=True), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(reports, stream, indent=2)
    raise SystemExit(0 if all(report["status"] == "VERIFIED" for report in reports) else 1)


if __name__ == "__main__":
    main()
