"""Explicit offline export from a checksum-verified, completed training run."""

import argparse
import hashlib
import json
from pathlib import Path


def export_completed(directory: Path, destination: Path) -> str:
    import torch

    def read(name):
        path = directory / name
        receipt = json.loads(path.with_suffix(path.suffix + ".sha256").read_text())
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if path.stat().st_size != receipt["bytes"] or digest != receipt["sha256"]:
            raise ValueError("Training checkpoint checksum mismatch")
        value = torch.load(path, weights_only=True, map_location="cpu", mmap=True)
        progress = value["progress"]
        if receipt["run_id"] != value["run_id"] or receipt["position"] != [progress["epoch"], progress["next_batch"], progress["steps"]]:
            raise ValueError("Training checkpoint receipt mismatch")
        return value

    best, last = read("best.ckpt"), read("last.ckpt")
    completed = json.loads((directory / "completed.json").read_text())
    bp, lp = best["progress"], last["progress"]
    if (
        best["format"] != 1 or last["format"] != 1 or not lp["completed"] or not completed["completed"]
        or not best["run_id"] == last["run_id"] == completed["run_id"]
        or not bp["best_epoch"] == lp["best_epoch"] == completed["best_epoch"]
        or not bp["best_map"] == lp["best_map"] == completed["best_map"]
        or bp["epoch"] != lp["best_epoch"] + 1
        or completed["position"] != [lp["epoch"], lp["next_batch"], lp["steps"]]
        or best["config"] != last["config"] or best["manifest_sha256"] != last["manifest_sha256"]
    ):
        raise ValueError("Best checkpoint does not belong to the completed run")
    if destination.exists():
        raise FileExistsError("Destination exists; choose a new versioned path")
    # Same inference schema as finetune_detection/src/evaluator.py; avoid importing
    # its dataset/COCO evaluation dependencies into an offline export command.
    from app.cv.detection import PREPROCESSING, OFFICIAL_FORMAT
    config = best["config"]
    artifact = {
        "format": OFFICIAL_FORMAT, "model": best["model"],
        "labels": {0: "__background__", 1: "skin_lesion"},
        "model_config": {key: config[key] for key in ("model", "min_size", "max_size", "trainable_backbone_layers")},
        "preprocessing": PREPROCESSING, "operating_score_threshold": config["operating_score_threshold"],
        "evaluation_protocol": "COCO bbox all areas, maxDets=100, score-ranked AP; operating IoU=0.50",
        "selection": {
            "experiment": config["experiment"], "run_id": best["run_id"], "manifest_sha256": best["manifest_sha256"],
            "best_epoch": bp["best_epoch"], "best_map_50_95": bp["best_map"], "selection_metric": "validation map_50_95",
        }, "environment": best["environment"],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        torch.save(artifact, temporary)
        torch.load(temporary, weights_only=True, map_location="cpu", mmap=True)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    with destination.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Xuất best checkpoint của run hoàn tất, có checksum.")
    parser.add_argument("directory", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(json.dumps({"sha256": export_completed(args.directory, args.destination)}))


if __name__ == "__main__":
    main()
