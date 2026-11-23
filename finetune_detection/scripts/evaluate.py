"""Export a completed experiment or evaluate one locked artifact on official ISIC test data."""

import argparse
import json
from pathlib import Path

import torch

from scripts.prepare_data import prepare_dataset
from src.annotations import sha256_file, write_json
from src.checkpoint import CheckpointStore
from src.dataset import LesionDataset, load_manifest
from src.evaluator import evaluate_model, export_inference
from src.model import load_inference
from src.trainer import make_loader
from src.utils import EXPERIMENTS, WORKSPACE, environment, require_device, run_directory


def official_test(args):
    if not args.confirm_final_selection:
        raise ValueError("Confirm final validation-based model selection with --confirm-final-selection")
    result = args.output / "official_test_metrics.json"
    lock = args.output / "official_test_protocol.json"
    artifact_hash = sha256_file(args.artifact)
    training = load_manifest(args.training_manifest)
    device = require_device()
    model, artifact = load_inference(args.artifact, device)
    if sha256_file(args.artifact) != artifact_hash:
        raise ValueError("Inference artifact changed during loading.")
    if artifact["selection"]["manifest_sha256"] != training["sha256"]:
        raise ValueError("Training manifest differs from the exported artifact")
    protocol = {
        "artifact_sha256": artifact_hash, "selection": artifact["selection"],
        "evaluation_protocol": artifact["evaluation_protocol"],
        "score_threshold": artifact["operating_score_threshold"], "split": "ISIC 2018 Task 1 official test",
    }
    if lock.exists():
        if json.loads(lock.read_text(encoding="utf-8")) != protocol:
            raise ValueError(f"Different model or protocol already locked at {lock}.")
    else:
        write_json(lock, protocol)
    if result.exists():
        recorded = json.loads(result.read_text(encoding="utf-8"))
        if recorded["protocol"] != protocol:
            raise ValueError(f"Existing results differ from {lock}")
        return recorded
    test_path = prepare_dataset(args.data, args.archives, args.download, kind="test")
    test = load_manifest(test_path, expected="test")
    for key in ("id", "image_sha256"):
        training_keys = {row[key] for row in training["records"]}
        if key == "id":
            training_keys.update(row["id"] for row in training["invalid"])
        if training_keys & {row[key] for row in test["records"]}:
            raise ValueError(f"Official test and training inputs overlap by {key}")
    torch.cuda.reset_peak_memory_stats(device)
    dataset = LesionDataset(test_path, test, "test")
    metrics = evaluate_model(model, make_loader(dataset, device), device, protocol["score_threshold"], "Official test")
    payload = {
        "protocol": protocol, "test_manifest_sha256": test["sha256"], "images": len(dataset),
        "metrics": metrics, "peak_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20,
        "environment": environment(device),
    }
    write_json(result, payload)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    export.add_argument("experiment", choices=EXPERIMENTS)
    export.add_argument("--runs-root", type=Path, default=WORKSPACE / "runs")
    export.add_argument("--drive-root", type=Path)
    export.add_argument("--run-name")
    export.add_argument("--output", type=Path, required=True)
    test = commands.add_parser("official-test")
    test.add_argument("--artifact", type=Path, required=True)
    test.add_argument("--training-manifest", type=Path, required=True)
    test.add_argument("--data", type=Path, default=WORKSPACE / "data" / "official-test")
    test.add_argument("--archives", type=Path)
    test.add_argument("--download", action="store_true")
    test.add_argument("--output", type=Path, required=True, help="A persistent directory shared by any official test invocation")
    test.add_argument("--confirm-final-selection", action="store_true")
    args = parser.parse_args()
    if args.command == "official-test":
        print(json.dumps(official_test(args), indent=2))
    else:
        local = run_directory(args.runs_root, args.experiment, args.run_name)
        drive = run_directory(args.drive_root, args.experiment, args.run_name) if args.drive_root else None
        store = CheckpointStore(local, drive)
        latest = store.latest()
        if latest is None or not latest["progress"]["completed"]:
            raise ValueError("Finish training and validation before exporting the selected model.")
        export_inference(store.best(latest), latest, args.output)


if __name__ == "__main__":
    main()
