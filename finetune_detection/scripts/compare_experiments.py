"""Compare verified best validation checkpoints across the three default runs."""

import argparse
from pathlib import Path

from src.checkpoint import CheckpointStore
from src.trainer import result_summary
from src.utils import EXPERIMENTS, WORKSPACE, run_directory


def compare(runs_root=None, drive_root=None):
    rows = []
    manifest_hashes, thresholds = set(), set()
    for experiment in EXPERIMENTS:
        local = run_directory(runs_root or WORKSPACE / "runs", experiment)
        drive = run_directory(drive_root, experiment) if drive_root else None
        store = CheckpointStore(local, drive)
        latest = store.latest()
        if latest is None or latest["progress"]["best_epoch"] is None:
            continue
        store.best(latest)
        manifest_hashes.add(latest["manifest_sha256"])
        thresholds.add(latest["config"]["operating_score_threshold"])
        rows.append(result_summary(latest["progress"], latest))
    if len(manifest_hashes) > 1 or len(thresholds) > 1:
        raise ValueError("Comparison requires a shared validation manifest and operating threshold.")
    ranked = sorted((row for row in rows if row["completed"]), key=lambda row: -row["best"]["map_50_95"])
    return rows, ranked[0]["experiment"] if ranked else None


def render_report(rows, candidate):
    if not rows:
        return "Chưa có kết quả validation."
    report = [
        "| Experiment | Trạng thái | Best Epoch | mAP@50 | mAP@50:95 | Recall | Mean IoU | FP/Image | ms/image | Peak MiB |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        best = row["best"]
        values = (best[key] for key in (
            "map_50", "map_50_95", "recall_50", "mean_matched_iou", "fp_per_image",
            "inference_ms_per_image", "peak_memory_mib",
        ))
        metrics = " | ".join(f"{value:.4f}" if value is not None else "N/A" for value in values)
        status = "Hoàn tất" if row["completed"] else "Chưa hoàn tất"
        report.append(f"| {row['experiment']} | {status} | {best['epoch']} | {metrics} |")
    if candidate:
        report.extend(["", f"**Ứng viên đứng đầu theo validation mAP@50:95** {candidate}. So sánh Recall và FP/Image trước khi chọn mô hình triển khai."])
    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=WORKSPACE / "runs")
    parser.add_argument("--drive-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows, candidate = compare(args.runs_root, args.drive_root)
    report = render_report(rows, candidate)
    print(report)
    if rows:
        path = args.output or (args.drive_root or args.runs_root) / "comparison.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
