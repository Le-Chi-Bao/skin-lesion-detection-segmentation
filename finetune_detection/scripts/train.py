"""Run a standard or uniquely named Faster R-CNN experiment on CUDA."""

import argparse
from pathlib import Path

from src.trainer import run_all_experiments, run_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=["E1", "E2", "E3", "ALL"])
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--runs-root", type=Path)
    parser.add_argument("--drive-root", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run-name")
    parser.add_argument("--resume", choices=["auto", "never"], default="auto")
    args = parser.parse_args()
    options = dict(manifest_path=args.manifest, runs_root=args.runs_root, drive_root=args.drive_root, resume=args.resume)
    if args.experiment == "ALL":
        if args.config or args.run_name:
            parser.error("ALL uses the default configurations and run directories.")
        run_all_experiments(**options)
    else:
        run_experiment(args.experiment, config_path=args.config, run_name=args.run_name, **options)


if __name__ == "__main__":
    main()
