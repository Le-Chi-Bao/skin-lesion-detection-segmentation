"""Configuration, reproducibility and run logging."""

import csv
import importlib.metadata
import os
import platform
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm.auto import tqdm

from .annotations import write_json

WORKSPACE = Path(__file__).resolve().parents[1]
EXPERIMENTS = ("E1", "E2", "E3")


def load_config(experiment, path=None):
    if experiment not in EXPERIMENTS:
        raise ValueError(f"Experiment must be one of {EXPERIMENTS}")
    with Path(path or WORKSPACE / "configs" / f"{experiment}.yaml").open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if config["experiment"] != experiment or config["optimizer"] != "sgd":
        raise ValueError("Configuration experiment and optimizer must match the run")
    for key in ("min_size", "max_size", "batch_size", "gradient_accumulation_steps", "epochs",
                "warmup_steps", "early_stopping_min_epochs", "early_stopping_patience", "checkpoint_every_steps"):
        if type(config[key]) is not int or config[key] < (0 if key == "warmup_steps" else 1):
            raise ValueError(f"Invalid configuration value for {key}: {config[key]}")
    if not 0 <= config["trainable_backbone_layers"] <= 5 or config["min_size"] > config["max_size"]:
        raise ValueError("Invalid model resize or trainable backbone layers")
    for key in ("operating_score_threshold", "warmup_start_factor", "min_lr_factor"):
        if not 0 <= config[key] <= 1:
            raise ValueError(f"{key} must be in [0, 1]")
    return config


def run_directory(root, experiment, name=None):
    if name is not None and (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name)
                             or name in {"checkpoints", "logs"}):
        raise ValueError("Run name must be a distinct directory name with letters, numbers, dots, underscores or hyphens")
    directory = Path(root) / experiment
    return directory / name if name else directory


def require_device(cpu=False):
    try:
        from torchvision.ops import nms
    except (ImportError, OSError, RuntimeError) as error:
        raise RuntimeError("Torchvision failed to import. Install a matching torch/torchvision wheel pair and restart.") from error
    if not cpu and not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required. In Colab select Runtime > Change runtime type > GPU and rerun.")
    device = torch.device("cpu" if cpu else "cuda:0")
    try:
        nms(torch.zeros((1, 4), device=device), torch.ones(1, device=device), 0.5)
        import pycocotools.mask  # Verify the backend and its compiled extension.
    except (ImportError, OSError, RuntimeError) as error:
        raise RuntimeError(f"Missing pycocotools or incompatible Torchvision operators on {device}. Run the dependency cell.") from error
    return device


def environment(device):
    packages = ("torch", "torchvision", "numpy", "Pillow", "PyYAML", "tqdm", "matplotlib", "pycocotools")
    versions = {name: importlib.metadata.version(name) for name in packages}
    return {
        "python": platform.python_version(), "platform": platform.platform(), "packages": versions,
        "cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
        "cudnn": torch.backends.cudnn.version(), "device_type": device.type,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "gpu_memory_bytes": torch.cuda.get_device_properties(device).total_memory if device.type == "cuda" else None,
    }


def seed_training(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def duration(seconds):
    hours, rest = divmod(int(seconds), 3600)
    minutes, seconds = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class RunLogs:
    def __init__(self, directory):
        self.directory = Path(directory) / "logs"
        self.directory.mkdir(parents=True, exist_ok=True)

    def message(self, message, display=True):
        if display:
            tqdm.write(message)
        with (self.directory / "training.log").open("a", encoding="utf-8") as stream:
            stream.write(f"{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())} {message}\n")

    def initialize(self, config, settings):
        (self.directory / "run_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        write_json(self.directory / "environment.json", settings)

    def history(self, rows):
        if not rows:
            return
        write_json(self.directory / "history.json", rows)
        path = self.directory / "metrics.csv"
        temporary = path.with_suffix(".part")
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)

        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        figure = Figure(figsize=(11, 7), layout="constrained")
        FigureCanvasAgg(figure)
        axes = figure.subplots(2, 2).flat
        for axis, columns, title in zip(axes, (
            ("train_loss", "loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"),
            ("map_50", "map_50_95", "recall_50"), ("learning_rate",), ("peak_memory_mib",),
        ), ("Training losses", "Validation metrics", "Learning rate", "Peak allocated VRAM (MiB)")):
            for column in columns:
                if any(row[column] is not None for row in rows):
                    axis.plot([row["epoch"] for row in rows], [row[column] for row in rows], label=column)
            if axis.lines:
                axis.legend()
            axis.set(title=title, xlabel="Epoch")
        path = self.directory / "training_curves.png"
        temporary = path.with_suffix(".part")
        figure.savefig(temporary, format="png", dpi=130)
        os.replace(temporary, path)
