"""Explicit device selection; importing this module does not allocate a model."""

import torch


def get_device(preference: str = "auto") -> torch.device:
    """Prefer CUDA in auto mode; reject an unavailable explicit CUDA request."""
    if preference not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu or cuda")
    if preference == "cpu":
        return torch.device("cpu")
    available = torch.cuda.is_available()
    if preference == "cuda" and not available:
        raise RuntimeError("CUDA was requested but is unavailable; use cpu or auto")
    return torch.device("cuda" if available else "cpu")
