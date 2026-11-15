"""Strict CPU load verification only. No model forward pass or inference."""

import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import traceback


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "checkpoints" / "segformer-gate2" / "real_only" / "best.pt"
report = {"status": "FAIL", "path": str(CHECKPOINT), "weights_only": True,
          "map_location": "cpu", "strict": True, "inference_run": False,
          "smp_version_source": "Trial version 0.5.0; upstream README specifies >=0.5, no manifest or lockfile pins the exact version",
          "code_revision": "d3d3195f13b02366cbc1332cbb57d7ca9234b7a7"}
try:
    import torch
    import torch.nn as nn
    import segmentation_models_pytorch as smp

    class SMPWrapper(nn.Module):
        """Same wrapper and state-dict namespace as upstream registry.py."""

        def __init__(self, model: nn.Module, name: str):
            super().__init__()
            self.model = model
            self.name = name

        def forward(self, x, provenance=None, **kw):
            return {"logits": self.model(x)}

    report["versions"] = {name: importlib.metadata.version(name) for name in
                          ("torch", "torchvision", "segmentation-models-pytorch", "timm", "numpy", "Pillow")}
    provenance = json.loads(CHECKPOINT.with_name("provenance.json").read_text(encoding="utf-8"))
    actual_sha256 = hashlib.file_digest(CHECKPOINT.open("rb"), "sha256").hexdigest()
    assert actual_sha256 == provenance["upstream_lfs_sha256"], "Checkpoint checksum mismatch"
    report["sha256"] = actual_sha256
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    report["deserialization_succeeded"] = True
    report["checkpoint_keys"] = list(checkpoint)
    cfg = checkpoint["cfg"]
    assert cfg["model"] == "segformer_mitb2" and cfg["model_overrides"] == {}
    state = checkpoint["model"]
    assert all(t.device.type == "cpu" for t in state.values())
    model = SMPWrapper(
        smp.Segformer(encoder_name="mit_b2", encoder_weights=None, in_channels=3, classes=1),
        "segformer_mitb2",
    )
    result = model.load_state_dict(state, strict=True)
    report.update({"missing_keys": result.missing_keys, "unexpected_keys": result.unexpected_keys})
    assert not result.missing_keys and not result.unexpected_keys
    actual = model.state_dict()
    assert all(torch.equal(actual[key], value) for key, value in state.items()), "Copied tensor values differ"
    model.eval()
    report.update({"status": "PASS", "tensor_values_match": True,
                   "state_dict_entries": len(state), "total_params": sum(p.numel() for p in model.parameters()),
                   "model_device": str(next(model.parameters()).device),
                   "model": cfg["model"], "encoder": "mit_b2", "encoder_weights": None,
                   "in_channels": 3, "classes": 1, "img_size": cfg["data"]["img_size"],
                   "checkpoint_epoch": checkpoint.get("epoch"), "checkpoint_which": checkpoint.get("which")})
except Exception as error:
    report.update({"error_type": type(error).__name__, "error": str(error),
                   "traceback": traceback.format_exc()})
path = Path(__file__).with_name("strict_load_result.json")
path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
sys.exit(0 if report["status"] == "PASS" else 1)
