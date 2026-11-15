"""Verify real CUDA arithmetic only; do not run a segmentation model."""

import json
from pathlib import Path
import sys
import traceback


report = {"python": sys.version, "executable": sys.executable, "prefix": sys.prefix,
          "status": "FAIL"}
try:
    import torch
    import torchvision

    report.update({"torch": torch.__version__, "torchvision": torchvision.__version__,
                   "cuda_build": torch.version.cuda,
                   "cuda_available": torch.cuda.is_available()})
    print(json.dumps(report, indent=2), flush=True)
    report["gpu"] = torch.cuda.get_device_name(0)
    report["compute_capability"] = list(torch.cuda.get_device_capability(0))
    report["compiled_cuda_architectures"] = torch.cuda.get_arch_list()
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, device="cuda:0")
    y = x @ x
    torch.cuda.synchronize(0)
    actual = y.cpu().tolist()
    expected = [[7.0, 10.0], [15.0, 22.0]]
    assert y.is_cuda and actual == expected, f"Unexpected result: {actual}"
    report.update({"status": "PASS", "result_device": str(y.device),
                   "result_cpu": actual, "expected_cpu": expected,
                   "synchronized": True})
except Exception as error:
    report.update({"error_type": type(error).__name__, "error": str(error),
                   "traceback": traceback.format_exc()})
path = Path(__file__).with_name("cuda_result.json")
path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
sys.exit(0 if report["status"] == "PASS" else 1)
