import random

import nbformat
import numpy as np
import pytest
import torch

from src.checkpoint import CheckpointStore
from src.metrics import DetectionMetrics
from src.evaluator import export_inference
from src.model import build_model, load_inference
from src.progress import LOSS_NAMES, training_metrics
from src.trainer import TrainingRun
from src.utils import RunLogs, WORKSPACE, load_config, require_device


class SyntheticDataset:
    def __init__(self, training):
        self.training = training

    def __len__(self):
        return 5 if self.training else 2

    def __getitem__(self, index):
        intensity = (index + 1) / 10
        if self.training:
            intensity += random.random() / 10 + np.random.random() / 10
        image = torch.full((3, 8, 8), intensity, dtype=torch.float32)
        return image, {"boxes": torch.tensor([[1., 1., 4., 4.]]), "labels": torch.tensor([1]),
                       "image_id": torch.tensor(index)}


class TinyDetector(torch.nn.Module):
    def __init__(self, config, pretrained):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.rand(()))

    def forward(self, images, targets=None):
        if self.training:
            factor = torch.stack([image.mean() for image in images]).mean() + torch.rand((), device=self.weight.device)
            return {name: (self.weight * factor - 0.25).square() for name in LOSS_NAMES[1:]}
        return [
            {"boxes": self.weight.new_tensor([[1., 1., 4., 4.]]), "scores": self.weight.new_tensor([0.9]),
             "labels": torch.ones(1, device=self.weight.device, dtype=torch.int64)}
            for _ in images
        ]


class InterruptingStore(CheckpointStore):
    def save(self, payload, improved=False, snapshot_step=False, report=print):
        result = super().save(payload, improved, snapshot_step, report)
        if snapshot_step and payload["progress"]["epoch"] == 2:
            raise KeyboardInterrupt("simulated Colab disconnect after saved optimizer step")
        return result


def build_run(config, store, factory=TinyDetector, dataset=None, device=None):
    return TrainingRun(config, dataset or SyntheticDataset(True), dataset or SyntheticDataset(False), "synthetic",
                       device or torch.device("cpu"), store, RunLogs(store.local), model_factory=factory, pretrained=False)


@pytest.mark.parametrize("device_name", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA Torch runtime unavailable"
))])
def test_mid_epoch_resume_matches_uninterrupted_training(tmp_path, device_name):
    config = {
        **load_config("E1"), "epochs": 2, "batch_size": 2, "gradient_accumulation_steps": 2,
        "checkpoint_every_steps": 1, "warmup_steps": 1,
    }
    device = torch.device(device_name)
    reference = CheckpointStore(tmp_path / "reference")
    complete = build_run(config, reference, device=device).run()
    assert complete["completed"] and complete["best"]["map_50_95"] == pytest.approx(1.0)
    interrupted = InterruptingStore(tmp_path / "first", tmp_path / "drive")
    with pytest.raises(KeyboardInterrupt):
        build_run(config, interrupted, device=device).run()
    restored = CheckpointStore(tmp_path / "fresh", tmp_path / "drive")
    resumed = build_run(config, restored, device=device)
    assert resumed.state["epoch"] == 2 and resumed.state["next_batch"] == 2
    assert resumed.run()["completed"]
    expected, actual = reference.latest(), restored.latest()
    assert expected["progress"]["steps"] == actual["progress"]["steps"] == 4
    assert expected["scheduler"] == actual["scheduler"]
    assert torch.equal(expected["model"]["weight"], actual["model"]["weight"])
    assert torch.equal(expected["rng"]["torch"], actual["rng"]["torch"])
    assert expected["rng"]["python"] == actual["rng"]["python"]
    assert expected["rng"]["numpy"] == actual["rng"]["numpy"]
    assert torch.equal(expected["optimizer"]["state"][0]["momentum_buffer"],
                       actual["optimizer"]["state"][0]["momentum_buffer"])
    logs = restored.local / "logs"
    assert all((logs / filename).is_file() for filename in
               ("metrics.csv", "history.json", "environment.json", "training.log", "run_config.yaml", "training_curves.png"))
    assert expected["scaler"] == actual["scaler"]
    assert all(torch.equal(left, right) for left, right in zip(expected["rng"]["cuda"], actual["rng"]["cuda"], strict=True))
    assert build_run(config, restored, device=device).run()["completed"]


def test_validation_comparison_and_inference_export(tmp_path, monkeypatch):
    from scripts.compare_experiments import compare, render_report
    from src import model

    config = {**load_config("E1"), "epochs": 1}
    store = CheckpointStore(tmp_path / "E1")
    results = build_run(config, store).run()
    latest = store.latest()
    artifact_path = export_inference(store.best(latest), latest, tmp_path / "export.pt")
    monkeypatch.setattr(model, "build_model", TinyDetector)
    restored, artifact = load_inference(artifact_path)
    assert torch.equal(restored.weight, latest["model"]["weight"])
    assert artifact["model_config"]["min_size"] == config["min_size"]
    assert artifact["preprocessing"]["range"] == [0, 1]
    assert artifact["labels"][1] == "skin_lesion"
    assert artifact["selection"]["run_id"] == results["run_id"]
    rows, candidate = compare(tmp_path)
    assert candidate == "E1" and len(rows) == 1
    assert "| E1 |" in render_report(rows, candidate)


@pytest.mark.parametrize("device_name", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA Torch runtime unavailable"
))])
def test_real_model_one_optimizer_update_and_inference(device_name):
    device = require_device(cpu=device_name == "cpu")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        config = {**load_config("E1"), "min_size": 64, "max_size": 96}
        model = build_model(config, pretrained=False).to(device)
        head = model.roi_heads.box_predictor
        assert head.cls_score.out_features == 2 and head.bbox_pred.out_features == 8
        assert model.backbone.body.layer2[0].conv1.weight.requires_grad
        assert not model.backbone.body.layer1[0].conv1.weight.requires_grad
        image = torch.rand(3, 64, 88, device=device)
        target = {"boxes": torch.tensor([[10., 11., 60., 50.]], device=device),
                  "labels": torch.tensor([1], dtype=torch.int64, device=device)}
        optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=0.0025)
        amp = device_name == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=amp, init_scale=32)
        with torch.amp.autocast(device_type=device_name, dtype=torch.float16, enabled=amp):
            losses = model([image], [target])
        assert set(losses) == set(LOSS_NAMES[1:])
        assert all(torch.isfinite(value).item() for value in losses.values())
        initial = head.cls_score.weight.detach().clone()
        scaler.scale(sum(losses.values())).backward()
        scaler.step(optimizer)
        scaler.update()
        assert not torch.equal(initial, head.cls_score.weight.detach())
        model.eval()
        with torch.inference_mode():
            outputs = model([image])
        assert len(outputs) == 1 and outputs[0]["boxes"].shape[1] == 4
        assert outputs[0]["labels"].dtype == torch.int64 and set(outputs[0]["labels"].tolist()) <= {1}
    finally:
        torch.set_num_threads(previous_threads)


def test_progress_reports_real_components_and_missing_values():
    losses = dict(zip(LOSS_NAMES, [1.2, 0.1, 0.2, 0.4, 0.5]))
    line = training_metrics(losses, 0.0025, None, 3.1)
    assert "Loss 1.2000" in line and "Cls 0.1000" in line and "RPN Box 0.5000" in line
    assert "LR 0.0025" in line and "VRAM N/A" in line and "Speed 3.10 img/s" in line


def test_metric_operating_point_and_empty_results():
    metrics = DetectionMetrics(threshold=0.5)
    gt = {"image_id": torch.tensor(1), "boxes": torch.tensor([[2., 2., 6., 6.]]), "labels": torch.tensor([1])}
    metrics.update({"boxes": torch.tensor([[2., 2., 6., 6.], [2., 2., 6., 6.], [0., 0., 8.000001, 8.]]),
                    "labels": torch.tensor([1, 1, 1]), "scores": torch.tensor([0.9, 0.8, 0.4])}, gt, 8, 8)
    empty = {"boxes": torch.empty(0, 4), "labels": torch.empty(0, dtype=torch.int64), "scores": torch.empty(0)}
    metrics.update(empty, {**gt, "image_id": torch.tensor(2)}, 8, 8)
    result = metrics.compute()
    assert result["recall_50"] == 0.5 and result["mean_matched_iou"] == 1.0 and result["fp_per_image"] == 0.5
    assert result["map_50"] == pytest.approx(51 / 101) and result["map_50_95"] == pytest.approx(51 / 101)
    only_empty = DetectionMetrics(threshold=0.5)
    only_empty.update(empty, gt, 8, 8)
    assert only_empty.compute() == {"map_50": 0, "map_50_95": 0, "recall_50": 0, "mean_matched_iou": None, "fp_per_image": 0}


def test_notebook_has_executable_cells():
    notebook = nbformat.read(WORKSPACE / "notebooks/Faster_RCNN_ISIC2018_Colab.ipynb", as_version=4)
    nbformat.validate(notebook)
    cells = [cell.source for cell in notebook.cells if cell.cell_type == "code"]
    assert len(cells) == 11
    for index, cell in enumerate(cells, 1):
        compile(cell, f"Colab cell {index}", "exec")
    for cell, marker in zip(cells, (
        "mem", "INSTALL_FALLBACK", "drive.mount", "ZipFile", "prepare_dataset", "plt.hist",
        "EXPERIMENT", "run_experiment", "preview_validation", "render_report", "export_inference",
    )):
        assert marker in cell
