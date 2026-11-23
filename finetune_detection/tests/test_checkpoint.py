import random

import numpy as np
import pytest
import torch

from src import checkpoint
from src.checkpoint import CheckpointStore, capture_rng, restore_rng, verify_compatibility
from src.utils import environment, load_config


def payload(epoch=1, next_batch=0, step=0):
    history = [{"epoch": index, "map_50_95": 0.5} for index in range(1, epoch)]
    return {
        "config": load_config("E1"), "manifest_sha256": "example", "environment": environment(torch.device("cpu")),
        "run_id": "test", "progress": {
            "epoch": epoch, "next_batch": next_batch, "steps": step, "best_epoch": 1 if history else None,
            "best_map": 0.5 if history else None, "completed": False, "history": history,
        },
        "model": {"example": torch.arange(4)}, "optimizer": {}, "scheduler": {"last_epoch": step},
        "scaler": {}, "amp_enabled": False, "rng": capture_rng(),
    }


def test_rng_restoration():
    state = capture_rng()
    expected = (random.random(), np.random.random(), torch.rand(1).item())
    restore_rng(state)
    assert (random.random(), np.random.random(), torch.rand(1).item()) == expected


def test_round_trip_and_previous_recovery(tmp_path):
    store = CheckpointStore(tmp_path / "local", tmp_path / "drive")
    first = payload(2, 0, 1)
    assert "Drive backup verified" in store.save(first, improved=True)
    assert "optimizer step 2" in store.save(payload(2, 2, 2), snapshot_step=True)
    loaded = store.latest()
    assert loaded["progress"]["steps"] == 2
    assert torch.equal(loaded["model"]["example"], torch.arange(4))
    assert store.best(loaded)["progress"]["epoch"] == 2
    verify_compatibility(loaded, first)
    with pytest.raises(ValueError, match="manifest_sha256"):
        verify_compatibility(loaded, {**first, "manifest_sha256": "different"})
    del loaded
    for root in store.directories():
        for path in (root / "last.ckpt", *root.glob("checkpoints/step-*.ckpt")):
            path.write_bytes(b"corrupt")
    recovered = store.latest()
    assert recovered["progress"]["steps"] == 1


def test_drive_promotion_failure_preserves_previous_and_local(tmp_path, monkeypatch):
    store = CheckpointStore(tmp_path / "local", tmp_path / "drive")
    store.save(payload(2, 0, 1), improved=True)
    replace = checkpoint.os.replace

    def fail_drive_seal(source, target):
        if target == store.drive / "last.ckpt.sha256":
            raise OSError("simulated interruption at Drive promotion")
        return replace(source, target)

    monkeypatch.setattr(checkpoint.os, "replace", fail_drive_seal)
    assert "Drive backup FAILED" in store.save(payload(2, 2, 2))
    recovered = CheckpointStore(tmp_path / "fresh", store.drive).latest()
    assert recovered["progress"]["steps"] == 1
    assert store.latest()["progress"]["steps"] == 2


def test_corrupt_run_never_starts_fresh(tmp_path):
    store = CheckpointStore(tmp_path)
    (tmp_path / "last.ckpt").write_bytes(b"invalid")
    with pytest.raises(ValueError, match="No verified"):
        store.latest()


def test_independent_run_ids_do_not_overwrite_each_other(tmp_path):
    local, drive = tmp_path / "local", tmp_path / "drive"
    CheckpointStore(local).save(payload())
    CheckpointStore(drive).save({**payload(), "run_id": "different"})
    with pytest.raises(ValueError, match="Conflicting run IDs"):
        CheckpointStore(local, drive).latest()


def test_completed_checkpoints_are_protected(tmp_path):
    store = CheckpointStore(tmp_path)
    store.save(payload(2, 0, 1), improved=True)
    completed = payload(3, 0, 2)
    completed["progress"]["completed"] = True
    store.save(completed)
    assert store.latest()["progress"]["completed"]
    (tmp_path / "last.ckpt").write_bytes(b"invalid")
    with pytest.raises(ValueError, match="No verified"):
        store.latest()


def test_completed_local_backup_retries_without_retraining(tmp_path, monkeypatch):
    store = CheckpointStore(tmp_path / "local", tmp_path / "drive")
    store.save(payload(2, 0, 1), improved=True)
    completed = payload(3, 0, 2)
    completed["progress"]["completed"] = True
    replace = checkpoint.os.replace

    def fail_drive_promotion(source, target):
        if target == store.drive / "last.ckpt":
            raise OSError("simulated completed checkpoint backup failure")
        return replace(source, target)

    with monkeypatch.context() as patch:
        patch.setattr(checkpoint.os, "replace", fail_drive_promotion)
        assert "Drive backup FAILED" in store.save(completed)
    store.backup_completed(store.latest())
    remote = CheckpointStore(tmp_path / "fresh", store.drive)
    assert remote.latest()["progress"]["completed"]
