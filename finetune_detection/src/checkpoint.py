"""Verified rotating checkpoints with independent local and Drive recovery."""

import json
import os
import pickle
import random
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from .annotations import sha256_file, write_json

VERSION = 1
RETAIN_STEPS = 2
CHECKPOINT_KEYS = {
    "format", "config", "manifest_sha256", "environment", "run_id", "progress",
    "model", "optimizer", "scheduler", "scaler", "amp_enabled", "rng",
}
READ_ERRORS = (OSError, RuntimeError, ValueError, TypeError, KeyError, IndexError, EOFError, pickle.UnpicklingError, zipfile.BadZipFile)


def capture_rng():
    algorithm, values, position, gaussian_cached, gaussian = np.random.get_state()
    return {
        "python": random.getstate(), "numpy": (algorithm, values.tolist(), position, gaussian_cached, gaussian),
        "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    algorithm, values, position, gaussian_cached, gaussian = state["numpy"]
    random.setstate(state["python"])
    np.random.set_state((algorithm, np.array(values, dtype=np.uint32), position, gaussian_cached, gaussian))
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        if torch.cuda.device_count() != len(state["cuda"]):
            raise ValueError("Checkpoint CUDA RNG count differs from the runtime.")
        torch.cuda.set_rng_state_all(state["cuda"])


def verify_compatibility(saved, current):
    for key in ("config", "manifest_sha256"):
        if saved[key] != current[key]:
            raise ValueError(f"Checkpoint {key} differs. Restore the original inputs or use a distinct run name.")
    previous, proposed = saved["environment"], current["environment"]
    for name in ("torch", "torchvision", "numpy", "Pillow"):
        if previous["packages"][name] != proposed["packages"][name]:
            raise ValueError(f"Checkpoint uses {name} {previous['packages'][name]}; runtime has {proposed['packages'][name]}.")
    if (previous["device_type"], previous["cuda_build"]) != (proposed["device_type"], proposed["cuda_build"]):
        raise ValueError("Checkpoint device type or CUDA build differs from the runtime.")


def seal_path(path):
    return path.with_suffix(path.suffix + ".sha256")


def read_seal(path):
    seal = json.loads(seal_path(path).read_text(encoding="utf-8"))
    if set(seal) != {"run_id", "position", "best_epoch", "best_map", "completed", "bytes", "sha256"}:
        raise ValueError("Checkpoint seal fields are incomplete")
    position = seal["position"]
    if (not isinstance(position, list) or len(position) != 3
            or any(type(value) is not int or value < 0 for value in position)):
        raise ValueError("Invalid checkpoint position in seal")
    return seal


def describe(payload):
    state = payload["progress"]
    return {
        "run_id": payload["run_id"], "position": [state["epoch"], state["next_batch"], state["steps"]],
        "best_epoch": state["best_epoch"], "best_map": state["best_map"], "completed": state["completed"],
    }


def validate_payload(path, mmap=True):
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=mmap)
    if set(payload) != CHECKPOINT_KEYS or payload["format"] != VERSION:
        raise ValueError("Missing checkpoint contents or unsupported checkpoint format")
    state = payload["progress"]
    if (state["epoch"] != len(state["history"]) + 1 or state["next_batch"] < 0 or state["steps"] < 0
            or payload["scheduler"]["last_epoch"] != state["steps"]
            or state["completed"] and (state["next_batch"] != 0 or state["best_epoch"] is None)):
        raise ValueError("Invalid epoch, scheduler or completion state")
    if state["best_epoch"] is not None:
        row = state["history"][state["best_epoch"] - 1]
        if row["epoch"] != state["best_epoch"] or row["map_50_95"] != state["best_map"]:
            raise ValueError("Best checkpoint does not match recorded validation history")
    return payload


def verify_bytes(path, seal):
    if path.stat().st_size != seal["bytes"] or sha256_file(path) != seal["sha256"]:
        raise ValueError(f"Checkpoint size or SHA-256 mismatch: {path}")


def publish(source, target, seal):
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = target.with_suffix(target.suffix + ".part")
    try:
        shutil.copyfile(source, stage)
        verify_bytes(stage, seal)
        # The previous verified checkpoint remains available through both writes.
        os.replace(stage, target)
        verify_bytes(target, seal)
        write_json(seal_path(target), seal)
        if read_seal(target) != seal:
            raise OSError(f"Checkpoint seal verification failed: {target}")
    finally:
        stage.unlink(missing_ok=True)


def rotate(target, previous, protect_best_epoch=None):
    if not target.exists():
        return
    try:
        seal = read_seal(target)
        verify_bytes(target, seal)
    except READ_ERRORS:
        return
    if protect_best_epoch is not None and seal["best_epoch"] != protect_best_epoch:
        # The previous best still belongs to the last committed training state.
        return
    publish(target, previous, seal)


def is_selected_best(seal, reference):
    return (
        seal["run_id"] == reference["run_id"]
        and seal["best_epoch"] == reference["best_epoch"]
        and seal["best_map"] == reference["best_map"]
        and reference["best_epoch"] is not None
        and seal["position"][:2] == [reference["best_epoch"] + 1, 0]
    )


class CheckpointStore:
    def __init__(self, local, drive=None):
        self.local = Path(local)
        self.drive = Path(drive) if drive is not None else None
        self.remote_best_seal = None
        if self.drive and self.local.resolve() == self.drive.resolve():
            raise ValueError("Local checkpoint storage and Drive backup must be different directories.")

    def directories(self):
        yield self.local
        if self.drive:
            yield self.drive

    def candidates(self, report, best=None):
        entries, present = [], False
        for root in self.directories():
            paths = [root / name for name in ("last.ckpt", "last.previous.ckpt", "best.ckpt", "best.previous.ckpt")]
            paths.extend(sorted(root.glob("checkpoints/step-*.ckpt"), reverse=True))
            for path in paths:
                if not (path.exists() or seal_path(path).exists()):
                    continue
                present = True
                try:
                    seal = read_seal(path)
                    if best is None or is_selected_best(seal, best):
                        entries.append((tuple(seal["position"]), path, seal))
                except READ_ERRORS as error:
                    report(f"Checkpoint metadata unreadable at {path}: {error}")
        return sorted(entries, key=lambda entry: entry[0], reverse=True), present

    def read(self, path, seal):
        if self.drive and path.is_relative_to(self.drive):
            # Verify Drive data after staging it locally. mmap works on the local filesystem.
            self.local.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".restore-", dir=self.local) as directory:
                staged = Path(directory) / path.name
                shutil.copyfile(path, staged)
                # A Windows file mapping can keep a temporary file locked until its tensors are freed.
                return self.read_local(staged, seal, mmap=os.name != "nt")
        return self.read_local(path, seal)

    @staticmethod
    def read_local(path, seal, mmap=True):
        verify_bytes(path, seal)
        payload = validate_payload(path, mmap=mmap)
        if any(seal[key] != value for key, value in describe(payload).items()):
            raise ValueError(f"Checkpoint metadata differs from payload: {path}")
        return payload

    def latest(self, report=tqdm.write):
        options, present = self.candidates(report)
        completed = []
        for root in self.directories():
            marker = root / "completed.json"
            if marker.is_file():
                completed.append(json.loads(marker.read_text(encoding="utf-8")))
        for _, path, seal in options:
            if completed and any(marker != describe_seal(seal) for marker in completed):
                continue
            try:
                payload = self.read(path, seal)
                if seal["best_epoch"] is not None and not is_selected_best(seal, seal):
                    self.best(payload, report)
            except READ_ERRORS as error:
                report(f"Skipping unusable checkpoint {path}: {error}")
                continue
            for _, other_path, other_seal in options:
                if other_seal["run_id"] == seal["run_id"]:
                    continue
                try:
                    self.read(other_path, other_seal)
                except READ_ERRORS:
                    continue
                raise ValueError(f"Conflicting run IDs in {path} and {other_path}; verify the local and Drive run directories.")
            return payload
        if present or completed:
            raise ValueError("No verified recoverable checkpoint for this run. Restore a valid backup or use a distinct run name.")
        return None

    def best(self, reference, report=tqdm.write):
        options, _ = self.candidates(report, best=describe(reference))
        # Prefer the dedicated best artifact when both best and last describe the same epoch.
        options.sort(key=lambda entry: entry[1].name not in {"best.ckpt", "best.previous.ckpt"})
        for _, path, seal in options:
            try:
                payload = self.read(path, seal)
                for key in ("manifest_sha256", "config"):
                    if payload[key] != reference[key]:
                        raise ValueError(f"Best checkpoint {key} differs")
                return payload
            except READ_ERRORS as error:
                report(f"Skipping unusable best checkpoint {path}: {error}")
        raise ValueError("No verified best checkpoint for the recorded validation epoch.")

    def restore_local_best(self, reference, report=tqdm.write):
        best = self.local / "best.ckpt"
        try:
            seal = read_seal(best)
            if is_selected_best(seal, describe(reference)):
                verify_bytes(best, seal)
                return
        except READ_ERRORS:
            pass
        payload = self.best(reference, report)
        self.local.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".best-", dir=self.local) as directory:
            source = Path(directory) / "best.ckpt"
            torch.save(payload, source)
            validate_payload(source)
            seal = {**describe(payload), "bytes": source.stat().st_size, "sha256": sha256_file(source)}
            self.promote_best(self.local, source, seal)

    def save(self, payload, improved=False, snapshot_step=False, report=tqdm.write):
        self.local.mkdir(parents=True, exist_ok=True)
        payload["format"] = VERSION
        with tempfile.TemporaryDirectory(prefix=".save-", dir=self.local) as directory:
            source = Path(directory) / "checkpoint.ckpt"
            torch.save(payload, source)
            validate_payload(source)
            seal = {**describe(payload), "bytes": source.stat().st_size, "sha256": sha256_file(source)}
            self.commit(self.local, source, seal, improved, snapshot_step)
            local_best = self.local / "best.ckpt"
            status = ["last local verified"]
            if improved:
                status.append("best local verified")
            if snapshot_step:
                status.append(f"optimizer step {seal['position'][2]}")
            if self.drive:
                try:
                    self.sync_best(local_best, seal)
                    self.commit(self.drive, source, seal, False, snapshot_step)
                    status.append("Drive backup verified")
                except READ_ERRORS as error:
                    status.append("Drive backup FAILED (local retained)")
                    report(f"Drive backup failed for {self.drive}: {error}")
            message = " | ".join(status)
            report(f"Checkpoint: {message}")
            return message

    def backup_completed(self, payload, report=tqdm.write):
        if not self.drive:
            return
        reference = describe(payload)
        last = self.drive / "last.ckpt"
        try:
            try:
                seal = read_seal(last)
                if describe_seal(seal) != reference:
                    raise ValueError("Drive checkpoint does not match completed run")
                verify_bytes(last, seal)
                self.sync_best(self.local / "best.ckpt", seal)
            except READ_ERRORS:
                self.restore_local_best(payload, report)
                with tempfile.TemporaryDirectory(prefix=".complete-", dir=self.local) as directory:
                    source = Path(directory) / "checkpoint.ckpt"
                    torch.save(payload, source)
                    validate_payload(source)
                    seal = {**reference, "bytes": source.stat().st_size, "sha256": sha256_file(source)}
                    self.sync_best(self.local / "best.ckpt", seal)
                    self.commit(self.drive, source, seal, False, False)
                report("Completed checkpoint Drive backup repaired and verified.")
        except READ_ERRORS as error:
            report(f"Completed checkpoint Drive backup failed; local checkpoint retained: {error}")

    def commit(self, root, source, seal, improved, snapshot_step):
        last = root / "last.ckpt"
        if improved:
            self.promote_best(root, source, seal)
        rotate(last, root / "last.previous.ckpt")
        publish(source, last, seal)
        if snapshot_step:
            folder = root / "checkpoints"
            target = folder / f"step-{seal['position'][2]:08d}.ckpt"
            publish(source, target, seal)
            for obsolete in sorted(folder.glob("step-*.ckpt"), reverse=True)[RETAIN_STEPS:]:
                obsolete.unlink(missing_ok=True)
                seal_path(obsolete).unlink(missing_ok=True)
        if seal["completed"]:
            write_json(root / "completed.json", describe_seal(seal))

    @staticmethod
    def promote_best(root, source, seal):
        last = root / "last.ckpt"
        try:
            protected = read_seal(last)["best_epoch"]
        except READ_ERRORS:
            protected = None
        best = root / "best.ckpt"
        rotate(best, root / "best.previous.ckpt", protected)
        publish(source, best, seal)

    def sync_best(self, local_best, reference):
        if reference["best_epoch"] is None:
            return
        remote_best = self.drive / "best.ckpt"
        try:
            remote_seal = read_seal(remote_best)
            if is_selected_best(remote_seal, reference):
                # Reuse the verified Drive backup between improvements.
                if remote_best.stat().st_size == remote_seal["bytes"] and self.remote_best_seal == remote_seal:
                    return
                verify_bytes(remote_best, remote_seal)
                self.remote_best_seal = remote_seal
                return
        except READ_ERRORS:
            pass
        for source in (local_best, self.local / "best.previous.ckpt", self.drive / "best.previous.ckpt"):
            try:
                seal = read_seal(source)
                if is_selected_best(seal, reference):
                    verify_bytes(source, seal)
                    # Copy first when the source is itself the protected Drive recovery file.
                    with tempfile.TemporaryDirectory(prefix=".best-", dir=self.local) as directory:
                        staged = Path(directory) / source.name
                        shutil.copyfile(source, staged)
                        self.promote_best(self.drive, staged, seal)
                    self.remote_best_seal = seal
                    return
            except READ_ERRORS:
                continue
        raise ValueError("Cannot copy the verified best checkpoint referenced by last.")

    def restore_logs(self, local_logs, report=tqdm.write):
        if not self.drive:
            return
        try:
            name = "training.log"
            remote = self.drive / "logs" / name
            local = local_logs / name
            if remote.is_file() and not local.exists():
                shutil.copyfile(remote, local)
        except OSError as error:
            report(f"Drive log restore failed: {error}")

    def backup_logs(self, local_logs, report=tqdm.write):
        if not self.drive:
            return
        try:
            directory = self.drive / "logs"
            directory.mkdir(parents=True, exist_ok=True)
            for path in local_logs.iterdir():
                if not path.is_file() or path.suffix == ".part":
                    continue
                stage = directory / (path.name + ".part")
                shutil.copyfile(path, stage)
                if sha256_file(path) != sha256_file(stage):
                    raise OSError(f"Log checksum mismatch: {path.name}")
                os.replace(stage, directory / path.name)
        except OSError as error:
            report(f"Drive log backup failed; local logs retained: {error}")


def describe_seal(seal):
    return {key: seal[key] for key in ("run_id", "position", "best_epoch", "best_map", "completed")}
