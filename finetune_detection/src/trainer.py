"""Optimizer-step training, local recovery and experiment entry points."""

import math
import time
import uuid
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .checkpoint import CheckpointStore, capture_rng, restore_rng, verify_compatibility
from .dataset import LesionDataset, detection_collate, load_manifest
from .evaluator import evaluate_model
from .model import build_model
from .progress import LOSS_NAMES, epoch_summary, training_bar, training_metrics
from .utils import (
    EXPERIMENTS, WORKSPACE, RunLogs, duration, environment, load_config, require_device,
    run_directory, seed_training,
)


def build_scheduler(optimizer, total_steps, config):
    warmup = min(config["warmup_steps"], max(0, total_steps - 1))
    floor, start = config["min_lr_factor"], config["warmup_start_factor"]

    def multiplier(step):
        if step < warmup:
            return start + (1 - start) * step / warmup
        phase = min(1, (step - warmup) / max(1, total_steps - warmup - 1))
        return floor + (1 - floor) * (1 + math.cos(math.pi * phase)) / 2

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def epoch_batches(size, batch_size, seed, epoch):
    generator = torch.Generator().manual_seed(seed + epoch)
    indices = torch.randperm(size, generator=generator).tolist()
    return [indices[start:start + batch_size] for start in range(0, size, batch_size)]


def make_loader(dataset, device, batch_size=1, batches=None):
    options = {
        "num_workers": 0, "pin_memory": device.type == "cuda", "collate_fn": detection_collate,
        # Creating a DataLoader iterator must not advance the model's global RNG on resume.
        "generator": torch.Generator().manual_seed(0),
    }
    if batches is not None:
        return DataLoader(dataset, batch_sampler=batches, **options)
    return DataLoader(dataset, batch_size=batch_size, **options)


def move_batch(images, targets, device):
    return (
        [image.to(device, non_blocking=True) for image in images],
        [{key: tensor.to(device, non_blocking=True) for key, tensor in target.items()} for target in targets],
    )


class TrainingRun:
    def __init__(self, config, train_data, validation_data, manifest_hash, device, store, logs,
                 resume="auto", model_factory=build_model, pretrained=True):
        self.config = config
        self.train_data, self.validation_data = train_data, validation_data
        self.device, self.store, self.logs = device, store, logs
        self.identity = {"config": config, "manifest_sha256": manifest_hash, "environment": environment(device)}
        saved = store.latest(logs.message)
        if saved is not None:
            if resume == "never":
                raise FileExistsError("This run already exists. Specify a unique run name.")
            verify_compatibility(saved, self.identity)
            self.identity["run_id"] = saved["run_id"]
            if self.identity["environment"] != saved["environment"]:
                logs.message("Runtime hardware or auxiliary packages changed since the checkpoint.")
        else:
            self.identity["run_id"] = uuid.uuid4().hex
        self.state = saved["progress"] if saved else {
            "epoch": 1, "next_batch": 0, "steps": 0, "amp_skips": 0, "samples": 0,
            "loss_sums": dict.fromkeys(LOSS_NAMES, 0.0), "epoch_seconds": 0.0,
            "training_seconds": 0.0, "epoch_peak_mib": None, "best_map": None, "best_epoch": None,
            "stale_epochs": 0, "history": [], "completed": False, "stop_reason": None,
        }
        self.started_from_checkpoint = saved is not None
        self.model = None
        if self.state["completed"]:
            self.identity["environment"] = saved["environment"]
            store.backup_completed(saved, logs.message)
            return
        if saved and self.state["best_epoch"] is not None:
            store.restore_local_best(saved, logs.message)
        seed_training(config["seed"])
        self.model = model_factory(config, pretrained=pretrained and saved is None).to(device)
        self.optimizer = torch.optim.SGD(
            (parameter for parameter in self.model.parameters() if parameter.requires_grad),
            lr=config["learning_rate"], momentum=config["momentum"], weight_decay=config["weight_decay"],
        )
        self.accumulation = config["gradient_accumulation_steps"]
        batches = math.ceil(len(train_data) / config["batch_size"])
        self.scheduler = build_scheduler(self.optimizer, math.ceil(batches / self.accumulation) * config["epochs"], config)
        self.amp = bool(config["amp"] and device.type == "cuda" and torch.cuda.get_device_capability(device) >= (5, 3))
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        if saved:
            if saved["amp_enabled"] != self.amp:
                raise ValueError("AMP capability differs from the checkpoint. Use a compatible GPU.")
            self.model.load_state_dict(saved["model"])
            self.optimizer.load_state_dict(saved["optimizer"])
            self.scheduler.load_state_dict(saved["scheduler"])
            self.scaler.load_state_dict(saved["scaler"])
            restore_rng(saved["rng"])
        self.optimizer.zero_grad(set_to_none=True)

    def snapshot(self, improved=False, snapshot_step=False):
        payload = {
            **self.identity, "progress": self.state, "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(), "amp_enabled": self.amp, "rng": capture_rng(),
        }
        return self.store.save(payload, improved, snapshot_step, self.logs.message)

    def train_epoch(self):
        state, config = self.state, self.config
        epoch = state["epoch"]
        batches = epoch_batches(len(self.train_data), config["batch_size"], config["seed"], epoch)
        start_batch = state["next_batch"]
        if (start_batch > len(batches) or (start_batch < len(batches) and start_batch % self.accumulation)
                or state["samples"] != sum(map(len, batches[:start_batch]))):
            raise ValueError("Checkpoint contains an invalid optimizer-step boundary")
        loader = make_loader(self.train_data, self.device, batches=batches[start_batch:])
        self.model.train()
        started = time.perf_counter()
        processed = 0
        with training_bar(config["experiment"], epoch, config["epochs"], len(batches), start_batch) as bar:
            for index, (images, targets) in enumerate(loader, start_batch):
                if index % self.accumulation == 0:
                    group_stop = min(index + self.accumulation, len(batches))
                    group_samples = sum(map(len, batches[index:group_stop]))
                count = len(images)
                images, targets = move_batch(images, targets, self.device)
                lr = self.optimizer.param_groups[0]["lr"]
                with torch.amp.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp):
                    losses = self.model(images, targets)
                    loss = sum(losses[name] for name in LOSS_NAMES[1:])
                readings = torch.stack([loss.detach(), *(losses[name].detach() for name in LOSS_NAMES[1:])]).float().cpu()
                if not torch.isfinite(readings).all():
                    raise FloatingPointError(f"NaN/Inf loss in epoch {epoch}, batch {index + 1}. Last verified checkpoint retained.")
                self.scaler.scale(loss * (count / group_samples)).backward()
                values = dict(zip(LOSS_NAMES, readings.tolist()))
                for name, value in values.items():
                    state["loss_sums"][name] += value * count
                state["samples"] += count
                processed += count
                due = False
                if index + 1 == group_stop:
                    previous_scale = self.scaler.get_scale()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    state["next_batch"] = index + 1
                    new_scale = self.scaler.get_scale()
                    if not math.isfinite(new_scale) or new_scale <= 0:
                        raise FloatingPointError("AMP scale became invalid; resume from the last verified checkpoint.")
                    if new_scale >= previous_scale:
                        state["steps"] += 1
                        self.scheduler.step()
                        due = state["steps"] % config["checkpoint_every_steps"] == 0
                    else:
                        state["amp_skips"] += 1
                        skips = state["amp_skips"]
                        if skips == 1 or skips & (skips - 1) == 0:
                            self.logs.message(f"AMP overflow skipped optimizer update ({skips} total)")
                memory = torch.cuda.memory_allocated(self.device) / 2**20 if self.device.type == "cuda" else None
                rate = processed / max(time.perf_counter() - started, 1e-9) if processed else None
                bar.set_postfix_str(training_metrics(values, lr, memory, rate), refresh=False)
                bar.update(1)
                if due:
                    self.record_time(started)
                    self.snapshot(snapshot_step=True)
                    self.store.backup_logs(self.logs.directory, self.logs.message)
                    started = time.perf_counter()
                    processed = 0
        self.record_time(started)

    def record_time(self, started):
        seconds = time.perf_counter() - started
        self.state["epoch_seconds"] += seconds
        self.state["training_seconds"] += seconds
        if self.device.type == "cuda":
            peak = torch.cuda.max_memory_allocated(self.device) / 2**20
            self.state["epoch_peak_mib"] = max(peak, self.state["epoch_peak_mib"] or 0)

    def run(self):
        state, config = self.state, self.config
        if state["completed"]:
            self.logs.message(f"{config['experiment']} already completed and validated. Best epoch {state['best_epoch']}.")
            if not (self.logs.directory / "history.json").is_file():
                self.logs.history(state["history"])
            if not (self.logs.directory / "run_config.yaml").is_file():
                self.logs.initialize(config, self.identity["environment"])
            self.store.backup_logs(self.logs.directory, self.logs.message)
            return result_summary(state, self.identity)
        self.logs.initialize(config, self.identity["environment"])
        if self.started_from_checkpoint:
            self.logs.history(state["history"])
        location = f"epoch {state['epoch']}, next batch {state['next_batch'] + 1}, optimizer step {state['steps']}"
        self.logs.message(f"{config['experiment']} | {config['model']} | {self.identity['environment']['gpu'] or self.device} | FP16 AMP {self.amp}")
        self.logs.message(f"{'Resumed' if self.started_from_checkpoint else 'Started'} run {self.identity['run_id']} at {location}")
        validation_loader = make_loader(self.validation_data, self.device)
        while not state["completed"]:
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
            self.train_epoch()
            previous_steps = state["history"][-1]["optimizer_steps"] if state["history"] else 0
            if state["steps"] == previous_steps:
                raise FloatingPointError(f"No optimizer updates completed in epoch {state['epoch']}. Check AMP overflow or dataset size.")
            started = time.perf_counter()
            metrics = evaluate_model(self.model, validation_loader, self.device, config["operating_score_threshold"])
            self.record_time(started)
            epoch = state["epoch"]
            improved = state["best_map"] is None or metrics["map_50_95"] > state["best_map"]
            if improved:
                state.update(best_map=metrics["map_50_95"], best_epoch=epoch, stale_epochs=0)
            else:
                state["stale_epochs"] += 1
            converged = epoch >= config["early_stopping_min_epochs"] and state["stale_epochs"] >= config["early_stopping_patience"]
            stop_reason = "early_stopping" if converged else "max_epochs" if epoch >= config["epochs"] else None
            row = {
                "epoch": epoch, **{name: state["loss_sums"][name] / state["samples"] for name in LOSS_NAMES},
                **metrics, "best_map": state["best_map"], "best_epoch": state["best_epoch"],
                "learning_rate": self.optimizer.param_groups[0]["lr"], "optimizer_steps": state["steps"],
                "amp_skips": state["amp_skips"], "epoch_seconds": state["epoch_seconds"],
                "training_seconds": state["training_seconds"], "peak_memory_mib": state["epoch_peak_mib"],
            }
            state["history"].append(row)
            state.update(
                epoch=epoch + 1, next_batch=0, samples=0, loss_sums=dict.fromkeys(LOSS_NAMES, 0.0),
                epoch_seconds=0.0, epoch_peak_mib=None, completed=stop_reason is not None, stop_reason=stop_reason,
            )
            status = self.snapshot(improved=improved)
            self.logs.history(state["history"])
            self.logs.message(epoch_summary(config["experiment"], row, status))
            self.store.backup_logs(self.logs.directory, self.logs.message)
        self.logs.message(
            f"{config['experiment']} finished ({state['stop_reason']}) | best epoch {state['best_epoch']} | "
            f"mAP@50:95 {state['best_map']:.4f} | elapsed {duration(state['training_seconds'])}"
        )
        self.store.backup_logs(self.logs.directory, self.logs.message)
        return result_summary(state, self.identity)


def result_summary(state, identity):
    return {
        "experiment": identity["config"]["experiment"], "run_id": identity["run_id"],
        "completed": state["completed"], "stop_reason": state["stop_reason"],
        "best": next((row for row in state["history"] if row["epoch"] == state["best_epoch"]), None),
    }


def run_experiment(experiment, resume="auto", *, manifest_path=None, runs_root=None,
                   drive_root=None, config_path=None, run_name=None):
    if resume not in {"auto", "never"}:
        raise ValueError("resume must be 'auto' or 'never'")
    if config_path and not run_name:
        raise ValueError("Custom configurations require a unique run name")
    config = load_config(experiment, config_path)
    device = require_device()
    manifest_path = Path(manifest_path or WORKSPACE / "data/training/manifest.json")
    manifest = load_manifest(manifest_path)
    local = run_directory(runs_root or WORKSPACE / "runs", experiment, run_name)
    drive = run_directory(drive_root, experiment, run_name) if drive_root else None
    logs = RunLogs(local)
    store = CheckpointStore(local, drive)
    store.restore_logs(logs.directory, logs.message)
    try:
        run = TrainingRun(
            config, LesionDataset(manifest_path, manifest, "train", config["augmentation"]),
            LesionDataset(manifest_path, manifest, "validation"), manifest["sha256"], device, store, logs, resume,
        )
        return run.run()
    except torch.cuda.OutOfMemoryError as error:
        logs.message("CUDA out of memory. Latest verified checkpoint retained. Use a larger GPU or a named custom configuration.")
        store.backup_logs(logs.directory, logs.message)
        raise RuntimeError(f"{experiment} exhausted GPU memory; configured batch size and resolution were preserved.") from error
    except (Exception, KeyboardInterrupt) as error:
        logs.message(f"{experiment} stopped: {type(error).__name__}: {error}. Latest verified checkpoint retained if available.")
        store.backup_logs(logs.directory, logs.message)
        raise


def run_all_experiments(**options):
    if options.keys() & {"config_path", "run_name"}:
        raise ValueError("Sequential mode uses the three standard experiment configurations")
    results = {}
    for experiment in EXPERIMENTS:
        results[experiment] = run_experiment(experiment, **options)
        torch.cuda.empty_cache()
    return results
