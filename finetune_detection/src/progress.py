"""Live notebook/terminal progress without per-batch log spam."""

from tqdm.auto import tqdm

LOSS_NAMES = ("train_loss", "loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg")
LOSS_LABELS = ("Loss", "Cls", "Box", "RPN Obj", "RPN Box")
BAR_FORMAT = "{desc} {n_fmt}/{total_fmt} [{percentage:3.0f}%] | {bar} | [{elapsed}<{remaining}]{postfix}"


def training_bar(experiment, epoch, max_epochs, batches, initial):
    return tqdm(total=batches, initial=initial, desc=f"{experiment} epoch {epoch:02d}/{max_epochs:02d} | Train",
                unit="batch", mininterval=0.75, dynamic_ncols=True, bar_format=BAR_FORMAT)


def validation_bar(samples, description):
    return tqdm(total=samples, desc=description, unit="image", mininterval=0.75,
                dynamic_ncols=True, bar_format=BAR_FORMAT)


def reading(value, precision=".4f", unit=""):
    return "N/A" if value is None else f"{value:{precision}}{unit}"


def training_metrics(losses, lr, memory, rate):
    components = [f"{label} {reading(losses.get(key))}" for key, label in zip(LOSS_NAMES, LOSS_LABELS)]
    return " | ".join([
        *components, f"LR {reading(lr, '.6g')}", f"VRAM {reading(memory, '.0f', ' MiB')}",
        f"Speed {reading(rate, '.2f', ' img/s')}",
    ])


def epoch_summary(experiment, row, checkpoint_status):
    from .utils import duration

    return "\n".join([
        f"{experiment} | Epoch {row['epoch']:02d} | Train loss {row['train_loss']:.4f}",
        f"  mAP@50 {row['map_50']:.4f} | mAP@50:95 {row['map_50_95']:.4f} | Recall@IoU0.5 {row['recall_50']:.4f}",
        f"  Mean matched IoU {reading(row['mean_matched_iou'])} | FP/Image {row['fp_per_image']:.4f} | "
        f"Best mAP {row['best_map']:.4f} (epoch {row['best_epoch']})",
        f"  LR {row['learning_rate']:.6g} | Epoch {duration(row['epoch_seconds'])} | Total {duration(row['training_seconds'])} | "
        f"Peak VRAM {reading(row['peak_memory_mib'], '.0f', ' MiB')} | {row['inference_ms_per_image']:.1f} ms/image",
        f"  {checkpoint_status}",
    ])
