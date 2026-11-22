"""Train one model with the shared protocol; keep the checkpoint with the best validation Dice.

Usage:
    uv run python -m spleenseg.train --model baseline --power-note "plugged in, Best performance"

Outputs go to outputs/runs/<model>-<date>-<time>/ (git-ignored):
    best.pt       weights with the best selection Dice
    log.csv       one row per validation: step, mean training loss, selection Dice, time
    summary.json  config, best score, timing, peak GPU memory, hardware, versions, git commit

"Selection Dice" is computed on the resampled 1.5 x 1.5 x 2 mm grid: a cheap way to choose the
checkpoint. Reported validation numbers come from predictions on the original grid. The best of
many validation checks is optimistic, so it is never presented as the model's performance.
"""

import argparse
import csv
import json
import platform
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from monai.data import CacheDataset, DataLoader
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.networks.nets import UNet
from monai.utils import set_determinism
from torch import nn

from spleenseg.config import DEFAULT_CONFIG, TrainConfig, load_config
from spleenseg.data import DEFAULT_DATASET_DIR, DEFAULT_SPLIT_FILE, case_dicts
from spleenseg.evaluate import git_commit
from spleenseg.metrics import dice
from spleenseg.model import UNet3D
from spleenseg.transforms import preprocessing, train_transforms

CACHE_THREADS = 4  # threads that preprocess every case once, before training starts


def build_model(cfg: TrainConfig) -> nn.Module:
    if cfg.architecture == "unet3d":
        return UNet3D(cfg.channels)
    if cfg.architecture == "monai_unet":
        # MONAI's reference U-Net for this task: residual units, strided convolutions instead
        # of pooling. Instance norm, as in the baseline: batch statistics are unreliable here.
        return UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=cfg.channels,
            strides=[2] * (len(cfg.channels) - 1),  # halve the size between levels
            num_res_units=2,
            norm="instance",
        )
    raise ValueError(f"Unknown architecture {cfg.architecture!r}")


@torch.no_grad()  # no gradients when only predicting: faster, less memory
def segment(
    model: Callable[[torch.Tensor], torch.Tensor],
    images: torch.Tensor,
    patch_size: Sequence[int],
    device: torch.device,
    use_amp: bool,
) -> torch.Tensor:
    """Predicted class of every voxel, (batch, 1, X, Y, Z), for whole preprocessed volumes.

    Shared by checkpoint selection and predict.py, so both always predict the same way.
    `model` is a PyTorch model already in eval mode, or any function with the same input and
    output, such as an ONNX Runtime session wrapped by predict.onnx_predictor.
    """
    with torch.autocast(device.type, enabled=use_amp):
        # The whole volume is too big for one pass: predict overlapping windows and blend them.
        logits = sliding_window_inference(
            images.to(device), patch_size, sw_batch_size=4, predictor=model
        )
    assert isinstance(logits, torch.Tensor)  # our models return one tensor
    # The predicted class is the larger of the two scores: no threshold to tune.
    return logits.argmax(dim=1, keepdim=True)


def selection_dice(
    model: nn.Module, loader: DataLoader, cfg: TrainConfig, device: torch.device, use_amp: bool
) -> float:
    """Mean Dice over the validation cases, on the resampled grid."""
    model.eval()
    scores = []
    for batch in loader:
        prediction = segment(model, batch["image"], cfg.patch_size, device, use_amp)
        scores.append(dice(prediction[0, 0].cpu().numpy(), batch["label"][0, 0].numpy()))
    return float(np.mean(scores))


def train(
    cfg: TrainConfig,
    dataset_dir: Path,
    split_file: Path,
    run_dir: Path,
    device: torch.device,
    power_note: str = "",
) -> dict[str, object]:
    run_dir.mkdir(parents=True, exist_ok=False)  # never overwrite an earlier run
    # Seeds Python, NumPy and PyTorch. GPU runs are still not bit-identical: "seeded", not
    # "reproducible".
    set_determinism(seed=cfg.seed)

    started = time.perf_counter()
    train_data = CacheDataset(
        case_dicts(split_file, dataset_dir, "train"),
        transform=train_transforms(patch_size=cfg.patch_size, num_samples=cfg.samples_per_case),
        num_workers=CACHE_THREADS,
        progress=False,  # we time the caching ourselves
    )
    val_data = CacheDataset(
        case_dicts(split_file, dataset_dir, "val"),
        transform=preprocessing(),
        num_workers=CACHE_THREADS,
        progress=False,  # we time the caching ourselves
    )
    cache_seconds = time.perf_counter() - started
    # MONAI's DataLoader stacks the 4 patches of each case into one batch of patches.
    train_loader = DataLoader(
        train_data, batch_size=cfg.cases_per_batch, shuffle=True, num_workers=cfg.num_workers
    )
    val_loader = DataLoader(val_data, batch_size=1, num_workers=cfg.num_workers)

    model = build_model(cfg).to(device)
    # Cross-entropy scores every voxel; the Dice term rewards overlap despite the tiny spleen.
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    use_amp = cfg.amp and device.type == "cuda"
    # With 16-bit floats, tiny gradients can round to zero: the scaler enlarges them first.
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    log_path = run_dir / "log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["step", "train_loss", "selection_dice", "elapsed_s"])

    step, best, best_step = 0, -1.0, 0
    losses: list[float] = []
    step_seconds: list[float] = []
    training_started = last_step_end = time.perf_counter()
    while step < cfg.max_steps:
        for batch in train_loader:
            model.train()
            images, labels = batch["image"].to(device), batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)  # forget the previous step's gradients
            with torch.autocast(device.type, enabled=use_amp):
                loss = loss_fn(model(images), labels)  # forward pass + how wrong it is
            scaler.scale(loss).backward()  # gradients: which way each weight should move
            scaler.step(optimizer)  # move every weight a small step that way
            scaler.update()
            losses.append(loss.item())  # .item() waits for the GPU, so the timing is real
            step += 1
            step_seconds.append(time.perf_counter() - last_step_end)  # data loading + compute

            if step % cfg.val_every_steps == 0 or step == cfg.max_steps:
                score = selection_dice(model, val_loader, cfg, device, use_amp)
                if score > best:
                    best, best_step = score, step
                    torch.save(model.state_dict(), run_dir / "best.pt")
                elapsed = time.perf_counter() - training_started
                with log_path.open("a", newline="", encoding="utf-8") as f:
                    row = [step, f"{np.mean(losses):.4f}", f"{score:.4f}", f"{elapsed:.0f}"]
                    csv.writer(f).writerow(row)
                print(
                    f"step {step}/{cfg.max_steps} | loss {np.mean(losses):.4f} | "
                    f"selection Dice {score:.4f} | best {best:.4f} at step {best_step}"
                )
                losses = []
            last_step_end = time.perf_counter()  # validation time is not a training step
            if step >= cfg.max_steps:
                break

    on_gpu = device.type == "cuda"
    summary: dict[str, object] = {
        "config": asdict(cfg),
        "best_selection_dice": best,
        "best_step": best_step,
        "steps": step,
        "cache_seconds": round(cache_seconds, 1),
        "training_seconds": round(time.perf_counter() - training_started, 1),
        "median_step_seconds": round(float(np.median(step_seconds)), 3),
        "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated(device) / 1e9, 2)
        if on_gpu
        else None,
        "amp": use_amp,
        "device": torch.cuda.get_device_name(device) if on_gpu else platform.processor(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "os": platform.platform(),
        "power_note": power_note,
        "git_commit": git_commit(),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="a [models.<name>] section of the config")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT_FILE)
    parser.add_argument("--run-dir", type=Path, help="default: outputs/runs/<model>-<date>-<time>")
    parser.add_argument(
        "--power-note", required=True, help='how the laptop was powered, e.g. "plugged in, Best"'
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.model)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.run_dir or Path("outputs/runs") / f"{args.model}-{stamp}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary = train(cfg, args.dataset_dir, args.split_file, run_dir, device, args.power_note)
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=2))
    print(f"Saved to {run_dir}")


if __name__ == "__main__":  # required on Windows, where data-loading workers re-import this file
    main()
