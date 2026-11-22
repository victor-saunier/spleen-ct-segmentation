"""Measure how fast the network predicts, with PyTorch and with ONNX Runtime.

Usage:
    uv run python -m spleenseg.benchmark --run-dir outputs/runs/<run> --onnx outputs/onnx/<run>.onnx
        --power-note "plugged in, power mode: ..."

Two measurements on one validation scan (never a test scan), for every available backend:
- window: one 96^3 window through the network;
- volume: the whole preprocessed scan through the sliding window, as predict.py does it.
Reading files, preprocessing and saving are not timed: they do not depend on the backend.
Each measurement starts with untimed warm-up runs (the first calls are slow while libraries
set up), then repeats the timed run and keeps the median, the fastest and the slowest.
Writes outputs/benchmark/<run name>-<date>-<time>.json with the hardware and versions.
"""

import argparse
import json
import os
import platform
import statistics
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import onnxruntime as ort
import torch

from spleenseg.data import DEFAULT_DATASET_DIR, DEFAULT_SPLIT_FILE, case_dicts
from spleenseg.evaluate import git_commit
from spleenseg.predict import load_run, onnx_predictor
from spleenseg.train import segment
from spleenseg.transforms import preprocessing


def _time(run: Callable[[], object], sync: Callable[[], None], warmup: int, repeats: int) -> dict:
    for _ in range(warmup):
        run()
        sync()
    seconds = []
    for _ in range(repeats):
        start = time.perf_counter()
        run()
        sync()  # the GPU works in the background: wait for it before stopping the clock
        seconds.append(time.perf_counter() - start)
    return {
        "median_s": round(statistics.median(seconds), 4),
        "min_s": round(min(seconds), 4),
        "max_s": round(max(seconds), 4),
        "repeats": repeats,
    }


def _time_backend(
    model: Callable[[torch.Tensor], torch.Tensor],
    device: torch.device,
    amp: bool,
    volume: torch.Tensor,
    patch_size: tuple[int, ...],
    warmup: int,
    repeats: int,
) -> dict[str, dict]:
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    window = torch.rand(1, 1, *patch_size)

    @torch.no_grad()
    def one_window() -> object:
        with torch.autocast(device.type, enabled=amp):
            return model(window.to(device))

    def whole_volume() -> object:
        return segment(model, volume, patch_size, device, amp)

    return {
        "window": _time(one_window, sync, warmup, repeats),
        "volume": _time(whole_volume, sync, warmup, repeats),
    }


def benchmark(
    run_dir: Path, onnx_path: Path, image: Path, warmup: int, repeats: int
) -> dict[str, object]:
    cpu = torch.device("cpu")
    torch_model, cfg = load_run(run_dir, cpu)
    onnx_model, _ = onnx_predictor(onnx_path)
    # name: (model, device, mixed precision); the GPU uses the precision it was trained with.
    backends: dict[str, tuple[Callable[[torch.Tensor], torch.Tensor], torch.device, bool]] = {
        "pytorch-cpu": (torch_model, cpu, False),
        "onnxruntime-cpu": (onnx_model, cpu, False),
    }
    if torch.cuda.is_available():
        gpu = torch.device("cuda")
        backends["pytorch-gpu"] = (load_run(run_dir, gpu)[0], gpu, cfg.amp)

    volume = preprocessing(keys=("image",))({"image": str(image)})["image"].as_tensor()[None]
    results = {}
    for name, (model, device, amp) in backends.items():
        results[name] = _time_backend(
            model, device, amp, volume, tuple(cfg.patch_size), warmup, repeats
        )
        print(name, json.dumps(results[name]))

    return {
        "run": run_dir.name,
        "onnx": onnx_path.name,
        "image": image.name,
        "volume_shape": list(volume.shape[2:]),
        "patch_size": list(cfg.patch_size),
        "warmup": warmup,
        "results": results,
        "cpu": platform.processor(),
        "cpu_logical_cores": os.cpu_count(),
        "torch_cpu_threads": torch.get_num_threads(),
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "torch": torch.__version__,
        "onnxruntime": ort.__version__,
        "os": platform.platform(),
        "git_commit": git_commit(),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True, help="has best.pt, summary.json")
    parser.add_argument("--onnx", type=Path, required=True, help="the same network, exported")
    parser.add_argument("--image", type=Path, help="default: the first validation case")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT_FILE)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out", type=Path, help="default: outputs/benchmark/<run>-<stamp>.json")
    parser.add_argument(
        "--power-note", required=True, help='a laptop is slower on battery: "plugged in, ..."'
    )
    args = parser.parse_args(argv)

    # case_dicts never lists test cases: timing uses validation data only.
    image = args.image or Path(case_dicts(args.split_file, args.dataset_dir, "val")[0]["image"])
    record = benchmark(args.run_dir, args.onnx, image, args.warmup, args.repeats)
    record["power_note"] = args.power_note
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = args.out or Path("outputs/benchmark") / f"{args.run_dir.name}-{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
