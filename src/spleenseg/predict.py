"""Predict spleen masks for the validation cases, back on each scan's original grid.

Usage:
    uv run python -m spleenseg.predict --run-dir outputs/runs/<model>-<date>-<time>
    uv run python -m spleenseg.evaluate --pred-dir outputs/predictions/<model>-<date>-<time>

Only the image is read: the label never touches prediction. The model predicts on the
preprocessed grid; the mask is then put back on the scan's own grid by undoing the geometric
preprocessing steps (crop, resampling, orientation) in reverse order, and saved with the scan's
own header, so evaluate.py compares it voxel by voxel with the original label.
"""

import argparse
import json
from collections.abc import Hashable, Mapping
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
from monai import transforms as mt
from monai.data import MetaTensor

from spleenseg.config import TrainConfig
from spleenseg.data import DEFAULT_DATASET_DIR, DEFAULT_SPLIT_FILE, case_dicts
from spleenseg.train import build_model, segment
from spleenseg.transforms import preprocessing


def save_on_original_grid(
    mask: torch.Tensor,
    preprocessed: Mapping[Hashable, Any],  # what `pipeline` returned, as MONAI types it
    pipeline: mt.Compose,
    image_path: Path,
    out_path: Path,
) -> None:
    """Undo `pipeline` on a (1, X, Y, Z) mask of the preprocessed grid; save it like the scan."""
    # Invertd reads the list of steps that preprocessing recorded inside the image's MetaTensor.
    # The mask must be a MetaTensor too: for a plain tensor Invertd only warns and does nothing.
    data = {**preprocessed, "mask": MetaTensor(mask.cpu())}
    restored = mt.Invertd(keys="mask", transform=pipeline, orig_keys="image", nearest_interp=True)(
        data
    )["mask"]
    scan = nib.Nifti1Image.from_filename(image_path)
    array = restored[0].numpy().astype(np.uint8)
    # Never write a mask that silently stayed on the wrong grid: evaluate.py would refuse it,
    # but failing here names the real cause.
    if array.shape != scan.shape or not np.allclose(np.asarray(restored.affine), scan.affine):
        raise RuntimeError(f"{out_path.name}: the mask did not return to the scan's grid.")
    out = nib.Nifti1Image(array, scan.affine, scan.header)
    out.set_data_dtype(np.uint8)  # the scan's header says float32: a mask is 0/1 integers
    nib.save(out, str(out_path))


def predict(run_dir: Path, image_paths: list[Path], out_dir: Path, device: torch.device) -> None:
    # Rebuild the model from the run's own record, not from today's config file.
    raw = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))["config"]
    cfg = TrainConfig(**raw)
    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device, weights_only=True))
    model.eval()
    use_amp = cfg.amp and device.type == "cuda"

    pipeline = preprocessing(keys=("image",))
    out_dir.mkdir(parents=True, exist_ok=True)
    for image_path in image_paths:
        data = pipeline({"image": str(image_path)})
        # [None] adds the batch axis; a plain tensor in gives a plain tensor out.
        mask = segment(model, data["image"].as_tensor()[None], cfg.patch_size, device, use_amp)
        save_on_original_grid(mask[0], data, pipeline, image_path, out_dir / image_path.name)
        print(f"{image_path.name}: saved")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True, help="has best.pt, summary.json")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT_FILE)
    parser.add_argument("--out-dir", type=Path, help="default: outputs/predictions/<run name>")
    args = parser.parse_args(argv)

    # case_dicts refuses the test split: the test set stays closed until both models are trained.
    cases = case_dicts(args.split_file, args.dataset_dir, "val")
    out_dir = args.out_dir or Path("outputs/predictions") / args.run_dir.name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predict(args.run_dir, [Path(c["image"]) for c in cases], out_dir, device)
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
