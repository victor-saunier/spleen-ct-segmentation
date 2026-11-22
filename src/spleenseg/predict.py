"""Predict spleen masks for the validation cases, back on each scan's original grid.

Usage:
    uv run python -m spleenseg.predict --run-dir outputs/runs/<model>-<date>-<time>
    uv run python -m spleenseg.evaluate --pred-dir outputs/predictions/<model>-<date>-<time>

`--split test` predicts the test cases instead: only for the single, pre-registered final
evaluation (docs/decisions/0006-test-evaluation.md).

Only the image is read: the label never touches prediction. The model predicts on the
preprocessed grid; the mask is then put back on the scan's own grid by undoing the geometric
preprocessing steps (crop, resampling, orientation) in reverse order, and saved with the scan's
own header, so evaluate.py compares it voxel by voxel with the original label.

Only the largest connected piece of the mask is kept (--keep-all-pieces turns this off): every
training label holds one spleen in one piece, so smaller pieces are false detections. This
assumes one spleen per scan: a real second spleen would be erased.
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
from scipy import ndimage
from torch import nn

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


def keep_largest_piece(mask: np.ndarray) -> np.ndarray:
    """Erase every piece of a 3D mask except the largest; voxels touching by a corner count."""
    pieces, n = ndimage.label(mask, structure=np.ones((3, 3, 3)))
    if n <= 1:  # empty or already one piece
        return mask
    largest = np.bincount(pieces.ravel())[1:].argmax() + 1  # [1:] skips the background
    return (pieces == largest).astype(mask.dtype)


def load_run(run_dir: Path, device: torch.device) -> tuple[nn.Module, TrainConfig]:
    """The trained model of a run folder, in eval mode, and the config it was trained with."""
    # Rebuild the model from the run's own record, not from today's config file.
    raw = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))["config"]
    cfg = TrainConfig(**raw)
    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device, weights_only=True))
    model.eval()
    return model, cfg


def predict(
    run_dir: Path,
    image_paths: list[Path],
    out_dir: Path,
    device: torch.device,
    keep_all_pieces: bool = False,
) -> None:
    model, cfg = load_run(run_dir, device)
    use_amp = cfg.amp and device.type == "cuda"

    pipeline = preprocessing(keys=("image",))
    out_dir.mkdir(parents=True, exist_ok=True)
    for image_path in image_paths:
        data = pipeline({"image": str(image_path)})
        # [None] adds the batch axis; a plain tensor in gives a plain tensor out.
        mask = segment(model, data["image"].as_tensor()[None], cfg.patch_size, device, use_amp)
        mask = mask[0].cpu()  # (1, X, Y, Z)
        if not keep_all_pieces:
            mask = torch.from_numpy(keep_largest_piece(mask[0].numpy()))[None]
        save_on_original_grid(mask, data, pipeline, image_path, out_dir / image_path.name)
        print(f"{image_path.name}: saved")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True, help="has best.pt, summary.json")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT_FILE)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument(
        "--out-dir", type=Path, help="default: outputs/predictions/[test/]<run name>"
    )
    parser.add_argument(
        "--keep-all-pieces", action="store_true", help="skip the largest-piece step (to measure it)"
    )
    args = parser.parse_args(argv)

    out_dir = args.out_dir or Path("outputs/predictions") / args.run_dir.name
    if args.split == "test":
        # Deliberately not case_dicts: it keeps the test set out of all training code, and
        # still refuses it. The test set is opened here, on explicit request only.
        cases = json.loads(args.split_file.read_text(encoding="utf-8"))["test"]
        images = [args.dataset_dir / "imagesTr" / f"{case}.nii.gz" for case in cases]
        out_dir = args.out_dir or Path("outputs/predictions/test") / args.run_dir.name
    else:
        images = [Path(c["image"]) for c in case_dicts(args.split_file, args.dataset_dir, "val")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predict(args.run_dir, images, out_dir, device, args.keep_all_pieces)
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
