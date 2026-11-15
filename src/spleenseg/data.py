"""Lists of {"image", "label"} file paths for the training code, from the committed split."""

import json
from pathlib import Path

DEFAULT_DATASET_DIR = Path("data/Task09_Spleen")
DEFAULT_SPLIT_FILE = Path("splits/split.json")

# The test set is closed until both models are trained: training code can never list it.
TRAINING_SPLITS = ("train", "val")


def case_dicts(split_file: Path, dataset_dir: Path, split: str) -> list[dict[str, str]]:
    if split not in TRAINING_SPLITS:
        raise ValueError(f"Training code may only use {TRAINING_SPLITS}, not {split!r}.")
    cases = json.loads(split_file.read_text(encoding="utf-8"))[split]
    return [
        {
            "image": str(dataset_dir / "imagesTr" / f"{case}.nii.gz"),
            "label": str(dataset_dir / "labelsTr" / f"{case}.nii.gz"),
        }
        for case in cases
    ]
