"""Case-level train / validation / test split of the 41 labelled Spleen cases.

Usage (after the download):
    uv run python -m spleenseg.split

CASE-level, not patient-level: the dataset ships no patient IDs and its source study scanned
patients before and after chemotherapy, so two cases could be the same person.
See docs/decisions/0001-case-level-split.md.

The written file (splits/split.json) is committed and is the source of truth: Python does not
promise that `random.shuffle` gives the same order in every version.
"""

import argparse
import json
import random
from pathlib import Path

SEED = 42  # arbitrary, fixed once before any training
N_VAL = 7
N_TEST = 7
DEFAULT_DATASET_JSON = Path("data/Task09_Spleen/dataset.json")
DEFAULT_SPLIT_FILE = Path("splits/split.json")


def case_id(image_path: str) -> str:
    """'./imagesTr/spleen_10.nii.gz' -> 'spleen_10'."""
    return Path(image_path).name.removesuffix(".nii.gz")


def labelled_case_ids(dataset_json: Path) -> list[str]:
    """IDs of the cases listed under "training" in dataset.json: the only ones with labels."""
    meta = json.loads(dataset_json.read_text(encoding="utf-8"))
    return sorted(case_id(entry["image"]) for entry in meta["training"])


def split_cases(case_ids: list[str], n_val: int, n_test: int, seed: int) -> dict[str, list[str]]:
    """Shuffle the IDs with a fixed seed and cut them into train / val / test."""
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("Duplicate case IDs: each case must appear once.")
    if n_val + n_test >= len(case_ids):
        raise ValueError("Not enough cases left for training.")

    # Sort first, so the order in which IDs are given can never change the split.
    shuffled = sorted(case_ids)
    random.Random(seed).shuffle(shuffled)
    return {
        "train": sorted(shuffled[n_val + n_test :]),
        "val": sorted(shuffled[n_test : n_test + n_val]),
        "test": sorted(shuffled[:n_test]),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset-json", type=Path, default=DEFAULT_DATASET_JSON)
    parser.add_argument("--out", type=Path, default=DEFAULT_SPLIT_FILE)
    args = parser.parse_args(argv)

    split = split_cases(labelled_case_ids(args.dataset_json), N_VAL, N_TEST, SEED)
    payload = {"level": "case", "seed": SEED, **split}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n": identical bytes on Windows and Linux, so the file never shows a spurious diff.
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    sizes = ", ".join(f"{name} {len(ids)}" for name, ids in split.items())
    print(f"Wrote {args.out}: {sizes}")


if __name__ == "__main__":
    main()
