"""Leakage checks on the committed split file, splits/split.json.

What these tests prove:
- no case ID is in two sets, and each set has the planned size;
- the file is exactly what split_cases() gives for its own seed (it was not edited by hand);
- when the data is present (local only, CI has none): the split uses exactly the 41 labelled
  cases, no unlabelled image, and no two cases contain the same image.

What they do NOT prove: that no PATIENT is in two sets. The dataset ships no patient IDs and
its source study scanned patients before and after chemotherapy, so two cases could be the
same person. See docs/decisions/0001-case-level-split.md.
"""

import gzip
import hashlib
import json
import re
from pathlib import Path

import pytest

from spleenseg.split import N_TEST, N_VAL, case_id, labelled_case_ids, split_cases

ROOT = Path(__file__).resolve().parents[1]
SPLIT = json.loads((ROOT / "splits" / "split.json").read_text(encoding="utf-8"))
SETS = ("train", "val", "test")
ALL_IDS = [i for name in SETS for i in SPLIT[name]]

DATASET_DIR = ROOT / "data" / "Task09_Spleen"
needs_data = pytest.mark.skipif(
    not (DATASET_DIR / "dataset.json").exists(), reason="dataset not downloaded (as in CI)"
)


def test_split_is_declared_case_level() -> None:
    assert SPLIT["level"] == "case"


def test_no_case_is_in_two_sets() -> None:
    assert len(ALL_IDS) == len(set(ALL_IDS))


def test_sets_have_the_planned_sizes() -> None:
    assert [len(SPLIT[name]) for name in SETS] == [41 - N_VAL - N_TEST, N_VAL, N_TEST]


def test_ids_look_like_spleen_cases() -> None:
    assert all(re.fullmatch(r"spleen_\d+", i) for i in ALL_IDS)


def test_file_matches_the_split_function() -> None:
    # If this ever fails after a Python upgrade, the shuffle changed: keep the committed file,
    # never regenerate it, or test cases could move into training.
    expected = {name: SPLIT[name] for name in SETS}
    assert split_cases(ALL_IDS, N_VAL, N_TEST, SPLIT["seed"]) == expected


@needs_data
def test_split_uses_exactly_the_labelled_cases() -> None:
    assert set(ALL_IDS) == set(labelled_case_ids(DATASET_DIR / "dataset.json"))


@needs_data
def test_no_unlabelled_image_is_used() -> None:
    meta = json.loads((DATASET_DIR / "dataset.json").read_text(encoding="utf-8"))
    unlabelled = {case_id(path) for path in meta["test"]}
    assert not set(ALL_IDS) & unlabelled


@needs_data
def test_no_two_cases_contain_the_same_image() -> None:
    # Hash the decompressed bytes, so a copy saved with different gzip metadata still matches.
    # This only catches exact copies, not two different scans of the same person.
    hashes = set()
    for i in ALL_IDS:
        with gzip.open(DATASET_DIR / "imagesTr" / f"{i}.nii.gz", "rb") as f:
            hashes.add(hashlib.file_digest(f, "sha256").hexdigest())
    assert len(hashes) == len(ALL_IDS)
