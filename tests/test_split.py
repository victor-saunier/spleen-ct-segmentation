"""Tests for the split function, on synthetic case IDs (no data needed)."""

import json
from pathlib import Path

import pytest

from spleenseg.split import case_id, labelled_case_ids, split_cases

IDS = [f"spleen_{i}" for i in range(41)]


def test_case_id_strips_folder_and_extension() -> None:
    assert case_id("./imagesTr/spleen_10.nii.gz") == "spleen_10"


def test_labelled_case_ids_reads_only_training_entries(tmp_path: Path) -> None:
    entry = {"image": "./imagesTr/spleen_2.nii.gz", "label": "./labelsTr/spleen_2.nii.gz"}
    meta = {"training": [entry], "test": ["./imagesTs/spleen_1.nii.gz"]}
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(meta), encoding="utf-8")

    assert labelled_case_ids(path) == ["spleen_2"]


def test_sets_are_disjoint_complete_and_sized() -> None:
    split = split_cases(IDS, n_val=7, n_test=7, seed=0)
    train, val, test = (set(split[k]) for k in ("train", "val", "test"))

    assert (len(train), len(val), len(test)) == (27, 7, 7)
    assert not (train & val) and not (train & test) and not (val & test)
    assert train | val | test == set(IDS)


def test_same_seed_gives_same_split_whatever_the_input_order() -> None:
    assert split_cases(IDS, 7, 7, seed=0) == split_cases(list(reversed(IDS)), 7, 7, seed=0)


def test_different_seed_gives_different_split() -> None:
    assert split_cases(IDS, 7, 7, seed=0) != split_cases(IDS, 7, 7, seed=1)


def test_duplicate_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        split_cases(["a", "a", "b", "c"], 1, 1, seed=0)


def test_too_few_cases_for_training_are_rejected() -> None:
    with pytest.raises(ValueError, match="training"):
        split_cases(["a", "b"], 1, 1, seed=0)
