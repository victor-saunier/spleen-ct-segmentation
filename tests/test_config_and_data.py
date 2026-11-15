import json
from dataclasses import replace
from pathlib import Path

import pytest

from spleenseg.config import DEFAULT_CONFIG, load_config
from spleenseg.data import DEFAULT_SPLIT_FILE, case_dicts

ROOT = Path(__file__).resolve().parents[1]


def test_the_committed_config_loads() -> None:
    cfg = load_config(ROOT / DEFAULT_CONFIG, "baseline")
    assert cfg.architecture == "unet3d"
    assert cfg.patch_size == (96, 96, 96)


def test_a_misspelled_key_is_an_error(tmp_path: Path) -> None:
    text = (ROOT / DEFAULT_CONFIG).read_text(encoding="utf-8").replace("max_steps", "max_step")
    path = tmp_path / "train.toml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(TypeError):
        load_config(path, "baseline")


def test_config_is_frozen() -> None:
    cfg = load_config(ROOT / DEFAULT_CONFIG, "baseline")
    with pytest.raises(AttributeError):
        cfg.max_steps = 1  # type: ignore[misc]
    assert replace(cfg, max_steps=1).max_steps == 1  # changes need an explicit new copy


def test_training_code_cannot_list_the_test_set() -> None:
    with pytest.raises(ValueError, match="may only use"):
        case_dicts(ROOT / DEFAULT_SPLIT_FILE, Path("data"), "test")


def test_no_test_case_reaches_the_training_lists() -> None:
    split = json.loads((ROOT / DEFAULT_SPLIT_FILE).read_text(encoding="utf-8"))
    listed = {
        Path(entry["image"]).name.removesuffix(".nii.gz")
        for name in ("train", "val")
        for entry in case_dicts(ROOT / DEFAULT_SPLIT_FILE, Path("data"), name)
    }
    assert listed == set(split["train"]) | set(split["val"])
    assert not listed & set(split["test"])
