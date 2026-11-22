"""Tests for the paired comparison, on hand-written per-case tables."""

import csv
import json
import math
from pathlib import Path

import pytest

from spleenseg.compare import compare, main


def _write(eval_dir: Path, rows: list[tuple[str, float, float]]) -> Path:
    eval_dir.mkdir(parents=True)
    with (eval_dir / "per_case.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["case", "dice", "hd95_mm"])
        writer.writerows(rows)
    return eval_dir


def test_a_model_better_on_every_case_gives_an_interval_above_0(tmp_path: Path) -> None:
    a = _write(tmp_path / "a", [("c1", 0.90, 5.0), ("c2", 0.80, 9.0), ("c3", 0.95, 3.0)])
    b = _write(tmp_path / "b", [("c1", 0.92, 4.0), ("c2", 0.83, 8.0), ("c3", 0.96, 2.0)])
    dice = compare(a, b)["dice"]
    assert isinstance(dice, dict)
    assert dice["mean"] == pytest.approx(0.02)
    assert dice["ci95"][0] > 0 and dice["ci_contains_0"] is False


def test_undefined_hd95_is_counted_not_dropped_silently(tmp_path: Path) -> None:
    a = _write(tmp_path / "a", [("c1", 0.9, 5.0), ("c2", 0.8, 9.0), ("c3", 0.9, 3.0)])
    b = _write(tmp_path / "b", [("c1", 0.9, 4.0), ("c2", 0.0, math.nan), ("c3", 0.9, 2.0)])
    hd95 = compare(a, b)["hd95_mm"]
    assert isinstance(hd95, dict)
    assert (hd95["n_defined"], hd95["n_undefined"]) == (2, 1)
    assert hd95["mean"] == pytest.approx(-1.0)


def test_different_cases_are_refused(tmp_path: Path) -> None:
    a = _write(tmp_path / "a", [("c1", 0.9, 5.0), ("c2", 0.8, 9.0)])
    b = _write(tmp_path / "b", [("c1", 0.9, 5.0), ("c3", 0.8, 9.0)])
    with pytest.raises(ValueError, match="same cases"):
        compare(a, b)


def test_the_command_saves_a_traceable_summary(tmp_path: Path) -> None:
    a = _write(tmp_path / "a", [("c1", 0.9, 5.0), ("c2", 0.8, 9.0)])
    b = _write(tmp_path / "b", [("c1", 0.8, 6.0), ("c2", 0.9, 8.0)])
    out = tmp_path / "out.json"
    main(["--a", str(a), "--b", str(b), "--out", str(out)])
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["n_cases"] == 2 and "git_commit" in saved and "command" in saved
