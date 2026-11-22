"""Paired comparison of two models evaluated on the same cases.

Usage:
    uv run python -m spleenseg.compare --a outputs/eval/<A>-<split> --b outputs/eval/<B>-<split>

Reads the per_case.csv that evaluate.py wrote for each model, takes the per-case difference
B - A for Dice and HD95, and a percentile bootstrap interval of the mean difference. Pairing by
case removes how different the cases are from each other. If the interval contains 0, neither
model can be called better. Writes outputs/compare/<A>-vs-<B>.json.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

from spleenseg.evaluate import git_commit
from spleenseg.metrics import bootstrap_ci

METRICS = ("dice", "hd95_mm")


def _per_case(eval_dir: Path) -> dict[str, dict[str, float]]:
    with (eval_dir / "per_case.csv").open(encoding="utf-8") as f:
        return {row["case"]: {m: float(row[m]) for m in METRICS} for row in csv.DictReader(f)}


def compare(a_dir: Path, b_dir: Path) -> dict[str, object]:
    a, b = _per_case(a_dir), _per_case(b_dir)
    # A paired comparison only means something on exactly the same cases.
    if a.keys() != b.keys():
        raise ValueError(f"{a_dir.name} and {b_dir.name} were not evaluated on the same cases.")
    result: dict[str, object] = {"a": a_dir.name, "b": b_dir.name, "n_cases": len(a)}
    for metric in METRICS:
        diffs = {case: b[case][metric] - a[case][metric] for case in sorted(a)}
        # HD95 is undefined (NaN) when a mask is empty: count those cases, never drop silently.
        defined = [d for d in diffs.values() if not math.isnan(d)]
        summary: dict[str, object] = {
            "b_minus_a_per_case": diffs,
            "n_defined": len(defined),
            "n_undefined": len(diffs) - len(defined),
        }
        if len(defined) >= 2:
            mean, low, high = bootstrap_ci(defined)
            summary |= {"mean": mean, "ci95": [low, high], "ci_contains_0": low <= 0 <= high}
        result[metric] = summary
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--a", type=Path, required=True, help="evaluate.py output of model A")
    parser.add_argument("--b", type=Path, required=True, help="evaluate.py output of model B")
    parser.add_argument("--out", type=Path, help="default: outputs/compare/<A>-vs-<B>.json")
    args = parser.parse_args(argv)

    result = compare(args.a, args.b) | {"command": " ".join(sys.argv), "git_commit": git_commit()}
    out = args.out or Path("outputs/compare") / f"{args.a.name}-vs-{args.b.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for metric in METRICS:
        s = result[metric]
        assert isinstance(s, dict)
        if "mean" in s:
            low, high = s["ci95"]
            print(f"{metric}: B - A = {s['mean']:+.4f}, 95 % interval [{low:+.4f}, {high:+.4f}]")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
