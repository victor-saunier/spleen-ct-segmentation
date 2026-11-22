"""Export the network of a trained run to a single ONNX file.

Usage:
    uv run python -m spleenseg.export --run-dir outputs/runs/<model>-<date>-<time>

Writes outputs/onnx/<run name>.onnx and <run name>.json (git-ignored). Only the network is
exported: it maps a batch of preprocessed 96^3 windows to two scores per voxel. Preprocessing,
the sliding window and post-processing stay in predict.py.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from spleenseg.evaluate import git_commit
from spleenseg.predict import load_run


def export(run_dir: Path, out_path: Path) -> dict[str, object]:
    model, cfg = load_run(run_dir, torch.device("cpu"))
    example = torch.zeros(1, 1, *cfg.patch_size)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (example,),
        str(out_path),
        input_names=["image"],
        output_names=["logits"],
        dynamic_shapes=({0: "batch"},),  # axis 0 of the input may be any number of windows
        external_data=False,  # one file, weights included: simpler to ship
        verbose=False,  # its progress messages contain emoji the Windows console cannot print
    )
    onnx.checker.check_model(str(out_path))  # a structurally valid ONNX graph

    # Quick self-check on random windows; the real check compares masks on validation scans.
    windows = torch.rand(2, 1, *cfg.patch_size, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        expected = model(windows).numpy()
    session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"image": windows.numpy()})[0]
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    record: dict[str, object] = {
        "run": run_dir.name,
        "trained_at_commit": summary["git_commit"],
        "exported_at_commit": git_commit(),
        "patch_size": list(cfg.patch_size),
        "random_windows_max_abs_diff": float(np.abs(expected - actual).max()),
        "random_windows_same_class_share": float((expected.argmax(1) == actual.argmax(1)).mean()),
        "size_mb": round(out_path.stat().st_size / 1e6, 2),
        "torch": torch.__version__,
        "onnx": onnx.__version__,
        "onnxruntime": ort.__version__,
    }
    out_path.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True, help="has best.pt, summary.json")
    parser.add_argument("--out", type=Path, help="default: outputs/onnx/<run name>.onnx")
    args = parser.parse_args(argv)
    out = args.out or Path("outputs/onnx") / f"{args.run_dir.name}.onnx"
    print(json.dumps(export(args.run_dir, out), indent=2))
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
