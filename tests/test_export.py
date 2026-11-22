"""Export a tiny trained-run folder to ONNX and check ONNX Runtime gives PyTorch's outputs."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from test_train import TINY

from spleenseg.export import main
from spleenseg.model import UNet3D


def test_exported_network_matches_pytorch(tmp_path: Path) -> None:
    # A run folder as train.py leaves it, with random weights.
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cfg = replace(TINY, amp=False)
    summary = {"config": asdict(cfg), "git_commit": "abc123"}
    (run_dir / "summary.json").write_text(json.dumps(summary))
    model = UNet3D(cfg.channels).eval()
    torch.save(model.state_dict(), run_dir / "best.pt")

    out = tmp_path / "model.onnx"
    main(["--run-dir", str(run_dir), "--out", str(out)])

    # One file with the weights inside (no separate model.onnx.data), plus the record.
    assert {p.name for p in tmp_path.glob("model.*")} == {"model.onnx", "model.json"}
    record = json.loads((tmp_path / "model.json").read_text(encoding="utf-8"))
    assert record["trained_at_commit"] == "abc123"

    # Any number of windows (dynamic batch axis), same scores as PyTorch up to rounding.
    windows = torch.randn(3, 1, *cfg.patch_size)
    with torch.no_grad():
        expected = model(windows).numpy()
    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"image": windows.numpy()})[0]
    assert actual.shape == expected.shape == (3, 2, *cfg.patch_size)
    np.testing.assert_allclose(actual, expected, atol=1e-4)
