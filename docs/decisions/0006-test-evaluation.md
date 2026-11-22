# 0006 - Test evaluation (pre-registered)

- Status: accepted
- Date: 2026-11-22
- Written and committed before any test case was read by any model or script. It is not edited
  after the test set is opened: results and any deviation are reported elsewhere (README).

## Context

Both models are trained and every choice was made on training or validation data (decision
records 0002 to 0005). The 7 test cases have never been read. Each look at test numbers followed
by a change would turn the test set into a second validation set, so it is used once, for both
models together, with everything below fixed in advance.

## What is fixed

| Item | Value |
|---|---|
| Test cases | spleen_14, spleen_20, spleen_21, spleen_41, spleen_44, spleen_52, spleen_61 (`splits/split.json`) |
| Baseline | `outputs/runs/baseline-20261117-233020/best.pt` (trained at commit f4e2030) |
| Second model | `outputs/runs/monai_unet-20261122-182222/best.pt` (trained at commit 2ba7943) |
| Prediction | `predict.py` as on the T07 branch: image only, sliding window, back to the original grid, largest connected piece kept |
| Metrics | Dice and HD95 (mm) per case, on the original grid (decision record 0003) |
| Main comparison | Dice, paired: per-case difference second model minus baseline, mean and 95 % percentile bootstrap interval (10,000 resamples, seed 0) |
| Secondary results | Each model's mean Dice and HD95 with bootstrap intervals; paired HD95 difference on the cases where both are defined; the same evaluation without the largest-piece step, to show its effect |
| Error analysis | The 3 cases with the lowest Dice for each model, drawn on the CT (label and prediction outlines) and described in words |

## Procedure (run once, from a clean working tree)

```bash
uv run python -m spleenseg.predict --run-dir outputs/runs/baseline-20261117-233020 --split test --out-dir outputs/predictions/test/baseline-20261117-233020
uv run python -m spleenseg.predict --run-dir outputs/runs/monai_unet-20261122-182222 --split test --out-dir outputs/predictions/test/monai_unet-20261122-182222
uv run python -m spleenseg.predict --run-dir outputs/runs/baseline-20261117-233020 --split test --keep-all-pieces --out-dir outputs/predictions/test/baseline-20261117-233020-all-pieces
uv run python -m spleenseg.predict --run-dir outputs/runs/monai_unet-20261122-182222 --split test --keep-all-pieces --out-dir outputs/predictions/test/monai_unet-20261122-182222-all-pieces
uv run python -m spleenseg.evaluate --split test --pred-dir outputs/predictions/test/<each of the 4 folders above>
uv run python -m spleenseg.compare --a outputs/eval/baseline-20261117-233020-test --b outputs/eval/monai_unet-20261122-182222-test
```

## How the main result will be worded

For the paired Dice difference (second model minus baseline):

- interval entirely above 0: "the MONAI U-Net scored higher than the baseline on these 7 test
  cases";
- interval entirely below 0: "the baseline scored higher than the MONAI U-Net on these 7 test
  cases";
- interval containing 0: "no detectable difference between the two models on 7 test cases".

Every statement adds: 7 test cases (the percentile bootstrap interval is too narrow with so
few), one training run and seed per model, a case-level split (decision record 0001).

## If something goes wrong

- A model predicts nothing on a case: Dice 0, HD95 undefined and counted, as in decision record
  0003. Nothing is re-run.
- A crash or a bug found after the test set is opened: fix it in a separate commit, re-run the
  procedure, and report both the first and the corrected results with the reason. The models,
  checkpoints, post-processing and metrics are never changed.
- No other re-run is allowed.
