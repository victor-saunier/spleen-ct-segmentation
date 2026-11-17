# 0004 - Training protocol

- Status: accepted
- Date: 2026-11-18

## Context

Two models are compared: a small baseline U-Net and a stronger model (task T06). The comparison
is only fair if both are trained with the same data, budget and selection rule. Training runs on
one laptop GPU with 8 GB of memory, so batch size and budget are limited by memory and time.
Validation has 7 cases and the test set (7 cases) stays closed until both models are trained.

## Decisions

| Topic | Choice | Options rejected |
|---|---|---|
| Shared settings | One `[shared]` section in `configs/train.toml` for every model; only the architecture differs | One config file per model: shared values can drift apart unnoticed. |
| Batch | 2 cases x 4 patches of 96^3 = 8 patches per step | Larger batch: not needed, and it must fit every compared model. |
| Loss | Dice + cross-entropy (`DiceCELoss`) | Cross-entropy alone: dominated by the 99.6 % background. Dice alone: less stable gradients. |
| Optimizer | AdamW, learning rate 1e-4, weight decay 1e-5, constant | SGD: needs more learning-rate tuning. A schedule: one more setting to justify. |
| Precision | Mixed precision (AMP) on GPU | Full 32-bit: about twice the memory. |
| Budget | 5000 optimizer steps for every model | Equal wall-clock time: would give the faster model more steps. |
| Checkpoint selection | Every 250 steps (20 checks), mean Dice on the 7 validation cases, resampled grid; keep the best | Last checkpoint: may be past the best point. More checks: more time and a more optimistic best score. |
| How the budget was chosen | From a pilot's measured speed and memory; the owner chose 5000 steps from the wall-clock estimate | From validation curves: would tune the protocol on the validation cases. |
| Prediction | Image only: preprocessing, sliding window, argmax, then `Invertd` (nearest neighbour) back to the scan's grid, saved with the scan's header | Reading the label to place the mask: the label would touch prediction, and new scans have none. Inverting probabilities with linear interpolation, then argmax: possibly smoother borders, but moves a 2-channel float volume onto the full grid. Kept as an option to compare on validation. |

## Measurements

All on an NVIDIA GeForce RTX 4060 Laptop GPU (8 GB), Windows 11, PyTorch 2.15.0 + CUDA 13.2,
laptop plugged in, Windows power mode "Hautes performances".

Pilot, 60 steps with the real training loop, 8 patches per step:

| Model | Weights | Median step | Waiting for data | Peak GPU memory, allocated / reserved |
|---|---|---|---|---|
| Baseline U-Net | 1.40 M | 1.10 s (first pilot), 0.78 s (repeated) | 0.35 s | 4.17 / 5.74 GB |
| MONAI U-Net, residual units (likely T06 model) | 4.81 M | 0.56 s | 0.38 s | 1.05 / 1.56 GB |

- Both models fit at 8 patches per step, so the batch was kept.
- The first baseline pilot was slower in its GPU part only (data waiting was the same). Repeated
  after the real run with the same code and settings, it matched the real run. The cause of the
  first, slower measurement is unknown. The budget was chosen from the slower estimate.
- The MONAI U-Net has 3.4 times more weights but needs 4 times less memory: it downsamples in its
  first layer, while the baseline computes its first and last levels (16 channels) at the full
  96^3. Activations, not weights, fill the GPU.

Baseline run (`baseline-20261117-233020`, commit f4e2030 on the T05b branch):

- 5000 steps in 66 min (3955 s), plus 28 s to cache the preprocessed cases.
- Median step 0.785 s; the average per 250 steps varied between 0.74 and 0.85 s.
- Peak GPU memory 4.17 GB allocated.
- Best selection checkpoint at step 4750 of 5000. The training loss was flat over the last
  750 steps.

Cost of the grid: sending the true validation labels through preprocessing and back (no model)
gives Dice 0.978 to 0.987. This is what the 1.5 x 1.5 x 2 mm grid costs a perfect mask with this
prediction method: a reference point, not a hard ceiling.

## Consequences

- T06 uses the same `[shared]` section unchanged. Its pilot memory must be checked again if the
  T06 model differs from the one measured here.
- Validation scores are optimistic: the same 7 cases choose the checkpoint. Reported results come
  from the test set, evaluated once for both models together.
- Runs are seeded, not bit-for-bit reproducible on GPU.
- Each run records the git commit. GitHub's rebase merge gives commits new hashes on main; the
  recorded hash stays reachable through the pull request's branch.
- With 20 checks and the best at step 4750, a longer budget might have helped slightly. The budget
  is not changed after seeing validation scores.
