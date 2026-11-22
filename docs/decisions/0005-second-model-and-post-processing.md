# 0005 - Second model and post-processing

- Status: accepted
- Date: 2026-11-22

## Context

The baseline (decision record 0004) is a small U-Net written from scratch. The project compares
it with a stronger, standard model trained under exactly the same protocol. Looking at the
baseline's validation masks showed small false pieces far from the spleen (in the arm, the
pelvis, bowel, a back muscle) in 6 of 7 cases.

## Decisions

| Topic | Choice | Options rejected |
|---|---|---|
| Second model | MONAI's `UNet` with residual units: 5 levels (16 to 256 channels), strided convolutions, PReLU, 4.81 M weights | nnU-Net: a full framework with its own preprocessing and training rules, which would break the shared protocol. UNETR / SwinUNETR (transformers): heavy on memory and data-hungry, with 27 training cases. SegResNet: reasonable, but nothing measured for it. |
| Normalisation | Instance norm | Batch norm (MONAI's spleen tutorial): unreliable statistics with a few patches per step, as for the baseline. |
| Protocol | The `[shared]` section of decision record 0004, unchanged | Tuning the second model separately: the comparison would no longer be fair. |
| Post-processing | Keep only the largest connected piece of the predicted mask (26-connectivity), for both models, at prediction time only | Keep every piece: the far pieces are certain errors. Remove pieces under a size limit: a new number to tune, on validation. |

## Why

- Residual units add each block's input back onto its output, so every block only learns a
  correction and the gradient has a direct path through a deeper network.
- Largest piece: 26 of the 27 training labels are a single piece; the other has a second piece of
  3 voxels next to 96,411. On this data, any smaller predicted piece is a false detection. The
  idea came from looking at validation masks; the justification comes from the training labels.
- The step runs in `predict.py` only. Training and checkpoint selection are unchanged, so the
  baseline did not need to be retrained. `--keep-all-pieces` skips it, to measure its effect.

## Measurements

Same hardware and settings as decision record 0004 (RTX 4060 Laptop GPU, 8 GB, plugged in,
power mode "Hautes performances").

- MONAI U-Net run (`monai_unet-20261122-182222`, commit 2ba7943 on the T06 branch): 5000 steps
  in 40 min (2393 s), plus 29 s caching; median step 0.473 s; peak GPU memory 1.05 GB
  allocated. The baseline needed 66 min and 4.17 GB.
- Effect of the largest-piece step on validation (7 cases, original grid):

| Model | Mean Dice, all pieces -> largest piece | Mean HD95, all pieces -> largest piece |
|---|---|---|
| Baseline | 0.948 -> 0.952 | 19.1 -> 3.8 mm |
| MONAI U-Net | 0.945 -> 0.949 | 20.8 -> 3.7 mm |

  The far pieces raised HD95 in 5 of 7 baseline cases, most in spleen_40 (98.7 -> 1.5 mm).
- Validation comparison, MONAI minus baseline, both with the step, paired bootstrap: Dice
  -0.0026 [-0.0088, +0.0034], HD95 -0.09 mm [-1.54, +1.77]. Both intervals contain 0: no
  detectable difference on validation.

## Consequences

- One spleen per scan is assumed: a real second spleen (an accessory spleen) would be erased.
- Validation numbers are optimistic: the 7 cases chose each checkpoint, and the idea for the
  post-processing came from them. The reported comparison is the test evaluation, once, for both
  models, in T07.
- The two models differ in several ways at once (depth, residual units, strided convolutions,
  activation): a difference between them could not be attributed to one of these.
- One run and one seed per model: a small difference could come from the random seed alone.
