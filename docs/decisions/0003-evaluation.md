# 0003 - Evaluation protocol

- Status: accepted
- Date: 2026-11-15

## Context

The metrics and the evaluation script are written before any model exists, so they cannot be
bent to fit a result. Training will select its checkpoint with the same Dice code, on the
validation set. The test set has 7 cases, so every mean comes with a lot of uncertainty.

## Decisions

| Topic | Choice | Options rejected |
|---|---|---|
| Overlap metric | Dice = 2 x overlap / (prediction + truth), in voxels | Accuracy: "background everywhere" is 99.6 % accurate on this data and finds no spleen. |
| Border metric | HD95: the larger of the two directed 95th percentiles of border-to-border distances, in mm, using the label's voxel size | Maximum Hausdorff distance: one stray voxel dominates it. Distances in voxels: a voxel is 0.6 to 7.5 mm here. |
| HD95 definition | Same as MONAI's `compute_hausdorff_distance(percentile=95)`, checked by a test | Pooling both directions into one percentile: a different number, harder to compare. |
| Empty prediction | Dice = 0. HD95 is undefined (NaN), and the number of such cases is reported | A penalty distance (e.g. the image diagonal): an invented number that moves the mean. |
| Where to compare | At the original resolution, against the original label files | In the resampled 1.5 x 1.5 x 2 mm space: the score would depend on our own preprocessing. |
| Uncertainty | Percentile bootstrap, 10,000 resamples, seed 0, 95 % interval of the mean | t-interval: assumes normal scores, which Dice often isn't. BCa: better with few cases, harder to explain. |
| Model comparison | Bootstrap of the per-case differences on the same cases | Comparing two separate intervals: ignores that both models see the same cases. |
| Spread | Standard deviation with ddof = 1 (sample estimate) | ddof = 0: underestimates the spread of 7 cases. |

## Why

- Dice and HD95 see different errors: Dice measures the bulk of the overlap, HD95 how far the
  borders are apart. HD95 ignores the worst 5 % of border distances on purpose, so a small
  distant false blob (under 5 % of the predicted border) is invisible to both metrics. The error
  analysis therefore looks at the masks themselves, and keeping only the largest connected
  component is a post-processing option to judge on the validation set.
- Our own short implementation makes every step explainable. The test against MONAI makes sure
  it computes the standard definition.
- Every summary records the command and the git commit (marked `+dirty` if tracked files were
  edited), so each published number can be traced to the code that produced it.

## Consequences

- Predictions must be resampled back to the original grid before evaluation. The script refuses a
  prediction whose shape or position differs from the label.
- The default split is validation. The test split must be named explicitly (`--split test`) and
  is used once, for the final numbers.
- With 7 cases, the percentile bootstrap interval is too narrow. Results will say so.
- HD95 averages only cover cases where it is defined. The count of undefined cases is always
  reported next to them.
