# Spleen CT Segmentation

3D segmentation of the spleen on abdominal CT scans with PyTorch and MONAI.

> **Research / portfolio project. NOT a medical device.**
> It is not validated for clinical use and must not be used for diagnosis or treatment decisions.

## Status

Work in progress. This README only reports numbers produced by real runs. Results will be
reported on the test set, once, after both models are trained.

## Problem

Given a 3D CT volume of the abdomen, predict for every voxel (3D pixel) whether it belongs to the
spleen. This is binary semantic segmentation.

## Data

- **Source:** [Medical Segmentation Decathlon](http://medicaldecathlon.com/), Task09 Spleen:
  61 CT volumes from Memorial Sloan Kettering Cancer Center (41 training + 20 testing).
  Only the 41 training volumes have labels, so only these are used here.
- **Labels:** per the dataset paper, the spleen was segmented semi-automatically, then the
  contours were adjusted by an expert abdominal radiologist.
- **Split:** 27 train / 7 validation / 7 test, **by case**, with a fixed seed, committed in
  [splits/split.json](splits/split.json) before any training.
  The dataset ships no patient IDs, and its source study imaged patients before and after
  chemotherapy, so a patient-level split **cannot be guaranteed**: two cases might be the
  same person. See [decision record 0001](docs/decisions/0001-case-level-split.md).
- **License:** [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), as stated on the
  official site. The data is **not** included in this repository.
- **Citation:**
  - Simpson et al., *A large annotated medical image dataset for the development and evaluation
    of segmentation algorithms*, [arXiv:1902.09063](https://arxiv.org/abs/1902.09063), 2019.
  - Antonelli et al., *The Medical Segmentation Decathlon*, Nature Communications, 2022,
    [doi:10.1038/s41467-022-30695-9](https://doi.org/10.1038/s41467-022-30695-9).

## Method

- **Preprocessing** (all cases): RAS orientation, resampling to 1.5 x 1.5 x 2.0 mm, intensity
  window -40 to 180 HU rescaled to [0, 1], crop to the body. Values come from statistics of the
  training cases only, see [decision record 0002](docs/decisions/0002-preprocessing.md).
- **Training input:** 96^3 patches, centred on spleen or body tissue with equal probability,
  with small rotation, zoom and intensity augmentations; no left-right flip.
- **Baseline model:** a small 3D U-Net written from scratch (1.4 M weights), two output
  scores per voxel, prediction = the larger one (no threshold).
- **Training** (same for every model): Dice + cross-entropy loss, AdamW, mixed precision,
  8 patches per step, 5000 steps. Every 250 steps the validation Dice picks the best
  checkpoint. Budget and batch were fixed from measured speed and GPU memory, not from
  validation scores. The baseline trains in about 66 min on a laptop RTX 4060 (8 GB). See
  [decision record 0004](docs/decisions/0004-training-protocol.md).
- **Prediction:** from the image only; the mask is put back on the scan's original grid by
  undoing the preprocessing, then evaluated there.
- **Evaluation:** per case, at the original resolution: Dice (overlap) and HD95 (95th-percentile
  border distance in mm, MONAI's definition). Mean, standard deviation and a 95 % percentile
  bootstrap interval; models are compared with a bootstrap of their per-case differences. An
  empty prediction gives Dice 0 and an undefined HD95, counted separately. The test set is used
  once. See [decision record 0003](docs/decisions/0003-evaluation.md).

## Results

None yet.

## Reproduce

```bash
uv sync                               # install the exact locked environment
uv run python -m spleenseg.download   # ~1.6 GB into data/ (git-ignored), MD5-verified
uv run python -m spleenseg.split      # rewrites splits/split.json; must equal the committed file
uv run python -m spleenseg.stats      # training-set statistics behind the preprocessing choices
uv run python -m spleenseg.preview    # outputs/preview_<case>.png, one training case
uv run python -m spleenseg.train --model baseline --power-note "plugged in"   # ~1 h on GPU
uv run python -m spleenseg.predict --run-dir outputs/runs/<run>   # validation cases
uv run python -m spleenseg.evaluate --pred-dir outputs/predictions/<run>   # validation by default
uv run pytest                         # data checks run only when data/ is present
```

## Limitations

- **Case-level split:** separating patients cannot be guaranteed (see Data above). If some
  cases are repeat scans of one person, test scores could be optimistic.
- More to be written as results come in.

## License

- Code: MIT, see [LICENSE](LICENSE).
- Data: CC BY-SA 4.0, owned by the dataset authors, not redistributed here.
