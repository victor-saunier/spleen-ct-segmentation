# Model card: spleen segmentation on CT (baseline 3D U-Net)

Format after Mitchell et al., *Model Cards for Model Reporting*, 2019. Every number below comes
from a recorded run; the decision records in [docs/decisions](docs/decisions) give the details.

> **Research / portfolio project. NOT a medical device.** Not validated for clinical use. Do not
> use it for diagnosis, treatment or any decision about a patient.

## Model details

- **What it does:** labels every voxel of an abdominal CT scan as spleen or not spleen.
- **Architecture:** 3D U-Net written for this project, 4 levels (16, 32, 64, 128 channels),
  instance normalisation, LeakyReLU, 1,401,986 weights, two output scores per voxel (the larger
  one wins, no threshold).
- **Compared model:** MONAI's U-Net with residual units (4.8 M weights), trained the same way. It
  scored lower on the test set, so the baseline is the model that is exported and packaged.
- **Input:** one CT volume (NIfTI). It is reoriented, resampled to 1.5 x 1.5 x 2.0 mm, clipped to
  -40..180 HU and rescaled to [0, 1], then read in 96^3 windows (sliding window).
- **Output:** a mask on the scan's original grid. Only the largest connected piece is kept.
- **Formats:** PyTorch checkpoint; ONNX export (5.69 MB, equivalent within rounding); a CPU
  Docker image that runs the whole prediction.
- **Author and date:** Victor Saunier, 2026. Code: MIT licence.

## Intended use

- Learning and research: a documented, reproducible baseline for 3D organ segmentation.
- Comparing methods on the Medical Segmentation Decathlon spleen task, with the limits below.

**Out of scope:**

- Any clinical use, including triage, measurement for a report, or treatment planning.
- Scans unlike the training data: other hospitals or scanners, CT phases other than portal
  venous, non-CT images. Nothing here measures performance on them.
- Patients unlike the training population (see below), including children: the dataset paper
  does not state ages.
- Scans with no spleen (after splenectomy) or more than one spleen: the largest-piece step
  assumes exactly one and would keep only one piece.

## Training data

- [Medical Segmentation Decathlon](http://medicaldecathlon.com/), Task09 Spleen, CC BY-SA 4.0.
  Per the dataset paper: 61 portal venous phase CT scans of patients undergoing chemotherapy
  treatment for liver metastases at Memorial Sloan Kettering Cancer Center (New York). The spleen
  was segmented semi-automatically, then the contours were adjusted by an expert abdominal
  radiologist. Ages and inclusion criteria are not stated.
- Only the 41 labelled scans are used, split **by case** with a fixed seed: 27 training, 7
  validation, 7 test. The dataset has no patient IDs and its source study scanned patients before
  and after chemotherapy, so the same patient may appear in two parts of the split.
- Training: Dice + cross-entropy loss, AdamW (learning rate 1e-4), mixed precision, 5000 steps of
  8 patches of 96^3, with small rotation, zoom and intensity changes. 66 min on a laptop RTX 4060
  (8 GB). The validation set chose the checkpoint (best at step 4750), so validation scores are
  optimistic and are not reported as results.

## Evaluation

- **Data:** the 7 test cases, never read before the evaluation, scored once, with a procedure
  committed and pushed beforehand ([decision record 0006](docs/decisions/0006-test-evaluation.md)).
- **Metrics,** per case on the original grid, then averaged: Dice (overlap, 1 = perfect) and HD95
  (95th-percentile distance between the borders, in mm, lower is better). 95 % intervals:
  percentile bootstrap of the mean, 10,000 resamples.

| Model | Mean Dice [95 % interval] | Mean HD95 [95 % interval] |
|---|---|---|
| Baseline 3D U-Net (this model) | 0.950 [0.936, 0.963] | 4.0 mm [1.9, 7.2] |
| MONAI U-Net, residual units | 0.935 [0.907, 0.954] | 5.3 mm [3.1, 7.8] |

- Paired Dice difference, MONAI U-Net minus baseline: -0.015 [-0.029, -0.006]. The baseline scored
  higher on these 7 cases; HD95 showed no detectable difference.
- Without the largest-piece step, mean HD95 would be 19.5 mm (baseline), because of small false
  pieces far from the spleen.
- Speed of the network on one validation scan: 0.75 s on the laptop GPU, 13.7 s with PyTorch on
  the CPU, 36.3 s with ONNX Runtime on the CPU; about 21 s for a whole prediction in the Docker
  image on the CPU ([decision records 0007 and 0008](docs/decisions)).

## Known failures

Seen in the lowest-Dice test cases:

- **Leaks into touching organs:** on spleen_41, both models extend into bowel touching the tip of
  the spleen, up to 35 mm outside it. The largest-piece step cannot remove a leak that touches the
  spleen.
- **Small spleens:** on spleen_44 (119 mL, 10 slices of 5 mm) the MONAI U-Net misses 24 mL; the
  baseline gets the volume right but leaks into neighbouring tissue on one slice.
- **Far false pieces** (bowel, pelvis, muscle, arm) appear without post-processing; the
  largest-piece step removes them only because exactly one spleen is assumed.

## Limitations and ethical considerations

- 7 test cases from one hospital and one CT phase: the intervals are wide, and with so few cases
  the percentile bootstrap makes them narrower than they should be.
- One training run per model, one seed: part of the difference between models may be chance.
- Case-level split: repeat scans of one patient could make test scores optimistic.
- The population is patients on chemotherapy for liver metastases; the source study is about
  spleen volume increase under chemotherapy. Performance on other populations is unknown.
- No information on sex, age or ethnicity, so no subgroup analysis is possible.

## Licence and availability

- Code: MIT. Data: CC BY-SA 4.0, owned by the dataset authors, not redistributed here.
- **The trained weights are not released.** Whether weights trained on CC BY-SA data count as
  adapted material is not settled ([decision record 0009](docs/decisions/0009-releases.md)).
  Anyone can retrain with the commands in the README; results will vary between runs.

## Citation of the data

- Simpson et al., *A large annotated medical image dataset for the development and evaluation of
  segmentation algorithms*, [arXiv:1902.09063](https://arxiv.org/abs/1902.09063), 2019.
- Antonelli et al., *The Medical Segmentation Decathlon*, Nature Communications, 2022,
  [doi:10.1038/s41467-022-30695-9](https://doi.org/10.1038/s41467-022-30695-9).
