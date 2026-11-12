# Spleen CT Segmentation

3D segmentation of the spleen on abdominal CT scans with PyTorch and MONAI.

> **Research / portfolio project. NOT a medical device.**
> It is not validated for clinical use and must not be used for diagnosis or treatment decisions.

## Status

Work in progress. This README only reports numbers produced by real runs; there are none yet.

## Problem

Given a 3D CT volume of the abdomen, predict for every voxel (3D pixel) whether it belongs to the
spleen. This is binary semantic segmentation.

## Data

- **Source:** [Medical Segmentation Decathlon](http://medicaldecathlon.com/), Task09 Spleen:
  61 CT volumes from Memorial Sloan Kettering Cancer Center (41 training + 20 testing).
  Only the 41 training volumes have labels, so only these are used here, split into
  train / validation / test by patient.
- **License:** [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), as stated on the
  official site. The data is **not** included in this repository.
- **Citation:**
  - Simpson et al., *A large annotated medical image dataset for the development and evaluation
    of segmentation algorithms*, [arXiv:1902.09063](https://arxiv.org/abs/1902.09063), 2019.
  - Antonelli et al., *The Medical Segmentation Decathlon*, Nature Communications, 2022,
    [doi:10.1038/s41467-022-30695-9](https://doi.org/10.1038/s41467-022-30695-9).

## Method

To be written.

## Results

None yet.

## Reproduce

To be written.

## Limitations

To be written.

## License

- Code: MIT, see [LICENSE](LICENSE).
- Data: CC BY-SA 4.0, owned by the dataset authors, not redistributed here.
