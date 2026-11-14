# 0002 - Preprocessing and augmentation

- Status: accepted
- Date: 2026-11-14

## Context

The network sees arrays of voxels, not millimetres or Hounsfield units, and a whole CT does not fit
through a 3D network on the 8 GB laptop GPU. Statistics of the **27 training cases only**
(`uv run python -m spleenseg.stats`; validation and test cases are never read for this):

| Quantity | Value |
|---|---|
| In-plane voxel size | 0.61 to 0.98 mm (median 0.79) |
| Slice thickness | 1.5 to 7.5 mm (median 5.0) |
| Largest spleen (bounding box) | 117 x 132 x 142.5 mm |
| Spleen share of the voxels | 0.08 % to 1.33 % (median 0.39 %) |
| Spleen intensity percentiles | 0.5th: -37 HU, 50th: 103 HU, 99.5th: 176 HU |

## Decisions

| Step | Choice | Options rejected |
|---|---|---|
| Voxel size | Resample to 1.5 x 1.5 x 2.0 mm (image: linear, mask: nearest neighbour) | Median spacing 0.79 x 0.79 x 5.0 mm (nnU-Net's rule): a 96^3 patch would only cover 76 x 76 mm in-plane, less than a spleen. 1 mm isotropic: about 4.5 times more voxels per case. |
| Intensities | Clip to -40 to 180 HU, rescale to [0, 1] | Raw HU: values in the thousands, with most of the range spent on air and bone. |
| Crop | Box around everything above -40 HU | No crop: more memory spent on air. |
| Patches | 96^3, 4 per case, centred on spleen or on body tissue with equal probability | Uniform random patches: with 0.39 % spleen, most would hold none. |
| Augmentations | Rotation up to 15 degrees around the head-feet axis, zoom up to 10 %, intensity scale and shift up to 10 %, each with probability 0.5 | Left-right flip (used by nnU-Net): creates a spleen on the wrong side. Elastic deformation: slower, harder to explain. |

## Why

- **1.5 x 1.5 x 2.0 mm:** a 96^3 patch then covers 144 x 144 x 192 mm, enough to hold the largest
  training spleen. The same spacing is used in MONAI's spleen tutorial.
- **-40 to 180 HU:** the 0.5th to 99.5th percentile of spleen intensities in the training cases,
  rounded outward to 10 HU. nnU-Net uses the same percentile rule for CT. MONAI's tutorial uses
  -57 to 164 HU, which is close.
- **Augmentations imitate real variation only:** patient position and body size, scanner and
  contrast timing. The strengths are judgement calls, not tuned values.
- **Padding guard:** after preprocessing, the shallowest training case is exactly 96 voxels deep.
  Anything smaller is padded instead of crashing.

## Consequences

- Both models use exactly these transforms, so their comparison stays fair.
- In-plane detail is lost (0.79 mm voxels become 1.5 mm). Thick slices are interpolated, which
  adds voxels but no information.
- The crop is coarse: the CT table, above -40 HU, stays inside the box.
- Preprocessing the 27 training cases took about 51 s on the development laptop, so training
  should cache the deterministic part.
- Any later change to these values may only be judged on the validation set, never on the test set.
