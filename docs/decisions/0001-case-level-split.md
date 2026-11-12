# 0001 - Case-level hold-out split

- Status: accepted
- Date: 2026-11-12

## Context

The 41 labelled cases of MSD Task09 Spleen must be split into train / validation / test without
leakage: the test set is used once, at the very end, and nothing about it may influence
training or model selection.

The goal was a **patient-level** split, because two scans of one person share their anatomy and
a model could "recognise" the person. Checked on 2026-11-12:

- The dataset paper (Simpson et al., arXiv:1902.09063) reports 61 portal venous phase CT scans
  of patients treated with chemotherapy for liver metastases. It gives **no patient count** and
  does not say whether each scan is a different patient.
- The files carry no patient ID: cases are named `spleen_N`.
- The source study it cites (PubMed 25617913, J Am Coll Surg 2015) measured spleen volume
  **before and after** 6 months of chemotherapy, so its patients were scanned more than once.
  Whether the 61 released scans include such pairs is unknown.

Inspection of the downloaded files (2026-11-12):

- `dataset.json` holds only the name, description, reference (MSKCC), licence (CC-BY-SA 4.0),
  release date and the image lists: no patient field.
- The NIfTI text header fields (`descrip`, `db_name`, `aux_file`, `intent_name`) are empty, or
  hold the same value (`5.0.10`) in all 41 images: nothing to group cases by.
- No two of the 41 images are identical (checked by `tests/test_split_file.py`).
- The archive also contains macOS metadata files (`._spleen_N.nii.gz`). The code takes the case
  list from `dataset.json`, never from a folder listing, so they are ignored.

## Options

1. **Case-level split, limitation documented.** Guarantees that no case is in two sets, checks
   for identical images, and states clearly that patients may not be separated.
2. **Case-level split plus a similarity screen** to flag pairs of scans that might be the same
   person. Heuristic: it can flag suspects but never prove there are none. Adds a dependency
   and several hours.
3. **Switch to a dataset that ships patient IDs.** The only way to guarantee patient-level
   separation, at the cost of redoing the license check and the data work.

Also considered: **k-fold cross-validation** instead of one hold-out split. More reliable on small
data, but it multiplies the training cost by k for each of the two models, which is too much
for a single laptop GPU.

## Decision

Option 1: a case-level hold-out split, 27 train / 7 validation / 7 test, seed 42, written to
`splits/split.json` and committed before any training.

## Why

- It is the strongest claim that can be made honestly with this data.
- The committed file is the source of truth: it does not depend on `random.shuffle` behaving the
  same in future Python versions, and git history shows it was fixed before any result existed.
- 7 test cases is a compromise: more would narrow the confidence interval, fewer would leave
  more data for training.

## Consequences

- Every document calls the split **case-level**, never patient-level.
- If some cases are repeat scans of one person across sets, test scores could be optimistic.
  The README and the model card state this limitation.
- `tests/test_split_file.py` checks that no case is in two sets, that the sizes are right, that
  the file matches the split code, and (locally, with the data) that only labelled cases are
  used and no image is duplicated. It does not, and cannot, check patients.
