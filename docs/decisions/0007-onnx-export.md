# 0007 - ONNX export, equivalence and speed

- Status: accepted
- Date: 2026-11-22

## Context

The trained model is Python code plus a weights file, so running it needs PyTorch and this
project's code. For deployment, a portable copy of the network is useful: ONNX stores it as a
standard graph of operations that any ONNX runtime can execute (Python, C++, other languages),
without PyTorch. Exporting re-expresses every operation, so the copy must be proven faithful, and
its speed measured rather than assumed.

## Decisions

| Topic | Choice | Options rejected |
|---|---|---|
| Model exported | The baseline U-Net | The MONAI U-Net: about 5 times faster per window on the CPU, but it scored lower on the test set. |
| What is exported | The network only: 96^3 windows in, two scores per voxel out, any number of windows | The whole pipeline: preprocessing, sliding window and post-processing use MONAI and are not graph operations. |
| File | One `.onnx` file with the weights inside (5.69 MB) | The exporter's default separate `.onnx.data` file: two files to keep together. |
| Runtime | ONNX Runtime 1.32, CPU | GPU runtime: an extra, large dependency; PyTorch already runs on the GPU. |
| Equivalence | Compare final masks on the 7 validation cases, voxel by voxel | Raw scores only: small score differences matter only if they change the predicted class. Test cases: the test set is used once (decision record 0006). |
| Speed | Own benchmark: one window and one whole validation scan per backend, warm-up, 5 repeats, median, GPU synchronised | Timing once: includes start-up costs and noise. |

The choice of the baseline used the test result. That is acceptable for deciding what to ship,
and produced no new test numbers.

## Measurements

Laptop: Intel Core i9-13900HX (32 logical cores, PyTorch using 24 threads), NVIDIA RTX 4060
Laptop GPU (8 GB), Windows 11, plugged in, power mode "Hautes performances". PyTorch 2.15.0,
ONNX Runtime 1.32.0.

Equivalence (baseline, 7 validation cases, about 590,000 predicted spleen voxels in total):

| Comparison | Voxels with a different class |
|---|---|
| ONNX Runtime vs PyTorch, both on the CPU in 32-bit | 14 |
| ONNX Runtime vs the GPU predictions behind the reported results (mixed precision) | 68 |
| PyTorch CPU vs the same GPU predictions | 82 |

The differing voxels are near-ties, where the two class scores are almost equal and rounding
picks the winner. Dice against the labels is unchanged to 4 decimals in 6 of 7 cases (spleen_47:
0.9536 to 0.9535). The export is equivalent within rounding; the GPU's mixed precision changes
more voxels than the export does.

Speed (`benchmark.py`, validation case spleen_16, preprocessed volume 271 x 234 x 241, median of
5 runs after 2 warm-up runs):

| Backend | One 96^3 window | Whole scan, sliding window |
|---|---|---|
| PyTorch, GPU, mixed precision | 0.016 s | 0.75 s |
| PyTorch, CPU | 0.24 s | 13.7 s |
| ONNX Runtime, CPU | 0.34 s | 36.3 s |

- ONNX Runtime is slower than PyTorch on this CPU. It is kept for portability, not speed.
- Its gap grows on whole scans because the sliding window sends 4 windows at a time: measured
  separately, its time per window rises from 0.31 s (1 window) to 0.74 s (4 windows), against
  0.22 s to 0.29 s for PyTorch. The cause inside ONNX Runtime was not investigated.

## Consequences

- `predict.py --onnx` runs the exported file through exactly the same steps; it still needs
  PyTorch and MONAI for preprocessing and the sliding window. Removing them would mean rewriting
  those steps and proving them equivalent too.
- On a CPU, the baseline takes about 14 s (PyTorch) to 36 s (ONNX Runtime) per scan, plus file
  reading and preprocessing. A smaller sliding-window batch could help ONNX Runtime; not tried.
- Timings come from one laptop and one scan; other hardware will differ.
- Licence: the weights were trained on CC BY-SA 4.0 data. Whether a published model file counts
  as adapted material is decided before any release (T11).
