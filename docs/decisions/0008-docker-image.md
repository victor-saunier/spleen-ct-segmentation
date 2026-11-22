# 0008 - Docker image for prediction

- Status: accepted
- Date: 2026-11-22

## Context

Running the model needs Python 3.12, PyTorch, MONAI and their dependencies at exact versions. `uv.lock` fixes the Python libraries, but not the operating system or the system
libraries underneath them. A Docker image freezes all of it: anyone with Docker runs the same
Linux, the same Python and the same libraries, without installing them, and CI can run the same
image later (T10).

The image has one job: predict spleen masks for CT files given on the command line, on the CPU.

## Decisions

| Topic | Choice | Options rejected |
|---|---|---|
| What runs inside | `spleenseg.predict` with PyTorch and MONAI, as on the laptop | ONNX Runtime only: a smaller image, but preprocessing, sliding window and the return to the original grid use MONAI and PyTorch, so they would have to be rewritten and proven equivalent again. |
| Hardware | CPU only (Linux uses the CPU PyTorch index already) | GPU image: needs the NVIDIA container toolkit on the host and a much larger CUDA PyTorch. |
| Inputs | A new `--image` option: predict the files given, not a split | Mounting the whole dataset and `splits/split.json`: a user has scan files, not this split. |
| Model weights and data | Not in the image: scans, model folder and output folder are mounted when the container starts | Weights inside the image: whether a published model file counts as adapted CC BY-SA material is decided before any release (T11). |
| Build | Two stages: uv installs the locked environment (`uv sync --locked --no-dev`), then only that environment is copied onto a clean `python:3.12-slim-trixie` | One stage: uv and its cache would stay in the image. |
| Layers | Libraries installed before our code is copied | Copying everything first: every code change would reinstall the libraries. |
| Base images | Pinned by digest (a hash of the content) | Tags only: a tag can be moved to different content. |
| User | An ordinary user `spleenseg`, not root | root: more rights than prediction needs. |
| Build context | `.dockerignore` allow-list: only `pyproject.toml`, `uv.lock`, `README.md`, `LICENSE`, `src/` | A deny-list: a forgotten folder (data, outputs) could be sent into the build. |

## Measurements

Laptop i9-13900HX, Windows 11, Docker Desktop (engine 29.9.3), plugged in.

- Size: `docker image ls` shows 450 MB content size (compressed layers) and 2.03 GB disk usage.
  The disk usage counts the compressed layers plus their unpacked copy (about 1.6 GB). The
  compiled `.pyc` files add about 211 MB; they are kept so the container does not compile them at
  every start.
- First build: about 68 s.
- Equivalence: on validation case spleen_16, the mask from the container is identical, voxel for
  voxel, to the mask from `predict.py` run on the laptop's CPU (Dice 0.9495 against the label).
- Time: about 21 s for that scan, from `docker run` to the saved mask (start, model loading,
  preprocessing, CPU prediction, saving).

## Consequences

- One command runs the model on any machine with Docker; the user mounts the scans and a model
  folder (`best.pt` + `summary.json`).
- The image is large because of PyTorch: the ONNX-only rewrite stays the way to shrink it, with
  its own equivalence proof.
- On a Linux host, the container's user may not be allowed to write into the mounted output
  folder; running with `--user "$(id -u):$(id -g)"` fixes that. Tested on Windows only.
- Real data: checked on one validation scan, on one machine. Since T10, CI builds the image on
  every pull request and requires its mask to equal `predict.py`'s on a synthetic scan with
  random weights (`tests/test_docker.py`), on Linux with `--user`.
