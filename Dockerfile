# CPU image that predicts spleen masks with spleenseg.predict.
#
# Build:  docker build -t spleenseg .
# Run:    see "Run with Docker" in README.md. Scans, model and output folder are mounted at run
#         time: no data and no model weights are inside the image.
#
# Two stages: the first installs the locked environment with uv, the second copies only that
# environment onto a clean Python image, without uv or any build tools.
# Base images are pinned by digest (a hash of their content): a tag can be moved, a digest not.

# ---- Stage 1: install the locked environment ----
FROM ghcr.io/astral-sh/uv:0.13.7-python3.12-trixie-slim@sha256:d4a752dd52f42e0c508bb0b6a912f15922bb32e42f66af30671c3fdc0d759f21 AS build

# Compile .pyc files now (faster start), copy files instead of linking from uv's cache, and use
# this image's Python: the final image has the same one at the same place.
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app

# The libraries first, in their own layer: a code change then rebuilds only the last layer, not
# the 1.2 GB of libraries. --locked: fail if uv.lock does not match pyproject.toml.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

# Then our package, installed as a normal (non-editable) package so src/ is not needed later.
COPY README.md LICENSE ./
COPY src ./src
RUN uv sync --locked --no-dev --no-editable

# ---- Stage 2: the image that runs ----
FROM python:3.12-slim-trixie@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b

# Run as an ordinary user, not root: fewer rights if anything goes wrong inside the container.
RUN useradd --create-home spleenseg
COPY --from=build /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"
USER spleenseg
WORKDIR /work

ENTRYPOINT ["python", "-m", "spleenseg.predict"]
CMD ["--help"]
