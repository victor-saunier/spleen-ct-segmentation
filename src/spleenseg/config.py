"""Training configuration: the shared protocol plus one model section, read from a TOML file."""

import tomllib
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CONFIG = Path("configs/train.toml")


@dataclass(frozen=True)
class TrainConfig:
    # [shared]: identical for every model
    seed: int
    patch_size: tuple[int, int, int]
    samples_per_case: int
    cases_per_batch: int
    learning_rate: float
    weight_decay: float
    max_steps: int
    val_every_steps: int
    num_workers: int
    amp: bool
    # [models.<name>]
    model: str
    architecture: str
    channels: tuple[int, ...]


def load_config(path: Path, model: str) -> TrainConfig:
    """Shared section + the named model section. Unknown or missing keys raise an error."""
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    shared = dict(raw["shared"])
    shared["patch_size"] = tuple(shared["patch_size"])
    section = raw["models"][model]
    return TrainConfig(
        **shared,
        model=model,
        architecture=section["architecture"],
        channels=tuple(section["channels"]),
    )
