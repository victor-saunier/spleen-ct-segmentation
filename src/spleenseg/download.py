"""Download and extract the Medical Segmentation Decathlon Task09 Spleen dataset.

Usage (from the repository root):
    uv run python -m spleenseg.download

Data license: CC BY-SA 4.0 (http://medicaldecathlon.com/). The data is never committed.
"""

import argparse
import hashlib
import json
import tarfile
import urllib.request
from pathlib import Path

# Link given on the official "Direct from AWS" page of medicaldecathlon.com.
URL = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar"
# MD5 published by MONAI (monai/apps/datasets.py): a source independent of the download host.
EXPECTED_MD5 = "410d4a301da4e5b2f6f86ec3ddba524e"
DEFAULT_DATA_DIR = Path("data")


def file_hash(path: Path, algorithm: str) -> str:
    """Return the hex digest of a file, read in 1 MB chunks so it never sits fully in memory."""
    digest = hashlib.new(algorithm)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _print_progress(blocks: int, block_size: int, total_size: int) -> None:
    if total_size > 0:
        percent = min(100, blocks * block_size * 100 // total_size)
        print(f"\r  {percent:3d} %", end="", flush=True)


def download(url: str, target: Path, expected_md5: str) -> Path:
    """Download `url` to `target` and keep it only if its MD5 matches `expected_md5`."""
    if target.exists():
        if file_hash(target, "md5") != expected_md5:
            raise ValueError(f"{target} exists but its checksum is wrong: delete it and rerun.")
        print(f"Already downloaded and verified: {target}")
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    # Download under a temporary name, so an interrupted download never looks complete.
    partial = target.with_name(target.name + ".part")
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, partial, reporthook=_print_progress)
    print()

    actual_md5 = file_hash(partial, "md5")
    if actual_md5 != expected_md5:
        partial.unlink()
        raise ValueError(f"Checksum mismatch: expected MD5 {expected_md5}, got {actual_md5}.")
    partial.replace(target)
    print(f"MD5 verified. SHA-256 (for the record): {file_hash(target, 'sha256')}")
    return target


def extract(archive: Path, out_dir: Path) -> None:
    """Extract a tar archive into `out_dir`, refusing any member that would land outside it."""
    with tarfile.open(archive) as tar:
        # The "data" filter rejects absolute paths, "../" tricks, links leaving out_dir, and
        # device files: a malicious archive cannot write anywhere else on the disk.
        tar.extractall(out_dir, filter="data")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args(argv)

    archive = download(URL, args.data_dir / "Task09_Spleen.tar", EXPECTED_MD5)
    dataset_dir = args.data_dir / "Task09_Spleen"
    if not (dataset_dir / "dataset.json").exists():
        print(f"Extracting to {args.data_dir}")
        extract(archive, args.data_dir)

    meta = json.loads((dataset_dir / "dataset.json").read_text(encoding="utf-8"))
    print(f"Ready: {len(meta['training'])} labelled cases, {len(meta['test'])} unlabelled.")


if __name__ == "__main__":
    main()
