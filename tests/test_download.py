"""Tests for the download script. No network: a local file:// URL stands in for AWS."""

import hashlib
import io
import tarfile
from pathlib import Path

import pytest

from spleenseg.download import download, extract, file_hash


def _make_source(tmp_path: Path) -> tuple[Path, str]:
    source = tmp_path / "source.tar"
    source.write_bytes(b"fake archive content")
    return source, hashlib.md5(b"fake archive content").hexdigest()


def test_file_hash_matches_hashlib(tmp_path: Path) -> None:
    source, md5 = _make_source(tmp_path)
    assert file_hash(source, "md5") == md5


def test_download_keeps_file_when_checksum_matches(tmp_path: Path) -> None:
    source, md5 = _make_source(tmp_path)
    target = tmp_path / "out" / "archive.tar"

    download(source.as_uri(), target, md5)

    assert target.read_bytes() == b"fake archive content"
    assert not (tmp_path / "out" / "archive.tar.part").exists()


def test_download_rejects_wrong_checksum_and_leaves_nothing(tmp_path: Path) -> None:
    source, _ = _make_source(tmp_path)
    target = tmp_path / "out" / "archive.tar"

    with pytest.raises(ValueError, match="Checksum mismatch"):
        download(source.as_uri(), target, "0" * 32)

    # Neither the final file nor the partial one may remain: nothing can pass as "downloaded".
    assert list((tmp_path / "out").iterdir()) == []


def _tar_with_member(path: Path, name: str) -> None:
    with tarfile.open(path, "w") as tar:
        info = tarfile.TarInfo(name)
        info.size = 5
        tar.addfile(info, io.BytesIO(b"hello"))


def test_extract_writes_regular_members(tmp_path: Path) -> None:
    archive = tmp_path / "ok.tar"
    _tar_with_member(archive, "Task09_Spleen/dataset.json")

    extract(archive, tmp_path / "out")

    assert (tmp_path / "out" / "Task09_Spleen" / "dataset.json").read_bytes() == b"hello"


def test_extract_refuses_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "evil.tar"
    _tar_with_member(archive, "../evil.txt")

    with pytest.raises(tarfile.OutsideDestinationError):
        extract(archive, tmp_path / "out")

    assert not (tmp_path / "evil.txt").exists()
