from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from neural_state_machine.phase3c_diagnostics.manifest import (
    canonical_json_bytes,
    safe_path,
    sha256_file,
)


def test_canonical_json_is_sorted_compact_and_newline_terminated() -> None:
    assert canonical_json_bytes({"b": 2, "a": 1}) == b'{"a":1,"b":2}\n'


def test_safe_path_rejects_parent_escape(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="containment"):
        safe_path(tmp_path, "../escape.json")


def test_sha256_file_hashes_exact_bytes(tmp_path: Path) -> None:
    path = tmp_path / "x.bin"
    path.write_bytes(b"abc\x00")
    assert sha256_file(path) == hashlib.sha256(b"abc\x00").hexdigest()


def test_sha256_file_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_text("x", encoding="utf-8")
    link = tmp_path / "link"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlink unavailable")
    with pytest.raises(ValueError, match="regular file"):
        sha256_file(link)
