from __future__ import annotations

import json
from pathlib import Path


def canonical_json_bytes(payload: object) -> bytes:
    return b"{}\n"


def safe_path(root: Path, relative: str) -> Path:
    return root / relative


def sha256_file(path: Path) -> str:
    return ""
