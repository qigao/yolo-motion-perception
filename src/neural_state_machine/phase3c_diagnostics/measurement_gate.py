from __future__ import annotations

from pathlib import Path


def seal_auxiliary_artifact(
    attempt_dir: Path,
    manifest: dict[str, object],
    payload: dict[str, object],
) -> dict[str, str]:
    del attempt_dir, manifest, payload
    return {}
