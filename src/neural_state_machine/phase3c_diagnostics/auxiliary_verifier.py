from __future__ import annotations

from pathlib import Path

from .manifest import load_json_object, sha256_file

_AUXILIARY_GROUPS = ("reset_scores", "reverse_checks", "drain_scores")


def _expected_auxiliary_keys(manifest: dict[str, object]) -> dict[str, tuple[str, ...]] | None:
    value = manifest.get("auxiliary_keys")
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != set(_AUXILIARY_GROUPS):
        raise RuntimeError("sealed manifest auxiliary key groups are invalid")
    resolved: dict[str, tuple[str, ...]] = {}
    for group in _AUXILIARY_GROUPS:
        rows = value.get(group)
        if not isinstance(rows, list) or any(not isinstance(row, str) or not row for row in rows):
            raise RuntimeError(f"sealed manifest auxiliary keys are invalid: {group}")
        keys = tuple(rows)
        if len(keys) != len(set(keys)):
            raise RuntimeError(f"sealed manifest auxiliary keys contain duplicates: {group}")
        resolved[group] = keys
    return resolved


def verify_auxiliary_artifact(
    manifest: dict[str, object],
    execution: dict[str, object],
    attempt_dir: Path,
) -> None:
    expected = _expected_auxiliary_keys(manifest)
    if expected is None:
        return
    metadata = execution.get("auxiliary")
    if not isinstance(metadata, dict) or set(metadata) != {"path", "sha256"}:
        raise RuntimeError("auxiliary artifact metadata is missing or invalid")
    relative = metadata.get("path")
    digest = metadata.get("sha256")
    if relative != "auxiliary.json" or not isinstance(digest, str) or len(digest) != 64:
        raise RuntimeError("auxiliary artifact path/hash metadata is invalid")
    path = attempt_dir / relative
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("auxiliary artifact must be a regular file")
    if sha256_file(path) != digest:
        raise RuntimeError("auxiliary artifact hash mismatch")
    payload = load_json_object(path)
    if set(payload) != set(_AUXILIARY_GROUPS):
        raise RuntimeError("auxiliary artifact groups are invalid")
    for group in _AUXILIARY_GROUPS:
        rows = payload.get(group)
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise RuntimeError(f"auxiliary artifact rows are invalid: {group}")
        observed: list[str] = []
        for row in rows:
            key = row.get("key")
            if not isinstance(key, str) or not key:
                raise RuntimeError(f"auxiliary artifact key is invalid: {group}")
            observed.append(key)
        if len(observed) != len(set(observed)) or set(observed) != set(expected[group]):
            raise RuntimeError(f"auxiliary artifact key grid differs: {group}")
