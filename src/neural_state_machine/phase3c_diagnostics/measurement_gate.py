from __future__ import annotations

from pathlib import Path

from .manifest import canonical_json_bytes, sha256_file

_AUXILIARY_GROUPS = ("reset_scores", "reverse_checks", "drain_scores")


def _expected_keys(manifest: dict[str, object]) -> dict[str, tuple[str, ...]]:
    value = manifest.get("auxiliary_keys")
    if not isinstance(value, dict) or set(value) != set(_AUXILIARY_GROUPS):
        raise RuntimeError("auxiliary key manifest is missing or invalid")
    resolved: dict[str, tuple[str, ...]] = {}
    for group in _AUXILIARY_GROUPS:
        rows = value.get(group)
        if not isinstance(rows, list) or any(not isinstance(row, str) or not row for row in rows):
            raise RuntimeError(f"auxiliary key manifest is invalid: {group}")
        keys = tuple(rows)
        if len(keys) != len(set(keys)):
            raise RuntimeError(f"auxiliary key manifest contains duplicates: {group}")
        resolved[group] = keys
    return resolved


def _observed_keys(payload: dict[str, object]) -> dict[str, tuple[str, ...]]:
    if set(payload) != set(_AUXILIARY_GROUPS):
        raise RuntimeError("auxiliary evidence groups differ from the sealed manifest")
    resolved: dict[str, tuple[str, ...]] = {}
    for group in _AUXILIARY_GROUPS:
        rows = payload.get(group)
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise RuntimeError(f"auxiliary evidence rows are invalid: {group}")
        keys: list[str] = []
        for row in rows:
            key = row.get("key")
            if not isinstance(key, str) or not key:
                raise RuntimeError(f"auxiliary evidence key is invalid: {group}")
            keys.append(key)
        if len(keys) != len(set(keys)):
            raise RuntimeError(f"auxiliary evidence contains duplicate keys: {group}")
        resolved[group] = tuple(keys)
    return resolved


def seal_auxiliary_artifact(
    attempt_dir: Path,
    manifest: dict[str, object],
    payload: dict[str, object],
) -> dict[str, str]:
    if not isinstance(attempt_dir, Path) or not attempt_dir.is_dir():
        raise ValueError("attempt_dir must be an existing directory")
    if not isinstance(manifest, dict) or not isinstance(payload, dict):
        raise ValueError("manifest and payload must be dictionaries")
    expected = _expected_keys(manifest)
    observed = _observed_keys(payload)
    for group in _AUXILIARY_GROUPS:
        if set(observed[group]) != set(expected[group]):
            raise RuntimeError(f"auxiliary evidence key grid differs: {group}")

    path = attempt_dir / "auxiliary.json"
    if path.exists() or path.is_symlink():
        raise FileExistsError("refusing to overwrite auxiliary.json")
    path.write_bytes(canonical_json_bytes(payload))
    return {"path": "auxiliary.json", "sha256": sha256_file(path)}
