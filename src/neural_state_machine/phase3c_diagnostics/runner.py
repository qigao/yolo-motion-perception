from __future__ import annotations

from pathlib import Path

from . import runner_core as _core
from .auxiliary import registered_auxiliary_keys
from .auxiliary_verifier import verify_auxiliary_artifact
from .manifest import canonical_json_bytes, load_json_object, sha256_file
from .runner_core import *  # noqa: F403


def prepare_manifest(
    root: Path,
    output: Path,
    *,
    profile: str = _core.REGISTERED_PROFILE,
) -> tuple[Path, str]:
    manifest_path, _ = _core.prepare_manifest(root, output, profile=profile)
    payload = load_json_object(manifest_path)
    if payload.get("registered_valid_profile") is True:
        keys = registered_auxiliary_keys(_core._registered_config())
        payload["auxiliary_keys"] = {
            "reset_scores": list(keys.reset_score_keys),
            "reverse_checks": list(keys.reverse_check_keys),
            "drain_scores": list(keys.drain_score_keys),
        }
        manifest_path.write_bytes(canonical_json_bytes(payload))
        digest = sha256_file(manifest_path)
        (output / "manifest.sha256").write_text(digest + "\n", encoding="ascii")
        return manifest_path, digest
    return manifest_path, sha256_file(manifest_path)


def verify_attempt(manifest_path: Path, attempt_dir: Path) -> dict[str, object]:
    execution = _core.verify_attempt(manifest_path, attempt_dir)
    verify_auxiliary_artifact(load_json_object(manifest_path), execution, attempt_dir)
    return execution
