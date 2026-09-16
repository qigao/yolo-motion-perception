from __future__ import annotations

import shutil
from pathlib import Path

from . import runner_core as _core
from .auxiliary import registered_auxiliary_keys
from .auxiliary_evidence import original_auxiliary_evidence, reference_reset_evidence
from .auxiliary_verifier import verify_auxiliary_artifact
from .evaluation import build_evaluation_bundles
from .manifest import canonical_json_bytes, load_json_object, sha256_file
from .measurement_gate import seal_auxiliary_artifact
from .replay import run_diagnostic_replay
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


def _registered_auxiliary_payload(*, attempt_id: str) -> dict[str, object]:
    config = _core._registered_config()
    bundles = build_evaluation_bundles(config)
    reset_scores: list[dict[str, object]] = []
    reverse_checks: list[dict[str, object]] = []
    drain_scores: list[dict[str, object]] = []

    for model in _core.profile_model_ids(_core.REGISTERED_PROFILE):
        if model.family == "original":
            replay = run_diagnostic_replay(
                model.seed,
                str(model.arm),
                config,
                str(model.condition),
                attempt_id=attempt_id,
            )
            evidence = original_auxiliary_evidence(model, replay, bundles, config)
            reset_scores.extend(evidence.reset_scores)
            reverse_checks.extend(evidence.reverse_checks)
            drain_scores.extend(evidence.drain_scores)
            continue
        if model.family != "reference":
            continue
        if model.reference_kind == "supervised_ridge":
            reference = _core._train_ridge_reference(model.seed, config)
        elif model.reference_kind == "immediate_identified":
            reference = _core._train_immediate_reference(model.seed, config)
        elif model.reference_kind == "source_visible_delayed":
            reference = _core._train_source_visible_reference(model.seed, config)
        else:
            raise RuntimeError(f"unsupported reference kind: {model.reference_kind}")
        reset_scores.extend(reference_reset_evidence(model, reference, bundles, config))

    return {
        "reset_scores": reset_scores,
        "reverse_checks": reverse_checks,
        "drain_scores": drain_scores,
    }


def _mark_staged_invalid(staged_attempt: Path, exc: Exception) -> None:
    execution_path = staged_attempt / "execution-manifest.json"
    if not execution_path.is_file():
        return
    execution = load_json_object(execution_path)
    execution["complete"] = False
    execution["diagnostic_valid"] = False
    execution["first_failure"] = {
        "type": type(exc).__name__,
        "message": str(exc),
    }
    execution_path.write_bytes(canonical_json_bytes(execution))


def run_registered_measurement(
    root: Path,
    manifest_path: Path,
    *,
    expected_manifest_sha256: str,
    attempt_id: str,
    approve_measurement: bool,
) -> Path:
    final_attempt = manifest_path.parent / f"attempt-{attempt_id}"
    if final_attempt.exists() or final_attempt.is_symlink():
        raise FileExistsError("attempt directory already exists")
    staging = manifest_path.parent / f".staging-{attempt_id}"
    if staging.exists() or staging.is_symlink():
        raise FileExistsError("staging directory already exists")
    staging.mkdir(parents=True)
    staged_manifest = staging / "manifest.json"
    staged_manifest.write_bytes(manifest_path.read_bytes())
    staged_attempt = staging / f"attempt-{attempt_id}"

    try:
        _core.run_registered_measurement(
            root,
            staged_manifest,
            expected_manifest_sha256=expected_manifest_sha256,
            attempt_id=attempt_id,
            approve_measurement=approve_measurement,
        )
        manifest = load_json_object(staged_manifest)
        auxiliary_payload = _registered_auxiliary_payload(attempt_id=attempt_id)
        auxiliary_metadata = seal_auxiliary_artifact(
            staged_attempt,
            manifest,
            auxiliary_payload,
        )
        execution_path = staged_attempt / "execution-manifest.json"
        execution = load_json_object(execution_path)
        execution["auxiliary"] = auxiliary_metadata
        execution_path.write_bytes(canonical_json_bytes(execution))
        verify_attempt(staged_manifest, staged_attempt)
        staged_attempt.replace(final_attempt)
        shutil.rmtree(staging)
        return final_attempt
    except Exception as exc:
        if staged_attempt.exists() and not final_attempt.exists():
            _mark_staged_invalid(staged_attempt, exc)
            staged_attempt.replace(final_attempt)
        shutil.rmtree(staging, ignore_errors=True)
        raise
