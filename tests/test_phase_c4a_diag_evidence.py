from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import pytest

from neural_state_machine.phase_c4a_diagnostics.evidence import (
    prepare_attribution_manifest,
    validate_attribution_manifest,
    verify_attribution_stage,
    write_attribution_bundle,
)


ROOT = Path(__file__).resolve().parents[1]
_STAGE = "c4a-failure-attribution"
_MANIFEST_KEYS = {
    "schema_version",
    "stage",
    "scientific_head",
    "scientific_hashes",
    "frozen_c4a",
    "formal_head",
    "environment",
    "registered_seeds",
    "registered_scores",
    "diagnostic_config",
    "permutation_lineage",
    "secondary_evaluation_manifest",
    "expected_result_keys",
}
_RESULT_KEYS = {
    "schema_version",
    "stage",
    "integrity_valid",
    "geometry",
    "registered_attribution",
    "permutation_rows",
    "permutation_summaries",
    "mechanism_table",
}
_MECHANISMS = (
    "design_geometry",
    "task_relevant_target_projection",
    "bias_action_shortcut",
    "local_block_structure",
    "evaluation_fixture_sensitivity",
    "unresolved_multiple_mechanisms",
)


def _canonical_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _load(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    assert path.read_bytes() == _canonical_bytes(payload)
    return payload


def _mechanism_table() -> list[dict[str, object]]:
    return [
        {
            "mechanism": mechanism,
            "integrity_valid": True,
            "observed_association": f"observed {mechanism}",
            "matched_counterfactual_evidence": f"matched {mechanism}",
            "interpretation": f"interpretation {mechanism}",
        }
        for mechanism in _MECHANISMS
    ]


def _result_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "stage": _STAGE,
        "integrity_valid": True,
        "geometry": [{"seed": 7, "rank": 42}],
        "registered_attribution": [{"seed": 7, "normal": 200, "shuffled": 147}],
        "permutation_rows": [{"seed": 7, "mode": "block10", "replicate": 0}],
        "permutation_summaries": [{"seed": 7, "mode": "block10", "count": 32}],
        "mechanism_table": _mechanism_table(),
    }


def test_prepare_manifest_is_canonical_registered_and_result_free(tmp_path: Path):
    manifest_path = prepare_attribution_manifest(ROOT, tmp_path)
    payload = validate_attribution_manifest(ROOT, manifest_path)

    assert manifest_path == tmp_path / "manifest.json"
    assert set(payload) == _MANIFEST_KEYS
    assert payload["schema_version"] == 1
    assert payload["stage"] == _STAGE
    assert isinstance(payload["scientific_head"], str)
    assert len(payload["scientific_head"]) == 40
    assert payload["frozen_c4a"] == {
        "scientific_head": "8ae3154950ed53c4d0a0f555463042ff72674d31",
        "manifest_sha256": "a990745cf6f0e8c5fd7ae5d9a7189737b30a40168a4db3df1a6d3043d6d5876a",
        "result_sha256": "7f398f474dba928d2a6aa993d0801563aa05af20faa8cd9a61ec94a6ca4697a4",
        "provenance_sha256": "8fb47b5e670754bc44c9ea70dd4cedce40e6f52165923142114577b4aec00351",
    }
    assert payload["formal_head"] == "8b2180ed24b6ff03db4b927ec29cdd9903b0ccac"
    assert payload["registered_seeds"] == [7, 17, 29]
    assert payload["registered_scores"] == [
        {"seed": 7, "normal": 200, "reset": 100, "shuffled": 147},
        {"seed": 17, "normal": 200, "reset": 100, "shuffled": 170},
        {"seed": 29, "normal": 200, "reset": 100, "shuffled": 160},
    ]
    assert payload["diagnostic_config"] == {
        "permutation_replicates": 32,
        "modes": ["block10", "global"],
        "near_zero_margin": 1e-9,
    }
    assert payload["permutation_lineage"] == 0x43344144
    assert len(payload["secondary_evaluation_manifest"]) == 27
    assert payload["expected_result_keys"] == [
        "schema_version",
        "stage",
        "integrity_valid",
        "geometry",
        "registered_attribution",
        "permutation_rows",
        "permutation_summaries",
        "mechanism_table",
    ]
    environment = payload["environment"]
    assert isinstance(environment, dict)
    assert environment["python"] == platform.python_version()
    assert environment["implementation"] == platform.python_implementation()
    assert isinstance(payload["scientific_hashes"], dict)
    assert payload["scientific_hashes"]
    assert all(len(value) == 64 for value in payload["scientific_hashes"].values())
    assert list(tmp_path.iterdir()) == [manifest_path]
    assert manifest_path.read_bytes() == _canonical_bytes(payload)

    flattened = json.dumps(payload).lower()
    assert "c4b" not in flattened
    assert "behavior_passed" not in flattened
    assert "operator_passed" not in flattened
    assert "outcome" not in flattened


def test_prepare_manifest_refuses_overwrite_and_symlink(tmp_path: Path):
    prepare_attribution_manifest(ROOT, tmp_path)
    with pytest.raises(FileExistsError, match="overwrite"):
        prepare_attribution_manifest(ROOT, tmp_path)

    other = tmp_path / "other"
    other.mkdir()
    target = other / "target.json"
    target.write_text("{}\n", encoding="utf-8")
    symlink_root = tmp_path / "symlink-case"
    symlink_root.mkdir()
    try:
        (symlink_root / "manifest.json").symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are unavailable on this platform")
    with pytest.raises(FileExistsError, match="overwrite"):
        prepare_attribution_manifest(ROOT, symlink_root)


def test_manifest_validation_fails_closed_on_binding_or_environment_mutation(tmp_path: Path):
    manifest_path = prepare_attribution_manifest(ROOT, tmp_path)
    payload = _load(manifest_path)

    mutations = (
        ("frozen_c4a", "manifest_sha256", "0" * 64),
        ("formal_head", None, "0" * 40),
        ("permutation_lineage", None, 123),
    )
    for key, nested, value in mutations:
        candidate = json.loads(json.dumps(payload))
        if nested is None:
            candidate[key] = value
        else:
            candidate[key][nested] = value
        manifest_path.write_bytes(_canonical_bytes(candidate))
        with pytest.raises(ValueError):
            validate_attribution_manifest(ROOT, manifest_path)

    candidate = json.loads(json.dumps(payload))
    candidate["environment"]["python"] = "0.0.0"
    manifest_path.write_bytes(_canonical_bytes(candidate))
    with pytest.raises(ValueError, match="environment"):
        validate_attribution_manifest(ROOT, manifest_path)


def test_no_result_verifier_accepts_manifest_only_and_strict_mode_rejects_it(tmp_path: Path):
    prepare_attribution_manifest(ROOT, tmp_path)

    verify_attribution_stage(tmp_path, allow_missing_result=True)
    with pytest.raises(ValueError, match="result"):
        verify_attribution_stage(tmp_path, allow_missing_result=False)


def test_write_bundle_is_canonical_raw_only_and_trace_bound(tmp_path: Path):
    manifest_path = prepare_attribution_manifest(ROOT, tmp_path)
    trace = tmp_path / "geometry-trace.bin"
    trace.write_bytes(b"tiny geometry trace\n")

    write_attribution_bundle(
        ROOT,
        manifest_path,
        tmp_path,
        _result_payload(),
        {"geometry-trace.bin": "tiny geometry trace"},
    )

    result = _load(tmp_path / "result.json")
    provenance = _load(tmp_path / "provenance.json")
    trace_index = _load(tmp_path / "trace-index.json")
    assert set(result) == _RESULT_KEYS
    assert "outcome" not in result
    assert "operator_passed" not in result
    assert "behavior_passed" not in result
    assert "all_passed" not in result
    assert provenance["manifest_sha256"] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert provenance["result_sha256"] == hashlib.sha256(
        (tmp_path / "result.json").read_bytes()
    ).hexdigest()
    assert trace_index == {
        "schema_version": 1,
        "stage": _STAGE,
        "files": [
            {
                "filename": "geometry-trace.bin",
                "size": len(trace.read_bytes()),
                "sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
                "semantic_role": "tiny geometry trace",
            }
        ],
    }
    verify_attribution_stage(tmp_path, allow_missing_result=False)

    with pytest.raises(FileExistsError, match="overwrite"):
        write_attribution_bundle(
            ROOT,
            manifest_path,
            tmp_path,
            _result_payload(),
            {"geometry-trace.bin": "tiny geometry trace"},
        )


def test_bundle_rejects_outcome_pass_fields_trace_mutation_and_unindexed_files(tmp_path: Path):
    manifest_path = prepare_attribution_manifest(ROOT, tmp_path)
    trace = tmp_path / "permutation-trace.bin"
    trace.write_bytes(b"trace\n")

    for forbidden in ("outcome", "operator_passed", "behavior_passed", "all_passed", "c4b"):
        payload = _result_payload()
        payload[forbidden] = True
        with pytest.raises(ValueError, match="result"):
            write_attribution_bundle(
                ROOT,
                manifest_path,
                tmp_path,
                payload,
                {"permutation-trace.bin": "tiny permutation trace"},
            )

    write_attribution_bundle(
        ROOT,
        manifest_path,
        tmp_path,
        _result_payload(),
        {"permutation-trace.bin": "tiny permutation trace"},
    )
    trace.write_bytes(b"mutated\n")
    with pytest.raises(ValueError, match="trace"):
        verify_attribution_stage(tmp_path, allow_missing_result=False)

    trace.write_bytes(b"trace\n")
    extra = tmp_path / "unindexed.bin"
    extra.write_bytes(b"not indexed\n")
    with pytest.raises(ValueError, match="unindexed"):
        verify_attribution_stage(tmp_path, allow_missing_result=False)


def test_verifier_rejects_partial_bundle_and_symlinked_canonical_files(tmp_path: Path):
    prepare_attribution_manifest(ROOT, tmp_path)
    (tmp_path / "provenance.json").write_bytes(_canonical_bytes({"partial": True}))
    with pytest.raises(ValueError, match="partial"):
        verify_attribution_stage(tmp_path, allow_missing_result=True)

    (tmp_path / "provenance.json").unlink()
    manifest = tmp_path / "manifest.json"
    raw = manifest.read_bytes()
    manifest.unlink()
    target = tmp_path / "manifest-target.json"
    target.write_bytes(raw)
    try:
        manifest.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are unavailable on this platform")
    with pytest.raises(ValueError, match="regular file"):
        verify_attribution_stage(tmp_path, allow_missing_result=True)
