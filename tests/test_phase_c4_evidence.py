from __future__ import annotations

import json
from pathlib import Path

import pytest

from neural_state_machine.phase_c4_evidence import (
    canonical_json_bytes,
    canonical_json_sha256,
    prepare_c4a_manifest,
    prepare_c4b_manifest,
    verify_c4_stage,
)


ROOT = Path(__file__).resolve().parents[1]
FORMAL_HEAD = "8b2180ed24b6ff03db4b927ec29cdd9903b0ccac"


def _load(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_canonical_json_is_sorted_compact_utf8_and_newline_terminated():
    payload = {"b": 1, "a": "μ"}
    rendered = canonical_json_bytes(payload)
    assert rendered == '{"a":"μ","b":1}\n'.encode("utf-8")
    assert canonical_json_sha256(payload) == __import__("hashlib").sha256(rendered).hexdigest()


def test_canonical_json_rejects_non_finite_values():
    with pytest.raises(ValueError):
        canonical_json_bytes({"bad": float("nan")})


def test_prepare_c4a_manifest_seals_registered_science_without_result(tmp_path):
    path = prepare_c4a_manifest(ROOT, tmp_path)
    payload = _load(path)

    assert path.name == "c4a-manifest.json"
    assert payload["schema_version"] == 1
    assert payload["stage"] == "c4a"
    assert payload["seeds"] == [7, 17, 29]
    assert payload["formal"]["commit"] == FORMAL_HEAD
    assert payload["delay_law"] == {
        "support": [1, 3, 5],
        "probabilities": [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
    }
    assert payload["config"]["training_decisions"] == 2_000
    assert payload["config"]["ridge_penalty"] == 1e-6
    assert payload["public_drain_feedback_count"] == 5
    assert payload["expected_scalar_design_rows"] == 2_005
    assert payload["thresholds"] == {
        "normal_min": 180,
        "per_delay_min": 34,
        "reset_exact": 100,
        "reset_per_delay_exact": 20,
        "shuffled_max_exclusive": 150,
    }
    assert len(payload["evaluation_manifest"]) == 27
    assert payload["c4a_binding"] is None
    assert not (tmp_path / "c4a-result.json").exists()
    verify_c4_stage(tmp_path, "c4a", True)


def test_no_result_verifier_rejects_any_result_presence(tmp_path):
    prepare_c4a_manifest(ROOT, tmp_path)
    (tmp_path / "c4a-result.json").write_bytes(canonical_json_bytes({}))
    with pytest.raises((ValueError, RuntimeError)):
        verify_c4_stage(tmp_path, "c4a", True)


def test_manifest_mutation_fails_closed(tmp_path):
    path = prepare_c4a_manifest(ROOT, tmp_path)
    payload = _load(path)
    payload["thresholds"]["normal_min"] = 179
    path.write_bytes(canonical_json_bytes(payload))
    with pytest.raises((ValueError, RuntimeError)):
        verify_c4_stage(tmp_path, "c4a", True)


def test_manifest_extra_key_fails_closed(tmp_path):
    path = prepare_c4a_manifest(ROOT, tmp_path)
    payload = _load(path)
    payload["unexpected"] = True
    path.write_bytes(canonical_json_bytes(payload))
    with pytest.raises((ValueError, RuntimeError)):
        verify_c4_stage(tmp_path, "c4a", True)


def test_prepare_c4b_rejects_unverified_or_failed_c4a_result(tmp_path):
    c4a_dir = tmp_path / "c4a"
    c4a_dir.mkdir()
    result = c4a_dir / "c4a-result.json"
    result.write_bytes(
        canonical_json_bytes(
            {
                "schema_version": 1,
                "stage": "c4a",
                "operator_passed": False,
            }
        )
    )
    (c4a_dir / "c4a-provenance.json").write_bytes(canonical_json_bytes({"stage": "c4a"}))

    with pytest.raises((ValueError, RuntimeError)):
        prepare_c4b_manifest(ROOT, result, tmp_path / "c4b")
