from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from neural_state_machine import r1_e1_protocol as protocol


def _expected_manifest() -> dict[str, object]:
    return {
        "phase": "R1-E1",
        "architectures": {
            "shallow": 0,
            "grouped2": 1,
            "grouped4": 2,
            "deep2": 3,
            "deep4": 4,
        },
        "budgets": {"64": 0, "256": 1},
        "seeds": [7, 17, 29, 43, 61],
        "reservoir": {
            "activation": "tanh",
            "spectral_radius": 0.9,
            "leak": 1.0,
            "recurrent_bias": False,
            "dtype": "float64",
            "input_weight_std": "1/sqrt(input_dim)",
            "recurrent_weight_std": "1/sqrt(hidden_dim)",
            "lineage_tag": 0x52314531,
        },
        "ridge": {
            "regularization": 1e-6,
            "bias": "enabled-unpenalized",
            "binary_target": "-1/+1",
            "binary_tie": 0,
            "multiclass_tie": "lowest-index",
        },
        "e1_a": {
            "delays": [1, 2, 5, 10, 20, 40, 80],
            "train_per_delay": 400,
            "evaluation_per_delay": 40,
            "train_tag": 0x45314154,
            "evaluation_tag": 0x45314145,
            "memory_threshold": 0.85,
            "reset_overall": [140, 280],
            "reset_per_delay": [20, 40],
        },
        "e1_b": {
            "classes": ["AB", "BA", "AA", "BB"],
            "horizons": [1, 5, 20, 40],
            "train_pairs_per_horizon": 200,
            "evaluation_pairs_per_horizon": 50,
            "train_tag": 0x45314254,
            "evaluation_tag": 0x45314245,
            "separability_threshold": 0.80,
            "reset_per_horizon": [50, 200],
        },
        "e1_c": {
            "classes": ["approach", "touch", "pick_up", "pass_by"],
            "corruptions": ["clean", "drop10", "wrong10", "occlusion4", "jitter", "mixed"],
            "train_pairs": 200,
            "evaluation_pairs": 50,
            "train_tag": 0x45314354,
            "evaluation_tag": 0x45314345,
            "corruption_tag": 0x45314343,
            "feature_count": 9,
            "sequence_frames": 20,
            "behavior_frames": 16,
            "wrong_donor_cycle": [0, 1, 2, 3, 0],
            "occlusion_frames": [8, 9, 10, 11],
            "jitter_sigma": 0.05,
        },
    }


def test_registered_manifest_payload_is_literal_and_json_safe() -> None:
    payload = protocol.registered_manifest_payload()

    assert payload == _expected_manifest()
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload


def test_r1_e1_import_isolation_gate_passes_current_sources() -> None:
    protocol.assert_r1_r2_import_isolation()


def test_import_isolation_detects_forbidden_r2_import(tmp_path: Path) -> None:
    bad = tmp_path / "r1_e1_bad.py"
    bad.write_text("from .phase3b_learners import TD0\n", encoding="utf-8")

    issues = protocol.scan_r1_r2_import_isolation((bad,))

    assert issues == (
        protocol.ValidityIssue("forbidden-import", "r1_e1_bad.py imports phase3b_learners"),
    )


def test_import_isolation_uses_ast_not_substring_comments(tmp_path: Path) -> None:
    safe = tmp_path / "r1_e1_safe.py"
    safe.write_text("# phase3 delayed_credit reward_learning\nVALUE = 'phase_c4'\n", encoding="utf-8")

    assert protocol.scan_r1_r2_import_isolation((safe,)) == ()
    ast.parse(safe.read_text(encoding="utf-8"))


def test_protocol_smoke_is_deterministic_small_and_never_registered_measurement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_measurement():
        pytest.fail("protocol smoke invoked registered measurement")

    monkeypatch.setattr(protocol, "run_registered_measurement", forbidden_measurement)

    first = protocol.protocol_smoke(seed=7)
    second = protocol.protocol_smoke(seed=7)

    assert first == second
    assert first["registered_measurement"] is False
    assert first["seed"] == 7
    assert first["arm_count"] == 10
    assert first["smoke_pairs"] == 1
    assert first["valid"] is True
    assert first["issues"] == []
    assert len(first["reservoir_parameter_digests"]) == 10
    assert len(set(first["reservoir_parameter_digests"].values())) == 10
    assert first["fixture_digests"] == second["fixture_digests"]
    assert set(first["fixture_digests"]) == {"e1_a", "e1_b", "e1_c", "e1_c_corruption"}


def test_validity_is_fail_closed_and_never_substitutes_zero_scores() -> None:
    issues = protocol.validate_measurement_payload(
        {
            "state_dim_matches": False,
            "parameter_digest_stable": True,
            "digests_present": True,
            "fixture_lineage_matches": True,
            "reset_controls_exact": True,
            "phase2a_compatibility": True,
            "deterministic": True,
            "r2_isolated": True,
            "no_posthoc_selection": True,
        }
    )

    assert issues == (protocol.ValidityIssue("state-dim", "state_dim must equal budget"),)
    with pytest.raises(protocol.ProtocolInvalid, match="state-dim"):
        protocol.require_valid(issues)


def test_validity_gate_reports_all_missing_section13_conditions() -> None:
    issues = protocol.validate_measurement_payload({})

    assert {issue.code for issue in issues} == {
        "state-dim",
        "parameter-stability",
        "digests",
        "fixture-lineage",
        "reset-controls",
        "phase2a-compatibility",
        "determinism",
        "r2-isolation",
        "posthoc-selection",
    }


def test_registered_runner_signature_rejects_overrides() -> None:
    with pytest.raises(TypeError):
        protocol.run_registered_measurement(seeds=(7,))
