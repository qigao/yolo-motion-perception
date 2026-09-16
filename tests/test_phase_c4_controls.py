from __future__ import annotations

import json
from pathlib import Path

import pytest

from neural_state_machine.phase_c4_controls import load_phase_c4_formal_contract


FORMAL_HEAD = "8b2180ed24b6ff03db4b927ec29cdd9903b0ccac"
THEOREMS = (
    "NarrativeDynamics.MarginalizedTemporalCredit.registered_law_valid",
    "NarrativeDynamics.MarginalizedTemporalCredit.expected_aggregate_decomposition",
    "NarrativeDynamics.MarginalizedTemporalCredit.candidate_support_bounded",
    "NarrativeDynamics.MarginalizedTemporalCredit.current_weight_observation",
    "NarrativeDynamics.MarginalizedTemporalCredit.immediate_reduction_to_phase3a",
    "NarrativeDynamics.MarginalizedTemporalCredit.invalid_candidates_zero",
)


def _payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "repository": "qigao/lean",
        "module": "NarrativeDynamics/Core/MarginalizedTemporalCredit.lean",
        "commit": FORMAL_HEAD,
        "lean_version": "4.32.0",
        "mathlib_version": "v4.32.0",
        "exact_head_ci_passed": True,
        "axiom_audit_reviewed": True,
        "sorry_ax_present": False,
        "custom_axiom_present": False,
        "theorems": list(THEOREMS),
    }


def _write_contract(root: Path, payload: dict[str, object]) -> Path:
    path = root / "docs/experiments/phase-c4-delay-marginalized-credit/formal-contract.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def test_committed_formal_contract_binds_exact_passing_head():
    contract = load_phase_c4_formal_contract()
    assert contract.commit == FORMAL_HEAD
    assert contract.theorems == THEOREMS


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("repository", "qigao/not-lean"),
        ("module", "NarrativeDynamics/Core/AnonymousTemporalCredit.lean"),
        ("commit", "bad"),
        ("lean_version", "4.31.0"),
        ("mathlib_version", "v4.31.0"),
        ("exact_head_ci_passed", False),
        ("axiom_audit_reviewed", False),
        ("sorry_ax_present", True),
        ("custom_axiom_present", True),
    ],
)
def test_formal_contract_rejects_invalid_fields(tmp_path, key, value):
    payload = _payload()
    payload[key] = value
    _write_contract(tmp_path, payload)
    with pytest.raises(ValueError):
        load_phase_c4_formal_contract(tmp_path)


def test_formal_contract_rejects_changed_theorem_set(tmp_path):
    payload = _payload()
    payload["theorems"] = list(THEOREMS[:-1])
    _write_contract(tmp_path, payload)
    with pytest.raises(ValueError):
        load_phase_c4_formal_contract(tmp_path)


def test_formal_contract_rejects_extra_keys(tmp_path):
    payload = _payload()
    payload["unexpected"] = True
    _write_contract(tmp_path, payload)
    with pytest.raises(ValueError):
        load_phase_c4_formal_contract(tmp_path)


def test_formal_contract_rejects_symlink(tmp_path):
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps(_payload()), encoding="utf-8")
    path = tmp_path / "docs/experiments/phase-c4-delay-marginalized-credit/formal-contract.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(outside)
    with pytest.raises(ValueError):
        load_phase_c4_formal_contract(tmp_path)
