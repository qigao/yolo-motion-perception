from __future__ import annotations

import json
from pathlib import Path

import pytest

from neural_state_machine.phase3c_formal_contract import (
    APPROVED_FORMAL_COMMIT,
    REQUIRED_THEOREMS,
    load_phase3c_formal_contract,
)


FORMAL_COMMIT = "3de297cee2a94a7fc309531334f720f1b34467c9"


def _valid_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "repository": "qigao/lean",
        "module": "NarrativeDynamics/Core/AnonymousTemporalCredit.lean",
        "commit": FORMAL_COMMIT,
        "lean_version": "4.32.0",
        "mathlib_version": "v4.32.0",
        "exact_head_ci_passed": True,
        "axiom_audit_reviewed": True,
        "sorry_ax_present": False,
        "custom_axiom_present": False,
        "theorems": list(REQUIRED_THEOREMS),
    }


def _write_contract(root: Path, payload: dict[str, object]) -> Path:
    path = root / "docs" / "experiments" / "phase-3c-formal-contract.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_committed_contract_binds_exact_gate_f_head() -> None:
    contract = load_phase3c_formal_contract()

    assert APPROVED_FORMAL_COMMIT == FORMAL_COMMIT
    assert contract.commit == FORMAL_COMMIT
    assert contract.theorems == REQUIRED_THEOREMS
    assert contract.exact_head_ci_passed is True
    assert contract.axiom_audit_reviewed is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("repository", "qigao/yolo-motion-perception"),
        ("module", "NarrativeDynamics/Core/TemporalCredit.lean"),
        ("lean_version", "4.31.0"),
        ("mathlib_version", "v4.31.0"),
        ("exact_head_ci_passed", False),
        ("axiom_audit_reviewed", False),
        ("sorry_ax_present", True),
        ("custom_axiom_present", True),
    ],
)
def test_contract_rejects_invalid_fixed_fields(
    tmp_path: Path, field: str, value: object
) -> None:
    payload = _valid_payload()
    payload[field] = value
    _write_contract(tmp_path, payload)

    with pytest.raises(ValueError):
        load_phase3c_formal_contract(tmp_path)


def test_contract_rejects_non_hex_or_wrong_commit(tmp_path: Path) -> None:
    for commit in ("not-a-sha", "0" * 40):
        payload = _valid_payload()
        payload["commit"] = commit
        _write_contract(tmp_path, payload)
        with pytest.raises(ValueError):
            load_phase3c_formal_contract(tmp_path)


def test_contract_rejects_missing_or_extra_theorem(tmp_path: Path) -> None:
    for theorems in (list(REQUIRED_THEOREMS[:-1]), [*REQUIRED_THEOREMS, "extra"]):
        payload = _valid_payload()
        payload["theorems"] = theorems
        _write_contract(tmp_path, payload)
        with pytest.raises(ValueError):
            load_phase3c_formal_contract(tmp_path)


def test_contract_rejects_schema_change(tmp_path: Path) -> None:
    payload = _valid_payload()
    payload["schema_version"] = 2
    _write_contract(tmp_path, payload)

    with pytest.raises(ValueError):
        load_phase3c_formal_contract(tmp_path)


def test_contract_rejects_symlink_file(tmp_path: Path) -> None:
    real = tmp_path / "real.json"
    real.write_text(json.dumps(_valid_payload()), encoding="utf-8")
    contract = tmp_path / "docs" / "experiments" / "phase-3c-formal-contract.json"
    contract.parent.mkdir(parents=True)
    contract.symlink_to(real)

    with pytest.raises(ValueError):
        load_phase3c_formal_contract(tmp_path)
