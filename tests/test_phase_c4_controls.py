from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest

from neural_state_machine.phase3c_schedule import AggregateFeedback, LatentRewardRecord
from neural_state_machine.phase_c4_controls import (
    FROZEN_C3_INPUT_PATHS,
    PhaseC4ProtocolAudit,
    frozen_input_hashes,
    load_phase_c4_formal_contract,
    validate_phase_c4_protocol,
)
from neural_state_machine.phase_c4_delay_model import DelayLaw, build_marginalized_features
from neural_state_machine.phase_c4_learner import DelayMarginalizedAnonymousCredit


FORMAL_HEAD = "8b2180ed24b6ff03db4b927ec29cdd9903b0ccac"
THEOREMS = (
    "NarrativeDynamics.MarginalizedTemporalCredit.registered_law_valid",
    "NarrativeDynamics.MarginalizedTemporalCredit.expected_aggregate_decomposition",
    "NarrativeDynamics.MarginalizedTemporalCredit.candidate_support_bounded",
    "NarrativeDynamics.MarginalizedTemporalCredit.current_weight_observation",
    "NarrativeDynamics.MarginalizedTemporalCredit.immediate_reduction_to_phase3a",
    "NarrativeDynamics.MarginalizedTemporalCredit.invalid_candidates_zero",
)
_EXPECTED_ACTIONS = (0, 1, 0, 1)


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


def _integer_sequence_digest(values: tuple[int, ...]) -> str:
    digest = hashlib.sha256()
    digest.update(len(values).to_bytes(8, "big", signed=False))
    for value in values:
        digest.update(value.to_bytes(8, "big", signed=True))
    return digest.hexdigest()


def _valid_audit() -> PhaseC4ProtocolAudit:
    return PhaseC4ProtocolAudit(
        decision_count=4,
        latent_reward_count=4,
        delivered_reward_count=4,
        real_feedback_count=4,
        drain_feedback_count=5,
        pending_final=0,
        action_digest=_integer_sequence_digest(_EXPECTED_ACTIONS),
        schedule_digest="b" * 64,
        call_digest="c" * 64,
        candidate_digest="d" * 64,
        parameter_digest="e" * 64,
        source_relabel_invariant=True,
        hidden_multiplicity_invariant=True,
        current_weight_probe_passed=True,
        bounded_history_passed=True,
        immediate_continuity_passed=True,
        repeatable=True,
    )


def _observer_feedback(step: int, value: float, *, relabel: int, multiplicity: int) -> AggregateFeedback:
    records = tuple(
        LatentRewardRecord(
            source_step=relabel + index,
            source_action=index % 2,
            reward=value / max(multiplicity, 1),
            delay=1 + 2 * (index % 3),
            due_step=step,
            delivery_step=step,
        )
        for index in range(multiplicity)
    )
    return AggregateFeedback(
        delivery_step=step,
        value=value,
        multiplicity=multiplicity,
        records=records,
    )


def _prepared_learner() -> DelayMarginalizedAnonymousCredit:
    learner = DelayMarginalizedAnonymousCredit(1, 2, step_size=0.1)
    for index, hidden in enumerate((1.0, 2.0, 3.0, 4.0, 5.0)):
        learner.select_for_training(
            np.array([hidden], dtype=np.float64),
            (index % 2,),
            np.random.default_rng(100 + index),
        )
        learner.learn(0.0)
    learner.select_for_training(
        np.array([6.0], dtype=np.float64),
        (1,),
        np.random.default_rng(200),
    )
    return learner


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


def test_protocol_audit_contains_no_behavior_result_fields():
    names = {field.name for field in fields(PhaseC4ProtocolAudit)}
    assert "accuracy" not in names
    assert "operator_passed" not in names
    assert "behavior_passed" not in names
    assert "post_training" not in names
    assert "state_reset" not in names
    assert "shuffled_control" not in names


def test_valid_protocol_audit_passes():
    validate_phase_c4_protocol(_valid_audit(), _EXPECTED_ACTIONS)


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("decision_count", 3),
        ("latent_reward_count", 3),
        ("delivered_reward_count", 3),
        ("real_feedback_count", 3),
        ("drain_feedback_count", 4),
        ("pending_final", 1),
        ("action_digest", "0" * 64),
        ("schedule_digest", "bad"),
        ("call_digest", "bad"),
        ("candidate_digest", "bad"),
        ("parameter_digest", "bad"),
        ("source_relabel_invariant", False),
        ("hidden_multiplicity_invariant", False),
        ("current_weight_probe_passed", False),
        ("bounded_history_passed", False),
        ("immediate_continuity_passed", False),
        ("repeatable", False),
    ],
)
def test_protocol_audit_fails_closed_on_each_mutation(field_name, bad_value):
    with pytest.raises(ValueError):
        validate_phase_c4_protocol(
            replace(_valid_audit(), **{field_name: bad_value}),
            _EXPECTED_ACTIONS,
        )


def test_registered_z_p_c_can_be_reconstructed_outside_learner():
    learner = _prepared_learner()
    history = learner.history_snapshot()
    features = build_marginalized_features(history, 5, 2, DelayLaw.registered())
    before = np.array([[0.5, 0.25], [-0.75, 0.125]], dtype=np.float64)
    learner._weights[:] = before

    update = learner.learn(0.25)

    expected_prediction = float(np.sum(before * features.expected_feature))
    expected_td_error = 0.25 - expected_prediction
    expected_after = before + 0.1 * expected_td_error * features.normalized_credit
    assert update.prediction == pytest.approx(expected_prediction, rel=0.0, abs=1e-15)
    assert update.td_error == pytest.approx(expected_td_error, rel=0.0, abs=1e-15)
    np.testing.assert_allclose(
        learner.parameter_snapshot(), expected_after, rtol=0.0, atol=1e-15
    )


def test_current_weight_perturbation_changes_prediction_through_same_z():
    left = _prepared_learner()
    right = _prepared_learner()
    history = left.history_snapshot()
    features = build_marginalized_features(history, 5, 2, DelayLaw.registered())
    left_weights = np.array([[0.5, 0.25], [-0.75, 0.125]], dtype=np.float64)
    right_weights = np.array([[1.5, 0.25], [-0.75, 0.125]], dtype=np.float64)
    left._weights[:] = left_weights
    right._weights[:] = right_weights

    left_update = left.learn(0.0)
    right_update = right.learn(0.0)

    expected_delta = float(np.sum((right_weights - left_weights) * features.expected_feature))
    assert right_update.prediction - left_update.prediction == pytest.approx(
        expected_delta, rel=0.0, abs=1e-15
    )


def test_source_labels_and_multiplicity_metadata_stay_external_to_c4():
    left = DelayMarginalizedAnonymousCredit(1, 2)
    right = DelayMarginalizedAnonymousCredit(1, 2)
    left_rng = np.random.default_rng(71)
    right_rng = np.random.default_rng(71)
    left_actions: list[int] = []
    right_actions: list[int] = []
    scalar_stream = (0.0, 1.0, -1.0, 0.0, 2.0, -2.0)

    for step, value in enumerate(scalar_stream):
        hidden = np.array([float(step + 1)], dtype=np.float64)
        left_actions.append(left.select_for_training(hidden, (0, 1), left_rng).action_index)
        right_actions.append(right.select_for_training(hidden, (0, 1), right_rng).action_index)
        left_feedback = _observer_feedback(step, value, relabel=10, multiplicity=1)
        right_feedback = _observer_feedback(step, value, relabel=100, multiplicity=2)
        left.learn(left_feedback.value)
        right.learn(right_feedback.value)

    assert tuple(left_actions) == tuple(right_actions)
    assert left.parameter_digest() == right.parameter_digest()


def _copy_frozen_inputs(destination: Path) -> None:
    source_root = Path(__file__).resolve().parents[1]
    for relative in FROZEN_C3_INPUT_PATHS:
        source = source_root / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def test_frozen_input_hashes_cover_registered_c3_inputs_and_detect_mutation(tmp_path):
    baseline = frozen_input_hashes()
    assert tuple(path for path, _ in baseline) == FROZEN_C3_INPUT_PATHS
    assert all(len(digest) == 64 for _, digest in baseline)

    _copy_frozen_inputs(tmp_path)
    copied = frozen_input_hashes(tmp_path)
    assert copied == baseline

    changed = tmp_path / FROZEN_C3_INPUT_PATHS[0]
    changed.write_bytes(changed.read_bytes() + b"\n")
    assert frozen_input_hashes(tmp_path) != baseline


def test_frozen_input_hashes_reject_missing_path(tmp_path):
    _copy_frozen_inputs(tmp_path)
    (tmp_path / FROZEN_C3_INPUT_PATHS[0]).unlink()
    with pytest.raises(ValueError, match="missing"):
        frozen_input_hashes(tmp_path)


def test_frozen_input_hashes_reject_symlink(tmp_path):
    _copy_frozen_inputs(tmp_path)
    target = tmp_path / FROZEN_C3_INPUT_PATHS[0]
    target.unlink()
    outside = tmp_path / "outside.txt"
    outside.write_text("not frozen\n", encoding="utf-8")
    target.symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        frozen_input_hashes(tmp_path)
