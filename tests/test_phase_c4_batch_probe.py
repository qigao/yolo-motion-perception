from __future__ import annotations

import inspect
from dataclasses import fields

import numpy as np
import pytest

import neural_state_machine.phase_c4_batch_probe as batch_probe_module
from neural_state_machine.phase_c4_batch_probe import (
    C4_RIDGE_REGULARIZATION,
    BatchProbeFit,
    build_batch_design,
    fit_anonymous_batch_probe,
)
from neural_state_machine.phase_c4_delay_model import (
    DecisionCreditRow,
    DelayLaw,
    candidate_indices,
)


def _row(index: int, action: int, feature: tuple[float, ...]) -> DecisionCreditRow:
    values = np.array(feature, dtype=np.float64)
    return DecisionCreditRow(
        decision_index=index,
        action_index=action,
        feature=values,
        denominator=float(values @ values),
    )


def _analytic_design() -> tuple[np.ndarray, np.ndarray]:
    design = np.array(
        [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [2.0, -1.0, 0.5, 1.5],
        ],
        dtype=np.float64,
    )
    target = np.array((1.0, -2.0, 0.5, 3.0, -1.25), dtype=np.float64)
    return design, target


def test_registered_surfaces_match_committed_plan():
    build_signature = inspect.signature(build_batch_design)
    fit_signature = inspect.signature(fit_anonymous_batch_probe)

    assert tuple(build_signature.parameters) == (
        "rows",
        "feedback",
        "action_count",
        "law",
    )
    assert tuple(fit_signature.parameters) == (
        "design",
        "target",
        "action_count",
        "feature_size",
        "penalty",
    )
    assert fit_signature.parameters["penalty"].default == 1e-6
    assert C4_RIDGE_REGULARIZATION == 1e-6


def test_fixed_ridge_matches_independent_normal_equation_reference():
    design, target = _analytic_design()

    fit = fit_anonymous_batch_probe(design, target, 2, 2)

    reference = np.linalg.solve(
        design.T @ design
        + C4_RIDGE_REGULARIZATION * np.eye(design.shape[1], dtype=np.float64),
        design.T @ target,
    )
    np.testing.assert_allclose(
        fit.weights.reshape(-1), reference, rtol=0.0, atol=1e-12
    )
    assert isinstance(fit, BatchProbeFit)
    assert {field.name for field in fields(fit)} == {
        "weights",
        "row_count",
        "column_count",
        "augmented_rank",
        "residual_norm",
        "singular_values",
        "penalty",
    }
    assert fit.row_count == design.shape[0]
    assert fit.column_count == design.shape[1]
    assert fit.augmented_rank == design.shape[1]
    assert fit.penalty == C4_RIDGE_REGULARIZATION
    assert fit.residual_norm >= 0.0
    assert fit.weights.flags.writeable is False
    assert fit.singular_values.flags.writeable is False


def test_bias_coordinates_are_penalized_too():
    design = np.array(
        [
            [1.0, 1.0, 0.0, 0.0],
            [2.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 2.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    target = np.array((1.0, 0.0, -1.0, 0.0, 0.25), dtype=np.float64)

    fit = fit_anonymous_batch_probe(design, target, 2, 2)

    full_penalty = C4_RIDGE_REGULARIZATION * np.eye(4, dtype=np.float64)
    expected = np.linalg.solve(design.T @ design + full_penalty, design.T @ target)
    unpenalized_bias = full_penalty.copy()
    unpenalized_bias[1, 1] = 0.0
    unpenalized_bias[3, 3] = 0.0
    wrong = np.linalg.solve(
        design.T @ design + unpenalized_bias,
        design.T @ target,
    )
    np.testing.assert_allclose(
        fit.weights.reshape(-1), expected, rtol=0.0, atol=1e-12
    )
    assert not np.allclose(
        fit.weights.reshape(-1), wrong, rtol=0.0, atol=1e-10
    )


def test_batch_design_uses_only_candidate_rows_and_public_drain_horizon():
    rows = (
        _row(0, 0, (1.0, 1.0)),
        _row(1, 1, (2.0, 1.0)),
        _row(2, 0, (3.0, 1.0)),
        _row(3, 1, (4.0, 1.0)),
    )
    feedback = tuple(float(index - 4) for index in range(9))
    law = DelayLaw.registered()

    design, target = build_batch_design(rows, feedback, 2, law)

    assert design.shape == (9, 4)
    assert target.shape == (9,)
    np.testing.assert_array_equal(target, np.asarray(feedback, dtype=np.float64))
    expected_last = np.zeros((2, 2), dtype=np.float64)
    expected_last[1] = np.array((4.0, 1.0), dtype=np.float64) / 3.0
    np.testing.assert_allclose(
        design[8], expected_last.reshape(-1, order="C"), rtol=0.0, atol=1e-15
    )
    assert candidate_indices(9, 4, law) == ()
    assert design.flags.writeable is False
    assert target.flags.writeable is False


def test_batch_design_information_boundary_excludes_privileged_sources():
    source = inspect.getsource(batch_probe_module)
    forbidden = (
        "LatentRewardRecord",
        "AggregateFeedback.records",
        "correct_actions",
        "phase3c_diagnostics",
        "SourceVisibleDelayedReference",
        "source_step",
        "due_step",
        "multiplicity",
    )
    for token in forbidden:
        assert token not in source


def test_fit_is_repeatable_and_does_not_alias_inputs():
    design, target = _analytic_design()
    left = fit_anonymous_batch_probe(design, target, 2, 2)
    right = fit_anonymous_batch_probe(design.copy(), target.copy(), 2, 2)
    frozen_weights = left.weights.copy()
    frozen_singular = left.singular_values.copy()

    design[:] = 99.0
    target[:] = 99.0

    assert left.weights.tobytes() == right.weights.tobytes()
    assert left.singular_values.tobytes() == right.singular_values.tobytes()
    np.testing.assert_array_equal(left.weights, frozen_weights)
    np.testing.assert_array_equal(left.singular_values, frozen_singular)


@pytest.mark.parametrize("penalty", [0.0, 2e-6, 1e-5, True, float("nan")])
def test_fit_rejects_any_nonregistered_penalty(penalty):
    design, target = _analytic_design()
    with pytest.raises(ValueError, match="penalty"):
        fit_anonymous_batch_probe(design, target, 2, 2, penalty=penalty)


@pytest.mark.parametrize(
    ("design", "target", "action_count", "feature_size"),
    [
        (np.zeros((0, 4)), np.zeros(0), 2, 2),
        (np.zeros(4), np.zeros(1), 2, 2),
        (np.zeros((3, 3)), np.zeros(3), 2, 2),
        (np.full((3, 4), np.nan), np.zeros(3), 2, 2),
        (np.zeros((3, 4)), np.zeros(2), 2, 2),
        (np.zeros((3, 4)), np.zeros((3, 1)), 2, 2),
        (np.zeros((3, 4)), np.array((0.0, np.inf, 0.0)), 2, 2),
        (np.zeros((3, 4)), np.zeros(3), 1, 4),
        (np.zeros((3, 4)), np.zeros(3), 2, 0),
    ],
)
def test_invalid_fit_inputs_fail_closed(design, target, action_count, feature_size):
    with pytest.raises(ValueError):
        fit_anonymous_batch_probe(
            design,
            target,
            action_count,
            feature_size,
        )


def test_invalid_design_inputs_fail_closed():
    law = DelayLaw.registered()
    rows = (_row(0, 0, (1.0, 1.0)),)

    with pytest.raises(ValueError):
        build_batch_design((), (0.0,), 2, law)
    with pytest.raises(ValueError):
        build_batch_design(rows, (), 2, law)
    with pytest.raises(ValueError):
        build_batch_design(rows, (0.0, float("nan")), 2, law)
    with pytest.raises(ValueError):
        build_batch_design(rows, (0.0,), 1, law)
    with pytest.raises(ValueError):
        build_batch_design(rows, (0.0,), 2, object())
