"""Diagnostic-only variance decomposition for the frozen Phase 3A failure."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from .action_value_benchmark import (
    ActionValueBenchmarkConfig,
    _build_fixture_bundle,
    _evaluate,
    _new_learner,
    _new_policy,
    _train_normal,
)
from .memory_benchmark import AccuracyCount
from .memory_probe import _collect_dataset, fit_linear_probe
from .memory_task import DelayedCueTask


@dataclass(frozen=True)
class AttributionMeasurement:
    """One causal arm, keeping the Phase 3A algorithm and reporting readouts."""

    arm: str
    reservoir_seed: int
    action_seed: int
    fixture_seed: int
    post_training: AccuracyCount
    post_per_delay: tuple[tuple[int, AccuracyCount], ...]
    q_margin_mean: float
    q_margin_p10: float
    q_margin_minimum: float
    supervised_reference: AccuracyCount
    supervised_per_delay: tuple[tuple[int, AccuracyCount], ...]
    action_digest: str
    reward_digest: str
    action_sequence_equal_to_baseline: bool
    reward_sequence_equal_to_baseline: bool
    fixture_digest: str
    evaluation_fixture_digest: str


@dataclass(frozen=True)
class _Run:
    measurement: AttributionMeasurement
    actions: tuple[int, ...]
    rewards: tuple[float, ...]


def attribution_payload(
    measurements: Sequence[AttributionMeasurement],
) -> dict[str, object]:
    """Serialize diagnostic measurements without acceptance-gate semantics."""
    values = tuple(measurements)
    if not values or any(not isinstance(value, AttributionMeasurement) for value in values):
        raise ValueError("measurements must be a non-empty sequence of AttributionMeasurement")
    return {
        "experiment": "phase-3a-failure-attribution",
        "diagnostic_only": True,
        "schema_version": 1,
        "arms": [_measurement_payload(value) for value in values],
    }


def run_failure_attribution(
    seed: int = 7,
    config: ActionValueBenchmarkConfig | None = None,
    *,
    reservoir_seeds: Sequence[int] = (17, 29),
    action_seeds: Sequence[int] = (17, 29),
    fixture_seeds: Sequence[int] = (17, 29),
) -> tuple[AttributionMeasurement, ...]:
    """Run the fixed baseline and single-factor diagnostic arms.

    The baseline uses one seed for all three lineages.  Each diagnostic arm
    varies exactly one lineage while retaining the other two.  This function is
    intentionally separate from the frozen evidence writer and never rewrites
    Phase 3A evidence.
    """
    base_seed = _validated_seed(seed)
    resolved_config = (
        ActionValueBenchmarkConfig() if config is None else _validated_config(config)
    )
    reservoirs = _validated_variants(reservoir_seeds, "reservoir_seeds")
    actions = _validated_variants(action_seeds, "action_seeds")
    fixtures = _validated_variants(fixture_seeds, "fixture_seeds")

    baseline = _run_arm(
        "baseline",
        base_seed,
        base_seed,
        base_seed,
        resolved_config,
    )
    measurements = [baseline.measurement]
    for reservoir_seed in reservoirs:
        measurements.append(
            _relative_measurement(
                _run_arm(
                    "reservoir",
                    reservoir_seed,
                    base_seed,
                    base_seed,
                    resolved_config,
                ),
                baseline,
            )
        )
    for action_seed in actions:
        measurements.append(
            _relative_measurement(
                _run_arm(
                    "action_rng",
                    base_seed,
                    action_seed,
                    base_seed,
                    resolved_config,
                ),
                baseline,
            )
        )
    for fixture_seed in fixtures:
        measurements.append(
            _relative_measurement(
                _run_arm(
                    "fixture",
                    base_seed,
                    base_seed,
                    fixture_seed,
                    resolved_config,
                ),
                baseline,
            )
        )
    return tuple(measurements)


def _run_arm(
    arm: str,
    reservoir_seed: int,
    action_seed: int,
    fixture_seed: int,
    config: ActionValueBenchmarkConfig,
) -> _Run:
    task = DelayedCueTask()
    bundle = _build_fixture_bundle(fixture_seed, config)
    policy = _new_policy(reservoir_seed, config)
    learner = _new_learner(config)
    action_rng = np.random.default_rng(
        np.random.SeedSequence([action_seed, 0x33414354])
    )
    trace = _train_normal(
        policy,
        learner,
        task,
        bundle.training,
        action_rng,
        config,
    )
    post = _evaluate(
        policy,
        learner,
        bundle.evaluation,
        reset_before_decision=False,
    )

    training_data = _collect_dataset(
        policy,
        bundle.training,
        reset_before_decision=False,
    )
    evaluation_data = _collect_dataset(
        policy,
        bundle.evaluation,
        reset_before_decision=False,
    )
    probe = fit_linear_probe(
        training_data.states,
        training_data.labels,
        regularization=1e-3,
    )
    supervised_choices = probe.predict(evaluation_data.states)
    supervised_matches = supervised_choices == evaluation_data.labels
    supervised = AccuracyCount(
        int(np.count_nonzero(supervised_matches)),
        int(supervised_matches.size),
    )
    supervised_per_delay = tuple(
        (
            delay,
            AccuracyCount(
                int(
                    np.count_nonzero(
                        supervised_matches[evaluation_data.delays == delay]
                    )
                ),
                int(np.count_nonzero(evaluation_data.delays == delay)),
            ),
        )
        for delay in range(1, 6)
    )
    measurement = AttributionMeasurement(
        arm=arm,
        reservoir_seed=reservoir_seed,
        action_seed=action_seed,
        fixture_seed=fixture_seed,
        post_training=post.overall,
        post_per_delay=post.per_delay,
        q_margin_mean=post.margin_mean,
        q_margin_p10=post.margin_p10,
        q_margin_minimum=post.margin_minimum,
        supervised_reference=supervised,
        supervised_per_delay=supervised_per_delay,
        action_digest=trace.action_digest,
        reward_digest=trace.reward_digest,
        action_sequence_equal_to_baseline=True,
        reward_sequence_equal_to_baseline=True,
        fixture_digest=bundle.training_fixture_digest,
        evaluation_fixture_digest=bundle.evaluation_fixture_digest,
    )
    return _Run(measurement, trace.actions, trace.rewards)


def _relative_measurement(run: _Run, baseline: _Run) -> AttributionMeasurement:
    measurement = run.measurement
    return replace(
        measurement,
        action_sequence_equal_to_baseline=run.actions == baseline.actions,
        reward_sequence_equal_to_baseline=run.rewards == baseline.rewards,
    )


def _validated_seed(value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("seed must be a non-negative integer")
    return value


def _validated_variants(values: object, name: str) -> tuple[int, ...]:
    try:
        resolved = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence of non-negative integers") from exc
    if any(type(value) is not int or value < 0 for value in resolved):
        raise ValueError(f"{name} must contain non-negative integers")
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{name} must not contain duplicates")
    return resolved


def _validated_config(value: object) -> ActionValueBenchmarkConfig:
    if not isinstance(value, ActionValueBenchmarkConfig):
        raise ValueError("config must be an ActionValueBenchmarkConfig")
    return value


def _accuracy_payload(value: AccuracyCount) -> dict[str, object]:
    return {
        "correct": value.correct,
        "total": value.total,
        "accuracy": value.accuracy,
    }


def _measurement_payload(value: AttributionMeasurement) -> dict[str, object]:
    return {
        "arm": value.arm,
        "reservoir_seed": value.reservoir_seed,
        "action_seed": value.action_seed,
        "fixture_seed": value.fixture_seed,
        "post_training": _accuracy_payload(value.post_training),
        "post_per_delay": {
            str(delay): _accuracy_payload(score)
            for delay, score in value.post_per_delay
        },
        "q_margin": {
            "mean": value.q_margin_mean,
            "p10": value.q_margin_p10,
            "minimum": value.q_margin_minimum,
        },
        "supervised_reference": _accuracy_payload(value.supervised_reference),
        "supervised_per_delay": {
            str(delay): _accuracy_payload(score)
            for delay, score in value.supervised_per_delay
        },
        "action_digest": value.action_digest,
        "reward_digest": value.reward_digest,
        "action_sequence_equal_to_baseline": value.action_sequence_equal_to_baseline,
        "reward_sequence_equal_to_baseline": value.reward_sequence_equal_to_baseline,
        "fixture_digest": value.fixture_digest,
        "evaluation_fixture_digest": value.evaluation_fixture_digest,
    }
