"""Diagnostic TD(0) versus eligibility-trace comparison for Phase 3A."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .action_value import (
    ActionValueDecision,
    ActionValueUpdate,
    NormalizedActionValue,
    _PendingCredit,
    readonly_float64_copy,
    validated_finite_scalar,
)
from .action_value_benchmark import (
    ActionValueBenchmarkConfig,
    _build_fixture_bundle,
    _evaluate,
    _new_learner,
    _new_policy,
    _train_normal,
)
from .memory_benchmark import AccuracyCount
from .memory_task import DelayedCueTask


class EligibilityTraceActionValue(NormalizedActionValue):
    """Action-value learner with a persistent accumulating eligibility trace.

    The trace is intentionally retained across immediate-reward episodes for
    this diagnostic comparison.  It is not used by the frozen Phase 3A
    acceptance benchmark.
    """

    def __init__(
        self,
        hidden_size: int,
        action_count: int,
        step_size: float = 0.1,
        *,
        discount: float = 0.9,
        trace_decay: float = 0.8,
    ) -> None:
        super().__init__(hidden_size, action_count, step_size)
        self.discount = _validated_unit_interval(discount, "discount")
        self.trace_decay = _validated_unit_interval(trace_decay, "trace_decay")
        self._eligibility = np.zeros_like(self._weights)

    def select_for_training(
        self,
        hidden_state: object,
        legal_action_indices: object,
        rng: np.random.Generator,
    ) -> ActionValueDecision:
        decision = super().select_for_training(hidden_state, legal_action_indices, rng)
        pending = self._pending
        if not isinstance(pending, _PendingCredit):
            raise RuntimeError("trace update requires pending feedback")
        self._eligibility *= self.discount * self.trace_decay
        self._eligibility[pending.action_index] += (
            pending.feature / pending.denominator
        )
        return decision

    def learn(self, reward: object) -> ActionValueUpdate:
        if self._pending is None:
            raise RuntimeError("learning requires pending feedback")
        reward_value = validated_finite_scalar(reward, "reward")
        pending = self._pending
        td_error = reward_value - pending.prediction
        self._weights += self.step_size * td_error * self._eligibility
        self._pending = None
        return ActionValueUpdate(
            action_index=pending.action_index,
            prediction_before=pending.prediction,
            reward=reward_value,
            td_error=td_error,
        )

    def eligibility_snapshot(self) -> np.ndarray:
        return readonly_float64_copy(self._eligibility)


@dataclass(frozen=True)
class CreditComparison:
    seed: int
    config: ActionValueBenchmarkConfig
    discount: float
    trace_decay: float
    td0_post_training: AccuracyCount
    td0_per_delay: tuple[tuple[int, AccuracyCount], ...]
    lambda_post_training: AccuracyCount
    lambda_per_delay: tuple[tuple[int, AccuracyCount], ...]
    td0_action_digest: str
    lambda_action_digest: str
    action_sequences_equal: bool
    training_fixture_digest: str
    lambda_training_fixture_digest: str
    evaluation_fixture_digest: str
    lambda_evaluation_fixture_digest: str
    td0_parameter_digest: str
    lambda_parameter_digest: str


def credit_comparison_payload(result: CreditComparison) -> dict[str, object]:
    """Serialize one diagnostic comparison without acceptance semantics."""
    if not isinstance(result, CreditComparison):
        raise ValueError("result must be a CreditComparison")
    return {
        "experiment": "phase-3a-credit-comparison",
        "diagnostic_only": True,
        "schema_version": 1,
        "seed": result.seed,
        "config": {
            "hidden_size": result.config.hidden_size,
            "recurrent_radius": result.config.recurrent_radius,
            "step_size": result.config.step_size,
            "training_episodes": result.config.training_episodes,
            "evaluation_blocks": result.config.evaluation_blocks,
            "checkpoint_interval": result.config.checkpoint_interval,
        },
        "discount": result.discount,
        "trace_decay": result.trace_decay,
        "td0": _result_payload(
            result.td0_post_training,
            result.td0_per_delay,
            result.td0_action_digest,
            result.td0_parameter_digest,
        ),
        "td_lambda": _result_payload(
            result.lambda_post_training,
            result.lambda_per_delay,
            result.lambda_action_digest,
            result.lambda_parameter_digest,
        ),
        "action_sequences_equal": result.action_sequences_equal,
        "training_fixture_digest": result.training_fixture_digest,
        "evaluation_fixture_digest": result.evaluation_fixture_digest,
    }


def credit_comparison_benchmark_payload(
    results: tuple[CreditComparison, ...],
) -> dict[str, object]:
    """Serialize a fixed-seed comparison collection."""
    if not results or any(not isinstance(result, CreditComparison) for result in results):
        raise ValueError("results must be a non-empty tuple of CreditComparison")
    return {
        "experiment": "phase-3a-credit-comparison",
        "diagnostic_only": True,
        "schema_version": 1,
        "seeds": [result.seed for result in results],
        "results": [credit_comparison_payload(result) for result in results],
    }


def run_credit_comparison(
    seed: int = 7,
    config: ActionValueBenchmarkConfig | None = None,
    *,
    discount: float = 0.9,
    trace_decay: float = 0.8,
) -> CreditComparison:
    """Compare TD(0) and persistent-trace readouts on identical lineages."""
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    resolved = ActionValueBenchmarkConfig() if config is None else config
    if not isinstance(resolved, ActionValueBenchmarkConfig):
        raise ValueError("config must be an ActionValueBenchmarkConfig")
    resolved_discount = _validated_unit_interval(discount, "discount")
    resolved_decay = _validated_unit_interval(trace_decay, "trace_decay")

    task = DelayedCueTask()
    bundle = _build_fixture_bundle(seed, resolved)
    td0_policy = _new_policy(seed, resolved)
    lambda_policy = _new_policy(seed, resolved)
    td0 = _new_learner(resolved)
    trace = EligibilityTraceActionValue(
        resolved.hidden_size,
        2,
        step_size=resolved.step_size,
        discount=resolved_discount,
        trace_decay=resolved_decay,
    )
    td0_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x33414354]))
    lambda_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x33414354]))
    td0_training = _train_normal(
        td0_policy, td0, task, bundle.training, td0_rng, resolved
    )
    lambda_training = _train_normal(
        lambda_policy, trace, task, bundle.training, lambda_rng, resolved
    )
    td0_post = _evaluate(
        td0_policy, td0, bundle.evaluation, reset_before_decision=False
    )
    lambda_post = _evaluate(
        lambda_policy, trace, bundle.evaluation, reset_before_decision=False
    )
    return CreditComparison(
        seed=seed,
        config=resolved,
        discount=resolved_discount,
        trace_decay=resolved_decay,
        td0_post_training=td0_post.overall,
        td0_per_delay=td0_post.per_delay,
        lambda_post_training=lambda_post.overall,
        lambda_per_delay=lambda_post.per_delay,
        td0_action_digest=td0_training.action_digest,
        lambda_action_digest=lambda_training.action_digest,
        action_sequences_equal=td0_training.actions == lambda_training.actions,
        training_fixture_digest=bundle.training_fixture_digest,
        lambda_training_fixture_digest=bundle.training_fixture_digest,
        evaluation_fixture_digest=bundle.evaluation_fixture_digest,
        lambda_evaluation_fixture_digest=bundle.evaluation_fixture_digest,
        td0_parameter_digest=td0_training.final_parameter_digest,
        lambda_parameter_digest=lambda_training.final_parameter_digest,
    )


def run_credit_comparison_benchmark(
    seeds: tuple[int, ...] = (7, 17, 29),
    config: ActionValueBenchmarkConfig | None = None,
    *,
    discount: float = 0.9,
    trace_decay: float = 0.8,
) -> tuple[CreditComparison, ...]:
    """Run the credit comparison for the pre-registered seed set."""
    if not isinstance(seeds, tuple) or not seeds:
        raise ValueError("seeds must be a non-empty tuple")
    if any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("seeds must contain non-negative integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must not contain duplicates")
    return tuple(
        run_credit_comparison(
            seed,
            config,
            discount=discount,
            trace_decay=trace_decay,
        )
        for seed in seeds
    )


def _validated_unit_interval(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and in [0.0, 1.0]")
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and in [0.0, 1.0]") from exc
    if not math.isfinite(resolved) or not 0.0 <= resolved <= 1.0:
        raise ValueError(f"{name} must be finite and in [0.0, 1.0]")
    return resolved


def _result_payload(
    overall: AccuracyCount,
    per_delay: tuple[tuple[int, AccuracyCount], ...],
    action_digest: str,
    parameter_digest: str,
) -> dict[str, object]:
    return {
        "post_training": {
            "correct": overall.correct,
            "total": overall.total,
            "accuracy": overall.accuracy,
        },
        "per_delay": {
            str(delay): {
                "correct": score.correct,
                "total": score.total,
                "accuracy": score.accuracy,
            }
            for delay, score in per_delay
        },
        "action_digest": action_digest,
        "parameter_digest": parameter_digest,
    }
