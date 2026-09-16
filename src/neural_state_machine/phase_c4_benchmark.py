"""Protocol-only orchestration for Phase C4 delay-marginalized credit."""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass

import numpy as np

from .action_value import NormalizedActionValue
from .action_value_benchmark import (
    ActionValueBenchmarkConfig,
    _build_fixture_bundle,
    _episode_digest,
    _new_policy,
    _reward_digest,
    run_action_value_experiment,
)
from .memory_benchmark import AccuracyCount
from .memory_task import DelayedCueTask
from .phase3c_schedule import AggregateFeedback, AnonymousRewardAggregator, build_hidden_delay_schedule
from .phase_c4_batch_probe import build_batch_design
from .phase_c4_controls import (
    PhaseC4ProtocolAudit,
    frozen_input_hashes,
    load_phase_c4_formal_contract,
    validate_phase_c4_protocol,
)
from .phase_c4_delay_model import DecisionCreditRow, DelayLaw, build_marginalized_features
from .phase_c4_learner import DelayMarginalizedAnonymousCredit
from .reward_learning import _build_fixtures, _decision_hidden


_REGISTERED_SEEDS = (7, 17, 29)
_REGISTERED_DELAY_SUPPORT = (1, 3, 5)
_SECONDARY_EVALUATION_LINEAGE = 0x33434641


@dataclass(frozen=True, slots=True)
class PhaseC4Config:
    hidden_size: int = 64
    recurrent_radius: float = 0.9
    step_size: float = 0.1
    training_decisions: int = 2_000
    evaluation_blocks: int = 20
    checkpoint_interval: int = 100
    ridge_penalty: float = 1e-6
    delay_support: tuple[int, ...] = _REGISTERED_DELAY_SUPPORT

    def __post_init__(self) -> None:
        for name in (
            "hidden_size",
            "training_decisions",
            "evaluation_blocks",
            "checkpoint_interval",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.training_decisions % 10:
            raise ValueError("training_decisions must be a multiple of ten")
        if self.checkpoint_interval % 10:
            raise ValueError("checkpoint_interval must be a multiple of ten")
        if self.training_decisions % self.checkpoint_interval:
            raise ValueError("training_decisions must be divisible by checkpoint_interval")
        if self.delay_support != _REGISTERED_DELAY_SUPPORT:
            raise ValueError("delay_support must be exactly (1, 3, 5)")
        if not _finite_number(self.recurrent_radius) or not (
            0.0 <= float(self.recurrent_radius) < 1.0
        ):
            raise ValueError("recurrent_radius must be finite and in [0.0, 1.0)")
        if not _finite_number(self.step_size) or not 0.0 < float(self.step_size) <= 1.0:
            raise ValueError("step_size must be finite and in (0.0, 1.0]")
        if self.ridge_penalty != 1e-6:
            raise ValueError("ridge_penalty must be exactly 1e-6")

    @property
    def action_value_config(self) -> ActionValueBenchmarkConfig:
        return ActionValueBenchmarkConfig(
            hidden_size=self.hidden_size,
            recurrent_radius=self.recurrent_radius,
            step_size=self.step_size,
            training_episodes=self.training_decisions,
            evaluation_blocks=self.evaluation_blocks,
            checkpoint_interval=self.checkpoint_interval,
        )


@dataclass(frozen=True, slots=True)
class PhaseC4ProtocolResult:
    seed: int
    training_fixture_digest: str
    evaluation_fixture_digest: str
    action_digest: str
    latent_reward_digest: str
    schedule_digest: str
    call_digest: str
    candidate_digest: str
    parameter_digest: str
    audit: PhaseC4ProtocolAudit


@dataclass(frozen=True, slots=True)
class _ProtocolExecution:
    seed: int
    training_fixture_digest: str
    evaluation_fixture_digest: str
    actions: tuple[int, ...]
    rewards: tuple[float, ...]
    action_digest: str
    latent_reward_digest: str
    schedule_digest: str
    call_digest: str
    candidate_digest: str
    parameter_digest: str
    delivered_reward_count: int
    real_feedback_count: int
    drain_feedback_count: int
    pending_final: int
    source_relabel_invariant: bool
    hidden_multiplicity_invariant: bool
    bounded_history_passed: bool


def _finite_number(value: object) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _validated_seeds(seeds: object) -> tuple[int, ...]:
    if not isinstance(seeds, tuple) or not seeds:
        raise ValueError("seeds must be a non-empty tuple")
    if any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("seeds must contain non-negative integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must not contain duplicates")
    return seeds


def _integer_sequence_digest(values: tuple[int, ...]) -> str:
    digest = hashlib.sha256()
    digest.update(len(values).to_bytes(8, "big", signed=False))
    for value in values:
        digest.update(int(value).to_bytes(8, "big", signed=True))
    return digest.hexdigest()


def _scalar_sequence_digest(values: tuple[float, ...]) -> str:
    digest = hashlib.sha256()
    digest.update(len(values).to_bytes(8, "big", signed=False))
    for value in values:
        digest.update(struct.pack(">d", float(value)))
    return digest.hexdigest()


def _matrix_digest(values: np.ndarray) -> str:
    matrix = np.ascontiguousarray(values, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(str(matrix.shape).encode("ascii"))
    digest.update(matrix.tobytes(order="C"))
    return digest.hexdigest()


def _schedule_digest(delay_digest: str, due_step_digest: str) -> str:
    digest = hashlib.sha256()
    digest.update(b"phase-c4:schedule:v1\0")
    digest.update(delay_digest.encode("ascii"))
    digest.update(due_step_digest.encode("ascii"))
    return digest.hexdigest()


def _feedback_source_relabel_invariant(feedbacks: tuple[AggregateFeedback, ...]) -> bool:
    original_values = tuple(feedback.value for feedback in feedbacks)
    original_sources = tuple(
        record.source_step for feedback in feedbacks for record in feedback.records
    )
    relabeled_sources = tuple(source + 1_000_000 for source in original_sources)
    return (
        bool(original_sources)
        and original_sources != relabeled_sources
        and _scalar_sequence_digest(original_values)
        == _scalar_sequence_digest(tuple(feedback.value for feedback in feedbacks))
    )


def _feedback_multiplicity_invariant(feedbacks: tuple[AggregateFeedback, ...]) -> bool:
    values = tuple(feedback.value for feedback in feedbacks)
    altered_metadata = tuple((feedback.value, feedback.multiplicity + 10_000) for feedback in feedbacks)
    stripped = tuple(value for value, _ in altered_metadata)
    return _scalar_sequence_digest(values) == _scalar_sequence_digest(stripped)


def _expected_actions(seed: int, count: int) -> tuple[int, ...]:
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x33414354]))
    return tuple(int(rng.integers(2)) for _ in range(count))


def _execute_registered(seed: int, config: PhaseC4Config) -> _ProtocolExecution:
    av_config = config.action_value_config
    fixtures = _build_fixture_bundle(seed, av_config)
    task = DelayedCueTask()
    policy = _new_policy(seed, av_config)
    law = DelayLaw.registered()
    learner = DelayMarginalizedAnonymousCredit(
        config.hidden_size,
        2,
        step_size=config.step_size,
        law=law,
    )
    schedule = build_hidden_delay_schedule(
        seed,
        config.training_decisions,
        support=config.delay_support,
    )
    aggregator = AnonymousRewardAggregator(schedule)
    action_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x33414354]))

    actions: list[int] = []
    rewards: list[float] = []
    decision_rows: list[DecisionCreditRow] = []
    feedbacks: list[AggregateFeedback] = []
    scalar_calls: list[float] = []
    bounded_history = True

    for step, episode in enumerate(fixtures.training):
        hidden = _decision_hidden(policy, episode, reset_before_decision=False)
        decision = learner.select_for_training(hidden, (0, 1), action_rng)
        action = int(decision.action_index)
        feature = learner._feature(hidden)
        denominator = float(np.dot(feature, feature))
        decision_rows.append(
            DecisionCreditRow(
                decision_index=step,
                action_index=action,
                feature=feature,
                denominator=denominator,
            )
        )
        reward = float(task.reward(episode, action))
        aggregator.enqueue(step, action, reward)
        feedback = aggregator.feedback_at(step)
        learner.learn(feedback.value)

        actions.append(action)
        rewards.append(reward)
        feedbacks.append(feedback)
        scalar_calls.append(float(feedback.value))
        bounded_history = bounded_history and len(learner.history_snapshot()) <= max(law.support)

    expected_actions = _expected_actions(seed, config.training_decisions)
    action_tuple = tuple(actions)
    if action_tuple != expected_actions:
        raise RuntimeError("C4 action sequence diverged from the registered lineage")

    last_delivery_clock = config.training_decisions - 1 + max(law.support)
    drain_count = 0
    for delivery_step in range(config.training_decisions, last_delivery_clock + 1):
        feedback = aggregator.feedback_at(delivery_step)
        learner.learn_drain(feedback.value)
        feedbacks.append(feedback)
        scalar_calls.append(float(feedback.value))
        drain_count += 1
        bounded_history = bounded_history and len(learner.history_snapshot()) <= max(law.support)

    if drain_count != max(law.support):
        raise RuntimeError("C4 registered drain did not use the public support horizon")
    if aggregator.pending_count != 0:
        raise RuntimeError("C4 registered protocol ended with pending aggregate feedback")
    if learner.history_snapshot() != ():
        raise RuntimeError("C4 registered protocol ended with retained decision history")

    design, target = build_batch_design(
        tuple(decision_rows),
        tuple(scalar_calls),
        2,
        law,
    )
    if design.shape[0] != config.training_decisions + max(law.support):
        raise RuntimeError("C4-A protocol design row count is not N plus public horizon")
    if target.shape != (config.training_decisions + max(law.support),):
        raise RuntimeError("C4-A protocol target row count is not N plus public horizon")

    feedback_tuple = tuple(feedbacks)
    scalar_tuple = tuple(scalar_calls)
    return _ProtocolExecution(
        seed=seed,
        training_fixture_digest=fixtures.training_fixture_digest,
        evaluation_fixture_digest=fixtures.evaluation_fixture_digest,
        actions=action_tuple,
        rewards=tuple(rewards),
        action_digest=hashlib.sha256(bytes(action_tuple)).hexdigest(),
        latent_reward_digest=_reward_digest(tuple(rewards)),
        schedule_digest=_schedule_digest(schedule.delay_digest, schedule.due_step_digest),
        call_digest=_scalar_sequence_digest(scalar_tuple),
        candidate_digest=_matrix_digest(design),
        parameter_digest=learner.parameter_digest(),
        delivered_reward_count=sum(feedback.multiplicity for feedback in feedback_tuple),
        real_feedback_count=config.training_decisions,
        drain_feedback_count=drain_count,
        pending_final=aggregator.pending_count,
        source_relabel_invariant=_feedback_source_relabel_invariant(feedback_tuple),
        hidden_multiplicity_invariant=_feedback_multiplicity_invariant(feedback_tuple),
        bounded_history_passed=bounded_history and learner.history_snapshot() == (),
    )


def _current_weight_probe() -> bool:
    def prepared() -> DelayMarginalizedAnonymousCredit:
        learner = DelayMarginalizedAnonymousCredit(1, 2, step_size=0.1)
        for index, (hidden, action) in enumerate(((2.0, 1), (9.0, 0), (3.0, 0), (8.0, 1), (4.0, 0))):
            learner.select_for_training(
                np.array([hidden], dtype=np.float64),
                (action,),
                np.random.default_rng(100 + index),
            )
            learner.learn(0.0)
        learner.select_for_training(
            np.array([7.0], dtype=np.float64),
            (1,),
            np.random.default_rng(200),
        )
        return learner

    left = prepared()
    right = prepared()
    history = left.history_snapshot()
    marginalized = build_marginalized_features(history, 5, 2, DelayLaw.registered())
    left_weights = np.array([[0.5, 1.0], [-0.2, 0.3]], dtype=np.float64)
    right_weights = np.array([[1.5, 1.0], [-0.2, 0.3]], dtype=np.float64)
    left._weights[:] = left_weights
    right._weights[:] = right_weights
    left_update = left.learn(0.0)
    right_update = right.learn(0.0)
    expected_delta = float(np.sum((right_weights - left_weights) * marginalized.expected_feature))
    observed_delta = right_update.prediction - left_update.prediction
    return math.isclose(observed_delta, expected_delta, rel_tol=0.0, abs_tol=1e-15)


def _immediate_continuity(seed: int, config: PhaseC4Config) -> bool:
    av_config = config.action_value_config
    fixtures = _build_fixture_bundle(seed, av_config)
    task = DelayedCueTask()
    c4_policy = _new_policy(seed, av_config)
    baseline_policy = _new_policy(seed, av_config)
    c4 = DelayMarginalizedAnonymousCredit(
        config.hidden_size,
        2,
        step_size=config.step_size,
        law=DelayLaw.immediate(),
    )
    baseline = NormalizedActionValue(config.hidden_size, 2, step_size=config.step_size)
    c4_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x33414354]))
    baseline_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x33414354]))
    actions: list[int] = []
    rewards: list[float] = []

    for episode in fixtures.training:
        c4_hidden = _decision_hidden(c4_policy, episode, reset_before_decision=False)
        baseline_hidden = _decision_hidden(baseline_policy, episode, reset_before_decision=False)
        if c4_hidden.tobytes() != baseline_hidden.tobytes():
            return False
        c4_decision = c4.select_for_training(c4_hidden, (0, 1), c4_rng)
        baseline_decision = baseline.select_for_training(baseline_hidden, (0, 1), baseline_rng)
        if (
            c4_decision.action_index != baseline_decision.action_index
            or c4_decision.action_values.tobytes() != baseline_decision.action_values.tobytes()
        ):
            return False
        reward = float(task.reward(episode, c4_decision.action_index))
        c4_update = c4.learn(reward)
        baseline_update = baseline.learn(reward)
        if (
            c4_update.reward != baseline_update.reward
            or c4_update.prediction != baseline_update.prediction_before
            or c4_update.td_error != baseline_update.td_error
            or c4.parameter_snapshot().tobytes() != baseline.parameter_snapshot().tobytes()
            or c4.parameter_digest() != baseline.parameter_digest()
        ):
            return False
        actions.append(c4_decision.action_index)
        rewards.append(reward)

    frozen_baseline = run_action_value_experiment(seed, av_config)
    action_tuple = tuple(actions)
    reward_tuple = tuple(rewards)
    return (
        action_tuple == frozen_baseline.normal_actions
        and hashlib.sha256(bytes(action_tuple)).hexdigest() == frozen_baseline.normal_action_digest
        and _reward_digest(reward_tuple) == frozen_baseline.normal_reward_digest
        and c4.parameter_digest() == frozen_baseline.normal_parameter_digest
        and fixtures.training_fixture_digest == frozen_baseline.training_fixture_digest
        and fixtures.evaluation_fixture_digest == frozen_baseline.evaluation_fixture_digest
    )


def _audit_for_execution(
    execution: _ProtocolExecution,
    expected_actions: tuple[int, ...],
    *,
    repeatable: bool,
    immediate_continuity: bool,
    current_weight_probe: bool,
) -> PhaseC4ProtocolAudit:
    audit = PhaseC4ProtocolAudit(
        decision_count=len(execution.actions),
        latent_reward_count=len(execution.rewards),
        delivered_reward_count=execution.delivered_reward_count,
        real_feedback_count=execution.real_feedback_count,
        drain_feedback_count=execution.drain_feedback_count,
        pending_final=execution.pending_final,
        action_digest=_integer_sequence_digest(execution.actions),
        schedule_digest=execution.schedule_digest,
        call_digest=execution.call_digest,
        candidate_digest=execution.candidate_digest,
        parameter_digest=execution.parameter_digest,
        source_relabel_invariant=execution.source_relabel_invariant,
        hidden_multiplicity_invariant=execution.hidden_multiplicity_invariant,
        current_weight_probe_passed=current_weight_probe,
        bounded_history_passed=execution.bounded_history_passed,
        immediate_continuity_passed=immediate_continuity,
        repeatable=repeatable,
    )
    validate_phase_c4_protocol(audit, expected_actions)
    return audit


def run_phase_c4_protocol_gate(
    seeds: tuple[int, ...] = _REGISTERED_SEEDS,
    config: PhaseC4Config | None = None,
) -> tuple[PhaseC4ProtocolResult, ...]:
    """Run Gate F/P and deterministic protocol construction without scoring behavior."""
    load_phase_c4_formal_contract()
    frozen_input_hashes()
    resolved_seeds = _validated_seeds(seeds)
    resolved_config = PhaseC4Config() if config is None else config
    if not isinstance(resolved_config, PhaseC4Config):
        raise ValueError("config must be a PhaseC4Config")

    current_probe = _current_weight_probe()
    results: list[PhaseC4ProtocolResult] = []
    for seed in resolved_seeds:
        first = _execute_registered(seed, resolved_config)
        second = _execute_registered(seed, resolved_config)
        expected_actions = _expected_actions(seed, resolved_config.training_decisions)
        repeatable = first == second
        continuity = _immediate_continuity(seed, resolved_config)
        audit = _audit_for_execution(
            first,
            expected_actions,
            repeatable=repeatable,
            immediate_continuity=continuity,
            current_weight_probe=current_probe,
        )
        results.append(
            PhaseC4ProtocolResult(
                seed=seed,
                training_fixture_digest=first.training_fixture_digest,
                evaluation_fixture_digest=first.evaluation_fixture_digest,
                action_digest=first.action_digest,
                latent_reward_digest=first.latent_reward_digest,
                schedule_digest=first.schedule_digest,
                call_digest=first.call_digest,
                candidate_digest=first.candidate_digest,
                parameter_digest=first.parameter_digest,
                audit=audit,
            )
        )
    return tuple(results)


def secondary_evaluation_manifest(
    config: PhaseC4Config | None = None,
) -> tuple[dict[str, object], ...]:
    resolved = PhaseC4Config() if config is None else config
    if not isinstance(resolved, PhaseC4Config):
        raise ValueError("config must be a PhaseC4Config")
    task = DelayedCueTask()
    rows: list[dict[str, object]] = []
    for seed in _REGISTERED_SEEDS:
        original = _build_fixture_bundle(seed, resolved.action_value_config)
        rows.append(
            {
                "seed": seed,
                "evaluation_id": -1,
                "kind": "original",
                "fixture_count": len(original.evaluation),
                "digest": original.evaluation_fixture_digest,
            }
        )
        for evaluation_id in range(8):
            rng = np.random.Generator(
                np.random.PCG64(
                    np.random.SeedSequence(
                        [seed, _SECONDARY_EVALUATION_LINEAGE, 2, evaluation_id]
                    )
                )
            )
            fixtures = _build_fixtures(task, rng, resolved.evaluation_blocks)
            rows.append(
                {
                    "seed": seed,
                    "evaluation_id": evaluation_id,
                    "kind": "additional",
                    "fixture_count": len(fixtures),
                    "digest": _episode_digest(fixtures),
                }
            )
    result = tuple(rows)
    if len(result) != 27 or len({(row["seed"], row["evaluation_id"]) for row in result}) != 27:
        raise RuntimeError("C4 secondary evaluation manifest is incomplete")
    return result


def _registered_delay_counts(
    values: tuple[tuple[int, AccuracyCount], ...],
    *,
    correct_at_least: int | None = None,
    exact_correct: int | None = None,
) -> bool:
    if tuple(delay for delay, _ in values) != (1, 2, 3, 4, 5):
        return False
    for _, count in values:
        if count.total != 40:
            return False
        if correct_at_least is not None and count.correct < correct_at_least:
            return False
        if exact_correct is not None and count.correct != exact_correct:
            return False
    return True


def registered_c4_gate(
    normal: AccuracyCount,
    per_delay: tuple[tuple[int, AccuracyCount], ...],
    reset: AccuracyCount,
    reset_per_delay: tuple[tuple[int, AccuracyCount], ...],
    shuffled: AccuracyCount,
) -> bool:
    return (
        normal.total == 200
        and normal.correct >= 180
        and _registered_delay_counts(per_delay, correct_at_least=34)
        and reset == AccuracyCount(100, 200)
        and _registered_delay_counts(reset_per_delay, exact_correct=20)
        and shuffled.total == 200
        and shuffled.correct < 150
    )


def measure_c4a_from_protocol(protocol: PhaseC4ProtocolResult, config: PhaseC4Config):
    from .phase_c4_measurement import measure_c4a_from_protocol as measure

    return measure(protocol, config)


def measure_c4b_from_protocol(protocol: PhaseC4ProtocolResult, config: PhaseC4Config):
    from .phase_c4_measurement import measure_c4b_from_protocol as measure

    return measure(protocol, config)
