"""Protocol-only orchestration for Phase 3C anonymous temporal credit."""

from __future__ import annotations

import hashlib
import math
import struct
from collections import Counter
from dataclasses import dataclass, replace

import numpy as np

from .action_value_benchmark import (
    ActionValueBenchmarkConfig,
    _build_fixture_bundle,
    _new_policy,
    _reward_digest,
    run_action_value_experiment,
)
from .memory_task import DelayedCueTask
from .phase3c_controls import (
    AnonymousProtocolAudit,
    integer_sequence_digest,
    learner_call_digest,
    validate_anonymous_protocol,
)
from .phase3c_formal_contract import load_phase3c_formal_contract
from .phase3c_learners import (
    AnonymousCurrentStepTD0,
    NormalizedAnonymousEligibilityCredit,
)
from .phase3c_schedule import (
    AggregateFeedback,
    AnonymousRewardAggregator,
    HiddenDelaySchedule,
    build_hidden_delay_schedule,
)
from .reward_learning import _decision_hidden


_REGISTERED_DELAY_SUPPORT = (1, 3, 5)
_ARMS = ("td0", "eligibility")


@dataclass(frozen=True)
class AnonymousCreditConfig:
    hidden_size: int = 64
    recurrent_radius: float = 0.9
    step_size: float = 0.1
    training_decisions: int = 2_000
    evaluation_blocks: int = 20
    checkpoint_interval: int = 100
    delay_support: tuple[int, ...] = _REGISTERED_DELAY_SUPPORT
    discount: float = 0.9
    trace_decay: float = 0.8

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
        if not _finite_number(self.recurrent_radius) or not 0.0 <= self.recurrent_radius < 1.0:
            raise ValueError("recurrent_radius must be finite and in [0.0, 1.0)")
        if not _finite_number(self.step_size) or not 0.0 < self.step_size <= 1.0:
            raise ValueError("step_size must be finite and in (0.0, 1.0]")
        for name in ("discount", "trace_decay"):
            value = getattr(self, name)
            if not _finite_number(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0.0, 1.0]")

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


@dataclass(frozen=True)
class Phase3CProtocolResult:
    seed: int
    arm: str
    training_fixture_digest: str
    evaluation_fixture_digest: str
    action_digest: str
    latent_reward_digest: str
    parameter_digest: str
    audit: AnonymousProtocolAudit
    immediate_continuity: bool
    repeatable: bool


@dataclass(frozen=True)
class _ProtocolExecution:
    seed: int
    arm: str
    training_fixture_digest: str
    evaluation_fixture_digest: str
    actions: tuple[int, ...]
    rewards: tuple[float, ...]
    action_digest: str
    latent_reward_digest: str
    parameter_digest: str
    audit: AnonymousProtocolAudit | None


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


def _action_digest(actions: tuple[int, ...]) -> str:
    if any(action not in (0, 1) for action in actions):
        raise ValueError("actions must contain only registered action indices")
    return hashlib.sha256(bytes(actions)).hexdigest()


def _aggregate_feedback_digest(feedbacks: tuple[AggregateFeedback, ...]) -> str:
    digest = hashlib.sha256()
    digest.update(b"phase3c:aggregate-feedback:v1\0")
    digest.update(len(feedbacks).to_bytes(8, "big", signed=False))
    for feedback in feedbacks:
        digest.update(feedback.delivery_step.to_bytes(8, "big", signed=True))
        digest.update(struct.pack(">d", feedback.value))
    return digest.hexdigest()


def _multiplicity_digest(schedule: HiddenDelaySchedule) -> str:
    counts = Counter(schedule.due_steps)
    flattened = tuple(
        value
        for due_step, multiplicity in sorted(counts.items())
        for value in (due_step, multiplicity)
    )
    return integer_sequence_digest(flattened)


def _new_arm(
    arm: str,
    config: AnonymousCreditConfig,
    *,
    immediate_control: bool,
) -> AnonymousCurrentStepTD0 | NormalizedAnonymousEligibilityCredit:
    if arm == "td0":
        return AnonymousCurrentStepTD0(
            config.hidden_size,
            2,
            step_size=config.step_size,
        )
    if arm == "eligibility":
        return NormalizedAnonymousEligibilityCredit(
            config.hidden_size,
            2,
            step_size=config.step_size,
            discount=0.0 if immediate_control else config.discount,
            trace_decay=config.trace_decay,
        )
    raise ValueError("arm must be td0 or eligibility")


def _source_relabel_invariant(feedbacks: tuple[AggregateFeedback, ...]) -> bool:
    original = tuple(feedback.value for feedback in feedbacks)
    relabeled = tuple(
        float(sum(record.reward for record in feedback.records))
        for feedback in feedbacks
    )
    return learner_call_digest(original) == learner_call_digest(relabeled)


def _hidden_multiplicity_invariant(feedbacks: tuple[AggregateFeedback, ...]) -> bool:
    original = tuple(feedback.value for feedback in feedbacks)
    metadata_changed = tuple(
        (feedback.value, feedback.multiplicity + 10_000)
        for feedback in feedbacks
    )
    stripped = tuple(value for value, _ in metadata_changed)
    return learner_call_digest(original) == learner_call_digest(stripped)


def _build_audit(
    *,
    arm: str,
    config: AnonymousCreditConfig,
    schedule: HiddenDelaySchedule,
    rewards: tuple[float, ...],
    feedbacks: tuple[AggregateFeedback, ...],
    real_feedback_count: int,
    drain_feedback_count: int,
    pending_final: int,
    trace_reset_count: int,
) -> AnonymousProtocolAudit:
    call_values = tuple(feedback.value for feedback in feedbacks)
    delay_histogram = tuple(sorted(Counter(schedule.delays).items()))
    rho = config.discount * config.trace_decay
    audit = AnonymousProtocolAudit(
        arm=arm,
        action_count=len(rewards),
        latent_record_count=len(rewards),
        delivered_record_count=sum(feedback.multiplicity for feedback in feedbacks),
        real_feedback_count=real_feedback_count,
        drain_feedback_count=drain_feedback_count,
        queue_pending_final=pending_final,
        delay_histogram=delay_histogram,
        inversion_count=schedule.inversion_count,
        collision_step_count=schedule.collision_step_count,
        multiplicity_histogram=schedule.multiplicity_histogram,
        delay_digest=schedule.delay_digest,
        due_step_digest=schedule.due_step_digest,
        multiplicity_digest=_multiplicity_digest(schedule),
        aggregate_feedback_digest=_aggregate_feedback_digest(feedbacks),
        learner_call_digest=learner_call_digest(call_values),
        latent_reward_sum=float(sum(rewards)),
        aggregate_feedback_sum=float(sum(call_values)),
        source_relabel_invariant=_source_relabel_invariant(feedbacks),
        hidden_multiplicity_invariant=_hidden_multiplicity_invariant(feedbacks),
        trace_reset_count=trace_reset_count,
        trace_coefficients=(
            (0, 1.0),
            (1, rho),
            (3, rho**3),
            (5, rho**5),
        ),
    )
    validate_anonymous_protocol(audit, action_count=len(rewards))
    return audit


def _execute_protocol(
    seed: int,
    arm: str,
    config: AnonymousCreditConfig,
    *,
    immediate_control: bool,
) -> _ProtocolExecution:
    av_config = config.action_value_config
    task = DelayedCueTask()
    fixtures = _build_fixture_bundle(seed, av_config)
    support = (0,) if immediate_control else config.delay_support
    schedule = build_hidden_delay_schedule(
        seed,
        config.training_decisions,
        support=support,
    )
    policy = _new_policy(seed, av_config)
    learner = _new_arm(arm, config, immediate_control=immediate_control)
    action_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x33414354]))

    expected_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x33414354]))
    expected_actions = tuple(
        int(expected_rng.integers(2)) for _ in range(config.training_decisions)
    )

    aggregator = AnonymousRewardAggregator(schedule)
    actions: list[int] = []
    rewards: list[float] = []
    feedbacks: list[AggregateFeedback] = []

    for step, episode in enumerate(fixtures.training):
        hidden = _decision_hidden(
            policy,
            episode,
            reset_before_decision=False,
        )
        decision = learner.select_for_training(hidden, (0, 1), action_rng)
        action = int(decision.action_index)
        reward = float(task.reward(episode, action))
        aggregator.enqueue(step, action, reward)
        feedback = aggregator.feedback_at(step)
        learner.learn(feedback.value)
        actions.append(action)
        rewards.append(reward)
        feedbacks.append(feedback)

    action_tuple = tuple(actions)
    reward_tuple = tuple(rewards)
    if action_tuple != expected_actions:
        raise RuntimeError("learner action sequence diverged from the pre-generated lineage")

    drain_feedback_count = 0
    last_due_step = max(schedule.due_steps)
    for delivery_step in range(config.training_decisions, last_due_step + 1):
        feedback = aggregator.feedback_at(delivery_step)
        learner.learn_drain(feedback.value)
        feedbacks.append(feedback)
        drain_feedback_count += 1

    parameter_digest = learner.parameter_digest()
    if isinstance(learner, NormalizedAnonymousEligibilityCredit):
        learner.end_run()
        trace_reset_count = learner.trace_reset_count
    else:
        trace_reset_count = 0

    audit = None
    if not immediate_control:
        audit = _build_audit(
            arm=arm,
            config=config,
            schedule=schedule,
            rewards=reward_tuple,
            feedbacks=tuple(feedbacks),
            real_feedback_count=config.training_decisions,
            drain_feedback_count=drain_feedback_count,
            pending_final=aggregator.pending_count,
            trace_reset_count=trace_reset_count,
        )

    return _ProtocolExecution(
        seed=seed,
        arm=arm,
        training_fixture_digest=fixtures.training_fixture_digest,
        evaluation_fixture_digest=fixtures.evaluation_fixture_digest,
        actions=action_tuple,
        rewards=reward_tuple,
        action_digest=_action_digest(action_tuple),
        latent_reward_digest=_reward_digest(reward_tuple),
        parameter_digest=parameter_digest,
        audit=audit,
    )


def _immediate_continuity(
    seed: int,
    arm: str,
    config: AnonymousCreditConfig,
) -> bool:
    baseline = run_action_value_experiment(seed, config.action_value_config)
    control = _execute_protocol(
        seed,
        arm,
        config,
        immediate_control=True,
    )
    task = DelayedCueTask()
    fixtures = _build_fixture_bundle(seed, config.action_value_config)
    expected_rewards = tuple(
        float(task.reward(episode, action))
        for episode, action in zip(fixtures.training, control.actions, strict=True)
    )
    return (
        control.actions == baseline.normal_actions
        and control.action_digest == baseline.normal_action_digest
        and control.rewards == expected_rewards
        and control.latent_reward_digest == baseline.normal_reward_digest
        and control.parameter_digest == baseline.normal_parameter_digest
        and control.training_fixture_digest == baseline.training_fixture_digest
        and control.evaluation_fixture_digest == baseline.evaluation_fixture_digest
    )


def _public_result(
    execution: _ProtocolExecution,
    *,
    immediate_continuity: bool,
    repeatable: bool,
) -> Phase3CProtocolResult:
    if execution.audit is None:
        raise RuntimeError("registered protocol result requires a Gate P audit")
    return Phase3CProtocolResult(
        seed=execution.seed,
        arm=execution.arm,
        training_fixture_digest=execution.training_fixture_digest,
        evaluation_fixture_digest=execution.evaluation_fixture_digest,
        action_digest=execution.action_digest,
        latent_reward_digest=execution.latent_reward_digest,
        parameter_digest=execution.parameter_digest,
        audit=execution.audit,
        immediate_continuity=immediate_continuity,
        repeatable=repeatable,
    )


def _require_matched_lineage(
    td0: Phase3CProtocolResult,
    eligibility: Phase3CProtocolResult,
) -> None:
    if (
        td0.training_fixture_digest != eligibility.training_fixture_digest
        or td0.evaluation_fixture_digest != eligibility.evaluation_fixture_digest
        or td0.action_digest != eligibility.action_digest
        or td0.latent_reward_digest != eligibility.latent_reward_digest
        or td0.audit.delay_digest != eligibility.audit.delay_digest
        or td0.audit.due_step_digest != eligibility.audit.due_step_digest
        or td0.audit.multiplicity_digest != eligibility.audit.multiplicity_digest
        or td0.audit.aggregate_feedback_digest != eligibility.audit.aggregate_feedback_digest
        or td0.audit.learner_call_digest != eligibility.audit.learner_call_digest
    ):
        raise RuntimeError("Phase 3C arms do not share the registered protocol lineage")


def run_phase3c_protocol_gate(
    seeds: tuple[int, ...] = (7, 17, 29),
    config: AnonymousCreditConfig | None = None,
) -> tuple[Phase3CProtocolResult, ...]:
    """Run structural Gate F/Gate P checks without anonymous behavioral evaluation."""
    load_phase3c_formal_contract()
    resolved_seeds = _validated_seeds(seeds)
    resolved_config = AnonymousCreditConfig() if config is None else config
    if not isinstance(resolved_config, AnonymousCreditConfig):
        raise ValueError("config must be an AnonymousCreditConfig")

    results: list[Phase3CProtocolResult] = []
    for seed in resolved_seeds:
        seed_results: dict[str, Phase3CProtocolResult] = {}
        for arm in _ARMS:
            first = _execute_protocol(
                seed,
                arm,
                resolved_config,
                immediate_control=False,
            )
            second = _execute_protocol(
                seed,
                arm,
                resolved_config,
                immediate_control=False,
            )
            continuity = _immediate_continuity(seed, arm, resolved_config)
            result = _public_result(
                first,
                immediate_continuity=continuity,
                repeatable=first == second,
            )
            if not result.immediate_continuity:
                raise RuntimeError("Phase 3C immediate control diverged from Phase 3A")
            if not result.repeatable:
                raise RuntimeError("Phase 3C registered protocol is not repeatable")
            seed_results[arm] = result
        _require_matched_lineage(seed_results["td0"], seed_results["eligibility"])
        results.extend(seed_results[arm] for arm in _ARMS)
    return tuple(results)
