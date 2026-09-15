"""Phase 3B delayed action-to-reward benchmark protocol."""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .action_value_benchmark import (
    ActionValueBenchmarkConfig,
    _build_fixture_bundle,
    _evaluate,
    _new_policy,
)
from .delayed_credit import DelayedRewardQueue, RewardDelivery
from .memory_benchmark import AccuracyCount
from .memory_task import DelayedCueTask
from .phase3b_controls import (
    TimelineAudit,
    delivery_timeline_digest,
    validate_fixed_delay_timeline,
)
from .phase3b_learners import DelayedTD0Adapter
from .reward_learning import _decision_hidden


_ALLOWED_ARMS = ("td0",)
_DEFAULT_REWARD_DELAYS = (0, 1, 3, 5)
_DEFAULT_SEEDS = (7, 17, 29)


@dataclass(frozen=True)
class DelayedCreditConfig:
    hidden_size: int = 64
    recurrent_radius: float = 0.9
    step_size: float = 0.1
    training_episodes: int = 2_000
    evaluation_blocks: int = 20
    checkpoint_interval: int = 100
    reward_delays: tuple[int, ...] = _DEFAULT_REWARD_DELAYS
    discount: float = 0.9
    trace_decay: float = 0.8

    def __post_init__(self) -> None:
        ActionValueBenchmarkConfig(
            hidden_size=self.hidden_size,
            recurrent_radius=self.recurrent_radius,
            step_size=self.step_size,
            training_episodes=self.training_episodes,
            evaluation_blocks=self.evaluation_blocks,
            checkpoint_interval=self.checkpoint_interval,
        )
        if not isinstance(self.reward_delays, tuple) or not self.reward_delays:
            raise ValueError("reward_delays must be a non-empty tuple")
        if any(type(delay) is not int or delay < 0 for delay in self.reward_delays):
            raise ValueError("reward_delays must contain non-negative integers")
        if len(set(self.reward_delays)) != len(self.reward_delays):
            raise ValueError("reward_delays must not contain duplicates")
        if self.reward_delays[0] != 0:
            raise ValueError("reward_delays must start with the immediate control 0")
        for name, value in (("discount", self.discount), ("trace_decay", self.trace_decay)):
            if isinstance(value, bool) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be finite and in [0.0, 1.0]")

    @property
    def action_value_config(self) -> ActionValueBenchmarkConfig:
        return ActionValueBenchmarkConfig(
            hidden_size=self.hidden_size,
            recurrent_radius=self.recurrent_radius,
            step_size=self.step_size,
            training_episodes=self.training_episodes,
            evaluation_blocks=self.evaluation_blocks,
            checkpoint_interval=self.checkpoint_interval,
        )


@dataclass(frozen=True)
class DelayedCreditCheckpoint:
    decision_count: int
    delivery_count: int
    unresolved_credit_count: int
    parameter_digest: str


@dataclass(frozen=True)
class DelayedCreditResult:
    seed: int
    arm: str
    reward_delay: int
    pre_training: AccuracyCount
    post_training: AccuracyCount
    state_reset: AccuracyCount
    per_delay: tuple[tuple[int, AccuracyCount], ...]
    reset_per_delay: tuple[tuple[int, AccuracyCount], ...]
    action_digest: str
    actions: tuple[int, ...]
    training_reward_digest: str
    training_fixture_digest: str
    evaluation_fixture_digest: str
    parameter_digest: str
    pending_feedback: bool
    queue_deliveries: int
    timeline: TimelineAudit
    checkpoints: tuple[DelayedCreditCheckpoint, ...]
    repeatable: bool


@dataclass(frozen=True)
class _ActionTrace:
    actions: tuple[int, ...]

    @property
    def action_digest(self) -> str:
        return hashlib.sha256(bytes(self.actions)).hexdigest()


@dataclass(frozen=True)
class _RunOnce:
    pre_training: object
    post_training: object
    state_reset: object
    action_trace: _ActionTrace
    training_reward_digest: str
    training_fixture_digest: str
    evaluation_fixture_digest: str
    parameter_digest: str
    pending_feedback: bool
    queue_deliveries: int
    timeline: TimelineAudit
    checkpoints: tuple[DelayedCreditCheckpoint, ...]


def run_delayed_credit(
    seed: int,
    reward_delay: int,
    arm: str = "td0",
    config: DelayedCreditConfig | None = None,
) -> DelayedCreditResult:
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    resolved = DelayedCreditConfig() if config is None else config
    if not isinstance(resolved, DelayedCreditConfig):
        raise ValueError("config must be a DelayedCreditConfig")
    if type(reward_delay) is not int or reward_delay not in resolved.reward_delays:
        raise ValueError("reward_delay must be one of config.reward_delays")
    if arm == "td_lambda":
        raise ValueError(
            "TD(lambda) is blocked under the corrected overlapping Phase 3B protocol"
        )
    if arm not in _ALLOWED_ARMS:
        raise ValueError("arm must be 'td0'")
    first = _run_once(seed, reward_delay, resolved)
    second = _run_once(seed, reward_delay, resolved)
    return DelayedCreditResult(
        seed=seed,
        arm=arm,
        reward_delay=reward_delay,
        pre_training=first.pre_training.overall,
        post_training=first.post_training.overall,
        state_reset=first.state_reset.overall,
        per_delay=first.post_training.per_delay,
        reset_per_delay=first.state_reset.per_delay,
        action_digest=first.action_trace.action_digest,
        actions=first.action_trace.actions,
        training_reward_digest=first.training_reward_digest,
        training_fixture_digest=first.training_fixture_digest,
        evaluation_fixture_digest=first.evaluation_fixture_digest,
        parameter_digest=first.parameter_digest,
        pending_feedback=first.pending_feedback,
        queue_deliveries=first.queue_deliveries,
        timeline=first.timeline,
        checkpoints=first.checkpoints,
        repeatable=first == second,
    )


def run_delayed_credit_benchmark(
    seeds: tuple[int, ...] = _DEFAULT_SEEDS,
    config: DelayedCreditConfig | None = None,
    *,
    arm: str = "td0",
) -> tuple[DelayedCreditResult, ...]:
    resolved = DelayedCreditConfig() if config is None else config
    if not isinstance(seeds, tuple) or not seeds:
        raise ValueError("seeds must be a non-empty tuple")
    if any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("seeds must contain non-negative integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must not contain duplicates")
    if arm == "td_lambda":
        raise ValueError(
            "TD(lambda) is blocked under the corrected overlapping Phase 3B protocol"
        )
    if arm not in _ALLOWED_ARMS:
        raise ValueError("arm must be 'td0'")
    return tuple(
        run_delayed_credit(seed, delay, arm, resolved)
        for seed in seeds
        for delay in resolved.reward_delays
    )


def delayed_credit_payload(results: tuple[DelayedCreditResult, ...]) -> dict[str, object]:
    if not results or any(not isinstance(result, DelayedCreditResult) for result in results):
        raise ValueError("results must be a non-empty tuple of DelayedCreditResult")
    return {
        "experiment": "phase-3b-delayed-credit",
        "schema_version": 1,
        "diagnostic_only": True,
        "results": [
            {
                "seed": result.seed,
                "arm": result.arm,
                "reward_delay": result.reward_delay,
                "pre_training": _count_payload(result.pre_training),
                "post_training": _count_payload(result.post_training),
                "state_reset": _count_payload(result.state_reset),
                "per_delay": _per_delay_payload(result.per_delay),
                "reset_per_delay": _per_delay_payload(result.reset_per_delay),
                "action_digest": result.action_digest,
                "training_reward_digest": result.training_reward_digest,
                "training_fixture_digest": result.training_fixture_digest,
                "evaluation_fixture_digest": result.evaluation_fixture_digest,
                "parameter_digest": result.parameter_digest,
                "pending_feedback": result.pending_feedback,
                "queue_deliveries": result.queue_deliveries,
                "timeline": _timeline_payload(result.timeline),
                "checkpoints": [
                    {
                        "decision_count": checkpoint.decision_count,
                        "delivery_count": checkpoint.delivery_count,
                        "unresolved_credit_count": checkpoint.unresolved_credit_count,
                        "parameter_digest": checkpoint.parameter_digest,
                    }
                    for checkpoint in result.checkpoints
                ],
                "repeatable": result.repeatable,
            }
            for result in results
        ],
    }


def _run_once(
    seed: int,
    reward_delay: int,
    config: DelayedCreditConfig,
) -> _RunOnce:
    task = DelayedCueTask()
    action_config = config.action_value_config
    fixtures = _build_fixture_bundle(seed, action_config)
    policy = _new_policy(seed, action_config)
    learner = DelayedTD0Adapter(
        action_config.hidden_size,
        2,
        step_size=action_config.step_size,
    )
    action_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x33414354]))
    queue = DelayedRewardQueue(max_delay=max(config.reward_delays))
    pre_training = _evaluate(policy, learner, fixtures.evaluation, reset_before_decision=False)
    actions: list[int] = []
    rewards: list[float] = []
    deliveries: list[RewardDelivery] = []
    checkpoints: list[DelayedCreditCheckpoint] = []
    max_pending_before_delivery = 0
    max_pending_after_delivery = 0
    decisions_with_prior_feedback_pending = 0

    for decision_step, episode in enumerate(fixtures.training):
        if queue.current_step != decision_step:
            raise RuntimeError("queue and decision clocks diverged")

        prior_pending = queue.pending_count > 0
        hidden = _decision_hidden(policy, episode, reset_before_decision=False)
        decision = learner.select_for_training(hidden, (0, 1), action_rng)
        action = int(decision.action_index)
        reward = float(task.reward(episode, action))
        queue.enqueue(action, reward, reward_delay)

        max_pending_before_delivery = max(
            max_pending_before_delivery, queue.pending_count
        )
        ready = queue.deliver_ready()
        for delivery in ready:
            update = learner.learn(delivery.reward)
            if update.action_index != delivery.action_index:
                raise RuntimeError("learner credit and queue action diverged")
            deliveries.append(delivery)
        max_pending_after_delivery = max(max_pending_after_delivery, queue.pending_count)
        decisions_with_prior_feedback_pending += int(prior_pending)
        actions.append(action)
        rewards.append(reward)

        decision_count = decision_step + 1
        if decision_count % config.checkpoint_interval == 0:
            checkpoints.append(
                DelayedCreditCheckpoint(
                    decision_count=decision_count,
                    delivery_count=len(deliveries),
                    unresolved_credit_count=learner.unresolved_credit_count,
                    parameter_digest=learner.parameter_digest(),
                )
            )

        if decision_count < len(fixtures.training):
            queue.advance()

    terminal_drain_count = 0
    while queue.pending_count:
        queue.advance()
        ready = queue.deliver_ready()
        for delivery in ready:
            update = learner.learn(delivery.reward)
            if update.action_index != delivery.action_index:
                raise RuntimeError("learner credit and queue action diverged")
            deliveries.append(delivery)
            terminal_drain_count += 1

    lag_counts = Counter(
        delivery.delivery_step - delivery.decision_step for delivery in deliveries
    )
    timeline = TimelineAudit(
        action_count=len(actions),
        delivery_count=len(deliveries),
        terminal_drain_count=terminal_drain_count,
        max_pending_before_delivery=max_pending_before_delivery,
        max_pending_after_delivery=max_pending_after_delivery,
        decisions_with_prior_feedback_pending=decisions_with_prior_feedback_pending,
        lag_histogram=tuple(sorted(lag_counts.items())),
        delivery_timeline_digest=delivery_timeline_digest(tuple(deliveries)),
        queue_pending_final=queue.pending_count,
        learner_unresolved_final=learner.unresolved_credit_count,
    )
    validate_fixed_delay_timeline(timeline, reward_delay)

    if queue.pending_count or learner.has_pending_feedback:
        raise RuntimeError("delayed training retained pending feedback")
    post_training = _evaluate(policy, learner, fixtures.evaluation, reset_before_decision=False)
    state_reset = _evaluate(policy, learner, fixtures.evaluation, reset_before_decision=True)
    return _RunOnce(
        pre_training=pre_training,
        post_training=post_training,
        state_reset=state_reset,
        action_trace=_ActionTrace(tuple(actions)),
        training_reward_digest=_reward_digest(tuple(rewards)),
        training_fixture_digest=fixtures.training_fixture_digest,
        evaluation_fixture_digest=fixtures.evaluation_fixture_digest,
        parameter_digest=learner.parameter_digest(),
        pending_feedback=bool(learner.has_pending_feedback),
        queue_deliveries=len(deliveries),
        timeline=timeline,
        checkpoints=tuple(checkpoints),
    )


def _reward_digest(rewards: tuple[float, ...]) -> str:
    values = np.ascontiguousarray(rewards, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(len(rewards).to_bytes(8, "little"))
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _count_payload(count: AccuracyCount) -> dict[str, object]:
    return {"correct": count.correct, "total": count.total, "accuracy": count.accuracy}


def _per_delay_payload(
    rows: tuple[tuple[int, AccuracyCount], ...],
) -> list[dict[str, object]]:
    return [{"delay": delay, **_count_payload(count)} for delay, count in rows]


def _timeline_payload(audit: TimelineAudit) -> dict[str, object]:
    return {
        "action_count": audit.action_count,
        "delivery_count": audit.delivery_count,
        "terminal_drain_count": audit.terminal_drain_count,
        "max_pending_before_delivery": audit.max_pending_before_delivery,
        "max_pending_after_delivery": audit.max_pending_after_delivery,
        "decisions_with_prior_feedback_pending": audit.decisions_with_prior_feedback_pending,
        "lag_histogram": [
            {"lag": lag, "count": count} for lag, count in audit.lag_histogram
        ],
        "delivery_timeline_digest": audit.delivery_timeline_digest,
        "queue_pending_final": audit.queue_pending_final,
        "learner_unresolved_final": audit.learner_unresolved_final,
    }
