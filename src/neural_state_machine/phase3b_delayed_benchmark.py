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
    _permute_reward_blocks,
)
from .delayed_credit import DelayedRewardQueue, RewardDelivery
from .memory_benchmark import AccuracyCount
from .memory_task import DelayedCueTask
from .phase3b_controls import (
    TimelineAudit,
    action_sequences_equal,
    delivery_timeline_digest,
    reward_block_multisets_equal,
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
class _TrainingRun:
    actions: tuple[int, ...]
    rewards: tuple[float, ...]
    action_digest: str
    reward_digest: str
    parameter_digest: str
    timeline: TimelineAudit
    checkpoints: tuple[DelayedCreditCheckpoint, ...]
    pending_feedback: bool


@dataclass(frozen=True)
class DelayedCreditResult:
    seed: int
    arm: str
    reward_delay: int
    pre_training: AccuracyCount
    post_training: AccuracyCount
    state_reset: AccuracyCount
    shuffled_control: AccuracyCount
    per_delay: tuple[tuple[int, AccuracyCount], ...]
    reset_per_delay: tuple[tuple[int, AccuracyCount], ...]
    shuffled_per_delay: tuple[tuple[int, AccuracyCount], ...]
    action_digest: str
    actions: tuple[int, ...]
    training_reward_digest: str
    normal_action_digest: str
    shuffled_action_digest: str
    normal_reward_assignment_digest: str
    shuffled_reward_assignment_digest: str
    training_fixture_digest: str
    evaluation_fixture_digest: str
    parameter_digest: str
    shuffled_parameter_digest: str
    pending_feedback: bool
    queue_deliveries: int
    timeline: TimelineAudit
    normal_timeline: TimelineAudit
    shuffled_timeline: TimelineAudit
    checkpoints: tuple[DelayedCreditCheckpoint, ...]
    normal_checkpoints: tuple[DelayedCreditCheckpoint, ...]
    shuffled_checkpoints: tuple[DelayedCreditCheckpoint, ...]
    action_sequences_equal: bool
    reward_block_multisets_equal: bool
    behavior_passed: bool
    repeatable: bool


@dataclass(frozen=True)
class _RunOnce:
    pre_training: object
    post_training: object
    state_reset: object
    shuffled_control: object
    normal_training: _TrainingRun
    shuffled_training: _TrainingRun
    training_fixture_digest: str
    evaluation_fixture_digest: str
    action_sequences_equal: bool
    reward_block_multisets_equal: bool
    behavior_passed: bool


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
    normal = first.normal_training
    shuffled = first.shuffled_training
    return DelayedCreditResult(
        seed=seed,
        arm=arm,
        reward_delay=reward_delay,
        pre_training=first.pre_training.overall,
        post_training=first.post_training.overall,
        state_reset=first.state_reset.overall,
        shuffled_control=first.shuffled_control.overall,
        per_delay=first.post_training.per_delay,
        reset_per_delay=first.state_reset.per_delay,
        shuffled_per_delay=first.shuffled_control.per_delay,
        action_digest=normal.action_digest,
        actions=normal.actions,
        training_reward_digest=normal.reward_digest,
        normal_action_digest=normal.action_digest,
        shuffled_action_digest=shuffled.action_digest,
        normal_reward_assignment_digest=normal.reward_digest,
        shuffled_reward_assignment_digest=shuffled.reward_digest,
        training_fixture_digest=first.training_fixture_digest,
        evaluation_fixture_digest=first.evaluation_fixture_digest,
        parameter_digest=normal.parameter_digest,
        shuffled_parameter_digest=shuffled.parameter_digest,
        pending_feedback=normal.pending_feedback,
        queue_deliveries=normal.timeline.delivery_count,
        timeline=normal.timeline,
        normal_timeline=normal.timeline,
        shuffled_timeline=shuffled.timeline,
        checkpoints=normal.checkpoints,
        normal_checkpoints=normal.checkpoints,
        shuffled_checkpoints=shuffled.checkpoints,
        action_sequences_equal=first.action_sequences_equal,
        reward_block_multisets_equal=first.reward_block_multisets_equal,
        behavior_passed=first.behavior_passed,
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
                "shuffled_control": _count_payload(result.shuffled_control),
                "per_delay": _per_delay_payload(result.per_delay),
                "reset_per_delay": _per_delay_payload(result.reset_per_delay),
                "shuffled_per_delay": _per_delay_payload(result.shuffled_per_delay),
                "normal_action_digest": result.normal_action_digest,
                "shuffled_action_digest": result.shuffled_action_digest,
                "normal_reward_assignment_digest": result.normal_reward_assignment_digest,
                "shuffled_reward_assignment_digest": result.shuffled_reward_assignment_digest,
                "training_fixture_digest": result.training_fixture_digest,
                "evaluation_fixture_digest": result.evaluation_fixture_digest,
                "parameter_digest": result.parameter_digest,
                "shuffled_parameter_digest": result.shuffled_parameter_digest,
                "normal_timeline": _timeline_payload(result.normal_timeline),
                "shuffled_timeline": _timeline_payload(result.shuffled_timeline),
                "normal_checkpoints": _checkpoint_payload(result.normal_checkpoints),
                "shuffled_checkpoints": _checkpoint_payload(result.shuffled_checkpoints),
                "action_sequences_equal": result.action_sequences_equal,
                "reward_block_multisets_equal": result.reward_block_multisets_equal,
                "behavior_passed": result.behavior_passed,
                "repeatable": result.repeatable,
            }
            for result in results
        ],
    }


def _run_once(seed: int, reward_delay: int, config: DelayedCreditConfig) -> _RunOnce:
    task = DelayedCueTask()
    action_config = config.action_value_config
    fixtures = _build_fixture_bundle(seed, action_config)

    schedule_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x33414354]))
    expected_actions = tuple(
        (0, 1)[int(schedule_rng.integers(2))] for _ in fixtures.training
    )
    normal_rewards = tuple(
        float(task.reward(episode, action))
        for episode, action in zip(fixtures.training, expected_actions, strict=True)
    )
    shuffle_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x33534846]))
    shuffled_rewards = _permute_reward_blocks(normal_rewards, shuffle_rng, block_size=10)

    normal_policy = _new_policy(seed, action_config)
    shuffled_policy = _new_policy(seed, action_config)
    normal_learner = DelayedTD0Adapter(
        action_config.hidden_size, 2, step_size=action_config.step_size
    )
    shuffled_learner = DelayedTD0Adapter(
        action_config.hidden_size, 2, step_size=action_config.step_size
    )
    pre_training = _evaluate(
        normal_policy, normal_learner, fixtures.evaluation, reset_before_decision=False
    )

    normal_training = _train_schedule(
        normal_policy,
        normal_learner,
        fixtures.training,
        expected_actions,
        normal_rewards,
        np.random.default_rng(np.random.SeedSequence([seed, 0x33414354])),
        reward_delay,
        config,
    )
    shuffled_training = _train_schedule(
        shuffled_policy,
        shuffled_learner,
        fixtures.training,
        expected_actions,
        shuffled_rewards,
        np.random.default_rng(np.random.SeedSequence([seed, 0x33414354])),
        reward_delay,
        config,
    )

    same_actions = action_sequences_equal(
        normal_training.actions, shuffled_training.actions
    )
    same_reward_multisets = reward_block_multisets_equal(
        normal_training.rewards, shuffled_training.rewards, block_size=10
    )
    if not same_actions:
        raise RuntimeError("normal and shuffled action lineages diverged")
    if not same_reward_multisets:
        raise RuntimeError("shuffled rewards changed a registered block multiset")
    if (
        normal_training.timeline.delivery_timeline_digest
        != shuffled_training.timeline.delivery_timeline_digest
    ):
        raise RuntimeError("normal and shuffled delivery timelines diverged")

    post_training = _evaluate(
        normal_policy, normal_learner, fixtures.evaluation, reset_before_decision=False
    )
    state_reset = _evaluate(
        normal_policy, normal_learner, fixtures.evaluation, reset_before_decision=True
    )
    shuffled_control = _evaluate(
        shuffled_policy,
        shuffled_learner,
        fixtures.evaluation,
        reset_before_decision=False,
    )
    behavior_passed = _behavior_gate(
        post_training.overall,
        post_training.per_delay,
        state_reset.overall,
        state_reset.per_delay,
        shuffled_control.overall,
    )
    return _RunOnce(
        pre_training=pre_training,
        post_training=post_training,
        state_reset=state_reset,
        shuffled_control=shuffled_control,
        normal_training=normal_training,
        shuffled_training=shuffled_training,
        training_fixture_digest=fixtures.training_fixture_digest,
        evaluation_fixture_digest=fixtures.evaluation_fixture_digest,
        action_sequences_equal=same_actions,
        reward_block_multisets_equal=same_reward_multisets,
        behavior_passed=behavior_passed,
    )


def _train_schedule(
    policy: object,
    learner: DelayedTD0Adapter,
    fixtures: tuple[object, ...],
    expected_actions: tuple[int, ...],
    reward_schedule: tuple[float, ...],
    action_rng: np.random.Generator,
    reward_delay: int,
    config: DelayedCreditConfig,
) -> _TrainingRun:
    if len(fixtures) != len(expected_actions) or len(fixtures) != len(reward_schedule):
        raise ValueError("training schedule lengths must match fixtures")
    queue = DelayedRewardQueue(max_delay=max(config.reward_delays))
    deliveries: list[RewardDelivery] = []
    checkpoints: list[DelayedCreditCheckpoint] = []
    actions: list[int] = []
    max_pending_before_delivery = 0
    max_pending_after_delivery = 0
    decisions_with_prior_feedback_pending = 0

    for decision_step, episode in enumerate(fixtures):
        if queue.current_step != decision_step:
            raise RuntimeError("queue and decision clocks diverged")
        prior_pending = queue.pending_count > 0
        hidden = _decision_hidden(policy, episode, reset_before_decision=False)
        decision = learner.select_for_training(hidden, (0, 1), action_rng)
        action = int(decision.action_index)
        if action != expected_actions[decision_step]:
            raise RuntimeError("action lineage diverged from the registered schedule")
        queue.enqueue(action, reward_schedule[decision_step], reward_delay)

        max_pending_before_delivery = max(
            max_pending_before_delivery, queue.pending_count
        )
        for delivery in queue.deliver_ready():
            update = learner.learn(delivery.reward)
            if update.action_index != delivery.action_index:
                raise RuntimeError("learner credit and queue action diverged")
            deliveries.append(delivery)
        max_pending_after_delivery = max(max_pending_after_delivery, queue.pending_count)
        decisions_with_prior_feedback_pending += int(prior_pending)
        actions.append(action)

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
        if decision_count < len(fixtures):
            queue.advance()

    terminal_drain_count = 0
    while queue.pending_count:
        queue.advance()
        for delivery in queue.deliver_ready():
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

    action_tuple = tuple(actions)
    reward_tuple = tuple(float(value) for value in reward_schedule)
    return _TrainingRun(
        actions=action_tuple,
        rewards=reward_tuple,
        action_digest=hashlib.sha256(bytes(action_tuple)).hexdigest(),
        reward_digest=_reward_digest(reward_tuple),
        parameter_digest=learner.parameter_digest(),
        timeline=timeline,
        checkpoints=tuple(checkpoints),
        pending_feedback=bool(learner.has_pending_feedback),
    )


def _behavior_gate(
    post_training: AccuracyCount,
    per_delay: tuple[tuple[int, AccuracyCount], ...],
    state_reset: AccuracyCount,
    reset_per_delay: tuple[tuple[int, AccuracyCount], ...],
    shuffled_control: AccuracyCount,
) -> bool:
    return (
        post_training.correct >= 180
        and all(count.correct >= 34 for _, count in per_delay)
        and state_reset.correct == 100
        and all(count.correct == 20 for _, count in reset_per_delay)
        and shuffled_control.correct < 150
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


def _checkpoint_payload(
    checkpoints: tuple[DelayedCreditCheckpoint, ...],
) -> list[dict[str, object]]:
    return [
        {
            "decision_count": checkpoint.decision_count,
            "delivery_count": checkpoint.delivery_count,
            "unresolved_credit_count": checkpoint.unresolved_credit_count,
            "parameter_digest": checkpoint.parameter_digest,
        }
        for checkpoint in checkpoints
    ]
