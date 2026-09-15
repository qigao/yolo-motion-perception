"""Phase 3B delayed action-to-reward benchmark protocol."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from .action_value_benchmark import (
    ActionValueBenchmarkConfig,
    _build_fixture_bundle,
    _evaluate,
    _new_policy,
)
from .delayed_credit import DelayedRewardQueue
from .memory_benchmark import AccuracyCount
from .memory_task import DelayedCueTask
from .phase3b_learners import DelayedTD0Adapter, EpisodeResetEligibilityTrace
from .reward_learning import _decision_hidden


_ALLOWED_ARMS = ("td0", "td_lambda")
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
    repeatable: bool


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
    if arm not in _ALLOWED_ARMS:
        raise ValueError("arm must be one of ('td0', 'td_lambda')")
    first = _run_once(seed, reward_delay, arm, resolved)
    second = _run_once(seed, reward_delay, arm, resolved)
    return DelayedCreditResult(
        seed=seed,
        arm=arm,
        reward_delay=reward_delay,
        pre_training=first[0].overall,
        post_training=first[1].overall,
        state_reset=first[2].overall,
        per_delay=first[1].per_delay,
        reset_per_delay=first[2].per_delay,
        action_digest=first[3].action_digest,
        actions=first[3].actions,
        training_reward_digest=first[4],
        training_fixture_digest=first[5],
        evaluation_fixture_digest=first[6],
        parameter_digest=first[7],
        pending_feedback=first[8],
        queue_deliveries=first[9],
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
    if arm not in _ALLOWED_ARMS:
        raise ValueError("arm must be one of ('td0', 'td_lambda')")
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
                "repeatable": result.repeatable,
            }
            for result in results
        ],
    }


def _run_once(
    seed: int,
    reward_delay: int,
    arm: str,
    config: DelayedCreditConfig,
) -> tuple[object, object, object, object, str, str, str, str, bool, int]:
    task = DelayedCueTask()
    action_config = config.action_value_config
    fixtures = _build_fixture_bundle(seed, action_config)
    policy = _new_policy(seed, action_config)
    if arm == "td0":
        learner = DelayedTD0Adapter(
            action_config.hidden_size,
            2,
            step_size=action_config.step_size,
        )
    else:
        learner = EpisodeResetEligibilityTrace(
            action_config.hidden_size,
            2,
            step_size=action_config.step_size,
            discount=config.discount,
            trace_decay=config.trace_decay,
        )
    action_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x33414354]))
    queue = DelayedRewardQueue(max_delay=max(config.reward_delays))
    pre_training = _evaluate(policy, learner, fixtures.evaluation, reset_before_decision=False)
    actions: list[int] = []
    rewards: list[float] = []
    deliveries = 0
    for episode in fixtures.training:
        hidden = _decision_hidden(policy, episode, reset_before_decision=False)
        decision = learner.select_for_training(hidden, (0, 1), action_rng)
        action = int(decision.action_index)
        reward = float(task.reward(episode, action))
        queue.enqueue(action, reward, reward_delay)
        for _ in range(reward_delay):
            queue.advance()
        ready = queue.deliver_ready()
        if len(ready) != 1:
            raise RuntimeError("delayed reward queue did not deliver exactly one reward")
        learner.learn(ready[0].reward)
        if hasattr(learner, "reset_episode"):
            learner.reset_episode()
        deliveries += len(ready)
        actions.append(action)
        rewards.append(reward)
    if queue.pending_count or learner.has_pending_feedback:
        raise RuntimeError("delayed training retained pending feedback")
    post_training = _evaluate(policy, learner, fixtures.evaluation, reset_before_decision=False)
    state_reset = _evaluate(policy, learner, fixtures.evaluation, reset_before_decision=True)
    return (
        pre_training,
        post_training,
        state_reset,
        _ActionTrace(tuple(actions)),
        _reward_digest(tuple(rewards)),
        fixtures.training_fixture_digest,
        fixtures.evaluation_fixture_digest,
        learner.parameter_digest(),
        bool(learner.has_pending_feedback),
        deliveries,
    )


@dataclass(frozen=True)
class _ActionTrace:
    actions: tuple[int, ...]

    @property
    def action_digest(self) -> str:
        return hashlib.sha256(bytes(self.actions)).hexdigest()


def _reward_digest(rewards: tuple[float, ...]) -> str:
    values = np.ascontiguousarray(rewards, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(len(rewards).to_bytes(8, "little"))
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _count_payload(count: AccuracyCount) -> dict[str, object]:
    return {"correct": count.correct, "total": count.total, "accuracy": count.accuracy}


def _per_delay_payload(rows: tuple[tuple[int, AccuracyCount], ...]) -> list[dict[str, object]]:
    return [{"delay": delay, **_count_payload(count)} for delay, count in rows]
