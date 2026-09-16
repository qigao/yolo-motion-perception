"""Explicit, Gate-P-bound measurement helpers for Phase C4-A and C4-B."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .action_value import NormalizedActionValue
from .action_value_benchmark import _build_fixture_bundle, _evaluate, _new_policy
from .memory_benchmark import AccuracyCount
from .memory_task import DelayedCueTask
from .phase3c_schedule import AnonymousRewardAggregator, build_hidden_delay_schedule
from .phase_c4_batch_probe import BatchProbeFit, build_batch_design, fit_anonymous_batch_probe
from .phase_c4_delay_model import DecisionCreditRow, DelayLaw
from .phase_c4_learner import DelayMarginalizedAnonymousCredit
from .reward_learning import _build_fixtures, _decision_hidden

_SECONDARY_LINEAGE = 0x33434641
_SHUFFLE_LINEAGE = 0x33534846


@dataclass(frozen=True, slots=True)
class SecondaryEvaluationScore:
    evaluation_id: int
    fixture_digest: str
    overall: AccuracyCount
    per_delay: tuple[tuple[int, AccuracyCount], ...]


@dataclass(frozen=True, slots=True)
class C4ASeedResult:
    seed: int
    design_row_count: int
    normal_fit: BatchProbeFit
    shuffled_fit: BatchProbeFit
    post_training: AccuracyCount
    per_delay: tuple[tuple[int, AccuracyCount], ...]
    state_reset: AccuracyCount
    reset_per_delay: tuple[tuple[int, AccuracyCount], ...]
    shuffled_control: AccuracyCount
    shuffled_per_delay: tuple[tuple[int, AccuracyCount], ...]
    secondary_scores: tuple[SecondaryEvaluationScore, ...]
    operator_passed: bool


@dataclass(frozen=True, slots=True)
class C4BSeedResult:
    seed: int
    post_training: AccuracyCount
    per_delay: tuple[tuple[int, AccuracyCount], ...]
    state_reset: AccuracyCount
    reset_per_delay: tuple[tuple[int, AccuracyCount], ...]
    shuffled_control: AccuracyCount
    shuffled_per_delay: tuple[tuple[int, AccuracyCount], ...]
    secondary_scores: tuple[SecondaryEvaluationScore, ...]
    normal_drain_feedback_count: int
    shuffled_drain_feedback_count: int
    action_lineage_equal: bool
    schedule_lineage_equal: bool
    behavior_passed: bool


@dataclass(slots=True)
class _OnlineReplay:
    learner: DelayMarginalizedAnonymousCredit
    actions: tuple[int, ...]
    schedule_digest: str
    drain_feedback_count: int


def _benchmark_module():
    from . import phase_c4_benchmark as benchmark

    return benchmark


def _assert_protocol_binding(protocol: object, config: object) -> None:
    benchmark = _benchmark_module()
    if not isinstance(protocol, benchmark.PhaseC4ProtocolResult):
        raise ValueError("protocol must be a PhaseC4ProtocolResult")
    if not isinstance(config, benchmark.PhaseC4Config):
        raise ValueError("config must be a PhaseC4Config")
    replay = benchmark._execute_registered(protocol.seed, config)
    expected = {
        "training_fixture_digest": replay.training_fixture_digest,
        "evaluation_fixture_digest": replay.evaluation_fixture_digest,
        "action_digest": replay.action_digest,
        "latent_reward_digest": replay.latent_reward_digest,
        "schedule_digest": replay.schedule_digest,
        "call_digest": replay.call_digest,
        "candidate_digest": replay.candidate_digest,
        "parameter_digest": replay.parameter_digest,
    }
    for field, value in expected.items():
        if getattr(protocol, field) != value:
            raise RuntimeError(f"protocol binding mismatch: {field}")
    benchmark.validate_phase_c4_protocol(
        protocol.audit,
        benchmark._expected_actions(protocol.seed, config.training_decisions),
    )


def _decision_rows_and_rewards(seed: int, config: object):
    benchmark = _benchmark_module()
    fixtures = _build_fixture_bundle(seed, config.action_value_config)
    policy = _new_policy(seed, config.action_value_config)
    task = DelayedCueTask()
    actions = benchmark._expected_actions(seed, config.training_decisions)
    rows: list[DecisionCreditRow] = []
    rewards: list[float] = []
    for step, (episode, action) in enumerate(zip(fixtures.training, actions, strict=True)):
        hidden = _decision_hidden(policy, episode, reset_before_decision=False)
        feature = np.concatenate((hidden, np.array([1.0], dtype=np.float64)))
        denominator = float(np.dot(feature, feature))
        rows.append(DecisionCreditRow(step, action, feature, denominator))
        rewards.append(float(task.reward(episode, action)))
    return fixtures, tuple(rows), actions, tuple(rewards)


def _aggregate_stream(
    seed: int,
    config: object,
    actions: tuple[int, ...],
    rewards: tuple[float, ...],
) -> tuple[tuple[float, ...], str, int]:
    benchmark = _benchmark_module()
    schedule = build_hidden_delay_schedule(
        seed,
        config.training_decisions,
        support=config.delay_support,
    )
    aggregator = AnonymousRewardAggregator(schedule)
    scalars: list[float] = []
    for step, (action, reward) in enumerate(zip(actions, rewards, strict=True)):
        aggregator.enqueue(step, action, reward)
        scalars.append(float(aggregator.feedback_at(step).value))
    last_clock = config.training_decisions - 1 + max(config.delay_support)
    drain_count = 0
    for delivery_step in range(config.training_decisions, last_clock + 1):
        scalars.append(float(aggregator.feedback_at(delivery_step).value))
        drain_count += 1
    if aggregator.pending_count != 0:
        raise RuntimeError("aggregate replay ended with pending rewards")
    return (
        tuple(scalars),
        benchmark._schedule_digest(schedule.delay_digest, schedule.due_step_digest),
        drain_count,
    )


def _permute_reward_blocks(
    rewards: tuple[float, ...],
    rng: np.random.Generator,
    *,
    block_size: int = 10,
) -> tuple[float, ...]:
    if not isinstance(rewards, tuple) or not rewards or len(rewards) % block_size:
        raise ValueError("rewards must be a non-empty tuple of complete blocks")
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a numpy.random.Generator")
    shuffled: list[float] = []
    for start in range(0, len(rewards), block_size):
        block = np.asarray(rewards[start : start + block_size], dtype=np.float64).copy()
        rng.shuffle(block)
        shuffled.extend(float(value) for value in block)
    return tuple(shuffled)


def _readout_from_fit(fit: BatchProbeFit, config: object) -> NormalizedActionValue:
    readout = NormalizedActionValue(config.hidden_size, 2, step_size=config.step_size)
    if fit.weights.shape != readout.parameter_snapshot().shape:
        raise RuntimeError("batch-fit shape does not match the evaluation readout")
    readout._weights[:] = fit.weights
    return readout


def _evaluate_readout(seed: int, config: object, readout: object):
    fixtures = _build_fixture_bundle(seed, config.action_value_config)
    policy = _new_policy(seed, config.action_value_config)
    post = _evaluate(policy, readout, fixtures.evaluation, reset_before_decision=False)
    reset = _evaluate(policy, readout, fixtures.evaluation, reset_before_decision=True)
    return post, reset


def _secondary_scores(seed: int, config: object, readout: object) -> tuple[SecondaryEvaluationScore, ...]:
    from .action_value_benchmark import _episode_digest

    task = DelayedCueTask()
    scores: list[SecondaryEvaluationScore] = []
    for evaluation_id in range(8):
        rng = np.random.Generator(
            np.random.PCG64(
                np.random.SeedSequence([seed, _SECONDARY_LINEAGE, 2, evaluation_id])
            )
        )
        fixtures = _build_fixtures(task, rng, config.evaluation_blocks)
        evaluation = _evaluate(
            _new_policy(seed, config.action_value_config),
            readout,
            fixtures,
            reset_before_decision=False,
        )
        scores.append(
            SecondaryEvaluationScore(
                evaluation_id=evaluation_id,
                fixture_digest=_episode_digest(fixtures),
                overall=evaluation.overall,
                per_delay=evaluation.per_delay,
            )
        )
    return tuple(scores)


def measure_c4a_from_protocol(protocol: object, config: object) -> C4ASeedResult:
    """Fit C4-A only after the supplied protocol result is independently reproduced."""
    benchmark = _benchmark_module()
    _assert_protocol_binding(protocol, config)
    fixtures, rows, actions, rewards = _decision_rows_and_rewards(protocol.seed, config)
    normal_scalars, schedule_digest, normal_drain = _aggregate_stream(
        protocol.seed, config, actions, rewards
    )
    if schedule_digest != protocol.schedule_digest or normal_drain != max(config.delay_support):
        raise RuntimeError("normal aggregate replay does not match the sealed protocol")
    design, normal_target = build_batch_design(rows, normal_scalars, 2, DelayLaw.registered())

    shuffle_rng = np.random.default_rng(
        np.random.SeedSequence([protocol.seed, _SHUFFLE_LINEAGE])
    )
    shuffled_rewards = _permute_reward_blocks(rewards, shuffle_rng, block_size=10)
    shuffled_scalars, shuffled_schedule_digest, shuffled_drain = _aggregate_stream(
        protocol.seed, config, actions, shuffled_rewards
    )
    if shuffled_schedule_digest != schedule_digest or shuffled_drain != normal_drain:
        raise RuntimeError("shuffled aggregate replay changed schedule lineage")
    shuffled_design, shuffled_target = build_batch_design(
        rows,
        shuffled_scalars,
        2,
        DelayLaw.registered(),
    )
    if design.tobytes() != shuffled_design.tobytes():
        raise RuntimeError("C4-A shuffled control changed the anonymous design matrix")

    normal_fit = fit_anonymous_batch_probe(
        design,
        normal_target,
        2,
        config.hidden_size + 1,
        penalty=config.ridge_penalty,
    )
    shuffled_fit = fit_anonymous_batch_probe(
        shuffled_design,
        shuffled_target,
        2,
        config.hidden_size + 1,
        penalty=config.ridge_penalty,
    )
    normal_readout = _readout_from_fit(normal_fit, config)
    shuffled_readout = _readout_from_fit(shuffled_fit, config)
    post, reset = _evaluate_readout(protocol.seed, config, normal_readout)
    shuffled_eval = _evaluate(
        _new_policy(protocol.seed, config.action_value_config),
        shuffled_readout,
        fixtures.evaluation,
        reset_before_decision=False,
    )
    secondary = _secondary_scores(protocol.seed, config, normal_readout)
    passed = benchmark.registered_c4_gate(
        post.overall,
        post.per_delay,
        reset.overall,
        reset.per_delay,
        shuffled_eval.overall,
    )
    return C4ASeedResult(
        seed=protocol.seed,
        design_row_count=int(design.shape[0]),
        normal_fit=normal_fit,
        shuffled_fit=shuffled_fit,
        post_training=post.overall,
        per_delay=post.per_delay,
        state_reset=reset.overall,
        reset_per_delay=reset.per_delay,
        shuffled_control=shuffled_eval.overall,
        shuffled_per_delay=shuffled_eval.per_delay,
        secondary_scores=secondary,
        operator_passed=passed,
    )


def _run_online(
    seed: int,
    config: object,
    rewards_override: tuple[float, ...] | None,
) -> _OnlineReplay:
    benchmark = _benchmark_module()
    fixtures = _build_fixture_bundle(seed, config.action_value_config)
    task = DelayedCueTask()
    policy = _new_policy(seed, config.action_value_config)
    learner = DelayMarginalizedAnonymousCredit(
        config.hidden_size,
        2,
        step_size=config.step_size,
        law=DelayLaw.registered(),
    )
    schedule = build_hidden_delay_schedule(
        seed,
        config.training_decisions,
        support=config.delay_support,
    )
    aggregator = AnonymousRewardAggregator(schedule)
    action_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x33414354]))
    actions: list[int] = []
    for step, episode in enumerate(fixtures.training):
        hidden = _decision_hidden(policy, episode, reset_before_decision=False)
        action = int(learner.select_for_training(hidden, (0, 1), action_rng).action_index)
        reward = (
            float(task.reward(episode, action))
            if rewards_override is None
            else float(rewards_override[step])
        )
        aggregator.enqueue(step, action, reward)
        learner.learn(aggregator.feedback_at(step).value)
        actions.append(action)
    drain_count = 0
    last_clock = config.training_decisions - 1 + max(config.delay_support)
    for delivery_step in range(config.training_decisions, last_clock + 1):
        learner.learn_drain(aggregator.feedback_at(delivery_step).value)
        drain_count += 1
    if aggregator.pending_count != 0 or learner.history_snapshot() != ():
        raise RuntimeError("online measurement replay did not fully drain")
    return _OnlineReplay(
        learner=learner,
        actions=tuple(actions),
        schedule_digest=benchmark._schedule_digest(schedule.delay_digest, schedule.due_step_digest),
        drain_feedback_count=drain_count,
    )


def measure_c4b_from_protocol(protocol: object, config: object) -> C4BSeedResult:
    """Replay the already-frozen online learner only after protocol reproduction."""
    benchmark = _benchmark_module()
    _assert_protocol_binding(protocol, config)
    fixtures, _, expected_actions, rewards = _decision_rows_and_rewards(protocol.seed, config)
    normal = _run_online(protocol.seed, config, None)
    shuffle_rng = np.random.default_rng(
        np.random.SeedSequence([protocol.seed, _SHUFFLE_LINEAGE])
    )
    shuffled_rewards = _permute_reward_blocks(rewards, shuffle_rng, block_size=10)
    shuffled = _run_online(protocol.seed, config, shuffled_rewards)
    action_lineage_equal = normal.actions == shuffled.actions == expected_actions
    schedule_lineage_equal = (
        normal.schedule_digest == shuffled.schedule_digest == protocol.schedule_digest
    )
    if not action_lineage_equal or not schedule_lineage_equal:
        raise RuntimeError("C4-B measurement replay changed a frozen lineage")
    if normal.drain_feedback_count != 5 or shuffled.drain_feedback_count != 5:
        raise RuntimeError("C4-B registered replay did not use exactly five drain clocks")

    post = _evaluate(
        _new_policy(protocol.seed, config.action_value_config),
        normal.learner,
        fixtures.evaluation,
        reset_before_decision=False,
    )
    reset = _evaluate(
        _new_policy(protocol.seed, config.action_value_config),
        normal.learner,
        fixtures.evaluation,
        reset_before_decision=True,
    )
    shuffled_eval = _evaluate(
        _new_policy(protocol.seed, config.action_value_config),
        shuffled.learner,
        fixtures.evaluation,
        reset_before_decision=False,
    )
    secondary = _secondary_scores(protocol.seed, config, normal.learner)
    passed = benchmark.registered_c4_gate(
        post.overall,
        post.per_delay,
        reset.overall,
        reset.per_delay,
        shuffled_eval.overall,
    )
    return C4BSeedResult(
        seed=protocol.seed,
        post_training=post.overall,
        per_delay=post.per_delay,
        state_reset=reset.overall,
        reset_per_delay=reset.per_delay,
        shuffled_control=shuffled_eval.overall,
        shuffled_per_delay=shuffled_eval.per_delay,
        secondary_scores=secondary,
        normal_drain_feedback_count=normal.drain_feedback_count,
        shuffled_drain_feedback_count=shuffled.drain_feedback_count,
        action_lineage_equal=action_lineage_equal,
        schedule_lineage_equal=schedule_lineage_equal,
        behavior_passed=passed,
    )
