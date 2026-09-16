from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from neural_state_machine.action_value_benchmark import (
    _build_fixture_bundle,
    _collect_checkpoint,
    _new_policy,
    _reward_digest,
)
from neural_state_machine.memory_task import DelayedCueTask
from neural_state_machine.phase3c_benchmark import (
    AnonymousCreditConfig,
    _ProtocolExecution,
    _action_digest,
    _build_audit,
    _execute_training,
    _new_arm,
)
from neural_state_machine.phase3c_learners import NormalizedAnonymousEligibilityCredit
from neural_state_machine.phase3c_schedule import (
    AggregateFeedback,
    AnonymousRewardAggregator,
    build_hidden_delay_schedule,
)
from neural_state_machine.reward_learning import _decision_hidden

from .provenance import apply_permutation, original_block10_permutation


class D0Failure(RuntimeError):
    def __init__(
        self,
        *,
        path: str,
        expected: object,
        observed: object,
        step: int | None = None,
        attempt_id: str | None = None,
    ) -> None:
        self.path = path
        self.expected = expected
        self.observed = observed
        self.step = step
        self.attempt_id = attempt_id
        super().__init__(
            f"{path}: expected={expected!r}, observed={observed!r}, "
            f"step={step!r}, attempt_id={attempt_id!r}"
        )


@dataclass(frozen=True, slots=True)
class StepCapture:
    step: int
    action: int
    reward: float
    feedback: float
    prediction: float
    td_error: float
    hidden: np.ndarray
    action_values: np.ndarray
    weights_before: np.ndarray
    weights_after: np.ndarray
    eligibility: np.ndarray | None
    prediction_trace: float | None
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ReplayResult:
    protocol: object
    scalar_calls: tuple[float, ...]
    action_values: tuple[tuple[float, ...], ...]
    hidden_bytes: tuple[bytes, ...]
    final_parameter_digest: str
    queue_pending_final: int
    metadata: tuple[dict[str, Any], ...]
    steps: tuple[StepCapture, ...] = ()
    drain_steps: tuple[StepCapture, ...] = ()


def _readonly(values: np.ndarray) -> np.ndarray:
    copied = np.array(values, dtype=np.float64, copy=True)
    copied.flags.writeable = False
    return copied


def _digest_weights(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def validate_call_stream(
    expected: tuple[float, ...], observed: tuple[float, ...], *, attempt_id: str
) -> None:
    if len(expected) != len(observed):
        raise D0Failure(
            path="scalar_calls.length",
            expected=len(expected),
            observed=len(observed),
            attempt_id=attempt_id,
        )
    for step, (left, right) in enumerate(zip(expected, observed, strict=True)):
        if type(left) is not type(right) or left != right:
            raise D0Failure(
                path=f"scalar_calls[{step}]",
                expected=left,
                observed=right,
                step=step,
                attempt_id=attempt_id,
            )


def assert_array_isolation(
    learner_array: np.ndarray, observer_array: np.ndarray, *, attempt_id: str
) -> None:
    learner = np.asarray(learner_array)
    observer = np.asarray(observer_array)
    if np.shares_memory(learner, observer):
        raise D0Failure(
            path="array_alias",
            expected="independent storage",
            observed="shared storage",
            attempt_id=attempt_id,
        )
    if observer.flags.writeable:
        raise D0Failure(
            path="observer_array.writeable",
            expected=False,
            observed=True,
            attempt_id=attempt_id,
        )


def _original_shuffle_rewards(seed: int, config: AnonymousCreditConfig) -> tuple[float, ...]:
    baseline = _execute_training(seed, "td0", config, immediate_control=False)
    permutation = original_block10_permutation(seed, config.training_decisions)
    return apply_permutation(baseline.protocol.rewards, permutation)


def _capture_training(
    seed: int,
    arm: str,
    config: AnonymousCreditConfig,
    *,
    reward_override: tuple[float, ...] | None,
) -> tuple[_ProtocolExecution, tuple[StepCapture, ...], tuple[StepCapture, ...], int]:
    av_config = config.action_value_config
    task = DelayedCueTask()
    fixtures = _build_fixture_bundle(seed, av_config)
    schedule = build_hidden_delay_schedule(
        seed, config.training_decisions, support=config.delay_support
    )
    policy = _new_policy(seed, av_config)
    learner = _new_arm(arm, config, immediate_control=False)
    action_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x33414354]))
    aggregator = AnonymousRewardAggregator(schedule)

    actions: list[int] = []
    rewards: list[float] = []
    feedbacks: list[AggregateFeedback] = []
    checkpoints = []
    block_hidden: list[np.ndarray] = []
    block_correct: list[int] = []
    block_td_errors: list[float] = []
    steps: list[StepCapture] = []
    drains: list[StepCapture] = []

    for step, episode in enumerate(fixtures.training):
        hidden = _decision_hidden(policy, episode, reset_before_decision=False)
        weights_before = _readonly(learner.parameter_snapshot())
        decision = learner.select_for_training(hidden, (0, 1), action_rng)
        action = int(decision.action_index)
        reward = (
            float(task.reward(episode, action))
            if reward_override is None
            else float(reward_override[step])
        )
        aggregator.enqueue(step, action, reward)
        feedback = aggregator.feedback_at(step)
        update = learner.learn(feedback.value)
        weights_after = _readonly(learner.parameter_snapshot())
        eligibility = None
        prediction_trace = None
        if isinstance(learner, NormalizedAnonymousEligibilityCredit):
            eligibility = _readonly(learner.eligibility_snapshot())
            prediction_trace = float(learner.prediction_trace)
        hidden_copy = _readonly(hidden)
        values_copy = _readonly(decision.action_values)
        metadata = {
            "delivery_step": feedback.delivery_step,
            "multiplicity": feedback.multiplicity,
            "records": tuple(
                {
                    "source_step": record.source_step,
                    "source_action": record.source_action,
                    "reward": record.reward,
                    "delay": record.delay,
                    "due_step": record.due_step,
                    "delivery_step": record.delivery_step,
                }
                for record in feedback.records
            ),
        }
        steps.append(
            StepCapture(
                step=step,
                action=action,
                reward=reward,
                feedback=float(feedback.value),
                prediction=float(update.prediction),
                td_error=float(update.td_error),
                hidden=hidden_copy,
                action_values=values_copy,
                weights_before=weights_before,
                weights_after=weights_after,
                eligibility=eligibility,
                prediction_trace=prediction_trace,
                metadata=metadata,
            )
        )
        actions.append(action)
        rewards.append(reward)
        feedbacks.append(feedback)
        block_hidden.append(hidden_copy)
        block_correct.append(episode.correct_action_index)
        block_td_errors.append(float(update.td_error))
        decision_count = step + 1
        if decision_count % config.checkpoint_interval == 0:
            checkpoints.append(
                _collect_checkpoint(
                    learner,
                    decision_count,
                    block_hidden,
                    block_correct,
                    block_td_errors,
                )
            )
            block_hidden = []
            block_correct = []
            block_td_errors = []

    if block_hidden or block_correct or block_td_errors:
        raise RuntimeError("diagnostic capture ended with an incomplete checkpoint interval")

    last_due_step = max(schedule.due_steps)
    drain_count = 0
    for delivery_step in range(config.training_decisions, last_due_step + 1):
        weights_before = _readonly(learner.parameter_snapshot())
        eligibility = None
        prediction_trace = None
        if isinstance(learner, NormalizedAnonymousEligibilityCredit):
            eligibility = _readonly(learner.eligibility_snapshot())
            prediction_trace = float(learner.prediction_trace)
        feedback = aggregator.feedback_at(delivery_step)
        update = learner.learn_drain(feedback.value)
        weights_after = _readonly(learner.parameter_snapshot())
        drains.append(
            StepCapture(
                step=delivery_step,
                action=-1,
                reward=float(feedback.value),
                feedback=float(feedback.value),
                prediction=float(update.prediction),
                td_error=float(update.td_error),
                hidden=_readonly(np.empty((0,), dtype=np.float64)),
                action_values=_readonly(np.empty((0,), dtype=np.float64)),
                weights_before=weights_before,
                weights_after=weights_after,
                eligibility=eligibility,
                prediction_trace=prediction_trace,
                metadata={
                    "delivery_step": feedback.delivery_step,
                    "multiplicity": feedback.multiplicity,
                    "records": tuple(
                        {
                            "source_step": record.source_step,
                            "source_action": record.source_action,
                            "reward": record.reward,
                            "delay": record.delay,
                            "due_step": record.due_step,
                            "delivery_step": record.delivery_step,
                        }
                        for record in feedback.records
                    ),
                },
            )
        )
        feedbacks.append(feedback)
        drain_count += 1

    parameter_digest = learner.parameter_digest()
    trace_reset_count = 0
    if isinstance(learner, NormalizedAnonymousEligibilityCredit):
        learner.end_run()
        trace_reset_count = learner.trace_reset_count

    reward_tuple = tuple(rewards)
    action_tuple = tuple(actions)
    audit = _build_audit(
        arm=arm,
        config=config,
        schedule=schedule,
        rewards=reward_tuple,
        feedbacks=tuple(feedbacks),
        real_feedback_count=config.training_decisions,
        drain_feedback_count=drain_count,
        pending_final=aggregator.pending_count,
        trace_reset_count=trace_reset_count,
    )
    protocol = _ProtocolExecution(
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
        checkpoints=tuple(checkpoints),
    )
    return protocol, tuple(steps), tuple(drains), aggregator.pending_count


def run_diagnostic_replay(
    seed: int,
    arm: str,
    config: AnonymousCreditConfig,
    condition: str = "normal",
    *,
    attempt_id: str = "d0",
) -> ReplayResult:
    if condition not in ("normal", "original_shuffle"):
        raise ValueError("condition must be normal or original_shuffle")
    reward_override = None
    if condition == "original_shuffle":
        reward_override = _original_shuffle_rewards(seed, config)
    untouched = _execute_training(
        seed,
        arm,
        config,
        immediate_control=False,
        reward_override=reward_override,
    )
    captured_protocol, steps, drains, pending_final = _capture_training(
        seed, arm, config, reward_override=reward_override
    )
    if captured_protocol != untouched.protocol:
        raise D0Failure(
            path="protocol",
            expected=untouched.protocol,
            observed=captured_protocol,
            attempt_id=attempt_id,
        )
    scalar_calls = tuple(step.feedback for step in (*steps, *drains))
    hidden_bytes = tuple(
        np.ascontiguousarray(step.hidden, dtype=np.float64).tobytes(order="C")
        for step in steps
    )
    action_values = tuple(tuple(float(value) for value in step.action_values) for step in steps)
    metadata = tuple(step.metadata for step in (*steps, *drains))
    for step in steps:
        assert_array_isolation(step.weights_after, _readonly(step.weights_after), attempt_id=attempt_id)
    return ReplayResult(
        protocol=untouched.protocol,
        scalar_calls=scalar_calls,
        action_values=action_values,
        hidden_bytes=hidden_bytes,
        final_parameter_digest=untouched.protocol.parameter_digest,
        queue_pending_final=pending_final,
        metadata=metadata,
        steps=steps,
        drain_steps=drains,
    )
