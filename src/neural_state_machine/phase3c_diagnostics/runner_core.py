from __future__ import annotations

from neural_state_machine.action_value_benchmark import _build_fixture_bundle
from neural_state_machine.phase3c_benchmark import AnonymousCreditConfig
from neural_state_machine.phase3c_schedule import build_hidden_delay_schedule

from . import runner_core_base as _base
from .contracts import ModelId
from .evaluation import EvaluationBundle
from .provenance import (
    apply_permutation,
    diagnostic_permutation,
    identity_permutation,
    original_block10_permutation,
    provenance_association_audit,
)
from .replay import ReplayResult, capture_reward_override_execution
from .runner_core_base import *  # noqa: F403


def __getattr__(name: str) -> object:
    return getattr(_base, name)


def _d1_audit_for_replay(
    seed: int,
    replay: ReplayResult,
    permutation: tuple[int, ...],
    normal_latent_rewards: tuple[float, ...],
    config: AnonymousCreditConfig,
) -> dict[str, object]:
    """Bind observer-only D1 statistics to one actual captured learner replay."""
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not isinstance(replay, ReplayResult):
        raise ValueError("replay must be a ReplayResult")
    if not isinstance(config, AnonymousCreditConfig):
        raise ValueError("config must be an AnonymousCreditConfig")
    if replay.protocol.seed != seed:
        raise ValueError("replay seed differs from requested D1 seed")
    if len(replay.steps) != config.training_decisions:
        raise ValueError("replay decision count differs from registered configuration")

    fixtures = _build_fixture_bundle(seed, config.action_value_config)
    labels = tuple(episode.correct_action_index for episode in fixtures.training)
    schedule = build_hidden_delay_schedule(
        seed,
        config.training_decisions,
        support=config.delay_support,
    )
    return provenance_association_audit(
        permutation=permutation,
        due_steps=schedule.due_steps,
        latent_rewards=normal_latent_rewards,
        actions=tuple(step.action for step in replay.steps),
        labels=labels,
        actual_feedback_calls=replay.scalar_calls,
    )


def _anonymous_model_row(
    model: ModelId,
    *,
    normal_rewards: tuple[float, ...],
    replays: dict[tuple[int, str, str], ReplayResult],
    config: AnonymousCreditConfig,
    bundles: tuple[EvaluationBundle, ...],
    attempt_id: str,
) -> dict[str, object]:
    """Train/score one anonymous row and bind its D1 evidence to that execution."""
    if not isinstance(model, ModelId) or model.family not in ("original", "permutation"):
        raise ValueError("anonymous row requires an original or permutation ModelId")
    if not isinstance(config, AnonymousCreditConfig):
        raise ValueError("config must be an AnonymousCreditConfig")
    if len(normal_rewards) != config.training_decisions:
        raise ValueError("normal reward stream length differs from registered training length")

    arm = str(model.arm)
    if model.family == "original":
        condition = str(model.condition)
        if condition == "normal":
            permutation = identity_permutation(config.training_decisions)
            reward_override = None
        elif condition == "original_shuffle":
            permutation = original_block10_permutation(model.seed, config.training_decisions)
            reward_override = apply_permutation(normal_rewards, permutation)
        else:
            raise ValueError("original model condition is not registered")
        try:
            replay = replays[(model.seed, arm, condition)]
        except KeyError as exc:
            raise RuntimeError("required D0 replay is missing for original row") from exc
        execution = _base._execute_training(
            model.seed,
            arm,
            config,
            immediate_control=False,
            reward_override=reward_override,
        )
        if execution.protocol != replay.protocol:
            raise RuntimeError("original scoring execution differs from the D0 replay endpoint")
        learner = execution.learner
        parameter_digest = execution.protocol.parameter_digest
    else:
        permutation = diagnostic_permutation(
            model.seed,
            config.training_decisions,
            str(model.mode),
            int(model.replicate),
        )
        reward_override = apply_permutation(normal_rewards, permutation)
        captured = capture_reward_override_execution(
            model.seed,
            arm,
            config,
            reward_override=reward_override,
            attempt_id=attempt_id,
        )
        replay = captured.replay
        learner = captured.learner
        parameter_digest = replay.protocol.parameter_digest
        if learner.parameter_digest() != parameter_digest:
            raise RuntimeError("captured permutation learner differs from its replay endpoint")

    d1 = _d1_audit_for_replay(
        model.seed,
        replay,
        permutation,
        normal_rewards,
        config,
    )
    scores = _base._score_learner(model.seed, learner, config, bundles)
    return {
        "parameter_digest": parameter_digest,
        "scores": scores,
        "d1": d1,
    }
