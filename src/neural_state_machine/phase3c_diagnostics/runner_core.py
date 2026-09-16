from __future__ import annotations

from neural_state_machine.action_value_benchmark import _build_fixture_bundle
from neural_state_machine.phase3c_benchmark import AnonymousCreditConfig
from neural_state_machine.phase3c_schedule import build_hidden_delay_schedule

from . import runner_core_base as _base
from .provenance import provenance_association_audit
from .replay import ReplayResult
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
