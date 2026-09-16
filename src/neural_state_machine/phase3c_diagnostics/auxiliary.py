from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, replace

import numpy as np

from neural_state_machine.action_value import NormalizedActionValue
from neural_state_machine.action_value_benchmark import _evaluate, _new_policy
from neural_state_machine.memory_task import DelayedCueEpisode
from neural_state_machine.phase3c_benchmark import AnonymousCreditConfig
from neural_state_machine.phase3c_schedule import build_hidden_delay_schedule

from .contracts import REGISTERED_ARMS, registered_model_ids
from .replay import ReplayResult


@dataclass(frozen=True, slots=True)
class AuxiliaryKeys:
    reset_score_keys: tuple[str, ...]
    reverse_check_keys: tuple[str, ...]
    drain_score_keys: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SnapshotScore:
    canonical_actions: tuple[int, ...]
    reversed_actions_mapped: tuple[int, ...]
    canonical_correct: int
    reset_correct: int
    total: int


def _evaluation_suffixes() -> tuple[str, ...]:
    return tuple(f"evaluation={evaluation_id}" for evaluation_id in range(-1, 8))


def registered_auxiliary_keys(config: AnonymousCreditConfig) -> AuxiliaryKeys:
    if not isinstance(config, AnonymousCreditConfig):
        raise ValueError("config must be an AnonymousCreditConfig")
    models = registered_model_ids()
    original = tuple(model for model in models if model.family == "original")
    reset_models = tuple(
        model for model in models if model.family in ("original", "reference")
    )
    suffixes = _evaluation_suffixes()
    reset = tuple(
        f"reset/{model.stable_key()}/{suffix}"
        for model in reset_models
        for suffix in suffixes
    )
    reverse = tuple(
        f"reverse/{model.stable_key()}/{suffix}"
        for model in original
        for suffix in suffixes
    )
    drain: list[str] = []
    schedule_by_seed = {
        seed: build_hidden_delay_schedule(
            seed,
            config.training_decisions,
            support=config.delay_support,
        )
        for seed in {model.seed for model in original}
    }
    for model in original:
        schedule = schedule_by_seed[model.seed]
        labels = ["pre_drain"]
        labels.extend(
            f"after_drain_{delivery_step}"
            for delivery_step in range(
                config.training_decisions,
                max(schedule.due_steps) + 1,
            )
        )
        if model.arm == "eligibility":
            labels.append("post_end_run")
        for label in labels:
            for suffix in suffixes:
                drain.append(f"drain/{model.stable_key()}/{label}/{suffix}")
    result = AuxiliaryKeys(reset, reverse, tuple(drain))
    for name, values in (
        ("reset", result.reset_score_keys),
        ("reverse", result.reverse_check_keys),
        ("drain", result.drain_score_keys),
    ):
        if len(values) != len(set(values)):
            raise RuntimeError(f"{name} auxiliary keys contain duplicates")
    return result


def learner_view_digest(replay: ReplayResult) -> str:
    if not isinstance(replay, ReplayResult):
        raise ValueError("replay must be a ReplayResult")
    digest = hashlib.sha256()
    digest.update(b"phase3c-diagnostic-learner-view-v1\0")
    digest.update(str(replay.protocol.action_digest).encode("ascii"))
    digest.update(str(replay.protocol.latent_reward_digest).encode("ascii"))
    digest.update(str(replay.final_parameter_digest).encode("ascii"))
    digest.update(len(replay.scalar_calls).to_bytes(8, "big", signed=False))
    for value in replay.scalar_calls:
        digest.update(struct.pack(">d", float(value)))
    for row in replay.action_values:
        digest.update(len(row).to_bytes(8, "big", signed=False))
        for value in row:
            digest.update(struct.pack(">d", float(value)))
    for hidden in replay.hidden_bytes:
        digest.update(len(hidden).to_bytes(8, "big", signed=False))
        digest.update(hidden)
    return digest.hexdigest()


def replay_with_metadata(
    replay: ReplayResult, metadata: tuple[dict[str, object], ...]
) -> ReplayResult:
    if not isinstance(replay, ReplayResult):
        raise ValueError("replay must be a ReplayResult")
    if not isinstance(metadata, tuple) or len(metadata) != len(replay.metadata):
        raise ValueError("metadata must preserve the observer row count")
    if any(not isinstance(row, dict) for row in metadata):
        raise ValueError("metadata rows must be dictionaries")
    return replace(replay, metadata=metadata)


def _readonly(values: np.ndarray) -> np.ndarray:
    copied = np.array(values, dtype=np.float64, copy=True)
    copied.flags.writeable = False
    return copied


def drain_weight_snapshots(
    replay: ReplayResult, arm: str
) -> tuple[tuple[str, np.ndarray], ...]:
    if not isinstance(replay, ReplayResult):
        raise ValueError("replay must be a ReplayResult")
    if arm not in REGISTERED_ARMS:
        raise ValueError("arm must be td0 or eligibility")
    if replay.protocol.arm != arm:
        raise ValueError("arm differs from replay protocol")
    if not replay.drain_steps:
        raise ValueError("registered delayed replay must contain drain calls")
    snapshots: list[tuple[str, np.ndarray]] = [
        ("pre_drain", _readonly(replay.drain_steps[0].weights_before))
    ]
    snapshots.extend(
        (f"after_drain_{step.step}", _readonly(step.weights_after))
        for step in replay.drain_steps
    )
    if arm == "eligibility":
        snapshots.append(
            ("post_end_run", _readonly(replay.drain_steps[-1].weights_after))
        )
    return tuple(snapshots)


def _readout_from_weights(
    weights: np.ndarray, config: AnonymousCreditConfig
) -> NormalizedActionValue:
    values = np.asarray(weights, dtype=np.float64)
    expected_shape = (2, config.hidden_size + 1)
    if values.shape != expected_shape or not np.all(np.isfinite(values)):
        raise ValueError(f"weights must be finite with shape {expected_shape}")
    learner = NormalizedActionValue(config.hidden_size, 2, config.step_size)
    learner._weights = np.array(values, dtype=np.float64, copy=True)
    return learner


def score_weight_snapshot(
    seed: int,
    weights: np.ndarray,
    fixtures: tuple[DelayedCueEpisode, ...],
    config: AnonymousCreditConfig,
) -> SnapshotScore:
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not isinstance(config, AnonymousCreditConfig):
        raise ValueError("config must be an AnonymousCreditConfig")
    if not isinstance(fixtures, tuple) or not fixtures:
        raise ValueError("fixtures must be a non-empty tuple")
    frozen_before = np.array(weights, dtype=np.float64, copy=True)

    canonical = _evaluate(
        _new_policy(seed, config.action_value_config),
        _readout_from_weights(weights, config),
        fixtures,
        reset_before_decision=False,
    )
    reversed_evaluation = _evaluate(
        _new_policy(seed, config.action_value_config),
        _readout_from_weights(weights, config),
        tuple(reversed(fixtures)),
        reset_before_decision=False,
    )
    reset = _evaluate(
        _new_policy(seed, config.action_value_config),
        _readout_from_weights(weights, config),
        fixtures,
        reset_before_decision=True,
    )
    if not np.array_equal(np.asarray(weights), frozen_before):
        raise RuntimeError("snapshot scoring mutated input weights")
    return SnapshotScore(
        canonical_actions=canonical.actions,
        reversed_actions_mapped=tuple(reversed(reversed_evaluation.actions)),
        canonical_correct=canonical.overall.correct,
        reset_correct=reset.overall.correct,
        total=canonical.overall.total,
    )
