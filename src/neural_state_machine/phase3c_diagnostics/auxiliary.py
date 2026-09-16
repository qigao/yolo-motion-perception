from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from neural_state_machine.phase3c_benchmark import AnonymousCreditConfig

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


def registered_auxiliary_keys(config: AnonymousCreditConfig) -> AuxiliaryKeys:
    del config
    return AuxiliaryKeys((), (), ())


def learner_view_digest(replay: ReplayResult) -> str:
    del replay
    return ""


def replay_with_metadata(replay: ReplayResult, metadata: tuple[dict[str, object], ...]) -> ReplayResult:
    return replace(replay, metadata=metadata)


def drain_weight_snapshots(replay: ReplayResult, arm: str) -> tuple[tuple[str, np.ndarray], ...]:
    del replay, arm
    return ()


def score_weight_snapshot(
    seed: int,
    weights: np.ndarray,
    fixtures: tuple[object, ...],
    config: AnonymousCreditConfig,
) -> SnapshotScore:
    del seed, weights, fixtures, config
    return SnapshotScore((), (), 0, 0, 0)
