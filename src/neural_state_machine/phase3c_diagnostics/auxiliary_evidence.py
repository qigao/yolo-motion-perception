from __future__ import annotations

from dataclasses import dataclass

from neural_state_machine.phase3c_benchmark import AnonymousCreditConfig

from .contracts import ModelId
from .evaluation import EvaluationBundle
from .replay import ReplayResult


@dataclass(frozen=True, slots=True)
class AuxiliaryEvidence:
    reset_scores: tuple[dict[str, object], ...]
    reverse_checks: tuple[dict[str, object], ...]
    drain_scores: tuple[dict[str, object], ...]


def original_auxiliary_evidence(
    model: ModelId,
    replay: ReplayResult,
    bundles: tuple[EvaluationBundle, ...],
    config: AnonymousCreditConfig,
) -> AuxiliaryEvidence:
    del model, replay, bundles, config
    return AuxiliaryEvidence((), (), ())
