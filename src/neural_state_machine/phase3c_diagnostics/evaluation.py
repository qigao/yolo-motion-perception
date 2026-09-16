from __future__ import annotations

from dataclasses import dataclass

from .contracts import EvaluationId


@dataclass(frozen=True, slots=True)
class EvaluationBundle:
    evaluation_id: EvaluationId
    fixtures: tuple[object, ...]
    digest: str


def build_evaluation_bundles(*args: object, **kwargs: object) -> tuple[EvaluationBundle, ...]:
    return ()
