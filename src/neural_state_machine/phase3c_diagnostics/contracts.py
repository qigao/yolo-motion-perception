from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelId:
    family: str
    seed: int
    arm: str | None = None
    condition: str | None = None
    mode: str | None = None
    replicate: int | None = None
    reference_kind: str | None = None


@dataclass(frozen=True)
class EvaluationId:
    seed: int
    evaluation_id: int


@dataclass(frozen=True)
class NullableMetric:
    value: float | None
    reason: str | None
    sample_count: int


def registered_model_ids() -> tuple[ModelId, ...]:
    return ()
