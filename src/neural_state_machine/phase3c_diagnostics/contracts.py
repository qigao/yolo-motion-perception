from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

REGISTERED_SEEDS = (7, 17, 29)
REGISTERED_ARMS = ("td0", "eligibility")
REGISTERED_CONDITIONS = ("normal", "original_shuffle")
PERMUTATION_MODES = ("block10", "global")
REFERENCE_KINDS = ("supervised_ridge", "immediate_identified", "source_visible_delayed")
REGISTERED_DELAY_SUPPORT = (1, 3, 5)

_REGISTERED_CONFIG = {
    "hidden_size": 64,
    "recurrent_radius": 0.9,
    "step_size": 0.1,
    "training_decisions": 2_000,
    "evaluation_blocks": 20,
    "checkpoint_interval": 100,
    "discount": 0.9,
    "trace_decay": 0.8,
}


def _int(value: object, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    return value


def _exact_keys(mapping: Mapping[str, object], expected: set[str], name: str) -> None:
    keys = set(mapping)
    if keys != expected:
        raise ValueError(f"{name} keys differ: expected {sorted(expected)}, got {sorted(keys)}")


@dataclass(frozen=True, slots=True)
class ModelId:
    family: str
    seed: int
    arm: str | None = None
    condition: str | None = None
    mode: str | None = None
    replicate: int | None = None
    reference_kind: str | None = None

    def __post_init__(self) -> None:
        seed = _int(self.seed, "seed")
        if seed not in REGISTERED_SEEDS:
            raise ValueError("seed must be one of the registered seeds")
        if self.family == "original":
            if self.arm not in REGISTERED_ARMS:
                raise ValueError("original model arm must be registered")
            if self.condition not in REGISTERED_CONDITIONS:
                raise ValueError("original model condition must be registered")
            if self.mode is not None or self.replicate is not None or self.reference_kind is not None:
                raise ValueError("original model has permutation/reference fields")
            return
        if self.family == "permutation":
            if self.arm not in REGISTERED_ARMS:
                raise ValueError("permutation model arm must be registered")
            if self.mode not in PERMUTATION_MODES:
                raise ValueError("permutation mode must be block10 or global")
            replicate = _int(self.replicate, "replicate")
            if not 0 <= replicate < 32:
                raise ValueError("replicate must be in [0, 31]")
            if self.condition is not None or self.reference_kind is not None:
                raise ValueError("permutation model has original/reference fields")
            return
        if self.family == "reference":
            if self.reference_kind not in REFERENCE_KINDS:
                raise ValueError("reference_kind must be registered")
            if any(value is not None for value in (self.arm, self.condition, self.mode, self.replicate)):
                raise ValueError("reference model has anonymous/permutation fields")
            return
        raise ValueError("family must be original, permutation or reference")

    def to_dict(self) -> dict[str, object]:
        return {
            "family": self.family,
            "seed": self.seed,
            "arm": self.arm,
            "condition": self.condition,
            "mode": self.mode,
            "replicate": self.replicate,
            "reference_kind": self.reference_kind,
        }

    @classmethod
    def from_dict(cls, mapping: Mapping[str, object]) -> "ModelId":
        _exact_keys(
            mapping,
            {"family", "seed", "arm", "condition", "mode", "replicate", "reference_kind"},
            "ModelId",
        )
        family = mapping["family"]
        if not isinstance(family, str):
            raise ValueError("family must be a string")
        seed = _int(mapping["seed"], "seed")
        for key in ("arm", "condition", "mode", "reference_kind"):
            value = mapping[key]
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{key} must be a string or null")
        replicate = mapping["replicate"]
        if replicate is not None:
            replicate = _int(replicate, "replicate")
        return cls(
            family=family,
            seed=seed,
            arm=mapping["arm"],
            condition=mapping["condition"],
            mode=mapping["mode"],
            replicate=replicate,
            reference_kind=mapping["reference_kind"],
        )


@dataclass(frozen=True, slots=True)
class EvaluationId:
    seed: int
    evaluation_id: int

    def __post_init__(self) -> None:
        seed = _int(self.seed, "seed")
        if seed not in REGISTERED_SEEDS:
            raise ValueError("seed must be one of the registered seeds")
        evaluation_id = _int(self.evaluation_id, "evaluation_id")
        if not -1 <= evaluation_id <= 7:
            raise ValueError("evaluation_id must be -1 (original) or in [0, 7]")

    @property
    def is_original(self) -> bool:
        return self.evaluation_id == -1


@dataclass(frozen=True, slots=True)
class NullableMetric:
    value: float | None
    reason: str | None
    sample_count: int

    def __post_init__(self) -> None:
        sample_count = _int(self.sample_count, "sample_count")
        if sample_count < 0:
            raise ValueError("sample_count must be non-negative")
        if self.value is None:
            if not isinstance(self.reason, str) or not self.reason:
                raise ValueError("undefined metric requires a reason")
            return
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise ValueError("metric value must be a finite number or null")
        if not math.isfinite(float(self.value)):
            raise ValueError("metric value must be finite")
        if self.reason is not None:
            raise ValueError("defined metric must not have a reason")

    def to_dict(self) -> dict[str, object]:
        return {"value": self.value, "reason": self.reason, "sample_count": self.sample_count}


def registered_config() -> dict[str, object]:
    return dict(_REGISTERED_CONFIG)


def validate_registered_config(mapping: Mapping[str, object]) -> None:
    expected = registered_config()
    _exact_keys(mapping, set(expected), "registered config")
    if dict(mapping) != expected:
        raise ValueError("registered config values differ")


def registered_model_ids() -> tuple[ModelId, ...]:
    rows: list[ModelId] = []
    for seed in REGISTERED_SEEDS:
        for condition in REGISTERED_CONDITIONS:
            for arm in REGISTERED_ARMS:
                rows.append(ModelId("original", seed, arm=arm, condition=condition))
        for mode in PERMUTATION_MODES:
            for replicate in range(32):
                for arm in REGISTERED_ARMS:
                    rows.append(
                        ModelId("permutation", seed, arm=arm, mode=mode, replicate=replicate)
                    )
        for reference_kind in REFERENCE_KINDS:
            rows.append(ModelId("reference", seed, reference_kind=reference_kind))
    result = tuple(rows)
    if len(result) != 405 or len(set(result)) != 405:
        raise RuntimeError("registered model grid is internally inconsistent")
    return result


def registered_evaluation_ids() -> tuple[EvaluationId, ...]:
    return tuple(
        EvaluationId(seed, evaluation_id)
        for seed in REGISTERED_SEEDS
        for evaluation_id in (-1, *range(8))
    )
