from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median

from .r1_e2_reservoir import (
    E2Architecture,
    E2ReservoirSpec,
    build_e2_reservoir,
)
from .r1_e3_benchmark import evaluate_real_track_arm
from .r1_e3_dataset import RegisteredDataset


REGISTERED_ARCHITECTURES = (
    E2Architecture.FLAT,
    E2Architecture.GROUPED4,
    E2Architecture.HIERARCHICAL2,
    E2Architecture.HIERARCHICAL4,
)
REGISTERED_SEEDS = (7, 17, 29, 43, 61)
REGISTERED_LABELS = ("approach", "touch", "pick_up", "pass_by")
INPUT_SIZE = 14
ARM_COUNT = len(REGISTERED_ARCHITECTURES) * len(REGISTERED_SEEDS)
TEMPORAL_WINDOW_BINS = (16, 17, 18, 19)
RIDGE_REGULARIZATION = 1e-6
_OUTCOME_A_MEDIAN_MIN = 0.05
_OUTCOME_A_POSITIVE_ARM_MIN = 16


class ProtocolInvalid(RuntimeError):
    pass


@dataclass(frozen=True)
class RegisteredArmResult:
    seed: int
    architecture: int
    result: object
    delta_macro_f1: float


@dataclass(frozen=True)
class RegisteredMeasurementResult:
    registered_measurement: bool
    manifest: dict[str, object]
    arms: tuple[RegisteredArmResult, ...]
    median_delta_macro_f1: float
    positive_arm_count: int
    outcome: str


def registered_manifest_payload(artifact_root_digest: object) -> dict[str, object]:
    digest = _validated_digest(artifact_root_digest)
    return {
        "phase": "R1-E3",
        "artifact_root_digest": digest,
        "architectures": {
            "flat": int(E2Architecture.FLAT),
            "grouped4": int(E2Architecture.GROUPED4),
            "hierarchical2": int(E2Architecture.HIERARCHICAL2),
            "hierarchical4": int(E2Architecture.HIERARCHICAL4),
        },
        "seeds": list(REGISTERED_SEEDS),
        "labels": list(REGISTERED_LABELS),
        "input_size": INPUT_SIZE,
        "neuron_budget": 256,
        "arm_count": ARM_COUNT,
        "primary_metric": "macro_f1",
        "temporal_window_bins": list(TEMPORAL_WINDOW_BINS),
        "ridge_regularization": RIDGE_REGULARIZATION,
        "decoder_arms": [
            "frame_only",
            "reservoir_instantaneous",
            "reservoir_temporal_mean",
        ],
        "outcome_a": {
            "median_delta_macro_f1_min": _OUTCOME_A_MEDIAN_MIN,
            "positive_arm_count_min": _OUTCOME_A_POSITIVE_ARM_MIN,
        },
        "outcome_b": {
            "median_delta_macro_f1_gt": 0.0,
            "requires_outcome_a_false": True,
        },
        "outcome_c": {
            "median_delta_macro_f1_lte": 0.0,
        },
    }


def classify_registered_outcome(deltas: object) -> str:
    values = _validated_deltas(deltas)
    median_delta = float(median(values))
    positive_count = sum(value > 0.0 for value in values)
    if (
        median_delta >= _OUTCOME_A_MEDIAN_MIN
        and positive_count >= _OUTCOME_A_POSITIVE_ARM_MIN
    ):
        return "A"
    if median_delta > 0.0:
        return "B"
    return "C"


def protocol_smoke(
    artifact_root_digest: object,
    *,
    seed: object = 7,
) -> dict[str, object]:
    digest = _validated_digest(artifact_root_digest)
    registered_seed = _validated_registered_seed(seed)

    parameter_digests: dict[str, str] = {}
    state_dims: dict[str, int] = {}
    for architecture in REGISTERED_ARCHITECTURES:
        reservoir = build_e2_reservoir(
            E2ReservoirSpec(
                architecture=architecture,
                input_size=INPUT_SIZE,
                seed=registered_seed,
            )
        )
        name = _architecture_name(architecture)
        parameter_digests[name] = reservoir.parameter_digest()
        state_dims[name] = reservoir.state_dim

    valid = (
        len(parameter_digests) == len(REGISTERED_ARCHITECTURES)
        and all(len(value) == 64 for value in parameter_digests.values())
        and all(value == 256 for value in state_dims.values())
    )
    issues = (
        []
        if valid
        else [
            {
                "code": "smoke-invalid",
                "detail": "R1-E3 reservoir smoke invariant failed",
            }
        ]
    )
    return {
        "registered_measurement": False,
        "seed": registered_seed,
        "artifact_root_digest": digest,
        "arm_count": ARM_COUNT,
        "valid": valid,
        "issues": issues,
        "reservoir_parameter_digests": parameter_digests,
        "state_dims": state_dims,
    }


def run_registered_measurement(
    dataset: RegisteredDataset,
) -> RegisteredMeasurementResult:
    if not isinstance(dataset, RegisteredDataset):
        raise ValueError("dataset must be RegisteredDataset")

    manifest = registered_manifest_payload(dataset.artifact_root_digest)
    arms: list[RegisteredArmResult] = []
    deltas: list[float] = []

    for seed in REGISTERED_SEEDS:
        for architecture in REGISTERED_ARCHITECTURES:
            result = evaluate_real_track_arm(
                E2ReservoirSpec(
                    architecture=architecture,
                    input_size=INPUT_SIZE,
                    seed=seed,
                ),
                dataset,
            )
            delta = _validated_delta(
                getattr(result, "delta_macro_f1", None),
                "delta_macro_f1",
            )
            result_artifact_digest = getattr(result, "artifact_root_digest", None)
            if (
                result_artifact_digest is not None
                and result_artifact_digest != dataset.artifact_root_digest
            ):
                raise ProtocolInvalid(
                    "artifact-root: benchmark result does not match registered dataset"
                )
            arms.append(
                RegisteredArmResult(
                    seed=seed,
                    architecture=int(architecture),
                    result=result,
                    delta_macro_f1=delta,
                )
            )
            deltas.append(delta)

    if len(arms) != ARM_COUNT:
        raise ProtocolInvalid("registered arm count mismatch")

    median_delta = float(median(deltas))
    positive_count = sum(value > 0.0 for value in deltas)
    return RegisteredMeasurementResult(
        registered_measurement=True,
        manifest=manifest,
        arms=tuple(arms),
        median_delta_macro_f1=median_delta,
        positive_arm_count=positive_count,
        outcome=classify_registered_outcome(deltas),
    )


def _validated_deltas(deltas: object) -> tuple[float, ...]:
    if not isinstance(deltas, (tuple, list)) or len(deltas) != ARM_COUNT:
        raise ValueError("registered outcome requires exactly 20 arm deltas")
    return tuple(
        _validated_delta(value, f"delta[{index}]")
        for index, value in enumerate(deltas)
    )


def _validated_delta(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validated_digest(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(
            "artifact_root_digest must be a lowercase SHA-256 digest"
        )
    return value


def _validated_registered_seed(seed: object) -> int:
    if type(seed) is not int or seed not in REGISTERED_SEEDS:
        raise ValueError("seed must be one of the registered R1-E3 seeds")
    return seed


def _architecture_name(architecture: E2Architecture) -> str:
    if architecture is E2Architecture.FLAT:
        return "flat"
    if architecture is E2Architecture.GROUPED4:
        return "grouped4"
    if architecture is E2Architecture.HIERARCHICAL2:
        return "hierarchical2"
    if architecture is E2Architecture.HIERARCHICAL4:
        return "hierarchical4"
    raise ValueError("architecture must be registered")
