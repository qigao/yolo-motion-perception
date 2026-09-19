from __future__ import annotations

import math
import statistics
from dataclasses import dataclass


OUTCOME_A = "A"
OUTCOME_B = "B"
OUTCOME_C = "C"

REGISTERED_ARCHITECTURES = (0, 1, 2, 3)
REGISTERED_SEEDS = (7, 17, 29, 43, 61)
REGISTERED_DELAYS = (1, 2, 5, 10, 15)
PRIMARY_DELAY = 10
ARM_COUNT = 20


@dataclass(frozen=True)
class ArmPrimaryResult:
    architecture: int
    seed: int
    delta10: float
    reset10: float

    def __post_init__(self) -> None:
        if type(self.architecture) is not int or self.architecture not in (
            REGISTERED_ARCHITECTURES
        ):
            raise ValueError("architecture must be registered")
        if type(self.seed) is not int or self.seed not in REGISTERED_SEEDS:
            raise ValueError("seed must be registered")
        for name in ("delta10", "reset10"):
            value = getattr(self, name)
            if isinstance(value, bool):
                raise ValueError(f"{name} must be finite")
            try:
                converted = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be finite") from exc
            if not math.isfinite(converted):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, converted)


@dataclass(frozen=True)
class OutcomeClassification:
    outcome: str
    median_delta10: float
    median_reset10: float
    positive_delta_arms: int
    positive_reset_arms: int


def registered_manifest_payload() -> dict[str, object]:
    return {
        "phase": "R1-E3M",
        "source_manifest_sha256": (
            "1db8cc200ba9eaab2c14ed260b06ed582ed75d54f57c3036e1312cd2f184f4cf"
        ),
        "window_seconds": 4.0,
        "bin_count": 20,
        "bin_hz": 5.0,
        "presence_gate": 16,
        "delays": list(REGISTERED_DELAYS),
        "primary_delay": PRIMARY_DELAY,
        "architectures": list(REGISTERED_ARCHITECTURES),
        "seeds": list(REGISTERED_SEEDS),
        "arm_count": ARM_COUNT,
        "reservoir": {
            "neurons": 256,
            "activation": "tanh",
            "spectral_radius": 0.9,
            "leak": 1.0,
            "dtype": "float64",
            "recurrent_bias": False,
        },
        "ridge": {
            "regularization": 1e-6,
            "bias": "enabled-unpenalized",
        },
        "primary_thresholds": {
            "median_delta10": 0.05,
            "positive_delta_arms": 16,
            "median_reset10": 0.05,
            "positive_reset_arms": 16,
        },
        "semantic_labels_used": False,
        "human_event_boundaries_used": False,
    }


def classify_outcome(
    arms: tuple[ArmPrimaryResult, ...],
) -> OutcomeClassification:
    if len(arms) != ARM_COUNT:
        raise ValueError("R1-E3M outcome requires exactly 20 arms")
    if any(not isinstance(arm, ArmPrimaryResult) for arm in arms):
        raise ValueError("arms must contain ArmPrimaryResult")

    identities = {(arm.architecture, arm.seed) for arm in arms}
    if len(identities) != ARM_COUNT:
        raise ValueError("duplicate R1-E3M arm identity")
    expected = {
        (architecture, seed)
        for architecture in REGISTERED_ARCHITECTURES
        for seed in REGISTERED_SEEDS
    }
    if identities != expected:
        raise ValueError("R1-E3M arm identities do not match registration")

    delta_values = [arm.delta10 for arm in arms]
    reset_values = [arm.reset10 for arm in arms]
    median_delta = float(statistics.median(delta_values))
    median_reset = float(statistics.median(reset_values))
    positive_delta = sum(value > 0.0 for value in delta_values)
    positive_reset = sum(value > 0.0 for value in reset_values)

    robust = (
        median_delta >= 0.05
        and positive_delta >= 16
        and median_reset >= 0.05
        and positive_reset >= 16
    )
    if robust:
        outcome = OUTCOME_A
    elif median_delta > 0.0 and median_reset > 0.0:
        outcome = OUTCOME_B
    else:
        outcome = OUTCOME_C

    return OutcomeClassification(
        outcome=outcome,
        median_delta10=median_delta,
        median_reset10=median_reset,
        positive_delta_arms=positive_delta,
        positive_reset_arms=positive_reset,
    )
