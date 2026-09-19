from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .r1_e2_reservoir import (
    E2Architecture,
    E2ReservoirSpec,
    build_e2_reservoir,
)


REGISTERED_ARCHITECTURES = (
    E2Architecture.FLAT,
    E2Architecture.GROUPED4,
    E2Architecture.HIERARCHICAL2,
    E2Architecture.HIERARCHICAL4,
)
REGISTERED_SEEDS = (7, 17, 29, 43, 61)
REGISTERED_BIN_COUNT = 20
INPUT_CHANNELS = 14
RESERVOIR_DIM = 256


@dataclass(frozen=True)
class ArmSpec:
    architecture: E2Architecture
    seed: int


@dataclass(frozen=True)
class RepresentationSet:
    c0: np.ndarray
    s4: np.ndarray
    trajectories: np.ndarray
    r1: np.ndarray
    r4: np.ndarray
    parameter_digest: str

    def __post_init__(self) -> None:
        arrays = {
            "c0": (self.c0, 14),
            "s4": (self.s4, 56),
            "r1": (self.r1, RESERVOIR_DIM),
            "r4": (self.r4, RESERVOIR_DIM),
        }
        sample_count: int | None = None
        for name, (raw, width) in arrays.items():
            values = _matrix(raw, name=name)
            if values.shape[1] != width:
                raise ValueError(f"{name} must have {width} features")
            if sample_count is None:
                sample_count = values.shape[0]
            elif values.shape[0] != sample_count:
                raise ValueError("representation sample counts must match")
            object.__setattr__(self, name, _readonly(values))

        trajectories = np.asarray(self.trajectories, dtype=np.float64)
        if (
            trajectories.ndim != 3
            or trajectories.shape[1:] != (REGISTERED_BIN_COUNT, RESERVOIR_DIM)
        ):
            raise ValueError(
                "trajectories must have shape (samples, 20, 256)"
            )
        if trajectories.shape[0] != sample_count:
            raise ValueError("trajectory sample count must match representations")
        if not np.all(np.isfinite(trajectories)):
            raise ValueError("trajectories must contain only finite values")
        object.__setattr__(self, "trajectories", _readonly(trajectories))

        if (
            not isinstance(self.parameter_digest, str)
            or len(self.parameter_digest) != 64
            or any(
                char not in "0123456789abcdef"
                for char in self.parameter_digest
            )
        ):
            raise ValueError("parameter_digest must be a lowercase SHA-256 digest")


def registered_arms() -> tuple[ArmSpec, ...]:
    return tuple(
        ArmSpec(architecture, seed)
        for architecture in REGISTERED_ARCHITECTURES
        for seed in REGISTERED_SEEDS
    )


def build_representations(
    sequences: np.ndarray,
    *,
    architecture: E2Architecture,
    seed: int,
) -> RepresentationSet:
    values = _sequences(sequences)
    _registered_architecture(architecture)
    _registered_seed(seed)

    reservoir = build_e2_reservoir(
        E2ReservoirSpec(
            architecture=architecture,
            input_size=INPUT_CHANNELS,
            seed=seed,
        )
    )
    parameter_digest = reservoir.parameter_digest()
    trajectories = np.zeros(
        (values.shape[0], REGISTERED_BIN_COUNT, RESERVOIR_DIM),
        dtype=np.float64,
    )

    for sample_index, sequence in enumerate(values):
        reservoir.reset()
        for step, observation in enumerate(sequence):
            trajectories[sample_index, step] = reservoir.advance(observation)
        if reservoir.parameter_digest() != parameter_digest:
            raise RuntimeError("reservoir parameter digest changed during arm")

    c0 = values[:, 19, :]
    s4 = values[:, 16:20, :].reshape(values.shape[0], 56)
    r1 = trajectories[:, 19, :]
    r4 = np.mean(
        trajectories[:, 16:20, :],
        axis=1,
        dtype=np.float64,
    )
    return RepresentationSet(
        c0=c0,
        s4=s4,
        trajectories=trajectories,
        r1=r1,
        r4=r4,
        parameter_digest=parameter_digest,
    )


def _sequences(values: object) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("sequences must be float64-compatible") from exc
    if (
        array.ndim != 3
        or array.shape[1:] != (REGISTERED_BIN_COUNT, INPUT_CHANNELS)
        or array.shape[0] == 0
    ):
        raise ValueError("sequences must have shape (samples, 20, 14)")
    if not np.all(np.isfinite(array)):
        raise ValueError("sequences must contain only finite values")
    return array


def _registered_architecture(value: object) -> E2Architecture:
    if not isinstance(value, E2Architecture) or value not in REGISTERED_ARCHITECTURES:
        raise ValueError("architecture must be a registered R1-E3M architecture")
    return value


def _registered_seed(value: object) -> int:
    if type(value) is not int or value not in REGISTERED_SEEDS:
        raise ValueError("seed must be a registered R1-E3M seed")
    return value


def _matrix(values: object, *, name: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be float64-compatible") from exc
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty rank-two array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _readonly(values: np.ndarray) -> np.ndarray:
    copied = np.array(values, dtype=np.float64, copy=True, order="C")
    copied.flags.writeable = False
    return copied
