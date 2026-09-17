from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from .r1_e1_reservoir import (
    ReservoirArchitecture,
    ReservoirSpec,
    build_reservoir,
)


_E2_BUDGET = 256
_E2_RADIUS = 0.9
_E2_LEAK = 1.0
_E2_ACTIVATION = "tanh"
_DIGEST_VERSION = b"r1-e2-reservoir-v1\0"


class E2Architecture(IntEnum):
    FLAT = 0
    GROUPED4 = 1
    HIERARCHICAL2 = 2
    HIERARCHICAL4 = 3


@dataclass(frozen=True)
class E2ReservoirSpec:
    architecture: E2Architecture
    input_size: int
    seed: int

    def __post_init__(self) -> None:
        if not isinstance(self.architecture, E2Architecture):
            raise ValueError("architecture must be a registered E2Architecture")
        if type(self.input_size) is not int or self.input_size <= 0:
            raise ValueError("input_size must be a positive Python integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative Python integer")


class E2Reservoir:
    def __init__(self, spec: E2ReservoirSpec) -> None:
        if not isinstance(spec, E2ReservoirSpec):
            raise ValueError("spec must be an E2ReservoirSpec")
        self._spec = spec
        self._e1_architecture = _e1_architecture(spec.architecture)
        self._core = build_reservoir(
            ReservoirSpec(
                architecture=self._e1_architecture,
                budget=_E2_BUDGET,
                input_size=spec.input_size,
                seed=spec.seed,
            )
        )

    @property
    def component_widths(self) -> tuple[int, ...]:
        return self._core.component_widths

    @property
    def state_dim(self) -> int:
        return self._core.state_dim

    @property
    def neuron_count(self) -> int:
        return _E2_BUDGET

    @property
    def activation(self) -> str:
        return _E2_ACTIVATION

    @property
    def spectral_radius(self) -> float:
        return _E2_RADIUS

    @property
    def leak(self) -> float:
        return _E2_LEAK

    @property
    def recurrent_edge_count(self) -> int:
        return sum(width * width for width in self.component_widths)

    @property
    def input_parameter_count(self) -> int:
        if self._spec.architecture in (E2Architecture.FLAT, E2Architecture.GROUPED4):
            return sum(width * self._spec.input_size for width in self.component_widths)

        total = self.component_widths[0] * self._spec.input_size
        for previous_width, width in zip(self.component_widths, self.component_widths[1:]):
            total += width * previous_width
        return total

    @property
    def total_parameter_count(self) -> int:
        return self.recurrent_edge_count + self.input_parameter_count

    def reset(self) -> None:
        self._core.reset()

    def advance(self, observation: np.ndarray) -> np.ndarray:
        return self._core.advance(observation)

    def parameter_digest(self) -> str:
        digest = hashlib.sha256()
        digest.update(_DIGEST_VERSION)
        digest.update(str(int(self._spec.architecture)).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(self._spec.input_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(self._spec.seed).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(self.neuron_count).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(self.recurrent_edge_count).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(self.input_parameter_count).encode("ascii"))
        digest.update(b"\0")
        digest.update(self._core.parameter_digest().encode("ascii"))
        return digest.hexdigest()


def build_e2_reservoir(spec: E2ReservoirSpec) -> E2Reservoir:
    return E2Reservoir(spec)


def _e1_architecture(architecture: E2Architecture) -> ReservoirArchitecture:
    if architecture is E2Architecture.FLAT:
        return ReservoirArchitecture.SHALLOW
    if architecture is E2Architecture.GROUPED4:
        return ReservoirArchitecture.GROUPED4
    if architecture is E2Architecture.HIERARCHICAL2:
        return ReservoirArchitecture.DEEP2
    if architecture is E2Architecture.HIERARCHICAL4:
        return ReservoirArchitecture.DEEP4
    raise ValueError("architecture must be a registered E2Architecture")
