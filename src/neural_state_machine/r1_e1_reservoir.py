from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from enum import IntEnum

import numpy as np


_RECURRENT_RADIUS = 0.9
_RESERVOIR_LINEAGE_TAG = 0x52314531
_BUDGET_IDS = {64: 0, 256: 1}
_DIGEST_VERSION = b"r1-e1-reservoir-v1\0"


class ReservoirArchitecture(IntEnum):
    SHALLOW = 0
    GROUPED2 = 1
    GROUPED4 = 2
    DEEP2 = 3
    DEEP4 = 4


@dataclass(frozen=True)
class ReservoirSpec:
    architecture: ReservoirArchitecture
    budget: int
    input_size: int
    seed: int

    def __post_init__(self) -> None:
        if not isinstance(self.architecture, ReservoirArchitecture):
            raise ValueError("architecture must be a registered ReservoirArchitecture")
        if type(self.budget) is not int or self.budget not in _BUDGET_IDS:
            raise ValueError("budget must be one of the registered values 64 or 256")
        if type(self.input_size) is not int or self.input_size <= 0:
            raise ValueError("input_size must be a positive Python integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative Python integer")


class _RecurrentComponent:
    def __init__(
        self,
        *,
        spec: ReservoirSpec,
        component_id: int,
        input_size: int,
        hidden_size: int,
    ) -> None:
        rng = np.random.default_rng(
            np.random.SeedSequence(
                [
                    spec.seed,
                    _RESERVOIR_LINEAGE_TAG,
                    _BUDGET_IDS[spec.budget],
                    int(spec.architecture),
                    component_id,
                ]
            )
        )
        input_weights = rng.normal(
            0.0,
            1.0 / math.sqrt(input_size),
            size=(hidden_size, input_size),
        )
        recurrent = rng.normal(
            0.0,
            1.0 / math.sqrt(hidden_size),
            size=(hidden_size, hidden_size),
        )
        radius = float(np.max(np.abs(np.linalg.eigvals(recurrent))))
        if not math.isfinite(radius) or radius <= 0.0:
            raise RuntimeError("recurrent initialization produced an invalid spectral radius")
        recurrent_weights = recurrent * (_RECURRENT_RADIUS / radius)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self._input_weights = _readonly_copy(input_weights)
        self._recurrent_weights = _readonly_copy(recurrent_weights)
        self._state = np.zeros(hidden_size, dtype=np.float64)

    def reset(self) -> None:
        self._state.fill(0.0)

    def advance(self, observation: np.ndarray) -> np.ndarray:
        self._state = np.tanh(
            self._input_weights @ observation + self._recurrent_weights @ self._state
        )
        return self._state


class Reservoir:
    def __init__(self, spec: ReservoirSpec) -> None:
        if not isinstance(spec, ReservoirSpec):
            raise ValueError("spec must be a ReservoirSpec")
        self._spec = spec
        count = _component_count(spec.architecture)
        if spec.budget % count:
            raise ValueError("budget must divide evenly across reservoir components")
        width = spec.budget // count
        self._component_widths = tuple(width for _ in range(count))

        components: list[_RecurrentComponent] = []
        current_input_size = spec.input_size
        for component_id, hidden_size in enumerate(self._component_widths):
            component_input_size = (
                current_input_size
                if _is_deep(spec.architecture)
                else spec.input_size
            )
            components.append(
                _RecurrentComponent(
                    spec=spec,
                    component_id=component_id,
                    input_size=component_input_size,
                    hidden_size=hidden_size,
                )
            )
            if _is_deep(spec.architecture):
                current_input_size = hidden_size
        self._components = tuple(components)

    @property
    def component_widths(self) -> tuple[int, ...]:
        return self._component_widths

    @property
    def state_dim(self) -> int:
        return sum(self._component_widths)

    def reset(self) -> None:
        for component in self._components:
            component.reset()

    def advance(self, observation: np.ndarray) -> np.ndarray:
        values = _validated_observation(observation, self._spec.input_size)
        if _is_deep(self._spec.architecture):
            parts = []
            current = values
            for component in self._components:
                current = component.advance(current)
                parts.append(current)
        else:
            parts = [component.advance(values) for component in self._components]
        return _readonly_copy(np.concatenate(parts))

    def parameter_digest(self) -> str:
        digest = hashlib.sha256()
        digest.update(_DIGEST_VERSION)
        digest.update(str(int(self._spec.architecture)).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(self._spec.budget).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(self._spec.input_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(self._spec.seed).encode("ascii"))
        digest.update(b"\0")
        digest.update(repr(self._component_widths).encode("ascii"))
        digest.update(b"\0")
        for component in self._components:
            _digest_array(digest, component._input_weights)
            _digest_array(digest, component._recurrent_weights)
        return digest.hexdigest()


def build_reservoir(spec: ReservoirSpec) -> Reservoir:
    return Reservoir(spec)


def _component_count(architecture: ReservoirArchitecture) -> int:
    if architecture is ReservoirArchitecture.SHALLOW:
        return 1
    if architecture in (ReservoirArchitecture.GROUPED2, ReservoirArchitecture.DEEP2):
        return 2
    return 4


def _is_deep(architecture: ReservoirArchitecture) -> bool:
    return architecture in (ReservoirArchitecture.DEEP2, ReservoirArchitecture.DEEP4)


def _validated_observation(observation: object, input_size: int) -> np.ndarray:
    try:
        values = np.asarray(observation, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("observation must be float64-compatible") from exc
    if values.ndim != 1 or values.shape != (input_size,):
        raise ValueError(f"observation must have shape ({input_size},)")
    if not np.all(np.isfinite(values)):
        raise ValueError("observation must contain only finite values")
    return values


def _readonly_copy(values: np.ndarray) -> np.ndarray:
    copied = np.array(values, dtype=np.float64, copy=True, order="C")
    copied.flags.writeable = False
    return copied


def _digest_array(digest: hashlib._Hash, values: np.ndarray) -> None:
    array = np.ascontiguousarray(values, dtype=np.float64)
    digest.update(str(array.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
