from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from neural_state_machine.r1_e2_reservoir import (
    E2ReservoirSpec,
    build_e2_reservoir,
)
from neural_state_machine.r1_e3m_probe import (
    MultiTargetRidgeProbe,
    RegressionMetrics,
    evaluate_regression,
    regression_prediction_digest,
)


@dataclass(frozen=True)
class FixedProbeControlResult:
    metrics: RegressionMetrics
    coefficient_digest: str
    prediction_digest: str


def collect_final_states(
    spec: E2ReservoirSpec,
    tensors: np.ndarray,
) -> np.ndarray:
    values = _validated_tensors(tensors)
    _validated_spec(spec)
    reservoir = build_e2_reservoir(spec)
    states = []
    for tensor in values:
        reservoir.reset()
        final_state = None
        for row in tensor:
            final_state = reservoir.advance(row)
        assert final_state is not None
        states.append(final_state)
    return _readonly(np.stack(states, axis=0))


def collect_reset_final_states(
    spec: E2ReservoirSpec,
    tensors: np.ndarray,
) -> np.ndarray:
    values = _validated_tensors(tensors)
    _validated_spec(spec)
    reservoir = build_e2_reservoir(spec)
    states = []
    for tensor in values:
        reservoir.reset()
        for row in tensor[:16]:
            reservoir.advance(row)
        reservoir.reset()
        final_state = None
        for row in tensor[16:20]:
            final_state = reservoir.advance(row)
        assert final_state is not None
        states.append(final_state)
    return _readonly(np.stack(states, axis=0))


def permute_prefix(
    tensors: np.ndarray,
    window_ids: tuple[str, ...],
) -> np.ndarray:
    values = _validated_tensors(tensors)
    if len(window_ids) != values.shape[0]:
        raise ValueError(
            "window_ids must match the tensor sample count"
        )

    output = np.array(values, dtype=np.float64, copy=True, order="C")
    for index, window_id in enumerate(window_ids):
        _validated_window_id(window_id)
        seed_bytes = hashlib.sha256(
            window_id.encode("ascii")
        ).digest()[:8]
        seed = int.from_bytes(seed_bytes, "big", signed=False)
        permutation = np.random.default_rng(seed).permutation(16)
        output[index, :16] = values[index, permutation]

    output.setflags(write=False)
    return output


def collect_prefix_permuted_final_states(
    spec: E2ReservoirSpec,
    tensors: np.ndarray,
    window_ids: tuple[str, ...],
) -> np.ndarray:
    return collect_final_states(
        spec,
        permute_prefix(tensors, window_ids),
    )


def evaluate_fixed_probe(
    probe: MultiTargetRidgeProbe,
    final_states: np.ndarray,
    targets: np.ndarray,
) -> FixedProbeControlResult:
    if not isinstance(probe, MultiTargetRidgeProbe):
        raise ValueError("probe must be a fitted MultiTargetRidgeProbe")
    predictions = probe.predict(final_states)
    return FixedProbeControlResult(
        metrics=evaluate_regression(targets, predictions),
        coefficient_digest=probe.coefficient_digest(),
        prediction_digest=regression_prediction_digest(predictions),
    )


def _validated_spec(spec: object) -> E2ReservoirSpec:
    if not isinstance(spec, E2ReservoirSpec):
        raise ValueError("spec must be E2ReservoirSpec")
    if spec.input_size != 6:
        raise ValueError("R1-E3M reservoir input_size must be 6")
    return spec


def _validated_tensors(values: object) -> np.ndarray:
    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ValueError("tensors must be NumPy-compatible") from exc
    if array.dtype != np.float64:
        raise ValueError("tensors must use float64")
    if array.ndim != 3 or array.shape[1:] != (20, 6):
        raise ValueError("tensors must have shape (samples, 20, 6)")
    if array.shape[0] == 0:
        raise ValueError("tensors must contain samples")
    if not np.isfinite(array).all():
        raise ValueError("tensors must contain only finite values")
    return array


def _validated_window_id(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError("window_id must be a lowercase SHA-256 digest")
    return value


def _readonly(values: np.ndarray) -> np.ndarray:
    copied = np.array(values, dtype=np.float64, copy=True, order="C")
    copied.setflags(write=False)
    return copied
