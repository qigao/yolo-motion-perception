from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .r1_e2_reservoir import (
    E2ReservoirSpec,
    build_e2_reservoir,
)
from .r1_e3m_benchmark import (
    REGISTERED_DELAYS,
    ArmMemoryFit,
    RepresentationScore,
    target_for_delay,
)
from .r1_e3m_regression import MultiOutputRidgeProbe, r2_summary
from .r1_e3m_representation import build_representations
from .r1_e3m_windows import UnlabeledWindow


_SHUFFLE_TAG = b"r1-e3m-prefix-shuffle-v1|"


@dataclass(frozen=True)
class HistoryDelayResult:
    delay: int
    full_r1: RepresentationScore
    reset_r1: RepresentationScore
    reverse_r1: RepresentationScore
    shuffle_r1: RepresentationScore
    full_r4: RepresentationScore
    reset_r4: RepresentationScore
    reverse_r4: RepresentationScore
    shuffle_r4: RepresentationScore

    @property
    def reset_drop_r1(self) -> float:
        return self.full_r1.macro_r2 - self.reset_r1.macro_r2

    @property
    def reverse_drop_r1(self) -> float:
        return self.full_r1.macro_r2 - self.reverse_r1.macro_r2

    @property
    def shuffle_drop_r1(self) -> float:
        return self.full_r1.macro_r2 - self.shuffle_r1.macro_r2


@dataclass(frozen=True)
class HistoryDestructionResult:
    delays: Mapping[int, HistoryDelayResult]


def transform_prefix(
    tensors: np.ndarray,
    sequence_ids: tuple[str, ...],
    *,
    mode: str,
) -> np.ndarray:
    values = _tensor_batch(tensors)
    if len(sequence_ids) != values.shape[0]:
        raise ValueError("sequence_ids must match tensor sample count")
    if any(not isinstance(item, str) or not item for item in sequence_ids):
        raise ValueError("sequence_ids must be non-empty strings")
    if mode not in ("reverse", "shuffle"):
        raise ValueError("mode must be reverse or shuffle")

    transformed = np.array(values, dtype=np.float64, copy=True, order="C")
    if mode == "reverse":
        transformed[:, :16, :] = values[:, 15::-1, :]
    else:
        for index, sequence_id in enumerate(sequence_ids):
            permutation = _shuffle_permutation(sequence_id)
            transformed[index, :16, :] = values[index, permutation, :]

    if not np.array_equal(transformed[:, 16:20], values[:, 16:20]):
        raise RuntimeError("history transformation changed frozen suffix")
    transformed.flags.writeable = False
    return transformed


def evaluate_history_destruction(
    fitted: ArmMemoryFit,
    windows: tuple[UnlabeledWindow, ...],
) -> HistoryDestructionResult:
    if not isinstance(fitted, ArmMemoryFit):
        raise ValueError("fitted must be ArmMemoryFit")
    evaluation = tuple(window for window in windows if window.split == "eval")
    if not evaluation:
        raise ValueError("history destruction requires evaluation windows")

    sequence_ids = tuple(window.sequence_id for window in evaluation)
    if len(set(sequence_ids)) != len(sequence_ids):
        raise ValueError("duplicate evaluation sequence_id")

    tensors = np.stack([window.tensor for window in evaluation], axis=0)
    normal = build_representations(
        tensors,
        architecture=fitted.architecture,
        seed=fitted.seed,
    )
    if normal.parameter_digest != fitted.parameter_digest:
        raise RuntimeError("reservoir parameter digest changed in history control")

    reversed_tensors = transform_prefix(
        tensors,
        sequence_ids,
        mode="reverse",
    )
    shuffled_tensors = transform_prefix(
        tensors,
        sequence_ids,
        mode="shuffle",
    )
    reversed_rep = build_representations(
        reversed_tensors,
        architecture=fitted.architecture,
        seed=fitted.seed,
    )
    shuffled_rep = build_representations(
        shuffled_tensors,
        architecture=fitted.architecture,
        seed=fitted.seed,
    )
    reset_r1, reset_r4, reset_digest = _reset_suffix_features(
        tensors,
        architecture=fitted.architecture,
        seed=fitted.seed,
    )

    for digest in (
        reversed_rep.parameter_digest,
        shuffled_rep.parameter_digest,
        reset_digest,
    ):
        if digest != fitted.parameter_digest:
            raise RuntimeError("reservoir parameter digest changed in history control")

    delays: dict[int, HistoryDelayResult] = {}
    for delay in REGISTERED_DELAYS:
        targets = target_for_delay(tensors, delay)
        probes = fitted.probes.get(delay)
        if probes is None:
            raise ValueError(f"missing fitted probe for delay {delay}")

        delays[delay] = HistoryDelayResult(
            delay=delay,
            full_r1=_score(probes.r1, normal.r1, targets),
            reset_r1=_score(probes.r1, reset_r1, targets),
            reverse_r1=_score(probes.r1, reversed_rep.r1, targets),
            shuffle_r1=_score(probes.r1, shuffled_rep.r1, targets),
            full_r4=_score(probes.r4, normal.r4, targets),
            reset_r4=_score(probes.r4, reset_r4, targets),
            reverse_r4=_score(probes.r4, reversed_rep.r4, targets),
            shuffle_r4=_score(probes.r4, shuffled_rep.r4, targets),
        )
    return HistoryDestructionResult(delays=delays)


def _reset_suffix_features(
    tensors: np.ndarray,
    *,
    architecture,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    values = _tensor_batch(tensors)
    reservoir = build_e2_reservoir(
        E2ReservoirSpec(
            architecture=architecture,
            input_size=14,
            seed=seed,
        )
    )
    digest = reservoir.parameter_digest()
    r1 = np.zeros((values.shape[0], 256), dtype=np.float64)
    r4 = np.zeros((values.shape[0], 256), dtype=np.float64)

    for sample_index, sequence in enumerate(values):
        reservoir.reset()
        for observation in sequence[:16]:
            reservoir.advance(observation)
        reservoir.reset()

        suffix_states = np.zeros((4, 256), dtype=np.float64)
        for offset, observation in enumerate(sequence[16:20]):
            suffix_states[offset] = reservoir.advance(observation)
        r1[sample_index] = suffix_states[-1]
        r4[sample_index] = np.mean(suffix_states, axis=0, dtype=np.float64)

        if reservoir.parameter_digest() != digest:
            raise RuntimeError("reservoir parameter digest changed during reset")

    r1.flags.writeable = False
    r4.flags.writeable = False
    return r1, r4, digest


def _score(
    probe: MultiOutputRidgeProbe,
    features: np.ndarray,
    targets: np.ndarray,
) -> RepresentationScore:
    return RepresentationScore(
        r2=r2_summary(targets, probe.predict(features)),
        coefficient_digest=probe.coefficient_digest(),
    )


def _shuffle_permutation(sequence_id: str) -> np.ndarray:
    base = hashlib.sha256(
        _SHUFFLE_TAG + sequence_id.encode("utf-8")
    ).digest()
    keyed = []
    for index in range(16):
        digest = hashlib.sha256(base + index.to_bytes(2, "big")).digest()
        keyed.append((digest, index))
    return np.asarray(
        [index for _, index in sorted(keyed)],
        dtype=np.int64,
    )


def _tensor_batch(values: object) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("tensors must be float64-compatible") from exc
    if array.ndim != 3 or array.shape[1:] != (20, 14) or array.shape[0] == 0:
        raise ValueError("tensors must have shape (samples, 20, 14)")
    if not np.all(np.isfinite(array)):
        raise ValueError("tensors must contain only finite values")
    return array
