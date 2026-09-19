from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .r1_e2_reservoir import E2Architecture
from .r1_e3m_regression import (
    MultiOutputRidgeProbe,
    R2Summary,
    fit_multioutput_ridge,
    r2_summary,
)
from .r1_e3m_representation import (
    RepresentationSet,
    build_representations,
)
from .r1_e3m_windows import UnlabeledWindow


REGISTERED_DELAYS = (1, 2, 5, 10, 15)
PRIMARY_DELAY = 10


@dataclass(frozen=True)
class DelayProbeSet:
    c0: MultiOutputRidgeProbe
    s4: MultiOutputRidgeProbe
    r1: MultiOutputRidgeProbe
    r4: MultiOutputRidgeProbe


@dataclass(frozen=True)
class ArmMemoryFit:
    architecture: E2Architecture
    seed: int
    parameter_digest: str
    probes: Mapping[int, DelayProbeSet]
    train_sequence_ids: tuple[str, ...]

    @property
    def probe_digests(self) -> dict[int, dict[str, str]]:
        return {
            delay: {
                "c0": probe_set.c0.coefficient_digest(),
                "s4": probe_set.s4.coefficient_digest(),
                "r1": probe_set.r1.coefficient_digest(),
                "r4": probe_set.r4.coefficient_digest(),
            }
            for delay, probe_set in sorted(self.probes.items())
        }


@dataclass(frozen=True)
class RepresentationScore:
    r2: R2Summary
    coefficient_digest: str

    @property
    def macro_r2(self) -> float:
        return self.r2.macro_r2


@dataclass(frozen=True)
class DelayBenchmarkResult:
    delay: int
    c0: RepresentationScore
    s4: RepresentationScore
    r1: RepresentationScore
    r4: RepresentationScore

    @property
    def delta_r1_vs_s4(self) -> float:
        return self.r1.macro_r2 - self.s4.macro_r2


@dataclass(frozen=True)
class ArmMemoryResult:
    architecture: E2Architecture
    seed: int
    parameter_digest: str
    eval_sequence_ids: tuple[str, ...]
    delays: Mapping[int, DelayBenchmarkResult]


def target_for_delay(
    tensors: np.ndarray,
    delay: int,
) -> np.ndarray:
    values = _tensor_batch(tensors)
    _registered_delay(delay)
    result = np.array(
        values[:, 19 - delay, :],
        dtype=np.float64,
        copy=True,
        order="C",
    )
    result.flags.writeable = False
    return result


def fit_arm_memory(
    windows: tuple[UnlabeledWindow, ...],
    *,
    architecture: E2Architecture,
    seed: int,
) -> ArmMemoryFit:
    _validate_window_identity(windows)
    training = tuple(window for window in windows if window.split == "train")
    if not training:
        raise ValueError("R1-E3M fitting requires training windows")

    tensors = _windows_tensor(training)
    representations = build_representations(
        tensors,
        architecture=architecture,
        seed=seed,
    )

    probes: dict[int, DelayProbeSet] = {}
    for delay in REGISTERED_DELAYS:
        targets = target_for_delay(tensors, delay)
        probes[delay] = DelayProbeSet(
            c0=fit_multioutput_ridge(representations.c0, targets),
            s4=fit_multioutput_ridge(representations.s4, targets),
            r1=fit_multioutput_ridge(representations.r1, targets),
            r4=fit_multioutput_ridge(representations.r4, targets),
        )

    return ArmMemoryFit(
        architecture=architecture,
        seed=seed,
        parameter_digest=representations.parameter_digest,
        probes=probes,
        train_sequence_ids=tuple(window.sequence_id for window in training),
    )


def evaluate_arm_memory(
    fitted: ArmMemoryFit,
    windows: tuple[UnlabeledWindow, ...],
) -> ArmMemoryResult:
    if not isinstance(fitted, ArmMemoryFit):
        raise ValueError("fitted must be ArmMemoryFit")
    _validate_window_identity(windows)
    evaluation = tuple(window for window in windows if window.split == "eval")
    if not evaluation:
        raise ValueError("R1-E3M evaluation requires evaluation windows")

    tensors = _windows_tensor(evaluation)
    representations = build_representations(
        tensors,
        architecture=fitted.architecture,
        seed=fitted.seed,
    )
    if representations.parameter_digest != fitted.parameter_digest:
        raise RuntimeError("reservoir parameter digest changed between fit/eval")

    results: dict[int, DelayBenchmarkResult] = {}
    for delay in REGISTERED_DELAYS:
        probe_set = fitted.probes.get(delay)
        if probe_set is None:
            raise ValueError(f"missing fitted probe for delay {delay}")
        targets = target_for_delay(tensors, delay)
        results[delay] = _score_delay(
            delay,
            targets,
            representations,
            probe_set,
        )

    return ArmMemoryResult(
        architecture=fitted.architecture,
        seed=fitted.seed,
        parameter_digest=fitted.parameter_digest,
        eval_sequence_ids=tuple(window.sequence_id for window in evaluation),
        delays=results,
    )


def _score_delay(
    delay: int,
    targets: np.ndarray,
    representations: RepresentationSet,
    probes: DelayProbeSet,
) -> DelayBenchmarkResult:
    return DelayBenchmarkResult(
        delay=delay,
        c0=_score(probes.c0, representations.c0, targets),
        s4=_score(probes.s4, representations.s4, targets),
        r1=_score(probes.r1, representations.r1, targets),
        r4=_score(probes.r4, representations.r4, targets),
    )


def _score(
    probe: MultiOutputRidgeProbe,
    features: np.ndarray,
    targets: np.ndarray,
) -> RepresentationScore:
    prediction = probe.predict(features)
    return RepresentationScore(
        r2=r2_summary(targets, prediction),
        coefficient_digest=probe.coefficient_digest(),
    )


def _windows_tensor(
    windows: tuple[UnlabeledWindow, ...],
) -> np.ndarray:
    values = np.stack([window.tensor for window in windows], axis=0)
    result = np.array(values, dtype=np.float64, copy=True, order="C")
    result.flags.writeable = False
    return result


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


def _registered_delay(value: object) -> int:
    if type(value) is not int or value not in REGISTERED_DELAYS:
        raise ValueError("delay must be a registered delay")
    return value


def _validate_window_identity(
    windows: tuple[UnlabeledWindow, ...],
) -> None:
    if not windows:
        raise ValueError("windows must be non-empty")
    ids: set[str] = set()
    for window in windows:
        if not isinstance(window, UnlabeledWindow):
            raise ValueError("windows must contain UnlabeledWindow")
        if window.sequence_id in ids:
            raise ValueError("duplicate sequence_id")
        ids.add(window.sequence_id)
        if window.split not in ("train", "eval"):
            raise ValueError("window split must be train or eval")
