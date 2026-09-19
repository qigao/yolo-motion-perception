from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median

import numpy as np

from neural_state_machine.r1_e2_reservoir import (
    E2Architecture,
    E2ReservoirSpec,
    build_e2_reservoir,
)
from neural_state_machine.r1_e3m_controls import (
    FixedProbeControlResult,
    collect_final_states,
    collect_prefix_permuted_final_states,
    collect_reset_final_states,
    evaluate_fixed_probe,
)
from neural_state_machine.r1_e3m_dataset import MechanismDataset
from neural_state_machine.r1_e3m_pairs import (
    HistoryPairSet,
    PairStateDiagnostic,
    build_history_pairs,
    score_history_pairs,
)
from neural_state_machine.r1_e3m_probe import (
    DELAYS,
    MultiTargetRidgeProbe,
    RegressionMetrics,
    delay_valid_mask,
    delayed_geometry_targets,
    evaluate_regression,
    fit_instantaneous_delay_probe,
    fit_reservoir_delay_probe,
    regression_prediction_digest,
)


REGISTERED_ARCHITECTURES = (
    E2Architecture.FLAT,
    E2Architecture.GROUPED4,
    E2Architecture.HIERARCHICAL2,
    E2Architecture.HIERARCHICAL4,
)
REGISTERED_SEEDS = (7, 17, 29, 43, 61)
ARM_COUNT = len(REGISTERED_ARCHITECTURES) * len(REGISTERED_SEEDS)
LONG_DELAYS = (5, 10, 15)


@dataclass(frozen=True)
class ProbeResult:
    metrics: RegressionMetrics
    coefficient_digest: str
    prediction_digest: str


@dataclass(frozen=True)
class DelayMemoryResult:
    delay: int
    training_sample_count: int
    evaluation_sample_count: int
    instantaneous: ProbeResult
    reservoir: ProbeResult
    reset_control: FixedProbeControlResult
    permuted_control: FixedProbeControlResult
    delta_r2: float
    h1_drop_r2: float
    h2_drop_r2: float


@dataclass(frozen=True)
class MemoryArmResult:
    seed: int
    architecture: int
    delays: tuple[DelayMemoryResult, ...]
    long_delay_delta: float
    h1_long_delay_drop: float
    h2_long_delay_drop: float
    reservoir_parameter_digest: str
    artifact_root_digest: str
    pair_set_digest: str
    pair_diagnostics: tuple[PairStateDiagnostic, ...]


@dataclass(frozen=True)
class RegisteredMemoryResult:
    arms: tuple[MemoryArmResult, ...]
    median_long_delay_delta: float
    positive_arm_count: int
    median_h1_long_delay_drop: float
    outcome: str
    pair_set_digest: str
    history_pair_count: int
    history_prefix_threshold: float


def registered_specs() -> tuple[E2ReservoirSpec, ...]:
    return tuple(
        E2ReservoirSpec(
            architecture=architecture,
            input_size=6,
            seed=seed,
        )
        for seed in REGISTERED_SEEDS
        for architecture in REGISTERED_ARCHITECTURES
    )


def evaluate_memory_arm(
    spec: E2ReservoirSpec,
    dataset: MechanismDataset,
    pair_set: HistoryPairSet,
) -> MemoryArmResult:
    _validated_spec(spec)
    if not isinstance(dataset, MechanismDataset):
        raise ValueError("dataset must be MechanismDataset")
    pair_set = _validated_pair_set(dataset, pair_set)

    training_tensors, _ = _sample_matrix(dataset.training)
    evaluation_tensors, evaluation_ids = _sample_matrix(
        dataset.evaluation
    )

    training_states = collect_final_states(spec, training_tensors)
    evaluation_states = collect_final_states(spec, evaluation_tensors)
    reset_states = collect_reset_final_states(spec, evaluation_tensors)
    permuted_states = collect_prefix_permuted_final_states(
        spec,
        evaluation_tensors,
        evaluation_ids,
    )

    parameter_digest = build_e2_reservoir(spec).parameter_digest()
    pair_diagnostics = score_history_pairs(
        pair_set,
        dataset.evaluation,
        evaluation_states,
        reset_states,
    )

    delay_results: list[DelayMemoryResult] = []
    for delay in DELAYS:
        training_mask = delay_valid_mask(
            training_tensors,
            delay,
        )
        evaluation_mask = delay_valid_mask(
            evaluation_tensors,
            delay,
        )
        if not np.any(evaluation_mask):
            raise ValueError(
                "no evaluation windows have an observed delayed target "
                f"for delay {delay}"
            )
        masked_evaluation_tensors = evaluation_tensors[evaluation_mask]
        targets = delayed_geometry_targets(
            masked_evaluation_tensors,
            delay,
        )

        instantaneous_probe = fit_instantaneous_delay_probe(
            training_tensors,
            delay,
        )
        reservoir_probe = fit_reservoir_delay_probe(
            training_states,
            training_tensors,
            delay,
        )

        instantaneous_predictions = instantaneous_probe.predict(
            masked_evaluation_tensors[:, 19, :]
        )
        reservoir_predictions = reservoir_probe.predict(
            evaluation_states[evaluation_mask]
        )

        instantaneous = _probe_result(
            instantaneous_probe,
            targets,
            instantaneous_predictions,
        )
        reservoir = _probe_result(
            reservoir_probe,
            targets,
            reservoir_predictions,
        )
        reset_control = evaluate_fixed_probe(
            reservoir_probe,
            reset_states[evaluation_mask],
            targets,
        )
        permuted_control = evaluate_fixed_probe(
            reservoir_probe,
            permuted_states[evaluation_mask],
            targets,
        )

        if reset_control.coefficient_digest != reservoir.coefficient_digest:
            raise RuntimeError("H1 control refit detected")
        if (
            permuted_control.coefficient_digest
            != reservoir.coefficient_digest
        ):
            raise RuntimeError("H2 control refit detected")

        delay_results.append(
            DelayMemoryResult(
                delay=delay,
                training_sample_count=int(np.count_nonzero(training_mask)),
                evaluation_sample_count=int(np.count_nonzero(evaluation_mask)),
                instantaneous=instantaneous,
                reservoir=reservoir,
                reset_control=reset_control,
                permuted_control=permuted_control,
                delta_r2=(
                    reservoir.metrics.mean_r2
                    - instantaneous.metrics.mean_r2
                ),
                h1_drop_r2=(
                    reservoir.metrics.mean_r2
                    - reset_control.metrics.mean_r2
                ),
                h2_drop_r2=(
                    reservoir.metrics.mean_r2
                    - permuted_control.metrics.mean_r2
                ),
            )
        )

    by_delay = {result.delay: result for result in delay_results}
    long_delay_delta = _mean(
        by_delay[delay].delta_r2 for delay in LONG_DELAYS
    )
    h1_long_delay_drop = _mean(
        by_delay[delay].h1_drop_r2 for delay in LONG_DELAYS
    )
    h2_long_delay_drop = _mean(
        by_delay[delay].h2_drop_r2 for delay in LONG_DELAYS
    )

    return MemoryArmResult(
        seed=spec.seed,
        architecture=int(spec.architecture),
        delays=tuple(delay_results),
        long_delay_delta=long_delay_delta,
        h1_long_delay_drop=h1_long_delay_drop,
        h2_long_delay_drop=h2_long_delay_drop,
        reservoir_parameter_digest=parameter_digest,
        artifact_root_digest=dataset.artifact_root_digest,
        pair_set_digest=pair_set.pair_digest,
        pair_diagnostics=pair_diagnostics,
    )


def run_registered_memory_benchmark(
    dataset: MechanismDataset,
    pair_set: HistoryPairSet,
) -> RegisteredMemoryResult:
    if not isinstance(dataset, MechanismDataset):
        raise ValueError("dataset must be MechanismDataset")
    pair_set = _validated_pair_set(dataset, pair_set)

    arms = tuple(
        evaluate_memory_arm(spec, dataset, pair_set)
        for spec in registered_specs()
    )
    if len(arms) != ARM_COUNT:
        raise RuntimeError("registered R1-E3M arm count mismatch")

    deltas = [arm.long_delay_delta for arm in arms]
    h1_drops = [arm.h1_long_delay_drop for arm in arms]
    return RegisteredMemoryResult(
        arms=arms,
        median_long_delay_delta=float(median(deltas)),
        positive_arm_count=sum(value > 0.0 for value in deltas),
        median_h1_long_delay_drop=float(median(h1_drops)),
        outcome=classify_memory_outcome(deltas, h1_drops),
        pair_set_digest=pair_set.pair_digest,
        history_pair_count=len(pair_set.pairs),
        history_prefix_threshold=pair_set.prefix_threshold,
    )


def classify_memory_outcome(
    long_delay_deltas: object,
    h1_long_delay_drops: object,
) -> str:
    deltas = _validated_arm_values(
        long_delay_deltas,
        "long_delay_deltas",
    )
    h1_drops = _validated_arm_values(
        h1_long_delay_drops,
        "h1_long_delay_drops",
    )
    median_delta = float(median(deltas))
    positive_count = sum(value > 0.0 for value in deltas)
    median_h1_drop = float(median(h1_drops))

    if (
        median_delta > 0.0
        and positive_count >= 16
        and median_h1_drop > 0.0
    ):
        return "M-A"
    if median_delta > 0.0:
        return "M-B"
    return "M-C"



def _validated_pair_set(
    dataset: MechanismDataset,
    pair_set: object,
) -> HistoryPairSet:
    if not isinstance(pair_set, HistoryPairSet):
        raise ValueError("history pair set must be HistoryPairSet")
    expected = build_history_pairs(
        dataset.training,
        dataset.evaluation,
    )
    if pair_set != expected:
        raise ValueError(
            "history pair set does not match frozen dataset inputs"
        )
    return pair_set


def _probe_result(
    probe: MultiTargetRidgeProbe,
    targets: np.ndarray,
    predictions: np.ndarray,
) -> ProbeResult:
    return ProbeResult(
        metrics=evaluate_regression(targets, predictions),
        coefficient_digest=probe.coefficient_digest(),
        prediction_digest=regression_prediction_digest(predictions),
    )


def _sample_matrix(
    samples: tuple[object, ...],
) -> tuple[np.ndarray, tuple[str, ...]]:
    if not samples:
        raise ValueError("sample collection must not be empty")
    tensors = np.stack(
        [getattr(sample, "tensor") for sample in samples],
        axis=0,
    )
    ids = tuple(str(getattr(sample, "window_id")) for sample in samples)
    return np.asarray(tensors, dtype=np.float64), ids


def _validated_spec(spec: object) -> E2ReservoirSpec:
    if not isinstance(spec, E2ReservoirSpec):
        raise ValueError("spec must be E2ReservoirSpec")
    if spec.architecture not in REGISTERED_ARCHITECTURES:
        raise ValueError("architecture is not registered for R1-E3M")
    if spec.seed not in REGISTERED_SEEDS:
        raise ValueError("seed is not registered for R1-E3M")
    if spec.input_size != 6:
        raise ValueError("R1-E3M reservoir input_size must be 6")
    return spec


def _validated_arm_values(
    values: object,
    name: str,
) -> tuple[float, ...]:
    if not isinstance(values, (tuple, list)) or len(values) != ARM_COUNT:
        raise ValueError(f"{name} must contain exactly 20 values")
    result = tuple(float(value) for value in values)
    if any(not math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _mean(values: object) -> float:
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("aggregate values must be finite and non-empty")
    return float(np.mean(array, dtype=np.float64))
