from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np

from .r1_e1_probe import prediction_digest
from .r1_e2_reservoir import E2ReservoirSpec, build_e2_reservoir
from .r1_e3_dataset import RegisteredDataset, RegisteredEpisode
from .r1_e3_readout import (
    fit_frame_only,
    fit_reservoir_instantaneous,
    fit_reservoir_temporal_mean,
    temporal_mean_last_four,
)


REGISTERED_LABELS = ("approach", "touch", "pick_up", "pass_by")
_CLASS_COUNT = len(REGISTERED_LABELS)
_INPUT_SIZE = 14
_TRAJECTORY_DIGEST_VERSION = b"r1-e3-trajectory-v1\0"
_EPISODE_DIGEST_VERSION = b"r1-e3-episode-ids-v1\0"


@dataclass(frozen=True)
class ClassificationMetrics:
    correct: int
    total: int
    accuracy: float
    macro_f1: float
    precision_per_class: tuple[float, ...]
    recall_per_class: tuple[float, ...]
    f1_per_class: tuple[float, ...]
    confusion_counts: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class ReadoutResult:
    metrics: ClassificationMetrics
    coefficient_digest: str
    prediction_digest: str


@dataclass(frozen=True)
class HistoryDestructionResult:
    reset_before_bin: int
    suffix_bins: tuple[int, ...]
    instantaneous: ReadoutResult
    temporal_mean: ReadoutResult


@dataclass(frozen=True)
class RealTrackArmResult:
    seed: int
    architecture: int
    frame_only: ReadoutResult
    reservoir_instantaneous: ReadoutResult
    reservoir_temporal_mean: ReadoutResult
    history_destruction: HistoryDestructionResult
    reservoir_parameter_digest: str
    trajectory_digest_b1: str
    trajectory_digest_b2: str
    training_episode_digest: str
    evaluation_episode_digest: str
    artifact_root_digest: str
    delta_macro_f1: float


def evaluate_real_track_arm(
    spec: E2ReservoirSpec,
    dataset: RegisteredDataset,
) -> RealTrackArmResult:
    _validated_spec(spec)
    if not isinstance(dataset, RegisteredDataset):
        raise ValueError("dataset must be RegisteredDataset")

    training_observations, training_labels = _episode_matrix(dataset.training)
    evaluation_observations, evaluation_labels = _episode_matrix(dataset.evaluation)

    reservoir = build_e2_reservoir(spec)
    parameter_digest = reservoir.parameter_digest()

    training_trajectories = _collect_trajectories(
        reservoir,
        training_observations,
    )
    evaluation_trajectories = _collect_trajectories(
        reservoir,
        evaluation_observations,
    )
    if reservoir.parameter_digest() != parameter_digest:
        raise RuntimeError("R1-E3 reservoir parameters changed during trajectory collection")

    frame_probe = fit_frame_only(
        training_observations,
        training_labels,
        class_count=_CLASS_COUNT,
    )
    instantaneous_probe = fit_reservoir_instantaneous(
        training_trajectories,
        training_labels,
        class_count=_CLASS_COUNT,
    )
    temporal_probe = fit_reservoir_temporal_mean(
        training_trajectories,
        training_labels,
        class_count=_CLASS_COUNT,
    )
    if reservoir.parameter_digest() != parameter_digest:
        raise RuntimeError("R1-E3 reservoir parameters changed during probe fitting")

    frame_predictions = frame_probe.predict(evaluation_observations[:, 19, :])
    instantaneous_predictions = instantaneous_probe.predict(
        evaluation_trajectories[:, 19, :]
    )
    temporal_predictions = temporal_probe.predict(
        temporal_mean_last_four(evaluation_trajectories)
    )

    frame_result = _readout_result(
        evaluation_labels,
        frame_predictions,
        frame_probe.coefficient_digest(),
    )
    instantaneous_result = _readout_result(
        evaluation_labels,
        instantaneous_predictions,
        instantaneous_probe.coefficient_digest(),
    )
    temporal_result = _readout_result(
        evaluation_labels,
        temporal_predictions,
        temporal_probe.coefficient_digest(),
    )

    reset_trajectories = _collect_history_destruction_trajectories(
        reservoir,
        evaluation_observations,
    )
    if reservoir.parameter_digest() != parameter_digest:
        raise RuntimeError("R1-E3 reservoir parameters changed during reset control")

    reset_instantaneous_predictions = instantaneous_probe.predict(
        reset_trajectories[:, -1, :]
    )
    reset_temporal_predictions = temporal_probe.predict(
        np.mean(reset_trajectories, axis=1, dtype=np.float64)
    )

    reset_control = HistoryDestructionResult(
        reset_before_bin=16,
        suffix_bins=(16, 17, 18, 19),
        instantaneous=_readout_result(
            evaluation_labels,
            reset_instantaneous_predictions,
            instantaneous_probe.coefficient_digest(),
        ),
        temporal_mean=_readout_result(
            evaluation_labels,
            reset_temporal_predictions,
            temporal_probe.coefficient_digest(),
        ),
    )

    trajectory_digest = _trajectory_digest(evaluation_trajectories)
    return RealTrackArmResult(
        seed=spec.seed,
        architecture=int(spec.architecture),
        frame_only=frame_result,
        reservoir_instantaneous=instantaneous_result,
        reservoir_temporal_mean=temporal_result,
        history_destruction=reset_control,
        reservoir_parameter_digest=parameter_digest,
        trajectory_digest_b1=trajectory_digest,
        trajectory_digest_b2=trajectory_digest,
        training_episode_digest=_episode_digest(dataset.training),
        evaluation_episode_digest=_episode_digest(dataset.evaluation),
        artifact_root_digest=dataset.artifact_root_digest,
        delta_macro_f1=(
            temporal_result.metrics.macro_f1
            - instantaneous_result.metrics.macro_f1
        ),
    )


def _validated_spec(spec: object) -> E2ReservoirSpec:
    if not isinstance(spec, E2ReservoirSpec):
        raise ValueError("spec must be E2ReservoirSpec")
    if spec.input_size != _INPUT_SIZE:
        raise ValueError("R1-E3 reservoir input_size must be 14")
    return spec


def _episode_matrix(
    episodes: tuple[RegisteredEpisode, ...],
) -> tuple[np.ndarray, np.ndarray]:
    if not episodes:
        raise ValueError("episode collection must not be empty")
    observations = np.stack([episode.tensor for episode in episodes], axis=0)
    labels = np.asarray(
        [REGISTERED_LABELS.index(episode.label) for episode in episodes],
        dtype=np.int64,
    )
    if set(labels.tolist()) != set(range(_CLASS_COUNT)):
        raise ValueError("episode collection must contain every registered class")
    return observations, labels


def _collect_trajectories(
    reservoir: object,
    observations: np.ndarray,
) -> np.ndarray:
    trajectories = []
    for episode in observations:
        reservoir.reset()
        states = [reservoir.advance(row) for row in episode]
        trajectories.append(np.stack(states, axis=0))
    return np.stack(trajectories, axis=0)


def _collect_history_destruction_trajectories(
    reservoir: object,
    observations: np.ndarray,
) -> np.ndarray:
    trajectories = []
    for episode in observations:
        reservoir.reset()
        for row in episode[:16]:
            reservoir.advance(row)
        reservoir.reset()
        suffix_states = [reservoir.advance(row) for row in episode[16:20]]
        trajectories.append(np.stack(suffix_states, axis=0))
    return np.stack(trajectories, axis=0)


def _readout_result(
    labels: np.ndarray,
    predictions: np.ndarray,
    coefficient_digest: str,
) -> ReadoutResult:
    return ReadoutResult(
        metrics=_classification_metrics(labels, predictions),
        coefficient_digest=coefficient_digest,
        prediction_digest=prediction_digest(predictions),
    )


def _classification_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
) -> ClassificationMetrics:
    expected = np.asarray(labels, dtype=np.int64)
    observed = np.asarray(predictions, dtype=np.int64)
    if expected.ndim != 1 or observed.ndim != 1 or expected.shape != observed.shape:
        raise ValueError("labels and predictions must be matching rank-one arrays")
    if expected.size == 0:
        raise ValueError("labels and predictions must not be empty")

    confusion = np.zeros((_CLASS_COUNT, _CLASS_COUNT), dtype=np.int64)
    for label, prediction in zip(expected, observed, strict=True):
        if (
            int(label) not in range(_CLASS_COUNT)
            or int(prediction) not in range(_CLASS_COUNT)
        ):
            raise ValueError("label or prediction is outside registered classes")
        confusion[int(label), int(prediction)] += 1

    precision = []
    recall = []
    f1 = []
    for index in range(_CLASS_COUNT):
        true_positive = int(confusion[index, index])
        false_positive = int(np.sum(confusion[:, index])) - true_positive
        false_negative = int(np.sum(confusion[index, :])) - true_positive
        p = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        r = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        score = 2.0 * p * r / (p + r) if p + r else 0.0
        precision.append(p)
        recall.append(r)
        f1.append(score)

    correct = int(np.count_nonzero(expected == observed))
    total = int(expected.size)
    accuracy = correct / total
    macro_f1 = float(np.mean(np.asarray(f1, dtype=np.float64)))
    values = (accuracy, macro_f1, *precision, *recall, *f1)
    if any(not math.isfinite(value) for value in values):
        raise RuntimeError("R1-E3 classification metrics are non-finite")

    return ClassificationMetrics(
        correct=correct,
        total=total,
        accuracy=accuracy,
        macro_f1=macro_f1,
        precision_per_class=tuple(float(value) for value in precision),
        recall_per_class=tuple(float(value) for value in recall),
        f1_per_class=tuple(float(value) for value in f1),
        confusion_counts=tuple(
            tuple(int(value) for value in row)
            for row in confusion
        ),
    )


def _trajectory_digest(trajectories: np.ndarray) -> str:
    values = np.ascontiguousarray(trajectories, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(_TRAJECTORY_DIGEST_VERSION)
    digest.update(str(values.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _episode_digest(episodes: tuple[RegisteredEpisode, ...]) -> str:
    digest = hashlib.sha256()
    digest.update(_EPISODE_DIGEST_VERSION)
    for episode in episodes:
        digest.update(episode.episode_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(episode.video_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(episode.label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(episode.tensor_digest.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()
