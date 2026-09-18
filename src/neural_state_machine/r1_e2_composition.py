from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np


COMPOSITION_CLASSES = ("ABC", "ACB", "BAC", "BCA", "CAB", "CBA")
COMPOSITION_HISTORIES = (5, 10, 20, 40)
E2_COMPOSITION_SEEDS = (7, 17, 29, 43, 61)
TRAIN_GROUPS_PER_HISTORY = 100
EVAL_GROUPS_PER_HISTORY = 25

_TRAIN_TAG = 0x45324254
_EVAL_TAG = 0x45324245
_FIXTURE_DIGEST_VERSION = b"r1-e2-composition-fixture-v1\0"


@dataclass(frozen=True)
class CompositionSequence:
    label: int
    history: int
    group_index: int
    frames: np.ndarray

    def __post_init__(self) -> None:
        if type(self.label) is not int or self.label not in range(len(COMPOSITION_CLASSES)):
            raise ValueError("label must be a registered composition class index")
        if self.history not in COMPOSITION_HISTORIES:
            raise ValueError("history must be registered for R1-E2 composition")
        if type(self.group_index) is not int or self.group_index < 0:
            raise ValueError("group_index must be a non-negative Python integer")
        values = _validated_frames(self.frames, self.history)
        object.__setattr__(self, "frames", _readonly_copy(values))


@dataclass(frozen=True)
class CompositionFixtureSet:
    history: int
    sequences: tuple[CompositionSequence, ...]
    fixture_digest: str

    def __post_init__(self) -> None:
        if self.history not in COMPOSITION_HISTORIES:
            raise ValueError("history must be registered for R1-E2 composition")
        if type(self.sequences) is not tuple or not self.sequences:
            raise ValueError("sequences must be a non-empty tuple")
        if any(sequence.history != self.history for sequence in self.sequences):
            raise ValueError("all sequences must match the fixture history")
        if type(self.fixture_digest) is not str or len(self.fixture_digest) != 64:
            raise ValueError("fixture_digest must be a SHA-256 hex digest")


@dataclass(frozen=True)
class ClassificationMetrics:
    correct: int
    total: int
    accuracy: float
    macro_f1: float
    confusion_counts: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class RepresentationGeometry:
    centroid_distances: tuple[float, ...]
    within_class_dispersion: tuple[float, ...]
    separation_ratio: float


@dataclass(frozen=True)
class CompositionReadoutResult:
    metrics: ClassificationMetrics
    geometry: RepresentationGeometry
    reset_correct: int
    reset_total: int
    coefficient_digest: str
    prediction_digest: str
    reset_prediction_digest: str


@dataclass(frozen=True)
class CompositionHistoryResult:
    history: int
    instantaneous: CompositionReadoutResult
    temporal_mean: CompositionReadoutResult
    reset_groups_equal: bool
    train_fixture_digest: str
    evaluation_fixture_digest: str
    reservoir_parameter_digest: str


@dataclass(frozen=True)
class CompositionArmResult:
    seed: int
    architecture: int
    histories: tuple[CompositionHistoryResult, ...]
    instantaneous_macro_accuracy: float
    temporal_mean_macro_accuracy: float


def build_composition_fixture_sets(
    seed: int,
    *,
    training: bool,
) -> tuple[CompositionFixtureSet, ...]:
    registered_seed = _validated_registered_seed(seed)
    groups = TRAIN_GROUPS_PER_HISTORY if training else EVAL_GROUPS_PER_HISTORY
    tag = _TRAIN_TAG if training else _EVAL_TAG
    rng = np.random.default_rng(np.random.SeedSequence([registered_seed, tag]))
    return tuple(
        _build_history_fixture(rng, history=history, group_count=groups)
        for history in COMPOSITION_HISTORIES
    )


def classification_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
) -> ClassificationMetrics:
    expected = _validated_labels(labels, "labels")
    observed = _validated_labels(predictions, "predictions")
    if expected.shape != observed.shape:
        raise ValueError("labels and predictions must have matching shapes")

    class_count = len(COMPOSITION_CLASSES)
    confusion = np.zeros((class_count, class_count), dtype=np.int64)
    for label, prediction in zip(expected, observed, strict=True):
        confusion[int(label), int(prediction)] += 1

    f1_scores = []
    for class_index in range(class_count):
        true_positive = int(confusion[class_index, class_index])
        false_positive = int(np.sum(confusion[:, class_index])) - true_positive
        false_negative = int(np.sum(confusion[class_index, :])) - true_positive
        denominator = 2 * true_positive + false_positive + false_negative
        f1_scores.append(0.0 if denominator == 0 else (2.0 * true_positive) / denominator)

    correct = int(np.count_nonzero(expected == observed))
    total = int(expected.size)
    accuracy = correct / total
    macro_f1 = float(np.mean(np.asarray(f1_scores, dtype=np.float64)))
    if not math.isfinite(macro_f1):
        raise RuntimeError("composition macro-F1 is non-finite")

    return ClassificationMetrics(
        correct=correct,
        total=total,
        accuracy=accuracy,
        macro_f1=macro_f1,
        confusion_counts=tuple(
            tuple(int(value) for value in row)
            for row in confusion
        ),
    )


def representation_geometry(
    representations: np.ndarray,
    labels: np.ndarray,
) -> RepresentationGeometry:
    values = _validated_representations(representations)
    class_labels = _validated_labels(labels, "labels")
    if values.shape[0] != class_labels.shape[0]:
        raise ValueError("representations and labels must match sample count")

    centroids = []
    dispersions = []
    for class_index in range(len(COMPOSITION_CLASSES)):
        members = values[class_labels == class_index]
        if members.shape[0] == 0:
            raise ValueError("labels must contain every composition class")
        centroid = np.mean(members, axis=0, dtype=np.float64)
        centroids.append(centroid)
        distances = np.linalg.norm(members - centroid, axis=1)
        dispersions.append(float(np.mean(distances, dtype=np.float64)))

    centroid_distances = []
    for left in range(len(centroids)):
        for right in range(left + 1, len(centroids)):
            centroid_distances.append(
                float(np.linalg.norm(centroids[left] - centroids[right]))
            )

    mean_between = float(np.mean(centroid_distances, dtype=np.float64))
    mean_within = float(np.mean(dispersions, dtype=np.float64))
    denominator = max(mean_within, float(np.finfo(np.float64).eps))
    ratio = mean_between / denominator

    all_values = (*centroid_distances, *dispersions, ratio)
    if any(not math.isfinite(value) or value < 0.0 for value in all_values):
        raise RuntimeError("composition geometry produced an invalid metric")

    return RepresentationGeometry(
        centroid_distances=tuple(centroid_distances),
        within_class_dispersion=tuple(dispersions),
        separation_ratio=ratio,
    )



def evaluate_composition_history(
    spec: E2ReservoirSpec,
    training: CompositionFixtureSet,
    evaluation: CompositionFixtureSet,
) -> CompositionHistoryResult:
    _validated_composition_spec(spec)
    if not isinstance(training, CompositionFixtureSet) or not isinstance(
        evaluation, CompositionFixtureSet
    ):
        raise ValueError("training and evaluation must be CompositionFixtureSet")
    if training.history != evaluation.history:
        raise ValueError("training and evaluation history must match")
    if training.seed is not None and training.seed != spec.seed:
        raise ValueError("training fixture seed must match reservoir seed")
    if evaluation.seed is not None and evaluation.seed != spec.seed:
        raise ValueError("evaluation fixture seed must match reservoir seed")

    reservoir = build_e2_reservoir(spec)
    parameter_digest = reservoir.parameter_digest()

    train_trajectories, train_labels = _collect_trajectories(
        reservoir, training.sequences
    )
    evaluation_trajectories, evaluation_labels = _collect_trajectories(
        reservoir, evaluation.sequences
    )
    reset_trajectories, reset_labels = _collect_reset_tail_trajectories(
        reservoir, evaluation.sequences
    )
    if reservoir.parameter_digest() != parameter_digest:
        raise RuntimeError("R1-E2 composition reservoir parameters changed")

    history = training.history
    train_final = train_trajectories[:, -1, :]
    evaluation_final = evaluation_trajectories[:, -1, :]
    reset_final = reset_trajectories[:, -1, :]

    train_temporal = causal_mean_pool_batch(train_trajectories, window=history)
    evaluation_temporal = causal_mean_pool_batch(
        evaluation_trajectories, window=history
    )
    reset_temporal = causal_mean_pool_batch(reset_trajectories, window=history)

    instant_probe = fit_instant_multiclass(
        train_final,
        train_labels,
        class_count=len(COMPOSITION_CLASSES),
    )
    temporal_probe = fit_temporal_mean_multiclass(
        train_trajectories,
        train_labels,
        class_count=len(COMPOSITION_CLASSES),
        window=history,
    )

    instant_predictions = instant_probe.predict(evaluation_final)
    temporal_predictions = temporal_probe.predict(evaluation_temporal)
    instant_reset_predictions = instant_probe.predict(reset_final)
    temporal_reset_predictions = temporal_probe.predict(reset_temporal)

    reset_groups_equal = _reset_groups_equal(
        reset_trajectories, evaluation.sequences
    )
    if not reset_groups_equal:
        raise RuntimeError(
            "R1-E2 composition reset trajectories differ within a nuisance group"
        )

    instant_reset_correct = int(
        np.count_nonzero(instant_reset_predictions == reset_labels)
    )
    temporal_reset_correct = int(
        np.count_nonzero(temporal_reset_predictions == reset_labels)
    )
    reset_total = int(reset_labels.size)
    if (
        instant_reset_correct * len(COMPOSITION_CLASSES) != reset_total
        or temporal_reset_correct * len(COMPOSITION_CLASSES) != reset_total
    ):
        raise RuntimeError("R1-E2 composition reset control is not exact chance")

    return CompositionHistoryResult(
        history=history,
        instantaneous=CompositionReadoutResult(
            metrics=classification_metrics(evaluation_labels, instant_predictions),
            geometry=representation_geometry(evaluation_final, evaluation_labels),
            reset_correct=instant_reset_correct,
            reset_total=reset_total,
            coefficient_digest=instant_probe.coefficient_digest(),
            prediction_digest=prediction_digest(instant_predictions),
            reset_prediction_digest=prediction_digest(instant_reset_predictions),
        ),
        temporal_mean=CompositionReadoutResult(
            metrics=classification_metrics(evaluation_labels, temporal_predictions),
            geometry=representation_geometry(evaluation_temporal, evaluation_labels),
            reset_correct=temporal_reset_correct,
            reset_total=reset_total,
            coefficient_digest=temporal_probe.coefficient_digest(),
            prediction_digest=prediction_digest(temporal_predictions),
            reset_prediction_digest=prediction_digest(temporal_reset_predictions),
        ),
        reset_groups_equal=True,
        train_fixture_digest=training.fixture_digest,
        evaluation_fixture_digest=evaluation.fixture_digest,
        reservoir_parameter_digest=parameter_digest,
    )


def run_composition_arm(
    spec: E2ReservoirSpec,
    training_sets: tuple[CompositionFixtureSet, ...],
    evaluation_sets: tuple[CompositionFixtureSet, ...],
) -> CompositionArmResult:
    _validated_composition_spec(spec)
    _validate_registered_fixture_sets(
        spec,
        training_sets,
        training=True,
    )
    _validate_registered_fixture_sets(
        spec,
        evaluation_sets,
        training=False,
    )

    results = tuple(
        evaluate_composition_history(spec, training, evaluation)
        for training, evaluation in zip(
            training_sets, evaluation_sets, strict=True
        )
    )
    return CompositionArmResult(
        seed=spec.seed,
        architecture=int(spec.architecture),
        histories=results,
        instantaneous_macro_accuracy=float(
            np.mean(
                [result.instantaneous.metrics.accuracy for result in results],
                dtype=np.float64,
            )
        ),
        temporal_mean_macro_accuracy=float(
            np.mean(
                [result.temporal_mean.metrics.accuracy for result in results],
                dtype=np.float64,
            )
        ),
    )


def _validated_composition_spec(spec: object) -> E2ReservoirSpec:
    if not isinstance(spec, E2ReservoirSpec):
        raise ValueError("spec must be E2ReservoirSpec")
    if spec.input_size != 7:
        raise ValueError("R1-E2 composition input_size must be seven")
    _validated_registered_seed(spec.seed)
    return spec


def _validate_registered_fixture_sets(
    spec: E2ReservoirSpec,
    fixture_sets: object,
    *,
    training: bool,
) -> None:
    if type(fixture_sets) is not tuple or len(fixture_sets) != len(
        COMPOSITION_HISTORIES
    ):
        raise ValueError("fixture sets must cover all registered histories")
    expected_groups = (
        TRAIN_GROUPS_PER_HISTORY if training else EVAL_GROUPS_PER_HISTORY
    )
    expected_count = expected_groups * len(COMPOSITION_CLASSES)
    kind = "training" if training else "evaluation"
    for expected_history, fixture_set in zip(
        COMPOSITION_HISTORIES, fixture_sets, strict=True
    ):
        if not isinstance(fixture_set, CompositionFixtureSet):
            raise ValueError(f"{kind} fixtures must be CompositionFixtureSet")
        if fixture_set.history != expected_history:
            raise ValueError(f"{kind} fixtures must use registered histories")
        if len(fixture_set.sequences) != expected_count:
            raise ValueError(f"registered {kind} count mismatch")
        if fixture_set.seed != spec.seed:
            raise ValueError(f"{kind} fixture seed must match reservoir seed")


def _collect_trajectories(
    reservoir: object,
    sequences: tuple[CompositionSequence, ...],
) -> tuple[np.ndarray, np.ndarray]:
    trajectories = []
    labels = []
    for sequence in sequences:
        reservoir.reset()
        states = [reservoir.advance(frame) for frame in sequence.frames]
        trajectories.append(np.stack(states, axis=0))
        labels.append(sequence.label)
    return (
        np.stack(trajectories, axis=0),
        np.asarray(labels, dtype=np.int64),
    )


def _collect_reset_tail_trajectories(
    reservoir: object,
    sequences: tuple[CompositionSequence, ...],
) -> tuple[np.ndarray, np.ndarray]:
    trajectories = []
    labels = []
    for sequence in sequences:
        reservoir.reset()
        for frame in sequence.frames[:5]:
            reservoir.advance(frame)
        reservoir.reset()
        states = [reservoir.advance(frame) for frame in sequence.frames[5:]]
        if len(states) != sequence.history:
            raise RuntimeError("reset tail length does not match registered history")
        trajectories.append(np.stack(states, axis=0))
        labels.append(sequence.label)
    return (
        np.stack(trajectories, axis=0),
        np.asarray(labels, dtype=np.int64),
    )


def _reset_groups_equal(
    trajectories: np.ndarray,
    sequences: tuple[CompositionSequence, ...],
) -> bool:
    if trajectories.shape[0] != len(sequences):
        return False
    grouped: dict[int, list[int]] = {}
    for index, sequence in enumerate(sequences):
        grouped.setdefault(sequence.group_index, []).append(index)
    for indices in grouped.values():
        if len(indices) != len(COMPOSITION_CLASSES):
            return False
        group = trajectories[np.asarray(indices, dtype=np.int64)]
        if not np.array_equal(
            group,
            np.repeat(group[:1], len(COMPOSITION_CLASSES), axis=0),
        ):
            return False
    return True


def _build_history_fixture(
    rng: np.random.Generator,
    *,
    history: int,
    group_count: int,
) -> CompositionFixtureSet:
    sequences = []
    event_index = {"A": 0, "B": 1, "C": 2}
    sequence_length = 5 + history

    for group_index in range(group_count):
        nuisance = rng.choice(
            np.array([-0.25, 0.25], dtype=np.float64),
            size=(sequence_length, 4),
            replace=True,
        )
        for label, order in enumerate(COMPOSITION_CLASSES):
            events = np.zeros((sequence_length, 3), dtype=np.float64)
            for frame_index, symbol in zip((0, 2, 4), order, strict=True):
                events[frame_index, event_index[symbol]] = 1.0
            frames = np.column_stack((events, nuisance))
            sequences.append(
                CompositionSequence(
                    label=label,
                    history=history,
                    group_index=group_index,
                    frames=frames,
                )
            )

    frozen = tuple(sequences)
    return CompositionFixtureSet(
        history=history,
        sequences=frozen,
        fixture_digest=_fixture_digest(frozen),
    )


def _fixture_digest(sequences: tuple[CompositionSequence, ...]) -> str:
    digest = hashlib.sha256()
    digest.update(_FIXTURE_DIGEST_VERSION)
    for sequence in sequences:
        digest.update(
            np.asarray(
                [sequence.label, sequence.history, sequence.group_index],
                dtype=np.int64,
            ).tobytes()
        )
        frames = np.ascontiguousarray(sequence.frames, dtype=np.float64)
        digest.update(str(frames.shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(frames.tobytes(order="C"))
    return digest.hexdigest()


def _validated_registered_seed(seed: object) -> int:
    if type(seed) is not int or seed not in E2_COMPOSITION_SEEDS:
        raise ValueError("seed must be one of the registered R1-E2 seeds")
    return seed


def _validated_frames(frames: object, history: int) -> np.ndarray:
    try:
        values = np.asarray(frames, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("frames must be float64-compatible") from exc
    if values.shape != (5 + history, 7):
        raise ValueError("frames have the wrong R1-E2 composition shape")
    if not np.all(np.isfinite(values)):
        raise ValueError("frames must contain only finite values")
    return values


def _validated_labels(labels: object, name: str) -> np.ndarray:
    raw = np.asarray(labels)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError(f"{name} must be a non-empty rank-one array")
    if raw.dtype.kind not in "iu" or raw.dtype.kind == "b":
        raise ValueError(f"{name} must contain integer class indices")
    values = np.asarray(raw, dtype=np.int64)
    if np.any(values < 0) or np.any(values >= len(COMPOSITION_CLASSES)):
        raise ValueError(f"{name} contains an out-of-range class index")
    return values


def _validated_representations(representations: object) -> np.ndarray:
    try:
        values = np.asarray(representations, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("representations must be float64-compatible") from exc
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("representations must contain samples and features")
    if not np.all(np.isfinite(values)):
        raise ValueError("representations must contain only finite values")
    return values


def _readonly_copy(values: np.ndarray) -> np.ndarray:
    copied = np.array(values, dtype=np.float64, copy=True, order="C")
    copied.flags.writeable = False
    return copied
