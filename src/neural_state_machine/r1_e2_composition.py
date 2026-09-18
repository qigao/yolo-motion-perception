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
