from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .r1_e1_probe import MulticlassRidgeProbe, fit_multiclass_ridge, prediction_digest
from .r1_e1_yolo_like import (
    BEHAVIOR_CLASSES,
    CORRUPTION_ARMS,
    BehaviorSequence,
    CleanFixtureSet,
    CorruptionPlan,
    build_clean_fixture_set,
    build_corruption_plan,
    corrupt_sequence,
)
from .r1_e2_reservoir import E2ReservoirSpec, build_e2_reservoir


E2_EPISODE_SEEDS = (7, 17, 29, 43, 61)
EPISODE_CLASSES = BEHAVIOR_CLASSES
EPISODE_CORRUPTIONS = CORRUPTION_ARMS
EPISODE_TEMPORAL_WINDOW = 4
_TRAIN_COUNT = 800
_EVAL_COUNT = 200
_CLASS_COUNT = len(EPISODE_CLASSES)
_RIDGE_REGULARIZATION = 1e-6


@dataclass(frozen=True)
class EpisodeMetrics:
    correct: int
    total: int
    accuracy: float
    macro_f1: float
    confusion_counts: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class EpisodeBaselineResult:
    arm_metrics: tuple[tuple[str, EpisodeMetrics], ...]
    coefficient_digest: str
    prediction_digests: tuple[tuple[str, str], ...]
    macro_corrupted_accuracy: float
    worst_corrupted_accuracy: float
    reset_correct: int
    reset_total: int


@dataclass(frozen=True)
class YoloEpisodeSetResult:
    seed: int
    architecture: int
    frame_only: EpisodeBaselineResult
    reservoir_instantaneous: EpisodeBaselineResult
    reservoir_temporal_mean: EpisodeBaselineResult
    reset_groups_equal: bool
    train_fixture_digest: str
    evaluation_fixture_digest: str
    corruption_digest: str
    reservoir_parameter_digest: str


def build_e2_episode_fixture_set(seed: int, *, training: bool) -> CleanFixtureSet:
    return build_clean_fixture_set(_validated_registered_seed(seed), training=training)


def build_e2_episode_corruption_plan(
    seed: int,
    fixtures: CleanFixtureSet,
) -> CorruptionPlan:
    return build_corruption_plan(_validated_registered_seed(seed), fixtures)


def evaluate_yolo_episode_set(
    spec: E2ReservoirSpec,
    training: CleanFixtureSet,
    evaluation: CleanFixtureSet,
    plan: CorruptionPlan,
) -> YoloEpisodeSetResult:
    _validated_episode_spec(spec)
    if not isinstance(training, CleanFixtureSet):
        raise ValueError("training must be CleanFixtureSet")
    if not isinstance(evaluation, CleanFixtureSet):
        raise ValueError("evaluation must be CleanFixtureSet")
    if not isinstance(plan, CorruptionPlan):
        raise ValueError("plan must be CorruptionPlan")
    if len(plan.entries) != len(evaluation.sequences):
        raise ValueError("corruption plan count must match evaluation fixtures")

    train_labels = _labels(training.sequences)
    train_frame_only = np.vstack(
        [sequence.frames[19] for sequence in training.sequences]
    )
    frame_probe = _fit_probe(train_frame_only, train_labels)

    reservoir = build_e2_reservoir(spec)
    parameter_digest = reservoir.parameter_digest()
    train_trajectories, reservoir_train_labels = _collect_clean_trajectories(
        reservoir,
        training.sequences,
    )
    if reservoir.parameter_digest() != parameter_digest:
        raise RuntimeError("R1-E2 episode reservoir parameters changed during training")

    instant_probe = _fit_probe(train_trajectories[:, -1, :], reservoir_train_labels)
    temporal_probe = _fit_probe(
        np.mean(
            train_trajectories[:, -EPISODE_TEMPORAL_WINDOW:, :],
            axis=1,
            dtype=np.float64,
        ),
        reservoir_train_labels,
    )

    frame_scores: list[tuple[str, EpisodeMetrics]] = []
    frame_digests: list[tuple[str, str]] = []
    instant_scores: list[tuple[str, EpisodeMetrics]] = []
    instant_digests: list[tuple[str, str]] = []
    temporal_scores: list[tuple[str, EpisodeMetrics]] = []
    temporal_digests: list[tuple[str, str]] = []

    for arm in EPISODE_CORRUPTIONS:
        arm_frames, labels = _build_arm_frames(evaluation, plan, arm)
        frame_predictions = frame_probe.predict(arm_frames[:, 19, :])
        frame_scores.append((arm, _classification_metrics(labels, frame_predictions)))
        frame_digests.append((arm, prediction_digest(frame_predictions)))

        trajectories = _collect_frame_trajectories(reservoir, arm_frames)
        if reservoir.parameter_digest() != parameter_digest:
            raise RuntimeError("R1-E2 episode reservoir parameters changed during evaluation")

        instant_predictions = instant_probe.predict(trajectories[:, -1, :])
        temporal_representation = np.mean(
            trajectories[:, -EPISODE_TEMPORAL_WINDOW:, :],
            axis=1,
            dtype=np.float64,
        )
        temporal_predictions = temporal_probe.predict(temporal_representation)

        instant_scores.append(
            (arm, _classification_metrics(labels, instant_predictions))
        )
        instant_digests.append((arm, prediction_digest(instant_predictions)))
        temporal_scores.append(
            (arm, _classification_metrics(labels, temporal_predictions))
        )
        temporal_digests.append((arm, prediction_digest(temporal_predictions)))

    reset_trajectories, reset_labels = _collect_reset_suffix_trajectories(
        reservoir,
        evaluation.sequences,
    )
    reset_groups_equal = _reset_groups_equal(reset_trajectories, evaluation.sequences)
    if not reset_groups_equal:
        raise RuntimeError(
            "R1-E2 episode reset suffix trajectories differ within a paired group"
        )

    reset_instant_predictions = instant_probe.predict(reset_trajectories[:, -1, :])
    reset_temporal_predictions = temporal_probe.predict(
        np.mean(reset_trajectories, axis=1, dtype=np.float64)
    )
    reset_total = int(reset_labels.size)
    reset_instant_correct = int(
        np.count_nonzero(reset_instant_predictions == reset_labels)
    )
    reset_temporal_correct = int(
        np.count_nonzero(reset_temporal_predictions == reset_labels)
    )
    if (
        reset_instant_correct * _CLASS_COUNT != reset_total
        or reset_temporal_correct * _CLASS_COUNT != reset_total
    ):
        raise RuntimeError("R1-E2 episode reset control is not exact chance")

    return YoloEpisodeSetResult(
        seed=spec.seed,
        architecture=int(spec.architecture),
        frame_only=_baseline_result(
            frame_scores,
            frame_probe,
            frame_digests,
            reset_correct=0,
            reset_total=0,
        ),
        reservoir_instantaneous=_baseline_result(
            instant_scores,
            instant_probe,
            instant_digests,
            reset_correct=reset_instant_correct,
            reset_total=reset_total,
        ),
        reservoir_temporal_mean=_baseline_result(
            temporal_scores,
            temporal_probe,
            temporal_digests,
            reset_correct=reset_temporal_correct,
            reset_total=reset_total,
        ),
        reset_groups_equal=True,
        train_fixture_digest=training.fixture_digest,
        evaluation_fixture_digest=evaluation.fixture_digest,
        corruption_digest=plan.digest,
        reservoir_parameter_digest=parameter_digest,
    )


def run_yolo_episode_arm(
    spec: E2ReservoirSpec,
    training: CleanFixtureSet,
    evaluation: CleanFixtureSet,
    plan: CorruptionPlan,
) -> YoloEpisodeSetResult:
    _validated_episode_spec(spec)
    if len(training.sequences) != _TRAIN_COUNT:
        raise ValueError("registered training count mismatch")
    if len(evaluation.sequences) != _EVAL_COUNT:
        raise ValueError("registered evaluation count mismatch")
    if len(plan.entries) != _EVAL_COUNT:
        raise ValueError("registered corruption plan count mismatch")

    expected_training = build_e2_episode_fixture_set(spec.seed, training=True)
    expected_evaluation = build_e2_episode_fixture_set(spec.seed, training=False)
    if training.fixture_digest != expected_training.fixture_digest:
        raise ValueError("training fixture seed lineage must match reservoir seed")
    if evaluation.fixture_digest != expected_evaluation.fixture_digest:
        raise ValueError("evaluation fixture seed lineage must match reservoir seed")
    expected_plan = build_e2_episode_corruption_plan(spec.seed, expected_evaluation)
    if plan.digest != expected_plan.digest:
        raise ValueError("corruption plan seed lineage must match reservoir seed")

    return evaluate_yolo_episode_set(spec, training, evaluation, plan)


def _fit_probe(states: np.ndarray, labels: np.ndarray) -> MulticlassRidgeProbe:
    return fit_multiclass_ridge(
        states,
        labels,
        class_count=_CLASS_COUNT,
        regularization=_RIDGE_REGULARIZATION,
    )


def _collect_clean_trajectories(
    reservoir: object,
    sequences: tuple[BehaviorSequence, ...],
) -> tuple[np.ndarray, np.ndarray]:
    frames = np.stack([sequence.frames for sequence in sequences], axis=0)
    return _collect_frame_trajectories(reservoir, frames), _labels(sequences)


def _collect_frame_trajectories(
    reservoir: object,
    frames: np.ndarray,
) -> np.ndarray:
    values = np.asarray(frames, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (20, 9):
        raise ValueError("episode frames must have shape (samples, 20, 9)")
    trajectories = []
    for episode in values:
        reservoir.reset()
        states = [reservoir.advance(frame) for frame in episode]
        trajectories.append(np.stack(states, axis=0))
    return np.stack(trajectories, axis=0)


def _build_arm_frames(
    fixtures: CleanFixtureSet,
    plan: CorruptionPlan,
    arm: str,
) -> tuple[np.ndarray, np.ndarray]:
    if arm not in EPISODE_CORRUPTIONS:
        raise ValueError("arm must be a registered episode corruption")
    if len(fixtures.sequences) != len(plan.entries):
        raise ValueError("corruption plan count must match evaluation fixtures")
    grouped = _paired_sequences(fixtures.sequences)
    frames = []
    labels = []
    for sequence, entry in zip(fixtures.sequences, plan.entries, strict=True):
        frames.append(
            corrupt_sequence(
                sequence,
                grouped[sequence.pair_index],
                entry,
                arm,
            )
        )
        labels.append(sequence.label)
    return (
        np.stack(frames, axis=0),
        np.asarray(labels, dtype=np.int64),
    )


def _collect_reset_suffix_trajectories(
    reservoir: object,
    sequences: tuple[BehaviorSequence, ...],
) -> tuple[np.ndarray, np.ndarray]:
    trajectories = []
    labels = []
    for sequence in sequences:
        reservoir.reset()
        for frame in sequence.frames[:16]:
            reservoir.advance(frame)
        reservoir.reset()
        suffix_states = [reservoir.advance(frame) for frame in sequence.frames[16:20]]
        trajectories.append(np.stack(suffix_states, axis=0))
        labels.append(sequence.label)
    return (
        np.stack(trajectories, axis=0),
        np.asarray(labels, dtype=np.int64),
    )


def _paired_sequences(
    sequences: tuple[BehaviorSequence, ...],
) -> dict[int, dict[int, BehaviorSequence]]:
    groups: dict[int, dict[int, BehaviorSequence]] = {}
    for sequence in sequences:
        groups.setdefault(sequence.pair_index, {})[sequence.label] = sequence
    if any(set(group) != set(range(_CLASS_COUNT)) for group in groups.values()):
        raise ValueError("paired fixture group must contain every episode class")
    return groups


def _reset_groups_equal(
    trajectories: np.ndarray,
    sequences: tuple[BehaviorSequence, ...],
) -> bool:
    grouped: dict[int, list[int]] = {}
    for index, sequence in enumerate(sequences):
        grouped.setdefault(sequence.pair_index, []).append(index)
    for indices in grouped.values():
        if len(indices) != _CLASS_COUNT:
            return False
        group = trajectories[np.asarray(indices, dtype=np.int64)]
        if not np.array_equal(
            group,
            np.repeat(group[:1], _CLASS_COUNT, axis=0),
        ):
            return False
    return True


def _classification_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
) -> EpisodeMetrics:
    expected = np.asarray(labels, dtype=np.int64)
    observed = np.asarray(predictions, dtype=np.int64)
    if expected.ndim != 1 or observed.ndim != 1 or expected.shape != observed.shape:
        raise ValueError("labels and predictions must be matching rank-one arrays")
    if expected.size == 0:
        raise ValueError("labels and predictions must not be empty")
    if (
        np.any(expected < 0)
        or np.any(expected >= _CLASS_COUNT)
        or np.any(observed < 0)
        or np.any(observed >= _CLASS_COUNT)
    ):
        raise ValueError("labels or predictions contain an out-of-range class index")

    confusion = np.zeros((_CLASS_COUNT, _CLASS_COUNT), dtype=np.int64)
    for label, prediction in zip(expected, observed, strict=True):
        confusion[int(label), int(prediction)] += 1

    f1_scores = []
    for class_index in range(_CLASS_COUNT):
        true_positive = int(confusion[class_index, class_index])
        false_positive = int(np.sum(confusion[:, class_index])) - true_positive
        false_negative = int(np.sum(confusion[class_index, :])) - true_positive
        denominator = 2 * true_positive + false_positive + false_negative
        f1_scores.append(
            0.0 if denominator == 0 else (2.0 * true_positive) / denominator
        )

    correct = int(np.count_nonzero(expected == observed))
    total = int(expected.size)
    accuracy = correct / total
    macro_f1 = float(np.mean(np.asarray(f1_scores, dtype=np.float64)))
    if not math.isfinite(macro_f1):
        raise RuntimeError("R1-E2 episode macro-F1 is non-finite")
    return EpisodeMetrics(
        correct=correct,
        total=total,
        accuracy=accuracy,
        macro_f1=macro_f1,
        confusion_counts=tuple(
            tuple(int(value) for value in row)
            for row in confusion
        ),
    )


def _baseline_result(
    arm_scores: list[tuple[str, EpisodeMetrics]],
    probe: MulticlassRidgeProbe,
    digests: list[tuple[str, str]],
    *,
    reset_correct: int,
    reset_total: int,
) -> EpisodeBaselineResult:
    corrupted = [score.accuracy for name, score in arm_scores if name != "clean"]
    macro = float(np.mean(np.asarray(corrupted, dtype=np.float64)))
    worst = float(np.min(np.asarray(corrupted, dtype=np.float64)))
    if not math.isfinite(macro) or not math.isfinite(worst):
        raise RuntimeError("R1-E2 episode robustness metric is non-finite")
    return EpisodeBaselineResult(
        arm_metrics=tuple(arm_scores),
        coefficient_digest=probe.coefficient_digest(),
        prediction_digests=tuple(digests),
        macro_corrupted_accuracy=macro,
        worst_corrupted_accuracy=worst,
        reset_correct=reset_correct,
        reset_total=reset_total,
    )


def _labels(sequences: tuple[BehaviorSequence, ...]) -> np.ndarray:
    labels = np.asarray([sequence.label for sequence in sequences], dtype=np.int64)
    if labels.size == 0 or set(labels.tolist()) != set(range(_CLASS_COUNT)):
        raise ValueError("episode fixtures must contain every behavior class")
    return labels


def _validated_episode_spec(spec: object) -> E2ReservoirSpec:
    if not isinstance(spec, E2ReservoirSpec):
        raise ValueError("spec must be E2ReservoirSpec")
    if spec.input_size != 9:
        raise ValueError("R1-E2 episode input_size must be nine")
    _validated_registered_seed(spec.seed)
    return spec


def _validated_registered_seed(seed: object) -> int:
    if type(seed) is not int or seed not in E2_EPISODE_SEEDS:
        raise ValueError("seed must be one of the registered R1-E2 episode seeds")
    return seed
