from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np

from .r1_e1_probe import fit_multiclass_ridge, prediction_digest
from .r1_e1_reservoir import ReservoirSpec, build_reservoir


HISTORY_HORIZONS = (1, 5, 20, 40)
HISTORY_CLASSES = ((0, 1), (1, 0), (0, 0), (1, 1))  # AB, BA, AA, BB
_TRAIN_PAIRS = 200
_EVAL_PAIRS = 50
_TRAIN_TAG = 0x45314254
_EVAL_TAG = 0x45314245
_FIXTURE_DIGEST_VERSION = b"r1-e1-history-fixture-v1\0"


@dataclass(frozen=True)
class HistorySequence:
    label: int
    horizon: int
    pair_index: int
    frames: np.ndarray

    def __post_init__(self) -> None:
        if type(self.label) is not int or self.label not in range(4):
            raise ValueError("label must be an integer in [0, 3]")
        if self.horizon not in HISTORY_HORIZONS:
            raise ValueError("horizon must be registered")
        if type(self.pair_index) is not int or self.pair_index < 0:
            raise ValueError("pair_index must be a non-negative integer")
        values = np.asarray(self.frames, dtype=np.float64)
        if values.shape != (3 + self.horizon, 6):
            raise ValueError("frames have the wrong E1-B shape")
        if not np.all(np.isfinite(values)):
            raise ValueError("frames must be finite")
        copied = np.array(values, dtype=np.float64, copy=True, order="C")
        copied.flags.writeable = False
        object.__setattr__(self, "frames", copied)


@dataclass(frozen=True)
class HistoryFixtureSet:
    horizon: int
    sequences: tuple[HistorySequence, ...]
    fixture_digest: str


@dataclass(frozen=True)
class CountSummary:
    count: int
    minimum: float
    median: float
    mean: float
    maximum: float


@dataclass(frozen=True)
class HistoryHorizonResult:
    horizon: int
    correct: int
    total: int
    reset_correct: int
    reset_total: int
    reset_groups_equal: bool
    normalized_margin: CountSummary
    within_class_cosine_distance: CountSummary
    between_class_cosine_distance: CountSummary
    coefficient_digest: str
    prediction_digest: str
    reset_prediction_digest: str
    train_fixture_digest: str
    evaluation_fixture_digest: str
    reservoir_parameter_digest: str

    @property
    def accuracy(self) -> float:
        return self.correct / self.total


@dataclass(frozen=True)
class HistorySeparabilityResult:
    seed: int
    architecture: int
    budget: int
    horizons: tuple[HistoryHorizonResult, ...]
    macro_accuracy: float
    contiguous_80_horizon: int | None


def build_history_fixture_sets(seed: int, *, training: bool) -> tuple[HistoryFixtureSet, ...]:
    _validate_seed(seed)
    pair_count = _TRAIN_PAIRS if training else _EVAL_PAIRS
    tag = _TRAIN_TAG if training else _EVAL_TAG
    rng = np.random.default_rng(np.random.SeedSequence([seed, tag]))
    return tuple(_build_horizon_fixture(rng, horizon, pair_count) for horizon in HISTORY_HORIZONS)


def run_history_separability(spec: ReservoirSpec) -> HistorySeparabilityResult:
    if not isinstance(spec, ReservoirSpec):
        raise ValueError("spec must be a ReservoirSpec")
    if spec.input_size != 6:
        raise ValueError("E1-B requires reservoir input_size 6")

    training_sets = build_history_fixture_sets(spec.seed, training=True)
    evaluation_sets = build_history_fixture_sets(spec.seed, training=False)
    horizon_results = []
    for training, evaluation in zip(training_sets, evaluation_sets, strict=True):
        horizon_results.append(_run_horizon(spec, training, evaluation))

    macro_accuracy = float(np.mean([result.accuracy for result in horizon_results]))
    if not math.isfinite(macro_accuracy):
        raise RuntimeError("E1-B produced a non-finite macro accuracy")
    contiguous = None
    for result in horizon_results:
        if result.accuracy < 0.80:
            break
        contiguous = result.horizon
    return HistorySeparabilityResult(
        seed=spec.seed,
        architecture=int(spec.architecture),
        budget=spec.budget,
        horizons=tuple(horizon_results),
        macro_accuracy=macro_accuracy,
        contiguous_80_horizon=contiguous,
    )


def _build_horizon_fixture(
    rng: np.random.Generator,
    horizon: int,
    pair_count: int,
) -> HistoryFixtureSet:
    sequences = []
    for pair_index in range(pair_count):
        nuisance = rng.choice(
            np.array([-0.25, 0.25], dtype=np.float64),
            size=(3 + horizon, 4),
            replace=True,
        )
        for label, (first, second) in enumerate(HISTORY_CLASSES):
            events = np.zeros((3 + horizon, 2), dtype=np.float64)
            events[0, first] = 1.0
            events[2, second] = 1.0
            frames = np.column_stack((events, nuisance))
            sequences.append(HistorySequence(label, horizon, pair_index, frames))
    fixture_digest = _fixture_digest(tuple(sequences))
    return HistoryFixtureSet(horizon, tuple(sequences), fixture_digest)


def _run_horizon(
    spec: ReservoirSpec,
    training: HistoryFixtureSet,
    evaluation: HistoryFixtureSet,
) -> HistoryHorizonResult:
    if training.horizon != evaluation.horizon:
        raise RuntimeError("E1-B fixture horizons do not match")
    if len(training.sequences) != _TRAIN_PAIRS * 4:
        raise RuntimeError("E1-B training count mismatch")
    if len(evaluation.sequences) != _EVAL_PAIRS * 4:
        raise RuntimeError("E1-B evaluation count mismatch")
    if not training.fixture_digest or not evaluation.fixture_digest:
        raise RuntimeError("E1-B fixture digest missing")

    reservoir = build_reservoir(spec)
    parameter_digest = reservoir.parameter_digest()
    train_states, train_labels = _collect_terminal_states(reservoir, training.sequences)
    evaluation_states, evaluation_labels = _collect_terminal_states(
        reservoir, evaluation.sequences
    )
    reset_states, reset_labels = _collect_reset_terminal_states(
        reservoir, evaluation.sequences
    )
    if reservoir.state_dim != spec.budget:
        raise RuntimeError("E1-B state width mismatch")
    if reservoir.parameter_digest() != parameter_digest:
        raise RuntimeError("E1-B reservoir parameters changed during collection")

    probe = fit_multiclass_ridge(
        train_states,
        train_labels,
        class_count=4,
        regularization=1e-6,
    )
    predictions = probe.predict(evaluation_states)
    reset_predictions = probe.predict(reset_states)
    scores = probe.predict_scores(evaluation_states)
    correct = int(np.count_nonzero(predictions == evaluation_labels))
    reset_correct = int(np.count_nonzero(reset_predictions == reset_labels))
    groups_equal = _reset_groups_equal(reset_states, evaluation.sequences)
    if not groups_equal:
        raise RuntimeError("E1-B reset terminal states differ within a paired group")
    if reset_correct != 50 or reset_predictions.size != 200:
        raise RuntimeError("E1-B reset control must be exactly 50/200")
    if reservoir.parameter_digest() != parameter_digest:
        raise RuntimeError("E1-B reservoir parameters changed during probe fitting")

    margins = _normalized_margins(scores, evaluation_labels)
    within, between = _cosine_distance_groups(evaluation_states, evaluation_labels)
    summaries = (_summary(margins), _summary(within), _summary(between))
    if any(not _summary_is_finite(summary) for summary in summaries):
        raise RuntimeError("E1-B produced a non-finite metric")

    return HistoryHorizonResult(
        horizon=training.horizon,
        correct=correct,
        total=int(predictions.size),
        reset_correct=reset_correct,
        reset_total=int(reset_predictions.size),
        reset_groups_equal=groups_equal,
        normalized_margin=summaries[0],
        within_class_cosine_distance=summaries[1],
        between_class_cosine_distance=summaries[2],
        coefficient_digest=probe.coefficient_digest(),
        prediction_digest=prediction_digest(predictions),
        reset_prediction_digest=prediction_digest(reset_predictions),
        train_fixture_digest=training.fixture_digest,
        evaluation_fixture_digest=evaluation.fixture_digest,
        reservoir_parameter_digest=parameter_digest,
    )


def _collect_terminal_states(
    reservoir: object,
    sequences: tuple[HistorySequence, ...],
) -> tuple[np.ndarray, np.ndarray]:
    states = []
    labels = []
    for sequence in sequences:
        reservoir.reset()
        state = None
        for frame in sequence.frames:
            state = reservoir.advance(frame)
        if state is None:
            raise RuntimeError("E1-B sequence unexpectedly empty")
        states.append(state)
        labels.append(sequence.label)
    return np.vstack(states), np.asarray(labels, dtype=np.int64)


def _collect_reset_terminal_states(
    reservoir: object,
    sequences: tuple[HistorySequence, ...],
) -> tuple[np.ndarray, np.ndarray]:
    states = []
    labels = []
    for sequence in sequences:
        reservoir.reset()
        for frame in sequence.frames[:3]:
            reservoir.advance(frame)
        reservoir.reset()
        state = None
        for frame in sequence.frames[3:]:
            state = reservoir.advance(frame)
        if state is None:
            raise RuntimeError("E1-B reset tail unexpectedly empty")
        states.append(state)
        labels.append(sequence.label)
    return np.vstack(states), np.asarray(labels, dtype=np.int64)


def _reset_groups_equal(
    states: np.ndarray,
    sequences: tuple[HistorySequence, ...],
) -> bool:
    if states.shape[0] != len(sequences):
        return False
    for offset in range(0, len(sequences), 4):
        group = states[offset : offset + 4]
        if group.shape[0] != 4 or not np.array_equal(
            group, np.repeat(group[:1], 4, axis=0)
        ):
            return False
    return True


def _normalized_margins(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    margins = np.empty(scores.shape[0], dtype=np.float64)
    epsilon = np.finfo(np.float64).eps
    for index, label in enumerate(labels):
        true_score = scores[index, label]
        other_score = np.max(np.delete(scores[index], label))
        denominator = max(float(np.linalg.norm(scores[index])), epsilon)
        margins[index] = (true_score - other_score) / denominator
    return margins


def _cosine_distance_groups(
    states: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    within = []
    between = []
    for left in range(states.shape[0]):
        for right in range(left + 1, states.shape[0]):
            distance = _cosine_distance(states[left], states[right])
            if labels[left] == labels[right]:
                within.append(distance)
            else:
                between.append(distance)
    return np.asarray(within, dtype=np.float64), np.asarray(between, dtype=np.float64)


def _cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 and right_norm == 0.0:
        return 0.0
    if left_norm == 0.0 or right_norm == 0.0:
        return 1.0
    similarity = float(np.dot(left, right) / (left_norm * right_norm))
    return 1.0 - max(-1.0, min(1.0, similarity))


def _summary(values: np.ndarray) -> CountSummary:
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise RuntimeError("E1-B summary values must be non-empty and finite")
    return CountSummary(
        count=int(values.size),
        minimum=float(np.min(values)),
        median=float(np.median(values)),
        mean=float(np.mean(values)),
        maximum=float(np.max(values)),
    )


def _summary_is_finite(summary: CountSummary) -> bool:
    return summary.count > 0 and all(
        math.isfinite(value)
        for value in (summary.minimum, summary.median, summary.mean, summary.maximum)
    )


def _fixture_digest(sequences: tuple[HistorySequence, ...]) -> str:
    digest = hashlib.sha256()
    digest.update(_FIXTURE_DIGEST_VERSION)
    for sequence in sequences:
        digest.update(np.asarray(sequence.label, dtype=np.int64).tobytes())
        digest.update(np.asarray(sequence.horizon, dtype=np.int64).tobytes())
        digest.update(np.asarray(sequence.pair_index, dtype=np.int64).tobytes())
        frames = np.ascontiguousarray(sequence.frames, dtype=np.float64)
        digest.update(str(frames.shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(frames.tobytes(order="C"))
    return digest.hexdigest()


def _validate_seed(seed: object) -> int:
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative Python integer")
    return seed
