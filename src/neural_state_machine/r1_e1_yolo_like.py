from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np

from .r1_e1_probe import fit_multiclass_ridge, prediction_digest
from .r1_e1_reservoir import ReservoirSpec, build_reservoir


BEHAVIOR_CLASSES = ("approach", "touch", "pick_up", "pass_by")
CORRUPTION_ARMS = ("clean", "drop10", "wrong10", "occlusion4", "jitter", "mixed")
_TRAIN_PAIRS = 200
_EVAL_PAIRS = 50
_TRAIN_TAG = 0x45314354
_EVAL_TAG = 0x45314345
_CORRUPTION_TAG = 0x45314343
_FIXTURE_DIGEST_VERSION = b"r1-e1-yolo-like-fixture-v1\0"
_CORRUPTION_DIGEST_VERSION = b"r1-e1-yolo-like-corruption-v1\0"
_MISSING = np.zeros(9, dtype=np.float64)
_MISSING.flags.writeable = False
_CONTINUOUS_DIMS = np.array([0, 2, 3, 4, 5, 7, 8], dtype=np.int64)
_UNIT_DIMS = np.array([0, 2, 3, 4, 5], dtype=np.int64)
_NUISANCE_DIMS = np.array([7, 8], dtype=np.int64)


@dataclass(frozen=True)
class BehaviorSequence:
    label: int
    behavior: str
    pair_index: int
    fixture_index: int
    frames: np.ndarray

    def __post_init__(self) -> None:
        if type(self.label) is not int or self.label not in range(4):
            raise ValueError("label must be an integer in [0, 3]")
        if self.behavior != BEHAVIOR_CLASSES[self.label]:
            raise ValueError("behavior must match label")
        if type(self.pair_index) is not int or self.pair_index < 0:
            raise ValueError("pair_index must be a non-negative integer")
        if type(self.fixture_index) is not int or self.fixture_index < 0:
            raise ValueError("fixture_index must be a non-negative integer")
        values = np.asarray(self.frames, dtype=np.float64)
        if values.shape != (20, 9):
            raise ValueError("frames must have shape (20, 9)")
        if not np.all(np.isfinite(values)):
            raise ValueError("frames must be finite")
        copied = np.array(values, dtype=np.float64, copy=True, order="C")
        copied.flags.writeable = False
        object.__setattr__(self, "frames", copied)


@dataclass(frozen=True)
class CleanFixtureSet:
    sequences: tuple[BehaviorSequence, ...]
    fixture_digest: str


@dataclass(frozen=True)
class CorruptionEntry:
    fixture_index: int
    label: int
    drop_indices: tuple[int, int]
    wrong_indices: tuple[int, int]
    occlusion_indices: tuple[int, int, int, int]
    mixed_drop_indices: tuple[int, int]
    mixed_wrong_index: int
    jitter_noise: np.ndarray
    mixed_jitter_noise: np.ndarray

    def __post_init__(self) -> None:
        if type(self.fixture_index) is not int or self.fixture_index < 0:
            raise ValueError("fixture_index must be a non-negative integer")
        if type(self.label) is not int or self.label not in range(4):
            raise ValueError("label must be an integer in [0, 3]")
        _validate_two_indices(self.drop_indices, "drop_indices")
        _validate_two_indices(self.wrong_indices, "wrong_indices")
        if self.occlusion_indices != (8, 9, 10, 11):
            raise ValueError("occlusion_indices must be exactly (8, 9, 10, 11)")
        _validate_two_indices(self.mixed_drop_indices, "mixed_drop_indices")
        if (
            type(self.mixed_wrong_index) is not int
            or not 0 <= self.mixed_wrong_index <= 15
            or self.mixed_wrong_index in self.mixed_drop_indices
        ):
            raise ValueError("mixed_wrong_index must be distinct and in [0, 15]")
        object.__setattr__(self, "jitter_noise", _validated_noise(self.jitter_noise))
        object.__setattr__(
            self,
            "mixed_jitter_noise",
            _validated_noise(self.mixed_jitter_noise),
        )


@dataclass(frozen=True)
class CorruptionPlan:
    entries: tuple[CorruptionEntry, ...]
    digest: str


@dataclass(frozen=True)
class AccuracyResult:
    correct: int
    total: int

    def __post_init__(self) -> None:
        if type(self.correct) is not int or type(self.total) is not int:
            raise ValueError("accuracy counts must be Python integers")
        if self.total <= 0 or not 0 <= self.correct <= self.total:
            raise ValueError("accuracy counts are invalid")

    @property
    def accuracy(self) -> float:
        return self.correct / self.total


@dataclass(frozen=True)
class YoloLikeRobustnessResult:
    seed: int
    architecture: int
    budget: int
    clean: AccuracyResult
    corrupted: tuple[tuple[str, AccuracyResult], ...]
    accuracy_drops: tuple[tuple[str, float], ...]
    macro_corrupted_accuracy: float
    worst_corrupted_accuracy: float
    confusion_matrices: tuple[tuple[str, np.ndarray], ...]
    train_fixture_digest: str
    evaluation_fixture_digest: str
    corruption_digest: str
    coefficient_digest: str
    prediction_digests: tuple[tuple[str, str], ...]
    reservoir_parameter_digest: str


def build_clean_fixture_set(seed: int, *, training: bool) -> CleanFixtureSet:
    _validate_seed(seed)
    pair_count = _TRAIN_PAIRS if training else _EVAL_PAIRS
    tag = _TRAIN_TAG if training else _EVAL_TAG
    rng = np.random.default_rng(np.random.SeedSequence([seed, tag]))
    sequences = []
    fixture_index = 0
    for pair_index in range(pair_count):
        nuisance = rng.choice(
            np.array([-0.05, 0.05], dtype=np.float64),
            size=(20, 2),
            replace=True,
        )
        for label, behavior in enumerate(BEHAVIOR_CLASSES):
            frames = _clean_template(label, nuisance)
            sequences.append(
                BehaviorSequence(label, behavior, pair_index, fixture_index, frames)
            )
            fixture_index += 1
    result = tuple(sequences)
    return CleanFixtureSet(result, _fixture_digest(result))


def build_corruption_plan(seed: int, fixtures: CleanFixtureSet) -> CorruptionPlan:
    _validate_seed(seed)
    if not isinstance(fixtures, CleanFixtureSet):
        raise ValueError("fixtures must be a CleanFixtureSet")
    if len(fixtures.sequences) != _EVAL_PAIRS * 4:
        raise ValueError("corruption plan requires the 200-sequence evaluation fixture")
    rng = np.random.default_rng(np.random.SeedSequence([seed, _CORRUPTION_TAG]))
    entries = []
    for expected_index, sequence in enumerate(fixtures.sequences):
        if sequence.fixture_index != expected_index:
            raise ValueError("fixture indices must be contiguous and ordered")
        drop = tuple(sorted(int(value) for value in rng.choice(16, size=2, replace=False)))
        wrong = tuple(sorted(int(value) for value in rng.choice(16, size=2, replace=False)))
        mixed_drop = tuple(
            sorted(int(value) for value in rng.choice(16, size=2, replace=False))
        )
        remaining = np.array(
            [index for index in range(16) if index not in mixed_drop],
            dtype=np.int64,
        )
        mixed_wrong = int(rng.choice(remaining))
        jitter_noise = rng.normal(0.0, 0.05, size=(16, len(_CONTINUOUS_DIMS)))
        mixed_jitter_noise = rng.normal(
            0.0,
            0.05,
            size=(16, len(_CONTINUOUS_DIMS)),
        )
        entries.append(
            CorruptionEntry(
                fixture_index=sequence.fixture_index,
                label=sequence.label,
                drop_indices=drop,
                wrong_indices=wrong,
                occlusion_indices=(8, 9, 10, 11),
                mixed_drop_indices=mixed_drop,
                mixed_wrong_index=mixed_wrong,
                jitter_noise=jitter_noise,
                mixed_jitter_noise=mixed_jitter_noise,
            )
        )
    result = tuple(entries)
    return CorruptionPlan(result, _corruption_digest(result))


def corrupt_sequence(
    sequence: BehaviorSequence,
    paired_by_label: dict[int, BehaviorSequence],
    entry: CorruptionEntry,
    arm: str,
) -> np.ndarray:
    if not isinstance(sequence, BehaviorSequence):
        raise ValueError("sequence must be a BehaviorSequence")
    if not isinstance(entry, CorruptionEntry):
        raise ValueError("entry must be a CorruptionEntry")
    if sequence.fixture_index != entry.fixture_index or sequence.label != entry.label:
        raise ValueError("corruption entry does not match sequence")
    if arm not in CORRUPTION_ARMS:
        raise ValueError("arm must be a registered corruption arm")
    if set(paired_by_label) != {0, 1, 2, 3}:
        raise ValueError("paired_by_label must contain all four behavior labels")
    if any(item.pair_index != sequence.pair_index for item in paired_by_label.values()):
        raise ValueError("donor sequences must come from the same paired realization")

    frames = np.array(sequence.frames, dtype=np.float64, copy=True, order="C")
    if arm == "clean":
        return _readonly(frames)
    if arm == "drop10":
        frames[list(entry.drop_indices)] = _MISSING
    elif arm == "wrong10":
        _apply_wrong(frames, sequence, paired_by_label, entry.wrong_indices)
    elif arm == "occlusion4":
        frames[list(entry.occlusion_indices)] = _MISSING
    elif arm == "jitter":
        _apply_jitter(frames, entry.jitter_noise, missing_indices=())
    elif arm == "mixed":
        frames[list(entry.mixed_drop_indices)] = _MISSING
        _apply_wrong(
            frames,
            sequence,
            paired_by_label,
            (entry.mixed_wrong_index,),
        )
        _apply_jitter(
            frames,
            entry.mixed_jitter_noise,
            missing_indices=entry.mixed_drop_indices,
        )
    _validate_corrupted_frames(sequence.frames, frames, arm, entry)
    return _readonly(frames)


def run_yolo_like_robustness(spec: ReservoirSpec) -> YoloLikeRobustnessResult:
    if not isinstance(spec, ReservoirSpec):
        raise ValueError("spec must be a ReservoirSpec")
    if spec.input_size != 9:
        raise ValueError("E1-C requires reservoir input_size 9")

    training = build_clean_fixture_set(spec.seed, training=True)
    evaluation = build_clean_fixture_set(spec.seed, training=False)
    if len(training.sequences) != _TRAIN_PAIRS * 4:
        raise RuntimeError("E1-C training count mismatch")
    if len(evaluation.sequences) != _EVAL_PAIRS * 4:
        raise RuntimeError("E1-C evaluation count mismatch")
    if not training.fixture_digest or not evaluation.fixture_digest:
        raise RuntimeError("E1-C fixture digest missing")
    plan = build_corruption_plan(spec.seed, evaluation)
    if not plan.digest:
        raise RuntimeError("E1-C corruption digest missing")

    reservoir = build_reservoir(spec)
    parameter_digest = reservoir.parameter_digest()
    train_states, train_labels = _collect_states(reservoir, training.sequences)
    if reservoir.state_dim != spec.budget:
        raise RuntimeError("E1-C state width mismatch")
    if reservoir.parameter_digest() != parameter_digest:
        raise RuntimeError("E1-C reservoir parameters changed during training collection")

    probe = fit_multiclass_ridge(
        train_states,
        train_labels,
        class_count=4,
        regularization=1e-6,
    )
    if reservoir.parameter_digest() != parameter_digest:
        raise RuntimeError("E1-C reservoir parameters changed during probe fitting")

    arm_scores = []
    arm_confusions = []
    arm_digests = []
    for arm in CORRUPTION_ARMS:
        states, labels = _collect_arm_states(reservoir, evaluation, plan, arm)
        if reservoir.parameter_digest() != parameter_digest:
            raise RuntimeError("E1-C reservoir parameters changed during evaluation")
        predictions = probe.predict(states)
        score = AccuracyResult(
            int(np.count_nonzero(predictions == labels)),
            int(predictions.size),
        )
        confusion = _confusion_matrix(labels, predictions)
        arm_scores.append((arm, score))
        arm_confusions.append((arm, confusion))
        arm_digests.append((arm, prediction_digest(predictions)))

    clean = arm_scores[0][1]
    corrupted = tuple(arm_scores[1:])
    drops = tuple((name, clean.accuracy - score.accuracy) for name, score in corrupted)
    corrupted_accuracies = np.asarray(
        [score.accuracy for _, score in corrupted],
        dtype=np.float64,
    )
    macro = float(np.mean(corrupted_accuracies))
    worst = float(np.min(corrupted_accuracies))
    if not math.isfinite(macro) or not math.isfinite(worst):
        raise RuntimeError("E1-C produced non-finite robustness metrics")

    return YoloLikeRobustnessResult(
        seed=spec.seed,
        architecture=int(spec.architecture),
        budget=spec.budget,
        clean=clean,
        corrupted=corrupted,
        accuracy_drops=drops,
        macro_corrupted_accuracy=macro,
        worst_corrupted_accuracy=worst,
        confusion_matrices=tuple(arm_confusions),
        train_fixture_digest=training.fixture_digest,
        evaluation_fixture_digest=evaluation.fixture_digest,
        corruption_digest=plan.digest,
        coefficient_digest=probe.coefficient_digest(),
        prediction_digests=tuple(arm_digests),
        reservoir_parameter_digest=parameter_digest,
    )


def _clean_template(label: int, nuisance: np.ndarray) -> np.ndarray:
    if label not in range(4):
        raise ValueError("label must be in [0, 3]")
    nuisance_values = np.asarray(nuisance, dtype=np.float64)
    if nuisance_values.shape != (20, 2):
        raise ValueError("nuisance must have shape (20, 2)")
    if not np.all(np.isin(nuisance_values, (-0.05, 0.05))):
        raise ValueError("nuisance values must be -0.05 or +0.05")

    frames = np.zeros((20, 9), dtype=np.float64)
    if label == 0:  # approach
        frames[:16, 0] = 1.0 - 0.8 * np.arange(16, dtype=np.float64) / 15.0
        frames[:16, 2] = 0.8
        frames[:16, 6] = 1.0
    elif label == 1:  # touch
        frames[:11, 0] = 1.0 - np.arange(11, dtype=np.float64) / 10.0
        frames[:11, 2] = 1.0
        frames[11:16, 1] = 1.0
        frames[:16, 6] = 1.0
    elif label == 2:  # pick_up
        frames[:9, 0] = 1.0 - np.arange(9, dtype=np.float64) / 8.0
        frames[:9, 2] = 1.0
        frames[9:16, 1] = 1.0
        frames[9:16, 2] = 1.0
        frames[10:16, 3] = 1.0
        frames[10:16, 4] = 1.0
        frames[9:16, 5] = np.arange(7, dtype=np.float64) / 6.0
        frames[:16, 6] = 1.0
    else:  # pass_by
        frames[:8, 0] = 1.0 - 0.9 * np.arange(8, dtype=np.float64) / 7.0
        frames[8:16, 0] = 0.1 + 0.9 * np.arange(8, dtype=np.float64) / 7.0
        frames[:16, 2] = 1.0
        frames[:16, 6] = 1.0

    frames[16:20, 0] = 0.5
    frames[16:20, 6] = 1.0
    frames[:, 7:9] = nuisance_values
    return frames


def _collect_states(
    reservoir: object,
    sequences: tuple[BehaviorSequence, ...],
) -> tuple[np.ndarray, np.ndarray]:
    states = []
    labels = []
    for sequence in sequences:
        reservoir.reset()
        state = None
        for frame in sequence.frames:
            state = reservoir.advance(frame)
        if state is None:
            raise RuntimeError("E1-C sequence unexpectedly empty")
        states.append(state)
        labels.append(sequence.label)
    return np.vstack(states), np.asarray(labels, dtype=np.int64)


def _collect_arm_states(
    reservoir: object,
    fixtures: CleanFixtureSet,
    plan: CorruptionPlan,
    arm: str,
) -> tuple[np.ndarray, np.ndarray]:
    if len(fixtures.sequences) != len(plan.entries):
        raise RuntimeError("E1-C corruption plan count mismatch")
    grouped = _paired_sequences(fixtures.sequences)
    states = []
    labels = []
    for sequence, entry in zip(fixtures.sequences, plan.entries, strict=True):
        frames = corrupt_sequence(sequence, grouped[sequence.pair_index], entry, arm)
        reservoir.reset()
        state = None
        for frame in frames:
            state = reservoir.advance(frame)
        if state is None:
            raise RuntimeError("E1-C corrupted sequence unexpectedly empty")
        states.append(state)
        labels.append(sequence.label)
    matrix = np.vstack(states)
    if matrix.shape != (200, reservoir.state_dim):
        raise RuntimeError("E1-C evaluation state matrix has the wrong shape")
    return matrix, np.asarray(labels, dtype=np.int64)


def _paired_sequences(
    sequences: tuple[BehaviorSequence, ...],
) -> dict[int, dict[int, BehaviorSequence]]:
    groups: dict[int, dict[int, BehaviorSequence]] = {}
    for sequence in sequences:
        groups.setdefault(sequence.pair_index, {})[sequence.label] = sequence
    if any(set(group) != {0, 1, 2, 3} for group in groups.values()):
        raise RuntimeError("E1-C paired fixture group is incomplete")
    return groups


def _apply_wrong(
    frames: np.ndarray,
    sequence: BehaviorSequence,
    paired_by_label: dict[int, BehaviorSequence],
    indices: tuple[int, ...],
) -> None:
    donor = paired_by_label[(sequence.label + 1) % 4]
    for index in indices:
        target_nuisance = frames[index, 7:9].copy()
        frames[index] = donor.frames[index]
        frames[index, 7:9] = target_nuisance


def _apply_jitter(
    frames: np.ndarray,
    noise: np.ndarray,
    *,
    missing_indices: tuple[int, ...],
) -> None:
    missing = set(missing_indices)
    for frame_index in range(16):
        if frame_index in missing:
            continue
        frames[frame_index, _CONTINUOUS_DIMS] += noise[frame_index]
    frames[:16, _UNIT_DIMS] = np.clip(frames[:16, _UNIT_DIMS], 0.0, 1.0)
    frames[:16, _NUISANCE_DIMS] = np.clip(
        frames[:16, _NUISANCE_DIMS],
        -0.1,
        0.1,
    )


def _validate_corrupted_frames(
    clean: np.ndarray,
    corrupted: np.ndarray,
    arm: str,
    entry: CorruptionEntry,
) -> None:
    if corrupted.shape != (20, 9) or not np.all(np.isfinite(corrupted)):
        raise RuntimeError("E1-C corruption produced invalid frames")
    if not np.array_equal(corrupted[16:20], clean[16:20]):
        raise RuntimeError("E1-C corruption changed the terminal neutral suffix")
    if arm in ("jitter", "mixed"):
        missing = set(entry.mixed_drop_indices if arm == "mixed" else ())
        for index in range(16):
            if index in missing:
                continue
            if corrupted[index, 1] != clean[index, 1] and arm == "jitter":
                raise RuntimeError("E1-C jitter changed overlap")
            if corrupted[index, 6] != clean[index, 6] and arm == "jitter":
                raise RuntimeError("E1-C jitter changed visibility")
        if np.any(corrupted[:16, _UNIT_DIMS] < 0.0) or np.any(
            corrupted[:16, _UNIT_DIMS] > 1.0
        ):
            raise RuntimeError("E1-C jitter escaped unit clipping bounds")
        if np.any(corrupted[:16, _NUISANCE_DIMS] < -0.1) or np.any(
            corrupted[:16, _NUISANCE_DIMS] > 0.1
        ):
            raise RuntimeError("E1-C jitter escaped nuisance clipping bounds")


def _confusion_matrix(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    matrix = np.zeros((4, 4), dtype=np.int64)
    for label, prediction in zip(labels, predictions, strict=True):
        matrix[int(label), int(prediction)] += 1
    matrix.flags.writeable = False
    return matrix


def _fixture_digest(sequences: tuple[BehaviorSequence, ...]) -> str:
    digest = hashlib.sha256()
    digest.update(_FIXTURE_DIGEST_VERSION)
    for sequence in sequences:
        digest.update(np.asarray(sequence.label, dtype=np.int64).tobytes())
        digest.update(np.asarray(sequence.pair_index, dtype=np.int64).tobytes())
        digest.update(np.asarray(sequence.fixture_index, dtype=np.int64).tobytes())
        frames = np.ascontiguousarray(sequence.frames, dtype=np.float64)
        digest.update(str(frames.shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(frames.tobytes(order="C"))
    return digest.hexdigest()


def _corruption_digest(entries: tuple[CorruptionEntry, ...]) -> str:
    digest = hashlib.sha256()
    digest.update(_CORRUPTION_DIGEST_VERSION)
    for entry in entries:
        for value in (
            entry.fixture_index,
            entry.label,
            *entry.drop_indices,
            *entry.wrong_indices,
            *entry.occlusion_indices,
            *entry.mixed_drop_indices,
            entry.mixed_wrong_index,
        ):
            digest.update(np.asarray(value, dtype=np.int64).tobytes())
        for noise in (entry.jitter_noise, entry.mixed_jitter_noise):
            values = np.ascontiguousarray(noise, dtype=np.float64)
            digest.update(str(values.shape).encode("ascii"))
            digest.update(b"\0")
            digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _validated_noise(values: object) -> np.ndarray:
    try:
        noise = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("jitter noise must be float64-compatible") from exc
    if noise.shape != (16, 7) or not np.all(np.isfinite(noise)):
        raise ValueError("jitter noise must be finite with shape (16, 7)")
    return _readonly(noise)


def _validate_two_indices(values: object, name: str) -> None:
    if type(values) is not tuple or len(values) != 2:
        raise ValueError(f"{name} must contain exactly two indices")
    if any(type(value) is not int or not 0 <= value <= 15 for value in values):
        raise ValueError(f"{name} values must be Python integers in [0, 15]")
    if len(set(values)) != 2:
        raise ValueError(f"{name} values must be distinct")


def _readonly(values: np.ndarray) -> np.ndarray:
    copied = np.array(values, dtype=np.float64, copy=True, order="C")
    copied.flags.writeable = False
    return copied


def _validate_seed(seed: object) -> int:
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative Python integer")
    return seed
