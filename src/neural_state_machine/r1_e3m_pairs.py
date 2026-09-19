from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from statistics import median

import numpy as np

from neural_state_machine.r1_e3m_dataset import MechanismSample


_PAIR_DIGEST_VERSION = b"r1-e3m-history-pairs-v1\0"


@dataclass(frozen=True)
class HistoryPair:
    left_window_id: str
    right_window_id: str
    suffix_distance: float
    prefix_distance: float

    def __post_init__(self) -> None:
        if self.left_window_id >= self.right_window_id:
            raise ValueError(
                "history pair IDs must be lexical and distinct"
            )
        for value in (self.suffix_distance, self.prefix_distance):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError("history-pair distances must be finite")


@dataclass(frozen=True)
class HistoryPairSet:
    prefix_threshold: float
    pairs: tuple[HistoryPair, ...]
    pair_digest: str

    def __post_init__(self) -> None:
        if (
            not math.isfinite(float(self.prefix_threshold))
            or float(self.prefix_threshold) < 0.0
        ):
            raise ValueError("prefix_threshold must be finite")
        if (
            not isinstance(self.pair_digest, str)
            or len(self.pair_digest) != 64
            or any(
                char not in "0123456789abcdef"
                for char in self.pair_digest
            )
        ):
            raise ValueError("pair_digest must be SHA-256")


@dataclass(frozen=True)
class PairStateDiagnostic:
    left_window_id: str
    right_window_id: str
    suffix_distance: float
    prefix_distance: float
    normal_state_distance: float
    reset_state_distance: float


def build_history_pairs(
    training: tuple[MechanismSample, ...],
    evaluation: tuple[MechanismSample, ...],
) -> HistoryPairSet:
    _validate_samples(training, "train")
    _validate_samples(evaluation, "eval")

    ordered_training = tuple(
        sorted(training, key=lambda sample: sample.window_id)
    )
    training_prefix_distances: list[float] = []
    for left_index, left in enumerate(ordered_training):
        for right in ordered_training[left_index + 1 :]:
            if (left.video_id, left.track_id) == (
                right.video_id,
                right.track_id,
            ):
                continue
            training_prefix_distances.append(
                _input_distance(
                    left.tensor[:16],
                    right.tensor[:16],
                )
            )
    if not training_prefix_distances:
        raise ValueError(
            "history-pair threshold requires different training track identities"
        )
    threshold = float(median(training_prefix_distances))

    pair_map: dict[tuple[str, str], HistoryPair] = {}
    ordered_evaluation = tuple(
        sorted(evaluation, key=lambda sample: sample.window_id)
    )
    for sample in ordered_evaluation:
        candidates: list[tuple[float, str, MechanismSample, float]] = []
        for candidate in ordered_evaluation:
            if candidate.window_id == sample.window_id:
                continue
            if (candidate.video_id, candidate.track_id) == (
                sample.video_id,
                sample.track_id,
            ):
                continue
            prefix_distance = _input_distance(
                sample.tensor[:16],
                candidate.tensor[:16],
            )
            if prefix_distance < threshold:
                continue
            suffix_distance = _input_distance(
                sample.tensor[16:20],
                candidate.tensor[16:20],
            )
            candidates.append(
                (
                    suffix_distance,
                    candidate.window_id,
                    candidate,
                    prefix_distance,
                )
            )
        if not candidates:
            continue

        (
            suffix_distance,
            _candidate_id,
            neighbor,
            prefix_distance,
        ) = min(candidates, key=lambda item: (item[0], item[1]))
        left_id, right_id = sorted(
            (sample.window_id, neighbor.window_id)
        )
        key = (left_id, right_id)
        pair_map[key] = HistoryPair(
            left_window_id=left_id,
            right_window_id=right_id,
            suffix_distance=suffix_distance,
            prefix_distance=prefix_distance,
        )

    pairs = tuple(pair_map[key] for key in sorted(pair_map))
    digest = _pair_digest(threshold, pairs)
    return HistoryPairSet(
        prefix_threshold=threshold,
        pairs=pairs,
        pair_digest=digest,
    )


def score_history_pairs(
    pair_set: HistoryPairSet,
    evaluation: tuple[MechanismSample, ...],
    normal_states: np.ndarray,
    reset_states: np.ndarray,
) -> tuple[PairStateDiagnostic, ...]:
    if not isinstance(pair_set, HistoryPairSet):
        raise ValueError("pair_set must be HistoryPairSet")
    _validate_samples(evaluation, "eval")
    normal = _state_matrix(
        normal_states,
        len(evaluation),
        "normal_states",
    )
    reset = _state_matrix(
        reset_states,
        len(evaluation),
        "reset_states",
    )
    if normal.shape != reset.shape:
        raise ValueError(
            "normal_states and reset_states must have matching shape"
        )

    index_by_id = {
        sample.window_id: index
        for index, sample in enumerate(evaluation)
    }
    diagnostics = []
    for pair in pair_set.pairs:
        try:
            left_index = index_by_id[pair.left_window_id]
            right_index = index_by_id[pair.right_window_id]
        except KeyError as exc:
            raise ValueError(
                "history pair references unknown evaluation window"
            ) from exc

        diagnostics.append(
            PairStateDiagnostic(
                left_window_id=pair.left_window_id,
                right_window_id=pair.right_window_id,
                suffix_distance=pair.suffix_distance,
                prefix_distance=pair.prefix_distance,
                normal_state_distance=_input_distance(
                    normal[left_index],
                    normal[right_index],
                ),
                reset_state_distance=_input_distance(
                    reset[left_index],
                    reset[right_index],
                ),
            )
        )
    return tuple(diagnostics)


def _nearest_suffix_neighbor(
    sample: MechanismSample,
    candidates: tuple[MechanismSample, ...],
) -> MechanismSample:
    eligible = [
        candidate
        for candidate in candidates
        if candidate.window_id != sample.window_id
        and (candidate.video_id, candidate.track_id)
        != (sample.video_id, sample.track_id)
    ]
    if not eligible:
        raise ValueError(
            "history-pair construction requires a different track identity"
        )
    return min(
        eligible,
        key=lambda candidate: (
            _input_distance(
                sample.tensor[16:20],
                candidate.tensor[16:20],
            ),
            candidate.window_id,
        ),
    )


def _validate_samples(
    samples: tuple[MechanismSample, ...],
    split: str,
) -> None:
    if not isinstance(samples, tuple) or len(samples) < 2:
        raise ValueError(
            "history-pair construction requires at least two samples"
        )
    if any(
        not isinstance(sample, MechanismSample)
        or sample.split != split
        for sample in samples
    ):
        raise ValueError(
            f"history-pair samples must all belong to {split}"
        )
    ids = [sample.window_id for sample in samples]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate window_id in history-pair inputs")


def _input_distance(left: np.ndarray, right: np.ndarray) -> float:
    left_values = np.asarray(left, dtype=np.float64)
    right_values = np.asarray(right, dtype=np.float64)
    if left_values.shape != right_values.shape:
        raise ValueError("distance inputs must have matching shape")
    if (
        left_values.size == 0
        or not np.isfinite(left_values).all()
        or not np.isfinite(right_values).all()
    ):
        raise ValueError("distance inputs must be finite and non-empty")
    return float(
        np.sqrt(
            np.mean(
                np.square(left_values - right_values),
                dtype=np.float64,
            )
        )
    )


def _state_matrix(
    values: object,
    expected_samples: int,
    name: str,
) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be float64-compatible") from exc
    if (
        array.ndim != 2
        or array.shape[0] != expected_samples
        or array.shape[1] == 0
    ):
        raise ValueError(
            f"{name} must have one non-empty row per evaluation sample"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _pair_digest(
    threshold: float,
    pairs: tuple[HistoryPair, ...],
) -> str:
    payload = {
        "prefix_threshold": threshold,
        "pairs": [
            {
                "left_window_id": pair.left_window_id,
                "right_window_id": pair.right_window_id,
                "suffix_distance": pair.suffix_distance,
                "prefix_distance": pair.prefix_distance,
            }
            for pair in pairs
        ],
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(_PAIR_DIGEST_VERSION)
    digest.update(encoded)
    return digest.hexdigest()
