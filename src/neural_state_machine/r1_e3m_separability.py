from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .r1_e2_reservoir import E2Architecture
from .r1_e3m_representation import build_representations
from .r1_e3m_windows import UnlabeledWindow


_STD_FLOOR = 1e-12


@dataclass(frozen=True)
class RawStandardizer:
    mean: np.ndarray
    std: np.ndarray

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float64)
        std = np.asarray(self.std, dtype=np.float64)
        if mean.shape != (14,) or std.shape != (14,):
            raise ValueError("raw standardizer must contain 14 channels")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
            raise ValueError("raw standardizer must be finite")
        if np.any(std < _STD_FLOOR):
            raise ValueError("raw standardizer std must respect floor")
        object.__setattr__(self, "mean", _readonly(mean))
        object.__setattr__(self, "std", _readonly(std))

    def transform(self, tensors: np.ndarray) -> np.ndarray:
        values = _tensor_batch(tensors)
        result = (values - self.mean[None, None, :]) / self.std[
            None, None, :
        ]
        return _readonly(result)


@dataclass(frozen=True)
class MatchedPair:
    sequence_a: str
    sequence_b: str
    video_a: str
    video_b: str
    suffix_distance: float
    prefix_distance: float
    r1_distance: float
    r4_distance: float


@dataclass(frozen=True)
class SeparabilityResult:
    pairs: tuple[MatchedPair, ...]
    correlations: Mapping[str, float | None]


def fit_raw_standardizer(
    windows: tuple[UnlabeledWindow, ...],
) -> RawStandardizer:
    training = tuple(window for window in windows if window.split == "train")
    if not training:
        raise ValueError("raw standardizer requires training windows")
    tensors = np.stack([window.tensor for window in training], axis=0)
    flat = tensors.reshape(-1, 14)
    mean = np.mean(flat, axis=0, dtype=np.float64)
    std = np.std(flat, axis=0, dtype=np.float64)
    std = np.maximum(std, _STD_FLOOR)
    return RawStandardizer(mean=mean, std=std)


def run_suffix_matched_separability(
    windows: tuple[UnlabeledWindow, ...],
    *,
    architecture: E2Architecture,
    seed: int,
) -> SeparabilityResult:
    standardizer = fit_raw_standardizer(windows)
    evaluation = tuple(
        sorted(
            (window for window in windows if window.split == "eval"),
            key=lambda window: window.sequence_id,
        )
    )
    if len(evaluation) < 2:
        raise ValueError("suffix matching requires evaluation windows")

    ids = [window.sequence_id for window in evaluation]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate evaluation sequence_id")

    tensors = np.stack([window.tensor for window in evaluation], axis=0)
    standardized = standardizer.transform(tensors)
    prefix = standardized[:, :16, :].reshape(len(evaluation), -1)
    suffix = standardized[:, 16:20, :].reshape(len(evaluation), -1)
    representations = build_representations(
        tensors,
        architecture=architecture,
        seed=seed,
    )

    selected: dict[tuple[str, str], MatchedPair] = {}
    for index, anchor in enumerate(evaluation):
        candidates = [
            candidate_index
            for candidate_index, candidate in enumerate(evaluation)
            if candidate_index != index
            and candidate.video_id != anchor.video_id
        ]
        if not candidates:
            continue

        neighbor_index = min(
            candidates,
            key=lambda candidate_index: (
                _distance(suffix[index], suffix[candidate_index]),
                evaluation[candidate_index].sequence_id,
            ),
        )
        neighbor = evaluation[neighbor_index]
        key = tuple(sorted((anchor.sequence_id, neighbor.sequence_id)))
        if key in selected:
            continue

        if anchor.sequence_id <= neighbor.sequence_id:
            a_index, b_index = index, neighbor_index
        else:
            a_index, b_index = neighbor_index, index
        a = evaluation[a_index]
        b = evaluation[b_index]
        selected[key] = MatchedPair(
            sequence_a=a.sequence_id,
            sequence_b=b.sequence_id,
            video_a=a.video_id,
            video_b=b.video_id,
            suffix_distance=_distance(suffix[a_index], suffix[b_index]),
            prefix_distance=_distance(prefix[a_index], prefix[b_index]),
            r1_distance=_distance(
                representations.r1[a_index],
                representations.r1[b_index],
            ),
            r4_distance=_distance(
                representations.r4[a_index],
                representations.r4[b_index],
            ),
        )

    pairs = tuple(selected[key] for key in sorted(selected))
    if not pairs:
        raise ValueError("suffix matching produced no cross-video pairs")

    prefix_distances = np.asarray(
        [pair.prefix_distance for pair in pairs],
        dtype=np.float64,
    )
    suffix_distances = np.asarray(
        [pair.suffix_distance for pair in pairs],
        dtype=np.float64,
    )
    r1_distances = np.asarray(
        [pair.r1_distance for pair in pairs],
        dtype=np.float64,
    )
    r4_distances = np.asarray(
        [pair.r4_distance for pair in pairs],
        dtype=np.float64,
    )
    correlations = {
        "prefix_vs_r1": _spearman(prefix_distances, r1_distances),
        "prefix_vs_r4": _spearman(prefix_distances, r4_distances),
        "suffix_vs_r1": _spearman(suffix_distances, r1_distances),
        "suffix_vs_r4": _spearman(suffix_distances, r4_distances),
    }
    return SeparabilityResult(pairs=pairs, correlations=correlations)


def _distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left - right))


def _spearman(
    left: np.ndarray,
    right: np.ndarray,
) -> float | None:
    if left.size < 2 or right.size != left.size:
        return None
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    left_centered = left_ranks - np.mean(left_ranks)
    right_centered = right_ranks - np.mean(right_ranks)
    denominator = math.sqrt(
        float(np.sum(left_centered**2))
        * float(np.sum(right_centered**2))
    )
    if denominator <= 0.0:
        return None
    return float(
        np.sum(left_centered * right_centered) / denominator
    )


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    position = 0
    while position < values.size:
        start = position
        value = values[order[position]]
        while (
            position + 1 < values.size
            and values[order[position + 1]] == value
        ):
            position += 1
        end = position
        average = (start + end) / 2.0 + 1.0
        ranks[order[start : end + 1]] = average
        position += 1
    return ranks


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


def _readonly(values: np.ndarray) -> np.ndarray:
    copied = np.array(values, dtype=np.float64, copy=True, order="C")
    copied.flags.writeable = False
    return copied
