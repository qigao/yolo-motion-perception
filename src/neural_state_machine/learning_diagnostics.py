"""Immutable hidden-state datasets for Phase 2C diagnostics.

This module deliberately separates numeric recurrent-state collection from the
diagnostic metadata that labels those states.  The frozen policy sees only task
stimuli; labels and delays are associated after every hidden row is copied.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np

from .memory_benchmark import AccuracyCount
from .memory_probe import FittedLinearProbe, fit_linear_probe
from .memory_task import DelayedCueEpisode, DelayedCueTask
from .policy import RecurrentPolicy
from .reward_learning import _build_fixtures, _decision_hidden


@dataclass(frozen=True)
class LearningDiagnosticsConfig:
    """The pre-registered configuration shared by all Phase 2C diagnostics."""

    hidden_size: int = 64
    recurrent_radius: float = 0.9
    learning_rate: float = 0.05
    temperature: float = 1.0
    training_episodes: int = 2_000
    evaluation_blocks: int = 20
    checkpoint_interval: int = 100

    def __post_init__(self) -> None:
        for name in (
            "hidden_size",
            "training_episodes",
            "evaluation_blocks",
            "checkpoint_interval",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.training_episodes % 10:
            raise ValueError("training_episodes must be a multiple of ten")
        if self.checkpoint_interval % 10:
            raise ValueError("checkpoint_interval must be a multiple of ten")
        if self.training_episodes % self.checkpoint_interval:
            raise ValueError("checkpoint_interval must divide training_episodes")
        if not _finite_number(self.recurrent_radius) or not (
            0.0 <= self.recurrent_radius < 1.0
        ):
            raise ValueError("recurrent_radius must be finite and in [0.0, 1.0)")
        for name in ("learning_rate", "temperature"):
            value = getattr(self, name)
            if not _finite_number(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class _HiddenDataset:
    states: np.ndarray
    labels: np.ndarray
    delays: np.ndarray
    fixture_digest: str
    state_digest: str

    def __post_init__(self) -> None:
        states = _validated_states(self.states)
        labels = _validated_integer_vector(self.labels, "labels", (0, 1))
        delays = _validated_integer_vector(self.delays, "delays", range(1, 6))
        if labels.shape[0] != states.shape[0] or delays.shape[0] != states.shape[0]:
            raise ValueError("dataset arrays must have matching sample counts")
        if set(labels.tolist()) != {0, 1}:
            raise ValueError("labels must contain both action classes")
        _validate_digest(self.fixture_digest, "fixture_digest")
        _validate_digest(self.state_digest, "state_digest")
        if self.state_digest != _array_digest(states):
            raise ValueError("state_digest must match states")

        object.__setattr__(self, "states", _readonly_copy(states, np.float64))
        object.__setattr__(self, "labels", _readonly_copy(labels, np.int64))
        object.__setattr__(self, "delays", _readonly_copy(delays, np.int64))


@dataclass(frozen=True)
class _DiagnosticFixtures:
    training_episodes: tuple[DelayedCueEpisode, ...]
    evaluation_episodes: tuple[DelayedCueEpisode, ...]
    training: _HiddenDataset
    evaluation: _HiddenDataset


@dataclass(frozen=True)
class MarginSummary:
    minimum: float
    percentile_10: float
    median: float


@dataclass(frozen=True)
class GeometryDiagnostic:
    training: AccuracyCount
    evaluation: AccuracyCount
    per_delay: tuple[tuple[int, AccuracyCount], ...]
    training_margins: MarginSummary
    evaluation_margins: MarginSummary
    per_delay_margins: tuple[tuple[int, MarginSummary], ...]
    delay_five_to_one_median_ratio: float
    probe_digest: str
    geometry_passed: bool


def _build_diagnostic_fixtures(
    seed: int,
    config: LearningDiagnosticsConfig,
) -> _DiagnosticFixtures:
    """Build the independent, label-isolated training and evaluation datasets."""
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative Python integer")
    if not isinstance(config, LearningDiagnosticsConfig):
        raise ValueError(  # noqa: TRY004 - protocol validation uses ValueError
            "config must be a LearningDiagnosticsConfig"
        )

    task = DelayedCueTask()
    training_rng = np.random.default_rng(
        np.random.SeedSequence([seed, 0x54524149])
    )
    evaluation_rng = np.random.default_rng(
        np.random.SeedSequence([seed, 0x4556414C])
    )
    training_episodes = _build_fixtures(
        task,
        training_rng,
        config.training_episodes // 10,
    )
    evaluation_episodes = _build_fixtures(task, evaluation_rng, config.evaluation_blocks)

    training = _collect_hidden_dataset(
        _new_frozen_policy(seed, config), training_episodes
    )
    evaluation = _collect_hidden_dataset(
        _new_frozen_policy(seed, config), evaluation_episodes
    )
    return _DiagnosticFixtures(
        training_episodes=training_episodes,
        evaluation_episodes=evaluation_episodes,
        training=training,
        evaluation=evaluation,
    )


def _collect_hidden_dataset(
    policy: object,
    fixtures: tuple[DelayedCueEpisode, ...],
) -> _HiddenDataset:
    """Copy all numeric hidden rows before accessing labels or delay metadata."""
    hidden_rows = [
        np.array(
            _decision_hidden(policy, episode, reset_before_decision=False),
            dtype=np.float64,
            copy=True,
            order="C",
        )
        for episode in fixtures
    ]
    if not hidden_rows:
        raise ValueError("fixtures must contain at least one episode")
    states = np.ascontiguousarray(np.vstack(hidden_rows), dtype=np.float64)
    fixture_digest = _fixture_digest(fixtures)
    state_digest = _array_digest(states)

    # This association is intentionally after the complete numeric collection.
    labels = np.asarray(
        [episode.correct_action_index for episode in fixtures], dtype=np.int64
    )
    delays = np.asarray([episode.delay_steps for episode in fixtures], dtype=np.int64)
    return _HiddenDataset(states, labels, delays, fixture_digest, state_digest)


def _new_frozen_policy(
    seed: int,
    config: LearningDiagnosticsConfig,
) -> RecurrentPolicy:
    return RecurrentPolicy(
        input_size=4,
        action_count=2,
        hidden_size=config.hidden_size,
        seed=seed,
        recurrent_radius=config.recurrent_radius,
    )


def _normalized_signed_margins(
    probe: FittedLinearProbe,
    dataset: _HiddenDataset,
) -> np.ndarray:
    """Return immutable, L2-normalized margins with the label sign applied."""
    if not isinstance(probe, FittedLinearProbe):
        raise ValueError("probe must be a FittedLinearProbe")  # noqa: TRY004
    if not isinstance(dataset, _HiddenDataset):
        raise ValueError("dataset must be a _HiddenDataset")  # noqa: TRY004
    states = _validated_states(dataset.states)
    labels = _validated_integer_vector(dataset.labels, "labels", (0, 1))
    if labels.shape[0] != states.shape[0]:
        raise ValueError("dataset arrays must have matching sample counts")
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("labels must contain both action classes")
    if states.shape[1] != probe.weights.size:
        raise ValueError("states must match the probe feature count")
    norm = float(np.linalg.norm(probe.weights))
    if not math.isfinite(norm) or norm == 0.0:
        raise ValueError("probe weight norm must be finite and non-zero")
    signed = np.where(labels == 0, -1.0, 1.0)
    margins = signed * (states @ probe.weights + probe.bias) / norm
    return _readonly_copy(margins, np.float64)


def _margin_summary(margins: np.ndarray) -> MarginSummary:
    values = np.asarray(margins, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("margins must be a non-empty finite rank-one array")
    return MarginSummary(
        minimum=float(np.min(values)),
        percentile_10=float(np.percentile(values, 10, method="linear")),
        median=float(np.median(values)),
    )


def _run_geometry(
    training: _HiddenDataset,
    evaluation: _HiddenDataset,
) -> GeometryDiagnostic:
    """Measure frozen Ridge accuracy and normalized signed-margin geometry."""
    if not isinstance(training, _HiddenDataset) or not isinstance(
        evaluation, _HiddenDataset
    ):
        raise ValueError(  # noqa: TRY004 - protocol validation uses ValueError
            "training and evaluation must be _HiddenDataset instances"
        )
    probe = fit_linear_probe(
        training.states,
        training.labels,
        regularization=1e-3,
    )
    training_margins = _normalized_signed_margins(probe, training)
    evaluation_margins = _normalized_signed_margins(probe, evaluation)
    training_choices = probe.predict(training.states)
    evaluation_choices = probe.predict(evaluation.states)
    training_matches = training_choices == training.labels
    evaluation_matches = evaluation_choices == evaluation.labels
    training_accuracy = AccuracyCount(
        int(np.count_nonzero(training_matches)), int(training_matches.size)
    )
    evaluation_accuracy = AccuracyCount(
        int(np.count_nonzero(evaluation_matches)), int(evaluation_matches.size)
    )
    per_delay: list[tuple[int, AccuracyCount]] = []
    per_delay_margins: list[tuple[int, MarginSummary]] = []
    for delay in range(1, 6):
        delay_mask = evaluation.delays == delay
        total = int(np.count_nonzero(delay_mask))
        if total == 0:
            raise ValueError(f"evaluation delay {delay} has zero samples")
        per_delay.append(
            (
                delay,
                AccuracyCount(
                    int(np.count_nonzero(evaluation_matches[delay_mask])), total
                ),
            )
        )
        per_delay_margins.append(
            (delay, _margin_summary(evaluation_margins[delay_mask]))
        )
    delay_summaries = dict(per_delay_margins)
    delay_one_median = delay_summaries[1].median
    if delay_one_median == 0.0:
        raise ValueError("delay one median margin must be non-zero")
    geometry_passed = (
        evaluation_accuracy == AccuracyCount(200, 200)
        and tuple(per_delay)
        == tuple((delay, AccuracyCount(40, 40)) for delay in range(1, 6))
        and bool(np.all(training_margins > 0.0))
        and bool(np.all(evaluation_margins > 0.0))
    )
    return GeometryDiagnostic(
        training=training_accuracy,
        evaluation=evaluation_accuracy,
        per_delay=tuple(per_delay),
        training_margins=_margin_summary(training_margins),
        evaluation_margins=_margin_summary(evaluation_margins),
        per_delay_margins=tuple(per_delay_margins),
        delay_five_to_one_median_ratio=(
            delay_summaries[5].median / delay_one_median
        ),
        probe_digest=probe.digest(),
        geometry_passed=geometry_passed,
    )


def _fixture_digest(fixtures: tuple[DelayedCueEpisode, ...]) -> str:
    digest = hashlib.sha256()
    for episode in fixtures:
        _digest_array(digest, episode.cue_stimulus)
        for stimulus in episode.delay_stimuli:
            _digest_array(digest, stimulus)
        _digest_array(digest, episode.decision_stimulus)
    return digest.hexdigest()


def _array_digest(values: np.ndarray) -> str:
    digest = hashlib.sha256()
    _digest_array(digest, values)
    return digest.hexdigest()


def _digest_array(digest: hashlib._Hash, values: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(values, dtype=np.float64)
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))


def _finite_number(value: object) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except TypeError:
        return False


def _validated_states(values: object) -> np.ndarray:
    try:
        states = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("states must be float64-compatible") from exc
    if states.ndim != 2 or states.shape[0] == 0 or states.shape[1] == 0:
        raise ValueError("states must be a non-empty rank-two matrix")
    if not np.all(np.isfinite(states)):
        raise ValueError("states must contain only finite values")
    return states


def _validated_integer_vector(
    values: object,
    name: str,
    allowed: object,
) -> np.ndarray:
    try:
        numeric = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be integer-compatible") from exc
    if numeric.ndim != 1 or not np.all(np.isfinite(numeric)):
        raise ValueError(f"{name} must be a finite rank-one vector")
    if not np.all(numeric == np.floor(numeric)):
        raise ValueError(f"{name} must contain integers")
    integer = np.asarray(numeric, dtype=np.int64)
    if not np.all(np.isin(integer, tuple(allowed))):
        raise ValueError(f"{name} contains an unsupported value")
    return integer


def _validate_digest(value: object, name: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a SHA-256 hexadecimal string")


def _readonly_copy(values: np.ndarray, dtype: np.dtype) -> np.ndarray:
    copied = np.array(values, dtype=dtype, copy=True, order="C")
    copied.flags.writeable = False
    return copied
