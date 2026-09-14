"""Immutable hidden-state datasets for Phase 2C diagnostics.

This module deliberately separates numeric recurrent-state collection from the
diagnostic metadata that labels those states.  The frozen policy sees only task
stimuli; labels and delays are associated after every hidden row is copied.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path

import numpy as np

from .memory_benchmark import AccuracyCount
from .memory_probe import FittedLinearProbe, fit_linear_probe
from .memory_task import DelayedCueEpisode, DelayedCueTask
from .policy import RecurrentPolicy
from .reward_learning import (
    RewardLearningConfig,
    _build_fixtures,
    _decision_hidden,
    run_reward_learning_benchmark,
)
from .reward_readout import RewardModulatedReadout


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


@dataclass(frozen=True)
class DiagnosticCheckpoint:
    """A read-only supervised evaluation after a scheduled training episode."""

    episode: int
    overall: AccuracyCount
    per_delay: tuple[tuple[int, AccuracyCount], ...]
    parameter_digest: str


@dataclass(frozen=True)
class SupervisedDiagnostic:
    """Results from the private online supervised softmax instrument."""

    overall: AccuracyCount
    per_delay: tuple[tuple[int, AccuracyCount], ...]
    checkpoints: tuple[DiagnosticCheckpoint, ...]
    parameter_digest: str
    supervised_passed: bool


@dataclass(frozen=True)
class GradientBlockDiagnostic:
    """Read-only sampled and supervised gradients for one ten-case block."""

    episode: int
    sampled_episode_gradients: tuple[tuple[float, ...], ...]
    supervised_episode_gradients: tuple[tuple[float, ...], ...]
    expected_bandit_to_supervised_norm_ratios: tuple[float, ...]
    sampled_gradient: tuple[float, ...]
    supervised_gradient: tuple[float, ...]
    mean_expected_bandit_to_supervised_norm_ratio: float
    cosine: float | None


@dataclass(frozen=True)
class RewardCheckpoint:
    """A mutation-free reward-readout evaluation after scheduled training."""

    episode: int
    overall: AccuracyCount
    per_delay: tuple[tuple[int, AccuracyCount], ...]
    correct_action_probabilities: tuple[float, ...]
    mean_correct_action_probability: float
    percentile_10_correct_action_probability: float
    mean_expected_bandit_to_supervised_norm_ratio: float
    gradient_blocks: tuple[GradientBlockDiagnostic, ...]
    zero_norm_block_count: int
    parameter_digest: str


@dataclass(frozen=True)
class RewardTrajectoryDiagnostic:
    """Instrumentation of an otherwise unchanged Phase 2B reward run."""

    overall: AccuracyCount
    per_delay: tuple[tuple[int, AccuracyCount], ...]
    checkpoints: tuple[RewardCheckpoint, ...]
    gradient_blocks: tuple[GradientBlockDiagnostic, ...]
    zero_norm_block_count: int
    total_training_reward: int
    final_block: AccuracyCount
    parameter_digest_before: str
    parameter_digest_after: str
    matrix_digests_before: tuple[str, str, str]
    matrix_digests_after: tuple[str, str, str]
    training_choice_digest: str
    training_reward_digest: str
    reward_passed: bool


@dataclass(frozen=True)
class LearningDiagnosticsResult:
    """One complete, immutable Phase 2C measurement and its validity status."""

    seed: int
    config: LearningDiagnosticsConfig
    geometry: GeometryDiagnostic
    supervised: SupervisedDiagnostic
    reward_trajectory: RewardTrajectoryDiagnostic
    classification: str
    matrix_digests_before: tuple[str, str, str]
    matrix_digests_after: tuple[str, str, str]
    phase_2b_portable_evidence_match: bool
    diagnostic_valid: bool
    repeatable: bool
    training_fixture_digest: str
    training_state_digest: str
    evaluation_fixture_digest: str
    evaluation_state_digest: str


@dataclass(frozen=True)
class _Phase2bRuntimeLocalEvidence:
    """Same-environment Phase 2B floating evidence excluded from portability."""

    initial_parameter_digest: str
    normal_parameter_digest: str
    shuffled_parameter_digest: str
    normal_matrix_digests_before: tuple[str, str, str]
    normal_matrix_digests_after: tuple[str, str, str]
    shuffled_matrix_digests_before: tuple[str, str, str]
    shuffled_matrix_digests_after: tuple[str, str, str]


@dataclass(frozen=True)
class _CompleteLearningDiagnosticsRun:
    """Private complete-run record that contains only equality-safe values."""

    geometry: GeometryDiagnostic
    supervised: SupervisedDiagnostic
    reward_trajectory: RewardTrajectoryDiagnostic
    matrix_digests_before: tuple[str, str, str]
    matrix_digests_after: tuple[str, str, str]
    phase_2b_runtime_local_evidence: _Phase2bRuntimeLocalEvidence | None
    phase_2b_portable_evidence_match: bool
    datasets_valid: bool
    checkpoints_valid: bool
    matrix_integrity: bool
    training_fixture_digest: str
    training_state_digest: str
    evaluation_fixture_digest: str
    evaluation_state_digest: str


@dataclass(frozen=True)
class _SupervisedDecision:
    action_index: int
    logits: np.ndarray
    probabilities: np.ndarray


class _DiagnosticSupervisedReadout:
    """Private deterministic readout whose only update receives a class label."""

    def __init__(
        self,
        hidden_size: int,
        learning_rate: float = 0.05,
        temperature: float = 1.0,
    ) -> None:
        if type(hidden_size) is not int or hidden_size <= 0:
            raise ValueError("hidden_size must be a positive integer")
        for name, value in (
            ("learning_rate", learning_rate),
            ("temperature", temperature),
        ):
            if not _finite_number(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")

        self.hidden_size = hidden_size
        self.learning_rate = float(learning_rate)
        self.temperature = float(temperature)
        self._weights = np.zeros((2, hidden_size), dtype=np.float64)
        self._biases = np.zeros(2, dtype=np.float64)

    def select_greedy(self, hidden_state: np.ndarray) -> _SupervisedDecision:
        """Return the lowest-index maximum without creating any learning state."""
        hidden = self._validated_hidden_state(hidden_state)
        logits, probabilities = self._distribution(hidden)
        return _SupervisedDecision(
            action_index=int(np.argmax(probabilities)),
            logits=_readonly_copy(logits, np.float64),
            probabilities=_readonly_copy(probabilities, np.float64),
        )

    def observe_label(self, hidden_state: np.ndarray, label: int) -> np.ndarray:
        """Apply one literal softmax update from an explicit class label."""
        hidden = self._validated_hidden_state(hidden_state)
        if type(label) is not int or label not in (0, 1):
            raise ValueError("label must be action index zero or one")
        _, probabilities = self._distribution(hidden)
        delta = np.eye(2, dtype=np.float64)[label] - probabilities
        self._weights += self.learning_rate * np.outer(delta, hidden)
        self._biases += self.learning_rate * delta
        return _readonly_copy(probabilities, np.float64)

    def parameter_digest(self) -> str:
        weights = np.ascontiguousarray(self._weights, dtype=np.float64)
        biases = np.ascontiguousarray(self._biases, dtype=np.float64)
        digest = hashlib.sha256()
        digest.update(str(weights.shape).encode("ascii"))
        digest.update(weights.tobytes(order="C"))
        digest.update(str(biases.shape).encode("ascii"))
        digest.update(biases.tobytes(order="C"))
        return digest.hexdigest()

    def _distribution(self, hidden_state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        logits = self._weights @ hidden_state + self._biases
        scaled = logits / self.temperature
        exponentials = np.exp(scaled - np.max(scaled))
        probabilities = exponentials / np.sum(exponentials)
        return logits, probabilities

    def _validated_hidden_state(self, hidden_state: object) -> np.ndarray:
        try:
            values = np.asarray(hidden_state, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("hidden_state must be float64-compatible") from exc
        if values.ndim != 1 or values.shape != (self.hidden_size,):
            raise ValueError(
                f"hidden_state must have shape ({self.hidden_size},)"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("hidden_state must contain only finite values")
        return values


def _run_supervised_diagnostic(
    training: _HiddenDataset,
    evaluation: _HiddenDataset,
    config: LearningDiagnosticsConfig,
) -> SupervisedDiagnostic:
    """Train only on frozen hidden rows, associating labels after each copy."""
    if not isinstance(training, _HiddenDataset) or not isinstance(
        evaluation, _HiddenDataset
    ):
        raise ValueError(  # noqa: TRY004 - protocol validation uses ValueError
            "training and evaluation must be _HiddenDataset instances"
        )
    if not isinstance(config, LearningDiagnosticsConfig):
        raise ValueError(  # noqa: TRY004 - protocol validation uses ValueError
            "config must be a LearningDiagnosticsConfig"
        )
    if training.states.shape != (config.training_episodes, config.hidden_size):
        raise ValueError("training states must match the configured episode count")
    expected_evaluation_rows = config.evaluation_blocks * 10
    if evaluation.states.shape != (expected_evaluation_rows, config.hidden_size):
        raise ValueError("evaluation states must match the configured block count")

    readout = _DiagnosticSupervisedReadout(
        hidden_size=config.hidden_size,
        learning_rate=config.learning_rate,
        temperature=config.temperature,
    )
    checkpoints: list[DiagnosticCheckpoint] = []
    for episode_index, (state, label) in enumerate(
        zip(training.states, training.labels, strict=True), start=1
    ):
        # The copied numeric vector crosses the supervised label boundary first.
        readout.observe_label(np.array(state, dtype=np.float64, copy=True), int(label))
        if episode_index % config.checkpoint_interval == 0:
            digest_before = readout.parameter_digest()
            overall, per_delay = _evaluate_supervised_readout(readout, evaluation)
            if readout.parameter_digest() != digest_before:
                raise RuntimeError("supervised checkpoint evaluation mutated parameters")
            checkpoints.append(
                DiagnosticCheckpoint(
                    episode=episode_index,
                    overall=overall,
                    per_delay=per_delay,
                    parameter_digest=digest_before,
                )
            )

    overall, per_delay = _evaluate_supervised_readout(readout, evaluation)
    return SupervisedDiagnostic(
        overall=overall,
        per_delay=per_delay,
        checkpoints=tuple(checkpoints),
        parameter_digest=readout.parameter_digest(),
        supervised_passed=(
            overall.total == 200
            and overall.correct >= 180
            and all(
                count.total == 40 and count.correct >= 34
                for _, count in per_delay
            )
        ),
    )


def _evaluate_supervised_readout(
    readout: _DiagnosticSupervisedReadout,
    evaluation: _HiddenDataset,
) -> tuple[AccuracyCount, tuple[tuple[int, AccuracyCount], ...]]:
    """Score greedy choices outside the supervised readout without mutation."""
    choices = np.fromiter(
        (
            readout.select_greedy(np.array(state, dtype=np.float64, copy=True)).action_index
            for state in evaluation.states
        ),
        dtype=np.int64,
        count=evaluation.states.shape[0],
    )
    matches = choices == evaluation.labels
    per_delay: list[tuple[int, AccuracyCount]] = []
    for delay in range(1, 6):
        delay_mask = evaluation.delays == delay
        total = int(np.count_nonzero(delay_mask))
        if total == 0:
            raise ValueError(f"evaluation delay {delay} has zero samples")
        per_delay.append(
            (
                delay,
                AccuracyCount(int(np.count_nonzero(matches[delay_mask])), total),
            )
        )
    return AccuracyCount(int(np.count_nonzero(matches)), int(matches.size)), tuple(
        per_delay
    )


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


def _run_reward_trajectory(
    seed: int,
    fixtures: _DiagnosticFixtures,
    config: LearningDiagnosticsConfig,
) -> RewardTrajectoryDiagnostic:
    """Replay Phase 2B action/reward learning while observing outside it."""
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative Python integer")
    if not isinstance(fixtures, _DiagnosticFixtures):
        raise ValueError("fixtures must be _DiagnosticFixtures")  # noqa: TRY004
    if not isinstance(config, LearningDiagnosticsConfig):
        raise ValueError("config must be a LearningDiagnosticsConfig")  # noqa: TRY004
    _validate_reward_trajectory_fixtures(fixtures, config)

    task = DelayedCueTask()
    policy = _new_frozen_policy(seed, config)
    readout = RewardModulatedReadout(
        config.hidden_size,
        2,
        config.learning_rate,
        config.temperature,
    )
    action_rng = np.random.default_rng(
        np.random.SeedSequence([seed, 0x4143544E])
    )
    matrix_digests_before = _frozen_matrix_digests(policy)
    parameter_digest_before = readout.parameter_digest()

    actions: list[int] = []
    rewards: list[float] = []
    correct_action_probabilities: list[float] = []
    expected_bandit_to_supervised_norm_ratios: list[float] = []
    sampled_block_vectors: list[tuple[float, ...]] = []
    supervised_block_vectors: list[tuple[float, ...]] = []
    ratio_block_values: list[float] = []
    gradient_blocks: list[GradientBlockDiagnostic] = []
    checkpoints: list[RewardCheckpoint] = []
    zero_norm_block_count = 0
    final_block_correct = 0
    final_block_start = config.training_episodes - 10

    for episode_index, episode in enumerate(fixtures.training_episodes, start=1):
        hidden = _readonly_copy(
            _decision_hidden(
                policy,
                episode,
                reset_before_decision=False,
            ),
            np.float64,
        )
        decision = readout.select_for_training(hidden, (0, 1), action_rng)
        selected_action = _validated_action_index(decision.action_index)
        probabilities = _probability_snapshot(decision.probabilities)
        float_reward = float(task.reward(episode, selected_action))

        # Metadata is intentionally observed only after action and reward are fixed.
        label = int(fixtures.training.labels[episode_index - 1])
        correct_action_probabilities.append(float(probabilities[label]))
        supervised, sampled, ratio = _reward_gradient_measurements(
            hidden,
            probabilities,
            selected_action,
            label,
            float_reward,
        )
        supervised_block_vectors.append(tuple(supervised.tolist()))
        sampled_block_vectors.append(tuple(sampled.tolist()))
        ratio_block_values.append(ratio)
        expected_bandit_to_supervised_norm_ratios.append(ratio)

        readout.learn(float_reward)
        if readout.has_pending_feedback:
            raise RuntimeError("reward readout has pending feedback after learning")

        actions.append(selected_action)
        rewards.append(float_reward)
        if episode_index - 1 >= final_block_start:
            final_block_correct += int(selected_action == label)

        if len(sampled_block_vectors) == 10:
            gradient_block, has_zero_norm = _gradient_block_diagnostic(
                episode_index,
                sampled_block_vectors,
                supervised_block_vectors,
                ratio_block_values,
            )
            gradient_blocks.append(gradient_block)
            zero_norm_block_count += int(has_zero_norm)
            sampled_block_vectors = []
            supervised_block_vectors = []
            ratio_block_values = []

        if episode_index % config.checkpoint_interval == 0:
            parameter_digest = readout.parameter_digest()
            overall, per_delay = _evaluate_reward_readout(
                policy,
                readout,
                fixtures.evaluation_episodes,
                fixtures.evaluation,
            )
            if readout.parameter_digest() != parameter_digest:
                raise RuntimeError("reward checkpoint evaluation mutated parameters")
            checkpoints.append(
                RewardCheckpoint(
                    episode=episode_index,
                    overall=overall,
                    per_delay=per_delay,
                    correct_action_probabilities=tuple(correct_action_probabilities),
                    mean_correct_action_probability=float(
                        np.mean(correct_action_probabilities)
                    ),
                    percentile_10_correct_action_probability=float(
                        np.percentile(
                            correct_action_probabilities,
                            10,
                            method="linear",
                        )
                    ),
                    mean_expected_bandit_to_supervised_norm_ratio=float(
                        np.mean(expected_bandit_to_supervised_norm_ratios)
                    ),
                    gradient_blocks=tuple(gradient_blocks),
                    zero_norm_block_count=zero_norm_block_count,
                    parameter_digest=parameter_digest,
                )
            )

    if sampled_block_vectors or supervised_block_vectors or ratio_block_values:
        raise RuntimeError("reward trajectory must finish complete ten-case blocks")
    if not checkpoints or checkpoints[-1].episode != config.training_episodes:
        raise RuntimeError("reward trajectory is missing its final checkpoint")

    overall = checkpoints[-1].overall
    per_delay = checkpoints[-1].per_delay
    parameter_digest_after = readout.parameter_digest()
    matrix_digests_after = _frozen_matrix_digests(policy)
    return RewardTrajectoryDiagnostic(
        overall=overall,
        per_delay=per_delay,
        checkpoints=tuple(checkpoints),
        gradient_blocks=tuple(gradient_blocks),
        zero_norm_block_count=zero_norm_block_count,
        total_training_reward=int(sum(rewards)),
        final_block=AccuracyCount(final_block_correct, 10),
        parameter_digest_before=parameter_digest_before,
        parameter_digest_after=parameter_digest_after,
        matrix_digests_before=matrix_digests_before,
        matrix_digests_after=matrix_digests_after,
        training_choice_digest=hashlib.sha256(bytes(actions)).hexdigest(),
        training_reward_digest=_reward_digest(rewards),
        reward_passed=(
            overall == AccuracyCount(overall.correct, 200)
            and overall.correct >= 180
            and all(score == AccuracyCount(score.correct, 40) and score.correct >= 34 for _, score in per_delay)
        ),
    )


def _validate_reward_trajectory_fixtures(
    fixtures: _DiagnosticFixtures,
    config: LearningDiagnosticsConfig,
) -> None:
    if len(fixtures.training_episodes) != config.training_episodes:
        raise ValueError("training fixtures must match the configured episode count")
    expected_evaluation_rows = config.evaluation_blocks * 10
    if len(fixtures.evaluation_episodes) != expected_evaluation_rows:
        raise ValueError("evaluation fixtures must match the configured block count")
    if fixtures.training.states.shape != (
        config.training_episodes,
        config.hidden_size,
    ):
        raise ValueError("training states must match the configured episode count")
    if fixtures.evaluation.states.shape != (
        expected_evaluation_rows,
        config.hidden_size,
    ):
        raise ValueError("evaluation states must match the configured block count")


def _evaluate_reward_readout(
    policy: RecurrentPolicy,
    readout: RewardModulatedReadout,
    episodes: tuple[DelayedCueEpisode, ...],
    evaluation: _HiddenDataset,
) -> tuple[AccuracyCount, tuple[tuple[int, AccuracyCount], ...]]:
    """Score the readout without exposing fixture metadata until after choice."""
    correct = 0
    per_delay_correct = {delay: 0 for delay in range(1, 6)}
    per_delay_total = {delay: 0 for delay in range(1, 6)}
    for index, episode in enumerate(episodes):
        hidden = _readonly_copy(
            _decision_hidden(policy, episode, reset_before_decision=False),
            np.float64,
        )
        action = _validated_action_index(
            readout.select_greedy(hidden, (0, 1)).action_index
        )
        label = int(evaluation.labels[index])
        delay = int(evaluation.delays[index])
        matched = int(action == label)
        correct += matched
        per_delay_correct[delay] += matched
        per_delay_total[delay] += 1
    per_delay = tuple(
        (delay, AccuracyCount(per_delay_correct[delay], per_delay_total[delay]))
        for delay in range(1, 6)
    )
    return AccuracyCount(correct, len(episodes)), per_delay


def _reward_gradient_measurements(
    hidden: np.ndarray,
    probabilities: np.ndarray,
    action: int,
    label: int,
    reward: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Reconstruct gradients from decision snapshots, never learner state."""
    augmented_hidden = np.concatenate((hidden, [1.0]))
    one_hot = np.eye(2, dtype=np.float64)
    supervised = np.outer(one_hot[label] - probabilities, augmented_hidden).ravel()
    sampled = (
        reward * np.outer(one_hot[action] - probabilities, augmented_hidden)
    ).ravel()
    expected_bandit = sum(
        probabilities[candidate]
        * (1.0 if candidate == label else -1.0)
        * np.outer(one_hot[candidate] - probabilities, augmented_hidden)
        for candidate in (0, 1)
    ).ravel()
    supervised_norm = float(np.linalg.norm(supervised))
    if supervised_norm == 0.0:
        ratio = 0.0
    else:
        ratio = float(np.linalg.norm(expected_bandit) / supervised_norm)
    return supervised, sampled, ratio


def _gradient_block_diagnostic(
    episode: int,
    sampled_episode_gradients: list[tuple[float, ...]],
    supervised_episode_gradients: list[tuple[float, ...]],
    expected_bandit_to_supervised_norm_ratios: list[float],
) -> tuple[GradientBlockDiagnostic, bool]:
    sampled = np.asarray(sampled_episode_gradients, dtype=np.float64)
    supervised = np.asarray(supervised_episode_gradients, dtype=np.float64)
    sampled_sum = np.sum(sampled, axis=0)
    supervised_sum = np.sum(supervised, axis=0)
    sampled_norm = float(np.linalg.norm(sampled_sum))
    supervised_norm = float(np.linalg.norm(supervised_sum))
    has_zero_norm = sampled_norm == 0.0 or supervised_norm == 0.0
    cosine = (
        None
        if has_zero_norm
        else float(np.dot(sampled_sum, supervised_sum) / (sampled_norm * supervised_norm))
    )
    return (
        GradientBlockDiagnostic(
            episode=episode,
            sampled_episode_gradients=tuple(sampled_episode_gradients),
            supervised_episode_gradients=tuple(supervised_episode_gradients),
            expected_bandit_to_supervised_norm_ratios=tuple(
                expected_bandit_to_supervised_norm_ratios
            ),
            sampled_gradient=tuple(sampled_sum.tolist()),
            supervised_gradient=tuple(supervised_sum.tolist()),
            mean_expected_bandit_to_supervised_norm_ratio=float(
                np.mean(expected_bandit_to_supervised_norm_ratios)
            ),
            cosine=cosine,
        ),
        has_zero_norm,
    )


def _validated_action_index(value: object) -> int:
    if type(value) is not int or value not in (0, 1):
        raise ValueError("reward readout must choose action zero or one")
    return value


def _probability_snapshot(values: object) -> np.ndarray:
    try:
        probabilities = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("reward readout probabilities must be float64-compatible") from exc
    if (
        probabilities.shape != (2,)
        or not np.all(np.isfinite(probabilities))
        or np.any(probabilities < 0.0)
        or not np.isclose(float(np.sum(probabilities)), 1.0, rtol=0.0, atol=1e-12)
    ):
        raise ValueError("reward readout probabilities must be a finite distribution")
    return _readonly_copy(probabilities, np.float64)


def _frozen_matrix_digests(policy: object) -> tuple[str, str, str]:
    try:
        matrices = (
            policy._input_weights,
            policy._recurrent_weights,
            policy._output_weights,
        )
    except AttributeError as exc:
        raise ValueError("policy must provide frozen recurrent matrices") from exc
    return tuple(_array_digest(matrix) for matrix in matrices)


def _reward_digest(rewards: list[float]) -> str:
    values = np.ascontiguousarray(rewards, dtype=np.float64)
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


_PHASE_2B_EVIDENCE = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "experiments"
    / "phase-2b-failure.json"
)
_PHASE_2B_PORTABLE_RESULT_KEYS = (
    "all_reset_hidden_equal",
    "final_block",
    "normal_training_choice_digest",
    "normal_training_reward_digest",
    "passed",
    "per_delay",
    "post_training",
    "pre_training",
    "recurrent_choice_digest",
    "repeatable",
    "reset_choice_digest",
    "reset_per_delay",
    "seed",
    "shuffled_choice_digest",
    "shuffled_control",
    "shuffled_final_block",
    "shuffled_per_delay",
    "shuffled_total_training_reward",
    "shuffled_training_choice_digest",
    "shuffled_training_reward_digest",
    "state_reset",
    "total_training_reward",
)


def _match_phase_2b_evidence(seed: int, runtime_entry: object) -> bool:
    """Match one public Phase 2B result on its cross-platform stable fields."""
    if type(seed) is not int or seed < 0 or not isinstance(runtime_entry, dict):
        return False
    try:
        expected_payload = json.loads(_PHASE_2B_EVIDENCE.read_text(encoding="utf-8"))
        expected_entry = next(
            entry
            for entry in expected_payload["results"]
            if isinstance(entry, dict) and entry.get("seed") == seed
        )
    except (OSError, StopIteration, TypeError, json.JSONDecodeError, KeyError):
        return False
    return all(
        runtime_entry.get(key) == expected_entry.get(key)
        for key in _PHASE_2B_PORTABLE_RESULT_KEYS
    )


def run_learning_diagnostics(
    seed: int = 7,
    config: LearningDiagnosticsConfig | None = None,
) -> LearningDiagnosticsResult:
    """Run the three diagnostic branches twice with fresh complete objects."""
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative Python integer")
    if config is None:
        resolved_config = LearningDiagnosticsConfig()
    elif isinstance(config, LearningDiagnosticsConfig):
        resolved_config = config
    else:
        raise ValueError("config must be a LearningDiagnosticsConfig")

    first = _run_learning_diagnostics_once(seed, resolved_config)
    second = _run_learning_diagnostics_once(seed, resolved_config)
    repeatable = first == second
    local_integrity = (
        _same_run_matrix_integrity(first)
        and _same_run_matrix_integrity(second)
        and first.matrix_digests_before == second.matrix_digests_before
        and first.matrix_digests_after == second.matrix_digests_after
        and first.phase_2b_runtime_local_evidence is not None
        and first.phase_2b_runtime_local_evidence
        == second.phase_2b_runtime_local_evidence
    )
    protocol_match = (
        first.phase_2b_portable_evidence_match
        and second.phase_2b_portable_evidence_match
        and first.datasets_valid
        and second.datasets_valid
        and first.checkpoints_valid
        and second.checkpoints_valid
        and local_integrity
        and repeatable
    )
    classification = _classify(
        protocol_match,
        first.geometry.geometry_passed,
        first.supervised.supervised_passed,
        first.reward_trajectory.reward_passed,
    )
    return LearningDiagnosticsResult(
        seed=seed,
        config=resolved_config,
        geometry=first.geometry,
        supervised=first.supervised,
        reward_trajectory=first.reward_trajectory,
        classification=classification,
        matrix_digests_before=first.matrix_digests_before,
        matrix_digests_after=first.matrix_digests_after,
        phase_2b_portable_evidence_match=first.phase_2b_portable_evidence_match,
        diagnostic_valid=protocol_match,
        repeatable=repeatable,
        training_fixture_digest=first.training_fixture_digest,
        training_state_digest=first.training_state_digest,
        evaluation_fixture_digest=first.evaluation_fixture_digest,
        evaluation_state_digest=first.evaluation_state_digest,
    )


def _run_learning_diagnostics_once(
    seed: int,
    config: LearningDiagnosticsConfig,
) -> _CompleteLearningDiagnosticsRun:
    """Build a complete independent execution without retaining mutable objects."""
    fixtures = _build_diagnostic_fixtures(seed, config)
    geometry = _run_geometry(fixtures.training, fixtures.evaluation)
    supervised = _run_supervised_diagnostic(
        fixtures.training,
        fixtures.evaluation,
        config,
    )
    reward_trajectory = _run_reward_trajectory(seed, fixtures, config)
    runtime_entry = _phase_2b_runtime_entry(seed, config)
    runtime_local_evidence = _phase_2b_runtime_local_evidence(runtime_entry)
    portable_match = _match_phase_2b_evidence(seed, runtime_entry) and (
        _reward_trajectory_matches_runtime_entry(reward_trajectory, runtime_entry)
    )
    return _CompleteLearningDiagnosticsRun(
        geometry=geometry,
        supervised=supervised,
        reward_trajectory=reward_trajectory,
        matrix_digests_before=reward_trajectory.matrix_digests_before,
        matrix_digests_after=reward_trajectory.matrix_digests_after,
        phase_2b_runtime_local_evidence=runtime_local_evidence,
        phase_2b_portable_evidence_match=portable_match,
        datasets_valid=_diagnostic_datasets_are_valid(fixtures, config),
        checkpoints_valid=_diagnostic_checkpoints_are_valid(
            supervised,
            reward_trajectory,
            config,
        ),
        matrix_integrity=(
            reward_trajectory.matrix_digests_before
            == reward_trajectory.matrix_digests_after
        ),
        training_fixture_digest=fixtures.training.fixture_digest,
        training_state_digest=fixtures.training.state_digest,
        evaluation_fixture_digest=fixtures.evaluation.fixture_digest,
        evaluation_state_digest=fixtures.evaluation.state_digest,
    )


def _phase_2b_runtime_entry(
    seed: int,
    config: LearningDiagnosticsConfig,
) -> object:
    """Return the unchanged Phase 2B runtime entry for this diagnostic seed."""
    reward_config = RewardLearningConfig(
        hidden_size=config.hidden_size,
        recurrent_radius=config.recurrent_radius,
        learning_rate=config.learning_rate,
        temperature=config.temperature,
        training_episodes=config.training_episodes,
        evaluation_blocks=config.evaluation_blocks,
    )
    runtime = run_reward_learning_benchmark((seed,), reward_config)
    results = runtime.get("results")
    if not isinstance(results, list):
        return None
    return next(
        (
            entry
            for entry in results
            if isinstance(entry, dict) and entry.get("seed") == seed
        ),
        None,
    )


def _phase_2b_runtime_local_evidence(
    runtime_entry: object,
) -> _Phase2bRuntimeLocalEvidence | None:
    """Freeze and validate Phase 2B's environment-local floating evidence."""
    if not isinstance(runtime_entry, dict):
        return None
    try:
        matrix_controls = runtime_entry["matrix_controls"]
        if not isinstance(matrix_controls, dict):
            return None
        parameter_digests = (
            runtime_entry["initial_parameter_digest"],
            runtime_entry["normal_parameter_digest"],
            runtime_entry["shuffled_parameter_digest"],
        )
        if not all(isinstance(digest, str) for digest in parameter_digests):
            return None
        for digest in parameter_digests:
            _validate_digest(digest, "parameter_digest")
        matrix_digest_sets = tuple(
            tuple(matrix_controls[name])
            for name in (
                "normal_before",
                "normal_after",
                "shuffled_before",
                "shuffled_after",
            )
        )
        if any(len(digests) != 3 for digests in matrix_digest_sets):
            return None
        for digest_set in matrix_digest_sets:
            for digest in digest_set:
                _validate_digest(digest, "matrix_digest")
    except (KeyError, TypeError, ValueError):
        return None

    (
        normal_before,
        normal_after,
        shuffled_before,
        shuffled_after,
    ) = matrix_digest_sets
    if (
        normal_before != normal_after
        or shuffled_before != shuffled_after
        or parameter_digests[0] == parameter_digests[1]
    ):
        return None
    return _Phase2bRuntimeLocalEvidence(
        initial_parameter_digest=parameter_digests[0],
        normal_parameter_digest=parameter_digests[1],
        shuffled_parameter_digest=parameter_digests[2],
        normal_matrix_digests_before=normal_before,
        normal_matrix_digests_after=normal_after,
        shuffled_matrix_digests_before=shuffled_before,
        shuffled_matrix_digests_after=shuffled_after,
    )


def _reward_trajectory_matches_runtime_entry(
    reward: RewardTrajectoryDiagnostic,
    runtime_entry: object,
) -> bool:
    """Compare the trace to Phase 2B's portable behavior, not float digests."""
    if not isinstance(runtime_entry, dict):
        return False
    try:
        post_training = runtime_entry["post_training"]
        runtime_per_delay = runtime_entry["per_delay"]
        if not isinstance(post_training, dict) or not isinstance(runtime_per_delay, list):
            return False
        expected_overall = AccuracyCount(
            int(post_training["correct"]),
            int(post_training["total"]),
        )
        expected_per_delay = tuple(
            (
                int(item["delay"]),
                AccuracyCount(int(item["correct"]), int(item["total"])),
            )
            for item in runtime_per_delay
            if isinstance(item, dict)
        )
        return (
            reward.overall == expected_overall
            and reward.per_delay == expected_per_delay
            and reward.total_training_reward == runtime_entry["total_training_reward"]
            and reward.training_choice_digest
            == runtime_entry["normal_training_choice_digest"]
            and reward.training_reward_digest
            == runtime_entry["normal_training_reward_digest"]
        )
    except (KeyError, TypeError, ValueError):
        return False


def _diagnostic_datasets_are_valid(
    fixtures: _DiagnosticFixtures,
    config: LearningDiagnosticsConfig,
) -> bool:
    """Recheck the immutable dataset schema used by every diagnostic branch."""
    expected_rows = (
        (fixtures.training, config.training_episodes),
        (fixtures.evaluation, config.evaluation_blocks * 10),
    )
    for dataset, row_count in expected_rows:
        if (
            dataset.states.shape != (row_count, config.hidden_size)
            or dataset.labels.shape != (row_count,)
            or dataset.delays.shape != (row_count,)
            or dataset.states.flags.writeable
            or dataset.labels.flags.writeable
            or dataset.delays.flags.writeable
            or dataset.state_digest != _array_digest(dataset.states)
        ):
            return False
        try:
            _validate_digest(dataset.fixture_digest, "fixture_digest")
            _validate_digest(dataset.state_digest, "state_digest")
        except ValueError:
            return False
    return (
        tuple(np.bincount(fixtures.training.labels, minlength=2))
        == (config.training_episodes // 2, config.training_episodes // 2)
        and tuple(np.bincount(fixtures.evaluation.labels, minlength=2))
        == (config.evaluation_blocks * 5, config.evaluation_blocks * 5)
        and tuple(np.bincount(fixtures.evaluation.delays, minlength=6)[1:])
        == (config.evaluation_blocks * 2,) * 5
    )


def _diagnostic_checkpoints_are_valid(
    supervised: SupervisedDiagnostic,
    reward: RewardTrajectoryDiagnostic,
    config: LearningDiagnosticsConfig,
) -> bool:
    """Ensure scheduled read-only checkpoints cover the whole frozen run."""
    expected_episodes = tuple(
        range(config.checkpoint_interval, config.training_episodes + 1, config.checkpoint_interval)
    )
    if (
        tuple(checkpoint.episode for checkpoint in supervised.checkpoints)
        != expected_episodes
        or tuple(checkpoint.episode for checkpoint in reward.checkpoints)
        != expected_episodes
        or not supervised.checkpoints
        or not reward.checkpoints
        or supervised.checkpoints[-1].parameter_digest != supervised.parameter_digest
        or reward.checkpoints[-1].parameter_digest != reward.parameter_digest_after
        or reward.checkpoints[-1].overall != reward.overall
        or reward.checkpoints[-1].per_delay != reward.per_delay
        or len(reward.gradient_blocks) != config.training_episodes // 10
        or tuple(block.episode for block in reward.gradient_blocks)
        != tuple(range(10, config.training_episodes + 1, 10))
    ):
        return False
    return all(
        checkpoint.overall.total == config.evaluation_blocks * 10
        and len(checkpoint.per_delay) == 5
        and all(score.total == config.evaluation_blocks * 2 for _, score in checkpoint.per_delay)
        for checkpoint in (*supervised.checkpoints, *reward.checkpoints)
    )


def _same_run_matrix_integrity(run: _CompleteLearningDiagnosticsRun) -> bool:
    """Confirm one run neither mutates nor fabricates its frozen matrix digests."""
    if not run.matrix_integrity or run.matrix_digests_before != run.matrix_digests_after:
        return False
    try:
        for digest in (*run.matrix_digests_before, *run.matrix_digests_after):
            _validate_digest(digest, "matrix_digest")
    except ValueError:
        return False
    return True


def _classify(
    protocol_match: bool,
    geometry_passed: bool,
    supervised_passed: bool,
    reward_passed: bool,
) -> str:
    """Select exactly one diagnostic outcome in the frozen priority order."""
    if not protocol_match:
        return "PROTOCOL_MISMATCH"
    if not geometry_passed:
        return "REPRESENTATION_FAILURE"
    if not supervised_passed:
        return "ONLINE_OPTIMIZATION_FAILURE"
    if not reward_passed:
        return "REWARD_CREDIT_FAILURE"
    return "NO_FAILURE_REPRODUCED"


def run_learning_diagnostics_benchmark(
    seeds: Sequence[int] = (7, 17, 29),
    config: LearningDiagnosticsConfig | None = None,
) -> dict[str, object]:
    """Measure the complete deterministic Phase 2C diagnostic benchmark."""
    resolved_seeds = _validated_benchmark_seeds(seeds)
    resolved_config = _validated_benchmark_config(config)
    results = [
        _diagnostics_benchmark_result(run_learning_diagnostics(seed, resolved_config))
        for seed in resolved_seeds
    ]
    classifications = [result["classification"] for result in results]
    if not all(isinstance(classification, str) for classification in classifications):
        raise ValueError("diagnostic result has an invalid classification")
    counts = Counter(classifications)
    payload: dict[str, object] = {
        "all_valid": all(
            result["diagnostic_valid"] is True and result["repeatable"] is True
            for result in results
        ),
        "classification_counts": dict(sorted(counts.items())),
        "config": _diagnostic_json_value(resolved_config),
        "phase": "2C",
        "phase_2b_evidence_digest": hashlib.sha256(
            _PHASE_2B_EVIDENCE.read_bytes()
        ).hexdigest(),
        "results": results,
        "seeds": list(resolved_seeds),
    }
    _validate_diagnostics_benchmark_payload(payload)
    return payload


def _validated_benchmark_seeds(seeds: object) -> tuple[int, ...]:
    """Validate ordered benchmark seeds before any policy can be constructed."""
    if isinstance(seeds, (str, bytes)) or not isinstance(seeds, Sequence):
        raise ValueError(  # noqa: TRY004
            "seeds must be a non-empty sequence of unique non-negative integers"
        )
    resolved = tuple(seeds)
    if not resolved:
        raise ValueError("seeds must be a non-empty sequence of unique non-negative integers")
    if any(type(seed) is not int or seed < 0 for seed in resolved):
        raise ValueError("seeds must be a non-empty sequence of unique non-negative integers")
    if len(set(resolved)) != len(resolved):
        raise ValueError("seeds must not contain duplicates")
    return resolved


def _validated_benchmark_config(
    config: LearningDiagnosticsConfig | None,
) -> LearningDiagnosticsConfig:
    """Resolve the only configuration accepted by the public benchmark."""
    if config is None:
        return LearningDiagnosticsConfig()
    if not isinstance(config, LearningDiagnosticsConfig):
        raise ValueError("config must be a LearningDiagnosticsConfig")  # noqa: TRY004
    return config


def _diagnostic_json_value(value: object) -> object:
    """Recursively project immutable diagnostic records to JSON primitives."""
    if is_dataclass(value):
        return {
            field.name: _diagnostic_json_value(getattr(value, field.name))
            for field in fields(value)
        }
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("diagnostic JSON values must be finite")
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        resolved = float(value)
        if not math.isfinite(resolved):
            raise ValueError("diagnostic JSON values must be finite")
        return resolved
    if isinstance(value, tuple | list):
        return [_diagnostic_json_value(item) for item in value]
    if isinstance(value, dict):
        if not all(type(key) is str for key in value):
            raise ValueError("diagnostic JSON object keys must be strings")
        return {key: _diagnostic_json_value(item) for key, item in value.items()}
    raise ValueError(f"diagnostic value is not JSON serializable: {type(value).__name__}")


def _diagnostics_benchmark_result(
    result: LearningDiagnosticsResult,
) -> dict[str, object]:
    """Project a complete result to stable evidence without duplicate raw traces."""
    return {
        "seed": result.seed,
        "config": _diagnostic_json_value(result.config),
        "geometry": _geometry_benchmark_value(result.geometry),
        "supervised": _supervised_benchmark_value(result.supervised),
        "reward_trajectory": _reward_trajectory_benchmark_value(result.reward_trajectory),
        "classification": result.classification,
        "matrix_digests_before": list(result.matrix_digests_before),
        "matrix_digests_after": list(result.matrix_digests_after),
        "phase_2b_portable_evidence_match": result.phase_2b_portable_evidence_match,
        "diagnostic_valid": result.diagnostic_valid,
        "repeatable": result.repeatable,
        "training_fixture_digest": result.training_fixture_digest,
        "training_state_digest": result.training_state_digest,
        "evaluation_fixture_digest": result.evaluation_fixture_digest,
        "evaluation_state_digest": result.evaluation_state_digest,
    }


def _geometry_benchmark_value(geometry: GeometryDiagnostic) -> dict[str, object]:
    """Serialize all geometry counts, margins, and its fitted-probe digest."""
    return {
        "training": _accuracy_benchmark_value(geometry.training),
        "evaluation": _accuracy_benchmark_value(geometry.evaluation),
        "per_delay": _per_delay_benchmark_value(geometry.per_delay),
        "training_margins": _margin_benchmark_value(geometry.training_margins),
        "evaluation_margins": _margin_benchmark_value(geometry.evaluation_margins),
        "per_delay_margins": [
            [delay, _margin_benchmark_value(summary)]
            for delay, summary in geometry.per_delay_margins
        ],
        "delay_five_to_one_median_ratio": geometry.delay_five_to_one_median_ratio,
        "probe_digest": geometry.probe_digest,
        "geometry_passed": geometry.geometry_passed,
    }


def _supervised_benchmark_value(supervised: SupervisedDiagnostic) -> dict[str, object]:
    """Serialize supervised counts and every scheduled parameter checkpoint."""
    return {
        "overall": _accuracy_benchmark_value(supervised.overall),
        "per_delay": _per_delay_benchmark_value(supervised.per_delay),
        "checkpoints": [
            {
                "episode": checkpoint.episode,
                "overall": _accuracy_benchmark_value(checkpoint.overall),
                "per_delay": _per_delay_benchmark_value(checkpoint.per_delay),
                "parameter_digest": checkpoint.parameter_digest,
            }
            for checkpoint in supervised.checkpoints
        ],
        "parameter_digest": supervised.parameter_digest,
        "supervised_passed": supervised.supervised_passed,
    }


def _reward_trajectory_benchmark_value(
    reward: RewardTrajectoryDiagnostic,
) -> dict[str, object]:
    """Serialize reward summaries while keeping raw gradient vectors private."""
    return {
        "overall": _accuracy_benchmark_value(reward.overall),
        "per_delay": _per_delay_benchmark_value(reward.per_delay),
        "checkpoints": [
            {
                "episode": checkpoint.episode,
                "overall": _accuracy_benchmark_value(checkpoint.overall),
                "per_delay": _per_delay_benchmark_value(checkpoint.per_delay),
                "mean_correct_action_probability": (
                    checkpoint.mean_correct_action_probability
                ),
                "percentile_10_correct_action_probability": (
                    checkpoint.percentile_10_correct_action_probability
                ),
                "mean_expected_bandit_to_supervised_norm_ratio": (
                    checkpoint.mean_expected_bandit_to_supervised_norm_ratio
                ),
                "zero_norm_block_count": checkpoint.zero_norm_block_count,
                "parameter_digest": checkpoint.parameter_digest,
            }
            for checkpoint in reward.checkpoints
        ],
        "gradient_blocks": [
            {
                "episode": block.episode,
                "mean_expected_bandit_to_supervised_norm_ratio": (
                    block.mean_expected_bandit_to_supervised_norm_ratio
                ),
                "cosine": block.cosine,
            }
            for block in reward.gradient_blocks
        ],
        "zero_norm_block_count": reward.zero_norm_block_count,
        "total_training_reward": reward.total_training_reward,
        "final_block": _accuracy_benchmark_value(reward.final_block),
        "parameter_digest_before": reward.parameter_digest_before,
        "parameter_digest_after": reward.parameter_digest_after,
        "matrix_digests_before": list(reward.matrix_digests_before),
        "matrix_digests_after": list(reward.matrix_digests_after),
        "training_choice_digest": reward.training_choice_digest,
        "training_reward_digest": reward.training_reward_digest,
        "reward_passed": reward.reward_passed,
    }


def _accuracy_benchmark_value(count: AccuracyCount) -> dict[str, int]:
    """Serialize literal correctness counts without derived rounded accuracy."""
    return {"correct": count.correct, "total": count.total}


def _per_delay_benchmark_value(
    per_delay: tuple[tuple[int, AccuracyCount], ...],
) -> list[list[object]]:
    """Serialize ordered delay/count pairs with their literal integer counts."""
    return [[delay, _accuracy_benchmark_value(count)] for delay, count in per_delay]


def _margin_benchmark_value(summary: MarginSummary) -> dict[str, float]:
    """Serialize each unrounded measured margin statistic."""
    return {
        "minimum": summary.minimum,
        "percentile_10": summary.percentile_10,
        "median": summary.median,
    }


def _validate_diagnostics_benchmark_payload(payload: object) -> None:
    """Reject malformed benchmark data before it becomes measured evidence."""
    if not isinstance(payload, dict) or set(payload) != {
        "all_valid",
        "classification_counts",
        "config",
        "phase",
        "phase_2b_evidence_digest",
        "results",
        "seeds",
    }:
        raise ValueError("diagnostics benchmark payload has an invalid top-level schema")
    _validate_diagnostic_json_primitives(payload)

    seeds = payload["seeds"]
    results = payload["results"]
    config = payload["config"]
    counts = payload["classification_counts"]
    if (
        payload["phase"] != "2C"
        or type(payload["all_valid"]) is not bool
        or not isinstance(seeds, list)
        or not isinstance(results, list)
        or not isinstance(config, dict)
        or not isinstance(counts, dict)
    ):
        raise ValueError("diagnostics benchmark payload has invalid required values")
    if (
        not seeds
        or any(type(seed) is not int or seed < 0 for seed in seeds)
        or len(set(seeds)) != len(seeds)
        or len(results) != len(seeds)
    ):
        raise ValueError("diagnostics benchmark payload has invalid seeds")
    expected_config_keys = {field.name for field in fields(LearningDiagnosticsConfig)}
    if set(config) != expected_config_keys:
        raise ValueError("diagnostics benchmark payload has an invalid configuration")
    try:
        resolved_config = LearningDiagnosticsConfig(**config)
    except (TypeError, ValueError) as exc:
        raise ValueError("diagnostics benchmark payload has an invalid configuration") from exc

    expected_result_keys = {field.name for field in fields(LearningDiagnosticsResult)}
    classifications: list[str] = []
    for seed, result in zip(seeds, results, strict=True):
        if not isinstance(result, dict) or set(result) != expected_result_keys:
            raise ValueError("diagnostics benchmark payload has an invalid result")
        if result["seed"] != seed or result["config"] != config:
            raise ValueError("diagnostics benchmark payload has mismatched result metadata")
        _validate_benchmark_result_summary(result, resolved_config)
        if (
            type(result["diagnostic_valid"]) is not bool
            or type(result["repeatable"]) is not bool
            or type(result["phase_2b_portable_evidence_match"]) is not bool
            or result["classification"]
            not in {
                "PROTOCOL_MISMATCH",
                "REPRESENTATION_FAILURE",
                "ONLINE_OPTIMIZATION_FAILURE",
                "REWARD_CREDIT_FAILURE",
                "NO_FAILURE_REPRODUCED",
            }
        ):
            raise ValueError("diagnostics benchmark payload has invalid result status")
        classifications.append(result["classification"])

    expected_counts = dict(sorted(Counter(classifications).items()))
    if counts != expected_counts or any(type(count) is not int for count in counts.values()):
        raise ValueError("diagnostics benchmark payload has invalid classification counts")
    if payload["all_valid"] is not all(
        result["diagnostic_valid"] is True and result["repeatable"] is True
        for result in results
    ):
        raise ValueError("diagnostics benchmark payload has an invalid validity gate")
    digest = payload["phase_2b_evidence_digest"]
    if (
        type(digest) is not str
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or digest != hashlib.sha256(_PHASE_2B_EVIDENCE.read_bytes()).hexdigest()
    ):
        raise ValueError("diagnostics benchmark payload has an invalid Phase 2B digest")


def _validate_benchmark_result_summary(
    result: dict[str, object],
    config: LearningDiagnosticsConfig,
) -> None:
    """Validate the explicit compact result schema before emitting evidence."""
    for key in (
        "training_fixture_digest",
        "training_state_digest",
        "evaluation_fixture_digest",
        "evaluation_state_digest",
    ):
        _validate_digest(result[key], key)
    _validate_benchmark_digest_list(result["matrix_digests_before"], "matrix_digests_before")
    _validate_benchmark_digest_list(result["matrix_digests_after"], "matrix_digests_after")

    geometry = _benchmark_schema_object(
        result["geometry"],
        {
            "training",
            "evaluation",
            "per_delay",
            "training_margins",
            "evaluation_margins",
            "per_delay_margins",
            "delay_five_to_one_median_ratio",
            "probe_digest",
            "geometry_passed",
        },
        "geometry",
    )
    _validate_benchmark_accuracy(geometry["training"], "geometry.training")
    _validate_benchmark_accuracy(geometry["evaluation"], "geometry.evaluation")
    _validate_benchmark_per_delay(geometry["per_delay"], "geometry.per_delay")
    _validate_benchmark_margin(geometry["training_margins"], "geometry.training_margins")
    _validate_benchmark_margin(geometry["evaluation_margins"], "geometry.evaluation_margins")
    _validate_benchmark_margin_per_delay(geometry["per_delay_margins"])
    if (
        type(geometry["delay_five_to_one_median_ratio"]) is not float
        or type(geometry["geometry_passed"]) is not bool
    ):
        raise ValueError("diagnostics benchmark payload has invalid geometry summaries")
    _validate_digest(geometry["probe_digest"], "geometry.probe_digest")

    supervised = _benchmark_schema_object(
        result["supervised"],
        {
            "overall",
            "per_delay",
            "checkpoints",
            "parameter_digest",
            "supervised_passed",
        },
        "supervised",
    )
    _validate_benchmark_accuracy(supervised["overall"], "supervised.overall")
    _validate_benchmark_per_delay(supervised["per_delay"], "supervised.per_delay")
    _validate_supervised_benchmark_checkpoints(supervised["checkpoints"], config)
    _validate_digest(supervised["parameter_digest"], "supervised.parameter_digest")
    if type(supervised["supervised_passed"]) is not bool:
        raise ValueError("diagnostics benchmark payload has an invalid supervised gate")

    reward = _benchmark_schema_object(
        result["reward_trajectory"],
        {
            "overall",
            "per_delay",
            "checkpoints",
            "gradient_blocks",
            "zero_norm_block_count",
            "total_training_reward",
            "final_block",
            "parameter_digest_before",
            "parameter_digest_after",
            "matrix_digests_before",
            "matrix_digests_after",
            "training_choice_digest",
            "training_reward_digest",
            "reward_passed",
        },
        "reward_trajectory",
    )
    _validate_benchmark_accuracy(reward["overall"], "reward.overall")
    _validate_benchmark_per_delay(reward["per_delay"], "reward.per_delay")
    _validate_reward_benchmark_checkpoints(reward["checkpoints"], config)
    _validate_gradient_block_summaries(reward["gradient_blocks"], config)
    _validate_benchmark_accuracy(reward["final_block"], "reward.final_block")
    _validate_benchmark_digest_list(reward["matrix_digests_before"], "reward.matrix_before")
    _validate_benchmark_digest_list(reward["matrix_digests_after"], "reward.matrix_after")
    for key in (
        "parameter_digest_before",
        "parameter_digest_after",
        "training_choice_digest",
        "training_reward_digest",
    ):
        _validate_digest(reward[key], f"reward.{key}")
    if (
        type(reward["zero_norm_block_count"]) is not int
        or reward["zero_norm_block_count"] < 0
        or type(reward["total_training_reward"]) is not int
        or type(reward["reward_passed"]) is not bool
    ):
        raise ValueError("diagnostics benchmark payload has invalid reward summaries")


def _benchmark_schema_object(
    value: object,
    keys: set[str],
    name: str,
) -> dict[str, object]:
    """Return a summary object only when its fields exactly match the schema."""
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"diagnostics benchmark payload has an invalid {name} schema")
    return value


def _validate_benchmark_accuracy(value: object, name: str) -> None:
    """Validate a compact literal correctness-count object."""
    count = _benchmark_schema_object(value, {"correct", "total"}, name)
    correct = count["correct"]
    total = count["total"]
    if (
        type(correct) is not int
        or type(total) is not int
        or total <= 0
        or not 0 <= correct <= total
    ):
        raise ValueError(f"diagnostics benchmark payload has an invalid {name} count")


def _validate_benchmark_per_delay(value: object, name: str) -> None:
    """Validate the complete ordered one-through-five delay-count series."""
    if not isinstance(value, list) or len(value) != 5:
        raise ValueError(f"diagnostics benchmark payload has an invalid {name}")
    for delay, pair in enumerate(value, start=1):
        if not isinstance(pair, list) or len(pair) != 2 or pair[0] != delay:
            raise ValueError(f"diagnostics benchmark payload has an invalid {name}")
        _validate_benchmark_accuracy(pair[1], f"{name}[{delay}]")


def _validate_benchmark_margin(value: object, name: str) -> None:
    """Validate one unrounded finite margin-summary object."""
    summary = _benchmark_schema_object(value, {"minimum", "percentile_10", "median"}, name)
    if any(type(summary[key]) is not float for key in summary):
        raise ValueError(f"diagnostics benchmark payload has an invalid {name}")


def _validate_benchmark_margin_per_delay(value: object) -> None:
    """Validate one margin summary for every frozen delay."""
    if not isinstance(value, list) or len(value) != 5:
        raise ValueError("diagnostics benchmark payload has invalid per-delay margins")
    for delay, pair in enumerate(value, start=1):
        if not isinstance(pair, list) or len(pair) != 2 or pair[0] != delay:
            raise ValueError("diagnostics benchmark payload has invalid per-delay margins")
        _validate_benchmark_margin(pair[1], f"per_delay_margins[{delay}]")


def _validate_supervised_benchmark_checkpoints(
    value: object,
    config: LearningDiagnosticsConfig,
) -> None:
    """Validate scheduled supervised count/digest checkpoint summaries."""
    checkpoints = _validate_benchmark_checkpoint_episodes(value, config, "supervised")
    for checkpoint in checkpoints:
        summary = _benchmark_schema_object(
            checkpoint,
            {"episode", "overall", "per_delay", "parameter_digest"},
            "supervised checkpoint",
        )
        _validate_benchmark_accuracy(summary["overall"], "supervised checkpoint overall")
        _validate_benchmark_per_delay(summary["per_delay"], "supervised checkpoint per_delay")
        _validate_digest(summary["parameter_digest"], "supervised checkpoint parameter_digest")


def _validate_reward_benchmark_checkpoints(
    value: object,
    config: LearningDiagnosticsConfig,
) -> None:
    """Validate reward checkpoint summaries without cumulative raw vectors."""
    checkpoints = _validate_benchmark_checkpoint_episodes(value, config, "reward")
    for checkpoint in checkpoints:
        summary = _benchmark_schema_object(
            checkpoint,
            {
                "episode",
                "overall",
                "per_delay",
                "mean_correct_action_probability",
                "percentile_10_correct_action_probability",
                "mean_expected_bandit_to_supervised_norm_ratio",
                "zero_norm_block_count",
                "parameter_digest",
            },
            "reward checkpoint",
        )
        _validate_benchmark_accuracy(summary["overall"], "reward checkpoint overall")
        _validate_benchmark_per_delay(summary["per_delay"], "reward checkpoint per_delay")
        if (
            type(summary["mean_correct_action_probability"]) is not float
            or type(summary["percentile_10_correct_action_probability"]) is not float
            or type(summary["mean_expected_bandit_to_supervised_norm_ratio"]) is not float
            or type(summary["zero_norm_block_count"]) is not int
            or summary["zero_norm_block_count"] < 0
        ):
            raise ValueError("diagnostics benchmark payload has invalid reward checkpoints")
        _validate_digest(summary["parameter_digest"], "reward checkpoint parameter_digest")


def _validate_benchmark_checkpoint_episodes(
    value: object,
    config: LearningDiagnosticsConfig,
    name: str,
) -> list[object]:
    """Require every hundred-episode checkpoint through the frozen final step."""
    expected = list(
        range(config.checkpoint_interval, config.training_episodes + 1, config.checkpoint_interval)
    )
    if not isinstance(value, list) or [item.get("episode") if isinstance(item, dict) else None for item in value] != expected:
        raise ValueError(f"diagnostics benchmark payload has invalid {name} checkpoints")
    return value


def _validate_gradient_block_summaries(
    value: object,
    config: LearningDiagnosticsConfig,
) -> None:
    """Validate per-block scale/cosine evidence without raw gradient arrays."""
    expected = list(range(10, config.training_episodes + 1, 10))
    if not isinstance(value, list) or [item.get("episode") if isinstance(item, dict) else None for item in value] != expected:
        raise ValueError("diagnostics benchmark payload has invalid gradient block episodes")
    for block in value:
        summary = _benchmark_schema_object(
            block,
            {"episode", "mean_expected_bandit_to_supervised_norm_ratio", "cosine"},
            "gradient block",
        )
        if (
            type(summary["mean_expected_bandit_to_supervised_norm_ratio"]) is not float
            or (summary["cosine"] is not None and type(summary["cosine"]) is not float)
        ):
            raise ValueError("diagnostics benchmark payload has invalid gradient block summaries")


def _validate_benchmark_digest_list(value: object, name: str) -> None:
    """Validate the fixed input/recurrent/legacy-output digest triple."""
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"diagnostics benchmark payload has an invalid {name}")
    for digest in value:
        _validate_digest(digest, name)


def _validate_diagnostic_json_primitives(value: object) -> None:
    """Enforce recursively finite primitive-only data for stable JSON evidence."""
    if value is None or type(value) in (bool, int, str):
        return
    if type(value) is float:
        if math.isfinite(value):
            return
        raise ValueError("diagnostics benchmark payload has non-finite JSON")
    if isinstance(value, list):
        for item in value:
            _validate_diagnostic_json_primitives(item)
        return
    if isinstance(value, dict):
        if not all(type(key) is str for key in value):
            raise ValueError("diagnostics benchmark payload has non-string JSON keys")
        for item in value.values():
            _validate_diagnostic_json_primitives(item)
        return
    raise ValueError("diagnostics benchmark payload is not primitive-only")


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
