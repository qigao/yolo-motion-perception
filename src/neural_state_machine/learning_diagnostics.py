"""Immutable hidden-state datasets for Phase 2C diagnostics.

This module deliberately separates numeric recurrent-state collection from the
diagnostic metadata that labels those states.  The frozen policy sees only task
stimuli; labels and delays are associated after every hidden row is copied.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
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
class _CompleteLearningDiagnosticsRun:
    """Private complete-run record that contains only equality-safe values."""

    geometry: GeometryDiagnostic
    supervised: SupervisedDiagnostic
    reward_trajectory: RewardTrajectoryDiagnostic
    matrix_digests_before: tuple[str, str, str]
    matrix_digests_after: tuple[str, str, str]
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
    portable_match = _match_phase_2b_evidence(seed, runtime_entry) and (
        _reward_trajectory_matches_runtime_entry(reward_trajectory, runtime_entry)
    )
    return _CompleteLearningDiagnosticsRun(
        geometry=geometry,
        supervised=supervised,
        reward_trajectory=reward_trajectory,
        matrix_digests_before=reward_trajectory.matrix_digests_before,
        matrix_digests_after=reward_trajectory.matrix_digests_after,
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


def run_learning_diagnostics_benchmark(*args: object, **kwargs: object) -> object:
    """Reserved public benchmark entry point implemented in the next task."""
    del args, kwargs
    raise NotImplementedError("run_learning_diagnostics_benchmark is not available yet")


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
