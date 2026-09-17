from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np

from .memory_probe import MemoryProbeConfig, fit_linear_probe, run_memory_probe
from .memory_task import Cue, DelayedCueEpisode, DelayedCueTask
from .policy import RecurrentPolicy
from .r1_e1_probe import fit_binary_ridge, prediction_digest
from .r1_e1_reservoir import Reservoir, ReservoirSpec, build_reservoir


MEMORY_DELAYS = (1, 2, 5, 10, 20, 40, 80)
_TRAIN_PAIRS_PER_DELAY = 200
_EVAL_PAIRS_PER_DELAY = 20
_E1A_TRAIN_TAG = 0x45314154
_E1A_EVAL_TAG = 0x45314145
_REGULARIZATION = 1e-6
_HISTORICAL_PHASE2A_SEEDS = frozenset((7, 17, 29))


@dataclass(frozen=True)
class MemoryAccuracy:
    correct: int
    total: int

    def __post_init__(self) -> None:
        if type(self.correct) is not int or type(self.total) is not int:
            raise ValueError("correct and total must be Python integers")
        if self.total <= 0 or not 0 <= self.correct <= self.total:
            raise ValueError("accuracy counts must satisfy 0 <= correct <= total")

    @property
    def accuracy(self) -> float:
        return self.correct / self.total


@dataclass(frozen=True)
class MemoryEpisode:
    label: int
    delay: int
    pair_id: int
    cue_stimulus: np.ndarray
    delay_stimuli: tuple[np.ndarray, ...]
    decision_stimulus: np.ndarray

    def __post_init__(self) -> None:
        if type(self.label) is not int or self.label not in (0, 1):
            raise ValueError("label must be zero or one")
        if type(self.delay) is not int or self.delay not in MEMORY_DELAYS:
            raise ValueError("delay must be registered for E1-A")
        if type(self.pair_id) is not int or self.pair_id < 0:
            raise ValueError("pair_id must be a non-negative Python integer")
        cue = _validated_frame(self.cue_stimulus)
        decision = _validated_frame(self.decision_stimulus)
        delays = tuple(_validated_frame(frame) for frame in self.delay_stimuli)
        if len(delays) != self.delay:
            raise ValueError("delay_stimuli must contain exactly delay frames")
        expected_cue = np.zeros(4, dtype=np.float64)
        expected_cue[self.label] = 1.0
        if not np.array_equal(cue, expected_cue):
            raise ValueError("cue_stimulus must encode the label exactly")
        if not np.array_equal(decision, np.array([0.0, 0.0, 0.0, 1.0])):
            raise ValueError("decision_stimulus must be the common decision frame")
        for frame in delays:
            if not np.all(frame[[0, 1, 3]] == 0.0) or not -0.25 <= frame[2] <= 0.25:
                raise ValueError("delay frames must contain only the bounded nuisance channel")
        object.__setattr__(self, "cue_stimulus", _readonly_copy(cue))
        object.__setattr__(self, "delay_stimuli", tuple(_readonly_copy(frame) for frame in delays))
        object.__setattr__(self, "decision_stimulus", _readonly_copy(decision))


@dataclass(frozen=True)
class MemoryFixtures:
    training: tuple[MemoryEpisode, ...]
    evaluation: tuple[MemoryEpisode, ...]
    training_digest: str
    evaluation_digest: str
    combined_digest: str


@dataclass(frozen=True)
class MemoryArmResult:
    training: MemoryAccuracy
    evaluation: MemoryAccuracy
    per_delay: tuple[tuple[int, MemoryAccuracy], ...]
    reset: MemoryAccuracy
    reset_per_delay: tuple[tuple[int, MemoryAccuracy], ...]
    macro_accuracy: float
    memory_horizon: int | None
    degradation_from_delay1: tuple[tuple[int, float], ...]
    all_reset_pairs_equal: bool
    valid: bool
    parameter_digest_before: str
    parameter_digest_after: str
    fixture_digest: str
    coefficient_digest: str
    prediction_digest: str
    reset_prediction_digest: str


@dataclass(frozen=True)
class Phase2ACompatibility:
    seed: int
    hidden_recurrence_matches: bool
    training_matches: bool
    evaluation_matches: bool
    per_delay_matches: bool
    reset_matches: bool
    probe_digest_matches: bool
    prediction_digest_matches: bool
    reset_prediction_digest_matches: bool
    legacy_acceptance_required: bool
    legacy_acceptance_passed: bool
    passed: bool


def build_memory_fixtures(seed: int) -> MemoryFixtures:
    validated_seed = _validated_seed(seed)
    train_rng = np.random.default_rng(
        np.random.SeedSequence([validated_seed, _E1A_TRAIN_TAG])
    )
    eval_rng = np.random.default_rng(
        np.random.SeedSequence([validated_seed, _E1A_EVAL_TAG])
    )
    training = _build_paired_fixtures(train_rng, _TRAIN_PAIRS_PER_DELAY)
    evaluation = _build_paired_fixtures(eval_rng, _EVAL_PAIRS_PER_DELAY)
    train_digest = _fixture_digest(training)
    eval_digest = _fixture_digest(evaluation)
    digest = hashlib.sha256()
    digest.update(b"r1-e1-a-fixtures-v1\0")
    digest.update(train_digest.encode("ascii"))
    digest.update(eval_digest.encode("ascii"))
    return MemoryFixtures(
        training=training,
        evaluation=evaluation,
        training_digest=train_digest,
        evaluation_digest=eval_digest,
        combined_digest=digest.hexdigest(),
    )


def contiguous_memory_horizon(
    per_delay: tuple[tuple[int, MemoryAccuracy], ...],
    *,
    threshold: float = 0.85,
) -> int | None:
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be finite and in [0,1]")
    if tuple(delay for delay, _ in per_delay) != MEMORY_DELAYS:
        raise ValueError("per_delay must use the registered E1-A delays in order")
    horizon: int | None = None
    for delay, score in per_delay:
        if score.accuracy < threshold:
            break
        horizon = delay
    return horizon


def run_memory_arm(spec: ReservoirSpec, fixtures: MemoryFixtures) -> MemoryArmResult:
    if not isinstance(spec, ReservoirSpec):
        raise ValueError("spec must be ReservoirSpec")
    if spec.input_size != 4:
        raise ValueError("E1-A reservoir input_size must be four")
    if not isinstance(fixtures, MemoryFixtures):
        raise ValueError("fixtures must be MemoryFixtures")
    reservoir = build_reservoir(spec)
    before = reservoir.parameter_digest()
    training_states, training_labels, training_delays, _ = _collect_states(
        reservoir,
        fixtures.training,
        reset_before_decision=False,
    )
    evaluation_states, evaluation_labels, evaluation_delays, _ = _collect_states(
        reservoir,
        fixtures.evaluation,
        reset_before_decision=False,
    )
    reset_states, reset_labels, reset_delays, reset_pairs = _collect_states(
        reservoir,
        fixtures.evaluation,
        reset_before_decision=True,
    )
    probe = fit_binary_ridge(
        training_states,
        training_labels,
        regularization=_REGULARIZATION,
    )
    train_predictions = probe.predict(training_states)
    predictions = probe.predict(evaluation_states)
    reset_predictions = probe.predict(reset_states)
    training = _score(training_labels, train_predictions)
    evaluation = _score(evaluation_labels, predictions)
    per_delay = _score_per_delay(evaluation_labels, predictions, evaluation_delays)
    reset = _score(reset_labels, reset_predictions)
    reset_per_delay = _score_per_delay(reset_labels, reset_predictions, reset_delays)
    all_reset_pairs_equal = _paired_states_equal(reset_states, reset_delays, reset_pairs)
    after = reservoir.parameter_digest()
    delay1_accuracy = per_delay[0][1].accuracy
    degradations = tuple(
        (delay, delay1_accuracy - score.accuracy) for delay, score in per_delay[1:]
    )
    valid = (
        reservoir.state_dim == spec.budget
        and before == after
        and all_reset_pairs_equal
        and reset == MemoryAccuracy(140, 280)
        and all(score == MemoryAccuracy(20, 40) for _, score in reset_per_delay)
        and all(
            len(value) == 64
            for value in (
                before,
                fixtures.combined_digest,
                probe.coefficient_digest(),
                prediction_digest(predictions),
                prediction_digest(reset_predictions),
            )
        )
    )
    return MemoryArmResult(
        training=training,
        evaluation=evaluation,
        per_delay=per_delay,
        reset=reset,
        reset_per_delay=reset_per_delay,
        macro_accuracy=float(np.mean([score.accuracy for _, score in per_delay])),
        memory_horizon=contiguous_memory_horizon(per_delay),
        degradation_from_delay1=degradations,
        all_reset_pairs_equal=all_reset_pairs_equal,
        valid=valid,
        parameter_digest_before=before,
        parameter_digest_after=after,
        fixture_digest=fixtures.combined_digest,
        coefficient_digest=probe.coefficient_digest(),
        prediction_digest=prediction_digest(predictions),
        reset_prediction_digest=prediction_digest(reset_predictions),
    )


def run_phase2a_compatibility(seed: int) -> Phase2ACompatibility:
    validated_seed = _validated_seed(seed)
    config = MemoryProbeConfig()
    legacy = run_memory_probe(validated_seed, config)
    task = DelayedCueTask()
    train_rng = np.random.default_rng(
        np.random.SeedSequence([validated_seed, 0x50524F42])
    )
    eval_rng = np.random.default_rng(
        np.random.SeedSequence([validated_seed, 0x4556414C])
    )
    training_fixtures = _legacy_build_fixtures(task, train_rng, config.training_blocks)
    evaluation_fixtures = _legacy_build_fixtures(task, eval_rng, config.evaluation_blocks)
    manual = _LegacyReservoir(validated_seed, config.hidden_size, config.recurrent_radius)
    training_states, training_labels, training_delays = _legacy_collect(
        manual,
        training_fixtures,
        reset_before_decision=False,
    )
    evaluation_states, evaluation_labels, evaluation_delays = _legacy_collect(
        manual,
        evaluation_fixtures,
        reset_before_decision=False,
    )
    reset_states, reset_labels, reset_delays = _legacy_collect(
        manual,
        evaluation_fixtures,
        reset_before_decision=True,
    )
    fitted = fit_linear_probe(
        training_states,
        training_labels,
        regularization=config.regularization,
    )
    train_predictions = fitted.predict(training_states)
    predictions = fitted.predict(evaluation_states)
    reset_predictions = fitted.predict(reset_states)
    training = _legacy_accuracy(training_labels, train_predictions)
    evaluation = _legacy_accuracy(evaluation_labels, predictions)
    per_delay = tuple(
        (
            delay,
            _legacy_accuracy(
                evaluation_labels[evaluation_delays == delay],
                predictions[evaluation_delays == delay],
            ),
        )
        for delay in range(1, 6)
    )
    reset = _legacy_accuracy(reset_labels, reset_predictions)
    hidden_match = _legacy_hidden_matches_policy(
        validated_seed,
        config,
        evaluation_fixtures[:10],
    )
    training_matches = training == legacy.training
    evaluation_matches = evaluation == legacy.recurrent
    per_delay_matches = per_delay == legacy.per_delay
    reset_matches = reset == legacy.state_reset and np.array_equal(
        reset_states,
        np.repeat(reset_states[:1], reset_states.shape[0], axis=0),
    ) == legacy.all_reset_hidden_equal
    probe_digest_matches = fitted.digest() == legacy.probe_digest
    prediction_digest_matches = (
        _legacy_choice_digest(predictions) == legacy.recurrent_choice_digest
    )
    reset_prediction_digest_matches = (
        _legacy_choice_digest(reset_predictions) == legacy.reset_choice_digest
    )
    acceptance_required = validated_seed in _HISTORICAL_PHASE2A_SEEDS
    acceptance_passed = _legacy_acceptance_passed(legacy)
    structural_pass = all(
        (
            hidden_match,
            training_matches,
            evaluation_matches,
            per_delay_matches,
            reset_matches,
            probe_digest_matches,
            prediction_digest_matches,
            reset_prediction_digest_matches,
        )
    )
    return Phase2ACompatibility(
        seed=validated_seed,
        hidden_recurrence_matches=hidden_match,
        training_matches=training_matches,
        evaluation_matches=evaluation_matches,
        per_delay_matches=per_delay_matches,
        reset_matches=reset_matches,
        probe_digest_matches=probe_digest_matches,
        prediction_digest_matches=prediction_digest_matches,
        reset_prediction_digest_matches=reset_prediction_digest_matches,
        legacy_acceptance_required=acceptance_required,
        legacy_acceptance_passed=acceptance_passed,
        passed=structural_pass and (acceptance_passed if acceptance_required else True),
    )


def _build_paired_fixtures(
    rng: np.random.Generator,
    pair_count: int,
) -> tuple[MemoryEpisode, ...]:
    episodes: list[MemoryEpisode] = []
    for delay in MEMORY_DELAYS:
        for pair_id in range(pair_count):
            nuisance = tuple(float(value) for value in rng.uniform(-0.25, 0.25, size=delay))
            delays = tuple(_delay_frame(value) for value in nuisance)
            for label in (0, 1):
                cue = np.zeros(4, dtype=np.float64)
                cue[label] = 1.0
                episodes.append(
                    MemoryEpisode(
                        label=label,
                        delay=delay,
                        pair_id=pair_id,
                        cue_stimulus=cue,
                        delay_stimuli=delays,
                        decision_stimulus=np.array([0.0, 0.0, 0.0, 1.0]),
                    )
                )
    return tuple(episodes)


def _collect_states(
    reservoir: Reservoir,
    episodes: tuple[MemoryEpisode, ...],
    *,
    reset_before_decision: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    states = []
    for episode in episodes:
        reservoir.reset()
        reservoir.advance(episode.cue_stimulus)
        for frame in episode.delay_stimuli:
            reservoir.advance(frame)
        if reset_before_decision:
            reservoir.reset()
        states.append(reservoir.advance(episode.decision_stimulus))
    return (
        np.vstack(states),
        np.asarray([episode.label for episode in episodes], dtype=np.int64),
        np.asarray([episode.delay for episode in episodes], dtype=np.int64),
        np.asarray([episode.pair_id for episode in episodes], dtype=np.int64),
    )


def _score(labels: np.ndarray, predictions: np.ndarray) -> MemoryAccuracy:
    return MemoryAccuracy(
        int(np.count_nonzero(labels == predictions)),
        int(labels.size),
    )


def _score_per_delay(
    labels: np.ndarray,
    predictions: np.ndarray,
    delays: np.ndarray,
) -> tuple[tuple[int, MemoryAccuracy], ...]:
    return tuple(
        (
            delay,
            _score(labels[delays == delay], predictions[delays == delay]),
        )
        for delay in MEMORY_DELAYS
    )


def _paired_states_equal(
    states: np.ndarray,
    delays: np.ndarray,
    pair_ids: np.ndarray,
) -> bool:
    for delay in MEMORY_DELAYS:
        ids = np.unique(pair_ids[delays == delay])
        for pair_id in ids:
            mask = (delays == delay) & (pair_ids == pair_id)
            pair = states[mask]
            if pair.shape[0] != 2 or not np.array_equal(pair[0], pair[1]):
                return False
    return True


def _fixture_digest(episodes: tuple[MemoryEpisode, ...]) -> str:
    digest = hashlib.sha256()
    digest.update(b"r1-e1-a-fixture-sequence-v1\0")
    for episode in episodes:
        digest.update(np.asarray([episode.label, episode.delay, episode.pair_id], dtype=np.int64).tobytes())
        for frame in (episode.cue_stimulus, *episode.delay_stimuli, episode.decision_stimulus):
            digest.update(np.ascontiguousarray(frame, dtype=np.float64).tobytes(order="C"))
    return digest.hexdigest()


def _validated_seed(seed: object) -> int:
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative Python integer")
    return seed


def _validated_frame(values: object) -> np.ndarray:
    try:
        frame = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("frame must be float64-compatible") from exc
    if frame.ndim != 1 or frame.shape != (4,) or not np.all(np.isfinite(frame)):
        raise ValueError("frame must be a finite four-vector")
    return frame


def _delay_frame(value: float) -> np.ndarray:
    return np.array([0.0, 0.0, value, 0.0], dtype=np.float64)


def _readonly_copy(values: np.ndarray) -> np.ndarray:
    copied = np.array(values, dtype=np.float64, copy=True, order="C")
    copied.flags.writeable = False
    return copied


class _LegacyReservoir:
    def __init__(self, seed: int, hidden_size: int, radius: float) -> None:
        rng = np.random.default_rng(seed)
        self._input = rng.normal(0.0, 1.0 / math.sqrt(4), size=(hidden_size, 4))
        recurrent = rng.normal(
            0.0,
            1.0 / math.sqrt(hidden_size),
            size=(hidden_size, hidden_size),
        )
        spectral_radius = float(np.max(np.abs(np.linalg.eigvals(recurrent))))
        self._recurrent = recurrent * (radius / spectral_radius)
        if radius == 0.0:
            self._recurrent.fill(0.0)
        self._state = np.zeros(hidden_size, dtype=np.float64)

    def reset(self) -> None:
        self._state.fill(0.0)

    def advance(self, frame: np.ndarray) -> np.ndarray:
        self._state = np.tanh(self._input @ frame + self._recurrent @ self._state)
        return self._state.copy()


def _legacy_balanced_cases(rng: np.random.Generator) -> list[tuple[Cue, int]]:
    cases = [(cue, delay) for cue in (Cue.LEFT, Cue.RIGHT) for delay in range(1, 6)]
    rng.shuffle(cases)
    return cases


def _legacy_build_fixtures(
    task: DelayedCueTask,
    rng: np.random.Generator,
    blocks: int,
) -> tuple[DelayedCueEpisode, ...]:
    return tuple(
        task.build_episode(cue, delay, rng)
        for _ in range(blocks)
        for cue, delay in _legacy_balanced_cases(rng)
    )


def _legacy_collect(
    reservoir: _LegacyReservoir,
    fixtures: tuple[DelayedCueEpisode, ...],
    *,
    reset_before_decision: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    states = []
    for episode in fixtures:
        reservoir.reset()
        reservoir.advance(episode.cue_stimulus)
        for frame in episode.delay_stimuli:
            reservoir.advance(frame)
        if reset_before_decision:
            reservoir.reset()
        states.append(reservoir.advance(episode.decision_stimulus))
    return (
        np.vstack(states),
        np.asarray([episode.correct_action_index for episode in fixtures], dtype=np.int64),
        np.asarray([episode.delay_steps for episode in fixtures], dtype=np.int64),
    )


def _legacy_hidden_matches_policy(
    seed: int,
    config: MemoryProbeConfig,
    fixtures: tuple[DelayedCueEpisode, ...],
) -> bool:
    manual = _LegacyReservoir(seed, config.hidden_size, config.recurrent_radius)
    policy = RecurrentPolicy(
        input_size=4,
        action_count=2,
        hidden_size=config.hidden_size,
        seed=seed,
        recurrent_radius=config.recurrent_radius,
    )
    for episode in fixtures:
        manual.reset()
        policy.reset_state()
        if not np.array_equal(manual.advance(episode.cue_stimulus), policy.advance(episode.cue_stimulus)):
            return False
        for frame in episode.delay_stimuli:
            if not np.array_equal(manual.advance(frame), policy.advance(frame)):
                return False
        if not np.array_equal(
            manual.advance(episode.decision_stimulus),
            policy.advance(episode.decision_stimulus),
        ):
            return False
    return True


def _legacy_accuracy(labels: np.ndarray, predictions: np.ndarray):
    from .memory_probe import ProbeAccuracy

    return ProbeAccuracy(int(np.count_nonzero(labels == predictions)), int(labels.size))


def _legacy_choice_digest(predictions: np.ndarray) -> str:
    values = np.ascontiguousarray(predictions, dtype=np.uint8)
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def _legacy_acceptance_passed(result: object) -> bool:
    expected_delays = tuple(range(1, 6))
    observed_delays = tuple(delay for delay, _ in result.per_delay)
    return (
        result.config == MemoryProbeConfig()
        and result.training.total == 2_000
        and result.recurrent.total == 200
        and result.recurrent.accuracy >= 0.90
        and observed_delays == expected_delays
        and all(score.total == 40 and score.accuracy >= 0.85 for _, score in result.per_delay)
        and result.state_reset.correct == 100
        and result.state_reset.total == 200
        and result.all_reset_hidden_equal
        and result.output_weight_digest_before == result.output_weight_digest_after
        and result.repeatable
    )
