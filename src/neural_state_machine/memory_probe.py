from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np

from .memory_task import Cue, DelayedCueEpisode, DelayedCueTask
from .policy import RecurrentPolicy


@dataclass(frozen=True)
class MemoryProbeConfig:
    hidden_size: int = 64
    recurrent_radius: float = 0.9
    training_blocks: int = 200
    evaluation_blocks: int = 20
    regularization: float = 1e-6

    def __post_init__(self) -> None:
        for name in ("hidden_size", "training_blocks", "evaluation_blocks"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.recurrent_radius, bool):
            raise ValueError(  # noqa: TRY004 - bool is invalid for numeric protocol
                "recurrent_radius must be finite and in [0.0, 1.0)"
            )
        try:
            radius = float(self.recurrent_radius)
        except (TypeError, ValueError) as exc:
            raise ValueError("recurrent_radius must be finite and in [0.0, 1.0)") from exc
        if not math.isfinite(radius) or not 0.0 <= radius < 1.0:
            raise ValueError("recurrent_radius must be finite and in [0.0, 1.0)")
        object.__setattr__(self, "recurrent_radius", radius)
        object.__setattr__(self, "regularization", _validated_regularization(self.regularization))


@dataclass(frozen=True)
class ProbeAccuracy:
    correct: int
    total: int

    def __post_init__(self) -> None:
        if type(self.correct) is not int or type(self.total) is not int:
            raise ValueError("correct and total must be integers")
        if self.total <= 0 or not 0 <= self.correct <= self.total:
            raise ValueError("counts must satisfy 0 <= correct <= total with positive total")

    @property
    def accuracy(self) -> float:
        return self.correct / self.total


@dataclass(frozen=True)
class _ProbeRun:
    seed: int
    config: MemoryProbeConfig
    training: ProbeAccuracy
    recurrent: ProbeAccuracy
    per_delay: tuple[tuple[int, ProbeAccuracy], ...]
    state_reset: ProbeAccuracy
    all_reset_hidden_equal: bool
    output_weight_digest_before: str
    output_weight_digest_after: str
    probe_digest: str
    recurrent_choice_digest: str
    reset_choice_digest: str


def _balanced_cases(rng: np.random.Generator) -> list[tuple[Cue, int]]:
    cases = [(cue, delay) for cue in (Cue.LEFT, Cue.RIGHT) for delay in range(1, 6)]
    rng.shuffle(cases)
    return cases


def _build_fixtures(
    task: DelayedCueTask,
    rng: np.random.Generator,
    blocks: int,
) -> tuple[DelayedCueEpisode, ...]:
    return tuple(
        task.build_episode(cue, delay, rng)
        for _ in range(blocks)
        for cue, delay in _balanced_cases(rng)
    )


@dataclass(frozen=True)
class _StateDataset:
    states: np.ndarray
    labels: np.ndarray
    delays: np.ndarray

    def __post_init__(self) -> None:
        states = _validated_state_matrix(self.states, require_samples=True)
        labels = np.asarray(self.labels, dtype=np.int64)
        delays = np.asarray(self.delays, dtype=np.int64)
        if labels.ndim != 1 or delays.ndim != 1:
            raise ValueError("dataset labels and delays must be rank one")
        if labels.shape[0] != states.shape[0] or delays.shape[0] != states.shape[0]:
            raise ValueError("dataset arrays must have matching sample counts")
        object.__setattr__(self, "states", _readonly_copy(states, dtype=np.float64))
        object.__setattr__(self, "labels", _readonly_copy(labels, dtype=np.int64))
        object.__setattr__(self, "delays", _readonly_copy(delays, dtype=np.int64))


def _collect_hidden(
    policy: RecurrentPolicy,
    episode: DelayedCueEpisode,
    *,
    reset_before_decision: bool,
) -> np.ndarray:
    policy.reset_state()
    policy.advance(episode.cue_stimulus)
    for stimulus in episode.delay_stimuli:
        policy.advance(stimulus)
    if reset_before_decision:
        policy.reset_state()
    return policy.advance(episode.decision_stimulus)


def _collect_dataset(
    policy: RecurrentPolicy,
    fixtures: tuple[DelayedCueEpisode, ...],
    *,
    reset_before_decision: bool,
) -> _StateDataset:
    states = np.vstack([
        _collect_hidden(policy, episode, reset_before_decision=reset_before_decision)
        for episode in fixtures
    ])
    return _StateDataset(
        states=states,
        labels=np.asarray([episode.correct_action_index for episode in fixtures], dtype=np.int64),
        delays=np.asarray([episode.delay_steps for episode in fixtures], dtype=np.int64),
    )


def _score_predictions(
    dataset: _StateDataset,
    choices: np.ndarray,
) -> tuple[ProbeAccuracy, tuple[tuple[int, ProbeAccuracy], ...]]:
    predicted = np.asarray(choices, dtype=np.int64)
    if predicted.ndim != 1 or predicted.shape != dataset.labels.shape:
        raise ValueError("choices must be rank one and match the dataset")
    matched = predicted == dataset.labels
    overall = ProbeAccuracy(int(np.count_nonzero(matched)), int(matched.size))
    per_delay = tuple(
        (
            delay,
            ProbeAccuracy(
                int(np.count_nonzero(matched[dataset.delays == delay])),
                int(np.count_nonzero(dataset.delays == delay)),
            ),
        )
        for delay in range(1, 6)
    )
    return overall, per_delay


def _choice_digest(choices: np.ndarray) -> str:
    values = np.ascontiguousarray(choices, dtype=np.uint8)
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def _readonly_copy(values: np.ndarray, *, dtype: np.dtype) -> np.ndarray:
    copied = np.array(values, dtype=dtype, copy=True, order="C")
    copied.flags.writeable = False
    return copied


def _validated_state_matrix(
    states: object,
    *,
    expected_features: int | None = None,
    require_samples: bool,
) -> np.ndarray:
    try:
        values = np.asarray(states, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("states must be float64-compatible") from exc
    if values.ndim != 2:
        raise ValueError("states must be rank two")
    if values.shape[1] == 0 or (require_samples and values.shape[0] == 0):
        raise ValueError("states must contain samples and features")
    if expected_features is not None and values.shape[1] != expected_features:
        raise ValueError(f"states must have {expected_features} features")
    if not np.all(np.isfinite(values)):
        raise ValueError("states must contain only finite values")
    return values


@dataclass(frozen=True)
class FittedLinearProbe:
    weights: np.ndarray
    bias: float

    def __post_init__(self) -> None:
        try:
            weights = np.asarray(self.weights, dtype=np.float64)
            bias = float(self.bias)
        except (TypeError, ValueError) as exc:
            raise ValueError("probe parameters must be float64-compatible") from exc
        if weights.ndim != 1 or weights.size == 0:
            raise ValueError("weights must be a non-empty rank-one array")
        if not np.all(np.isfinite(weights)) or not math.isfinite(bias):
            raise ValueError("probe parameters must contain only finite values")
        object.__setattr__(self, "weights", _readonly_copy(weights, dtype=np.float64))
        object.__setattr__(self, "bias", bias)

    def predict(self, states: np.ndarray) -> np.ndarray:
        values = _validated_state_matrix(
            states,
            expected_features=self.weights.size,
            require_samples=False,
        )
        choices = np.where(values @ self.weights + self.bias > 0.0, 1, 0)
        return _readonly_copy(choices, dtype=np.int64)

    def digest(self) -> str:
        values = np.ascontiguousarray(self.weights, dtype=np.float64)
        digest = hashlib.sha256()
        digest.update(str(values.shape).encode("ascii"))
        digest.update(values.tobytes(order="C"))
        digest.update(np.asarray(self.bias, dtype=np.float64).tobytes())
        return digest.hexdigest()


def _validated_regularization(value: object) -> float:
    if isinstance(value, bool):
        raise ValueError("regularization must be finite and positive")  # noqa: TRY004
    try:
        regularization = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("regularization must be finite and positive") from exc
    if not math.isfinite(regularization) or regularization <= 0.0:
        raise ValueError("regularization must be finite and positive")
    return regularization


def fit_linear_probe(
    states: np.ndarray,
    labels: np.ndarray,
    *,
    regularization: float,
) -> FittedLinearProbe:
    values = _validated_state_matrix(states, require_samples=True)
    raw_labels = np.asarray(labels)
    if raw_labels.ndim != 1 or raw_labels.shape[0] != values.shape[0]:
        raise ValueError("labels must be rank one and match the sample count")
    if raw_labels.dtype.kind not in "iu" or raw_labels.dtype.kind == "b":
        raise ValueError("labels must contain integer action indices")
    action_indices = np.asarray(raw_labels, dtype=np.int64)
    if not np.all(np.isin(action_indices, (0, 1))):
        raise ValueError("labels must contain only zero and one")
    if set(action_indices.tolist()) != {0, 1}:
        raise ValueError("labels must contain both classes")
    strength = _validated_regularization(regularization)
    design = np.column_stack((values, np.ones(values.shape[0])))
    penalty = np.diag([strength] * values.shape[1] + [0.0])
    target = np.where(action_indices == 0, -1.0, 1.0)
    try:
        parameters = np.linalg.solve(
            design.T @ design + penalty,
            design.T @ target,
        )
    except np.linalg.LinAlgError as exc:
        raise RuntimeError("linear probe solve failed") from exc
    return FittedLinearProbe(parameters[:-1], float(parameters[-1]))


def _run_probe_once(seed: int, config: MemoryProbeConfig) -> _ProbeRun:
    policy = RecurrentPolicy(
        input_size=4,
        action_count=2,
        hidden_size=config.hidden_size,
        seed=seed,
        recurrent_radius=config.recurrent_radius,
    )
    task = DelayedCueTask()
    output_before = policy.output_weight_digest()
    training_rng = np.random.default_rng(
        np.random.SeedSequence([seed, 0x50524F42])
    )
    evaluation_rng = np.random.default_rng(
        np.random.SeedSequence([seed, 0x4556414C])
    )
    training_fixtures = _build_fixtures(task, training_rng, config.training_blocks)
    evaluation_fixtures = _build_fixtures(
        task, evaluation_rng, config.evaluation_blocks
    )
    training_data = _collect_dataset(
        policy, training_fixtures, reset_before_decision=False
    )
    recurrent_data = _collect_dataset(
        policy, evaluation_fixtures, reset_before_decision=False
    )
    reset_data = _collect_dataset(
        policy, evaluation_fixtures, reset_before_decision=True
    )
    fitted = fit_linear_probe(
        training_data.states,
        training_data.labels,
        regularization=config.regularization,
    )
    training_choices = fitted.predict(training_data.states)
    recurrent_choices = fitted.predict(recurrent_data.states)
    reset_choices = fitted.predict(reset_data.states)
    training, _ = _score_predictions(training_data, training_choices)
    recurrent, per_delay = _score_predictions(recurrent_data, recurrent_choices)
    state_reset, _ = _score_predictions(reset_data, reset_choices)
    all_reset_equal = np.array_equal(
        reset_data.states,
        np.repeat(reset_data.states[:1], reset_data.states.shape[0], axis=0),
    )
    return _ProbeRun(
        seed=seed,
        config=config,
        training=training,
        recurrent=recurrent,
        per_delay=per_delay,
        state_reset=state_reset,
        all_reset_hidden_equal=all_reset_equal,
        output_weight_digest_before=output_before,
        output_weight_digest_after=policy.output_weight_digest(),
        probe_digest=fitted.digest(),
        recurrent_choice_digest=_choice_digest(recurrent_choices),
        reset_choice_digest=_choice_digest(reset_choices),
    )
