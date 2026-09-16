# Phase 2A Linear Memory Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether the frozen recurrent hidden state retains one-to-five-step vanished-cue information that one shared deterministic Ridge linear probe can decode.

**Architecture:** Keep the existing recurrent policy, delayed-cue task, and failed reward-learning baseline frozen. Add an independent Phase 2A measurement module that builds balanced hidden-state datasets using only `reset_state()` and `advance()`, fits one closed-form Ridge probe across all delays, evaluates recurrent and reset-ablation features, and exposes a repeatable three-seed JSON benchmark. The probe is an offline instrument and never becomes the game controller or modifies policy parameters.

**Tech Stack:** Python 3.10+, NumPy, pytest, Ruff, setuptools, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-14-delayed-cue-memory-design.md`

## Global Constraints

- Work only on the standalone orphan branch `experiment/neural-state-machine`; do not merge from or depend on `master`.
- Tasks 1–6 of `docs/superpowers/plans/2026-09-14-delayed-cue-memory.md` are the frozen foundation. This plan supersedes only that plan's original Tasks 7–8.
- Do not modify `src/neural_state_machine/policy.py`, `controller.py`, `memory_task.py`, or `memory_benchmark.py` unless a separately demonstrated defect receives a new design review.
- NumPy remains the only runtime dependency. Do not add scikit-learn, PyTorch, an RL framework, FlyVis, fly-brain, ROS2, YOLO, or a game engine.
- Phase 2A calls only `RecurrentPolicy.reset_state()` and `RecurrentPolicy.advance()` on the state-evolution path. It must not call `decide()` or `learn()`, inspect logits, or create decision eligibility.
- Probe features contain only the current decision-time hidden vector. Cue identity, correct action, delay length, episode phase, and task metadata must never be passed to `FittedLinearProbe.predict()`.
- Fit exactly one probe on the combined delay 1–5 training dataset. Reuse that same object for training, recurrent evaluation, per-delay scoring, and reset ablation; do not fit delay-specific probes or add delay-derived features.
- Use defaults `hidden_size=64`, `recurrent_radius=0.9`, `training_blocks=200`, `evaluation_blocks=20`, and `regularization=1e-6`.
- Use training RNG lineage `SeedSequence([seed, 0x50524F42])` and evaluation RNG lineage `SeedSequence([seed, 0x4556414C])`. Build evaluation fixtures once and reuse them in recurrent and reset modes.
- Fixed acceptance seeds are exactly `7`, `17`, and `29`. Do not weaken the `0.90` overall, `0.85` per-delay, or exact `100/200 == 0.50` reset thresholds.
- If any fixed seed fails, report that the current reservoir lacks robust linearly decodable 1–5-step memory and stop. Do not change thresholds, seeds, task difficulty, block counts, regularization, hidden size, or recurrent radius without a revised approved design.
- Preserve the Phase 1 API, numeric tests, and benchmark JSON. Keep the existing reward-learning failure visible as a non-gating Phase 2B baseline.
- Use RED–GREEN–REFACTOR for every behavior change. Run the focused failing test before writing the production implementation.
- Commit after every task using the exact commit subject shown in that task. Do not combine tasks into one large commit.

## File and responsibility map

| File | Responsibility | Change |
|---|---|---|
| `src/neural_state_machine/memory_probe.py` | Probe configuration, immutable Ridge probe, hidden-state collection, scoring, repeatability, acceptance, and benchmark payload | Create |
| `tests/test_memory_probe.py` | Probe math/validation, protocol isolation, RNG fixtures, controls, results, acceptance, and CLI contracts | Create |
| `scripts/benchmark_memory_probe.py` | Compact sorted JSON CLI and nonzero scientific-gate exit | Create |
| `src/neural_state_machine/__init__.py` | Export the Phase 2A public API | Modify |
| `README.md` | Explain Phase 2A/2B separation, commands, fixed results, and interpretation boundary | Modify |
| `.github/workflows/ci.yml` | Run the Phase 1 and Phase 2A benchmarks on Python 3.10–3.12 | Modify |

## Locked interfaces

The tasks below must use these names and signatures consistently:

| Symbol | Exact public signature or fields |
|---|---|
| `MemoryProbeConfig` | `hidden_size: int = 64`, `recurrent_radius: float = 0.9`, `training_blocks: int = 200`, `evaluation_blocks: int = 20`, `regularization: float = 1e-6` |
| `ProbeAccuracy` | `correct: int`, `total: int`, read-only property `accuracy: float` |
| `FittedLinearProbe` | `weights: np.ndarray`, `bias: float`; methods `predict(states: np.ndarray) -> np.ndarray` and `digest() -> str` |
| `fit_linear_probe` | `(states: np.ndarray, labels: np.ndarray, *, regularization: float) -> FittedLinearProbe` |
| `MemoryProbeResult` | `seed`, `config`, `training`, `recurrent`, `per_delay`, `state_reset`, `all_reset_hidden_equal`, both output digests, probe digest, both choice digests, and `repeatable` with the types defined in Task 4 |
| `run_memory_probe` | `(seed: int = 7, config: MemoryProbeConfig \| None = None) -> MemoryProbeResult` |
| `run_memory_probe_benchmark` | `(seeds: Sequence[int] = (7, 17, 29), config: MemoryProbeConfig \| None = None) -> dict[str, object]` |

`MemoryProbeResult` reports accuracy through `ProbeAccuracy.correct`, `.total`, and the derived `.accuracy` property. `run_memory_probe_benchmark` serializes those values explicitly; no NumPy scalar or array may escape into its dictionary.

---

### Task 1: Implement the immutable Ridge probe

**Files:**
- Create: `src/neural_state_machine/memory_probe.py`
- Create: `tests/test_memory_probe.py`

**Interfaces:**
- Consumes: NumPy only.
- Produces: `FittedLinearProbe(weights: ndarray, bias: float)`, `predict(states) -> ndarray`, `digest() -> str`, and `fit_linear_probe(states, labels, *, regularization) -> FittedLinearProbe`.
- Later tasks add experiment types and orchestration to the same focused module.

- [ ] **Step 1: Write RED tests for construction, prediction, copying, tie behavior, and digest**

Add these imports and tests to `tests/test_memory_probe.py`:

```python
import hashlib

import numpy as np
import pytest

from neural_state_machine.memory_probe import FittedLinearProbe


def test_fitted_probe_defensively_copies_readonly_float64_weights() -> None:
    source = np.array([1, -2], dtype=np.int64)
    probe = FittedLinearProbe(source, 0)
    source[:] = 99

    np.testing.assert_array_equal(probe.weights, [1.0, -2.0])
    assert probe.weights.dtype == np.float64
    assert not probe.weights.flags.writeable
    assert not np.shares_memory(probe.weights, source)


def test_prediction_is_readonly_independent_and_uses_left_on_exact_tie() -> None:
    states = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 8.0]])
    probe = FittedLinearProbe(np.array([1.0, 0.0]), 0.0)
    prediction = probe.predict(states)
    states[:] = 100.0

    np.testing.assert_array_equal(prediction, [1, 0, 0])
    assert prediction.dtype == np.int64
    assert not prediction.flags.writeable


def test_probe_digest_covers_shape_weights_and_float64_bias() -> None:
    probe = FittedLinearProbe(np.array([1.5, -2.0]), 0.25)
    expected = hashlib.sha256()
    expected.update(str((2,)).encode("ascii"))
    expected.update(np.array([1.5, -2.0], dtype=np.float64).tobytes(order="C"))
    expected.update(np.asarray(0.25, dtype=np.float64).tobytes())

    assert probe.digest() == expected.hexdigest()
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
.venv/bin/pytest tests/test_memory_probe.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'neural_state_machine.memory_probe'`.

- [ ] **Step 3: Implement immutable parameters, input validation, prediction, and digest**

Create `src/neural_state_machine/memory_probe.py` with the module imports and these exact behaviors:

```python
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


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
```

- [ ] **Step 4: Write RED validation tests for direct construction and prediction**

Append parametrized cases covering scalar/rank-two/empty/non-finite weights, non-finite bias, scalar/rank-one/rank-three/non-finite state matrices, and wrong feature width:

```python
@pytest.mark.parametrize(
    ("weights", "bias"),
    [
        (1.0, 0.0),
        (np.ones((1, 2)), 0.0),
        (np.array([]), 0.0),
        (np.array([np.nan]), 0.0),
        (np.array([1.0]), np.inf),
    ],
)
def test_fitted_probe_rejects_invalid_parameters(weights: object, bias: object) -> None:
    with pytest.raises(ValueError):
        FittedLinearProbe(weights, bias)


@pytest.mark.parametrize(
    "states",
    [1.0, np.ones(2), np.ones((1, 1, 2)), [[np.nan, 0.0]], [[1.0]]],
)
def test_predict_rejects_invalid_state_matrices(states: object) -> None:
    with pytest.raises(ValueError):
        FittedLinearProbe(np.ones(2), 0.0).predict(states)
```

Run the two validation tests. Expected: PASS because Step 3 included the complete validation boundary.

- [ ] **Step 5: Write RED tests for the exact Ridge algorithm and fit failures**

Extend the module import to include `fit_linear_probe`, then add:

```python
def test_fit_linear_probe_uses_closed_form_ridge_without_bias_penalty() -> None:
    states = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    labels = np.array([0, 0, 1, 1], dtype=np.int64)

    probe = fit_linear_probe(states, labels, regularization=1.0)

    np.testing.assert_allclose(probe.weights, [6.0 / 11.0], rtol=0.0, atol=1e-15)
    assert probe.bias == pytest.approx(0.0, abs=1e-15)
    np.testing.assert_array_equal(probe.predict(states), labels)


@pytest.mark.parametrize(
    ("states", "labels"),
    [
        (np.empty((0, 2)), np.empty(0, dtype=np.int64)),
        (np.empty((2, 0)), np.array([0, 1])),
        (np.ones(2), np.array([0, 1])),
        (np.array([[0.0], [np.inf]]), np.array([0, 1])),
        (np.ones((2, 1)), np.array([[0], [1]])),
        (np.ones((2, 1)), np.array([0])),
        (np.ones((2, 1)), np.array([0.0, 1.0])),
        (np.ones((2, 1)), np.array([False, True])),
        (np.ones((2, 1)), np.array([0, 2])),
        (np.ones((2, 1)), np.array([0, 0])),
    ],
)
def test_fit_linear_probe_rejects_invalid_datasets(states: object, labels: object) -> None:
    with pytest.raises(ValueError):
        fit_linear_probe(states, labels, regularization=1e-6)


@pytest.mark.parametrize("regularization", [0.0, -1.0, np.inf, np.nan, True])
def test_fit_linear_probe_rejects_invalid_regularization(regularization: object) -> None:
    with pytest.raises(ValueError):
        fit_linear_probe(
            np.array([[-1.0], [1.0]]),
            np.array([0, 1]),
            regularization=regularization,
        )


def test_fit_wraps_numpy_solve_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    failure = np.linalg.LinAlgError("singular")

    def fail_solve(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        raise failure

    monkeypatch.setattr(np.linalg, "solve", fail_solve)
    with pytest.raises(RuntimeError, match="linear probe solve failed") as caught:
        fit_linear_probe(
            np.array([[-1.0], [1.0]]),
            np.array([0, 1]),
            regularization=1e-6,
        )
    assert caught.value.__cause__ is failure
```

Run:

```bash
.venv/bin/pytest tests/test_memory_probe.py -q
```

Expected: RED with `ImportError` or `NameError` for `fit_linear_probe`.

- [ ] **Step 6: Implement the exact Ridge fit and error translation**

```python
def _validated_regularization(value: object) -> float:
    if isinstance(value, bool):
        raise ValueError("regularization must be finite and positive")
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
```

- [ ] **Step 7: Verify Task 1 and commit**

Run:

```bash
.venv/bin/pytest tests/test_memory_probe.py -q
.venv/bin/ruff check src/neural_state_machine/memory_probe.py tests/test_memory_probe.py
```

Expected: both commands pass.

Commit:

```bash
git add src/neural_state_machine/memory_probe.py tests/test_memory_probe.py
git commit -m "feat: add deterministic ridge memory probe"
```

---

### Task 2: Build isolated hidden-state datasets

**Files:**
- Modify: `src/neural_state_machine/memory_probe.py`
- Modify: `tests/test_memory_probe.py`

**Interfaces:**
- Consumes: `Cue`, `DelayedCueEpisode`, `DelayedCueTask`, and `RecurrentPolicy.reset_state()/advance()`.
- Produces: `MemoryProbeConfig`, private `_StateDataset`, `_build_fixtures(task, rng, blocks)`, `_collect_hidden(policy, episode, *, reset_before_decision)`, and `_collect_dataset(policy, fixtures, *, reset_before_decision)`.
- No function in this task calls `decide()`, `learn()`, or imports reward-learning helpers from `memory_benchmark.py`.

- [ ] **Step 1: Write RED tests for configuration validation**

Extend imports with `MemoryProbeConfig`, then add:

```python
@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("hidden_size", 0),
        ("hidden_size", True),
        ("training_blocks", -1),
        ("training_blocks", 1.5),
        ("evaluation_blocks", 0),
        ("evaluation_blocks", False),
    ],
)
def test_memory_probe_config_rejects_invalid_counts(name: str, value: object) -> None:
    values = dict(
        hidden_size=64,
        recurrent_radius=0.9,
        training_blocks=200,
        evaluation_blocks=20,
        regularization=1e-6,
    )
    values[name] = value
    with pytest.raises(ValueError):
        MemoryProbeConfig(**values)


@pytest.mark.parametrize("radius", [-0.1, 1.0, np.inf, np.nan, True])
def test_memory_probe_config_rejects_invalid_radius(radius: object) -> None:
    with pytest.raises(ValueError):
        MemoryProbeConfig(recurrent_radius=radius)


@pytest.mark.parametrize("strength", [0.0, -1.0, np.inf, np.nan, True])
def test_memory_probe_config_rejects_invalid_regularization(strength: object) -> None:
    with pytest.raises(ValueError):
        MemoryProbeConfig(regularization=strength)


def test_memory_probe_config_defaults_are_the_locked_protocol() -> None:
    assert MemoryProbeConfig() == MemoryProbeConfig(
        hidden_size=64,
        recurrent_radius=0.9,
        training_blocks=200,
        evaluation_blocks=20,
        regularization=1e-6,
    )
```

Run the config tests. Expected: RED because `MemoryProbeConfig` is absent.

- [ ] **Step 2: Implement the frozen validated configuration**

```python
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
            raise ValueError("recurrent_radius must be finite and in [0.0, 1.0)")
        try:
            radius = float(self.recurrent_radius)
        except (TypeError, ValueError) as exc:
            raise ValueError("recurrent_radius must be finite and in [0.0, 1.0)") from exc
        if not math.isfinite(radius) or not 0.0 <= radius < 1.0:
            raise ValueError("recurrent_radius must be finite and in [0.0, 1.0)")
        object.__setattr__(self, "recurrent_radius", radius)
        object.__setattr__(
            self,
            "regularization",
            _validated_regularization(self.regularization),
        )
```

- [ ] **Step 3: Write RED tests for balanced independent fixtures**

Import `Counter`, `Cue`, `DelayedCueTask`, and the module as `probe_module`. Add:

```python
from collections import Counter

from neural_state_machine import Cue, DelayedCueTask, RecurrentPolicy
from neural_state_machine import memory_probe as probe_module


def test_probe_fixtures_are_shuffled_balanced_fresh_blocks() -> None:
    fixtures = probe_module._build_fixtures(
        DelayedCueTask(),
        np.random.default_rng(21),
        3,
    )

    assert type(fixtures) is tuple
    assert len(fixtures) == 30
    for start in range(0, 30, 10):
        block = fixtures[start : start + 10]
        assert Counter(episode.correct_action_index for episode in block) == {0: 5, 1: 5}
        assert Counter(episode.delay_steps for episode in block) == {
            1: 2,
            2: 2,
            3: 2,
            4: 2,
            5: 2,
        }
        assert len({(episode.cue, episode.delay_steps) for episode in block}) == 10
    assert len({episode.delay_stimuli[0][2] for episode in fixtures}) == 30
```

Run this test. Expected: RED because `_build_fixtures` is absent.

- [ ] **Step 4: Implement local balanced fixtures without coupling to Phase 2B**

Add imports from `.memory_task` and `.policy`, then implement:

```python
from .memory_task import Cue, DelayedCueEpisode, DelayedCueTask
from .policy import RecurrentPolicy


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
```

This deliberate ten-line duplication keeps the memory measurement independent from private reward-training helpers.

- [ ] **Step 5: Write RED tests for exact hidden collection and reset ablation**

```python
def _manual_hidden(
    policy: RecurrentPolicy,
    episode,
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


@pytest.mark.parametrize("reset", [False, True])
def test_collect_hidden_matches_the_literal_advance_only_protocol(reset: bool) -> None:
    episode = DelayedCueTask().build_episode(Cue.RIGHT, 4, np.random.default_rng(4))
    actual_policy = RecurrentPolicy(4, 2, hidden_size=8, seed=7)
    reference_policy = RecurrentPolicy(4, 2, hidden_size=8, seed=7)

    actual = probe_module._collect_hidden(
        actual_policy,
        episode,
        reset_before_decision=reset,
    )
    expected = _manual_hidden(
        reference_policy,
        episode,
        reset_before_decision=reset,
    )

    np.testing.assert_array_equal(actual, expected)
    assert not actual.flags.writeable
    with pytest.raises(RuntimeError, match="preceding decision"):
        actual_policy.learn(1.0)


def test_reset_dataset_has_identical_states_and_literal_side_labels() -> None:
    fixtures = probe_module._build_fixtures(
        DelayedCueTask(), np.random.default_rng(3), 2
    )
    policy = RecurrentPolicy(4, 2, hidden_size=8, seed=9)
    dataset = probe_module._collect_dataset(
        policy,
        fixtures,
        reset_before_decision=True,
    )

    assert dataset.states.shape == (20, 8)
    assert dataset.labels.shape == (20,)
    assert dataset.delays.shape == (20,)
    assert not dataset.states.flags.writeable
    assert not dataset.labels.flags.writeable
    assert not dataset.delays.flags.writeable
    assert np.array_equal(dataset.states, np.repeat(dataset.states[:1], 20, axis=0))
    np.testing.assert_array_equal(dataset.labels, [episode.correct_action_index for episode in fixtures])
    np.testing.assert_array_equal(dataset.delays, [episode.delay_steps for episode in fixtures])
```

Run these tests. Expected: RED because `_collect_hidden`, `_collect_dataset`, and `_StateDataset` are absent.

- [ ] **Step 6: Implement immutable side-metadata datasets and the literal collector**

```python
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
    states = np.vstack(
        [
            _collect_hidden(
                policy,
                episode,
                reset_before_decision=reset_before_decision,
            )
            for episode in fixtures
        ]
    )
    return _StateDataset(
        states=states,
        labels=np.asarray(
            [episode.correct_action_index for episode in fixtures],
            dtype=np.int64,
        ),
        delays=np.asarray([episode.delay_steps for episode in fixtures], dtype=np.int64),
    )
```

- [ ] **Step 7: Prove policy weights and decision eligibility remain untouched**

Add a test that snapshots all three matrices and the public output digest, collects both modes, then compares exact bytes:

```python
def test_dataset_collection_never_changes_policy_parameters_or_creates_eligibility() -> None:
    policy = RecurrentPolicy(4, 2, hidden_size=8, seed=12)
    fixtures = probe_module._build_fixtures(
        DelayedCueTask(), np.random.default_rng(8), 2
    )
    input_before = policy._input_weights.copy()
    recurrent_before = policy._recurrent_weights.copy()
    output_before = policy._output_weights.copy()
    digest_before = policy.output_weight_digest()

    probe_module._collect_dataset(policy, fixtures, reset_before_decision=False)
    probe_module._collect_dataset(policy, fixtures, reset_before_decision=True)

    np.testing.assert_array_equal(policy._input_weights, input_before)
    np.testing.assert_array_equal(policy._recurrent_weights, recurrent_before)
    np.testing.assert_array_equal(policy._output_weights, output_before)
    assert policy.output_weight_digest() == digest_before
    with pytest.raises(RuntimeError, match="preceding decision"):
        policy.learn(1.0)
```

- [ ] **Step 8: Verify Task 2 and commit**

Run:

```bash
.venv/bin/pytest tests/test_memory_probe.py -q
.venv/bin/ruff check src/neural_state_machine/memory_probe.py tests/test_memory_probe.py
```

Expected: both commands pass.

Commit:

```bash
git add src/neural_state_machine/memory_probe.py tests/test_memory_probe.py
git commit -m "feat: collect delayed cue hidden states"
```

---

### Task 3: Measure one complete Phase 2A run

**Files:**
- Modify: `src/neural_state_machine/memory_probe.py`
- Modify: `tests/test_memory_probe.py`

**Interfaces:**
- Consumes: `MemoryProbeConfig`, `FittedLinearProbe`, dataset helpers, and `fit_linear_probe`.
- Produces: `ProbeAccuracy`, private `_ProbeRun`, `_score_predictions(dataset, choices)`, `_choice_digest(choices)`, and `_run_probe_once(seed, config) -> _ProbeRun`.
- `_ProbeRun` is array-free so independent runs can be compared with dataclass equality.

- [ ] **Step 1: Write RED tests for literal integer scoring and choice digests**

```python
def test_probe_scoring_uses_literal_labels_and_groups_all_five_delays() -> None:
    dataset = probe_module._StateDataset(
        states=np.arange(20, dtype=np.float64).reshape(10, 2),
        labels=np.array([0, 1] * 5),
        delays=np.repeat(np.arange(1, 6), 2),
    )
    choices = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 0], dtype=np.int64)

    overall, per_delay = probe_module._score_predictions(dataset, choices)

    assert overall == probe_module.ProbeAccuracy(7, 10)
    assert overall.accuracy == 0.7
    assert per_delay == (
        (1, probe_module.ProbeAccuracy(1, 2)),
        (2, probe_module.ProbeAccuracy(2, 2)),
        (3, probe_module.ProbeAccuracy(1, 2)),
        (4, probe_module.ProbeAccuracy(2, 2)),
        (5, probe_module.ProbeAccuracy(1, 2)),
    )
    expected = hashlib.sha256(np.asarray(choices, dtype=np.uint8).tobytes()).hexdigest()
    assert probe_module._choice_digest(choices) == expected
```

Run the test. Expected: RED because the scoring types and helpers are absent.

- [ ] **Step 2: Implement validated counts, scoring, and platform-stable choice digest**

```python
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
```

- [ ] **Step 3: Write a RED independent-reference test for one run**

The test must construct the same state source, RNG lineages, fixtures, datasets, and fit explicitly. It must compare every field and demonstrate that training/evaluation distractors are independent:

```python
def test_run_probe_once_matches_independent_protocol() -> None:
    seed = 7
    config = MemoryProbeConfig(
        hidden_size=9,
        recurrent_radius=0.7,
        training_blocks=3,
        evaluation_blocks=2,
        regularization=1e-4,
    )
    policy = RecurrentPolicy(
        input_size=4,
        action_count=2,
        hidden_size=config.hidden_size,
        seed=seed,
        recurrent_radius=config.recurrent_radius,
    )
    task = DelayedCueTask()
    train_fixtures = probe_module._build_fixtures(
        task,
        np.random.default_rng(np.random.SeedSequence([seed, 0x50524F42])),
        config.training_blocks,
    )
    evaluation_fixtures = probe_module._build_fixtures(
        task,
        np.random.default_rng(np.random.SeedSequence([seed, 0x4556414C])),
        config.evaluation_blocks,
    )
    assert train_fixtures[0].delay_stimuli[0].tobytes() != evaluation_fixtures[0].delay_stimuli[0].tobytes()
    before = policy.output_weight_digest()
    training_data = probe_module._collect_dataset(
        policy, train_fixtures, reset_before_decision=False
    )
    recurrent_data = probe_module._collect_dataset(
        policy, evaluation_fixtures, reset_before_decision=False
    )
    reset_data = probe_module._collect_dataset(
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
    training, _ = probe_module._score_predictions(training_data, training_choices)
    recurrent, per_delay = probe_module._score_predictions(
        recurrent_data, recurrent_choices
    )
    state_reset, _ = probe_module._score_predictions(reset_data, reset_choices)

    actual = probe_module._run_probe_once(seed, config)

    assert actual.seed == seed
    assert actual.config == config
    assert actual.training == training
    assert actual.recurrent == recurrent
    assert actual.per_delay == per_delay
    assert actual.state_reset == state_reset
    assert actual.all_reset_hidden_equal is np.array_equal(
        reset_data.states,
        np.repeat(reset_data.states[:1], reset_data.states.shape[0], axis=0),
    )
    assert actual.output_weight_digest_before == before
    assert actual.output_weight_digest_after == before
    assert actual.probe_digest == fitted.digest()
    assert actual.recurrent_choice_digest == probe_module._choice_digest(recurrent_choices)
    assert actual.reset_choice_digest == probe_module._choice_digest(reset_choices)
```

Run this test. Expected: RED because `_ProbeRun` and `_run_probe_once` are absent.

- [ ] **Step 4: Implement a single fit shared by every metric and ablation**

```python
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
```

The fitted probe is created once. `labels` are passed only to `fit_linear_probe` and `_score_predictions`; prediction receives each `.states` array alone.

- [ ] **Step 5: Add explicit decision-stimulus and frozen-matrix controls**

Add one test that compares left/right decision arrays exactly and another that snapshots private input/recurrent/output matrices around `_run_probe_once`'s constituent collection steps. Use the existing `test_dataset_collection_never_changes_policy_parameters_or_creates_eligibility` for byte-level matrix evidence, and add this direct task invariant:

```python
def test_probe_protocol_has_one_shared_decision_vector_for_both_cues() -> None:
    task = DelayedCueTask()
    left = task.build_episode(Cue.LEFT, 1, np.random.default_rng(1))
    right = task.build_episode(Cue.RIGHT, 5, np.random.default_rng(2))

    np.testing.assert_array_equal(left.decision_stimulus, right.decision_stimulus)
    np.testing.assert_array_equal(left.decision_stimulus, [0.0, 0.0, 0.0, 1.0])
```

- [ ] **Step 6: Verify Task 3 and commit**

Run:

```bash
.venv/bin/pytest tests/test_memory_probe.py -q
.venv/bin/pytest tests/test_policy.py tests/test_memory_task.py tests/test_memory_benchmark.py -q
.venv/bin/ruff check src/neural_state_machine/memory_probe.py tests/test_memory_probe.py
```

Expected: all commands pass; the reward baseline tests remain unchanged.

Commit:

```bash
git add src/neural_state_machine/memory_probe.py tests/test_memory_probe.py
git commit -m "feat: measure delayed cue probe runs"
```

---

### Task 4: Add public repeatability, acceptance, benchmark, and CLI

**Files:**
- Modify: `src/neural_state_machine/memory_probe.py`
- Modify: `src/neural_state_machine/__init__.py`
- Modify: `tests/test_memory_probe.py`
- Create: `scripts/benchmark_memory_probe.py`

**Interfaces:**
- Consumes: `_run_probe_once(seed, config) -> _ProbeRun`.
- Produces: public `MemoryProbeResult`, `run_memory_probe`, and `run_memory_probe_benchmark` with the locked signatures above.
- Produces private `_passes_acceptance(result) -> bool` and `_result_payload(result) -> dict[str, object]` used by the benchmark and CLI.

- [ ] **Step 1: Write RED tests for seeds and full independent repeatability**

Extend imports with the three public orchestration symbols. Add:

```python
@pytest.mark.parametrize("seed", [-1, True, 1.5, "7", None])
def test_run_memory_probe_rejects_invalid_seed_before_running(seed: object) -> None:
    with pytest.raises(ValueError, match="non-negative Python integer"):
        probe_module.run_memory_probe(seed, MemoryProbeConfig(training_blocks=1, evaluation_blocks=1))


def test_run_memory_probe_compares_two_complete_independent_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = MemoryProbeConfig(training_blocks=1, evaluation_blocks=1)
    first = probe_module._ProbeRun(
        seed=7,
        config=config,
        training=probe_module.ProbeAccuracy(10, 10),
        recurrent=probe_module.ProbeAccuracy(9, 10),
        per_delay=tuple(
            (delay, probe_module.ProbeAccuracy(2, 2)) for delay in range(1, 6)
        ),
        state_reset=probe_module.ProbeAccuracy(5, 10),
        all_reset_hidden_equal=True,
        output_weight_digest_before="output",
        output_weight_digest_after="output",
        probe_digest="probe",
        recurrent_choice_digest="recurrent",
        reset_choice_digest="reset",
    )
    calls: list[int] = []

    def fake_run(seed: int, received: MemoryProbeConfig):
        calls.append(seed)
        assert received is config
        return first

    monkeypatch.setattr(probe_module, "_run_probe_once", fake_run)
    result = probe_module.run_memory_probe(7, config)

    assert calls == [7, 7]
    assert result.repeatable is True
    assert result.probe_digest == "probe"


def test_real_small_probe_run_is_exactly_repeatable() -> None:
    config = MemoryProbeConfig(
        hidden_size=8,
        recurrent_radius=0.7,
        training_blocks=2,
        evaluation_blocks=2,
        regularization=1e-4,
    )

    first = probe_module.run_memory_probe(11, config)
    second = probe_module.run_memory_probe(11, config)

    assert first == second
    assert first.repeatable is True
```

The monkeypatch is limited to proving two invocations; deterministic real-run equality is tested at the end of this task.

Run these tests. Expected: RED because the public wrapper is absent.

- [ ] **Step 2: Implement the public result and exact full-object comparison**

```python
@dataclass(frozen=True)
class MemoryProbeResult:
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
    repeatable: bool


def _validated_seed(seed: object) -> int:
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative Python integer")
    return seed


def _public_result(run: _ProbeRun, *, repeatable: bool) -> MemoryProbeResult:
    return MemoryProbeResult(
        seed=run.seed,
        config=run.config,
        training=run.training,
        recurrent=run.recurrent,
        per_delay=run.per_delay,
        state_reset=run.state_reset,
        all_reset_hidden_equal=run.all_reset_hidden_equal,
        output_weight_digest_before=run.output_weight_digest_before,
        output_weight_digest_after=run.output_weight_digest_after,
        probe_digest=run.probe_digest,
        recurrent_choice_digest=run.recurrent_choice_digest,
        reset_choice_digest=run.reset_choice_digest,
        repeatable=repeatable,
    )


def run_memory_probe(
    seed: int = 7,
    config: MemoryProbeConfig | None = None,
) -> MemoryProbeResult:
    validated_seed = _validated_seed(seed)
    if config is None:
        resolved = MemoryProbeConfig()
    elif isinstance(config, MemoryProbeConfig):
        resolved = config
    else:
        raise ValueError("config must be a MemoryProbeConfig")
    first = _run_probe_once(validated_seed, resolved)
    second = _run_probe_once(validated_seed, resolved)
    return _public_result(first, repeatable=first == second)
```

- [ ] **Step 3: Write RED tests for exact acceptance and JSON-safe payloads**

Use `dataclasses.replace` to vary one criterion at a time:

```python
from dataclasses import replace


def _passing_result(seed: int = 7) -> probe_module.MemoryProbeResult:
    config = MemoryProbeConfig()
    return probe_module.MemoryProbeResult(
        seed=seed,
        config=config,
        training=probe_module.ProbeAccuracy(2000, 2000),
        recurrent=probe_module.ProbeAccuracy(180, 200),
        per_delay=tuple(
            (delay, probe_module.ProbeAccuracy(36, 40)) for delay in range(1, 6)
        ),
        state_reset=probe_module.ProbeAccuracy(100, 200),
        all_reset_hidden_equal=True,
        output_weight_digest_before="same",
        output_weight_digest_after="same",
        probe_digest="probe",
        recurrent_choice_digest="recurrent",
        reset_choice_digest="reset",
        repeatable=True,
    )


def test_acceptance_requires_every_locked_control() -> None:
    passing = _passing_result()
    assert probe_module._passes_acceptance(passing)
    assert not probe_module._passes_acceptance(
        replace(passing, recurrent=probe_module.ProbeAccuracy(179, 200))
    )
    assert not probe_module._passes_acceptance(
        replace(
            passing,
            per_delay=((1, probe_module.ProbeAccuracy(33, 40)), *passing.per_delay[1:]),
        )
    )
    assert not probe_module._passes_acceptance(
        replace(passing, per_delay=passing.per_delay[:-1])
    )
    assert not probe_module._passes_acceptance(
        replace(passing, state_reset=probe_module.ProbeAccuracy(101, 200))
    )
    assert not probe_module._passes_acceptance(
        replace(passing, all_reset_hidden_equal=False)
    )
    assert not probe_module._passes_acceptance(
        replace(passing, output_weight_digest_after="changed")
    )
    assert not probe_module._passes_acceptance(replace(passing, repeatable=False))


def test_benchmark_payload_is_stable_json_data(monkeypatch: pytest.MonkeyPatch) -> None:
    results = {seed: _passing_result(seed) for seed in (7, 17, 29)}
    monkeypatch.setattr(
        probe_module,
        "run_memory_probe",
        lambda seed, config=None: results[seed],
    )

    payload = probe_module.run_memory_probe_benchmark()

    assert payload["phase"] == "2A"
    assert payload["all_passed"] is True
    assert payload["reward_baseline_gates_phase_2a"] is False
    assert [entry["seed"] for entry in payload["results"]] == [7, 17, 29]
    assert payload["results"][0]["recurrent"] == {
        "correct": 180,
        "total": 200,
        "accuracy": 0.9,
    }
```

Run these tests. Expected: RED because acceptance and benchmark functions are absent.

- [ ] **Step 4: Implement acceptance and explicit JSON serialization**

```python
def _accuracy_payload(score: ProbeAccuracy) -> dict[str, object]:
    return {
        "correct": score.correct,
        "total": score.total,
        "accuracy": score.accuracy,
    }


def _passes_acceptance(result: MemoryProbeResult) -> bool:
    expected_delays = tuple(range(1, 6))
    observed_delays = tuple(delay for delay, _ in result.per_delay)
    per_delay_total = sum(score.total for _, score in result.per_delay)
    per_delay_correct = sum(score.correct for _, score in result.per_delay)
    return (
        result.config == MemoryProbeConfig()
        and result.training.total == 2_000
        and result.recurrent.total == 200
        and result.recurrent.accuracy >= 0.90
        and observed_delays == expected_delays
        and all(
            score.total == 40 and score.accuracy >= 0.85
            for _, score in result.per_delay
        )
        and per_delay_total == result.recurrent.total
        and per_delay_correct == result.recurrent.correct
        and result.state_reset == ProbeAccuracy(100, 200)
        and result.all_reset_hidden_equal
        and result.output_weight_digest_before == result.output_weight_digest_after
        and result.repeatable
    )


def _result_payload(result: MemoryProbeResult) -> dict[str, object]:
    return {
        "seed": result.seed,
        "config": {
            "hidden_size": result.config.hidden_size,
            "recurrent_radius": result.config.recurrent_radius,
            "training_blocks": result.config.training_blocks,
            "evaluation_blocks": result.config.evaluation_blocks,
            "regularization": result.config.regularization,
        },
        "passed": _passes_acceptance(result),
        "training": _accuracy_payload(result.training),
        "recurrent": _accuracy_payload(result.recurrent),
        "per_delay": {
            str(delay): _accuracy_payload(score) for delay, score in result.per_delay
        },
        "state_reset": _accuracy_payload(result.state_reset),
        "all_reset_hidden_equal": result.all_reset_hidden_equal,
        "output_weight_digest_before": result.output_weight_digest_before,
        "output_weight_digest_after": result.output_weight_digest_after,
        "probe_digest": result.probe_digest,
        "recurrent_choice_digest": result.recurrent_choice_digest,
        "reset_choice_digest": result.reset_choice_digest,
        "repeatable": result.repeatable,
    }


def run_memory_probe_benchmark(
    seeds: Sequence[int] = (7, 17, 29),
    config: MemoryProbeConfig | None = None,
) -> dict[str, object]:
    try:
        seed_values = tuple(seeds)
    except TypeError as exc:
        raise ValueError("seeds must be a non-empty sequence") from exc
    if not seed_values:
        raise ValueError("seeds must be a non-empty sequence")
    validated = tuple(_validated_seed(seed) for seed in seed_values)
    if len(set(validated)) != len(validated):
        raise ValueError("seeds must not contain duplicates")
    results = tuple(run_memory_probe(seed, config) for seed in validated)
    return {
        "phase": "2A",
        "all_passed": all(_passes_acceptance(result) for result in results),
        "reward_baseline_gates_phase_2a": False,
        "results": [_result_payload(result) for result in results],
    }
```

Add validation tests for empty, duplicate, boolean, negative, float, and string seed sequences.

- [ ] **Step 5: Export the locked public API**

Modify `src/neural_state_machine/__init__.py` to import and include these names in `__all__`:

```python
from .memory_probe import (
    FittedLinearProbe,
    MemoryProbeConfig,
    MemoryProbeResult,
    ProbeAccuracy,
    fit_linear_probe,
    run_memory_probe,
    run_memory_probe_benchmark,
)
```

Add an import-surface test:

```python
def test_phase_2a_public_api_is_exported() -> None:
    import neural_state_machine as package

    for name in (
        "FittedLinearProbe",
        "MemoryProbeConfig",
        "MemoryProbeResult",
        "ProbeAccuracy",
        "fit_linear_probe",
        "run_memory_probe",
        "run_memory_probe_benchmark",
    ):
        assert name in package.__all__
        assert getattr(package, name) is getattr(probe_module, name)
```

- [ ] **Step 6: Add the compact sorted JSON CLI and its subprocess test**

Create `scripts/benchmark_memory_probe.py`:

```python
import json

from neural_state_machine.memory_probe import run_memory_probe_benchmark


def main() -> int:
    result = run_memory_probe_benchmark()
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

Add:

```python
import json
import subprocess
import sys


def test_probe_benchmark_cli_emits_one_compact_sorted_json_object() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/benchmark_memory_probe.py"],
        check=False,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert completed.returncode == 0
    assert completed.stderr == ""
    assert completed.stdout.count("\n") == 1
    assert completed.stdout.rstrip() == json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    )
    assert payload["all_passed"] is True
    assert [entry["seed"] for entry in payload["results"]] == [7, 17, 29]
```

- [ ] **Step 7: Run the fixed scientific gate before committing**

Run:

```bash
.venv/bin/pytest tests/test_memory_probe.py -q
.venv/bin/python scripts/benchmark_memory_probe.py
```

Expected for an implementation eligible to continue: tests pass; the CLI exits `0`; all three seed entries have `passed:true`; recurrent overall is at least `180/200`; each delay is at least `34/40`; reset is exactly `100/200`; reset states are identical; policy output digests match; and `repeatable` is true.

If the command exits `1` or a fixed seed misses any threshold, stop this plan at the scientific gate. Preserve the exact JSON as evidence, report that the frozen reservoir does not robustly preserve linearly decodable 1–5-step memory, and request a new design review. Do not modify a scientific constant or commit a knowingly failing acceptance test.

- [ ] **Step 8: Commit only after the scientific gate passes**

```bash
git add src/neural_state_machine/memory_probe.py src/neural_state_machine/__init__.py tests/test_memory_probe.py scripts/benchmark_memory_probe.py
git commit -m "feat: add phase two-a memory probe benchmark"
```

---

### Task 5: Document the separated results and wire CI

**Files:**
- Modify: `README.md`
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: the exact Phase 2A JSON produced by Task 4 and the already recorded Phase 2B failure table in the approved spec.
- Produces: user-facing commands and interpretation boundaries; CI evidence on Python 3.10, 3.11, and 3.12.

- [ ] **Step 1: Capture the exact passing Phase 2A result for documentation**

Run:

```bash
.venv/bin/python scripts/benchmark_memory_probe.py
```

Copy only the reported integer counts and accuracies for seeds 7, 17, and 29 into the README table. Do not recompute, round upward, or describe a threshold miss as a pass.

- [ ] **Step 2: Add the Phase 2A/2B README section**

After `## Run`, add:

```markdown
## Delayed-cue memory gates

Phase 2 separates two claims that the first experiment combined:

- **Phase 2A — memory decodability:** one frozen Ridge linear probe measures
  whether the current recurrent hidden vector retains a vanished left/right cue
  across delays 1–5.
- **Phase 2B — reward learning:** a separate online reward-modulated readout
  must learn to use that information. Its current fixed-seed baseline fails and
  does not determine the Phase 2A result.

Run the Phase 2A scientific gate:

```bash
python scripts/benchmark_memory_probe.py
```

The command prints one compact sorted JSON object and exits nonzero if any of
seeds 7, 17, or 29 misses the fixed gate: at least 90% overall, at least 85%
for every delay, exact 50% after resetting state immediately before the shared
decision input, identical reset features, unchanged policy output weights, and
exact independent-run repeatability.

The frozen Phase 2B reward-learning baseline remains:

| Seed | Reward learner | Reset | Delay 1 / 2 / 3 / 4 / 5 |
|---:|---:|---:|---|
| 7 | 0.61 | 0.50 | 1.00 / 0.50 / 0.50 / 0.50 / 0.55 |
| 17 | 0.84 | 0.50 | 1.00 / 0.85 / 0.65 / 0.90 / 0.80 |
| 29 | 0.80 | 0.50 | 1.00 / 1.00 / 1.00 / 0.50 / 0.50 |

A passing Phase 2A result supports only linear availability of cue information
in recurrent activity. It does not show that the current reward rule learns
the readout, that semantic attractors emerged, or that the system has general
game intelligence or biological plausibility.
```

Immediately after the fixed-gate paragraph, insert the Phase 2A table emitted by this command verbatim:

```bash
.venv/bin/python - <<'PY'
import json
import subprocess
import sys

payload = json.loads(
    subprocess.check_output(
        [sys.executable, "scripts/benchmark_memory_probe.py"],
        text=True,
    )
)
print("| Seed | Phase 2A overall | Reset | Delay 1 / 2 / 3 / 4 / 5 |")
print("|---:|---:|---:|---|")
for result in payload["results"]:
    recurrent = result["recurrent"]
    reset = result["state_reset"]
    delays = " / ".join(
        f'{result["per_delay"][str(delay)]["correct"]}/'
        f'{result["per_delay"][str(delay)]["total"]}'
        for delay in range(1, 6)
    )
    print(
        f'| {result["seed"]} | {recurrent["correct"]}/{recurrent["total"]} '
        f'| {reset["correct"]}/{reset["total"]} | {delays} |'
    )
PY
```

This makes the committed values literal measurements rather than estimates.

- [ ] **Step 3: Update commands, package layout, and design-document links**

Add `python scripts/benchmark_memory_probe.py` to the verification command block. Add `memory_task.py`, `memory_benchmark.py`, and `memory_probe.py` to the package layout with respectively “immutable delayed-cue protocol,” “Phase 2B reward-learning baseline,” and “Phase 2A frozen linear measurement.” Add links to:

```markdown
- `docs/superpowers/specs/2026-09-14-delayed-cue-memory-design.md`
- `docs/superpowers/plans/2026-09-14-delayed-cue-memory.md`
- `docs/superpowers/plans/2026-09-14-phase-2a-memory-probe.md`
```

- [ ] **Step 4: Wire the Phase 2A benchmark into every CI Python version**

Append this step after the existing Phase 1 deterministic benchmark in `.github/workflows/ci.yml`:

```yaml
      - name: Delayed-cue linear memory probe
        run: python scripts/benchmark_memory_probe.py
```

Do not rename or remove the existing `pytest -q`, `ruff check .`, or `python scripts/benchmark.py` steps.

- [ ] **Step 5: Verify documentation and CI locally**

Run:

```bash
.venv/bin/pytest -q
.venv/bin/ruff check .
.venv/bin/python scripts/benchmark.py
.venv/bin/python scripts/benchmark_memory_probe.py
```

Expected: all commands exit `0`; the Phase 1 benchmark is byte-for-byte identical to its frozen output; the Phase 2A JSON matches the README counts; and the existing reward-learning tests remain green without entering the Phase 2A pass calculation.

- [ ] **Step 6: Commit documentation and CI**

```bash
git add README.md .github/workflows/ci.yml
git commit -m "docs: report separated memory experiment gates"
```

---

### Task 6: Perform the final scientific and scope audit

**Files:**
- Verify only: all files changed by Tasks 1–5
- Create no source, test, documentation, or workflow files in this task.

**Interfaces:**
- Consumes: committed Task 1–5 implementation and the approved design specification.
- Produces: verification evidence and a clean, reviewable branch head.

- [ ] **Step 1: Confirm the diff stays inside the approved Phase 2A surface**

Run:

```bash
plan_head="$(git log -1 --format=%H -- docs/superpowers/plans/2026-09-14-phase-2a-memory-probe.md)"
git diff --name-status "$plan_head"..HEAD
git diff --check "$plan_head"..HEAD
git status --short
```

Expected changed paths are only:

```text
.github/workflows/ci.yml
README.md
scripts/benchmark_memory_probe.py
src/neural_state_machine/__init__.py
src/neural_state_machine/memory_probe.py
tests/test_memory_probe.py
```

The resolved `plan_head` is the plan-only commit and excludes this document from the implementation range. `git diff --check` prints nothing and the worktree is clean.

- [ ] **Step 2: Audit forbidden controller semantics and calls**

Run:

```bash
rg -n "cue_memory|remembered_cue|AttackState|RetreatState|behavior_tree|transition_table|delay_counter|episode_phase|observation_history" src tests scripts
rg -n "\.decide\(|\.learn\(|logits" src/neural_state_machine/memory_probe.py
rg -n "fit_linear_probe\(" src/neural_state_machine/memory_probe.py
```

Expected: the first two commands return no matches in the new Phase 2A module. The fit search shows the public definition plus exactly one orchestration call in `_run_probe_once`; no delay-loop fit call exists.

- [ ] **Step 3: Inspect the diff for feature leakage and frozen-module changes**

Run:

```bash
plan_head="$(git log -1 --format=%H -- docs/superpowers/plans/2026-09-14-phase-2a-memory-probe.md)"
git diff "$plan_head"..HEAD -- src/neural_state_machine/memory_probe.py tests/test_memory_probe.py
git diff --exit-code "$plan_head"..HEAD -- src/neural_state_machine/policy.py src/neural_state_machine/controller.py src/neural_state_machine/memory_task.py src/neural_state_machine/memory_benchmark.py
```

Confirm from the first diff that `predict()` receives only state matrices; labels appear only in fitting and scoring; delays appear only in balanced fixture construction and post-prediction scoring; evaluation fixtures are built once and reused; reset occurs immediately before decision `advance()`; and the output readout never participates. Expected for the second command: exit `0` with no output.

- [ ] **Step 4: Run the complete verification suite from a clean process**

Run:

```bash
.venv/bin/pytest -q
.venv/bin/ruff check .
.venv/bin/python scripts/benchmark.py
.venv/bin/python scripts/benchmark_memory_probe.py
```

Expected: every command exits `0`. Record the pytest count, both exact JSON lines, all fixed-seed Phase 2A counts, and all digests in the handoff.

- [ ] **Step 5: Verify independent benchmark bytes**

Run the Phase 2A command twice in separate processes and compare their files:

```bash
probe_a="$(mktemp)"
probe_b="$(mktemp)"
.venv/bin/python scripts/benchmark_memory_probe.py > "$probe_a"
.venv/bin/python scripts/benchmark_memory_probe.py > "$probe_b"
cmp "$probe_a" "$probe_b"
python -m json.tool "$probe_a" >/dev/null
```

Expected: `cmp` and JSON validation both exit `0`. Remove the two explicit temporary files after recording success:

```bash
rm "$probe_a" "$probe_b"
```

- [ ] **Step 6: Review the final commit series**

Run:

```bash
plan_head="$(git log -1 --format=%H -- docs/superpowers/plans/2026-09-14-phase-2a-memory-probe.md)"
git log --oneline "$plan_head"..HEAD
git status --short --branch
```

Expected implementation commit subjects, newest first:

```text
docs: report separated memory experiment gates
feat: add phase two-a memory probe benchmark
feat: measure delayed cue probe runs
feat: collect delayed cue hidden states
feat: add deterministic ridge memory probe
```

Do not create an empty audit commit. If the audit reveals a defect, return to the owning task's RED–GREEN cycle, make one focused fix commit, rerun every audit command, and report the additional commit explicitly.
