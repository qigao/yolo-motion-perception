# Delayed-Cue Neural Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove that a cue which has disappeared from the current stimulus remains behaviorally available through recurrent neural activity, and that erasing that activity removes the learned behavior.

**Architecture:** Extract the current seeded reservoir into a task-agnostic `RecurrentPolicy`, retain `RecurrentController` as a numerically compatible Phase 1 adapter, and add a separate delayed-cue protocol plus deterministic training/evaluation harness. Only the output readout learns; the task never injects cue identity, delay count, phase, or the correct answer into the controller at decision time.

**Tech Stack:** Python 3.10+, NumPy, pytest, Ruff, setuptools, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-14-delayed-cue-memory-design.md`

## Global Constraints

- Work only on the standalone orphan branch `experiment/neural-state-machine`; do not read from, merge, or modify `master`.
- Preserve the Phase 1 public API and its seeded numeric behavior. The existing benchmark for seed `7` is a frozen compatibility contract.
- NumPy remains the only runtime dependency. Do not add PyTorch, an RL framework, YOLO, FlyVis, fly-brain, ROS2, or a game engine.
- Controller code may contain numeric recurrent state and one-decision eligibility only. It must not contain a cue variable, semantic memory label, behavior state, behavior tree, transition table, episode phase, delay counter, or observation history.
- Training may modify only the output matrix. Input and recurrent matrices remain seeded and frozen.
- The decision stimulus is exactly `[0.0, 0.0, 0.0, 1.0]` for both cues, and legal actions are exactly numeric indices `0` and `1`, mapped to `MOVE_LEFT` and `MOVE_RIGHT`.
- Default training uses 2,000 episodes in balanced ten-case blocks, epsilon `0.25 -> 0.02`, and acceptance seeds `7`, `17`, and `29`.
- Do not weaken the `0.90` overall, `0.85` per-delay, or exact `0.50` ablation thresholds to obtain a green build. A miss is a reported hypothesis failure and a new design-review gate.
- Use RED-GREEN-REFACTOR for every behavior change. Run the focused failing test before writing its production implementation.
- Commit after every task using the exact commit subject shown in that task. Do not combine tasks into one large commit.

## File and responsibility map

| File | Responsibility | Change |
|---|---|---|
| `src/neural_state_machine/policy.py` | Generic numeric reservoir, legal-action selection, one-shot eligibility, output digest | Create |
| `src/neural_state_machine/controller.py` | Phase 1 `GameObservation`/`Action` compatibility adapter | Refactor |
| `src/neural_state_machine/memory_task.py` | Delayed-cue stimulus protocol and scoring truth | Create |
| `src/neural_state_machine/memory_benchmark.py` | Balanced fixtures, training, recurrent evaluation, reset ablation, results | Create |
| `src/neural_state_machine/__init__.py` | Stable public exports | Modify |
| `scripts/benchmark_memory.py` | Stable JSON command-line benchmark | Create |
| `tests/test_phase1_compatibility.py` | Frozen Phase 1 trajectory and benchmark values | Create |
| `tests/test_policy.py` | Generic policy dynamics, validation, masking, exploration, learning | Create |
| `tests/test_memory_task.py` | Stimulus equality, delay generation, scoring, validation | Create |
| `tests/test_memory_benchmark.py` | Training/evaluation mechanics, immutability, repeatability, thresholds | Create |
| `README.md` | Phase 2 experiment, claims, commands, interpretation boundary | Modify |
| `.github/workflows/ci.yml` | Run both Phase 1 and Phase 2 benchmarks on Python 3.10-3.12 | Modify |

---

### Task 1: Freeze the Phase 1 numerical compatibility contract

**Files:**
- Create: `tests/test_phase1_compatibility.py`
- Test: `tests/test_phase1_compatibility.py`

**Interfaces:**
- Consumes: existing `RecurrentController`, `GameObservation`, and `run_benchmark`.
- Produces: literal regression values that the generic-policy extraction must preserve.
- No production file changes are allowed in this task.

- [ ] Create a controller golden test using this exact observation and seed:

```python
import numpy as np

from neural_state_machine import GameObservation, RecurrentController


def test_phase_one_seeded_controller_output_is_frozen() -> None:
    observation = GameObservation(
        enemy_distance=0.25,
        enemy_direction=1,
        health=0.8,
        incoming_threat=0.1,
        healing_distance=0.75,
        healing_direction=-1,
        left_blocked=False,
        right_blocked=False,
    )
    decision = RecurrentController(hidden_size=12, seed=41, learning_rate=0.2).step(
        observation
    )

    assert decision.action.name == "ATTACK"
    np.testing.assert_allclose(
        decision.logits,
        [
            0.5458970211878782,
            0.5281858971630009,
            0.8572334978938736,
            -0.7405802043742861,
        ],
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        decision.hidden_state,
        [
            -0.8185519255871977,
            -0.2200558324105483,
            -0.45274824294880944,
            0.8680670805389271,
            -0.8680026124715702,
            0.3918013883639866,
            -0.6692504695763907,
            0.5308668694766017,
            -0.06530864276167365,
            -0.8351635483337463,
            -0.22228422572190384,
            -0.45996089701532794,
        ],
        rtol=0.0,
        atol=1e-12,
    )
```

- [ ] Add a benchmark golden test that compares all structural fields exactly and the four floating-point values with `pytest.approx(abs=1e-12, rel=0.0)`. Freeze these values:

```python
EXPECTED_PHASE_ONE = {
    "seed": 7,
    "replay_equal": True,
    "context_separation": 1.2417694746099712,
    "positive_margin_delta": 0.6795494788487084,
    "negative_margin_delta": -0.6795494788487085,
    "episode": {
        "steps": 5,
        "total_reward": 0.7999999999999999,
        "actions": ["ATTACK", "ATTACK", "ATTACK", "ATTACK", "ATTACK"],
        "terminal": {
            "player_position": 1,
            "enemy_position": 2,
            "player_health": 1,
            "enemy_health": 0,
            "healing_available": True,
            "done": True,
        },
    },
}
```

- [ ] Run `.venv/bin/pytest tests/test_phase1_compatibility.py -q`. Expected: PASS against the unrefactored controller. This is a characterization task, so no initial RED is expected.
- [ ] Run `.venv/bin/pytest -q` and confirm the total test count increases without altering any existing test.
- [ ] Commit only the new characterization file:

```bash
git add tests/test_phase1_compatibility.py
git commit -m "test: freeze phase one numerical baseline"
```

---

### Task 2: Add the generic recurrent policy dynamics and validation

**Files:**
- Create: `src/neural_state_machine/policy.py`
- Create: `tests/test_policy.py`
- Modify: `src/neural_state_machine/__init__.py`
- Test: `tests/test_policy.py`

**Interfaces:**
- Produces frozen `PolicyDecision(action_index: int, logits: ndarray, hidden_state: ndarray)`.
- Produces `RecurrentPolicy(input_size, action_count, hidden_size=16, seed=0, learning_rate=0.1, recurrent_radius=0.8)`.
- Produces `advance(stimulus)`, `decide(stimulus, legal_action_indices, *, explore_probability=0.0, rng=None)`, `reset_state()`, and `output_weight_digest()`.
- Task 3 adds `learn(reward)` after its eligibility behavior has a failing test.

- [ ] Write import and construction tests first. Validate these exact invalid groups with parametrized cases:

```python
@pytest.mark.parametrize("name,value", [
    ("input_size", 0), ("input_size", True),
    ("action_count", 1), ("action_count", 2.5),
    ("hidden_size", -1), ("hidden_size", False),
])
def test_policy_rejects_invalid_sizes(name: str, value: object) -> None:
    kwargs = dict(input_size=4, action_count=2, hidden_size=8, seed=7)
    kwargs[name] = value
    with pytest.raises(ValueError):
        RecurrentPolicy(**kwargs)


@pytest.mark.parametrize("radius", [-0.01, 1.0, np.inf, np.nan])
def test_policy_rejects_invalid_recurrent_radius(radius: float) -> None:
    with pytest.raises(ValueError):
        RecurrentPolicy(4, 2, recurrent_radius=radius)
```

Also reject non-integer/negative seeds, and non-positive or non-finite learning rates.

- [ ] Run `.venv/bin/pytest tests/test_policy.py -q`. Expected RED: `ModuleNotFoundError: neural_state_machine.policy`.
- [ ] Implement constructor validation and preserve this exact random draw order, which is required for the Phase 1 adapter:

```python
rng = np.random.default_rng(seed)
self._input_weights = rng.normal(
    0.0, 1.0 / math.sqrt(input_size), size=(hidden_size, input_size)
)
recurrent = rng.normal(
    0.0, 1.0 / math.sqrt(hidden_size), size=(hidden_size, hidden_size)
)
radius = float(np.max(np.abs(np.linalg.eigvals(recurrent))))
self._recurrent_weights = recurrent * (recurrent_radius / radius)
self._output_weights = rng.normal(
    0.0, 1.0 / math.sqrt(hidden_size), size=(action_count, hidden_size)
)
```

For `recurrent_radius == 0.0`, fill the recurrent matrix with zeros after the same three RNG operations; do not skip or reorder a draw.

- [ ] Add RED tests showing that `advance` rejects a scalar, rank-two array, wrong-length vector, non-numeric value, and every `NaN`/infinite vector. The implementation must normalize through a single helper:

```python
def _validated_stimulus(self, stimulus: np.ndarray) -> np.ndarray:
    try:
        values = np.asarray(stimulus, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("stimulus must be float64-compatible") from exc
    if values.ndim != 1 or values.shape != (self.input_size,):
        raise ValueError(f"stimulus must have shape ({self.input_size},)")
    if not np.all(np.isfinite(values)):
        raise ValueError("stimulus must contain only finite values")
    return values
```

- [ ] Implement one shared hidden evolution method used exactly once by `advance` and exactly once by `decide`:

```python
def _evolve(self, stimulus: np.ndarray) -> None:
    values = self._validated_stimulus(stimulus)
    self._hidden_state = np.tanh(
        self._input_weights @ values
        + self._recurrent_weights @ self._hidden_state
    )
```

`advance` returns a read-only copy and clears no state. It must not create learning eligibility.

- [ ] Add RED tests for legal action masks: `()`, duplicates `(0, 0)`, booleans, non-integers, negative indices, and indices equal to `action_count` all raise `ValueError` before state evolution. `(1, 0)` is valid, but selection still uses lowest numeric index for a tie.
- [ ] Implement deterministic `decide` by computing raw logits, copying them, setting every illegal position to `-np.inf`, and applying `int(np.argmax(masked_logits))`. Return the read-only masked vector in `PolicyDecision`; this makes the legal-action mask observable and gives NumPy's lowest-index tie behavior.
- [ ] Add a narrow tie-break contract test by zeroing `_output_weights` inside the test, calling `decide(np.zeros(4), (1, 0))`, and asserting index `0`. The test may touch this private matrix only to create an otherwise unreachable exact tie. Add a second test asserting every illegal returned logit is negative infinity and no illegal action can be selected.
- [ ] Add tests proving all arrays returned by `advance` and `decide` are non-writeable independent copies, and that modifying a source stimulus after the call does not change a returned value.
- [ ] Implement `output_weight_digest()` as SHA-256 over a C-contiguous `float64` copy plus its shape encoded as ASCII. This exposes identity, not writable parameters:

```python
def output_weight_digest(self) -> str:
    values = np.ascontiguousarray(self._output_weights, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(str(values.shape).encode("ascii"))
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()
```

- [ ] Export `PolicyDecision` and `RecurrentPolicy` from `src/neural_state_machine/__init__.py`.
- [ ] Run `.venv/bin/pytest tests/test_policy.py -q`, then `.venv/bin/ruff check src/neural_state_machine/policy.py tests/test_policy.py`.
- [ ] Commit:

```bash
git add src/neural_state_machine/policy.py src/neural_state_machine/__init__.py tests/test_policy.py
git commit -m "feat: add generic recurrent policy"
```

---

### Task 3: Implement exploration and one-decision learning eligibility

**Files:**
- Modify: `src/neural_state_machine/policy.py`
- Modify: `tests/test_policy.py`
- Test: `tests/test_policy.py`

**Interfaces:**
- Completes `decide(stimulus, legal_action_indices, *, explore_probability, rng)`.
- Completes `learn(reward) -> None` with clipped reward and consumed eligibility.
- Tightens `reset_state()` to clear transient hidden state and eligibility but not weights.

- [ ] Add RED tests for exploration validation. Values below `0`, above `1`, `NaN`, and infinities raise `ValueError`; `explore_probability > 0` with `rng is None` raises `ValueError`; a non-`numpy.random.Generator` RNG raises `ValueError`.
- [ ] Add a deterministic exploration test using two independent `np.random.default_rng(73)` generators and 50 decisions at probability `1.0`. Assert both action sequences are identical, contain only the legal indices `(0, 2)`, and include both legal actions.
- [ ] Implement epsilon-greedy selection. In the following code, `legal` is the validated tuple and `masked_logits` is the read-only candidate vector produced in Task 2. Make the exploration branch consume exactly one uniform decision draw and one `rng.choice` draw only when exploration is selected:

```python
if explore_probability > 0.0 and rng.random() < explore_probability:
    action_index = int(rng.choice(np.asarray(legal, dtype=np.int64)))
else:
    action_index = int(np.argmax(masked_logits))
```

The stored eligibility is always the chosen action index plus a copy of the decision hidden vector.

- [ ] Add RED tests for learning lifecycle:

```python
def test_learning_requires_one_unconsumed_decision() -> None:
    policy = RecurrentPolicy(4, 2, seed=7)
    with pytest.raises(RuntimeError, match="preceding decision"):
        policy.learn(1.0)

    policy.decide(np.zeros(4), (0, 1))
    policy.learn(1.0)
    with pytest.raises(RuntimeError, match="preceding decision"):
        policy.learn(1.0)
```

Also prove `advance` does not make `learn` legal, and `reset_state` consumes pending eligibility.

- [ ] Add RED margin tests for positive reward, negative reward, and reward clipping. From the same reset stimulus and seed, `learn(100.0)` must yield the same new digest and margin as `learn(1.0)`; `-100.0` must match `-1.0`.
- [ ] Implement `learn` by validating a finite reward, clipping to `[-1.0, 1.0]`, updating only the selected output row, and setting both eligibility fields to `None` immediately after the update:

```python
clipped = max(-1.0, min(1.0, float(reward)))
self._output_weights[action_index] += (
    self.learning_rate * clipped * decision_hidden
)
self._last_action_index = None
self._last_hidden_state = None
```

Validation of a non-finite reward occurs before mutation, but a rejected reward does not consume valid eligibility; add a test that a following finite reward still succeeds.

- [ ] Add a digest-locality test: `advance`, `decide`, and `reset_state` do not alter the output digest; exactly one nonzero `learn` call does.
- [ ] Run `.venv/bin/pytest tests/test_policy.py -q` and `.venv/bin/ruff check src/neural_state_machine/policy.py tests/test_policy.py`.
- [ ] Commit:

```bash
git add src/neural_state_machine/policy.py tests/test_policy.py
git commit -m "feat: add masked exploration and one-shot plasticity"
```

---

### Task 4: Refactor `RecurrentController` into a Phase 1 compatibility adapter

**Files:**
- Modify: `src/neural_state_machine/controller.py`
- Modify: `tests/test_controller.py`
- Test: `tests/test_phase1_compatibility.py`
- Test: `tests/test_benchmark.py`

**Interfaces:**
- Consumes: `RecurrentPolicy(input_size=9, action_count=4, recurrent_radius=0.8)`.
- Preserves: `RecurrentController(hidden_size=16, seed=0, learning_rate=0.1)`, `step`, `learn`, and `reset_state`.
- Preserves: `NeuralDecision` and all current `Action` numeric values.

- [ ] Add an adapter delegation test before refactoring. Monkeypatch `controller._policy.decide` with a recorder returning a `PolicyDecision`, then assert `step` passes the exact nine-element output of `encode_observation` and legal mask `(0, 1, 2, 3)`.
- [ ] Run `.venv/bin/pytest tests/test_controller.py tests/test_phase1_compatibility.py -q`. Expected RED: the current controller has no `_policy` attribute.
- [ ] Replace matrix ownership in `RecurrentController` with this exact construction and conversion:

```python
self._policy = RecurrentPolicy(
    input_size=9,
    action_count=len(Action),
    hidden_size=hidden_size,
    seed=seed,
    learning_rate=learning_rate,
    recurrent_radius=0.8,
)

def step(self, observation: GameObservation) -> NeuralDecision:
    result = self._policy.decide(
        encode_observation(observation), tuple(range(len(Action)))
    )
    return NeuralDecision(
        action=Action(result.action_index),
        logits=result.logits,
        hidden_state=result.hidden_state,
    )
```

Delegate `learn` and `reset_state` directly. Retain `hidden_size` and `learning_rate` public attributes with their existing values.

- [ ] Run the frozen tests immediately. Expected: the seed `41` logits/hidden state match within `1e-12`, and the complete seed `7` benchmark matches its literal baseline. If they fail, fix draw order or numeric operations; do not update the golden values.
- [ ] Run the entire Phase 1 suite: `.venv/bin/pytest tests/test_encoding.py tests/test_controller.py tests/test_environment.py tests/test_experiment.py tests/test_benchmark.py tests/test_phase1_compatibility.py -q`.
- [ ] Run `.venv/bin/python scripts/benchmark.py` and compare its JSON to the Task 1 literal contract.
- [ ] Run `.venv/bin/ruff check src/neural_state_machine/controller.py tests/test_controller.py`.
- [ ] Commit:

```bash
git add src/neural_state_machine/controller.py tests/test_controller.py
git commit -m "refactor: adapt phase one controller to generic policy"
```

---

### Task 5: Implement the delayed-cue protocol without controller-visible state

**Files:**
- Create: `src/neural_state_machine/memory_task.py`
- Create: `tests/test_memory_task.py`
- Modify: `src/neural_state_machine/__init__.py`
- Test: `tests/test_memory_task.py`

**Interfaces:**
- Produces `Cue(IntEnum)` with `LEFT = 0`, `RIGHT = 1`.
- Produces frozen `DelayedCueEpisode(cue, delay_steps, cue_stimulus, delay_stimuli, decision_stimulus, correct_action_index)`.
- Produces `DelayedCueTask.build_episode(cue, delay_steps, rng)` and `DelayedCueTask.reward(episode, action_index)`.
- The cue and correct-action fields belong to the experiment fixture only; no method passes them to `RecurrentPolicy`.

- [ ] Write the exact protocol test first:

```python
def test_left_and_right_share_an_identical_decision_stimulus() -> None:
    task = DelayedCueTask()
    left = task.build_episode(Cue.LEFT, 3, np.random.default_rng(11))
    right = task.build_episode(Cue.RIGHT, 3, np.random.default_rng(11))

    np.testing.assert_array_equal(left.cue_stimulus, [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(right.cue_stimulus, [0.0, 1.0, 0.0, 0.0])
    np.testing.assert_array_equal(left.decision_stimulus, right.decision_stimulus)
    np.testing.assert_array_equal(left.decision_stimulus, [0.0, 0.0, 0.0, 1.0])
```

- [ ] Run `.venv/bin/pytest tests/test_memory_task.py -q`. Expected RED: the module does not exist.
- [ ] Implement a module-level read-only decision literal and a helper that always returns a fresh read-only copy. Cue vectors and every delay vector must also be independent read-only `float64` arrays.
- [ ] Add tests that each episode contains exactly `delay_steps` delay vectors and every delay vector has zeros at indices `0`, `1`, and `3`, with index `2` finite and within `[-0.25, 0.25]`.
- [ ] Add a same-seed generation test comparing every array with `np.array_equal`, and a different-seed test proving at least one delay distractor differs.
- [ ] Add reward truth tests: left/index `0` and right/index `1` return `+1.0`; the opposite legal index returns `-1.0`; all other values raise `ValueError`.
- [ ] Add validation tests: non-`Cue` cue values, delay lengths `0`, `6`, `True`, and `1.5`, plus non-`Generator` RNGs, all raise `ValueError`.
- [ ] Search the production module after implementation:

```bash
rg -n "last_cue|remembered|behavior_state|transition|delay_count|episode_phase" src/neural_state_machine
```

Expected: no controller or policy field stores any of these concepts. Descriptive documentation in `memory_task.py` is allowed; runtime state in `policy.py` or `controller.py` is not.

- [ ] Export `Cue`, `DelayedCueEpisode`, and `DelayedCueTask` from `__init__.py`.
- [ ] Run focused tests and Ruff.
- [ ] Commit:

```bash
git add src/neural_state_machine/memory_task.py src/neural_state_machine/__init__.py tests/test_memory_task.py
git commit -m "feat: add delayed cue experiment protocol"
```

---

### Task 6: Add balanced training, evaluation, and state-reset ablation

**Files:**
- Create: `src/neural_state_machine/memory_benchmark.py`
- Create: `tests/test_memory_benchmark.py`
- Modify: `src/neural_state_machine/__init__.py`
- Test: `tests/test_memory_benchmark.py`

**Interfaces:**
- Produces frozen `MemoryExperimentConfig` with defaults:

```python
@dataclass(frozen=True)
class MemoryExperimentConfig:
    hidden_size: int = 64
    learning_rate: float = 0.05
    recurrent_radius: float = 0.9
    training_episodes: int = 2_000
    epsilon_start: float = 0.25
    epsilon_end: float = 0.02
    evaluation_blocks: int = 20
```

- Produces frozen `AccuracyCount(correct: int, total: int)` with computed `accuracy`.
- Produces frozen internal `_RunResult` containing pre/post/reset counts, per-delay counts, training reward, final-block counts, and weight/choice digests.
- Produces `_run_once(seed, config) -> _RunResult`; public repeatability wrappers are added in Task 7.

- [ ] Write configuration tests first. `training_episodes` must be a positive multiple of ten; `evaluation_blocks`, `hidden_size` must be positive integers; rates/radius obey the policy bounds; epsilon endpoints are finite in `[0, 1]` and `epsilon_start >= epsilon_end`. Run focused tests and confirm RED because the module is absent. Add `AccuracyCount` tests requiring integer `0 <= correct <= total`, positive `total`, and `accuracy == correct / total`.
- [ ] Implement `_balanced_cases(rng)` as the exact Cartesian product of `(Cue.LEFT, Cue.RIGHT)` and delays `1..5`, then shuffle the ten pairs using the supplied generator. Test that every block contains each pair exactly once, never relies on order, and two same-seed generators produce the same order.
- [ ] Implement `_epsilon(episode_index, config)` with endpoints included:

```python
fraction = episode_index / (config.training_episodes - 1)
return config.epsilon_start + fraction * (
    config.epsilon_end - config.epsilon_start
)
```

Because `training_episodes` is a positive multiple of ten, the denominator is never zero. Test index `0`, index `training_episodes - 1`, and a midpoint.

- [ ] Implement `_run_episode(policy, task, episode, *, explore_probability, rng, learn, reset_before_decision=False)` with the only controller-facing sequence:

```python
policy.reset_state()
policy.advance(episode.cue_stimulus)
for delay_stimulus in episode.delay_stimuli:
    policy.advance(delay_stimulus)
if reset_before_decision:
    policy.reset_state()
decision = policy.decide(
    episode.decision_stimulus,
    (int(Cue.LEFT), int(Cue.RIGHT)),
    explore_probability=explore_probability,
    rng=rng if explore_probability > 0.0 else None,
)
reward = task.reward(episode, decision.action_index)
if learn:
    policy.learn(reward)
return decision.action_index, reward
```

Do not pass `episode.cue`, `delay_steps`, or `correct_action_index` to the policy.

- [ ] Add a spy-policy test that records method inputs and proves the sequence is cue `advance`, N delay `advance` calls, then one shared decision `decide`; no fixture label appears in a stimulus.
- [ ] Build immutable evaluation fixtures once using a distinct RNG lineage:

```python
train_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x54524149]))
evaluation_rng = np.random.default_rng(
    np.random.SeedSequence([seed, 0x4556414C])
)
```

For each evaluation block, generate all ten cue/delay pairs with new distractors. Reuse the resulting fixture tuple for pre-training recurrent evaluation, post-training recurrent evaluation, and post-training reset ablation.

- [ ] Implement `_evaluate(policy, task, fixtures, *, reset_before_decision)` by calling `_run_episode` with zero exploration, `rng=None`, `learn=False`, and the supplied reset flag. For ablation, the helper processes cue and delay normally, then calls `policy.reset_state()` immediately before `decide` on the shared decision vector. Count expected labels only from `episode.correct_action_index`, never from a prior prediction.
- [ ] Add a synthetic-policy evaluation test showing `correct` and `total` counts are literal integers and per-delay totals are balanced. With 20 blocks, overall total is `200`, each delay total is `40`, and left/right counts are equal.
- [ ] Add a reset-ablation invariant test using a real untrained or manually trained policy: all reset decisions must be the same action because the decision vector and starting hidden state are identical; over balanced fixtures, accuracy is exactly `100 / 200 == 0.5`.
- [ ] Implement training in complete ten-case blocks. Each episode builds fresh delay distractors, uses the current linear epsilon, gets reward `+1/-1`, and calls `learn` exactly once. Track integer total reward and the literal correct/total count for episodes `1990..1999`.
- [ ] Prove frozen reservoir weights without exposing them publicly. In `tests/test_memory_benchmark.py`, snapshot private input/recurrent matrices before training and use `np.array_equal` after training; also assert the output digest changes after at least one nonzero update.
- [ ] Add an evaluation immutability test comparing `output_weight_digest()` immediately before and after each evaluation mode.
- [ ] Implement `_run_once` in this exact order: create policy and fixed evaluation fixtures, pre-evaluate, train, post-evaluate recurrent, post-evaluate reset, compute digests. Choice digests are SHA-256 over the ordered byte sequence of selected indices; keep separate recurrent and reset choice digests.
- [ ] Export `AccuracyCount`, `MemoryExperimentConfig`, and the later public benchmark types from `__init__.py` only after import cycles are clean.
- [ ] Run `.venv/bin/pytest tests/test_memory_benchmark.py -q` and Ruff. At this task, do not yet assert the final `0.90/0.85` scientific thresholds; test mechanics and invariants only.
- [ ] Commit:

```bash
git add src/neural_state_machine/memory_benchmark.py src/neural_state_machine/__init__.py tests/test_memory_benchmark.py
git commit -m "feat: add delayed cue training and ablation"
```

---

### Task 7: Add repeatable three-seed acceptance benchmark and CLI

**Files:**
- Modify: `src/neural_state_machine/memory_benchmark.py`
- Modify: `tests/test_memory_benchmark.py`
- Modify: `src/neural_state_machine/__init__.py`
- Create: `scripts/benchmark_memory.py`
- Test: `tests/test_memory_benchmark.py`
- Test: `scripts/benchmark_memory.py`

**Interfaces:**
- Produces frozen `MemoryExperimentResult` with `seed`, `config`, all accuracy/count metrics, total training reward, final-block accuracy/count, three digests, and `repeatable`.
- Produces `run_memory_experiment(seed=7, config=MemoryExperimentConfig()) -> MemoryExperimentResult`.
- Produces `run_memory_benchmark(seeds: Sequence[int] = (7, 17, 29), config: MemoryExperimentConfig | None = None) -> dict[str, object]`.
- CLI prints one sorted, compact JSON object and exits nonzero if any acceptance rule fails.

- [ ] Implement `run_memory_experiment` by calling `_run_once` twice with newly constructed policies and RNGs. Set `repeatable = first == second`; return the first run's values plus that boolean. Do not compare a run object to itself or reuse a trained policy.
- [ ] Add a same-seed test asserting complete dataclass equality across two public calls, `repeatable is True`, and exact equality of weight/recurrent-choice/reset-choice digests.
- [ ] Add the committed acceptance test for seeds `7`, `17`, and `29`:

```python
@pytest.mark.parametrize("seed", [7, 17, 29])
def test_default_memory_experiment_meets_acceptance(seed: int) -> None:
    result = run_memory_experiment(seed)
    assert result.repeatable is True
    assert result.post_training.accuracy >= 0.90
    assert result.state_reset.accuracy == 0.50
    assert all(score.accuracy >= 0.85 for _, score in result.per_delay)
```

- [ ] Run the acceptance test alone. If any seed fails, stop implementation and record seed, overall accuracy, each delay accuracy, and the exact digests. Do not change thresholds, seeds, task difficulty, training episode count, reservoir radius, hidden size, or learning rule without returning to the design-review gate.
- [ ] Implement JSON conversion with stable field order through `json.dumps(payload, sort_keys=True, separators=(",", ":"))`. Include configuration, counts and accuracies, training metrics, digests, per-seed pass flags, and a top-level `all_passed`.
- [ ] Add a subprocess test that runs `scripts/benchmark_memory.py` twice, parses both lines as JSON, asserts byte-for-byte stdout equality, confirms seeds `[7, 17, 29]`, and confirms `all_passed is True`.
- [ ] Add a CLI `main() -> int` that returns `0` only when all acceptance checks pass and uses `raise SystemExit(main())` under the normal module guard.
- [ ] Run:

```bash
.venv/bin/pytest tests/test_memory_benchmark.py -q
.venv/bin/python scripts/benchmark_memory.py
.venv/bin/ruff check src/neural_state_machine/memory_benchmark.py tests/test_memory_benchmark.py scripts/benchmark_memory.py
```

Expected: all three seeds pass, ablation is exactly `0.5`, and the second benchmark invocation is byte-identical.

- [ ] Commit:

```bash
git add src/neural_state_machine/memory_benchmark.py src/neural_state_machine/__init__.py tests/test_memory_benchmark.py scripts/benchmark_memory.py
git commit -m "feat: add reproducible memory acceptance benchmark"
```

---

### Task 8: Document, wire CI, and perform the no-hidden-FSM audit

**Files:**
- Modify: `README.md`
- Modify: `.github/workflows/ci.yml`
- Modify only if an audit exposes a defect: Phase 2 source or test files from Tasks 2-7

**Interfaces:**
- Produces user commands for Phase 1 and Phase 2 benchmarks.
- Produces CI evidence on Python 3.10, 3.11, and 3.12.
- Produces an auditable interpretation boundary: remembered cue, not general intelligence or a biological fly brain.

- [ ] Add a README Phase 2 section containing:
  - the cue/delay/decision stimulus table;
  - the 2,000-episode balanced training protocol;
  - recurrent versus state-reset evaluation;
  - thresholds and fixed seeds `7`, `17`, `29`;
  - exact commands `python scripts/benchmark.py` and `python scripts/benchmark_memory.py`;
  - the statement that only output weights learn;
  - the prohibition on explicit cue memory/FSM state;
  - the narrow supported claim and the Phase 3/fly-brain boundary.
- [ ] Modify `.github/workflows/ci.yml` by retaining the existing install, unit-test, Ruff, and Phase 1 benchmark steps, then adding:

```yaml
      - name: Delayed-cue memory benchmark
        run: python scripts/benchmark_memory.py
```

- [ ] Run the complete local verification in this order and preserve the output for the handoff:

```bash
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/pytest -q
.venv/bin/ruff check .
.venv/bin/python scripts/benchmark.py
.venv/bin/python scripts/benchmark_memory.py
git diff --check
```

- [ ] Compare Phase 1 benchmark JSON numerically against Task 1. Any mismatch blocks completion; do not revise the baseline.
- [ ] Run the controller-state audit:

```bash
rg -n "last_cue|remembered_cue|current_phase|episode_phase|delay_count|attack_state|retreat_state|heal_state|behavior_tree|transition_table" src/neural_state_machine
```

Expected: no matches in `policy.py` or `controller.py`. A task-protocol description elsewhere is acceptable only if it is not stored in or passed to the policy.

- [ ] Audit task inputs directly: collect all decision stimuli from both cues and all delays and assert one unique byte representation; inspect `_run_episode` to confirm only numeric stimulus arrays and legal indices cross the policy boundary.
- [ ] Commit documentation and CI only after every command passes:

```bash
git add README.md .github/workflows/ci.yml
git commit -m "docs: add delayed cue memory benchmark"
```

- [ ] Push the exact local head to `origin/experiment/neural-state-machine` without touching `master`.
- [ ] Confirm `git rev-list --parents --max-count=1 HEAD` still shows an orphan root with zero parents and `git merge-base HEAD origin/master` exits nonzero.
- [ ] Inspect GitHub Actions for the exact pushed SHA. Require all three Python matrix jobs to pass unit tests, Ruff, the unchanged Phase 1 benchmark, and the new memory benchmark.
- [ ] If CI differs from local results, use systematic debugging and fix the actual portability defect in a new focused commit. Do not weaken the scientific acceptance criteria.

## Completion handoff

Completion evidence must include:

- the exact final commit SHA and remote branch;
- complete pytest count and result;
- Ruff result;
- unchanged Phase 1 benchmark JSON;
- Phase 2 metrics for seeds `7`, `17`, and `29`, including all five per-delay accuracies and exact `0.50` reset ablations;
- repeatability and digest evidence;
- exact-head GitHub Actions URL;
- a clear statement if the hypothesis failed instead of claiming success.

Only after this evidence passes should a new Phase 3 design session consider multiple remembered cues or conflicting attack/retreat/heal objectives.
