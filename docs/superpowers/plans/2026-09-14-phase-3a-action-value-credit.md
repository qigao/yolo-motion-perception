# Phase 3A Normalized Action-Value Credit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether a normalized action-local TD(0) readout can acquire the vanished-cue mapping from immediate scalar reward on all three frozen seeds, while preserving prior evidence and proving the update kernel's exact real-number algebra in Lean.

**Architecture:** Keep the recurrent policy, delayed-cue task, Phase 2B learner, and Phase 2C diagnostics byte-for-byte frozen. Add a zero-initialized linear action-value table over the augmented decision feature `[hidden; 1]`, train it with seeded uniform actions and one immediate terminal reward, and measure it through a separate deterministic benchmark with reset and within-block shuffled-reward controls. Treat Lean proofs, NumPy correspondence tests, and behavioral evidence as three distinct verification layers.

**Tech Stack:** Python 3.10–3.12, NumPy, pytest, Ruff, setuptools, Lean 4.30.0, Mathlib commit `c5ea00351c28e24afc9f0f84379aa41082b1188f`, Lake, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-14-phase-3a-action-value-credit-design.md`

## Global Constraints

- Work only on the standalone orphan branch `experiment/neural-state-machine`; do not merge, rebase onto, copy from, or modify `master`.
- Commit and push every completed task to `origin/experiment/neural-state-machine`; never force-push.
- NumPy remains the only Python runtime dependency. Lean and Mathlib live only under `formal/` and in their independent CI job.
- Keep `src/neural_state_machine/policy.py`, `reward_readout.py`, `reward_learning.py`, `learning_diagnostics.py`, `memory_task.py`, `memory_probe.py`, and `memory_benchmark.py` byte-for-byte unchanged.
- Keep `docs/experiments/phase-2b-failure.json`, `docs/experiments/phase-2c-diagnostics.json`, and every Phase 1/2 test expectation byte-for-byte unchanged.
- The learner receives only a finite decision-time hidden vector, legal numeric action indices, a NumPy Generator for training selection, and one finite scalar terminal reward.
- Never pass cue identity, correct action, delay, task phase, probe output, observation history, or episode objects into the learner.
- Do not infer a supervised target from binary reward, request counterfactual rewards, alter the recurrent substrate, add actor-critic/BPTT/STDP, or add temporal eligibility decay.
- Call delays `cue-to-decision delays`; Phase 3A has immediate action-to-reward feedback and must never be described as delayed-reward learning.
- Fixed scientific configuration: `hidden_size=64`, `recurrent_radius=0.9`, `step_size=0.1`, `training_episodes=2000`, `evaluation_blocks=20`, `checkpoint_interval=100`, delays `[1,2,3,4,5]`, ordered seeds `[7,17,29]`.
- Fixed seed lineages: training fixtures `[seed, 0x54524149]`, evaluation fixtures `[seed, 0x4556414C]`, Phase 3A behavior actions `[seed, 0x33414354]`, and Phase 3A reward shuffle `[seed, 0x33534846]`.
- Normal and shuffled paths reconstruct separate behavior generators from the same `[seed, 0x33414354]` lineage; their complete action sequences must be exactly equal.
- The shuffled path consumes a within-each-ten-case-block permutation of the literal normal rewards. Each corresponding block must preserve its exact reward multiset.
- Per-seed acceptance: post-training at least `180/200`, every delay at least `34/40`, reset exactly `100/200`, every reset delay exactly `20/40`, reset hidden vectors exactly equal, shuffled below `150/200`, changed value parameters, unchanged recurrent matrices, no pending feedback, and exact independent-run equality.
- Pooled shuffled accuracy must be in `[0.40, 0.60]`; all seed gates and the pooled gate must pass independently.
- A behavioral miss is committed as truthful failure evidence and returns to design review. Do not tune the step size, seeds, counts, task, reservoir, or thresholds after observing it.
- The formal result proves exact real-number equations only. Never claim that Lean verified NumPy, stochastic convergence, or behavioral acceptance.

## File Structure

| File | Responsibility |
|---|---|
| `src/neural_state_machine/action_value.py` | Validate inputs, own the action-value matrix and one pending credit record, select uniform/greedy actions, and apply normalized terminal TD(0). |
| `tests/test_action_value.py` | Unit contracts for construction, ownership, validation atomicity, RNG behavior, lifecycle, update equations, and digests. |
| `src/neural_state_machine/action_value_benchmark.py` | Fixed fixtures, policy replay, normal/shuffled training, checkpoints, controls, results, acceptance, serialization, and portable projection. |
| `tests/test_action_value_benchmark.py` | Boundary spies, seed-lineage/fairness controls, deterministic replay, gates, schema, frozen matrices, and public benchmark behavior. |
| `scripts/benchmark_action_value.py` | Print one compact deterministic JSON payload; optionally write only the approved evidence path atomically; exit from `all_passed`. |
| `scripts/verify_action_value_evidence.py` | Re-run twice, enforce same-environment equality and local float/matrix integrity, then compare the portable projection with committed evidence. |
| `docs/experiments/phase-3a-action-value.json` | Immutable measured payload generated once from the approved implementation without tuning. |
| `formal/lakefile.toml` | Define the standalone proof package and pin Mathlib by commit. |
| `formal/lean-toolchain` | Pin `leanprover/lean4:v4.30.0`. |
| `formal/lake-manifest.json` | Commit Lake's resolved dependency graph. |
| `formal/NeuralStateMachine/ActionValue.lean` | Define the abstract update and prove all seven algebraic obligations. |
| `formal/NeuralStateMachine/AxiomAudit.lean` | Import the public theorems and print their axioms in build logs. |
| `src/neural_state_machine/__init__.py` | Export only approved Phase 3A public types and entry points. |
| `.github/workflows/ci.yml` | Preserve the Python matrix and add Phase 3A verification plus an independent pinned Lean job. |
| `README.md` | Report measured Phase 3A results and the immediate-reward/Lean claim boundaries after evidence exists. |

---

### Task 1: Add Immutable Action-Value Selection

**Files:**
- Create: `src/neural_state_machine/action_value.py`
- Create: `tests/test_action_value.py`

**Interfaces:**
- Consumes: finite float64-compatible `hidden_state`, validated legal action indices, and `np.random.Generator`.
- Produces: `ActionValueDecision`, `NormalizedActionValue.parameter_snapshot()`, `parameter_digest()`, `select_greedy()`, `select_for_training()`, and `has_pending_feedback`.

- [ ] **Step 1: Write failing constructor and initialization tests**

Add tests with this public surface:

```python
import numpy as np
import pytest

from neural_state_machine.action_value import NormalizedActionValue


def test_action_value_starts_at_exact_zero() -> None:
    learner = NormalizedActionValue(hidden_size=3, action_count=2, step_size=0.1)
    snapshot = learner.parameter_snapshot()
    assert snapshot.dtype == np.float64
    assert snapshot.shape == (2, 4)
    assert np.array_equal(snapshot, np.zeros((2, 4), dtype=np.float64))
    assert snapshot.flags.writeable is False
    assert learner.has_pending_feedback is False


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"hidden_size": True}, "hidden_size"),
        ({"hidden_size": 0}, "hidden_size"),
        ({"action_count": True}, "action_count"),
        ({"action_count": 1}, "action_count"),
        ({"step_size": True}, "step_size"),
        ({"step_size": 0.0}, "step_size"),
        ({"step_size": 1.0000001}, "step_size"),
        ({"step_size": float("nan")}, "step_size"),
    ],
)
def test_constructor_rejects_invalid_values(kwargs: dict[str, object], message: str) -> None:
    values = {"hidden_size": 3, "action_count": 2, "step_size": 0.1}
    values.update(kwargs)
    with pytest.raises(ValueError, match=message):
        NormalizedActionValue(**values)
```

- [ ] **Step 2: Run the constructor tests and confirm RED**

Run:

```bash
.venv/bin/pytest tests/test_action_value.py -q
```

Expected: collection fails because `neural_state_machine.action_value` does not exist.

- [ ] **Step 3: Implement construction, immutable snapshots, and canonical digesting**

Create these exact public and private shapes:

```python
@dataclass(frozen=True)
class ActionValueDecision:
    action_index: int
    action_values: np.ndarray


@dataclass(frozen=True)
class _PendingCredit:
    action_index: int
    feature: np.ndarray
    denominator: float
    prediction: float


class NormalizedActionValue:
    def __init__(self, hidden_size: int, action_count: int, step_size: float = 0.1) -> None:
        self.hidden_size = validated_positive_integer(hidden_size, "hidden_size")
        self.action_count = validated_action_count(action_count)
        self.step_size = validated_step_size(step_size)
        self._weights = np.zeros(
            (self.action_count, self.hidden_size + 1), dtype=np.float64
        )
        self._pending: _PendingCredit | None = None

    @property
    def has_pending_feedback(self) -> bool:
        return self._pending is not None

    def parameter_snapshot(self) -> np.ndarray:
        return readonly_float64_copy(self._weights)

    def parameter_digest(self) -> str:
        values = np.ascontiguousarray(self._weights, dtype=np.float64)
        digest = hashlib.sha256()
        digest.update(str(values.shape).encode("ascii"))
        digest.update(values.tobytes(order="C"))
        return digest.hexdigest()
```

Use strict `type(value) is int` integer checks and reject booleans explicitly for numeric fields.

- [ ] **Step 4: Write failing feature, legal-action, and greedy-selection tests**

Cover rank, shape, dtype compatibility, non-finite elements, empty/duplicate/non-integer/out-of-range legal indices, exact tie-breaking to the lowest numeric legal action, masking of illegal actions, and evaluation non-mutation. Independently modify a returned `action_values` copy attempt and prove neither the learner nor an earlier snapshot changes.

Use an internal test helper that installs a known matrix only through a fresh object and direct private setup in the test:

```python
learner = NormalizedActionValue(2, 3)
learner._weights[:] = np.array(
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
    dtype=np.float64,
)
decision = learner.select_greedy(np.array([2.0, 1.0]), (2, 0))
assert decision.action_index == 0
assert np.isneginf(decision.action_values[1])
assert learner.has_pending_feedback is False
```

- [ ] **Step 5: Implement feature augmentation, legal masking, and greedy selection**

Implement the common validated path exactly as:

```python
def _feature(self, hidden_state: object) -> np.ndarray:
    try:
        hidden = np.asarray(hidden_state, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("hidden_state must be float64-compatible") from exc
    if hidden.ndim != 1 or hidden.shape != (self.hidden_size,):
        raise ValueError(f"hidden_state must have shape ({self.hidden_size},)")
    if not np.all(np.isfinite(hidden)):
        raise ValueError("hidden_state must contain only finite values")
    feature = np.concatenate((hidden, np.array([1.0], dtype=np.float64)))
    feature.flags.writeable = False
    return feature

def select_greedy(
    self, hidden_state: object, legal_action_indices: object
) -> ActionValueDecision:
    feature = self._feature(hidden_state)
    legal = self._legal_actions(legal_action_indices)
    values = self._weights @ feature
    masked = np.full(self.action_count, -np.inf, dtype=np.float64)
    masked[list(legal)] = values[list(legal)]
    maximum = max(masked[index] for index in legal)
    selected = min(index for index in legal if masked[index] == maximum)
    return ActionValueDecision(selected, readonly_float64_copy(masked))
```

- [ ] **Step 6: Write failing uniform-training-selection and lifecycle tests**

Require:

- two separately constructed `default_rng(19)` generators produce the same 1,000-action sequence;
- all three legal actions appear with counts inside `[280, 390]` while a fourth illegal action never appears;
- radically different installed Q matrices yield the same sequence under independently reconstructed same-seed generators;
- RNG validation occurs before advancing RNG state;
- a valid selection creates exactly one pending record;
- a second training selection raises `RuntimeError` without consuming the original record or advancing the supplied generator;
- mutating the caller's hidden array after selection cannot alter stored credit;
- `select_greedy` neither creates nor replaces pending credit.

Consume each temporary pending record in the sequence test by calling the private test-only helper `_clear_pending_for_test()` defined inside the test module as `learner._pending = None`; production code must not expose a discard-credit API.

- [ ] **Step 7: Implement uniform selection after complete validation**

Validate hidden, legal indices, RNG type, and absence of pending feedback before the first RNG draw. Sample an offset with `rng.integers(len(legal))`, not from Q values:

```python
def select_for_training(
    self,
    hidden_state: object,
    legal_action_indices: object,
    rng: np.random.Generator,
) -> ActionValueDecision:
    feature = self._feature(hidden_state)
    legal = self._legal_actions(legal_action_indices)
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a numpy.random.Generator")
    if self._pending is not None:
        raise RuntimeError("feedback is already pending")
    action_index = legal[int(rng.integers(len(legal)))]
    raw_values = self._weights @ feature
    masked = np.full(self.action_count, -np.inf, dtype=np.float64)
    masked[list(legal)] = raw_values[list(legal)]
    denominator = float(np.dot(feature, feature))
    self._pending = _PendingCredit(
        action_index=action_index,
        feature=readonly_float64_copy(feature),
        denominator=denominator,
        prediction=float(raw_values[action_index]),
    )
    return ActionValueDecision(action_index, readonly_float64_copy(masked))
```

- [ ] **Step 8: Run focused tests and commit the selection layer**

Run:

```bash
.venv/bin/pytest tests/test_action_value.py -q
.venv/bin/ruff check src/neural_state_machine/action_value.py tests/test_action_value.py
git diff --check
git add src/neural_state_machine/action_value.py tests/test_action_value.py
git commit -m "feat: add normalized action-value selection"
git push origin experiment/neural-state-machine
```

---

### Task 2: Apply Normalized Terminal TD(0)

**Files:**
- Modify: `src/neural_state_machine/action_value.py`
- Modify: `tests/test_action_value.py`
- Modify: `src/neural_state_machine/__init__.py`

**Interfaces:**
- Consumes: the one `_PendingCredit` created by `select_for_training()` and one finite scalar reward.
- Produces: immutable `ActionValueUpdate`, `NormalizedActionValue.learn(reward)`, and package exports.

- [ ] **Step 1: Write failing lifecycle and invalid-feedback atomicity tests**

Add tests that `learn(1.0)` before selection and a second learn after consumption raise `RuntimeError`. For `None`, an object, `True`, `nan`, `inf`, and `-inf`, snapshot weights and the pending record before calling `learn`; require a `ValueError`, byte-identical parameters, and `has_pending_feedback is True`, then prove `learn(1.0)` still succeeds.

- [ ] **Step 2: Write the independent normalized-update correspondence test**

Use nontrivial weights, a feature `[0.25, -0.5, 1.0]`, and a deterministic selected action. Snapshot `before`, obtain the decision prediction, then independently compute:

```python
feature = np.array([0.25, -0.5, 1.0], dtype=np.float64)
denominator = float(np.dot(feature, feature))
expected = before.copy()
td_error = reward - decision.action_values[decision.action_index]
expected[decision.action_index] += 0.1 * td_error * feature / denominator
update = learner.learn(reward)
np.testing.assert_allclose(
    learner.parameter_snapshot(), expected, rtol=0.0, atol=1e-15
)
assert update.td_error == pytest.approx(td_error, rel=0.0, abs=1e-15)
```

The expected value must be reconstructed from immutable public before-state and decision snapshots, never from a production helper.

- [ ] **Step 3: Write equation, row-isolation, and identity tests**

For several finite features and rewards, require:

```python
q_before = decision.action_values[action]
learner.learn(reward)
q_after = learner.select_greedy(hidden, legal).action_values[action]
assert q_after == pytest.approx(q_before + alpha * (reward - q_before), abs=1e-12)
assert reward - q_after == pytest.approx(
    (1.0 - alpha) * (reward - q_before), abs=1e-12
)
```

Also require all unselected rows to be byte-identical, exact zero TD error to leave the full matrix and digest byte-identical, `step_size=1.0` to interpolate the selected same-sample prediction to reward within `1e-12`, and no reward clipping by distinguishing updates for `reward=1.0` and `reward=100.0`.

- [ ] **Step 4: Run the update tests and confirm RED**

Run:

```bash
.venv/bin/pytest tests/test_action_value.py -q
```

Expected: failures identify the absent `ActionValueUpdate` and `learn()` behavior.

- [ ] **Step 5: Implement the one-shot normalized update**

Use this result and method shape:

```python
@dataclass(frozen=True)
class ActionValueUpdate:
    action_index: int
    prediction_before: float
    reward: float
    td_error: float


def learn(self, reward: object) -> ActionValueUpdate:
    if self._pending is None:
        raise RuntimeError("learning requires pending feedback")
    reward_value = validated_finite_scalar(reward, "reward")
    pending = self._pending
    td_error = reward_value - pending.prediction
    self._weights[pending.action_index] += (
        self.step_size * td_error * pending.feature / pending.denominator
    )
    self._pending = None
    return ActionValueUpdate(
        action_index=pending.action_index,
        prediction_before=pending.prediction,
        reward=reward_value,
        td_error=td_error,
    )
```

Validate the reward before copying or clearing `_pending`. Do not clip the reward and do not update any row other than `pending.action_index`.

- [ ] **Step 6: Add package exports and identity tests**

Export exactly:

```python
from .action_value import ActionValueDecision, ActionValueUpdate, NormalizedActionValue
```

Add tests asserting package-root identities equal the defining module objects. Do not export `_PendingCredit` or validation helpers.

- [ ] **Step 7: Run focused regression checks and commit**

Run:

```bash
.venv/bin/pytest tests/test_action_value.py tests/test_policy.py tests/test_reward_readout.py tests/test_phase1_compatibility.py -q
.venv/bin/ruff check src/neural_state_machine/action_value.py src/neural_state_machine/__init__.py tests/test_action_value.py
git diff --check
git add src/neural_state_machine/action_value.py src/neural_state_machine/__init__.py tests/test_action_value.py
git commit -m "feat: add normalized terminal td zero update"
git push origin experiment/neural-state-machine
```

---

### Task 3: Build Fixed Fixtures and Mutation-Free Evaluation

**Files:**
- Create: `src/neural_state_machine/action_value_benchmark.py`
- Create: `tests/test_action_value_benchmark.py`

**Interfaces:**
- Consumes: frozen `reward_learning._build_fixtures()`, `_decision_hidden()`, `_matrix_copies()`, `_matrix_digests()`, `_require_frozen()`, `AccuracyCount`, `DelayedCueTask`, `RecurrentPolicy`, and `NormalizedActionValue`.
- Produces: `ActionValueBenchmarkConfig`, `_Evaluation`, `_FixtureBundle`, `_build_fixture_bundle()`, and `_evaluate()`.

- [ ] **Step 1: Write failing fixed-configuration validation tests**

Define this frozen configuration:

```python
@dataclass(frozen=True)
class ActionValueBenchmarkConfig:
    hidden_size: int = 64
    recurrent_radius: float = 0.9
    step_size: float = 0.1
    training_episodes: int = 2_000
    evaluation_blocks: int = 20
    checkpoint_interval: int = 100
```

Test strict boolean rejection, positive integer constraints, `training_episodes % 10 == 0`, `evaluation_blocks > 0`, `checkpoint_interval % 10 == 0`, `training_episodes % checkpoint_interval == 0`, finite `recurrent_radius` in `[0,1)`, and finite `step_size` in `(0,1]`. Assert the dataclass exposes no seed-lineage fields.

- [ ] **Step 2: Write failing fixture-lineage and balance tests**

Define:

```python
@dataclass(frozen=True)
class _FixtureBundle:
    training: tuple[DelayedCueEpisode, ...]
    evaluation: tuple[DelayedCueEpisode, ...]
    training_fixture_digest: str
    evaluation_fixture_digest: str
```

For each seed, independently reconstruct bundles and require equality of semantic fixtures and digests. Require 200 balanced ten-case training blocks, 20 balanced ten-case evaluation blocks, 100 evaluation labels per action, 40 evaluation rows per delay, immutable stimulus arrays, and exact equality with frozen Phase 2B fixture construction from lineages `[seed, 0x54524149]` and `[seed, 0x4556414C]`.

- [ ] **Step 3: Implement configuration and fixture construction**

Construct no label-derived feature. The fixture digest must hash cue/delay/scoring metadata and all stimulus array shapes/bytes in episode order:

```python
def _build_fixture_bundle(
    seed: int, config: ActionValueBenchmarkConfig
) -> _FixtureBundle:
    task = DelayedCueTask()
    training_rng = np.random.default_rng(
        np.random.SeedSequence([seed, 0x54524149])
    )
    evaluation_rng = np.random.default_rng(
        np.random.SeedSequence([seed, 0x4556414C])
    )
    training = _build_fixtures(
        task, training_rng, config.training_episodes // 10
    )
    evaluation = _build_fixtures(task, evaluation_rng, config.evaluation_blocks)
    return _FixtureBundle(
        training=training,
        evaluation=evaluation,
        training_fixture_digest=_episode_digest(training),
        evaluation_fixture_digest=_episode_digest(evaluation),
    )
```

- [ ] **Step 4: Write failing evaluation-boundary and reset tests**

Use recording doubles to require `_evaluate()` calls only `policy.reset_state()`, `policy.advance(numeric_stimulus)`, and `learner.select_greedy(hidden, (0, 1))`. It must never call `select_for_training()` or `learn()`, and the learner must never receive an episode, cue, correct action, delay, or phase.

With a real policy and zero-valued learner, require ordinary and reset evaluation each choose action zero for all 200 balanced cases; reset hidden vectors are exactly equal; reset is `100/200` and every delay `20/40`; parameters, pending state, and all three policy matrices are unchanged.

- [ ] **Step 5: Implement immutable evaluation summaries**

Use:

```python
@dataclass(frozen=True)
class _Evaluation:
    overall: AccuracyCount
    per_delay: tuple[tuple[int, AccuracyCount], ...]
    actions: tuple[int, ...]
    action_digest: str
    hidden_digest: str
    all_hidden_equal: bool
    margin_mean: float
    margin_p10: float
    margin_minimum: float
```

Compute the correct-action margin only after `select_greedy()` returns:

```python
correct = episode.correct_action_index
other = 1 - correct
margin = float(decision.action_values[correct] - decision.action_values[other])
```

The metadata is used solely for scoring and stratification after action selection.

- [ ] **Step 6: Run focused tests and commit the protocol base**

Run:

```bash
.venv/bin/pytest tests/test_action_value_benchmark.py -q
.venv/bin/ruff check src/neural_state_machine/action_value_benchmark.py tests/test_action_value_benchmark.py
git diff --check
git add src/neural_state_machine/action_value_benchmark.py tests/test_action_value_benchmark.py
git commit -m "feat: add phase three-a evaluation protocol"
git push origin experiment/neural-state-machine
```

---

### Task 4: Add Fair Normal and Shuffled-Reward Training

**Files:**
- Modify: `src/neural_state_machine/action_value_benchmark.py`
- Modify: `tests/test_action_value_benchmark.py`

**Interfaces:**
- Consumes: one shared immutable fixture tuple, independent same-lineage action RNGs, one independent shuffle RNG, frozen recurrent policies, and independent value learners.
- Produces: `_TrainingTrace`, `ActionValueCheckpoint`, `_train_normal()`, `_train_shuffled()`, and `_run_once()`.

- [ ] **Step 1: Write the scalar-feedback boundary spy test**

Record every learner call for a 20-episode configuration. Require exactly this order per episode:

```text
policy reset and numeric stimulus replay
learner.select_for_training(hidden, (0, 1), action_rng)
task.reward(episode, selected_action)
learner.learn(literal_scalar_reward)
post-update bookkeeping
```

Assert `select_for_training()` sees only the numeric hidden vector, `(0,1)`, and a Generator; `learn()` sees only a float; task metadata is accessed only after the selection returns.

- [ ] **Step 2: Write failing same-action-schedule and generator-ownership tests**

Construct normal and shuffled action generators separately from the identical `SeedSequence([seed, 0x33414354])`. Require:

```python
assert normal_actions == shuffled_actions
assert normal_action_digest == shuffled_action_digest
assert normal_action_rng is not shuffled_action_rng
```

Use a state-recording wrapper to prove neither path shares a Generator object or advanced state and that changing installed Q values does not change the sampled action schedule.

- [ ] **Step 3: Write failing literal-reward permutation tests**

First complete the normal path while recording every `(action, literal reward)` pair. Before starting the shuffled path, permute the recorded literal normal rewards within each consecutive ten-case block using a separately constructed `default_rng(SeedSequence([seed, 0x33534846]))`. Replay the same immutable fixtures through a fresh policy and use a separately reconstructed same-lineage action generator.

Require for every block:

```python
assert sorted(normal_rewards[start:start + 10]) == sorted(
    shuffled_rewards[start:start + 10]
)
```

Also require at least one block order differs, the full normal/shuffled reward digests are recorded, and a shuffled-training task double whose `reward()` raises is accepted because shuffled training never queries task reward.

- [ ] **Step 4: Implement immutable schedules and training traces**

Use these shapes:

```python
@dataclass(frozen=True)
class ActionValueCheckpoint:
    episode: int
    accuracy: AccuracyCount
    margin_mean: float
    margin_p10: float
    margin_minimum: float
    td_error_mean: float
    td_error_abs_mean: float
    td_error_p90: float
    td_error_maximum: float


@dataclass(frozen=True)
class _TrainingTrace:
    actions: tuple[int, ...]
    rewards: tuple[float, ...]
    action_counts: tuple[tuple[int, int], ...]
    action_digest: str
    reward_digest: str
    checkpoints: tuple[ActionValueCheckpoint, ...]
    final_parameter_digest: str
    pending_feedback: bool
```

Correct action and delay remain local bookkeeping variables consumed only after the learner update for checkpoint scoring; never pass them into learner methods.

- [ ] **Step 5: Implement checkpoint collection without mutation**

Every 100 episodes, summarize only the completed block since the prior checkpoint. Compute TD statistics from returned `ActionValueUpdate.td_error`. Compute correct-action value margins by calling `select_greedy()` on already materialized hidden copies after feedback has been consumed. Snapshot the learner before and after checkpoint collection and require identical parameter digests and `has_pending_feedback is False`.

- [ ] **Step 6: Write failing recurrent-freeze and assembly tests**

For `_run_once(seed, config)`, require separately constructed normal/shuffled policies and learners; byte snapshots and digests of input, recurrent, and legacy output matrices before/after each path; different learner identities; changed final learner digests; no pending feedback; identical normal/shuffled action sequences; per-block reward-multiset equality; and exact evaluation fixture reuse.

- [ ] **Step 7: Implement one complete run**

The one-run order is fixed:

```python
fixtures = _build_fixture_bundle(seed, config)
normal_policy = _new_policy(seed, config)
shuffled_policy = _new_policy(seed, config)
normal_learner = _new_learner(config)
shuffled_learner = _new_learner(config)
pre_training = _evaluate(
    normal_policy,
    normal_learner,
    fixtures.evaluation,
    reset_before_decision=False,
)
normal_training = _train_normal(
    normal_policy,
    normal_learner,
    task,
    fixtures.training,
    normal_action_rng,
    config,
)
shuffled_rewards = _permute_reward_blocks(
    normal_training.rewards, shuffle_rng, block_size=10
)
shuffled_training = _train_shuffled(
    shuffled_policy,
    shuffled_learner,
    fixtures.training,
    shuffled_action_rng,
    shuffled_rewards,
    config,
)
post_training = _evaluate(
    normal_policy,
    normal_learner,
    fixtures.evaluation,
    reset_before_decision=False,
)
state_reset = _evaluate(
    normal_policy,
    normal_learner,
    fixtures.evaluation,
    reset_before_decision=True,
)
shuffled_control = _evaluate(
    shuffled_policy,
    shuffled_learner,
    fixtures.evaluation,
    reset_before_decision=False,
)
```

`_train_normal()` and `_train_shuffled()` each select once and consume feedback once per episode. The shuffled helper receives the already materialized reward tuple and never calls `task.reward()`.

- [ ] **Step 8: Run focused tests and commit the training controls**

Run:

```bash
.venv/bin/pytest tests/test_action_value.py tests/test_action_value_benchmark.py -q
.venv/bin/ruff check src/neural_state_machine/action_value_benchmark.py tests/test_action_value_benchmark.py
git diff --check
git add src/neural_state_machine/action_value_benchmark.py tests/test_action_value_benchmark.py
git commit -m "feat: add fair action-value reward controls"
git push origin experiment/neural-state-machine
```

---

### Task 5: Add Public Results, Acceptance Gates, and JSON CLI

**Files:**
- Modify: `src/neural_state_machine/action_value_benchmark.py`
- Modify: `src/neural_state_machine/__init__.py`
- Modify: `tests/test_action_value_benchmark.py`
- Create: `scripts/benchmark_action_value.py`

**Interfaces:**
- Consumes: `_run_once()` and fixed acceptance constants.
- Produces: `ActionValueExperimentResult`, `run_action_value_experiment()`, `run_action_value_benchmark()`, stable JSON, and CLI exit status.

- [ ] **Step 1: Define and test the public immutable result contract**

Add this result shape, using tuples rather than mutable lists internally:

```python
@dataclass(frozen=True)
class ActionValueExperimentResult:
    seed: int
    config: ActionValueBenchmarkConfig
    pre_training: AccuracyCount
    post_training: AccuracyCount
    state_reset: AccuracyCount
    shuffled_control: AccuracyCount
    per_delay: tuple[tuple[int, AccuracyCount], ...]
    reset_per_delay: tuple[tuple[int, AccuracyCount], ...]
    shuffled_per_delay: tuple[tuple[int, AccuracyCount], ...]
    normal_checkpoints: tuple[ActionValueCheckpoint, ...]
    shuffled_checkpoints: tuple[ActionValueCheckpoint, ...]
    normal_action_counts: tuple[tuple[int, int], ...]
    shuffled_action_counts: tuple[tuple[int, int], ...]
    normal_actions: tuple[int, ...]
    shuffled_actions: tuple[int, ...]
    normal_action_digest: str
    shuffled_action_digest: str
    normal_reward_digest: str
    shuffled_reward_digest: str
    action_sequences_equal: bool
    reward_block_multisets_equal: bool
    initial_parameter_digest: str
    normal_parameter_digest: str
    shuffled_parameter_digest: str
    normal_matrix_digests_before: tuple[str, str, str]
    normal_matrix_digests_after: tuple[str, str, str]
    shuffled_matrix_digests_before: tuple[str, str, str]
    shuffled_matrix_digests_after: tuple[str, str, str]
    training_fixture_digest: str
    evaluation_fixture_digest: str
    decision_hidden_digest: str
    reset_hidden_digest: str
    post_margin_mean: float
    post_margin_p10: float
    post_margin_minimum: float
    all_reset_hidden_equal: bool
    normal_pending_feedback: bool
    shuffled_pending_feedback: bool
    repeatable: bool
```

Test exact field types, ordered delay tuples, 20 checkpoints ending at episode 2,000, and read-only nested data.

- [ ] **Step 2: Add seed/config validation and independent-run repeatability**

Implement:

```python
def run_action_value_experiment(
    seed: int = 7,
    config: ActionValueBenchmarkConfig | None = None,
) -> ActionValueExperimentResult:
    resolved_seed = _validated_seed(seed)
    resolved_config = _validated_config(config)
    first = _run_once(resolved_seed, resolved_config)
    second = _run_once(resolved_seed, resolved_config)
    return _public_result(
        resolved_seed,
        resolved_config,
        first,
        repeatable=first == second,
    )
```

Tests must prove two full public calls are equal and that `repeatable` comes from two reconstructed executions rather than self-comparison.

- [ ] **Step 3: Encode the immutable per-seed and pooled gates**

Implement `_passes_acceptance()` with literal count checks:

```python
return (
    result.repeatable
    and result.post_training.total == 200
    and result.post_training.correct >= 180
    and all(score == AccuracyCount(score.correct, 40) and score.correct >= 34
            for _, score in result.per_delay)
    and result.state_reset == AccuracyCount(100, 200)
    and all(score == AccuracyCount(20, 40) for _, score in result.reset_per_delay)
    and result.all_reset_hidden_equal
    and result.shuffled_control.total == 200
    and result.shuffled_control.correct < 150
    and result.action_sequences_equal
    and result.reward_block_multisets_equal
    and result.normal_parameter_digest != result.initial_parameter_digest
    and result.shuffled_parameter_digest != result.initial_parameter_digest
    and result.normal_matrix_digests_before == result.normal_matrix_digests_after
    and result.shuffled_matrix_digests_before == result.shuffled_matrix_digests_after
    and not result.normal_pending_feedback
    and not result.shuffled_pending_feedback
)
```

Top-level `all_passed` is `all(per_seed_passed) and 0.40 <= pooled_accuracy <= 0.60`. Test that no strong seed compensates for a failed seed and boundaries 0.40/0.60 are inclusive.

- [ ] **Step 4: Serialize a stable schema-versioned payload**

`run_action_value_benchmark(seeds=(7,17,29), config=None)` returns a dict with sorted serialization compatibility and these top-level keys:

```python
{
    "all_passed": bool,
    "config": dict,
    "evidence_schema_version": 1,
    "frozen_evidence_sha256": {
        "phase_2b": str,
        "phase_2c": str,
    },
    "phase": "3A",
    "results": list,
    "rng_lineages": {
        "behavior_action": ["seed", 0x33414354],
        "evaluation_fixture": ["seed", 0x4556414C],
        "reward_shuffle": ["seed", 0x33534846],
        "training_fixture": ["seed", 0x54524149],
    },
    "seeds": list,
    "shuffled_pooled": {"accuracy": float, "correct": int, "total": int},
}
```

Each result includes all literal counts, per-delay rows, checkpoint summaries, value-margin summaries, TD-error summaries, both complete 2,000-action sequences, digests, integrity booleans, `repeatable`, and `passed`. Serialize with `allow_nan=False`, sorted keys, and no rounded replacement for raw values. Compute both frozen-evidence hashes from the committed Phase 2 files on every run so runtime and evidence projections can compare them.

- [ ] **Step 5: Add package exports and the compact CLI**

Export `ActionValueBenchmarkConfig`, `ActionValueCheckpoint`, `ActionValueExperimentResult`, `run_action_value_experiment`, and `run_action_value_benchmark`. Create:

```python
from __future__ import annotations

import json

from neural_state_machine import run_action_value_benchmark


def main() -> int:
    payload = run_action_value_benchmark()
    print(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return int(not payload["all_passed"])


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Run the pre-registered scientific gate exactly once**

Run:

```bash
.venv/bin/pytest tests/test_action_value.py tests/test_action_value_benchmark.py -q
.venv/bin/python scripts/benchmark_action_value.py
```

If any fixed gate misses, stop implementation, preserve the exact stdout payload, add only the failure-evidence path described in Task 6, commit it truthfully, and return to design review. Do not execute Tasks 7–9 as though the behavioral hypothesis passed.

- [ ] **Step 7: Add subprocess determinism tests after observing the fixed result**

Run the CLI twice in clean subprocesses. Require byte-identical stdout, one JSON line, ordered seeds `[7,17,29]`, phase `3A`, schema version `1`, and exit code equal to `int(not payload["all_passed"])`. The test must accept truthful nonzero status if the fixed experiment failed; it must never rewrite `all_passed`.

- [ ] **Step 8: Commit the public benchmark without evidence**

Run:

```bash
.venv/bin/pytest tests/test_action_value.py tests/test_action_value_benchmark.py -q
.venv/bin/ruff check src/neural_state_machine/action_value.py src/neural_state_machine/action_value_benchmark.py src/neural_state_machine/__init__.py tests/test_action_value.py tests/test_action_value_benchmark.py scripts/benchmark_action_value.py
git diff --check
git add src/neural_state_machine/action_value_benchmark.py src/neural_state_machine/__init__.py tests/test_action_value_benchmark.py scripts/benchmark_action_value.py
git commit -m "feat: add phase three-a acceptance benchmark"
git push origin experiment/neural-state-machine
```

---

### Task 6: Freeze and Verify Phase 3A Evidence

**Files:**
- Modify: `scripts/benchmark_action_value.py`
- Create: `scripts/verify_action_value_evidence.py`
- Create: `docs/experiments/phase-3a-action-value.json`
- Modify: `tests/test_action_value_benchmark.py`

**Interfaces:**
- Consumes: stable benchmark payload, the implementation commit being measured, and frozen Phase 2B/2C evidence bytes.
- Produces: one immutable evidence file, `_portable_phase_3a_payload()`, and fail-closed verification.

- [ ] **Step 1: Add evidence provenance fields before measurement**

Keep `run_action_value_benchmark()` independent of Git. It already emits the two frozen-evidence hashes from file bytes. Only when `--evidence` is supplied, enrich the artifact copy with:

```python
{
    "source_commit": _source_commit(),
}
```

`_source_commit()` runs `git rev-parse HEAD` without a shell and rejects nonzero status or a non-40-hex result. Tests monkeypatch subprocess output and cover malformed/missing Git state without altering scientific fields. The live compact benchmark output has no `source_commit`; the committed evidence records the exact Task 5 implementation commit that produced it.

- [ ] **Step 2: Add an atomic approved-path evidence writer**

Mirror the Phase 2C writer's safety properties: `--evidence` may resolve only to `docs/experiments/phase-3a-action-value.json`; reject symlinks and either frozen Phase 2 evidence path; render `json.dumps(..., sort_keys=True, indent=2, allow_nan=False) + "\n"`; write a same-directory temporary file, `fsync`, and `os.replace`; clean a failed temporary file; do not rewrite identical bytes.

- [ ] **Step 3: Define the portable projection**

Exclude only environment-local raw float/matrix/parameter digests whose bytes may vary with NumPy/LAPACK. Keep phase, schema, fixed config, ordered seeds, all counts, per-delay gates, action/reward schedule digests, fixture semantic digests, reward-block fairness, pending state, repeatability, per-seed `passed`, pooled control, frozen evidence hashes, and `all_passed`. Validate the evidence-only `source_commit` as exactly 40 lowercase hexadecimal characters and require `git merge-base --is-ancestor <source_commit> HEAD`; do not compare it to the current runtime HEAD or inject it into the live benchmark payload.

The projection must validate types, required keys, totals, digest format, exact seed order, exact config, checkpoint episode sequence, and finite numeric summaries before returning data. Missing or malformed fields raise `RuntimeError`; no permissive `.get()` defaults.

- [ ] **Step 4: Implement same-environment and committed-evidence verification**

Create:

```python
def verify_action_value_evidence() -> dict[str, object]:
    expected = _load_committed_evidence()
    _verify_source_commit_ancestor(expected["source_commit"])
    first = run_action_value_benchmark()
    second = run_action_value_benchmark()
    if first != second:
        raise RuntimeError("Phase 3A is not byte-stable in this environment")
    _verify_local_float_integrity(first)
    if _portable_phase_3a_payload(first) != _portable_phase_3a_payload(expected):
        raise RuntimeError("Phase 3A portable runtime differs from evidence")
    return _portable_phase_3a_payload(first)
```

Local integrity requires unchanged normal/shuffled matrix digests, changed normal and shuffled value digests from exact-zero initialization, equal action schedules, equal per-block reward multisets, and no pending feedback. The script prints compact sorted JSON and exits zero only after verification.

- [ ] **Step 5: Add fail-closed evidence-integrity tests**

Parametrize missing keys, wrong phase/schema/config/seeds, duplicate seed results, malformed counts/accuracies/digests/checkpoints, changed frozen evidence hashes, flipped fairness booleans, modified pass flags, and corrupted JSON. Require every mutation to fail verification. Also test same-environment first/second inequality and approved-writer path/symlink restrictions.

- [ ] **Step 6: Verify and commit the evidence tooling before measurement**

Run:

```bash
.venv/bin/pytest tests/test_action_value_benchmark.py -q
.venv/bin/ruff check scripts/benchmark_action_value.py scripts/verify_action_value_evidence.py tests/test_action_value_benchmark.py
git diff --check
git add scripts/benchmark_action_value.py scripts/verify_action_value_evidence.py tests/test_action_value_benchmark.py
git commit -m "test: add phase three-a evidence verification"
git push origin experiment/neural-state-machine
```

- [ ] **Step 7: Generate and verify the measured artifact without tuning**

With a clean worktree at the exact pushed Step 6 commit, run:

```bash
.venv/bin/python scripts/benchmark_action_value.py --evidence docs/experiments/phase-3a-action-value.json
.venv/bin/python scripts/verify_action_value_evidence.py
sha256sum docs/experiments/phase-3a-action-value.json docs/experiments/phase-2b-failure.json docs/experiments/phase-2c-diagnostics.json
```

The evidence-only `source_commit` must equal the Step 6 commit and be an ancestor of the later evidence commit. If `all_passed` is false, retain that truthful value and stop after Step 8 for design review.

- [ ] **Step 8: Commit only the immutable measured artifact**

Run:

```bash
git add docs/experiments/phase-3a-action-value.json
git commit -m "experiment: freeze phase three-a action-value evidence"
git push origin experiment/neural-state-machine
```

Use commit subject `experiment: record phase three-a failure` when the fixed gate is false. Never modify thresholds or previously frozen evidence to change that result.

---

### Task 7: Formalize the Update Kernel in Pinned Lean

**Files:**
- Create: `formal/lakefile.toml`
- Create: `formal/lean-toolchain`
- Create: `formal/lake-manifest.json`
- Create: `formal/NeuralStateMachine.lean`
- Create: `formal/NeuralStateMachine/ActionValue.lean`
- Create: `formal/NeuralStateMachine/AxiomAudit.lean`
- Create: `tests/test_formal_contract.py`

**Interfaces:**
- Consumes: the displayed normalized-update equation from the approved spec.
- Produces: seven named Lean theorems, a committed dependency lock, axiom output, and source-policy tests.

- [ ] **Step 1: Write failing formal-project contract tests**

In Python, require exact `formal/lean-toolchain` contents `leanprover/lean4:v4.30.0\n`, Mathlib revision `c5ea00351c28e24afc9f0f84379aa41082b1188f`, presence of all seven theorem names, a committed `lake-manifest.json`, and absence under `formal/**/*.lean` of regex tokens `\bsorry\b`, `\badmit\b`, declaration-level `\baxiom\b`, `native_decide`, and `run_tac`.

- [ ] **Step 2: Create the pinned Lake project and resolve once**

Use:

```toml
name = "neuralStateMachine"
version = "0.1.0"
defaultTargets = ["NeuralStateMachine"]

[[lean_lib]]
name = "NeuralStateMachine"

[[require]]
name = "mathlib"
git = "https://github.com/leanprover-community/mathlib4.git"
rev = "c5ea00351c28e24afc9f0f84379aa41082b1188f"
```

Then run from `formal/`:

```bash
lake update
lake build
```

Commit the generated `formal/lake-manifest.json`; do not regenerate dependencies in CI.

- [ ] **Step 3: Define the abstract table, feature, prediction, and selected update**

Set `formal/NeuralStateMachine.lean` to:

```lean
import NeuralStateMachine.ActionValue
import NeuralStateMachine.AxiomAudit
```

In `formal/NeuralStateMachine/ActionValue.lean`, use finite types and Euclidean dot products:

```lean
import Mathlib.Analysis.InnerProductSpace.PiL2
import Mathlib.Data.Fintype.BigOperators

namespace NeuralStateMachine.ActionValue

open scoped BigOperators

variable {Action Feature : Type} [Fintype Action] [DecidableEq Action]
variable [Fintype Feature] [DecidableEq Feature]

abbrev Vector (Feature : Type) := Feature → ℝ
abbrev Table (Action Feature : Type) := Action → Feature → ℝ

def prediction (w : Table Action Feature) (a : Action) (x : Vector Feature) : ℝ :=
  ∑ i, w a i * x i

def normSq (x : Vector Feature) : ℝ := ∑ i, x i * x i

def update
    (w : Table Action Feature) (selected : Action) (x : Vector Feature)
    (reward alpha : ℝ) : Table Action Feature :=
  fun action i =>
    if action = selected then
      w action i + alpha * (reward - prediction w selected x) * x i / normSq x
    else
      w action i
```

Use an augmented index `Option Feature` and define the bias coordinate as `none`, with `augmentedFeature h none = 1` and `augmentedFeature h (some i) = h i`.

- [ ] **Step 4: Prove positive augmented norm and selected prediction identity**

Prove:

```lean
theorem augmentedFeature_normSq_pos (h : Feature → ℝ) :
    0 < normSq (augmentedFeature h) := by
  have hone : (0 : ℝ) < (augmentedFeature h none) ^ 2 := by simp [augmentedFeature]
  have hnonneg : ∀ i, 0 ≤ (augmentedFeature h i) ^ 2 := fun i => sq_nonneg _
  simpa [normSq, mul_self] using
    Fintype.sum_pos_iff_of_nonempty.mpr ⟨none, hone, hnonneg⟩
```

If the named Mathlib lemma differs in Lean 4.30, replace only the proof tactic/lemma, not the theorem statement. Prove `selectedPrediction_after_update` by expanding `prediction`/`update`, pulling scalar factors through the finite sum, and cancelling `normSq x` using the explicit hypothesis `0 < normSq x`.

- [ ] **Step 5: Prove error contraction and row/lifecycle identities**

Add the exact theorem names and assumptions:

```lean
theorem selectedError_after_update
    (hpos : 0 < normSq x) :
    reward - prediction (update w selected x reward alpha) selected x =
      (1 - alpha) * (reward - prediction w selected x) := by
  rw [selectedPrediction_after_update hpos]
  ring

theorem selectedError_abs_le
    (hpos : 0 < normSq x) (halpha0 : 0 < alpha) (halpha1 : alpha ≤ 1) :
    |reward - prediction (update w selected x reward alpha) selected x| ≤
      |reward - prediction w selected x| := by
  rw [selectedError_after_update hpos, abs_mul]
  have : |1 - alpha| ≤ 1 := by
    rw [abs_of_nonneg (sub_nonneg.mpr halpha1)]
    linarith
  exact mul_le_of_le_one_left (abs_nonneg _) this
```

Prove `selectedError_abs_lt` with `alpha < 1` and nonzero prior error, `unselectedAction_unchanged` by simplifying the false action equality branch, and `zeroError_update_identity` by function extensionality plus simplification of the zero TD term.

- [ ] **Step 6: Add the axiom audit module**

Import `NeuralStateMachine.ActionValue` and include:

```lean
#print axioms NeuralStateMachine.ActionValue.augmentedFeature_normSq_pos
#print axioms NeuralStateMachine.ActionValue.selectedPrediction_after_update
#print axioms NeuralStateMachine.ActionValue.selectedError_after_update
#print axioms NeuralStateMachine.ActionValue.selectedError_abs_le
#print axioms NeuralStateMachine.ActionValue.selectedError_abs_lt
#print axioms NeuralStateMachine.ActionValue.unselectedAction_unchanged
#print axioms NeuralStateMachine.ActionValue.zeroError_update_identity
```

Record ordinary Mathlib foundations exactly as printed; do not hide them or add new declaration-level axioms.

- [ ] **Step 7: Build, scan, and commit the formal layer**

Run:

```bash
(cd formal && lake build)
.venv/bin/pytest tests/test_formal_contract.py -q
if rg -n '\b(sorry|admit|axiom|native_decide|run_tac)\b' formal --glob '*.lean'; then exit 1; fi
git diff --check
git add formal tests/test_formal_contract.py
git commit -m "proof: verify normalized action-value update"
git push origin experiment/neural-state-machine
```

The scan's only passing outcome is no matches.

---

### Task 8: Add Explicit Python-to-Theorem Correspondence Tests

**Files:**
- Modify: `tests/test_action_value.py`
- Create: `docs/experiments/phase-3a-theorem-boundary.md`

**Interfaces:**
- Consumes: immutable Python before/after snapshots and the public Lean theorem statements.
- Produces: an auditable equation map without claiming cross-language verification.

- [ ] **Step 1: Add table-driven Python equation cases**

Parametrize hidden vectors, selected actions, step sizes `0.1` and `1.0`, and rewards `-3.5`, `-1.0`, `0.0`, `1.0`, and `8.25`. For every case, independently reconstruct `x`, `s`, `q`, `delta`, and expected full table; require finite values, selected prediction interpolation, error-factor identity, unselected-row byte identity, and non-increasing same-sample absolute error for `0 < alpha <= 1`.

- [ ] **Step 2: Add strict-contraction and zero-error cases**

For `0 < alpha < 1` and nonzero TD error, require `abs(error_after) < abs(error_before)` within a guard that first proves the values differ by more than the chosen `1e-12` comparison tolerance. Separately require exact byte identity when reward equals the stored decision-time prediction.

- [ ] **Step 3: Document the correspondence boundary**

The document must state:

```text
Lean proves the displayed update equation over exact real-valued finite functions.
Python tests reconstruct the NumPy update independently within explicit float tolerances.
The Lean build does not execute, extract, or verify the Python implementation.
The algebraic theorem does not establish stochastic convergence or Phase 3A acceptance.
```

Map each Lean theorem name to the corresponding Python test name in a table.

- [ ] **Step 4: Run and commit the boundary checks**

Run:

```bash
.venv/bin/pytest tests/test_action_value.py tests/test_formal_contract.py -q
(cd formal && lake build)
.venv/bin/ruff check tests/test_action_value.py tests/test_formal_contract.py
git diff --check
git add tests/test_action_value.py docs/experiments/phase-3a-theorem-boundary.md
git commit -m "test: bind td equation to executable checks"
git push origin experiment/neural-state-machine
```

---

### Task 9: Report Results, Wire CI, and Audit the Orphan Branch

**Files:**
- Modify: `README.md`
- Modify: `.github/workflows/ci.yml`
- Test: complete repository and committed evidence

**Interfaces:**
- Consumes: committed Phase 3A evidence, verifier, formal build, and all frozen prior-phase commands.
- Produces: truthful documentation, Python-version CI coverage, independent formal CI, and final branch-integrity evidence.

- [ ] **Step 1: Add the measured README section**

Report the exact committed Phase 3A per-seed post/reset/shuffled counts, every per-delay count, pooled shuffled count, step size, episode counts, repeatability, and evidence SHA-256. State side by side that Phase 2B remains failed for seeds 7 and 29 and Phase 3A tests a different immediate scalar-reward estimator.

Include the update:

```text
x = [hidden; 1]
q = W[action] dot x
delta = reward - q
W[action] = W[action] + 0.1 * delta * x / (x dot x)
```

State explicitly that cue-to-decision delay is not action-to-reward delay, recurrent weights did not learn, the task is binary and small, Phase 3B is not implemented, and Lean proves only the exact abstract algebra.

- [ ] **Step 2: Extend the existing Python CI matrix**

Keep Python 3.10, 3.11, and 3.12 jobs and every existing step. Append:

```yaml
      - name: Phase 3A action-value benchmark
        run: python scripts/benchmark_action_value.py
      - name: Verify frozen Phase 3A evidence
        run: python scripts/verify_action_value_evidence.py
```

- [ ] **Step 3: Add an independent formal job**

Add:

```yaml
  formal:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: leanprover/lean-action@v1
        with:
          lake-package-directory: formal
      - name: Build pinned proofs
        working-directory: formal
        run: lake build
      - name: Reject proof shortcuts
        run: |
          if rg -n '\b(sorry|admit|axiom|native_decide|run_tac)\b' formal --glob '*.lean'; then
            exit 1
          fi
```

The committed `lake-manifest.json` must remain unchanged after this job.

- [ ] **Step 4: Audit the learner boundary and terminology**

Run:

```bash
rg -n 'correct_action_index|[.]cue|delay_steps|episode_phase|current_phase|probe|reward.delay|delayed.reward' src/neural_state_machine/action_value.py
rg -n 'reward.delay|delayed.reward' src/neural_state_machine/action_value_benchmark.py scripts/benchmark_action_value.py scripts/verify_action_value_evidence.py README.md
```

Expected: no matches in the learner; inspect every benchmark/README terminology match and remove any claim that cue-to-decision delay is delayed reward.

- [ ] **Step 5: Prove every frozen artifact remained unchanged**

Resolve the committed spec SHA, then compare its tree to the current tree for all frozen paths:

```bash
spec_commit=$(git log -1 --format=%H -- docs/superpowers/specs/2026-09-14-phase-3a-action-value-credit-design.md)
git diff --exit-code "$spec_commit"..HEAD -- \
  src/neural_state_machine/policy.py \
  src/neural_state_machine/reward_readout.py \
  src/neural_state_machine/reward_learning.py \
  src/neural_state_machine/learning_diagnostics.py \
  src/neural_state_machine/memory_task.py \
  src/neural_state_machine/memory_probe.py \
  src/neural_state_machine/memory_benchmark.py \
  docs/experiments/phase-2b-failure.json \
  docs/experiments/phase-2c-diagnostics.json
```

Also inspect `git diff --name-only "$spec_commit"..HEAD -- tests` and confirm no existing Phase 1/2 expectation changed; only new Phase 3A tests may appear.

- [ ] **Step 6: Run complete local verification**

Run in this order:

```bash
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/pytest -q
.venv/bin/ruff check .
.venv/bin/python scripts/benchmark.py
.venv/bin/python scripts/benchmark_memory_probe.py
.venv/bin/python scripts/verify_reward_learning_failure.py
.venv/bin/python scripts/benchmark_learning_diagnostics.py
.venv/bin/python scripts/benchmark_action_value.py
.venv/bin/python scripts/verify_action_value_evidence.py
(cd formal && lake build)
if rg -n '\b(sorry|admit|axiom|native_decide|run_tac)\b' formal --glob '*.lean'; then exit 1; fi
git diff --check
git status --short --branch
```

Record the pytest count, Ruff output, every benchmark exit status, Phase 3A JSON digest, and the formal axiom output.

- [ ] **Step 7: Commit documentation and CI, then push**

Run:

```bash
git add README.md .github/workflows/ci.yml
git commit -m "docs: report phase three-a action-value evidence"
git push origin experiment/neural-state-machine
```

- [ ] **Step 8: Verify the exact remote branch**

Require the remote head SHA equals local `HEAD`, the remote tree equals the local tree, and GitHub compare between `master` and `experiment/neural-state-machine` reports no common ancestor. Do not create a merge or pull request.

- [ ] **Step 9: Require exact-head GitHub Actions success**

Open the workflow run for the exact pushed SHA. Require all Python 3.10/3.11/3.12 jobs and the independent formal job to pass. If CI differs from local evidence, invoke `superpowers:systematic-debugging`, reproduce the discrepancy, and commit a focused portability fix without changing any scientific gate or frozen artifact.

## Completion Handoff

The final report must include:

- exact final remote SHA, orphan branch URL, and exact-head Actions URL;
- every Phase 3A implementation commit and subject;
- complete pytest count, Ruff result, and Lean build/axiom-audit result;
- unchanged Phase 1 result, Phase 2A counts, Phase 2B failure evidence hash, and Phase 2C evidence hash/classifications;
- exact Phase 3A pre/post/reset/shuffled counts for each seed and all five delay strata;
- checkpoint summaries, correct-action margin summaries, and TD-error summaries;
- individual and pooled shuffled-reward counts;
- normal/shuffled action-sequence equality and per-block reward-multiset equality;
- value-parameter and recurrent-matrix integrity evidence;
- exact independent-run repeatability and committed evidence SHA-256;
- the narrow conclusion: immediate scalar reward did or did not train this normalized action-value readout;
- an explicit statement that true action-to-reward delay remains Phase 3B and is not yet implemented.

Do not proceed into Phase 3B, FlyVis/fly-brain integration, YOLO, or game-engine work without a separate reviewed design.
