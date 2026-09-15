# Phase 3B Overlapping Delayed Credit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> This plan supersedes `docs/superpowers/plans/2026-09-15-phase-3b-delayed-credit.md`. The superseded plan is retained only as audit history and must not be executed.

**Goal:** Replace the invalid serialized Phase 3B harness with a deterministic overlapping action-to-reward protocol in which later decisions occur while earlier rewards remain pending, while preserving exact `d_r=0` Phase 3A TD(0) behavior and separating protocol validity from behavioral acceptance.

**Architecture:** Keep the frozen Phase 3A learner and evidence unchanged. Extend the environment-owned reward queue with auditable decision/delivery timing, add a Phase 3B-only multi-inflight TD(0) adapter that consumes scalar rewards FIFO without receiving delay metadata, then rewrite the benchmark around a global decision-step clock with select-before-deliver ordering and an explicit terminal drain. Gate P proves timeline validity before any behavioral result can be interpreted. Only after Gate P passes may Arm A be measured and an immutable Phase 3B artifact be written. TD(λ) is excluded until a separate overlapping-trace design is approved.

**Tech Stack:** Python 3.10–3.12, NumPy, pytest, Ruff, existing deterministic fixture/policy helpers, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-15-phase-3b-delayed-credit-design.md`

## Global Constraints

- Work only on `experiment/neural-state-machine`; do not modify or merge `master` as part of this plan.
- Keep these files byte-for-byte frozen: `src/neural_state_machine/policy.py`, `src/neural_state_machine/reward_readout.py`, `src/neural_state_machine/reward_learning.py`, `src/neural_state_machine/learning_diagnostics.py`, `src/neural_state_machine/memory_task.py`, `src/neural_state_machine/memory_probe.py`, `src/neural_state_machine/memory_benchmark.py`, and `src/neural_state_machine/action_value.py`.
- Keep `docs/experiments/phase-2b-failure.json`, `docs/experiments/phase-2c-diagnostics.json`, and `docs/experiments/phase-3a-action-value.json` byte-for-byte unchanged.
- The learner receives only a finite hidden vector, legal numeric action indices, a seeded NumPy generator, and one finite scalar reward when a delivery occurs.
- Never pass cue identity, correct action, either delay axis, task phase, episode objects, timestamps, due steps, queue identifiers, or reward-schedule labels into the learner API.
- Use ordered seeds `[7, 17, 29]`, training decisions `2000`, evaluation episodes `200`, cue-to-decision delays `[1, 2, 3, 4, 5]`, action-to-reward delays `[0, 1, 3, 5]`, hidden size `64`, recurrent radius `0.9`, step size `0.1`, and checkpoint interval `100`.
- `d_r` is the number of later real action selections that occur before an interior reward is delivered.
- At each real decision step: select action, enqueue its reward, deliver rewards due at that same decision step, then advance the clock.
- Terminal drain may advance time and deliver rewards but must never create synthetic actions.
- Fixed-delay delivery order is FIFO decision order.
- `d_r=0` must exactly reproduce frozen Phase 3A normal TD(0): action sequence, reward sequence, final parameter digest, post-training counts, and reset counts.
- Equality of action/reward digests across reward delays is allowed. Structural timeline evidence proves delay.
- Gate P failure means only `harness invalid`; behavior from that run is not Phase 3B evidence.
- TD(λ), eligibility traces, actor-critic, BPTT, STDP, recurrent-weight training, replay, reward shaping, adaptive delay step sizes, variable per-action delay, and hyperparameter search are outside this plan.
- Do not create `docs/experiments/phase-3b-delayed-credit.json` until Gate P passes on the exact implementation intended for measurement.
- After Gate P passes, a behavior failure is valid evidence and must be frozen without tuning.

## File Map

| Path | Responsibility |
|---|---|
| `src/neural_state_machine/delayed_credit.py` | Fixed-delay FIFO queue plus immutable decision/due/delivery metadata. |
| `src/neural_state_machine/phase3b_learners.py` | Multi-inflight TD(0) adapter; historical trace code is not selectable by corrected Phase 3B. |
| `src/neural_state_machine/phase3b_controls.py` | Timeline digest and exact Gate P invariants. |
| `src/neural_state_machine/phase3b_delayed_benchmark.py` | Select-before-deliver training, terminal drain, controls, checkpoints, and results. |
| `tests/test_delayed_credit.py` | Queue timing, metadata, atomicity, exactly-once behavior. |
| `tests/test_phase3b_learners.py` | FIFO credit and exact TD(0) update behavior. |
| `tests/test_phase3b_controls.py` | Exact overlap/high-water/lag/terminal-drain validation. |
| `tests/test_phase3b_delayed_benchmark.py` | End-to-end timeline, continuity, controls, repeatability. |
| `scripts/benchmark_phase3b_delayed_credit.py` | Arm A registered benchmark, Gate P CLI, schema-v2 payload, approved-path writer. |
| `scripts/verify_phase3b_delayed_credit.py` | Fail-closed committed-artifact verifier. |
| `tests/test_phase3b_evidence.py` | Schema-v2/writer/verifier mutation tests. |
| `.github/workflows/ci.yml` | Reusable unit, Ruff, Gate P, and later frozen-artifact verification. |
| `docs/experiments/phase-3b-delayed-credit.json` | Immutable corrected Arm A measurement, only after Gate P. |
| `docs/experiments/phase-3b-delayed-credit-report.md` | Bounded human-readable interpretation. |

---

### Task 1: Make reward deliveries self-auditing on the decision clock

**Files:**
- Modify: `src/neural_state_machine/delayed_credit.py`
- Modify: `tests/test_delayed_credit.py`

**Interfaces:**
- Consumes: `enqueue(action_index, reward, delay)`, `advance()`, `deliver_ready()`.
- Produces: `PendingReward(sequence, action_index, reward, decision_step, due_step)` and `RewardDelivery(sequence, action_index, reward, decision_step, due_step, delivery_step)`.

- [ ] **Step 1: Write the failing metadata test.**

```python
def test_delivery_records_exact_decision_due_and_delivery_steps() -> None:
    queue = DelayedRewardQueue(max_delay=5)
    pending = queue.enqueue(action_index=1, reward=1.0, delay=3)
    assert pending.sequence == 0
    assert pending.decision_step == 0
    assert pending.due_step == 3
    assert queue.advance() == 1
    assert queue.deliver_ready() == ()
    assert queue.advance() == 2
    assert queue.deliver_ready() == ()
    assert queue.advance() == 3
    assert queue.deliver_ready() == (
        RewardDelivery(
            sequence=0,
            action_index=1,
            reward=1.0,
            decision_step=0,
            due_step=3,
            delivery_step=3,
        ),
    )
```

Add a second test that enqueues one reward at step 0 and one at step 1 and asserts sequence numbers `(0, 1)` plus the correct `decision_step` on each delivery.

- [ ] **Step 2: Run RED.**

```bash
pytest -q tests/test_delayed_credit.py
```

Expected: failures because the current dataclasses lack timeline metadata.

- [ ] **Step 3: Implement immutable queue metadata.**

```python
@dataclass(frozen=True)
class PendingReward:
    sequence: int
    action_index: int
    reward: float
    decision_step: int
    due_step: int


@dataclass(frozen=True)
class RewardDelivery:
    sequence: int
    action_index: int
    reward: float
    decision_step: int
    due_step: int
    delivery_step: int
```

Initialize `_next_sequence = 0`. `enqueue()` captures `decision_step=current_step`, sets `due_step=current_step + delay`, and increments `_next_sequence` only after all validation passes. `deliver_ready()` stamps `delivery_step=current_step`. `reset()` restores step, sequence, and pending queue to zero/empty.

- [ ] **Step 4: Verify existing fail-closed behavior.**

```bash
pytest -q tests/test_delayed_credit.py
ruff check src/neural_state_machine/delayed_credit.py tests/test_delayed_credit.py
git diff --check
```

- [ ] **Step 5: Commit.**

```bash
git add src/neural_state_machine/delayed_credit.py tests/test_delayed_credit.py
git commit -m "feat: audit delayed reward timeline"
git push origin experiment/neural-state-machine
```

### Task 2: Implement Phase 3B-only multi-inflight TD(0)

**Files:**
- Modify: `src/neural_state_machine/phase3b_learners.py`
- Modify: `tests/test_phase3b_learners.py`
- Frozen: `src/neural_state_machine/action_value.py`

**Interfaces:**
- Produces: `DelayedTD0Adapter.select_for_training`, `select_greedy`, `learn`, `unresolved_credit_count`, `has_pending_feedback`, `parameter_snapshot`, and `parameter_digest`.
- `learn(reward)` consumes the oldest unresolved internal credit and returns `ActionValueUpdate` so the environment can audit the consumed action after the fact.

- [ ] **Step 1: Write the failing FIFO test.**

```python
def test_td0_adapter_allows_multiple_unresolved_decisions_and_learns_fifo() -> None:
    learner = DelayedTD0Adapter(hidden_size=2, action_count=2, step_size=0.1)
    rng = np.random.default_rng(113)
    first = learner.select_for_training(np.array([1.0, 0.0]), (0,), rng)
    second = learner.select_for_training(np.array([0.0, 1.0]), (1,), rng)
    assert (first.action_index, second.action_index) == (0, 1)
    assert learner.unresolved_credit_count == 2
    first_update = learner.learn(1.0)
    second_update = learner.learn(-1.0)
    assert (first_update.action_index, second_update.action_index) == (0, 1)
    assert learner.unresolved_credit_count == 0
    assert learner.has_pending_feedback is False
```

Add a test that captures `parameter_snapshot()` and `unresolved_credit_count`, calls `learn(float("nan"))`, expects `ValueError`, and proves both are unchanged.

- [ ] **Step 2: Run RED.**

```bash
pytest -q tests/test_phase3b_learners.py
```

Expected: the current wrapper rejects the second selection or lacks `unresolved_credit_count`.

- [ ] **Step 3: Implement a FIFO credit deque using frozen Phase 3A math.**

Use `collections.deque[_PendingCredit]`. Selection remains byte-for-byte equivalent in arithmetic/order to Phase 3A:

```python
feature = self._feature(hidden_state)
legal = self._legal_actions(legal_action_indices)
if not isinstance(rng, np.random.Generator):
    raise ValueError("rng must be a numpy.random.Generator")
action_index = legal[int(rng.integers(len(legal)))]
raw_values = self._weights @ feature
denominator = float(np.dot(feature, feature))
self._credits.append(
    _PendingCredit(
        action_index=action_index,
        feature=readonly_float64_copy(feature),
        denominator=denominator,
        prediction=float(raw_values[action_index]),
    )
)
```

Learning validates and computes before mutation:

```python
if not self._credits:
    raise RuntimeError("learning requires pending feedback")
reward_value = validated_finite_scalar(reward, "reward")
pending = self._credits[0]
td_error = reward_value - pending.prediction
with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
    delta = self.step_size * td_error * pending.feature / pending.denominator
    candidate = self._weights[pending.action_index] + delta
if not np.all(np.isfinite(candidate)):
    raise ValueError("finite reward would overflow action-value weights")
self._weights[pending.action_index] = candidate
self._credits.popleft()
return ActionValueUpdate(
    action_index=pending.action_index,
    prediction_before=pending.prediction,
    reward=reward_value,
    td_error=td_error,
)
```

Override `has_pending_feedback` as `bool(self._credits)` and expose `unresolved_credit_count` as `len(self._credits)`.

- [ ] **Step 4: Keep TD(λ) historical-only.**

Do not delete `EpisodeResetEligibilityTrace`, because Phase 3A diagnostic tests may import it. Add a docstring/comment that corrected Phase 3B does not select it, and do not add multi-inflight semantics to it.

- [ ] **Step 5: Verify and commit.**

```bash
pytest -q tests/test_phase3b_learners.py tests/test_phase3a_credit_compare.py tests/test_phase3a_credit_extreme.py
ruff check src/neural_state_machine/phase3b_learners.py tests/test_phase3b_learners.py
git diff -- src/neural_state_machine/action_value.py
git diff --check
git add src/neural_state_machine/phase3b_learners.py tests/test_phase3b_learners.py
git commit -m "feat: add multi-inflight delayed td0 credit"
git push origin experiment/neural-state-machine
```

The `action_value.py` diff must be empty.

### Task 3: Define exact Gate P timeline invariants

**Files:**
- Modify: `src/neural_state_machine/phase3b_controls.py`
- Modify: `tests/test_phase3b_controls.py`

**Interfaces:**
- Produces `TimelineAudit` and `validate_fixed_delay_timeline(audit, reward_delay)`.

- [ ] **Step 1: Add the audit type and failing validation tests.**

The target type is:

```python
@dataclass(frozen=True)
class TimelineAudit:
    action_count: int
    delivery_count: int
    terminal_drain_count: int
    max_pending_before_delivery: int
    max_pending_after_delivery: int
    decisions_with_prior_feedback_pending: int
    lag_histogram: tuple[tuple[int, int], ...]
    delivery_timeline_digest: str
    queue_pending_final: int
    learner_unresolved_final: int
```

Use this valid `N=8, d_r=3` fixture:

```python
audit = TimelineAudit(
    action_count=8,
    delivery_count=8,
    terminal_drain_count=3,
    max_pending_before_delivery=4,
    max_pending_after_delivery=3,
    decisions_with_prior_feedback_pending=7,
    lag_histogram=((3, 8),),
    delivery_timeline_digest="0" * 64,
    queue_pending_final=0,
    learner_unresolved_final=0,
)
validate_fixed_delay_timeline(audit, reward_delay=3)
```

Create copies with terminal drain `0`, high-water `1`, lag histogram `((0, 8),)`, delivery count `7`, queue final `1`, and learner final `1`; each must raise `ValueError`.

- [ ] **Step 2: Run RED.**

```bash
pytest -q tests/test_phase3b_controls.py
```

- [ ] **Step 3: Implement structural timeline hashing.**

Hash no reward values:

```python
digest = hashlib.sha256()
for delivery in deliveries:
    digest.update(
        f"{delivery.sequence}:{delivery.decision_step}:"
        f"{delivery.due_step}:{delivery.delivery_step}\n".encode("ascii")
    )
return digest.hexdigest()
```

- [ ] **Step 4: Implement exact validation with explicit branches.**

```python
if audit.delivery_count != audit.action_count:
    raise ValueError("delivery count must equal action count")
if audit.queue_pending_final != 0:
    raise ValueError("queue must be empty after terminal drain")
if audit.learner_unresolved_final != 0:
    raise ValueError("learner must have zero unresolved credits")

n = audit.action_count
if reward_delay == 0:
    if audit.terminal_drain_count != 0:
        raise ValueError("zero delay must not require terminal drain")
    if audit.max_pending_after_delivery != 0:
        raise ValueError("zero delay must leave no pending reward after delivery")
    if audit.lag_histogram != ((0, n),):
        raise ValueError("zero delay must have zero delivery lag")
else:
    if not 0 < reward_delay < n:
        raise ValueError("reward delay must be smaller than action count")
    if audit.terminal_drain_count != reward_delay:
        raise ValueError("terminal drain count must equal fixed reward delay")
    if audit.max_pending_before_delivery != reward_delay + 1:
        raise ValueError("pending high-water before delivery is invalid")
    if audit.max_pending_after_delivery != reward_delay:
        raise ValueError("pending high-water after delivery is invalid")
    if audit.decisions_with_prior_feedback_pending <= 0:
        raise ValueError("nonzero delay must overlap later decisions")
    if audit.lag_histogram != ((reward_delay, n),):
        raise ValueError("delivery lag histogram is invalid")
```

- [ ] **Step 5: Verify and commit.**

```bash
pytest -q tests/test_phase3b_controls.py
ruff check src/neural_state_machine/phase3b_controls.py tests/test_phase3b_controls.py
git diff --check
git add src/neural_state_machine/phase3b_controls.py tests/test_phase3b_controls.py
git commit -m "test: define phase three-b protocol gate"
git push origin experiment/neural-state-machine
```

### Task 4: Rewrite training as select-before-deliver overlap

**Files:**
- Modify: `src/neural_state_machine/phase3b_delayed_benchmark.py`
- Modify: `tests/test_phase3b_delayed_benchmark.py`

**Interfaces:**
- Keep `run_delayed_credit(seed, reward_delay, arm="td0", config=None)`.
- Corrected Phase 3B accepts only `arm="td0"`; `td_lambda` raises a blocked-arm `ValueError`.
- Each result exposes its `TimelineAudit`, actions, rewards, parameter digest, checkpoints, and repeatability.

- [ ] **Step 1: Add the RED overlap test.**

```python
def test_reward_delay_three_overlaps_real_decisions() -> None:
    config = DelayedCreditConfig(
        training_episodes=100,
        evaluation_blocks=2,
        checkpoint_interval=50,
    )
    result = run_delayed_credit(7, 3, "td0", config)
    assert result.timeline.terminal_drain_count == 3
    assert result.timeline.max_pending_before_delivery == 4
    assert result.timeline.max_pending_after_delivery == 3
    assert result.timeline.lag_histogram == ((3, 100),)
    assert result.timeline.decisions_with_prior_feedback_pending > 0
```

Add:

```python
with pytest.raises(ValueError, match="blocked"):
    run_delayed_credit(7, 1, "td_lambda", config)
```

- [ ] **Step 2: Run RED.**

```bash
pytest -q tests/test_phase3b_delayed_benchmark.py
```

- [ ] **Step 3: Implement one timeline primitive.**

Use this exact ordering for real decisions:

```python
for decision_step, episode in enumerate(fixtures):
    if queue.current_step != decision_step:
        raise RuntimeError("queue and decision clocks diverged")

    prior_pending = queue.pending_count > 0
    hidden = _decision_hidden(policy, episode, reset_before_decision=False)
    decision = learner.select_for_training(hidden, (0, 1), action_rng)
    action = int(decision.action_index)
    reward = float(reward_schedule[decision_step])
    queue.enqueue(action, reward, reward_delay)

    max_pending_before = max(max_pending_before, queue.pending_count)
    for delivery in queue.deliver_ready():
        update = learner.learn(delivery.reward)
        if update.action_index != delivery.action_index:
            raise RuntimeError("learner credit and queue action diverged")
        deliveries.append(delivery)
    max_pending_after = max(max_pending_after, queue.pending_count)
    decisions_with_prior_pending += int(prior_pending)

    if decision_step + 1 < len(fixtures):
        queue.advance()
```

Terminal drain contains no selections:

```python
terminal_drain_count = 0
while queue.pending_count:
    queue.advance()
    ready = queue.deliver_ready()
    for delivery in ready:
        update = learner.learn(delivery.reward)
        if update.action_index != delivery.action_index:
            raise RuntimeError("learner credit and queue action diverged")
        deliveries.append(delivery)
        terminal_drain_count += 1
```

- [ ] **Step 4: Define checkpoints at decision boundaries.**

After each `checkpoint_interval` real decisions and after due deliveries for that step, record exactly:

```text
decision_count
delivery_count
unresolved_credit_count
parameter_digest
```

Do not require unresolved count zero for nonzero delay. Do not synthesize TD errors for unresolved credits.

- [ ] **Step 5: Build and validate `TimelineAudit` before evaluating behavior.**

Compute the lag histogram from every `RewardDelivery`, compute the structural digest, capture final queue/learner counts, then call `validate_fixed_delay_timeline`. If it raises, propagate the failure and do not serialize accuracy as Phase 3B evidence.

- [ ] **Step 6: Strengthen exact `d_r=0` continuity.**

```python
legacy = run_action_value_experiment(7, config.action_value_config)
delayed = run_delayed_credit(7, 0, "td0", config)
assert delayed.actions == legacy.normal_actions
assert delayed.action_digest == legacy.normal_action_digest
assert delayed.training_reward_digest == legacy.normal_reward_digest
assert delayed.parameter_digest == legacy.normal_parameter_digest
assert delayed.post_training == legacy.post_training
assert delayed.per_delay == legacy.per_delay
assert delayed.state_reset == legacy.state_reset
assert delayed.reset_per_delay == legacy.reset_per_delay
```

- [ ] **Step 7: Verify and commit.**

```bash
pytest -q tests/test_delayed_credit.py tests/test_phase3b_learners.py tests/test_phase3b_controls.py tests/test_phase3b_delayed_benchmark.py
ruff check src/neural_state_machine/delayed_credit.py src/neural_state_machine/phase3b_learners.py src/neural_state_machine/phase3b_controls.py src/neural_state_machine/phase3b_delayed_benchmark.py tests/test_delayed_credit.py tests/test_phase3b_learners.py tests/test_phase3b_controls.py tests/test_phase3b_delayed_benchmark.py
git diff --check
git add src/neural_state_machine/phase3b_delayed_benchmark.py tests/test_phase3b_delayed_benchmark.py
git commit -m "fix: overlap phase three-b delayed decisions"
git push origin experiment/neural-state-machine
```

### Task 5: Rebuild shuffled and reset controls on the same timeline

**Files:**
- Modify: `src/neural_state_machine/phase3b_delayed_benchmark.py`
- Modify: `src/neural_state_machine/phase3b_controls.py`
- Modify: `tests/test_phase3b_delayed_benchmark.py`
- Modify: `tests/test_phase3b_controls.py`

**Interfaces:**
- Produces normal/shuffled action digests, reward-assignment digests, timeline digests, parameter digests, post/reset/shuffled counts, and reward-block-multiset status.

- [ ] **Step 1: Write timing-identity RED tests.**

```python
assert result.normal_action_digest == result.shuffled_action_digest
assert result.normal_timeline.delivery_timeline_digest == result.shuffled_timeline.delivery_timeline_digest
assert result.normal_timeline.lag_histogram == result.shuffled_timeline.lag_histogram
assert result.reward_block_multisets_equal is True
```

For a deterministic test seed/config where the permutation is known non-identity, also assert `normal_reward_assignment_digest != shuffled_reward_assignment_digest`.

- [ ] **Step 2: Precompute expected actions/rewards from independent lineages.**

```python
schedule_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x33414354]))
expected_actions = tuple((0, 1)[int(schedule_rng.integers(2))] for _ in fixtures.training)
normal_rewards = tuple(
    float(task.reward(episode, action))
    for episode, action in zip(fixtures.training, expected_actions, strict=True)
)
shuffle_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x33534846]))
shuffled_rewards = _permute_reward_blocks(normal_rewards, shuffle_rng, block_size=10)
```

Normal and shuffled training each reconstruct their own action RNG from `[seed, 0x33414354]`. At every decision, assert the actual selected action equals `expected_actions[decision_step]`. A mismatch is a hard lineage error.

- [ ] **Step 3: Run both reward schedules through the Task 4 primitive.**

Only scalar rewards differ. Action sequence, decision steps, due steps, queue occupancy, lag histogram, terminal drain, and timeline digest must match exactly.

- [ ] **Step 4: Implement reset evaluation with a fresh zero learner.**

Create a new `DelayedTD0Adapter` with the same shape/step size and no training, then call the existing `_evaluate(..., reset_before_decision=True)` on the same evaluation fixtures. Do not zero or mutate the trained learner.

- [ ] **Step 5: Compute behavior status only after both timelines are valid.**

Per `(seed, d_r)` require:

```text
post-training >= 180/200
each cue-to-decision delay >= 34/40
reset == 100/200
reset per cue delay == 20/40
shuffled < 150/200
```

Store `behavior_passed` per result. Protocol validity remains a separate field.

- [ ] **Step 6: Verify and commit.**

```bash
pytest -q tests/test_phase3b_controls.py tests/test_phase3b_delayed_benchmark.py
ruff check src/neural_state_machine/phase3b_controls.py src/neural_state_machine/phase3b_delayed_benchmark.py tests/test_phase3b_controls.py tests/test_phase3b_delayed_benchmark.py
git diff --check
git add src/neural_state_machine/phase3b_controls.py src/neural_state_machine/phase3b_delayed_benchmark.py tests/test_phase3b_controls.py tests/test_phase3b_delayed_benchmark.py
git commit -m "test: rebuild phase three-b delayed controls"
git push origin experiment/neural-state-machine
```

### Task 6: Replace diagnostic-only schema with Gate P / Arm A schema v2

**Files:**
- Modify: `scripts/benchmark_phase3b_delayed_credit.py`
- Modify: `scripts/verify_phase3b_delayed_credit.py`
- Modify: `tests/test_phase3b_evidence.py`
- Do not create yet: `docs/experiments/phase-3b-delayed-credit.json`

**Interfaces:**
- `build_payload` runs only corrected Arm A.
- Top-level schema version is `2` with booleans `protocol_valid`, `behavior_passed`, and `all_passed`.
- `all_passed` is exactly `protocol_valid and behavior_passed`.
- `write_evidence` refuses any payload whose `protocol_valid` is not exactly `True`.

- [ ] **Step 1: Write schema-v2 RED tests.**

```python
payload = build_payload(
    tmp_path,
    seeds=(7,),
    config=DelayedCreditConfig(
        training_episodes=100,
        evaluation_blocks=2,
        checkpoint_interval=50,
    ),
)
assert payload["experiment"] == "phase-3b-delayed-credit"
assert payload["schema_version"] == 2
assert type(payload["protocol_valid"]) is bool
assert type(payload["behavior_passed"]) is bool
assert payload["all_passed"] == (
    payload["protocol_valid"] and payload["behavior_passed"]
)
assert {row["arm"] for row in payload["results"]} == {"td0"}
```

Before calling `build_payload`, create `tmp_path / "docs/experiments/phase-3a-action-value.json"` with deterministic bytes so the frozen hash field is available.

- [ ] **Step 2: Write the writer-refusal test without a synthetic helper.**

```python
payload = build_payload(
    tmp_path,
    seeds=(7,),
    config=DelayedCreditConfig(
        training_episodes=100,
        evaluation_blocks=2,
        checkpoint_interval=50,
    ),
)
payload["protocol_valid"] = False
payload["behavior_passed"] = False
payload["all_passed"] = False
target = tmp_path / "docs/experiments/phase-3b-delayed-credit.json"
with pytest.raises(ValueError, match="protocol_valid"):
    write_evidence(payload, target, tmp_path)
assert not target.exists()
```

- [ ] **Step 3: Build registered Arm A aggregation.**

Run ordered seeds `(7, 17, 29)` and delays `(0, 1, 3, 5)`. Do not construct TD(λ) rows. `protocol_valid` is true only if every row passes structural timeline validation, normal/shuffled timing identity, `d_r=0` Phase 3A continuity, fixture/action lineages, repeatability, finite-value checks, zero terminal unresolved state, and frozen Phase 3A hash validation.

Set top-level `behavior_passed` to the conjunction of the registered non-control `d_r in (1, 3, 5)` per-result behavior flags after protocol validity is true. If protocol validity is false, force top-level behavior false. This keeps the `d_r=0` control as continuity evidence rather than a second delayed-credit acceptance opportunity.

- [ ] **Step 4: Add a protocol-gate CLI that cannot write evidence.**

```bash
python scripts/benchmark_phase3b_delayed_credit.py --require-protocol-valid
```

It prints deterministic JSON, exits nonzero when `protocol_valid` is false, and never writes the evidence file. Keep `--evidence docs/experiments/phase-3b-delayed-credit.json` as the only writer path and allow it only when protocol validity is true.

- [ ] **Step 5: Make the verifier fail closed.**

Reject schema version 1, any `td_lambda` row, missing timeline fields, malformed 64-hex digests, duplicate `(seed, d_r)` rows, changed frozen Phase 3A hash, `protocol_valid != True`, inconsistent `all_passed`, or any committed/runtime mismatch in the deterministic projection.

- [ ] **Step 6: Verify without creating evidence.**

```bash
pytest -q tests/test_phase3b_evidence.py tests/test_phase3b_delayed_benchmark.py tests/test_phase3b_controls.py
ruff check scripts/benchmark_phase3b_delayed_credit.py scripts/verify_phase3b_delayed_credit.py tests/test_phase3b_evidence.py
git diff -- docs/experiments/phase-3b-delayed-credit.json
git diff --check
```

The evidence diff must be empty.

- [ ] **Step 7: Commit.**

```bash
git add scripts/benchmark_phase3b_delayed_credit.py scripts/verify_phase3b_delayed_credit.py tests/test_phase3b_evidence.py
git commit -m "feat: gate phase three-b evidence on protocol validity"
git push origin experiment/neural-state-machine
```

### Task 7: Put Gate P on the reusable CI path

**Files:**
- Modify: `.github/workflows/ci.yml`
- No evidence artifact in this task.

- [ ] **Step 1: Require local Gate P before changing CI.**

```bash
pytest -q
ruff check .
git diff --check
python scripts/benchmark_phase3b_delayed_credit.py --require-protocol-valid
```

If the final command fails, stop with `harness invalid`. Do not weaken structural checks or write evidence.

- [ ] **Step 2: Add one reusable Gate P step.**

```yaml
      - name: Phase 3B overlapping protocol gate
        run: python scripts/benchmark_phase3b_delayed_credit.py --require-protocol-valid
```

Do not add `--evidence`.

- [ ] **Step 3: Commit and push.**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: enforce phase three-b protocol gate"
git push origin experiment/neural-state-machine
```

- [ ] **Step 4: Require exact-head remote success.**

Record the pushed SHA and require the GitHub Actions run for that exact SHA to pass Python 3.10/3.11/3.12 tests, Ruff, and the Phase 3B protocol gate. Any failure stops execution before Task 8.

### Task 8: Freeze the first valid Arm A measurement

**Files:**
- Create only after Task 7 exact-head success: `docs/experiments/phase-3b-delayed-credit.json`
- Create: `docs/experiments/phase-3b-delayed-credit-report.md`
- Modify: `README.md`
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- The artifact contains only corrected Arm A (`td0`) rows.
- `protocol_valid=true` is mandatory for writing; `behavior_passed` may be false.

- [ ] **Step 1: Generate one registered measurement from the exact Gate P implementation.**

```bash
python scripts/benchmark_phase3b_delayed_credit.py \
  --evidence docs/experiments/phase-3b-delayed-credit.json
```

If the writer refuses, stop at `harness invalid`; do not tune or retry with changed scientific parameters.

- [ ] **Step 2: Verify candidate bytes.**

```bash
python scripts/verify_phase3b_delayed_credit.py \
  --evidence docs/experiments/phase-3b-delayed-credit.json
python scripts/benchmark_phase3b_delayed_credit.py --require-protocol-valid
```

Both must agree on the deterministic protocol projection.

- [ ] **Step 3: Write the report from the frozen JSON.**

The report must include Gate P status; exact `d_r=0` continuity; per-seed/per-delay post, reset, and shuffled counts; exact lag histogram; queue high-water and terminal drain; Arm A threshold status; and the explicit statement that corrected Phase 3B contains no TD(λ) comparison. If performance changes with delay, report the observation without attributing a mechanism.

- [ ] **Step 4: Add permanent committed-artifact verification to CI.**

```yaml
      - name: Verify frozen Phase 3B delayed-credit evidence
        run: python scripts/verify_phase3b_delayed_credit.py
```

The verifier must accept protocol-valid behavioral failure evidence.

- [ ] **Step 5: Run final local verification.**

```bash
pytest -q
ruff check .
git diff --check
python scripts/verify_reward_learning_failure.py
python scripts/verify_phase3b_delayed_credit.py
git diff HEAD~1 -- \
  src/neural_state_machine/action_value.py \
  src/neural_state_machine/policy.py \
  src/neural_state_machine/reward_learning.py \
  docs/experiments/phase-2b-failure.json \
  docs/experiments/phase-2c-diagnostics.json \
  docs/experiments/phase-3a-action-value.json
```

The frozen-path diff must be empty.

- [ ] **Step 6: Commit and push immutable evidence.**

```bash
git add \
  docs/experiments/phase-3b-delayed-credit.json \
  docs/experiments/phase-3b-delayed-credit-report.md \
  README.md \
  .github/workflows/ci.yml
git commit -m "docs: freeze corrected phase three-b arm a evidence"
git push origin experiment/neural-state-machine
```

- [ ] **Step 7: Require final exact-head CI and record one bounded conclusion.**

Use exactly one classification:

```text
A. Gate P failed -> harness invalid; no delayed-credit conclusion.
B. Gate P passed and all registered nonzero delays passed Arm A thresholds -> immediate reward was not necessary under this fixed-delay overlapping protocol.
C. Gate P passed but at least one registered nonzero delay failed/degraded while d_r=0 preserved Phase 3A -> overlapping delayed feedback is associated with the measured limitation.
```

Do not infer TD(λ) benefit. A TD(λ) experiment requires a new committed design amendment first.

---

## Plan Self-Review Checklist

Before Task 1 execution, verify:

- All production changes are outside frozen Phase 3A files.
- Corrected measurement implements only multi-inflight TD(0).
- Every nonzero `d_r` creates real action overlap before delivery.
- Terminal drain contains no synthetic decisions.
- Timeline validity precedes behavior thresholds.
- `d_r=0` continuity includes final parameter digest.
- Normal and shuffled runs share action/timeline lineages; only reward assignment differs.
- Action/reward digest equality across delays is permitted.
- Evidence cannot be written before `protocol_valid == True`.
- Protocol-valid behavior failure can be frozen.
- No Task contains a TD(λ) acceptance path.
- The old serialized plan remains audit history only.

## Completion Handoff

Report the exact remote SHA; Task 1–8 commit subjects; local pytest/Ruff results; exact-head CI; Gate P structural metrics for every registered `d_r`; `d_r=0` continuity digests; normal/shuffled action and timeline digests; per-seed/per-delay post/reset/shuffled counts; frozen Phase 3A artifact hash; immutable Phase 3B artifact hash; and conclusion A/B/C. If any structural invariant fails, stop at `harness invalid` and do not report behavior as Phase 3B evidence.
