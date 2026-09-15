# Phase 3B Overlapping Delayed Credit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> This plan supersedes `docs/superpowers/plans/2026-09-15-phase-3b-delayed-credit.md`. The superseded plan is retained as audit history and must not be executed.

**Goal:** Replace the invalid serialized Phase 3B harness with a deterministic overlapping action-to-reward protocol in which later decisions occur while earlier rewards remain pending, while preserving exact `d_r=0` Phase 3A TD(0) behavior and separating protocol validity from behavioral acceptance.

**Architecture:** Keep the frozen Phase 3A learner and evidence unchanged. Extend the environment-owned reward queue so deliveries carry auditable decision/delivery timing, add a Phase 3B-only multi-inflight TD(0) adapter that consumes scalar rewards FIFO without receiving delay metadata, then rewrite the benchmark around a global decision-step clock with select-before-deliver ordering and an explicit terminal drain. Gate P proves timeline validity before any behavioral result can be interpreted; only after Gate P passes may Arm A be measured and an immutable Phase 3B artifact be written. TD(λ) is excluded from this implementation plan until a separate overlapping-trace design is approved.

**Tech Stack:** Python 3.10–3.12, NumPy, pytest, Ruff, existing deterministic fixture/policy helpers, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-15-phase-3b-delayed-credit-design.md`

## Global Constraints

- Work only on branch `experiment/neural-state-machine`; do not modify or merge `master` as part of this plan.
- Keep these files byte-for-byte frozen: `src/neural_state_machine/policy.py`, `src/neural_state_machine/reward_readout.py`, `src/neural_state_machine/reward_learning.py`, `src/neural_state_machine/learning_diagnostics.py`, `src/neural_state_machine/memory_task.py`, `src/neural_state_machine/memory_probe.py`, `src/neural_state_machine/memory_benchmark.py`, and `src/neural_state_machine/action_value.py`.
- Keep `docs/experiments/phase-2b-failure.json`, `docs/experiments/phase-2c-diagnostics.json`, and `docs/experiments/phase-3a-action-value.json` byte-for-byte unchanged.
- The learner may receive only a finite hidden vector, legal numeric action indices, a seeded NumPy generator, and one finite scalar reward when a delivery occurs.
- Never pass cue identity, correct action, cue-to-decision delay, action-to-reward delay, task phase, episode objects, timestamps, due steps, queue identifiers, or reward-schedule labels into the learner API.
- Use ordered seeds `[7, 17, 29]`, training decisions `2000`, evaluation episodes `200`, cue-to-decision delays `[1, 2, 3, 4, 5]`, action-to-reward delays `[0, 1, 3, 5]`, hidden size `64`, recurrent radius `0.9`, step size `0.1`, and checkpoint interval `100`.
- `d_r` means the number of later real action selections that occur before an interior reward is delivered.
- At every real decision step, select/enqueue before delivering rewards due at that step.
- Do not invent synthetic actions during terminal drain.
- For fixed `d_r`, delivery order must remain FIFO decision order.
- `d_r=0` must exactly reproduce the frozen Phase 3A normal TD(0) action sequence, reward sequence, final parameter digest, post-training counts, and reset counts for the same seed/configuration.
- Equality of action/reward digests across reward delays is allowed. Timeline evidence, not outcome divergence, proves that delay was exercised.
- Gate P failure means only `harness invalid`; no accuracy or parameter result from that run is scientific Phase 3B evidence.
- TD(λ), eligibility traces, actor-critic, BPTT, STDP, recurrent-weight training, replay buffers, reward shaping, adaptive delay-dependent step sizes, variable per-action delay, and hyperparameter search are outside this plan.
- Do not create or freeze `docs/experiments/phase-3b-delayed-credit.json` until Gate P passes on the exact implementation intended for measurement.
- After Gate P passes, a behavior failure is still valid evidence and must be frozen without tuning thresholds, seeds, fixtures, delays, counts, or parameters.

## File Map

| Path | Responsibility after this plan |
|---|---|
| `src/neural_state_machine/delayed_credit.py` | Environment-owned fixed-delay FIFO queue with immutable decision sequence, decision step, due step, and delivery step metadata. |
| `src/neural_state_machine/phase3b_learners.py` | Phase 3B-only multi-inflight TD(0) adapter; existing trace class may remain as historical diagnostic code but is not selectable by corrected Phase 3B. |
| `src/neural_state_machine/phase3b_controls.py` | Timeline digesting and exact structural Gate P validation, plus existing action/reward/queue controls. |
| `src/neural_state_machine/phase3b_delayed_benchmark.py` | Corrected global decision-step training loop, terminal drain, normal/shuffled controls, reset evaluation, checkpoints, and result payloads. |
| `tests/test_delayed_credit.py` | Queue timing/metadata/exactly-once tests. |
| `tests/test_phase3b_learners.py` | Multi-inflight FIFO credit, exact TD(0) update, numerical atomicity, and blocked-trace tests. |
| `tests/test_phase3b_controls.py` | Exact `d_r` overlap/high-water/terminal-drain validation and timeline digest tests. |
| `tests/test_phase3b_delayed_benchmark.py` | End-to-end corrected timeline, `d_r=0` Phase 3A continuity, shuffled timing identity, reset, repeatability, and Gate P tests. |
| `scripts/benchmark_phase3b_delayed_credit.py` | Arm A-only registered benchmark, Gate P CLI, deterministic schema-v2 payload, and approved-path atomic writer after protocol validity. |
| `scripts/verify_phase3b_delayed_credit.py` | Fail-closed schema, frozen-hash, protocol-validity, repeatability, and committed-artifact verifier. |
| `tests/test_phase3b_evidence.py` | Schema-v2, writer refusal before Gate P, artifact determinism, and verifier mutation tests. |
| `.github/workflows/ci.yml` | Reusable corrected Phase 3B unit/Ruff/Gate P checks; no one-off measurement job. |
| `docs/experiments/phase-3b-delayed-credit.json` | Immutable Arm A Phase 3B evidence created only after exact Gate P success. |
| `docs/experiments/phase-3b-delayed-credit-report.md` | Human-readable interpretation of the frozen Arm A measurement without claims about TD(λ). |

---

### Task 1: Make queue deliveries self-auditing on the decision clock

**Files:**
- Modify: `src/neural_state_machine/delayed_credit.py`
- Modify: `tests/test_delayed_credit.py`

**Interfaces:**
- Consumes: `enqueue(action_index: int, reward: float, delay: int)`, `advance()`, and `deliver_ready()`.
- Produces: immutable `PendingReward(sequence, action_index, reward, decision_step, due_step)` and `RewardDelivery(sequence, action_index, reward, decision_step, due_step, delivery_step)` records. Existing `current_step`, `pending_count`, and reset behavior remain public.

- [ ] **Step 1: Write RED tests for decision/delivery metadata.**

Add tests equivalent to:

```python
def test_delivery_records_exact_decision_due_and_delivery_steps() -> None:
    queue = DelayedRewardQueue(max_delay=5)
    pending = queue.enqueue(action_index=1, reward=1.0, delay=3)
    assert pending.sequence == 0
    assert pending.decision_step == 0
    assert pending.due_step == 3

    for expected in (1, 2, 3):
        assert queue.advance() == expected
        if expected < 3:
            assert queue.deliver_ready() == ()

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

Also add a FIFO test that enqueues one item at step 0, advances to step 1, enqueues a second item, and proves sequence numbers are `0, 1` and each delivery preserves its own decision step.

- [ ] **Step 2: Run the focused tests and confirm RED.**

Run:

```bash
pytest -q tests/test_delayed_credit.py
```

Expected: failures because the current dataclasses do not expose `sequence`, `decision_step`, or `delivery_step`.

- [ ] **Step 3: Extend queue records without changing timing semantics.**

Implement the record shapes and a monotonic sequence counter:

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

Initialize `self._next_sequence = 0`. In `enqueue()`, capture `decision_step=self._current_step`, assign the current sequence, increment only after validation succeeds, and preserve `due_step = current_step + delay`. In `deliver_ready()`, set `delivery_step=self._current_step`. In `reset()`, clear the queue and restore both step and sequence to zero.

- [ ] **Step 4: Preserve all existing fail-closed queue behavior.**

Run:

```bash
pytest -q tests/test_delayed_credit.py
ruff check src/neural_state_machine/delayed_credit.py tests/test_delayed_credit.py
git diff --check
```

Expected: all queue tests pass, including invalid reward/delay atomicity and stale-delivery protection.

- [ ] **Step 5: Commit Task 1.**

```bash
git add src/neural_state_machine/delayed_credit.py tests/test_delayed_credit.py
git commit -m "feat: audit delayed reward timeline"
git push origin experiment/neural-state-machine
```

### Task 2: Replace single-pending Arm A with a Phase 3B-only multi-inflight TD(0) adapter

**Files:**
- Modify: `src/neural_state_machine/phase3b_learners.py`
- Modify: `tests/test_phase3b_learners.py`
- Do not modify: `src/neural_state_machine/action_value.py`

**Interfaces:**
- Consumes: the same hidden vectors, legal actions, RNG, scalar reward, and numerical validation helpers used by Phase 3A.
- Produces: `DelayedTD0Adapter.select_for_training(...)`, `select_greedy(...)`, `learn(reward)`, `unresolved_credit_count`, `has_pending_feedback`, `parameter_snapshot()`, and `parameter_digest()`.
- `learn(reward)` consumes the oldest unresolved decision and returns `ActionValueUpdate` whose `action_index` identifies the consumed internal credit for environment-side audit.

- [ ] **Step 1: Replace the old one-pending tests with RED multi-inflight tests.**

Add:

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

Add another test that calls `learn()` with NaN while two credits are pending and proves weights plus both unresolved credits remain unchanged.

- [ ] **Step 2: Confirm RED against the current thin wrapper.**

Run:

```bash
pytest -q tests/test_phase3b_learners.py
```

Expected: the second selection raises `feedback is already pending` or `unresolved_credit_count` is missing.

- [ ] **Step 3: Implement a dedicated FIFO credit deque while reusing frozen Phase 3A math.**

Keep `NormalizedActionValue` frozen. In `phase3b_learners.py`, make `DelayedTD0Adapter` a Phase 3B-specific implementation/subclass that stores immutable `_PendingCredit` instances in `collections.deque`.

The selection logic must remain equivalent to Phase 3A:

```python
feature = self._feature(hidden_state)
legal = self._legal_actions(legal_action_indices)
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

The reward update must validate and compute before mutating the deque:

```python
pending = self._credits[0]
reward_value = validated_finite_scalar(reward, "reward")
td_error = reward_value - pending.prediction
delta = self.step_size * td_error * pending.feature / pending.denominator
candidate = self._weights[pending.action_index] + delta
if not np.all(np.isfinite(candidate)):
    raise ValueError("finite reward would overflow action-value weights")
self._weights[pending.action_index] = candidate
self._credits.popleft()
return ActionValueUpdate(...)
```

Override `has_pending_feedback` to mean `bool(self._credits)` and expose `unresolved_credit_count = len(self._credits)`. Keep greedy evaluation and parameter snapshots/digests mathematically identical to Phase 3A.

- [ ] **Step 4: Explicitly block the old trace learner from corrected Phase 3B selection.**

Do not delete `EpisodeResetEligibilityTrace`; it may remain for historical Phase 3A diagnostic tests. Add a comment/docstring that it is not an approved overlapping Phase 3B arm, and do not add any multi-inflight behavior to it in this task.

- [ ] **Step 5: Verify the adapter and frozen Phase 3A regressions.**

Run:

```bash
pytest -q tests/test_phase3b_learners.py tests/test_phase3a_credit_compare.py tests/test_phase3a_credit_extreme.py
ruff check src/neural_state_machine/phase3b_learners.py tests/test_phase3b_learners.py
git diff -- src/neural_state_machine/action_value.py
```

Expected: tests pass and the final diff command is empty.

- [ ] **Step 6: Commit Task 2.**

```bash
git add src/neural_state_machine/phase3b_learners.py tests/test_phase3b_learners.py
git commit -m "feat: add multi-inflight delayed td0 credit"
git push origin experiment/neural-state-machine
```

### Task 3: Add exact timeline statistics and Gate P structural validation

**Files:**
- Modify: `src/neural_state_machine/phase3b_controls.py`
- Modify: `tests/test_phase3b_controls.py`

**Interfaces:**
- Produces frozen `TimelineAudit` with `action_count`, `delivery_count`, `terminal_drain_count`, `max_pending_before_delivery`, `max_pending_after_delivery`, `decisions_with_prior_feedback_pending`, `lag_histogram`, `delivery_timeline_digest`, `queue_pending_final`, and `learner_unresolved_final`.
- Produces `delivery_timeline_digest(deliveries)` and `validate_fixed_delay_timeline(audit, reward_delay)`.

- [ ] **Step 1: Write RED tests for the exact registered invariants.**

Create compact fixtures for `N=8` and assert:

```python
def test_fixed_delay_three_requires_real_overlap() -> None:
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

Add mutations that set terminal drain to `0`, high-water to `1`, lag to `0`, delivery count to `7`, or either final unresolved count to `1`; every mutation must raise `ValueError`.

For `d_r=0`, require terminal drain `0`, max pending after delivery `0`, lag histogram `((0, N),)`, and zero final unresolved counts.

- [ ] **Step 2: Confirm RED.**

Run:

```bash
pytest -q tests/test_phase3b_controls.py
```

Expected: missing `TimelineAudit` and validator symbols.

- [ ] **Step 3: Implement deterministic timeline hashing.**

Hash only structural fields, never reward values:

```python
for delivery in deliveries:
    digest.update(
        f"{delivery.sequence}:{delivery.decision_step}:"
        f"{delivery.due_step}:{delivery.delivery_step}\n".encode("ascii")
    )
```

This allows normal and shuffled runs to have identical timeline digests even when reward assignments differ.

- [ ] **Step 4: Implement exact structural validation.**

For `N = audit.action_count`:

```python
if reward_delay == 0:
    require(audit.terminal_drain_count == 0)
    require(audit.max_pending_after_delivery == 0)
    require(audit.lag_histogram == ((0, N),))
else:
    require(0 < reward_delay < N)
    require(audit.terminal_drain_count == reward_delay)
    require(audit.max_pending_before_delivery == reward_delay + 1)
    require(audit.max_pending_after_delivery == reward_delay)
    require(audit.decisions_with_prior_feedback_pending > 0)
    require(audit.lag_histogram == ((reward_delay, N),))
require(audit.delivery_count == N)
require(audit.queue_pending_final == 0)
require(audit.learner_unresolved_final == 0)
```

Use explicit `ValueError` messages naming the violated invariant.

- [ ] **Step 5: Run focused controls and commit.**

```bash
pytest -q tests/test_phase3b_controls.py
ruff check src/neural_state_machine/phase3b_controls.py tests/test_phase3b_controls.py
git diff --check
git add src/neural_state_machine/phase3b_controls.py tests/test_phase3b_controls.py
git commit -m "test: define phase three-b protocol gate"
git push origin experiment/neural-state-machine
```

### Task 4: Rewrite the benchmark around select-before-deliver overlap

**Files:**
- Modify: `src/neural_state_machine/phase3b_delayed_benchmark.py`
- Modify: `tests/test_phase3b_delayed_benchmark.py`

**Interfaces:**
- `run_delayed_credit(seed, reward_delay, arm="td0", config=None)` remains the public entry point.
- Corrected Phase 3B accepts only `arm="td0"`; requesting `td_lambda` raises a clear blocked-arm error.
- Training produces both a `TimelineAudit` and action/reward/parameter traces.

- [ ] **Step 1: Add the RED test that reproduces the original harness defect.**

For a small `training_episodes=100` config and `reward_delay=3`, require:

```python
result = run_delayed_credit(7, 3, "td0", config)
assert result.timeline.terminal_drain_count == 3
assert result.timeline.max_pending_before_delivery == 4
assert result.timeline.max_pending_after_delivery == 3
assert result.timeline.lag_histogram == ((3, 100),)
assert result.timeline.decisions_with_prior_feedback_pending > 0
```

The current serialized harness must fail these assertions before implementation changes.

Also add:

```python
with pytest.raises(ValueError, match="TD.*lambda.*blocked"):
    run_delayed_credit(7, 1, "td_lambda", config)
```

- [ ] **Step 2: Run the RED test and capture the failure reason.**

```bash
pytest -q tests/test_phase3b_delayed_benchmark.py
```

Expected: the current harness either lacks timeline fields or reports no overlapping pending credits.

- [ ] **Step 3: Implement one corrected training primitive.**

Add a private helper with this ordering:

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

    max_before = max(max_before, queue.pending_count)
    ready = queue.deliver_ready()
    for delivery in ready:
        update = learner.learn(delivery.reward)
        if update.action_index != delivery.action_index:
            raise RuntimeError("learner credit and queue action diverged")
        deliveries.append(delivery)
    max_after = max(max_after, queue.pending_count)
    decisions_with_prior_pending += int(prior_pending)

    if decision_step + 1 < len(fixtures):
        queue.advance()
```

After the last real decision, drain without selecting synthetic actions:

```python
terminal_drain_count = 0
while queue.pending_count:
    queue.advance()
    for delivery in queue.deliver_ready():
        update = learner.learn(delivery.reward)
        if update.action_index != delivery.action_index:
            raise RuntimeError("learner credit and queue action diverged")
        deliveries.append(delivery)
        terminal_drain_count += 1
```

Never call `reset_episode()` inside this loop.

- [ ] **Step 4: Define checkpoint semantics on decision boundaries.**

At each `checkpoint_interval` real decisions, after delivering rewards due at that decision step, record an immutable checkpoint containing:

```text
decision_count
delivery_count
unresolved_credit_count
parameter_digest
```

Do not require unresolved credit to be zero at a checkpoint. Do not fabricate TD errors for rewards that have not arrived. `d_r=0` checkpoints naturally have `unresolved_credit_count == 0`; nonzero delays have the registered pipeline occupancy.

- [ ] **Step 5: Build `TimelineAudit` only after terminal drain and validate it immediately.**

Construct the lag histogram from every `RewardDelivery.delivery_step - decision_step`, calculate the structural digest, record final queue/learner counts, and call `validate_fixed_delay_timeline(...)`. If validation fails, raise before post-training accuracy is interpreted or serialized as evidence.

- [ ] **Step 6: Prove exact `d_r=0` continuity against Phase 3A.**

Keep and strengthen the existing continuity test:

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

- [ ] **Step 7: Run corrected benchmark tests and commit.**

```bash
pytest -q tests/test_delayed_credit.py tests/test_phase3b_learners.py tests/test_phase3b_controls.py tests/test_phase3b_delayed_benchmark.py
ruff check src/neural_state_machine/delayed_credit.py src/neural_state_machine/phase3b_learners.py src/neural_state_machine/phase3b_controls.py src/neural_state_machine/phase3b_delayed_benchmark.py tests/test_delayed_credit.py tests/test_phase3b_learners.py tests/test_phase3b_controls.py tests/test_phase3b_delayed_benchmark.py
git diff --check
git add src/neural_state_machine/phase3b_delayed_benchmark.py tests/test_phase3b_delayed_benchmark.py
git commit -m "fix: overlap phase three-b delayed decisions"
git push origin experiment/neural-state-machine
```

### Task 5: Rebuild shuffled/reset controls on the corrected timeline

**Files:**
- Modify: `src/neural_state_machine/phase3b_delayed_benchmark.py`
- Modify: `src/neural_state_machine/phase3b_controls.py`
- Modify: `tests/test_phase3b_delayed_benchmark.py`
- Modify: `tests/test_phase3b_controls.py`

**Interfaces:**
- Produces normal and shuffled action digests, reward-assignment digests, delivery-timeline digests, parameter digests, post-training/reset/shuffled evaluation counts, and block-multiset control status.

- [ ] **Step 1: Write RED tests for timing-identical shuffled controls.**

Require, for the same seed/config/delay:

```python
assert result.normal_action_digest == result.shuffled_action_digest
assert result.normal_timeline.delivery_timeline_digest == result.shuffled_timeline.delivery_timeline_digest
assert result.reward_block_multisets_equal is True
assert result.normal_timeline.lag_histogram == result.shuffled_timeline.lag_histogram
```

Also require the shuffled reward-assignment digest to differ when the deterministic permutation is non-identity.

- [ ] **Step 2: Build the reward schedule without coupling it to learner weights.**

Because training action choice is seeded random and independent of learned values, precompute the expected action sequence with an independently reconstructed RNG from `[seed, 0x33414354]`:

```python
schedule_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x33414354]))
expected_actions = tuple((0, 1)[int(schedule_rng.integers(2))] for _ in fixtures.training)
normal_rewards = tuple(
    float(task.reward(episode, action))
    for episode, action in zip(fixtures.training, expected_actions, strict=True)
)
```

Build shuffled rewards with an independent `[seed, 0x33534846]` generator and the existing fixed block-size permutation rule. The actual normal and shuffled learners each receive a separately reconstructed `[seed, 0x33414354]` RNG, and every actual selected action must equal the corresponding precomputed action. Any mismatch is a hard lineage failure.

- [ ] **Step 3: Run normal and shuffled schedules through the same timeline primitive.**

Only the scalar reward sequence may differ. Decision steps, due steps, queue occupancy, action sequence, and delivery timing must be identical. Compute a reward-assignment digest from delivered scalar values in decision-credit order so the permutation is auditable separately from timeline structure.

- [ ] **Step 4: Implement reset evaluation with exact-zero value parameters.**

Instantiate a fresh `DelayedTD0Adapter` with the same hidden size/action count/step size and no training. Evaluate the same frozen evaluation fixtures with the same `_evaluate(..., reset_before_decision=True)` path used by the Phase 3A boundary. Do not mutate the trained learner to perform reset evaluation.

- [ ] **Step 5: Add behavior gates but keep them downstream of Gate P.**

For each valid `(seed, d_r)` result compute:

```text
normal overall >= 180/200
every cue-to-decision delay >= 34/40
reset == 100/200
reset per delay == 20/40
shuffled < 150/200
```

Store the boolean as `behavior_passed`. It must never be consulted until both normal and shuffled timelines have passed structural validation.

- [ ] **Step 6: Run focused tests and commit.**

```bash
pytest -q tests/test_phase3b_controls.py tests/test_phase3b_delayed_benchmark.py
ruff check src/neural_state_machine/phase3b_controls.py src/neural_state_machine/phase3b_delayed_benchmark.py tests/test_phase3b_controls.py tests/test_phase3b_delayed_benchmark.py
git diff --check
git add src/neural_state_machine/phase3b_controls.py src/neural_state_machine/phase3b_delayed_benchmark.py tests/test_phase3b_controls.py tests/test_phase3b_delayed_benchmark.py
git commit -m "test: rebuild phase three-b delayed controls"
git push origin experiment/neural-state-machine
```

### Task 6: Replace the diagnostic schema with a fail-closed Gate P / Arm A evidence schema

**Files:**
- Modify: `scripts/benchmark_phase3b_delayed_credit.py`
- Modify: `scripts/verify_phase3b_delayed_credit.py`
- Modify: `tests/test_phase3b_evidence.py`
- Do not create yet: `docs/experiments/phase-3b-delayed-credit.json`

**Interfaces:**
- `build_payload(...)` runs only corrected Arm A.
- Schema version becomes `2`.
- Top-level fields include `protocol_valid`, `behavior_passed`, and `all_passed = protocol_valid and behavior_passed`.
- `write_evidence(...)` refuses to write when `protocol_valid` is not exactly `True`.

- [ ] **Step 1: Write RED schema-v2 tests.**

Require a compact test payload to contain:

```python
assert payload["experiment"] == "phase-3b-delayed-credit"
assert payload["schema_version"] == 2
assert type(payload["protocol_valid"]) is bool
assert type(payload["behavior_passed"]) is bool
assert payload["all_passed"] == (
    payload["protocol_valid"] and payload["behavior_passed"]
)
assert {row["arm"] for row in payload["results"]} == {"td0"}
```

Each result must include both normal/shuffled timeline audit fields, delivery-lag histogram, terminal drain count, action/reward/timeline/parameter digests, post/reset/shuffled counts, checkpoint summaries, repeatability, and queue/learner final counts.

- [ ] **Step 2: Add writer refusal before protocol validity.**

Add:

```python
def test_writer_refuses_protocol_invalid_payload(tmp_path: Path) -> None:
    payload = valid_schema_v2_payload(protocol_valid=False, behavior_passed=False)
    target = tmp_path / "docs/experiments/phase-3b-delayed-credit.json"
    with pytest.raises(ValueError, match="protocol_valid"):
        write_evidence(payload, target, tmp_path)
    assert not target.exists()
```

- [ ] **Step 3: Implement Arm A-only payload construction and Gate P aggregation.**

Run `(seed, d_r)` for ordered seeds `(7, 17, 29)` and delays `(0, 1, 3, 5)`. Do not construct TD(λ) results. Set `protocol_valid` to true only when every result passes normal/shuffled structural validation, `d_r=0` continuity, fixture lineage, action-lineage, repeatability, finite-value, and frozen Phase 3A hash checks.

Set `behavior_passed` from the pre-registered thresholds only after `protocol_valid` is true; otherwise force it to false. Derive `all_passed` mechanically from both fields.

- [ ] **Step 4: Add an explicit protocol-gate CLI that does not write evidence.**

Support:

```bash
python scripts/benchmark_phase3b_delayed_credit.py --require-protocol-valid
```

This command prints the deterministic payload and exits nonzero if `protocol_valid` is false. It must not write `docs/experiments/phase-3b-delayed-credit.json`.

Keep `--evidence docs/experiments/phase-3b-delayed-credit.json` as the only approved writer path, and permit it only when protocol validity is true.

- [ ] **Step 5: Make the verifier recompute and compare the valid protocol.**

`verify_phase3b_delayed_credit()` must reject schema version 1, any TD(λ) row, missing timeline fields, malformed digests, a changed frozen Phase 3A hash, `protocol_valid != True`, inconsistent `all_passed`, duplicate `(seed, d_r)` rows, or any committed/runtime mismatch in the deterministic evidence projection.

- [ ] **Step 6: Run evidence tests without writing the artifact.**

```bash
pytest -q tests/test_phase3b_evidence.py tests/test_phase3b_delayed_benchmark.py tests/test_phase3b_controls.py
ruff check scripts/benchmark_phase3b_delayed_credit.py scripts/verify_phase3b_delayed_credit.py tests/test_phase3b_evidence.py
git diff -- docs/experiments/phase-3b-delayed-credit.json
```

Expected: tests pass and no evidence file diff exists.

- [ ] **Step 7: Commit Task 6.**

```bash
git add scripts/benchmark_phase3b_delayed_credit.py scripts/verify_phase3b_delayed_credit.py tests/test_phase3b_evidence.py
git commit -m "feat: gate phase three-b evidence on protocol validity"
git push origin experiment/neural-state-machine
```

### Task 7: Put Gate P on the reusable CI path and stop on protocol failure

**Files:**
- Modify: `.github/workflows/ci.yml`
- No evidence artifact is created in this task.

**Interfaces:**
- Existing Python 3.10/3.11/3.12 matrix remains.
- Corrected Phase 3B unit tests and Ruff remain reusable jobs/steps.
- Add one reusable Gate P execution using the exact checked-out head; do not add a one-off artifact-emitting diagnostic job.

- [ ] **Step 1: Run the full local regression before changing CI.**

```bash
pytest -q
ruff check .
git diff --check
python scripts/benchmark_phase3b_delayed_credit.py --require-protocol-valid
```

If the final command fails, stop here. Record `harness invalid`; do not weaken the gate, write evidence, or modify thresholds.

- [ ] **Step 2: Add the reusable Gate P step.**

Append to the existing `standalone` job after corrected Phase 3B unit/Ruff checks:

```yaml
      - name: Phase 3B overlapping protocol gate
        run: python scripts/benchmark_phase3b_delayed_credit.py --require-protocol-valid
```

Do not add `--evidence` to CI at this stage.

- [ ] **Step 3: Commit and push CI wiring.**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: enforce phase three-b protocol gate"
git push origin experiment/neural-state-machine
```

- [ ] **Step 4: Require exact-head remote success before measurement.**

Record the pushed SHA. Inspect the GitHub Actions run for that exact SHA and require all three Python matrix jobs, full tests, Ruff, and the Phase 3B protocol gate to succeed. If any fails, stop and debug the failing gate; do not proceed to Task 8.

### Task 8: Freeze the first valid Arm A measurement and report only what it proves

**Files:**
- Create only after Task 7 exact-head Gate P success: `docs/experiments/phase-3b-delayed-credit.json`
- Create: `docs/experiments/phase-3b-delayed-credit-report.md`
- Modify: `README.md` only to link the new report/artifact and summarize the bounded conclusion.
- Modify: `.github/workflows/ci.yml` to verify the committed artifact after it exists.

**Interfaces:**
- The immutable JSON contains only corrected Arm A (`td0`) results for the registered seeds and delays.
- Behavioral failure is allowed as valid evidence when `protocol_valid=true`; `all_passed` may therefore be false.

- [ ] **Step 1: Generate the registered measurement once from the exact Gate P implementation.**

Run:

```bash
python scripts/benchmark_phase3b_delayed_credit.py \
  --evidence docs/experiments/phase-3b-delayed-credit.json
```

Do not alter code/configuration after seeing the measurement in order to improve counts. If the writer refuses because protocol validity is false, classify the run `harness invalid`, delete any partial temporary output, and stop.

- [ ] **Step 2: Verify the committed candidate bytes before git add.**

Run:

```bash
python scripts/verify_phase3b_delayed_credit.py \
  --evidence docs/experiments/phase-3b-delayed-credit.json
python scripts/benchmark_phase3b_delayed_credit.py --require-protocol-valid
```

Both commands must agree on protocol validity and the deterministic result projection.

- [ ] **Step 3: Write the report from the frozen JSON without re-running/tuning.**

The report must state:

```text
1. Gate P protocol validity result.
2. Exact d_r=0 continuity result.
3. For each seed and d_r, post/reset/shuffled counts.
4. Exact lag histogram, queue high-water, and terminal-drain invariants.
5. Whether Arm A behavior passed the pre-registered thresholds.
6. If performance changes with delay, report the observed change without claiming a mechanism.
7. No TD(lambda) comparison exists under the corrected protocol.
8. No claim about emergent state machines, general game competence, or general RL ability is supported.
```

Do not call equal action/reward digests a failure when timeline evidence is valid.

- [ ] **Step 4: Add permanent artifact verification to CI.**

Only now add:

```yaml
      - name: Verify frozen Phase 3B delayed-credit evidence
        run: python scripts/verify_phase3b_delayed_credit.py
```

The verifier must accept a valid-protocol artifact whether `behavior_passed` is true or false.

- [ ] **Step 5: Run the complete final verification.**

```bash
pytest -q
ruff check .
git diff --check
python scripts/verify_reward_learning_failure.py
python scripts/verify_phase3b_delayed_credit.py
```

Also verify frozen files have no diff:

```bash
git diff HEAD~1 -- \
  src/neural_state_machine/action_value.py \
  src/neural_state_machine/policy.py \
  src/neural_state_machine/reward_learning.py \
  docs/experiments/phase-2b-failure.json \
  docs/experiments/phase-2c-diagnostics.json \
  docs/experiments/phase-3a-action-value.json
```

Expected: empty frozen-path diff.

- [ ] **Step 6: Commit the immutable measurement and reporting surface.**

```bash
git add \
  docs/experiments/phase-3b-delayed-credit.json \
  docs/experiments/phase-3b-delayed-credit-report.md \
  README.md \
  .github/workflows/ci.yml
git commit -m "docs: freeze corrected phase three-b arm a evidence"
git push origin experiment/neural-state-machine
```

- [ ] **Step 7: Require final exact-head CI and record one bounded decision.**

For the exact pushed SHA, require all matrix jobs plus committed-artifact verification to pass. Then choose exactly one conclusion supported by the artifact:

```text
A. Gate P failed -> harness invalid. No delayed-credit conclusion.
B. Gate P passed; nonzero d_r Arm A passed -> immediate reward was not necessary under this fixed-delay overlapping protocol.
C. Gate P passed; Arm A degraded/failed while d_r=0 preserved Phase 3A -> overlapping delayed feedback is associated with the measured limitation.
```

Do not infer TD(λ) benefit. A follow-up TD(λ) design requires a new committed spec amendment before implementation.

---

## Plan Self-Review Checklist

Before starting Task 1, verify these plan properties against the corrected spec:

- Every production change is outside the frozen Phase 3A files.
- The only learner implemented for corrected measurement is multi-inflight TD(0).
- Every nonzero reward delay causes later real decisions before earlier reward delivery.
- Terminal drain contains rewards only, never synthetic actions.
- Timeline validity is checked before behavior thresholds.
- `d_r=0` continuity is exact, including the final parameter digest.
- Normal and shuffled runs share action/timeline lineages; only reward assignment changes.
- Action/reward digest equality across delays is permitted.
- Evidence writing is impossible before `protocol_valid == True`.
- A valid-protocol behavioral failure can still be frozen as evidence.
- No implementation task contains a TD(λ) acceptance path.
- The superseded serialized plan is retained only for audit history.

## Completion Handoff

At completion, report the exact remote SHA; Task 1–8 commit subjects; local pytest/Ruff results; exact-head CI run; Gate P structural metrics for every registered `d_r`; `d_r=0` continuity digests; normal/shuffled action and timeline digests; per-seed/per-delay post/reset/shuffled counts; frozen Phase 3A artifact hash; immutable Phase 3B artifact hash; and the single bounded conclusion A/B/C above. If any structural invariant fails, stop at `harness invalid` and do not report behavior as Phase 3B evidence.
