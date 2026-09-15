# Phase 3B Delayed Credit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic, fail-closed benchmark that measures true action-to-reward delay for TD(0) and episode-reset TD(λ), without changing frozen Phase 3A behavior or evidence.

**Architecture:** Build a standalone delayed-reward queue around the existing one-pending-credit learner boundary. First prove queue timing and immediate-reward equivalence, then add delayed TD(0), an episode-reset eligibility-trace arm, fixed controls, and an immutable evidence writer/verifier. Phase 3B is a new experiment surface; Phase 3A remains byte-for-byte frozen.

**Tech Stack:** Python 3.10–3.12, NumPy, pytest, Ruff, existing deterministic fixture/policy helpers, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-15-phase-3b-delayed-credit-design.md`

## Global Constraints

- Work only on orphan branch `experiment/neural-state-machine`; never merge, rebase onto, copy from, or modify `master`.
- Commit and push every completed task; never force-push.
- Keep NumPy as the only Python runtime dependency.
- Keep Phase 1/2 implementation files and evidence byte-for-byte unchanged.
- Keep `docs/experiments/phase-3a-action-value.json` byte-for-byte unchanged.
- The learner receives only finite hidden state, legal numeric actions, a seeded NumPy Generator, and a finite scalar reward when delivered.
- Do not pass cue identity, correct action, delay labels, task phase, probe output, episode objects, or reward timing metadata into the learner.
- Use ordered seeds `[7, 17, 29]`, training episodes `2000`, evaluation episodes `200`, cue-to-decision delays `[1, 2, 3, 4, 5]`, action-to-reward delays `[0, 1, 3, 5]`, hidden size `64`, recurrent radius `0.9`, step size `0.1`, checkpoint interval `100`, and TD(λ) parameters `discount=0.9`, `trace_decay=0.8`.
- Reuse the Phase 3A acceptance gate: normal at least `180/200`, each cue-to-decision delay at least `34/40`, reset exactly `100/200`, reset per delay exactly `20/40`, shuffled below `150/200`.
- Do not tune thresholds, seeds, counts, delays, fixtures, or parameters after observing results.
- Overflow and non-finite values are hard failures; failed updates must not mutate weights, traces, or pending credit.
- No actor-critic, BPTT, STDP, recurrent-weight training, replay, reward shaping, adaptive delay step sizes, or hyperparameter search.
- The Phase 3B artifact is separate from all Phase 3A artifacts.

## File Map

| Path | Responsibility |
|---|---|
| `src/neural_state_machine/delayed_credit.py` | FIFO action-to-reward queue with exactly-once delivery and strict delay validation. |
| `tests/test_delayed_credit.py` | Queue RED/green tests: timing, stale/duplicate/underflow/overflow, reset, and immediate equivalence. |
| `src/neural_state_machine/phase3b_delayed_benchmark.py` | Fixed fixtures, delayed training/evaluation, Arm A/Arm B adapters, controls, summaries, and deterministic digests. |
| `tests/test_phase3b_delayed_benchmark.py` | Lineage, fairness, queue integration, reset, shuffled, repeatability, and gate contract tests. |
| `scripts/benchmark_phase3b_delayed_credit.py` | Compact benchmark output and approved-path atomic evidence writer. |
| `scripts/verify_phase3b_delayed_credit.py` | Fail-closed replay, projection, artifact, and frozen-hash verification. |
| `docs/experiments/phase-3b-delayed-credit.json` | Immutable measured Phase 3B artifact; created only after implementation and verification. |
| `README.md` | Added only after measured evidence exists; reports Phase 3B boundaries and results. |
| `.github/workflows/ci.yml` | Added only after local implementation passes; preserves existing jobs and appends Phase 3B verification. |

---

### Task 1: Implement the delayed-reward queue

**Files:**
- Create: `src/neural_state_machine/delayed_credit.py`
- Create: `tests/test_delayed_credit.py`
- Modify: `src/neural_state_machine/__init__.py` only to export the queue's public type after tests pass.

**Interfaces:**
- Consumes: one selected action identifier, one finite terminal reward, a non-negative integer delay, and monotonically advancing integer steps.
- Produces: `DelayedRewardQueue`, `PendingReward`, `RewardDelivery`, `enqueue(action_index, reward, delay)`, `advance()`, `deliver_ready()`, `reset()`, and `pending_count`.

- [ ] **Step 1: Write RED tests for the public queue contract.**

```python
def test_reward_is_delivered_after_exact_registered_delay() -> None:
    queue = DelayedRewardQueue(max_delay=5)
    queue.enqueue(action_index=1, reward=1.0, delay=3)
    assert queue.deliver_ready() == ()
    assert queue.advance() == 1
    assert queue.deliver_ready() == ()
    assert queue.advance() == 2
    assert queue.deliver_ready() == ()
    assert queue.advance() == 3
    assert queue.deliver_ready() == (RewardDelivery(action_index=1, reward=1.0, due_step=3),)
```

Also add tests that `delay=0` is ready immediately, negative/non-integer/too-large delays are rejected, rewards must be finite non-booleans, duplicate delivery is impossible, stale delivery raises `RuntimeError`, and `reset()` removes all pending entries.

- [ ] **Step 2: Run the focused tests and verify RED.**

Run `pytest tests/test_delayed_credit.py -q`. Expected result: collection fails because `neural_state_machine.delayed_credit` does not exist.

- [ ] **Step 3: Implement the smallest queue with explicit step semantics.**

Use an immutable `RewardDelivery` dataclass and a private FIFO keyed by due step. `enqueue()` records `due_step = current_step + delay`; `deliver_ready()` returns all entries due at `current_step` in insertion order and removes them exactly once; `advance()` increments one step and rejects advancing after reset only through normal state; `reset()` returns to step zero with an empty queue. Never silently drop or reorder entries.

- [ ] **Step 4: Run focused tests and add numerical/lifecycle cases.**

Run `pytest tests/test_delayed_credit.py -q` and `ruff check src/neural_state_machine/delayed_credit.py tests/test_delayed_credit.py`. Add assertions that snapshots remain unchanged after invalid operations and that two queues with the same calls have byte-identical delivery digests.

- [ ] **Step 5: Commit the queue.**

```bash
git add src/neural_state_machine/delayed_credit.py tests/test_delayed_credit.py src/neural_state_machine/__init__.py
git commit -m "feat: add delayed reward queue"
git push origin experiment/neural-state-machine
```

### Task 2: Add the delayed benchmark protocol and immediate-control harness

**Files:**
- Create: `src/neural_state_machine/phase3b_delayed_benchmark.py`
- Create: `tests/test_phase3b_delayed_benchmark.py`

**Interfaces:**
- Consumes: `ActionValueBenchmarkConfig`, `_build_fixture_bundle`, frozen policy helpers, `NormalizedActionValue`, `DelayedRewardQueue`, and the registered seed/delay configuration.
- Produces: `DelayedCreditConfig`, `DelayedCreditResult`, `run_delayed_credit(seed, reward_delay, arm, config=None)`, `run_delayed_credit_benchmark()`, and a JSON-safe payload builder.

- [ ] **Step 1: Write RED tests for configuration and `d_r=0` continuity.**

Require exact defaults for seeds, both delay axes, hidden size, radius, step size, episode counts, checkpoint interval, and TD(λ) parameters. Require invalid arm names, negative delays, duplicate seeds, and unordered seed inputs to fail closed. For `reward_delay=0`, require the delayed harness to produce the same action digest, fixture digests, normal counts, reset counts, and parameter digest as the existing Phase 3A normal path for the same seed/config.

- [ ] **Step 2: Run the tests and verify RED.**

Run `pytest tests/test_phase3b_delayed_benchmark.py -q`. Expected result: import or symbol failures because the new benchmark module is absent.

- [ ] **Step 3: Implement immutable config validation and lineages.**

Define a frozen configuration object with exact tuple fields. Reuse training/evaluation fixture construction and policy helpers; do not duplicate recurrent dynamics. Derive reward-delay scheduling from `SeedSequence([seed, 0x3352444C])`, behavior actions from `[seed, 0x33414354]`, and reward shuffle from `[seed, 0x33534846]`. Reject duplicate seeds and preserve `[7, 17, 29]` ordering.

- [ ] **Step 4: Implement training with an environment-owned queue.**

At each episode, select one action using the existing learner boundary, enqueue the environment's scalar terminal reward with the registered `reward_delay`, advance the queue exactly that many steps, deliver exactly one reward, and call `learner.learn(reward)`. The learner must never receive `reward_delay` or queue state. For `reward_delay=0`, call `learn()` immediately after enqueue readiness.

- [ ] **Step 5: Run the continuity and queue-integration tests.**

Run `pytest tests/test_delayed_credit.py tests/test_phase3b_delayed_benchmark.py -q` and `ruff check src/neural_state_machine/delayed_credit.py src/neural_state_machine/phase3b_delayed_benchmark.py tests/test_delayed_credit.py tests/test_phase3b_delayed_benchmark.py`. Expected: queue timing passes and the `d_r=0` result matches the frozen Phase 3A boundary.

- [ ] **Step 6: Commit the harness.**

```bash
git add src/neural_state_machine/phase3b_delayed_benchmark.py tests/test_phase3b_delayed_benchmark.py
git commit -m "feat: add phase three-b delayed benchmark harness"
git push origin experiment/neural-state-machine
```

### Task 3: Add Arm A delayed TD(0) and Arm B episode-reset TD(λ)

**Files:**
- Modify: `src/neural_state_machine/phase3b_delayed_benchmark.py`
- Create: `src/neural_state_machine/phase3b_learners.py`
- Create: `tests/test_phase3b_learners.py`

**Interfaces:**
- Consumes: the delayed benchmark's decision-time hidden features and delivered scalar rewards.
- Produces: `DelayedTD0Adapter`, `EpisodeResetEligibilityTrace`, `select_for_training()`, `learn()`, `reset_episode()`, finite snapshots, and parameter digests.

- [ ] **Step 1: Write RED tests for adapter equivalence and episode trace reset.**

Require Arm A to delegate to the existing normalized action-value learner without changing its update equation. Require Arm B to accumulate a trace across decision steps within one episode, reset trace values to exact zero at terminal boundaries, reject duplicate pending feedback, and preserve action sequence equality under identical action RNGs.

- [ ] **Step 2: Run focused tests and verify RED.**

Run `pytest tests/test_phase3b_learners.py -q`. Expected result: missing adapter symbols.

- [ ] **Step 3: Implement Arm A as a thin adapter.**

Do not fork the TD(0) formula. Wrap `NormalizedActionValue` so the benchmark can call a common protocol while retaining parameter snapshots, pending-state checks, and overflow guards.

- [ ] **Step 4: Implement Arm B with terminal reset and candidate-state validation.**

Use `discount=0.9` and `trace_decay=0.8`. Update a candidate trace before committing it; reject non-finite trace candidates. On `learn()`, compute a candidate full weight matrix, reject any non-finite element without mutating weights or pending credit, then commit. `reset_episode()` clears only eligibility and pending state; it does not alter learned value weights.

- [ ] **Step 5: Run focused learner and regression tests.**

Run `pytest tests/test_phase3b_learners.py tests/test_phase3a_credit_compare.py tests/test_phase3a_credit_extreme.py -q` and `ruff check src/neural_state_machine/phase3b_learners.py tests/test_phase3b_learners.py`. The existing diagnostic TD(λ) tests and Phase 3A numerical-contract tests must remain unchanged and pass in an environment with pytest available.

- [ ] **Step 6: Commit the learner adapters.**

```bash
git add src/neural_state_machine/phase3b_learners.py src/neural_state_machine/phase3b_delayed_benchmark.py tests/test_phase3b_learners.py
git commit -m "feat: add delayed td credit learner arms"
git push origin experiment/neural-state-machine
```

### Task 4: Add shuffled, reset, action-lineage, and repeatability controls

**Files:**
- Modify: `src/neural_state_machine/phase3b_delayed_benchmark.py`
- Modify: `tests/test_phase3b_delayed_benchmark.py`

**Interfaces:**
- Consumes: both learner arms and the delayed queue.
- Produces: normal/shuffled/reset result fields, per-delay counts, action/reward/fixture digests, checkpoint summaries, and `all_passed` gate fields.

- [ ] **Step 1: Write RED control tests.**

Require normal and shuffled action digests to be equal, each reward-shuffle block to preserve its reward multiset, reset evaluations to use exact-zero value parameters, repeated runs to be byte-identical, and queue delivery counts to equal action counts with no stale entries. Mutate each control field and require payload validation to reject it.

- [ ] **Step 2: Implement independent RNG reconstruction.**

Construct normal and shuffled behavior generators separately from the same `[seed, 0x33414354]` lineage. Consume a separate reward-shuffle generator only for within-block permutations. Never reuse a generator object between arms when the protocol requires independent reconstruction.

- [ ] **Step 3: Implement reset and checkpoint measurements.**

Record post-training, reset, and shuffled accuracy overall and per cue-to-decision delay. Record checkpoints at episodes `100, 200, ..., 2000`, plus action and reward digests, pending state, finite-state checks, and parameter/recurrent integrity fields.

- [ ] **Step 4: Implement fixed gates without tuning.**

Compute the gate from the exact constants in the Global Constraints section. A failed arm or delay remains failed evidence; never replace it with a better-performing configuration.

- [ ] **Step 5: Run control tests and commit.**

Run `pytest tests/test_phase3b_delayed_benchmark.py tests/test_phase3b_learners.py -q`, `ruff check .`, and `git diff --check`. Commit:

```bash
git add src/neural_state_machine/phase3b_delayed_benchmark.py tests/test_phase3b_delayed_benchmark.py
git commit -m "test: add phase three-b integrity controls"
git push origin experiment/neural-state-machine
```

### Task 5: Add evidence serialization, atomic writer, and fail-closed verifier

**Files:**
- Create: `scripts/benchmark_phase3b_delayed_credit.py`
- Create: `scripts/verify_phase3b_delayed_credit.py`
- Create: `tests/test_phase3b_evidence.py`

**Interfaces:**
- Consumes: `run_delayed_credit_benchmark()` and validated result payloads.
- Produces: compact deterministic JSON, approved path `docs/experiments/phase-3b-delayed-credit.json`, portable projection, and `verify_phase3b_delayed_credit()`.

- [ ] **Step 1: Write RED schema and writer tests.**

Require exact phase/schema/config/seeds/delay axes, finite numeric summaries, valid digest formats, queue statistics, arm names, checkpoints, gates, repeatability, and `all_passed`. Reject missing keys, wrong types, duplicate seeds, changed frozen evidence hashes, NaN/Inf, arbitrary evidence paths, symlinks, and malformed source commits.

- [ ] **Step 2: Implement the compact benchmark script.**

Print one sorted JSON payload to stdout. Add `--evidence` only for the exact approved path. Exit zero only when the computed payload's `all_passed` is true; retain and serialize false gates without rewriting prior artifacts.

- [ ] **Step 3: Implement atomic evidence writing.**

Render `json.dumps(..., sort_keys=True, indent=2, allow_nan=False) + "\n"`; create a same-directory temporary file, flush and `fsync`, then `os.replace`. Reject symlinks, parent-directory escapes, and both Phase 2 evidence paths. Do not rewrite identical bytes.

- [ ] **Step 4: Implement verifier and portable projection.**

Run the benchmark twice and require exact equality. Verify action-lineage equality, reward-block fairness, queue counts, finite states, unchanged frozen evidence hashes, and the portable projection. The committed artifact's `source_commit` must be a lowercase 40-hex ancestor of the verifier's current HEAD; do not compare it with runtime output fields that are intentionally environment-local.

- [ ] **Step 5: Run evidence tests and commit tooling.**

Run `pytest tests/test_phase3b_evidence.py tests/test_phase3b_delayed_benchmark.py -q`, `ruff check scripts/benchmark_phase3b_delayed_credit.py scripts/verify_phase3b_delayed_credit.py tests/test_phase3b_evidence.py`, and `git diff --check`. Commit:

```bash
git add scripts/benchmark_phase3b_delayed_credit.py scripts/verify_phase3b_delayed_credit.py tests/test_phase3b_evidence.py
 git commit -m "test: add phase three-b evidence verification"
git push origin experiment/neural-state-machine
```

### Task 6: Measure once and freeze Phase 3B evidence

**Files:**
- Create: `docs/experiments/phase-3b-delayed-credit.json`
- Test: all Phase 3B tests and existing frozen tests.

**Interfaces:**
- Consumes: exact pushed implementation commit and verifier.
- Produces: one immutable measured artifact and one truthful decision boundary.

- [ ] **Step 1: Run the full local verification before measurement.**

Run `pytest -q`, `ruff check .`, the existing Phase 1/2 benchmark commands, Phase 3A verifier, and the new Phase 3B benchmark/verifier. If dependencies are unavailable, record that limitation rather than claiming a pass.

- [ ] **Step 2: Generate the artifact once.**

Run:

```bash
python scripts/benchmark_phase3b_delayed_credit.py --evidence docs/experiments/phase-3b-delayed-credit.json
python scripts/verify_phase3b_delayed_credit.py
sha256sum docs/experiments/phase-3a-action-value.json docs/experiments/phase-3b-delayed-credit.json
```

Do not regenerate after observing a failure except to reproduce the exact same bytes in the same environment.

- [ ] **Step 3: Commit the measured artifact.**

```bash
git add docs/experiments/phase-3b-delayed-credit.json
git commit -m "experiment: freeze phase three-b delayed-credit evidence"
git push origin experiment/neural-state-machine
```

### Task 7: Document results and wire exact-head CI

**Files:**
- Modify: `README.md`
- Modify: `.github/workflows/ci.yml`
- Create: `docs/experiments/phase-3b-delayed-credit-report.md`
- Create: `tests/test_phase3b_documentation.py`

**Interfaces:**
- Consumes: committed Phase 3B artifact and verifier output.
- Produces: truthful result report, CI verification, and frozen-boundary audit.

- [ ] **Step 1: Write documentation contract tests.**

Require README/report language to distinguish cue-to-decision delay from action-to-reward delay, state whether each arm passed, include artifact SHA-256, and explicitly state that no Phase 3B result proves emergent state machines or game competence.

- [ ] **Step 2: Add the result report without changing evidence.**

Record exact per-seed/per-arm/per-reward-delay counts, controls, queue statistics, repeatability, and the narrow decision rule. Keep failed results truthful and do not revise Phase 3A wording.

- [ ] **Step 3: Append CI jobs while preserving existing jobs.**

Add Python-version benchmark/verifier steps and an independent exact-head Phase 3B check. Do not remove or alter existing Python, formal, or frozen-evidence jobs.

- [ ] **Step 4: Run documentation, full tests, and diff checks.**

Run `pytest -q`, `ruff check .`, `git diff --check`, Phase 3A verifier, Phase 3B verifier, and the frozen-path diff audit. Commit:

```bash
git add README.md .github/workflows/ci.yml docs/experiments/phase-3b-delayed-credit-report.md tests/test_phase3b_documentation.py
git commit -m "docs: report phase three-b delayed credit evidence"
git push origin experiment/neural-state-machine
```

### Task 8: Exact-head remote audit and decision handoff

**Files:**
- No production changes; audit remote branch and workflow results.

- [ ] **Step 1: Verify orphan and frozen boundaries.**

Confirm `master` and `experiment/neural-state-machine` have no common ancestor, the remote SHA equals the pushed SHA, and every frozen path has the same blob digest as before Phase 3B.

- [ ] **Step 2: Verify exact-head CI.**

Require all Python matrix jobs, evidence verifier, and any formal job to pass for the exact pushed SHA. If workflow status is unavailable because the orphan branch has no PR-triggered run, report that fact instead of claiming CI success.

- [ ] **Step 3: Record one decision.**

Classify the result as: TD(0) succeeds under delay; TD(λ) adds measured benefit; both fail under delay; or harness invalid. Do not start game-engine, YOLO, FlyVis, fly-brain, or neural-state interpretation work from this plan.

- [ ] **Step 4: Commit only any required audit report.**

If the audit has no code/document change, leave the branch unchanged and report the exact remote SHA, artifact digest, and CI availability.

---

## Completion Handoff

The executor must report the exact remote SHA, all task commit subjects, full test and Ruff results, exact-head CI status, queue timing evidence, action/reward/fixture digests, per-seed/per-delay/per-arm counts, reset and shuffled controls, numerical-integrity results, frozen artifact hashes, and the single delayed-credit decision. No claim may exceed the measured protocol.
