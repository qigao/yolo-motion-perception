# Phase 2C Learning Failure Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Build a deterministic, label-isolated diagnostic ladder that explains whether Phase 2B failed because of recurrent representation, online optimization, or sampled reward credit while preserving the portable Phase 2B failure evidence and environment-local floating-point integrity.

**Architecture:** Collect complete immutable hidden datasets from the frozen recurrent policy before exposing labels to analyzers. Compare frozen Ridge geometry, a private online supervised softmax diagnostic, and an instrumented reproduction of the unchanged reward learner, then classify each seed using a fixed priority table. Treat the committed Phase 2B failure as a portable semantic regression contract, with stricter byte equality inside each runtime environment, so green CI means reproducible evidence rather than behavioral success.

**Tech Stack:** Python 3.10+, NumPy, pytest, Ruff, setuptools, GitHub Actions.

**Spec:** docs/superpowers/specs/2026-09-14-phase-2c-learning-diagnostics-design.md

## Global Constraints

- Work only on the standalone branch experiment/neural-state-machine.
- Do not merge, rebase onto, copy from, or modify master.
- NumPy remains the only runtime dependency.
- Do not modify policy.py, reward_readout.py, reward_learning.py, controller.py, memory_task.py, memory_probe.py, or memory_benchmark.py.
- Do not modify docs/experiments/phase-2b-failure.json.
- Existing Phase 1 and Phase 2A tests and benchmark evidence remain unchanged.
- The only permitted Phase 2B test edit changes obsolete success assertions into portable committed-failure regression assertions plus local float-integrity assertions.
- Labels and delay metadata may enter diagnostic analyzers only after hidden collection completes.
- RecurrentPolicy and RewardModulatedReadout never receive cue labels, correct actions, delay values, phases, probe outputs, or episode objects beyond the already frozen task-scoring boundary.
- Phase 2C does not tune learning rate, temperature, recurrent radius, seeds, episode counts, checkpoint interval, delay range, distractor amplitude, task vectors, or acceptance thresholds after observing results.
- Fixed defaults are hidden size 64, recurrent radius 0.9, learning rate 0.05, temperature 1.0, 2,000 training episodes, 20 evaluation blocks, checkpoint interval 100, and seeds 7, 17, and 29.
- Ridge geometry requires 200/200 overall, 40/40 at every delay, and strictly positive normalized signed margins.
- Online supervised behavior is reported against 180/200 overall and 34/40 per delay; a miss is a valid diagnostic result, not permission to tune.
- Every production behavior begins with a focused failing test.
- Commit after every task with the exact subject specified by that task.
- Push every completed commit to experiment/neural-state-machine without force.
- If Phase 2B portable reproduction differs from its committed JSON projection, or same-environment float integrity fails, classify PROTOCOL_MISMATCH, preserve evidence, and stop.
- If an integrity or determinism gate fails, preserve the exact output and return to design review.

## File and responsibility map

| File | Responsibility | Change |
|---|---|---|
| src/neural_state_machine/learning_diagnostics.py | Config, immutable datasets, geometry, supervised diagnostic, reward trajectory, classification, results | Create |
| tests/test_learning_diagnostics.py | Validation, numerics, boundaries, checkpoints, classifications, determinism | Create |
| scripts/verify_reward_learning_failure.py | Portable Phase 2B known-failure verifier plus local float integrity | Create |
| scripts/benchmark_learning_diagnostics.py | Stable Phase 2C JSON, evidence writer, exit status | Create |
| docs/experiments/phase-2c-diagnostics.json | Deterministic measured diagnostic evidence | Create after scientific run |
| src/neural_state_machine/__init__.py | Export approved Phase 2C public API | Modify |
| tests/test_reward_learning.py | Replace obsolete pass expectation with portable failure regression | Modify |
| README.md | Explain Phase 2B failure and Phase 2C diagnosis | Modify after evidence |
| .github/workflows/ci.yml | Verify Phase 2B failure and run Phase 2C benchmark | Modify |

---

### Task 1: Freeze the portable Phase 2B failure as a passing regression contract

**Files:**
- Create: scripts/verify_reward_learning_failure.py
- Modify: tests/test_reward_learning.py
- Test: tests/test_reward_learning.py
- Evidence: docs/experiments/phase-2b-failure.json

**Interfaces:**
- Consumes: run_reward_learning_benchmark() -> dict[str, object]
- Produces: `_portable_phase_2b_payload(payload) -> dict[str, object]`, `verify_reward_learning_failure() -> dict[str, object]`, and a CLI that exits zero only for portable known-failure reproduction plus local float integrity

- [ ] **Step 1: Replace the obsolete seed acceptance assertions with exact frozen counts**

Change the parametrized test to use literal expected results:

~~~python
@pytest.mark.parametrize(
    ("seed", "overall", "per_delay", "gate_passed"),
    [
        (7, 169, (40, 40, 40, 20, 29), False),
        (17, 200, (40, 40, 40, 40, 40), True),
        (29, 171, (40, 40, 40, 31, 20), False),
    ],
)
def test_phase_two_b_observed_failure_is_frozen(
    seed: int,
    overall: int,
    per_delay: tuple[int, ...],
    gate_passed: bool,
) -> None:
    result = run_reward_learning_experiment(seed)
    assert result.post_training == AccuracyCount(overall, 200)
    assert tuple(score.correct for _, score in result.per_delay) == per_delay
    assert (overall >= 180 and all(value >= 34 for value in per_delay)) is gate_passed
    assert result.state_reset == AccuracyCount(100, 200)
    assert result.shuffled_control == AccuracyCount(100, 200)
    assert result.repeatable is True
~~~

- [ ] **Step 2: Add a failing portable-evidence test for the missing verifier**

~~~python
def test_phase_two_b_runtime_payload_matches_committed_failure() -> None:
    from scripts.verify_reward_learning_failure import (
        _portable_phase_2b_payload,
        verify_reward_learning_failure,
    )

    payload = verify_reward_learning_failure()
    assert payload["all_passed"] is False
    expected = json.loads(_PHASE_2B_EVIDENCE.read_text())
    assert payload == _portable_phase_2b_payload(expected)
~~~

Run:

~~~bash
.venv/bin/pytest tests/test_reward_learning.py -q
~~~

Expected RED: ModuleNotFoundError for scripts.verify_reward_learning_failure after the frozen observed-count tests pass.

- [ ] **Step 3: Implement the portable verifier and local integrity checks**

Create scripts/verify_reward_learning_failure.py. `_portable_phase_2b_payload`
must retain the top-level phase, config, ordered seeds, all_passed, and pooled
shuffled count. In each result retain the seed/pass flags, all accuracy counts,
per-delay and final-block counts, total rewards, reset equality, repeatability,
and every action/reward sequence digest. Exclude `initial_parameter_digest`,
`normal_parameter_digest`, `shuffled_parameter_digest`, and `matrix_controls`
from cross-environment comparison.

The verifier runs the benchmark twice from newly reconstructed objects,
requires both full payloads to be exactly equal in the current environment,
compares their portable projection to the committed evidence projection, and
checks per result:

- `matrix_controls.normal_before == matrix_controls.normal_after`;
- `matrix_controls.shuffled_before == matrix_controls.shuffled_after`;
- the learned normal and shuffled parameter digests differ from their initial
  digest as required by the frozen training protocol.

The core shape is:

~~~python
from __future__ import annotations

import json
from pathlib import Path

from neural_state_machine import run_reward_learning_benchmark

_EVIDENCE = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "experiments"
    / "phase-2b-failure.json"
)


def verify_reward_learning_failure() -> dict[str, object]:
    expected = json.loads(_EVIDENCE.read_text(encoding="utf-8"))
    first = run_reward_learning_benchmark()
    second = run_reward_learning_benchmark()
    if expected.get("phase") != "2B":
        raise RuntimeError("Phase 2B evidence has an unexpected phase")
    if expected.get("seeds") != [7, 17, 29]:
        raise RuntimeError("Phase 2B evidence has unexpected seeds")
    if expected.get("all_passed") is not False:
        raise RuntimeError("Phase 2B evidence must record a failed gate")
    if first != second:
        raise RuntimeError("Phase 2B is not byte-stable in this environment")
    if _portable_phase_2b_payload(first) != _portable_phase_2b_payload(expected):
        raise RuntimeError("Phase 2B portable runtime differs from frozen evidence")
    _verify_local_float_integrity(first)
    return _portable_phase_2b_payload(first)


def main() -> int:
    payload = verify_reward_learning_failure()
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
~~~

- [ ] **Step 4: Correct existing benchmark/CLI assertions without weakening evidence**

Require the public benchmark's portable projection to equal the committed
projection and require `all_passed` is false. Run the unchanged
benchmark_reward_learning.py twice; require both current-environment stdout
payloads to be byte-identical, exit one, satisfy local float integrity, and
have the committed portable projection. Remove only assertions that demand
all_passed is true or exit status zero.

- [ ] **Step 5: Run focused verification**

~~~bash
.venv/bin/pytest tests/test_reward_learning.py -q
.venv/bin/python scripts/verify_reward_learning_failure.py
.venv/bin/ruff check tests/test_reward_learning.py scripts/verify_reward_learning_failure.py
~~~

Expected GREEN: all focused tests pass; the verifier exits zero; its payload
still records all_passed=false.

- [ ] **Step 6: Commit and push**

~~~bash
git add tests/test_reward_learning.py scripts/verify_reward_learning_failure.py
git commit -m "test: freeze phase two-b failure contract"
git push origin experiment/neural-state-machine
~~~

---

### Task 2: Add fixed diagnostic configuration and immutable hidden datasets

**Files:**
- Create: src/neural_state_machine/learning_diagnostics.py
- Create: tests/test_learning_diagnostics.py
- Test: tests/test_learning_diagnostics.py

**Interfaces:**
- Consumes: RewardLearningConfig, DelayedCueEpisode, DelayedCueTask, RecurrentPolicy, reward_learning._build_fixtures(), reward_learning._decision_hidden()
- Produces:
  - LearningDiagnosticsConfig
  - _HiddenDataset
  - _DiagnosticFixtures
  - _build_diagnostic_fixtures(seed: int, config: LearningDiagnosticsConfig) -> _DiagnosticFixtures

- [ ] **Step 1: Write failing configuration tests**

Require this frozen dataclass and strict boolean rejection:

~~~python
@dataclass(frozen=True)
class LearningDiagnosticsConfig:
    hidden_size: int = 64
    recurrent_radius: float = 0.9
    learning_rate: float = 0.05
    temperature: float = 1.0
    training_episodes: int = 2_000
    evaluation_blocks: int = 20
    checkpoint_interval: int = 100
~~~

Parametrize invalid values covering zero, negatives, booleans, floats for
integer fields, NaN/infinity, training counts not divisible by ten, checkpoint
intervals not divisible by ten, and intervals that do not divide
training_episodes.

- [ ] **Step 2: Write failing immutable-dataset tests**

Use this internal shape:

~~~python
@dataclass(frozen=True)
class _HiddenDataset:
    states: np.ndarray
    labels: np.ndarray
    delays: np.ndarray
    fixture_digest: str
    state_digest: str
~~~

Require exact float64/int64 dtypes, C contiguity, read-only arrays, independent
copies, matching row counts, finite states, action labels in zero/one, delays
in one through five, both classes, and immutable digest strings.

- [ ] **Step 3: Write the label-boundary recording test**

Use a recording policy whose reset_state and advance methods accept numeric
arrays only. Require all hidden rows to be collected before the helper reads
correct_action_index or delay_steps by wrapping fixture metadata access in a
guard that opens only after collection. Require no label-bearing value in any
policy call.

- [ ] **Step 4: Write exact fixture and seed-lineage tests**

For each seed require 2,000 training episodes in 200 balanced ten-case blocks,
200 evaluation episodes in 20 balanced ten-case blocks, 100 evaluation labels
per action, 40 rows per delay, fresh immutable distractors, and identical
fixture/state digests from independent reconstruction.

Run:

~~~bash
.venv/bin/pytest tests/test_learning_diagnostics.py -q
~~~

Expected RED: ModuleNotFoundError for neural_state_machine.learning_diagnostics.

- [ ] **Step 5: Implement configuration, dataset validation, and fixture collection**

Build exact RNGs:

~~~python
training_rng = np.random.default_rng(
    np.random.SeedSequence([seed, 0x54524149])
)
evaluation_rng = np.random.default_rng(
    np.random.SeedSequence([seed, 0x4556414C])
)
~~~

Construct one frozen policy per dataset collection with input size 4, action
count 2, the fixed hidden size, seed, and recurrent radius. Copy hidden rows
before associating labels and delays. Hash array shape plus contiguous bytes;
hash every episode stimulus shape plus bytes for fixture_digest.

- [ ] **Step 6: Run focused tests and Ruff**

~~~bash
.venv/bin/pytest tests/test_learning_diagnostics.py -q
.venv/bin/ruff check src/neural_state_machine/learning_diagnostics.py tests/test_learning_diagnostics.py
~~~

- [ ] **Step 7: Commit and push**

~~~bash
git add src/neural_state_machine/learning_diagnostics.py tests/test_learning_diagnostics.py
git commit -m "feat: add immutable phase two-c datasets"
git push origin experiment/neural-state-machine
~~~

---

### Task 3: Measure frozen Ridge geometry and normalized signed margins

**Files:**
- Modify: src/neural_state_machine/learning_diagnostics.py
- Modify: tests/test_learning_diagnostics.py
- Test: tests/test_learning_diagnostics.py

**Interfaces:**
- Consumes: _HiddenDataset, fit_linear_probe()
- Produces:
  - MarginSummary
  - GeometryDiagnostic
  - _normalized_signed_margins(probe, dataset) -> np.ndarray
  - _run_geometry(training, evaluation) -> GeometryDiagnostic

- [ ] **Step 1: Write the literal normalized-margin RED test**

Use a hand-constructed FittedLinearProbe and state matrix. Independently
calculate:

~~~python
signed = np.where(labels == 0, -1.0, 1.0)
expected = signed * (states @ probe.weights + probe.bias) / np.linalg.norm(
    probe.weights
)
~~~

Require exact shape, float64 dtype, read-only output, zero relative tolerance,
and 1e-12 absolute tolerance.

- [ ] **Step 2: Write invalid-geometry RED tests**

Reject zero probe norm, empty data, missing either class, mismatched labels,
non-finite states, and a delay with zero samples. Verify failures do not mutate
datasets or probe parameters.

- [ ] **Step 3: Write percentile and per-delay RED tests**

Define:

~~~python
@dataclass(frozen=True)
class MarginSummary:
    minimum: float
    percentile_10: float
    median: float
~~~

For literal margin fixtures, require np.percentile(values, 10, method="linear")
and np.median semantics. Require delays ordered one through five and
delay_five_to_one_median_ratio computed only from the two medians.

- [ ] **Step 4: Run focused RED**

~~~bash
.venv/bin/pytest tests/test_learning_diagnostics.py -q
~~~

Expected RED: missing margin interfaces.

- [ ] **Step 5: Implement geometry using the frozen Phase 2A fitter**

Call:

~~~python
probe = fit_linear_probe(
    training.states,
    training.labels,
    regularization=1e-3,
)
~~~

Use the fitted probe for both train and evaluation. Store overall and per-delay
AccuracyCount values, summaries, probe digest, and a geometry_passed boolean
that is true only for 200/200, five 40/40 delay counts, and all margins greater
than zero.

- [ ] **Step 6: Add real-seed frozen-gate tests**

For seeds 7, 17, and 29 require 200/200, 40/40 for every delay, strictly
positive minima, finite ratios, and unchanged dataset/policy digests. Do not
assert an unregistered magnitude threshold for the ratio.

- [ ] **Step 7: Run tests and Ruff**

~~~bash
.venv/bin/pytest tests/test_learning_diagnostics.py -q
.venv/bin/ruff check src/neural_state_machine/learning_diagnostics.py tests/test_learning_diagnostics.py
~~~

- [ ] **Step 8: Commit and push**

~~~bash
git add src/neural_state_machine/learning_diagnostics.py tests/test_learning_diagnostics.py
git commit -m "feat: measure phase two-c state geometry"
git push origin experiment/neural-state-machine
~~~

---

### Task 4: Add the private online supervised softmax diagnostic

**Files:**
- Modify: src/neural_state_machine/learning_diagnostics.py
- Modify: tests/test_learning_diagnostics.py
- Test: tests/test_learning_diagnostics.py

**Interfaces:**
- Consumes: _HiddenDataset, LearningDiagnosticsConfig, AccuracyCount
- Produces:
  - DiagnosticCheckpoint
  - SupervisedDiagnostic
  - _DiagnosticSupervisedReadout
  - _run_supervised_diagnostic(training, evaluation, config) -> SupervisedDiagnostic

- [ ] **Step 1: Write zero-init, softmax, and greedy RED tests**

Require two zero-initialized float64 weight rows and biases, stable
temperature-scaled softmax, action zero on an exact tie, immutable probability
snapshots, and constructor validation matching LearningDiagnosticsConfig.

- [ ] **Step 2: Write the literal label-update RED test**

For a known hidden vector and label, independently compute:

~~~python
delta = np.eye(2, dtype=np.float64)[label] - probabilities
expected_weights = before_weights + learning_rate * np.outer(delta, hidden)
expected_biases = before_biases + learning_rate * delta
~~~

Require every element within absolute tolerance 1e-12 and prove label update
does not sample an action or accept scalar reward through a learn method.

- [ ] **Step 3: Write checkpoint schedule RED tests**

Define:

~~~python
@dataclass(frozen=True)
class DiagnosticCheckpoint:
    episode: int
    overall: AccuracyCount
    per_delay: tuple[tuple[int, AccuracyCount], ...]
    parameter_digest: str
~~~

For training_episodes=20 and checkpoint_interval=10, require checkpoints
exactly at 10 and 20. Evaluation must not change the digest at each checkpoint.

- [ ] **Step 4: Write the label-isolation RED test**

A spy around the frozen policy and RewardModulatedReadout must observe no label
calls. Only _DiagnosticSupervisedReadout.observe_label(hidden, label) may
receive the correct action, and it must receive a copied numeric hidden vector
rather than an episode.

- [ ] **Step 5: Run focused RED**

~~~bash
.venv/bin/pytest tests/test_learning_diagnostics.py -q
~~~

Expected RED: supervised diagnostic interfaces are absent.

- [ ] **Step 6: Implement the private supervised readout and training loop**

Process training.states and training.labels in stored order. At each checkpoint
evaluate greedily on evaluation.states, score labels outside the readout, and
store immutable counts and digests. Set supervised_passed only when final
overall is at least 180/200 and every delay is at least 34/40.

- [ ] **Step 7: Run focused tests and Ruff**

~~~bash
.venv/bin/pytest tests/test_learning_diagnostics.py -q
.venv/bin/ruff check src/neural_state_machine/learning_diagnostics.py tests/test_learning_diagnostics.py
~~~

- [ ] **Step 8: Commit and push**

~~~bash
git add src/neural_state_machine/learning_diagnostics.py tests/test_learning_diagnostics.py
git commit -m "feat: add online supervised diagnostic"
git push origin experiment/neural-state-machine
~~~

---

### Task 5: Instrument the unchanged reward trajectory and match frozen evidence

**Files:**
- Modify: src/neural_state_machine/learning_diagnostics.py
- Modify: tests/test_learning_diagnostics.py
- Test: tests/test_learning_diagnostics.py
- Evidence: docs/experiments/phase-2b-failure.json

**Interfaces:**
- Consumes: RewardModulatedReadout, DelayedCueTask.reward(), _DiagnosticFixtures, run_reward_learning_benchmark()
- Produces:
  - GradientBlockDiagnostic
  - RewardCheckpoint
  - RewardTrajectoryDiagnostic
  - _run_reward_trajectory(seed, fixtures, config) -> RewardTrajectoryDiagnostic
  - _match_phase_2b_evidence(seed, runtime_entry) -> bool

- [ ] **Step 1: Write correct-action probability RED tests**

Given literal returned probability vectors and labels, require the diagnostic
to select the correct component only after action selection, then calculate
mean and np.percentile(values, 10, method="linear") exactly.

- [ ] **Step 2: Write independent gradient RED tests**

For augmented x = concatenate((hidden, [1.0])), independently compute:

~~~python
supervised = np.outer(one_hot_label - probabilities, x)
sampled = reward * np.outer(one_hot_action - probabilities, x)
~~~

Require the diagnostic's flattened per-episode vectors and exact
expected-bandit-to-supervised norm ratio to match. Accumulate ten vectors per
block and calculate cosine by dot product divided by both norms. Require None
when either norm is zero and increment zero_norm_block_count.

- [ ] **Step 3: Write reward-boundary RED tests**

Use spies to prove the call sequence contains only:

~~~text
readout.select_for_training(hidden, (0, 1), action_rng)
task.reward(episode, selected_action)
readout.learn(float_reward)
~~~

Assert label/delay access used for diagnostics occurs after select_for_training
and cannot alter the stored decision, reward, or learn argument.

- [ ] **Step 4: Write checkpoint and non-mutation RED tests**

For a 20-episode configuration require checkpoints at 10 and 20. Snapshot the
readout digest before and after each greedy checkpoint evaluation and require
equality. Require no pending feedback after every completed training episode.

- [ ] **Step 5: Write portable Phase 2B reproduction RED tests**

Load the committed evidence. For each seed, run the unchanged public Phase 2B
benchmark and require its portable result projection to equal the matching
evidence projection. Require the trajectory final overall, per-delay counts,
total reward, choice digest, and reward digest to equal the same entry. Require
the trajectory's parameter and matrix digests to match a second independent
run in the current environment and satisfy before/after immutability; do not
compare floating byte digests across environments.

- [ ] **Step 6: Run focused RED**

~~~bash
.venv/bin/pytest tests/test_learning_diagnostics.py -q
~~~

Expected RED: reward trajectory interfaces are absent.

- [ ] **Step 7: Implement trajectory instrumentation outside the learner**

Use the exact training action RNG:

~~~python
action_rng = np.random.default_rng(
    np.random.SeedSequence([seed, 0x4143544E])
)
~~~

Call the existing readout API unchanged. Compute all diagnostic values from
read-only copies of hidden, returned probabilities, selected action, and later
task reward. Never write diagnostic data into the learner.

- [ ] **Step 8: Run focused tests, evidence verifier, and Ruff**

~~~bash
.venv/bin/pytest tests/test_learning_diagnostics.py tests/test_reward_learning.py -q
.venv/bin/python scripts/verify_reward_learning_failure.py
.venv/bin/ruff check src/neural_state_machine/learning_diagnostics.py tests/test_learning_diagnostics.py
~~~

- [ ] **Step 9: Commit and push**

~~~bash
git add src/neural_state_machine/learning_diagnostics.py tests/test_learning_diagnostics.py
git commit -m "feat: trace phase two-b reward learning"
git push origin experiment/neural-state-machine
~~~

---

### Task 6: Add deterministic diagnosis classification and public results

**Files:**
- Modify: src/neural_state_machine/learning_diagnostics.py
- Modify: tests/test_learning_diagnostics.py
- Modify: src/neural_state_machine/__init__.py
- Test: tests/test_learning_diagnostics.py

**Interfaces:**
- Consumes: GeometryDiagnostic, SupervisedDiagnostic, RewardTrajectoryDiagnostic
- Produces:
  - LearningDiagnosticsResult
  - run_learning_diagnostics(seed: int = 7, config: LearningDiagnosticsConfig | None = None) -> LearningDiagnosticsResult
  - _classify(protocol_match, geometry_passed, supervised_passed, reward_passed) -> str

- [ ] **Step 1: Write classification priority RED tests**

Parametrize all meaningful combinations and require:

~~~python
(False, True, True, False) -> "PROTOCOL_MISMATCH"
(True, False, True, False) -> "REPRESENTATION_FAILURE"
(True, True, False, False) -> "ONLINE_OPTIMIZATION_FAILURE"
(True, True, True, False) -> "REWARD_CREDIT_FAILURE"
(True, True, True, True) -> "NO_FAILURE_REPRODUCED"
~~~

Protocol mismatch must win regardless of later flags. Representation failure
must win before optimizer or reward outcomes.

- [ ] **Step 2: Write frozen public-result RED tests**

LearningDiagnosticsResult contains seed, config, geometry, supervised,
reward_trajectory, classification, matrix digests before/after,
phase_2b_portable_evidence_match, diagnostic_valid, and repeatable. Require
mutation to raise FrozenInstanceError and nested arrays to be absent from the
public surface.

- [ ] **Step 3: Write seed and configuration validation RED tests**

Reject booleans, negative values, floats, strings, and None as seeds before any
policy or RNG constructor is called. Reject non-LearningDiagnosticsConfig
config objects.

- [ ] **Step 4: Write true independent-repeatability RED tests**

Call the internal complete run twice with reconstructed objects. Require full
internal result equality, then set repeatable from that equality. A test double
that returns different second-run digests must produce repeatable=false and
diagnostic_valid=false.

- [ ] **Step 5: Run focused RED**

~~~bash
.venv/bin/pytest tests/test_learning_diagnostics.py -q
~~~

Expected RED: public result and classification interfaces are absent.

- [ ] **Step 6: Implement classification, validity, and public assembly**

diagnostic_valid is true only when portable evidence matches, all
environment-local frozen matrix digests match before/after and across the two
independent executions, datasets and checkpoints validate, every seed run has
one classification, and repeatable is true. Behavioral pass/fail selects the
classification but does not independently invalidate a correctly measured
run.

- [ ] **Step 7: Export only the approved public API**

Add to src/neural_state_machine/__init__.py:

~~~python
from .learning_diagnostics import (
    LearningDiagnosticsConfig,
    LearningDiagnosticsResult,
    run_learning_diagnostics,
    run_learning_diagnostics_benchmark,
)
~~~

Do not export _DiagnosticSupervisedReadout, _HiddenDataset, or internal
analyzers.

- [ ] **Step 8: Run focused tests and Ruff**

~~~bash
.venv/bin/pytest tests/test_learning_diagnostics.py tests/test_reward_learning.py -q
.venv/bin/ruff check src/neural_state_machine/learning_diagnostics.py src/neural_state_machine/__init__.py tests/test_learning_diagnostics.py
~~~

- [ ] **Step 9: Commit and push**

~~~bash
git add src/neural_state_machine/learning_diagnostics.py src/neural_state_machine/__init__.py tests/test_learning_diagnostics.py
git commit -m "feat: classify phase two-c learning failures"
git push origin experiment/neural-state-machine
~~~

---

### Task 7: Add the three-seed benchmark and immutable diagnostic evidence

**Files:**
- Modify: src/neural_state_machine/learning_diagnostics.py
- Modify: tests/test_learning_diagnostics.py
- Create: scripts/benchmark_learning_diagnostics.py
- Create after a valid run: docs/experiments/phase-2c-diagnostics.json
- Test: tests/test_learning_diagnostics.py
- Test: scripts/benchmark_learning_diagnostics.py

**Interfaces:**
- Consumes: run_learning_diagnostics()
- Produces:
  - run_learning_diagnostics_benchmark(seeds: Sequence[int] = (7, 17, 29), config: LearningDiagnosticsConfig | None = None) -> dict[str, object]
  - compact stdout CLI
  - optional deterministic pretty evidence output

- [ ] **Step 1: Write benchmark validation RED tests**

Reject empty seeds, duplicates, booleans, negatives, floats, non-sequences,
and invalid config before constructing policies. Require output phase "2C",
ordered seeds [7, 17, 29], classification_counts, phase_2b_evidence_digest,
results, and all_valid.

- [ ] **Step 2: Write stable serialization RED tests**

Run the CLI twice in subprocesses. Require byte-identical stdout, empty stderr,
exactly one newline, sorted compact JSON, and exit status equal to
int(not payload["all_valid"]).

- [ ] **Step 3: Write evidence-writer RED tests**

The CLI accepts:

~~~text
--evidence docs/experiments/phase-2c-diagnostics.json
~~~

Require the file to contain the same semantic payload as stdout, formatted by
json.dumps(payload, sort_keys=True, indent=2) plus one newline. Require an
existing non-identical file to cause a nonzero exit unless the target is the
approved Phase 2C evidence path; never permit writing over Phase 2B evidence.

- [ ] **Step 4: Run focused RED**

~~~bash
.venv/bin/pytest tests/test_learning_diagnostics.py -q
~~~

Expected RED: benchmark and CLI do not exist.

- [ ] **Step 5: Implement benchmark and CLI**

Use argparse with only the optional --evidence path. Compute
classification_counts from literal result classifications. all_valid is true
only when every result has diagnostic_valid=true and repeatable=true. Serialize
counts as integers and retain unrounded floating measurements.

- [ ] **Step 6: Run the pre-registered three-seed diagnostic gate**

~~~bash
.venv/bin/pytest tests/test_learning_diagnostics.py -q
.venv/bin/python scripts/benchmark_learning_diagnostics.py
~~~

If all_valid is false, save the compact stdout and failing seed details in the
commit message body, commit the truthful failing evidence, push, and stop for
design review. Do not change any fixed value.

- [ ] **Step 7: Generate and verify immutable evidence only after a valid run**

~~~bash
.venv/bin/python scripts/benchmark_learning_diagnostics.py   --evidence docs/experiments/phase-2c-diagnostics.json
.venv/bin/python -m json.tool docs/experiments/phase-2c-diagnostics.json >/dev/null
~~~

Run the command a second time and require no file-content change.

- [ ] **Step 8: Run focused tests and Ruff**

~~~bash
.venv/bin/pytest tests/test_learning_diagnostics.py tests/test_reward_learning.py -q
.venv/bin/ruff check src/neural_state_machine/learning_diagnostics.py tests/test_learning_diagnostics.py scripts/benchmark_learning_diagnostics.py
~~~

- [ ] **Step 9: Commit and push**

~~~bash
git add src/neural_state_machine/learning_diagnostics.py tests/test_learning_diagnostics.py scripts/benchmark_learning_diagnostics.py docs/experiments/phase-2c-diagnostics.json
git commit -m "feat: add phase two-c diagnostic benchmark"
git push origin experiment/neural-state-machine
~~~

---

### Task 8: Document evidence, wire CI, and perform the final audit

**Files:**
- Modify: README.md
- Modify: .github/workflows/ci.yml
- Test: complete repository
- Audit: all frozen files and evidence

**Interfaces:**
- Consumes: Phase 2B and Phase 2C benchmark JSON
- Produces: documented interpretation and exact-head three-version CI

- [ ] **Step 1: Add the Phase 2C README section**

Document:

- Phase 2A proves decodability;
- Phase 2B remains a pre-registered failed reward-learning result;
- Phase 2C diagnoses rather than repairs that failure;
- the three diagnostic branches and label-isolation boundary;
- fixed configuration and seeds;
- exact classifications, supervised counts, per-delay counts, and margin
  summaries copied from docs/experiments/phase-2c-diagnostics.json;
- command python scripts/benchmark_learning_diagnostics.py;
- green CI means portable failure reproduction, local float integrity, and
  valid diagnostics;
- the next phase requires a new design review.

Do not describe Phase 2B as passing and do not round away literal counts.

- [ ] **Step 2: Append portable failure verification and diagnostics to CI**

Retain every existing CI step. Add after Phase 2A:

~~~yaml
      - name: Verify frozen Phase 2B failure
        run: python scripts/verify_reward_learning_failure.py

      - name: Phase 2C learning diagnostics
        run: python scripts/benchmark_learning_diagnostics.py
~~~

Do not run the expected-nonzero benchmark_reward_learning.py directly as a
success command.

- [ ] **Step 3: Run the no-label-leakage audit**

~~~bash
rg -n "correct_action_index|[.]cue|delay_steps|episode_phase|current_phase|remembered_cue|last_cue|probe"   src/neural_state_machine/policy.py   src/neural_state_machine/reward_readout.py   src/neural_state_machine/reward_learning.py   src/neural_state_machine/controller.py
~~~

Inspect every pre-existing match. The diff introduced by Phase 2C must add no
match to these frozen files.

- [ ] **Step 4: Prove frozen artifacts are unchanged**

Use the Phase 2C spec commit as the comparison base:

~~~bash
git diff --exit-code 4272b225621ef4413a82a518412ed0b4b3f50c39..HEAD --   src/neural_state_machine/policy.py   src/neural_state_machine/reward_readout.py   src/neural_state_machine/reward_learning.py   src/neural_state_machine/controller.py   src/neural_state_machine/memory_task.py   src/neural_state_machine/memory_probe.py   src/neural_state_machine/memory_benchmark.py   docs/experiments/phase-2b-failure.json   tests/test_policy.py   tests/test_phase1_compatibility.py   tests/test_memory_task.py   tests/test_memory_benchmark.py   tests/test_memory_probe.py
~~~

Expected: exit zero and no output.

- [ ] **Step 5: Run complete verification in order**

~~~bash
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/pytest -q
.venv/bin/ruff check .
.venv/bin/python scripts/benchmark.py
.venv/bin/python scripts/benchmark_memory_probe.py
.venv/bin/python scripts/verify_reward_learning_failure.py
.venv/bin/python scripts/benchmark_learning_diagnostics.py
git diff --check
git status --short --branch
~~~

Read every output. Require zero command failures. Confirm the Phase 1 JSON is
byte-identical to its frozen output, Phase 2A remains 200/200 and 40/40 at all
delays for all seeds, Phase 2B matches its failure JSON's portable projection
and passes local float-integrity checks, and Phase 2C all_valid is true.

- [ ] **Step 6: Commit documentation and CI**

~~~bash
git add README.md .github/workflows/ci.yml
git commit -m "docs: report phase two-c learning diagnosis"
git push origin experiment/neural-state-machine
~~~

- [ ] **Step 7: Verify branch isolation and remote tree**

~~~bash
git fetch origin
test "$(git rev-parse HEAD)" = "$(git rev-parse origin/experiment/neural-state-machine)"
test "$(git rev-parse HEAD^{tree})" = "$(git rev-parse origin/experiment/neural-state-machine^{tree})"
test -z "$(git merge-base origin/master origin/experiment/neural-state-machine || true)"
~~~

Expected: all commands exit zero; the experiment branch still has no merge
base with master.

- [ ] **Step 8: Require exact-head GitHub Actions success**

Inspect the push run for the exact final SHA. Require standalone jobs for
Python 3.10, 3.11, and 3.12 to pass unit tests, Ruff, Phase 1, Phase 2A,
frozen Phase 2B failure verification, and Phase 2C diagnostics.

If CI differs from local verification, use superpowers:systematic-debugging,
add one focused portability test before any implementation fix, commit the fix
separately, push without force, and repeat exact-head verification.

## Completion handoff

The final report must include:

- exact final remote SHA and experiment branch URL;
- every implementation commit and subject;
- complete pytest count and Ruff result;
- unchanged Phase 1 JSON;
- unchanged Phase 2A counts and digests;
- portable Phase 2B failure-evidence reproduction and local float integrity;
- Phase 2C classification for each seed;
- Ridge margin minimum, 10th percentile, and median for each delay;
- online supervised overall and per-delay counts for every seed;
- reward and supervised checkpoint summaries;
- expected-gradient scale and zero-norm/cosine evidence;
- frozen reservoir and legacy-output digests;
- independent-run repeatability;
- exact Phase 2C evidence path and digest;
- exact-head GitHub Actions URL;
- a truthful invalid-diagnostic conclusion if any integrity gate misses.
