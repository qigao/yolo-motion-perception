# Phase 2C Learning Failure Diagnostics Design

## Decision

Phase 2C will diagnose why the pre-registered Phase 2B reward learner failed
for seeds 7 and 29 even though Phase 2A proved that a frozen linear probe can
decode the vanished cue perfectly.

This phase is measurement-only. It will not tune Phase 2B, replace its learning
rule, alter its fixed seeds or thresholds, or introduce a new production game
controller. It adds an isolated diagnostic ladder over the same frozen
recurrent states:

1. offline Ridge geometry and normalized signed margins;
2. an online supervised softmax readout used only as a diagnostic instrument;
3. an instrumented reproduction of the unchanged Phase 2B reward learner.

Labels and delay metadata may enter the diagnostic analyzers only after the
complete numeric hidden datasets have been collected. They never enter
RecurrentPolicy, RewardModulatedReadout, the game controller, or action
selection.

## Evidence that motivates Phase 2C

The immutable Phase 2B evidence is stored in
docs/experiments/phase-2b-failure.json. Under hidden size 64, recurrent radius
0.9, learning rate 0.05, temperature 1.0, 2,000 training episodes, 20
evaluation blocks, and seeds 7, 17, and 29, it records:

| Seed | Reward evaluation | Delay 1 / 2 / 3 / 4 / 5 | Reset | Shuffled |
|---:|---:|---|---:|---:|
| 7 | 169/200 | 40/40 / 40/40 / 40/40 / 20/40 / 29/40 | 100/200 | 100/200 |
| 17 | 200/200 | 40/40 / 40/40 / 40/40 / 40/40 / 40/40 | 100/200 | 100/200 |
| 29 | 171/200 | 40/40 / 40/40 / 40/40 / 31/40 / 20/40 | 100/200 | 100/200 |

The pooled shuffled control is exactly 300/600. Every reset evaluation is
exactly balanced, all reset hidden vectors are equal, every independent rerun
is identical, and the input, recurrent, and legacy output matrices remain
byte-identical.

Phase 2A used a frozen Ridge probe and decoded 200/200 for every one of the
same seeds, including 40/40 at every delay. Therefore Phase 2C must distinguish
among fragile representation geometry, online optimization difficulty, and
bandit reward-credit difficulty without changing the observed system.

## Scope

### In scope

- Construct the exact Phase 2B training and evaluation fixtures from the same
  SeedSequence lineages.
- Collect immutable training and evaluation hidden-state datasets from a
  frozen RecurrentPolicy.
- Reuse the Phase 2A Ridge fitting equation to measure linear geometry.
- Compute normalized signed margins by seed and delay.
- Train a separate online supervised softmax readout over the collected hidden
  vectors in the exact fixture order.
- Reproduce the Phase 2B reward learner with read-only checkpoints every 100
  episodes.
- Measure correct-action probability and sampled-versus-supervised gradient
  behavior without feeding those labels into the reward learner.
- Produce one deterministic classification for each seed and a stable JSON
  report.
- Convert the known Phase 2B CI failure into an exact frozen-failure
  regression: green means the failure is reproduced exactly, not that Phase
  2B passed.

### Out of scope

- Changing any Phase 2B production module or learning equation.
- Tuning learning rate, temperature, recurrent radius, episode count, seed,
  task vectors, delay range, distractor range, or acceptance thresholds.
- Adding temporal eligibility traces, BPTT, recurrent plasticity, STDP, a
  nonlinear classifier, or a reinforcement-learning framework.
- Passing labels, cues, delays, phases, probe predictions, or episode objects
  into RecurrentPolicy or RewardModulatedReadout.
- Adding multi-memory, attack, retreat, heal, boss, upgrade, FlyVis, or
  fly-brain behavior.
- Claiming biological plausibility, semantic attractors, or a general
  replacement for finite-state machines.

A later design review may select a new learning mechanism after Phase 2C
identifies the failure class. Phase 2C itself cannot make that selection.

## Frozen boundaries

The following production and scientific artifacts remain byte-for-byte
unchanged:

- src/neural_state_machine/policy.py
- src/neural_state_machine/reward_readout.py
- src/neural_state_machine/reward_learning.py
- src/neural_state_machine/controller.py
- src/neural_state_machine/memory_task.py
- src/neural_state_machine/memory_probe.py
- src/neural_state_machine/memory_benchmark.py
- docs/experiments/phase-2b-failure.json
- existing Phase 1 and Phase 2A tests and benchmark evidence

The only Phase 2B test change permitted is replacing assertions that still
expect all_passed=true with assertions that reproduce the committed
all_passed=false evidence exactly. No production behavior or scientific
threshold changes with that test correction.

NumPy remains the only runtime dependency.

## Architecture

The new module is src/neural_state_machine/learning_diagnostics.py.

Its data flow is:

    immutable delayed-cue fixtures
                |
                v
       frozen RecurrentPolicy
                |
                v
       immutable hidden datasets
          /           |           \
         v            v            v
    Ridge margin   supervised   reward replay
      analysis      diagnostic   instrumentation
          \           |           /
                v
       deterministic diagnosis

The fixture contains labels and delays for experiment bookkeeping. Hidden
collection sends only cue, delay, and shared decision numeric vectors into the
policy. Dataset construction completes and freezes all hidden arrays before
any analyzer receives labels.

The diagnostic module may call existing deterministic fixture and hidden
collection helpers from reward_learning.py, but it may not alter them. It
constructs fresh policy and readout objects for every independent run.

## Fixed configuration and RNG lineages

LearningDiagnosticsConfig uses these defaults:

    hidden_size = 64
    recurrent_radius = 0.9
    learning_rate = 0.05
    temperature = 1.0
    training_episodes = 2_000
    evaluation_blocks = 20
    checkpoint_interval = 100

Acceptance seeds remain 7, 17, and 29.

Training fixtures use:

    SeedSequence([seed, 0x54524149])

Evaluation fixtures use:

    SeedSequence([seed, 0x4556414C])

The reward learner action generator uses:

    SeedSequence([seed, 0x4143544E])

The supervised path is deterministic and does not sample actions, so it owns
no action generator. It still processes the exact training tuple in the exact
same order as the reward path.

Every complete public run reconstructs all fixtures, policies, analyzers,
readouts, and generators. No trained object or advanced generator is reused
for repeatability checks.

## Immutable hidden datasets

A private frozen HiddenDataset contains:

- states: C-contiguous read-only float64 matrix;
- labels: C-contiguous read-only int64 vector;
- delays: C-contiguous read-only int64 vector;
- fixture_digest: SHA-256 over literal stimulus shapes and bytes;
- state_digest: SHA-256 over state shape and bytes.

Training contains exactly 2,000 rows in 200 balanced ten-case blocks.
Evaluation contains exactly 200 rows in 20 balanced ten-case blocks. Every
evaluation delay has 40 rows and every action label has 100 rows.

Collection performs this sequence for each episode:

    policy.reset_state()
    policy.advance(cue_stimulus)
    policy.advance(each delay stimulus in order)
    hidden = policy.advance(shared decision_stimulus)

Only after the returned hidden is copied into the dataset may the diagnostic
layer associate it with the fixture label and delay.

A boundary spy must prove that policy calls receive numeric arrays only and
that RewardModulatedReadout calls receive only hidden arrays, legal indices,
the seeded action generator, and later scalar rewards.

## Diagnostic A: Ridge geometry

Phase 2C reuses fit_linear_probe from memory_probe.py rather than creating a
new fitting algorithm. The probe trains on the 2,000-state training dataset
and predicts the 200-state evaluation dataset.

For label y in zero or one, define signed class value s as -1 for zero and +1
for one. For fitted weights w, bias b, and hidden vector h, normalized signed
margin is:

    margin = s * (dot(w, h) + b) / norm(w, 2)

A zero weight norm is a protocol error because a normalized margin would be
undefined.

For each seed and each delay, record:

- integer correct and total counts;
- minimum normalized signed margin;
- 10th percentile using NumPy linear percentile semantics;
- median normalized signed margin.

Also record the Delay-5-to-Delay-1 median-margin ratio. If the Delay 1 median
is zero, the ratio is a protocol error rather than infinity or an omitted
field.

The geometry gate remains the frozen Phase 2A gate:

- 200/200 overall;
- 40/40 at every delay;
- every normalized signed margin strictly positive.

The magnitude and delay ratio are reported measurements, not post-hoc pass
thresholds. Phase 2C will not invent a fragility threshold after observing
them.

## Diagnostic B: online supervised softmax

DiagnosticSupervisedReadout is private to learning_diagnostics.py and cannot
be exported as a production controller. It uses:

- two actions;
- zero-initialized float64 weights and biases;
- learning rate 0.05;
- temperature 1.0;
- the same stable legal-action softmax calculation as Phase 2B;
- greedy evaluation with lowest-index tie behavior.

For hidden vector h, probability vector p, and diagnostic label y, one online
update is:

    weights += learning_rate * outer(one_hot(y) - p, h)
    biases += learning_rate * (one_hot(y) - p)

The label enters this diagnostic method directly. The method name and type
must make that supervision explicit; it must not imitate the scalar-reward
learn interface.

Every 100 training episodes, the diagnostic performs a read-only greedy
evaluation over the same immutable 200-case evaluation dataset. It records
checkpoints 100, 200, ..., 2,000, final integer counts, per-delay counts, and a
parameter digest.

The pre-registered supervised behavioral gate is unchanged from Phase 2B:

- at least 180/200 overall;
- at least 34/40 at every delay.

Missing the gate is a valid diagnostic result, not permission to tune. It
selects ONLINE_OPTIMIZATION_FAILURE when Ridge geometry passed.

## Diagnostic C: unchanged reward trajectory

The reward trajectory reconstructs the exact Phase 2B training path using the
public RewardModulatedReadout behavior without modifying it.

For every training episode it performs:

    hidden = frozen policy decision state
    decision = reward_readout.select_for_training(hidden, (0, 1), action_rng)
    reward = delayed_cue_task.reward(episode, decision.action_index)
    record diagnostic quantities outside the learner
    reward_readout.learn(reward)

Every 100 episodes it performs a greedy, mutation-free evaluation on the
fixed evaluation dataset.

For each checkpoint it records:

- overall and per-delay integer accuracy;
- mean correct-action probability;
- 10th percentile correct-action probability;
- mean expected-bandit-to-supervised gradient scale;
- cosine between the accumulated sampled reward gradient and accumulated
  supervised gradient over each completed ten-case block.

Diagnostic labels are used only to calculate these measurements after the
learner has selected its action. They never affect selection or learning.

For a two-action policy, the exact expected bandit gradient has the same
direction as the supervised gradient but a probability-dependent scale. The
diagnostic records the exact norm ratio rather than assuming it is benign.
This reveals whether low correct-action probability suppresses recovery on
long-delay cases.

When either accumulated gradient has zero norm, block cosine is represented
as null in JSON and counted explicitly. It is never silently replaced with
zero or one.

The final reward result must match the corresponding seed entry in
phase-2b-failure.json for all counts and all Phase 2B digests. A mismatch is
PROTOCOL_MISMATCH and invalidates all downstream interpretation.

## Diagnosis classification

Each seed receives exactly one of these values, checked in priority order:

| Priority | Condition | Classification |
|---:|---|---|
| 1 | Phase 2B final reproduction differs from frozen evidence | PROTOCOL_MISMATCH |
| 2 | Ridge geometry misses its frozen gate | REPRESENTATION_FAILURE |
| 3 | Ridge passes and supervised online readout misses its gate | ONLINE_OPTIMIZATION_FAILURE |
| 4 | Ridge and supervised pass while reward learner misses | REWARD_CREDIT_FAILURE |
| 5 | Ridge, supervised, and reward learner all pass | NO_FAILURE_REPRODUCED |

The known Phase 2B evidence is not used to predeclare which non-mismatch
classification will occur. The diagnostic implementation must calculate it.

Phase 2C validity and behavioral success are separate:

- diagnostic_valid means evidence reproduction, frozen artifacts, boundary
  isolation, determinism, and schema checks all passed;
- the classification describes the measured learning outcome;
- diagnostic_valid does not mean the reward learner passed.

The benchmark exits zero when every seed has a valid, unique diagnosis and
all integrity gates pass. It may exit zero while reporting
REWARD_CREDIT_FAILURE because that is a successful diagnosis of a preserved
scientific failure.

## Public result surface

The package exports only:

- LearningDiagnosticsConfig;
- LearningDiagnosticsResult;
- run_learning_diagnostics;
- run_learning_diagnostics_benchmark.

DiagnosticSupervisedReadout and HiddenDataset remain private.

LearningDiagnosticsResult is a frozen dataclass containing literal counts,
margin summaries, checkpoint tuples, parameter and fixture digests, frozen
matrix digests, the Phase 2B evidence-match flag, repeatability, diagnostic
validity, and the classification string.

The benchmark returns one compact JSON-compatible dictionary with:

- phase equal to 2C;
- fixed configuration and ordered seeds;
- one complete result per seed;
- the frozen Phase 2B evidence digest;
- aggregate classification counts;
- all_valid.

scripts/benchmark_learning_diagnostics.py prints exactly one sorted compact
JSON line and exits with status zero only when all_valid is true.

## Known Phase 2B failure in tests and CI

The current test suite contains success assertions that necessarily fail for
seeds 7 and 29. Phase 2C does not delete the gate or claim it passed. Instead,
tests/test_reward_learning.py will become an exact known-failure regression:

1. run the unchanged Phase 2B benchmark;
2. require all_passed to be false;
3. load docs/experiments/phase-2b-failure.json;
4. require the complete runtime payload to equal the committed evidence;
5. require the CLI to exit with status one and print the same payload.

A dedicated scripts/verify_reward_learning_failure.py command performs the
same comparison and exits zero only when the known failure reproduces exactly.
If Phase 2B unexpectedly passes, drifts, or fails differently, verification
fails.

This makes CI green for a reproducible scientific result. Green means software
and evidence integrity, not Phase 2B behavioral acceptance.

The existing CI matrix retains unit tests, Ruff, the Phase 1 benchmark, and the
Phase 2A probe. It adds:

    python scripts/verify_reward_learning_failure.py
    python scripts/benchmark_learning_diagnostics.py

All steps run on Python 3.10, 3.11, and 3.12 against the exact pushed commit.

## Validation and error behavior

Configuration rejects booleans, non-integer sizes or counts, non-positive
sizes or rates, invalid radius, training counts not divisible by ten,
checkpoint intervals not divisible by ten, and checkpoint intervals that do
not divide the training count.

Dataset construction rejects non-finite, mismatched-rank, mismatched-length,
writable-output, empty, or incorrectly typed arrays.

The supervised analyzer rejects a wrong-sized or non-finite hidden vector,
an invalid label, a second label update without a new observation if the
implementation uses a pending record, and mutation attempts on snapshots.

Margin computation rejects mismatched states and labels, an empty class,
non-finite probe output, and zero probe norm.

Evidence verification rejects a missing file, malformed JSON, an unexpected
phase, unexpected seeds, all_passed=true, duplicate seeds, or any semantic
difference from the reproduced runtime payload.

A protocol error produces no fallback classification and causes diagnostic
validity and the benchmark exit status to fail.

## TDD and verification

Every production behavior begins with a focused failing test. Tests cover:

- fixed configuration and strict validation;
- exact fixture balance and RNG lineages;
- hidden collection before label association;
- array immutability and independent ownership;
- boundary spies for policy and reward readout;
- supervised softmax and update numerics reconstructed independently;
- normalized margin values reconstructed independently;
- exact percentile semantics;
- checkpoints exactly at 100 through 2,000;
- no mutation during checkpoint evaluation;
- Phase 2B runtime payload equality with frozen evidence;
- matrix digests before and after every path;
- unique classification priority;
- independent complete-run equality;
- stable byte-identical JSON output and exit status;
- Python 3.10, 3.11, and 3.12 CI.

Final verification runs:

    python -m pip install -e '.[dev]'
    pytest -q
    ruff check .
    python scripts/benchmark.py
    python scripts/benchmark_memory_probe.py
    python scripts/verify_reward_learning_failure.py
    python scripts/benchmark_learning_diagnostics.py

The final audit searches the controller and reward learner for diagnostic
labels and metadata. No label-bearing Phase 2C symbol may appear in
policy.py, reward_readout.py, reward_learning.py, or controller.py.

## File map

| File | Responsibility | Change |
|---|---|---|
| src/neural_state_machine/learning_diagnostics.py | datasets, analyzers, checkpoints, classifications, results | create |
| tests/test_learning_diagnostics.py | numerical, boundary, determinism, and diagnostic tests | create |
| scripts/benchmark_learning_diagnostics.py | stable Phase 2C JSON and validity exit status | create |
| scripts/verify_reward_learning_failure.py | exact known-failure verification | create |
| docs/experiments/phase-2c-diagnostics.json | immutable measured Phase 2C evidence | create after execution |
| src/neural_state_machine/__init__.py | approved public diagnostic exports | modify |
| tests/test_reward_learning.py | exact Phase 2B failure regression | modify |
| .github/workflows/ci.yml | failure verification and Phase 2C benchmark | modify |
| README.md | Phase 2B failure and Phase 2C diagnostic interpretation | modify |
| docs/superpowers/specs/2026-09-14-phase-2c-learning-diagnostics-design.md | reviewed design | create |

## Acceptance gates

Phase 2C is complete only if:

1. Every seed receives exactly one classification.
2. Every reproduced Phase 2B payload equals the frozen evidence entry.
3. Ridge geometry remains 200/200 overall, 40/40 at each delay, with strictly
   positive normalized signed margins.
4. Online supervised results are reported against the pre-registered 180/200
   and 34/40 gates without tuning.
5. All reward and supervised checkpoints are recorded at exactly every 100
   episodes through 2,000.
6. No checkpoint evaluation changes learner parameters.
7. Input, recurrent, and legacy output matrices remain byte-identical.
8. Labels and metadata remain outside the policy and reward learner boundary.
9. Two independently reconstructed complete executions are exactly equal.
10. The known Phase 2B failure verifier exits zero only for exact evidence
    reproduction.
11. The Phase 2C benchmark is byte-stable and all_valid reflects integrity,
    not reward success.
12. The complete test suite, Ruff, Phase 1, Phase 2A, exact Phase 2B failure
    verification, and Phase 2C benchmark pass on Python 3.10, 3.11, and 3.12.

A missed integrity gate is a Phase 2C failure. A supervised behavioral miss is
a diagnostic outcome and selects ONLINE_OPTIMIZATION_FAILURE; it is not an
implementation failure.

## Interpretation boundary and next gate

Phase 2C can support one of four causal interpretations after protocol
mismatch has been excluded:

- recurrent representation is insufficient;
- representation exists but the fixed online optimizer cannot acquire it;
- supervised online acquisition works but scalar sampled reward credit does
  not reliably acquire it;
- the prior reward failure cannot be reproduced.

Even a valid REWARD_CREDIT_FAILURE diagnosis does not authorize a temporal
eligibility implementation. It permits a new design review comparing expected
policy gradients, baselines, eligibility traces, or recurrent plasticity.
Those alternatives receive their own pre-registered spec and tests.
