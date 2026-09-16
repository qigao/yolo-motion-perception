# Phase 3A Normalized Action-Value Credit Design

## Decision

Phase 3A will replace neither the recurrent memory substrate nor the frozen
Phase 2B learner. It will add an independent action-value readout that learns
terminal scalar rewards from the decision-time recurrent hidden vector with a
normalized TD(0) update.

The selected action-value mechanism is:

```text
frozen recurrent hidden state
        |
        v
augmented feature x = [hidden; 1]
        |
        v
seeded uniform behavior action
        |
        v
immediate terminal scalar reward
        |
        v
normalized action-local TD(0) update
```

The training behavior policy is independent of the cue, correct action,
delay, reward, and current action values. Greedy action-value selection is
used only for evaluation. This isolates whether scalar reward can acquire the
linearly available hidden-state information without the probability-dependent
sample weighting that made the Phase 2B policy-gradient result seed-sensitive.

Phase 3A also introduces a small Lean specification of the normalized update
kernel. Lean proves algebraic properties of the mathematical model. Python
tests and deterministic evidence establish behavior of the NumPy
implementation. Neither evidence class is described as proving the other.

Phase 3B is deliberately not implemented by this phase. Phase 3A preserves a
credit-record boundary that a later, separately reviewed design can extend to
true action-to-reward delays and TD(lambda) eligibility decay.

## Correction to the experimental vocabulary

The Phase 2 task delays the cue relative to the decision. It does not delay
reward relative to the selected action:

```text
cue -> one to five cue-free frames -> shared decision frame -> action -> reward
```

`DelayedCueTask.reward()` returns immediately after the action, and
`RewardModulatedReadout.learn()` immediately consumes the one pending
eligibility stored by `select_for_training()`.

Consequently:

- Phase 2A establishes vanished-cue information in the decision-time hidden
  vector;
- Phase 2B tests a sampled policy-gradient credit estimator with immediate
  terminal reward;
- Phase 2C localizes the failed seeds to reward-credit acquisition rather than
  representation or ordinary supervised online acquisition;
- Phase 3A tests a different immediate-reward estimator;
- only a later Phase 3B may claim to test action-to-reward temporal credit.

No Phase 3A document, API, benchmark, or report may call the Phase 2 delay a
reward delay.

## Evidence motivating the change

The fixed Phase 2 configuration uses hidden size 64, recurrent radius 0.9,
learning rate 0.05, temperature 1.0, 2,000 training episodes, 200 evaluation
episodes, delays one through five, and seeds 7, 17, and 29.

The frozen Phase 2C evidence records:

| Seed | Ridge | Online supervised | Sampled reward | Diagnosis |
|---:|---:|---:|---:|---|
| 7 | 200/200 | 197/200 | 169/200 | `REWARD_CREDIT_FAILURE` |
| 17 | 200/200 | 200/200 | 200/200 | `NO_FAILURE_REPRODUCED` |
| 29 | 200/200 | 199/200 | 171/200 | `REWARD_CREDIT_FAILURE` |

Every Ridge delay is 40/40. The normalized signed-margin medians do not decay
materially from delay one to delay five. The ordinary supervised online
softmax optimizer also passes the frozen behavioral gate for every seed.

Phase 2C records the cosine between each ten-case block's accumulated sampled
reward gradient and accumulated supervised gradient. Negative block cosines
occur 76 times for seed 7, 25 times for seed 17, and 74 times for seed 29 out
of 200 blocks. Individual binary reward updates have a useful direction, but
their probability-dependent magnitudes weight different hidden vectors
unevenly. The accumulated stochastic direction can therefore oppose the
corresponding supervised direction even though the representation remains
linearly usable.

Phase 3A changes the credit estimator rather than the reservoir, task vectors,
seed set, sample counts, or behavioral thresholds.

## Claim under test

For the frozen two-action delayed-cue task, an action-value readout receiving
only:

1. the current recurrent hidden vector;
2. one action selected by an independent exploration policy; and
3. the resulting scalar terminal reward;

can learn values whose greedy ordering maps the vanished cue to the correct
action for every pre-registered seed and delay gate.

Passing supports the narrow claim that immediate scalar reward can train an
action-value readout over this recurrent memory representation. It does not
establish delayed-reward learning, recurrent plasticity, semantic attractors,
multiple simultaneous memories, complex game behavior, biological
plausibility, or replacement of general finite-state machines.

## Scope

### In scope

- A new action-value module independent of `RewardModulatedReadout`.
- Linear action values over the decision-time hidden vector plus a bias
  feature.
- A normalized terminal TD(0) update.
- A separately seeded uniform training behavior policy.
- Greedy, mutation-free evaluation.
- The exact Phase 2 training and evaluation fixtures, recurrent policies,
  seeds, counts, delays, and behavioral gates.
- Reset-state and shuffled-reward controls.
- Frozen Phase 1, Phase 2A, Phase 2B, and Phase 2C compatibility checks.
- Deterministic evidence and portable semantic verification.
- A pinned Lean project proving the normalized update kernel's algebraic
  obligations.
- Explicit separation of theorem, implementation, and behavioral evidence.

### Out of scope

- Modifying, deleting, or relabeling the Phase 2B failure evidence.
- Modifying the Phase 2C diagnostic evidence or classifications.
- Tuning on seeds 7, 17, or 29 after observing Phase 3A outcomes.
- Changing the reservoir, hidden size, recurrent radius, task stimuli, delay
  distribution, training count, evaluation count, or acceptance threshold.
- Passing cue labels, correct actions, delays, phases, probe predictions, or
  episode objects into the action-value learner.
- Converting binary reward into a reconstructed supervised target.
- Querying counterfactual rewards for actions that were not selected.
- Actor-critic, BPTT, recurrent plasticity, STDP, temporal eligibility decay,
  multiple decisions per episode, or a reinforcement-learning framework.
- YOLO, FlyVis, fly-brain, ROS2, a game engine, or a real perception stream.
- A claim that Lean verifies NumPy floating-point execution.
- A claim that an algebraic contraction theorem proves stochastic convergence
  or behavioral acceptance.

## Frozen boundaries

The following artifacts remain byte-for-byte unchanged during Phase 3A:

- `src/neural_state_machine/policy.py`
- `src/neural_state_machine/reward_readout.py`
- `src/neural_state_machine/reward_learning.py`
- `src/neural_state_machine/learning_diagnostics.py`
- `src/neural_state_machine/memory_task.py`
- `src/neural_state_machine/memory_probe.py`
- `src/neural_state_machine/memory_benchmark.py`
- `docs/experiments/phase-2b-failure.json`
- `docs/experiments/phase-2c-diagnostics.json`
- all Phase 1, Phase 2A, Phase 2B, and Phase 2C test expectations

Phase 3A may call existing deterministic fixture and hidden-state helpers. It
may not change their behavior. Fresh policies, learners, fixtures, and random
generators are constructed for every independent run.

## Action-value model

### Feature vector

For hidden size `n`, the learner validates a finite float64 vector
`h` of shape `(n,)` and constructs:

```text
x = concatenate(h, [1.0])
s = dot(x, x)
```

The final component is a bias feature. It also guarantees `s >= 1`, so the
normalizing denominator is finite and strictly positive whenever the validated
hidden vector is finite.

The learner owns a float64 matrix `W` with shape
`(action_count, hidden_size + 1)`, initialized to exact zero. Action value is:

```text
Q(h, a) = dot(W[a], x)
```

The public decision snapshot contains independent, read-only action values and
the selected action index. Internal features stored for pending feedback are
independent copies and cannot be mutated through caller-owned arrays.

### Training action selection

`select_for_training(hidden, legal_action_indices, rng)` samples uniformly
from the validated legal-action tuple with an independent NumPy generator.
It does not inspect action values when selecting the training action.

This is an explicit contextual-bandit behavior policy, not the learned target
policy. Its purpose is to guarantee continued exploration without leaking task
labels or allowing early random value estimates to suppress observations of an
action.

The action RNG uses a new documented `SeedSequence` lineage distinct from
fixtures, Phase 2B action sampling, Phase 2C diagnostics, evaluation, and
reward shuffling. Normal and shuffled-control paths reconstruct separate
generators from this same lineage, so they produce exactly the same action
sequence without sharing mutable RNG state. The complete action sequence and
lineage are included in the evidence.

Only one feedback record may be pending. A second training selection before
feedback is rejected. The pending record contains the selected action,
augmented feature, denominator, and decision-time value.

### Greedy evaluation

`select_greedy(hidden, legal_action_indices)` returns the legal action with the
largest value. The lowest numeric action index wins an exact tie. Evaluation
creates no pending feedback and changes no parameter, random generator, or
policy state beyond the separately reconstructed recurrent trajectory.

### Normalized terminal TD(0)

Let selected action be `a`, augmented feature be `x`, denominator be
`s = dot(x, x)`, prediction be `q = dot(W[a], x)`, terminal scalar reward be
`r`, and fixed step size be `alpha = 0.1`.

The pending feedback update is:

```text
delta = r - q
W[a] <- W[a] + alpha * delta * x / s
```

All unselected rows remain unchanged. Reward must be finite and is not clipped
or converted to a class label. The delayed-cue task already produces literal
`-1.0` or `1.0`; preserving that value makes the equation and evidence
unambiguous.

The pending feedback record is cleared after exactly one successful update.
Invalid reward leaves parameters and pending credit unchanged, permitting the
caller to supply valid feedback rather than silently losing the decision.

The step size is pre-registered as 0.1 before the benchmark is run. Phase 3A
does not search alternative step sizes. A failed behavioral gate remains a
result and triggers a new design review rather than tuning this specification.

## Mathematical obligations

For the selected action at one fixed feature vector, define:

```text
q       = dot(W[a], x)
delta   = r - q
W'[a]   = W[a] + alpha * delta * x / dot(x, x)
```

Then:

```text
dot(W'[a], x)
  = q + alpha * (r - q)
```

and therefore:

```text
r - dot(W'[a], x)
  = (1 - alpha) * (r - q)
```

For `0 < alpha <= 1`, the absolute prediction error on that same
state-action sample cannot increase. For `0 < alpha < 1` and nonzero error, it
strictly decreases.

These are single-update algebraic statements. They do not imply that an update
on one hidden vector cannot interfere with another non-orthogonal hidden
vector, that stochastic iteration converges, or that the Phase 3A benchmark
passes.

## Lean proof surface

Phase 3A adds a minimal pinned Lean 4 project under `formal/`. Mathlib is pinned
by a committed `lake-manifest.json`; CI must not resolve an unpinned moving
dependency.

The proof model uses a finite action type, a finite feature type, and exact
real-number inner-product algebra. It does not model IEEE-754 operations,
NumPy arrays, pseudorandom generation, or the recurrent policy.

Required theorem obligations are:

| Theorem | Statement |
|---|---|
| `augmentedFeature_normSq_pos` | A feature containing the constant-one component has positive squared norm. |
| `selectedPrediction_after_update` | The updated selected prediction equals `q + alpha * (r - q)`. |
| `selectedError_after_update` | The updated selected error equals `(1 - alpha) * (r - q)`. |
| `selectedError_abs_le` | For `0 < alpha <= 1`, selected absolute error does not increase. |
| `selectedError_abs_lt` | For `0 < alpha < 1` and nonzero error, selected absolute error strictly decreases. |
| `unselectedAction_unchanged` | Every unselected action row is unchanged. |
| `zeroError_update_identity` | Zero TD error leaves the complete value table unchanged. |

Lifecycle properties such as pending-feedback uniqueness, failed-validation
atomicity, and reset behavior are executable state-machine contracts verified
by Python tests. They are not mislabeled as algebraic Lean theorems unless a
later design introduces a formal lifecycle transition system.

The Lean source contains no `sorry`, `admit`, declaration-level new axiom, or
native-evaluation shortcut. CI reports `#print axioms` for the public theorems.
Ordinary Mathlib foundations such as classical choice or quotient soundness,
if present, are recorded rather than hidden.

## Python-to-theorem correspondence boundary

Python independently reconstructs each update from immutable before-state
snapshots and compares it with the production result. Focused tests cover
exact zero/identity cases and tolerance-based nontrivial float cases.

The correspondence claim is limited to:

> The Python kernel implements the same displayed normalized TD(0) equation
> within the stated float comparison contract, and Lean proves the displayed
> real-number equation under its explicit assumptions.

It must not be shortened to "Lean verified the Python learner." Python
validation, array ownership, RNG behavior, evidence serialization, and
floating-point portability remain test and CI responsibilities.

## Phase 3A experiment protocol

### Fixed configuration

Phase 3A fixes:

```text
hidden_size       = 64
recurrent_radius  = 0.9
step_size         = 0.1
training_episodes = 2_000
evaluation_blocks = 20
delays             = [1, 2, 3, 4, 5]
seeds              = [7, 17, 29]
```

The fixture RNG lineages remain identical to Phase 2B and Phase 2C. Phase 3A
adds only its own behavior-action generator lineage. The normal and shuffled
paths own separately reconstructed generators with that same lineage. Their
action sequences must be exactly equal, while no generator object or advanced
generator state is shared between paths.

### Training path

For each immutable training episode:

```text
reset frozen recurrent state
advance cue stimulus
advance every cue-free delay stimulus
advance shared decision stimulus -> hidden
uniformly select one legal action
obtain immediate scalar terminal reward
apply one normalized TD(0) update
```

Correct action and delay are available to experiment bookkeeping only after
the learner has selected and learned from scalar reward. They are used for
scoring and stratification, never for selection or update.

### Evaluation path

Pre-training and post-training evaluation reconstruct the same fixed 200-case
evaluation dataset. Each case resets and replays the frozen recurrent policy,
then performs a greedy action-value selection. Evaluation does not learn.

The reset ablation resets recurrent state immediately before the common
decision stimulus. It must produce identical hidden vectors and exactly
100/200 on the balanced fixture.

### Shuffled-reward control

The shuffled learner receives the same hidden vectors and the exact same
uniform behavior-action sequence as the normal learner. The experiment first
materializes the normal scalar rewards produced by those state-action pairs,
then independently permutes those literal rewards within every fixed ten-case
training block using a separate RNG lineage. Every block therefore preserves
its actual reward multiset while breaking the state-action-reward association.

The learner still sees only scalar reward. The shuffled control tests whether
the readout can pass merely from fixture order, the identical action schedule,
hidden geometry, reward marginal distribution, or update drift without the
correct state-action-reward relation.

## Behavioral and integrity gates

For each seed, Phase 3A requires:

- post-training greedy accuracy of at least 180/200;
- at least 34/40 for every delay one through five;
- reset accuracy exactly 100/200;
- every reset delay exactly 20/40;
- all reset hidden vectors exactly equal;
- shuffled-reward greedy accuracy below 150/200;
- normal value parameters changed from exact-zero initialization;
- recurrent input, recurrent, and legacy output matrices unchanged;
- no pending feedback after training;
- exact equality of two independently reconstructed complete executions in
  the same environment.

The pooled shuffled accuracy across all three seeds must remain between 0.40
and 0.60 inclusive.

The benchmark-level gate passes only when every seed passes and the pooled
control passes. Seed 17 cannot compensate for seeds 7 or 29.

Phase 3A success means the new action-value mechanism passes this fixed gate.
It does not rewrite Phase 2B from failure to success. The report presents both
results side by side.

## Diagnostics and evidence

The benchmark records, for every seed:

- pre-training, post-training, reset, and shuffled counts;
- post-training, reset, and shuffled counts per delay;
- checkpoints every 100 episodes through 2,000;
- mean, 10th-percentile, and minimum correct-action value margin;
- TD-error mean, absolute mean, 90th percentile, and maximum per checkpoint;
- selected-action counts, action-sequence digests, and exact equality of the
  normal and shuffled action sequences;
- normal and shuffled reward-sequence digests;
- equality of normal and shuffled reward multisets within every training
  block;
- initial and final value-parameter digests;
- recurrent matrix digests before and after every path;
- fixture and decision-hidden digests;
- pending-feedback state;
- repeatability and per-seed pass flags.

The immutable measured payload is written only after the implementation and
tests exist and the benchmark has been run without tuning:

```text
docs/experiments/phase-3a-action-value.json
```

The JSON carries its source commit, schema version, fixed configuration, and
the SHA-256 digests of both frozen Phase 2 evidence files. A verifier compares
portable semantic fields across environments and requires full byte identity
between independent runs within the same environment. Raw floating parameter
and matrix digests remain environment-local controls because the reservoir
construction depends on NumPy/LAPACK eigenvalue calculations.

Green CI means theorem checking, software integrity, evidence reproduction,
and the explicitly reported behavioral gate all passed at that exact commit.
It does not establish untested scientific claims.

## Validation and error behavior

The action-value object rejects:

- boolean or non-positive sizes;
- action counts below two;
- a non-finite step size or a step size outside `(0, 1]`;
- hidden vectors of the wrong rank, shape, or dtype compatibility;
- non-finite hidden values;
- empty, duplicate, non-integer, or out-of-range legal actions;
- a non-Generator RNG;
- a second training selection while feedback is pending;
- feedback without a pending selection;
- non-finite scalar reward.

Validation failure is atomic: no parameter, pending record, or RNG is changed
unless that operation's documented validation has completed. Caller-owned
arrays and returned snapshots cannot mutate internal state.

Benchmark configuration rejects invalid seeds, duplicate seeds, counts that
violate block balance, invalid checkpoint divisibility, and any attempt to
change a frozen Phase 2 fixture lineage through the Phase 3A public surface.

An integrity or protocol failure produces no fallback success result. The CLI
exits nonzero and does not replace committed evidence.

## TDD and verification

Implementation follows RED then GREEN in focused commits. Required coverage
includes:

- construction and strict validation;
- exact-zero initialization and immutable snapshots;
- legal-action greedy selection and tie behavior;
- uniform training selection independent of action values;
- one-pending-feedback lifecycle;
- normalized update reconstructed independently;
- selected prediction and error identities;
- unselected-row isolation;
- zero-error identity;
- invalid-feedback atomicity;
- no cue label or metadata across the learner boundary;
- exact RNG lineages and independent generator ownership;
- fixed fixture reuse and frozen matrix controls;
- reset and shuffled-reward ablations;
- per-delay and per-seed gates;
- checkpoint non-mutation;
- stable JSON schema, portable projection, and same-environment bytes;
- Python 3.10, 3.11, and 3.12;
- pinned Lean build, theorem declarations, forbidden-token scan, and axiom
  audit.

Final Phase 3A verification will include:

```text
python -m pip install -e '.[dev]'
pytest -q
ruff check .
python scripts/benchmark.py
python scripts/benchmark_memory_probe.py
python scripts/verify_reward_learning_failure.py
python scripts/benchmark_learning_diagnostics.py
python scripts/benchmark_action_value.py
python scripts/verify_action_value_evidence.py
lake build
```

The exact Lean invocation may include a project directory flag selected by the
implementation plan, but it must build from the committed pinned project and
must not alter the Python package dependency graph.

## Planned file map

No file in this table is authorized for implementation until this design is
reviewed and a separate implementation plan is committed.

| File | Responsibility | Planned change |
|---|---|---|
| `src/neural_state_machine/action_value.py` | normalized action-value object and immutable decision snapshots | create |
| `src/neural_state_machine/action_value_benchmark.py` | fixed Phase 3A protocol, controls, results, gates | create |
| `tests/test_action_value.py` | unit equations, validation, lifecycle, ownership | create |
| `tests/test_action_value_benchmark.py` | protocol, boundary, controls, determinism, gates | create |
| `scripts/benchmark_action_value.py` | one stable JSON output and behavioral exit status | create |
| `scripts/verify_action_value_evidence.py` | portable evidence plus environment-local integrity verification | create |
| `docs/experiments/phase-3a-action-value.json` | immutable measured evidence | create after execution |
| `formal/lakefile.toml` | minimal proof project | create |
| `formal/lean-toolchain` | exact Lean toolchain pin | create |
| `formal/lake-manifest.json` | exact Mathlib dependency pin | create |
| `formal/NeuralStateMachine/ActionValue.lean` | normalized-update definitions and theorems | create |
| `formal/NeuralStateMachine/AxiomAudit.lean` | public theorem axiom reporting | create |
| `src/neural_state_machine/__init__.py` | approved Phase 3A public exports | modify |
| `.github/workflows/ci.yml` | Phase 3A evidence and independent Lean proof jobs | modify |
| `README.md` | measured Phase 3A result and claim boundary | modify after evidence |

## Phase 3B compatibility boundary

Phase 3A stores its decision-time feature and action in an explicit pending
credit record. A later design may generalize that record to an eligibility
state:

```text
e_0 = x / dot(x, x)
e_(k+1) = gamma * lambda * e_k
```

If terminal reward arrives after `d` reward-free transitions and no
intermediate parameter update changes the stored prediction, a future update
may take the form:

```text
W[a] <- W[a] + alpha * (r - q_decision) * e_d
```

The corresponding candidate theorem obligations are:

```text
e_d = (gamma * lambda)^d * e_0
```

and, at the original decision feature:

```text
new_error = (1 - alpha * (gamma * lambda)^d) * old_error
```

These equations are design compatibility notes, not Phase 3A claims or
implemented behavior. Phase 3B must separately define reward timing,
intermediate observations, termination, trace reset, gamma, lambda, evidence,
and acceptance gates after Phase 3A evidence is frozen.

## Alternatives rejected

### Actor-critic baseline

An action-independent critic can reduce policy-gradient variance without
biasing the expected gradient. It also introduces a second learned function,
a second error surface, and additional coupled hyperparameters. Phase 3A first
tests the simpler action-value estimator that extends directly to TD traces.

### Binary reward-to-target reconstruction

With exactly two actions and deterministic rewards of plus or minus one, the
selected action and reward can reveal the correct action. Converting that fact
into a target would likely reproduce supervised learning, but it hides the
binary task structure inside the learner and does not extend honestly to a
general action set. Phase 3A forbids this conversion.

### Counterfactual reward queries

Evaluating both legal actions at the same hidden state would provide
full-information value targets, but it is not the single-action scalar-feedback
boundary under test.

### Temporal eligibility in Phase 3A

The present reward is immediate. Adding decay steps before establishing an
immediate-reward learner would confound estimator reliability with temporal
credit persistence. Eligibility decay belongs to Phase 3B.

### Reservoir or task changes

Phase 2A and Phase 2C already show that the unchanged hidden representation is
linearly usable. Modifying the recurrent system, features supplied to it, or
task difficulty would discard the causal diagnosis instead of testing it.

## Acceptance

Phase 3A implementation may begin only after:

1. this design is committed to `experiment/neural-state-machine`;
2. the user reviews and approves the committed specification;
3. a separate detailed implementation plan is written, self-reviewed,
   committed, and approved for execution.

Phase 3A is complete only when the fixed behavioral gate, integrity checks,
Lean proof job, existing frozen-phase checks, and Python-version CI matrix all
pass on the exact pushed commit. A behavioral miss is committed as new failure
evidence and returned to design review; it is never converted into success by
changing thresholds or rewriting Phase 2 history.
