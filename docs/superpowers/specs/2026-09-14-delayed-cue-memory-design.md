# Delayed-Cue Neural Memory Experiment Design

## Decision

Phase 2 is split into two independent scientific questions:

- **Phase 2A — memory decodability:** determine whether the frozen recurrent
  hidden state contains enough information to recover a cue that is absent
  from the current stimulus.
- **Phase 2B — reward learning:** determine whether an online reward-modulated
  readout can learn to use that information.

Phase 2A is the current gate. It uses one deterministic Ridge linear probe over
hidden states from all delay lengths. The probe is a measurement instrument,
not the game controller. It neither calls nor trains the policy action readout.

Phase 2B remains a separate later experiment. Failure of the existing reward
learner does not invalidate a passing Phase 2A result, and a passing probe does
not allow Phase 2B to be described as successful.

No explicit cue memory, semantic behavior state, transition table, BPTT, RL
framework, or nonlinear probe is introduced.

## Why the design changed

The first Phase 2 design combined two claims:

1. recurrent activity retains the vanished cue;
2. a selected-action Hebbian reward update can learn the correct readout.

The implemented mechanics showed that these claims cannot be inferred from
one score. With the approved defaults, the reward learner was exactly
repeatable and the reset ablation returned exactly 50%, but all three required
seeds failed the accuracy gate:

| Seed | Recurrent post-training | Reset ablation | Delay 1 / 2 / 3 / 4 / 5 |
|---:|---:|---:|---|
| 7 | 0.61 | 0.50 | 1.00 / 0.50 / 0.50 / 0.50 / 0.55 |
| 17 | 0.84 | 0.50 | 1.00 / 0.85 / 0.65 / 0.90 / 0.80 |
| 29 | 0.80 | 0.50 | 1.00 / 1.00 / 1.00 / 0.50 / 0.50 |

These results show that transient state affects behavior, but they do not tell
us whether longer-delay failures come from lost neural information or from an
inadequate learning rule. Phase 2A isolates that distinction.

The thresholds, seeds, delayed-cue task, reservoir defaults, and evaluation
set are not weakened or replaced in response to the failure.

## Claim under test

After a left or right cue, recurrent activity must preserve linearly decodable
cue information through one to five cue-free delay steps. At the decision
frame:

- left-cue and right-cue stimulus vectors are element-for-element equal;
- one shared linear probe is used for every delay length;
- cue identity, correct action, delay length, and episode phase are absent from
  the probe features;
- the only feature supplied to the probe is the current recurrent hidden
  vector after processing the shared decision stimulus;
- resetting recurrent state immediately before the decision stimulus makes
  all ablated features identical and returns balanced accuracy to exactly 50%.

Passing supports the narrow statement that the vanished cue remains linearly
available in recurrent neural activity. It does not establish that the current
reward learner can acquire the behavior.

## Alternatives considered

### Selected: one Ridge linear probe

Ridge regression has a deterministic closed-form solution, needs no iterative
optimizer, and measures linear decodability without changing the recurrent
system. A single probe across all delay lengths prevents the harness from
smuggling delay identity into a bank of delay-specific classifiers.

### Rejected: nearest-class-centroid probe

A centroid difference is simpler and more interpretable, but can understate
linearly available information when irrelevant hidden dimensions have large
variance. Ridge regression controls those dimensions while remaining linear.

### Rejected: logistic or nonlinear probe

Logistic regression adds convergence and optimizer choices. Kernels, MLPs,
k-nearest neighbors, and other nonlinear probes could decode information that
is not directly available to a linear action readout. They would weaken the
interpretation boundary.

### Deferred to Phase 2B: change the reward update

A symmetric two-row reward update or another online rule may outperform the
current selected-row update. Changing it before measuring hidden-state
decodability would continue to conflate representation and learning.

## Existing foundation

The completed Phase 2 foundation remains valid:

- `RecurrentPolicy` owns generic seeded recurrent dynamics;
- `RecurrentController` preserves the Phase 1 nine-input/four-action API and
  numerical benchmark;
- `DelayedCueTask` creates validated immutable four-channel episodes;
- `memory_benchmark.py` contains the reward-learning mechanics and the failed
  Phase 2B baseline.

Phase 2A does not change `RecurrentPolicy`, `RecurrentController`,
`DelayedCueTask`, or the reward learning rule.

## Architecture

```text
DelayedCueTask
    |
    | cue / delay / shared decision vectors
    v
RecurrentPolicy.advance
    |
    | current hidden vector only
    v
Hidden-state dataset
    |
    | training labels remain in experiment harness
    v
FittedLinearProbe
    |
    | left/right prediction
    v
Recurrent evaluation + state-reset ablation
```

The policy action output matrix exists for Phase 1 compatibility but is never
used to generate a Phase 2A feature or prediction and is never modified. Its
public digest may be sampled before and after the experiment solely to prove
that it stayed unchanged. Phase 2A calls only `reset_state()` and `advance()`
on the state-evolution path.

## Delayed-cue stimulus contract

The existing four-element task contract is unchanged:

```text
index 0: left-cue channel
index 1: right-cue channel
index 2: bounded distractor channel
index 3: decision channel
```

```text
left cue       [1, 0, 0, 0]
right cue      [0, 1, 0, 0]
delay          [0, 0, d, 0], d in [-0.25, 0.25]
decision       [0, 0, 0, 1]
```

Delay length remains an integer from one through five. The shared decision
vector is exactly equal for every cue and delay. Task labels and metadata are
used only to construct balanced datasets and score predictions.

## Hidden-state collection

For each episode, Phase 2A executes exactly:

```python
policy.reset_state()
policy.advance(episode.cue_stimulus)
for stimulus in episode.delay_stimuli:
    policy.advance(stimulus)
hidden = policy.advance(episode.decision_stimulus)
```

The returned decision-time hidden vector is the complete probe feature. The
collector does not call `decide()` or `learn()`, inspect logits, or create
eligibility.

For the state-reset ablation, the collector performs the same cue and delay
steps, then calls `reset_state()` immediately before the final
`advance(decision_stimulus)`. Because every ablated episode then starts the
decision step from zero state with the same stimulus and frozen matrices, the
resulting hidden vectors must be exactly equal.

## Dataset protocol

Every block contains the Cartesian product of:

- cue: left and right;
- delay: one, two, three, four, and five.

The ten cases are shuffled by a seeded generator. Every episode receives new
seeded distractors.

`MemoryProbeConfig` defaults are:

```python
hidden_size = 64
recurrent_radius = 0.9
training_blocks = 200
evaluation_blocks = 20
regularization = 1e-6
```

This produces 2,000 balanced training samples and 200 balanced evaluation
samples. The reservoir and sample defaults match the failed reward experiment
where they overlap.

Each run constructs exactly one state source:

```python
RecurrentPolicy(
    input_size=4,
    action_count=2,
    hidden_size=config.hidden_size,
    seed=seed,
    recurrent_radius=config.recurrent_radius,
)
```

The policy's default learning rate is irrelevant because Phase 2A never calls
`learn()`.

RNG lineages are fixed and independent:

```python
training_rng = np.random.default_rng(
    np.random.SeedSequence([seed, 0x50524F42])
)
evaluation_rng = np.random.default_rng(
    np.random.SeedSequence([seed, 0x4556414C])
)
```

Training and evaluation fixtures never share distractor draws. Evaluation
fixtures are built once and reused for recurrent and reset modes.

## Linear probe

### Interfaces

`memory_probe.py` provides the frozen data type
`FittedLinearProbe(weights: np.ndarray, bias: float)` and these operations:

```text
FittedLinearProbe.predict(states: np.ndarray) -> np.ndarray
FittedLinearProbe.digest() -> str
fit_linear_probe(states: np.ndarray, labels: np.ndarray,
                 *, regularization: float) -> FittedLinearProbe
```

`states` has shape `(samples, hidden_size)`. Labels are literal action indices:
`0` for left and `1` for right. Fitting maps them internally to `-1.0` and
`+1.0`.

### Fit algorithm

The implementation augments the feature matrix with a constant bias column:

```python
design = np.column_stack((states, np.ones(states.shape[0])))
penalty = np.diag([regularization] * states.shape[1] + [0.0])
target = np.where(labels == 0, -1.0, 1.0)
parameters = np.linalg.solve(
    design.T @ design + penalty,
    design.T @ target,
)
weights = parameters[:-1]
bias = float(parameters[-1])
```

Only feature weights are regularized; the bias is not. Prediction selects
right for scores strictly greater than zero and left otherwise, giving the
lowest-index action deterministic tie behavior.

One fitted probe is trained on the combined delay 1–5 dataset and reused for
all reported metrics. Delay-specific probes and delay-derived features are
forbidden.

Weights are stored as an independent read-only `float64` array. The probe
digest is SHA-256 over the weight shape, C-contiguous weight bytes, and the
`float64` bias bytes.

Direct construction of `FittedLinearProbe` validates and defensively copies
its weights, so callers cannot introduce non-finite, writable, or aliased
parameters.

## Evaluation and controls

### Recurrent mode

The fitted probe predicts from decision-time hidden vectors produced by the
continuous cue-to-delay-to-decision trajectory.

### State-reset ablation

The same fitted probe predicts from decision-time hidden vectors after state
is cleared immediately before the shared decision input. All ablated feature
vectors must be exactly equal. The classifier therefore produces one constant
choice, yielding exactly 100 correct of 200 balanced samples.

### Frozen-policy control

Phase 2A must prove:

- the policy output-weight digest is identical before and after dataset
  collection, fitting, and both evaluation modes;
- private input and recurrent matrices are element-for-element unchanged in
  tests;
- no `decide()` or `learn()` call occurs;
- the probe receives no task metadata or label during prediction.

Training labels are visible only to `fit_linear_probe`. Evaluation labels are
visible only to the scorer after prediction.

## Result contract

`MemoryProbeResult` reports:

- seed and complete `MemoryProbeConfig`;
- training correct/total and accuracy;
- recurrent evaluation correct/total and accuracy;
- recurrent accuracy for every delay length 1–5;
- state-reset correct/total and accuracy;
- whether all ablated hidden vectors are exactly equal;
- output-weight digest before and after the experiment;
- fitted-probe, recurrent-choice, and reset-choice digests;
- exact repeatability of two independent runs with the same seed.

Accuracy is always derived from literal correct/total counts. Expected labels
come from episode truth, never from another prediction.

The public orchestration surface is:

```text
run_memory_probe(seed: int = 7,
                 config: MemoryProbeConfig | None = None) -> MemoryProbeResult
run_memory_probe_benchmark(seeds: Sequence[int] = (7, 17, 29),
                           config: MemoryProbeConfig | None = None)
    -> dict[str, object]
```

`run_memory_probe` executes two completely independent runs and sets the
result's repeatability field from full run-object equality. The benchmark CLI
prints one compact JSON object with sorted keys and exits nonzero when any
fixed-seed acceptance condition fails.

## Phase 2A acceptance criteria

The committed default benchmark must satisfy all of the following:

1. Recurrent probe accuracy is at least `0.90` over 200 balanced evaluation
   samples.
2. Accuracy for every individual delay length 1–5 is at least `0.85`.
3. State-reset accuracy is exactly `0.50` (`100/200`).
4. All state-reset decision-time hidden vectors are exactly equal.
5. Left-cue and right-cue decision stimulus vectors are exactly equal.
6. One shared probe is used across all delays and receives no delay feature.
7. Two independent runs with the same seed return identical result objects and
   digests.
8. Seeds `7`, `17`, and `29` all pass criteria 1–7.
9. Policy input, recurrent, and output weights remain unchanged.
10. Phase 1 tests and benchmark JSON remain numerically unchanged.
11. The existing reward-learning failure remains reported as a Phase 2B
    baseline and is not included in Phase 2A pass/fail.
12. `pytest -q`, `ruff check .`, the Phase 1 benchmark, and the Phase 2A probe
    benchmark pass on Python 3.10, 3.11, and 3.12.

If any fixed seed misses the accuracy thresholds, Phase 2A reports that the
current reservoir does not provide robust linearly decodable 1–5-step memory.
Thresholds, fixed seeds, task difficulty, training/evaluation block counts,
regularization, hidden size, and recurrent radius must not be changed merely
to make CI green. Another design revision is required.

## Error handling

- `MemoryProbeConfig` rejects non-integer or non-positive sizes/block counts,
  recurrent radii outside `[0, 1)`, and non-positive/non-finite
  regularization.
- Probe fitting rejects non-finite or non-rank-two states, zero samples or
  features, non-rank-one labels, mismatched sample counts, labels outside
  `{0, 1}`, and a dataset missing either class.
- Prediction rejects non-finite or non-rank-two states and feature widths that
  differ from the fitted probe.
- A numerical solve failure raises `RuntimeError` with the original
  `numpy.linalg.LinAlgError` as its cause.
- Returned probe weights and prediction arrays are independent read-only
  copies.
- Probe orchestration rejects seeds that are not non-negative Python integers
  before constructing either RNG lineage or the policy.

## Files and dependency limits

Expected new files:

```text
src/neural_state_machine/memory_probe.py
tests/test_memory_probe.py
scripts/benchmark_memory_probe.py
```

Expected modifications are limited to public exports, README, and CI. Existing
policy, controller, delayed-cue task, and reward-baseline modules remain
unchanged unless a separately demonstrated defect requires a reviewed fix.

NumPy remains the only runtime dependency.

## Interpretation boundary

Results are interpreted as follows:

| Phase 2A probe | Phase 2B reward learner | Supported conclusion |
|---|---|---|
| pass | fail | Neural state contains usable memory; current reward rule cannot reliably learn its readout. |
| fail | fail | The current reservoir does not robustly preserve linearly decodable 1–5-step memory. |
| pass | pass | Both memory representation and the separately tested reward-learning mechanism succeed. |

A Phase 2A pass does not demonstrate semantic attractors, online learning,
open-world game intelligence, biological plausibility, or replacement of all
state machines.

## Plan supersession and next gates

Tasks 1–6 of
`docs/superpowers/plans/2026-09-14-delayed-cue-memory.md` remain the implemented
and reviewed foundation. Its original Tasks 7–8 are superseded because their
combined reward-learning acceptance was falsified before implementation.

After this revised spec is approved, a new implementation plan must cover only
the Phase 2A probe, its benchmark, documentation, CI, and final audit.

If Phase 2A passes, the next design session may address Phase 2B with a revised
reward-learning rule. Multiple concurrent memories and conflicting
attack/retreat/heal objectives remain Phase 3. A `fly-brain` adapter remains
later still and must face the same Phase 2A/2B separation before biological
topology is credited with improvement.
