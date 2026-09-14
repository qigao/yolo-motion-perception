# Phase 2B Online Reward Readout Learning Design

## Decision

Phase 2B will test whether an online reward-modulated action readout can learn
to use the vanished-cue information already demonstrated by Phase 2A.

The recurrent reservoir remains frozen. A new, independent
`RewardModulatedReadout` receives only the current numeric hidden vector,
selects an action, stores one decision's local eligibility, and later receives
one scalar reward. It updates its own output weights and biases with a
reward-modulated softmax policy-gradient rule.

The existing `RecurrentPolicy.learn()` implementation and
`memory_benchmark.py` results remain frozen as the Phase 2B V0 failure
baseline. Phase 2B does not rewrite that history to make the old experiment
pass.

No cue label, correct action, delay length, episode phase, observation history,
linear-probe prediction, BPTT gradient, or explicit behavior state crosses the
controller boundary.

## Why Phase 2B is the next gate

Phase 2A separated representation from learning and produced the following
fixed result for seeds 7, 17, and 29:

- a single frozen Ridge probe decoded 200/200 evaluation episodes;
- every delay from one through five decoded 40/40;
- resetting recurrent state immediately before the shared decision input
  produced exactly 100/200 and identical hidden vectors;
- the recurrent policy output weights remained byte-for-byte unchanged.

The frozen V0 selected-row Hebbian learner did not meet the same behavioral
gate:

| Seed | V0 post-training | Reset | Delay 1 / 2 / 3 / 4 / 5 |
|---:|---:|---:|---:|---|
| 7 | 0.61 | 0.50 | 1.00 / 0.50 / 0.50 / 0.50 / 0.55 |
| 17 | 0.84 | 0.50 | 1.00 / 0.85 / 0.65 / 0.90 / 0.80 |
| 29 | 0.80 | 0.50 | 1.00 / 1.00 / 1.00 / 0.50 / 0.50 |

The third row above is reported exactly as produced by the frozen baseline
module and its committed tests. Phase 2B changes the learning mechanism while
keeping the delayed-cue task, reservoir, seeds, sample counts, and acceptance
thresholds fixed.

## Claim under test

For a two-action delayed-cue task, a readout using only:

1. the current recurrent hidden vector;
2. the action it sampled;
3. the probabilities it assigned to legal actions; and
4. a later scalar reward;

can learn a policy that maps a vanished left/right cue to the correct action
after one to five cue-free delay frames.

Passing supports the narrow statement that an online reward signal can train a
linear action readout over the recurrent memory representation. It does not
establish semantic attractors, multiple simultaneous memories, general game
intelligence, a complete state-machine replacement, or biological
plausibility.

## Alternatives considered

### Selected: reward-modulated softmax eligibility

The selected rule is the score-function gradient for a categorical policy. It
uses a presynaptic hidden vector, a postsynaptic action/probability term, and a
global reward signal. Negative reward suppresses the sampled action and
redistributes probability toward the other legal actions.

Unlike a two-row special case, the rule has a defined extension to more than
two actions. Unlike supervised fitting, it never receives the correct action.
It remains a linear readout and therefore does not claim more representational
power than Phase 2A measured.

### Rejected: symmetric two-row Hebbian update

For exactly two actions, updating the selected row by `+reward * hidden` and
the unselected row by `-reward * hidden` is compact and likely effective.
However, negative reward does not identify one correct alternative when three
or more actions are legal. Baking the binary-task assumption into the learning
API would create another boundary that Phase 3 must replace.

### Rejected: supervised or probe-derived target

Passing `episode.correct_action_index`, the cue, or a Phase 2A probe
prediction into the learner would demonstrate supervised imitation, not
reward learning. It would make the scientific gate circular and is forbidden.

### Deferred: train the recurrent reservoir

BPTT, recurrent plasticity, dopamine-gated STDP, and connectome-specific
learning may become later comparisons. Phase 2A already shows that frozen
recurrent activity contains sufficient information, so changing the reservoir
now would conflate representation and readout learning again.

## Architectural boundary

The new path is composed rather than inserted into the existing Phase 1
controller:

```text
DelayedCueTask numeric frames
        |
        v
frozen RecurrentPolicy.advance()
        |
        | decision-time hidden vector
        v
RewardModulatedReadout.select_for_training()
        |
        | sampled action
        v
DelayedCueTask.reward()
        |
        | scalar reward only
        v
RewardModulatedReadout.learn()
```

Evaluation uses the same state source and readout but calls
`select_greedy()`, which creates no eligibility and changes no parameter.

The existing `RecurrentPolicy` still owns an output matrix for Phase 1
compatibility. Phase 2B never reads it for action selection and never modifies
it. Input, recurrent, and legacy output matrices are snapshotted around the
experiment as frozen-system controls.

## Reward-modulated readout

### Public data and construction

`reward_readout.py` provides:

```text
RewardReadoutDecision(
    action_index: int,
    logits: ndarray,
    probabilities: ndarray,
)

RewardModulatedReadout(
    hidden_size: int,
    action_count: int,
    learning_rate: float = 0.05,
    temperature: float = 1.0,
)
```

Weights have shape `(action_count, hidden_size)`; biases have shape
`(action_count,)`. Both start at exact zero so the initial policy is
unbiased. Returned arrays are independent, read-only `float64` values.

The acceptance configuration fixes `learning_rate=0.05` and
`temperature=1.0` before the three acceptance seeds are evaluated.

### Legal-action softmax

For hidden vector `h`, legal action set `L`, weights `W`, and bias `b`:

```text
z = W h + b
p(a | h, L) = exp((z_a - max(z_L)) / temperature)
               / sum(exp((z_j - max(z_L)) / temperature), j in L)
```

Illegal logits are returned as negative infinity and illegal probabilities as
zero. Only legal actions participate in normalization.

`select_for_training(hidden, legal_action_indices, rng)` samples one legal
action from the categorical distribution with the supplied seeded NumPy
generator. It stores one pending eligibility and rejects a second training
selection until feedback consumes the first.

`select_greedy(hidden, legal_action_indices)` selects the maximum legal
probability, with the lowest numeric index winning exact ties. It neither
creates nor consumes eligibility.

### Eligibility and update

Let `a` be the sampled action, `e_a` its one-hot vector over all actions,
and `p` the probability vector with zeros for illegal actions. The readout
stores independent copies of:

```text
E_weights = (e_a - p) outer h
E_bias = e_a - p
```

After the environment returns scalar reward `r`, `learn(r)` validates that
it is finite, clips it to `[-1, 1]`, and performs:

```text
W <- W + learning_rate * clipped(r) * E_weights
b <- b + learning_rate * clipped(r) * E_bias
```

The pending eligibility is consumed even when the clipped reward is zero.
Calling `learn` without pending eligibility, calling it twice, or beginning a
second training decision before feedback raises `RuntimeError`.

The update does not store a cue, task label, phase, delay count, reward
history, or semantic behavior state.

### Digests

`parameter_digest()` is SHA-256 over explicit shape metadata followed by the
C-contiguous `float64` bytes of weights and biases. It provides exact
repeatability and immutability evidence; it is not a serialization format.

## Phase 2B experiment protocol

### Fixed configuration

`RewardLearningConfig` defaults are:

```python
hidden_size = 64
recurrent_radius = 0.9
learning_rate = 0.05
temperature = 1.0
training_episodes = 2_000
evaluation_blocks = 20
```

Acceptance seeds remain `(7, 17, 29)`. Every training block contains exactly
the Cartesian product of cues left/right and delays 1–5, shuffled by a seeded
generator. Every episode receives fresh seeded distractors.

The delayed-cue stimulus contract is unchanged:

```text
left cue       [1, 0, 0, 0]
right cue      [0, 1, 0, 0]
delay          [0, 0, d, 0], d in [-0.25, 0.25]
decision       [0, 0, 0, 1]
```

At decision time, both cue classes present exactly the same numeric stimulus.

### Normal training

For every episode:

```python
policy.reset_state()
policy.advance(episode.cue_stimulus)
for stimulus in episode.delay_stimuli:
    policy.advance(stimulus)
hidden = policy.advance(episode.decision_stimulus)
decision = readout.select_for_training(hidden, (0, 1), action_rng)
reward = task.reward(episode, decision.action_index)
readout.learn(reward)
```

Only `task.reward` sees the experiment fixture and correct answer. The
readout receives the chosen action implicitly through its stored eligibility
and receives only the returned scalar reward.

There is no epsilon-greedy layer. Seeded categorical sampling from the softmax
policy supplies exploration and makes the learning equation match the action
distribution that generated the eligibility.

### Evaluation fixtures

Two hundred immutable evaluation episodes are constructed once from a
dedicated RNG lineage: 20 balanced blocks, giving 200 overall cases and 40
cases at each delay. The same fixture objects are reused for:

- pre-training greedy evaluation;
- post-training greedy recurrent evaluation;
- post-training state-reset ablation;
- the independently trained shuffled-reward control.

Evaluation never calls `learn` and must leave every digest unchanged.

### State-reset ablation

The ablation processes cue and delay frames normally, then calls
`policy.reset_state()` immediately before processing the shared decision
stimulus. Every ablated hidden vector must therefore be exactly equal.

Because the evaluation set is balanced and greedy tie behavior is
deterministic, one constant action must produce exactly 100/200 overall and
20/40 at every delay.

### Shuffled-reward control

The control constructs a fresh reservoir and zero-initialized readout with the
same configuration and training stimuli as the normal run. It uses independent
generators initialized from fixed lineages.

Within every ten-case training block, five `+1` and five `-1` rewards are
shuffled independently of cue, correct action, sampled action, and delay. The
control never calls `task.reward` during training. This preserves reward
magnitude and count while removing task information.

The control is evaluated against the real correct actions only after
training. It must not meet the normal Phase 2B learning gate. Across the pooled
600 evaluation cases from the three fixed seeds, its accuracy must remain in
`[0.40, 0.60]`; each individual seed must remain below `0.75`.

## Deterministic RNG separation

Every random concern receives its own `SeedSequence` lineage:

```python
training_fixture_rng = SeedSequence([seed, 0x54524149])
training_action_rng = SeedSequence([seed, 0x4143544E])
evaluation_rng = SeedSequence([seed, 0x4556414C])
shuffled_reward_rng = SeedSequence([seed, 0x53485546])
shuffled_action_rng = SeedSequence([seed, 0x53414354])
```

Normal and shuffled runs use independently constructed policies, readouts, and
generators. Repeating a complete run reconstructs every object; no trained
state or advanced generator is reused.

The benchmark records at least:

- pre-training, post-training, reset, and shuffled-control integer counts;
- per-delay post-training counts;
- normal and shuffled training reward totals;
- final ten-case training counts;
- readout parameter digests;
- legacy policy input, recurrent, and output digests before and after;
- ordered greedy-choice digests;
- exact independent-run repeatability.

## Validation and error behavior

Constructors reject booleans, non-integer sizes, non-positive sizes, non-finite
rates, non-positive temperature, and invalid action counts.

Selection rejects:

- hidden arrays with the wrong shape, rank, dtype compatibility, or non-finite
  values;
- empty, duplicate, non-integer, or out-of-range legal action indices;
- a non-`numpy.random.Generator` training RNG;
- a second training decision while feedback is pending.

Learning rejects non-numeric or non-finite reward and missing eligibility.
Evaluation paths assert that no eligibility exists before and after use.

## File boundaries

The implementation plan may create:

| File | Responsibility |
|---|---|
| `src/neural_state_machine/reward_readout.py` | Softmax action selection, one-shot eligibility, reward update, digest |
| `src/neural_state_machine/reward_learning.py` | Balanced Phase 2B training, controls, evaluation, results |
| `scripts/benchmark_reward_learning.py` | Stable JSON acceptance command |
| `tests/test_reward_readout.py` | Numerical rule, masking, validation, lifecycle tests |
| `tests/test_reward_learning.py` | Boundary, control, repeatability, and scientific-gate tests |

The plan may modify only public exports, README documentation, and CI wiring in
addition to those new files.

These existing scientific artifacts remain unchanged unless a separately
demonstrated defect receives a new reviewed design:

- `src/neural_state_machine/policy.py`;
- `src/neural_state_machine/controller.py`;
- `src/neural_state_machine/memory_task.py`;
- `src/neural_state_machine/memory_benchmark.py`;
- `src/neural_state_machine/memory_probe.py`;
- all Phase 1 and Phase 2A tests and benchmark outputs.

NumPy remains the only runtime dependency.

## Acceptance gates

Phase 2B passes only if all of the following hold:

1. Seeds 7, 17, and 29 independently reach at least 180/200 post-training
   recurrent accuracy.
2. Every delay for every seed reaches at least 34/40.
3. Every state-reset ablation is exactly 100/200 overall and 20/40 per delay,
   with all reset hidden vectors exactly equal.
4. The shuffled-reward control stays below 150/200 for every seed and its
   pooled 600-case accuracy is in `[0.40, 0.60]`.
5. Each normal readout digest changes during training; evaluation does not
   change it.
6. The legacy policy input, recurrent, and output matrices are byte-identical
   before and after every normal and control run.
7. Two independently reconstructed executions of each seed are exactly equal,
   including integer counts, ordered choices, and all digests.
8. A boundary spy proves that cue, correct action, delay, and phase never enter
   the policy or readout.
9. The complete existing test suite, Ruff, Phase 1 benchmark, frozen Phase 2A
   benchmark, and frozen V0 reward-baseline tests remain unchanged and pass.
10. GitHub Actions passes the exact pushed commit on Python 3.10, 3.11, and
    3.12.

Thresholds, seeds, episode counts, delay range, distractor amplitude, and task
vectors are frozen by this design. They are not weakened after observing an
acceptance failure.

## Failure protocol

If the reward learner misses any normal, reset, shuffled-control,
repeatability, or frozen-system gate:

1. report the exact seed, counts, per-delay results, and digests;
2. retain the failing test and benchmark output as evidence;
3. do not pass labels, probe predictions, or task metadata into the learner;
4. do not tune the committed acceptance seeds or weaken the gates;
5. return to a new design review before changing the learning rule,
   hyperparameters, reservoir, or task.

A failure means the selected online rule did not acquire the behavior under
the pre-registered protocol. It does not invalidate Phase 2A's memory result.

## Interpretation and next gate

The result matrix remains explicit:

| Phase 2A memory | Phase 2B reward learning | Supported conclusion |
|---|---|---|
| pass | fail | Neural state retains the cue; the tested online rule cannot reliably learn the action mapping. |
| pass | pass | Frozen recurrent memory can drive a separately learned reward-modulated action readout. |

Only after a Phase 2B pass may Phase 3 design multiple concurrent memories or
conflicting attack/retreat/heal objectives. A `fly-brain` adapter remains a
later comparison behind the same stimulus/activity/action/reward boundary and
must pass the same representation-versus-learning separation.
