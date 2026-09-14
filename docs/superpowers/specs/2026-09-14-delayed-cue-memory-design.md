# Delayed-Cue Neural Memory Experiment Design

## Decision

Phase 2 will test whether behavior can depend on a stimulus that is no longer
present in the current observation. The experiment uses a delayed left/right
cue task, a frozen recurrent reservoir, and a trainable action readout. It does
not add an explicit memory flag, behavior state, transition table, BPTT, or an
RL framework.

This is the next required evidence after Phase 1. Phase 1 established
deterministic recurrent trajectories and reward-driven output plasticity, but
its fully observable combat task could be completed by repeatedly attacking.
It did not establish that recurrent state was necessary for the behavior.

## Claim under test

After seeing a left or right cue, the controller must retain enough information
through one to five cue-free delay steps to choose the corresponding direction
on a later decision frame.

At the decision frame:

- the left-cue and right-cue stimulus vectors are element-for-element equal;
- the permitted actions are exactly `MOVE_LEFT` and `MOVE_RIGHT`;
- the only permitted source of different deterministic choices is the
  controller's recurrent hidden state and learned readout;
- resetting hidden state immediately before the decision must remove the
  information and return performance to the balanced 50% baseline.

If these conditions hold, action differences cannot be attributed to the
current observation or an enumerated task state supplied to the controller.

## Alternatives considered

### Selected: frozen recurrent reservoir plus learned readout

Random seeded input and recurrent weights remain fixed. Reward feedback trains
only the output weights. This isolates whether recurrent activity carries the
cue and keeps the result deterministic and interpretable.

### Rejected for Phase 2: train the full recurrent network

BPTT, PPO, DQN, or another optimizer could produce a stronger policy, but it
would add optimizer state, gradient behavior, and hyperparameters before the
basic memory hypothesis has been isolated.

### Rejected: explicit memory variable or FSM state

Recording `last_cue = LEFT` would make the task trivial by putting the answer in
a predefined state. It cannot test whether memory exists in neural dynamics.

## Architectural change

The current `RecurrentController` directly encodes `GameObservation`, owns the
recurrent matrices, selects actions, and learns. Phase 2 extracts those generic
dynamics into `RecurrentPolicy` while preserving the existing controller as a
compatibility adapter.

```text
                         ┌──────────────────────────┐
GameObservation ─encode─►│ RecurrentController     │
                         │ game-specific adapter    │
                         └────────────┬─────────────┘
                                      │ numeric stimulus
                                      ▼
                         ┌──────────────────────────┐
                         │ RecurrentPolicy          │
                         │ generic neural dynamics  │
                         └────────────┬─────────────┘
                                      ▲
                                      │ cue/delay/decision vectors
                         ┌────────────┴─────────────┐
                         │ DelayedCueTask           │
                         │ experiment protocol      │
                         └──────────────────────────┘
```

The Phase 1 public API and its 57 tests remain valid. The extraction must not
change Phase 1 seeded trajectories, selected actions, plasticity deltas, or
benchmark JSON.

## Component boundaries

### `RecurrentPolicy`

`RecurrentPolicy` is a numeric neural-dynamics component. It has no knowledge
of games, cues, health, or semantic action names.

Construction parameters:

```text
input_size: positive integer
action_count: integer >= 2
hidden_size: positive integer
seed: integer
learning_rate: finite positive float
recurrent_radius: finite float in [0, 1)
```

It owns:

- seeded input, recurrent, and output matrices;
- the recurrent hidden vector;
- eligibility from the most recent decision;
- no task-state or observation history outside that vector.

Public operations:

```python
advance(stimulus: ndarray) -> ndarray
decide(
    stimulus: ndarray,
    legal_action_indices: tuple[int, ...],
    *,
    explore_probability: float = 0.0,
    rng: numpy.random.Generator | None = None,
) -> PolicyDecision
learn(reward: float) -> None
reset_state() -> None
```

`advance` evolves hidden state without creating an action or eligibility. It is
used for cue and delay frames. `decide` evolves hidden state once for the
decision stimulus, masks illegal logits, selects an action, and records the
decision hidden vector for learning.

With zero exploration, selection is deterministic argmax with lowest-index tie
breaking. Positive exploration requires an explicitly supplied seeded random
generator. Exploration samples uniformly from legal actions; it never selects
masked actions.

Returned arrays are read-only copies. Stimuli must be finite one-dimensional
`float64`-compatible arrays of exactly `input_size` elements.

### Phase 1 compatibility adapter

`RecurrentController` retains its existing constructor and methods:

```python
step(GameObservation) -> NeuralDecision
learn(reward) -> None
reset_state() -> None
```

It owns a nine-input, four-action `RecurrentPolicy`, converts
`GameObservation` using the existing encoder, and converts numeric action
indices back to `Action`. The refactor must preserve the initialization draw
order and all Phase 1 numerical outputs exactly.

### `DelayedCueTask`

The task is an experiment protocol, not a controller state machine. It creates
stimulus sequences and scores the final direction choice. The controller never
receives the cue label, correct action, delay count, episode phase, or score.

The four-element stimulus contract is:

```text
index 0: left-cue channel
index 1: right-cue channel
index 2: bounded distractor channel
index 3: decision channel
```

For a left cue:

```text
cue frame      [1, 0, 0, 0]
delay frame    [0, 0, d, 0]  where d is seeded in [-0.25, 0.25]
decision frame [0, 0, 0, 1]
```

For a right cue, the cue frame is `[0, 1, 0, 0]`; all other contracts are
identical. Delay length is an integer from one through five.

The decision frame is a shared immutable literal. A test compares left-cue and
right-cue decision vectors using exact array equality.

Task phases may exist in the test harness for validation and sequencing, but
phase identifiers are never encoded into the controller stimulus except for
the single shared decision channel. They are hard experiment protocol, not
learned behavior states.

## Training protocol

Training uses the same recurrent reservoir across all episodes and changes only
the output matrix.

Each training block contains the Cartesian product of:

- cue: left and right;
- delay: one, two, three, four, and five steps.

The ten cases are shuffled by a seeded generator. Repeating complete balanced
blocks prevents cue-frequency bias. Delay distractors use the same generator
but are not reused by evaluation.

For each episode:

1. reset recurrent hidden state;
2. call `advance` for the cue frame;
3. call `advance` for every delay frame;
4. call `decide` on the shared decision frame with only left/right legal;
5. return `+1` for the correct direction and `-1` for the wrong direction;
6. call `learn` once with that reward.

Training uses seeded epsilon-greedy exploration. The default schedule is linear
decay from `0.25` to `0.02` over 2,000 episodes. Evaluation uses zero
exploration and performs no learning.

No training sample or evaluation result changes input or recurrent matrices.

## Evaluation protocol

Evaluation uses new deterministic distractor streams derived from an evaluation
seed distinct from the training seed. It contains equal numbers of left and
right cues for every delay length.

Two modes evaluate the same trained readout:

### Recurrent mode

The hidden state flows continuously from cue through delay to decision.

### State-reset ablation

The same cue and delay frames are processed, but `reset_state()` is called
immediately before the shared decision frame. Because the decision stimulus is
identical and the evaluator is deterministic, the controller makes one
constant choice. A balanced cue set therefore produces exactly 50% accuracy.

The ablation resets transient hidden state only. It does not reset learned
output weights, substitute another model, or retrain a baseline.

## Metrics

`MemoryExperimentResult` reports:

- seed and configuration;
- pre-training recurrent accuracy;
- post-training recurrent accuracy;
- post-training state-reset accuracy;
- post-training accuracy for each delay length 1–5;
- total training reward and training accuracy in the final balanced block;
- exact repeatability of two independent runs using the same seed;
- a deterministic digest of learned output weights and evaluation choices.

Accuracy is computed from literal correct/total counts. Metrics must not reuse
controller predictions to derive expected labels.

## Acceptance criteria

The committed benchmark must satisfy all of the following with the documented
default configuration:

1. Post-training recurrent accuracy is at least `0.90` over the balanced
   evaluation set.
2. Accuracy for each individual delay length from one through five is at least
   `0.85`.
3. State-reset ablation accuracy is exactly `0.50`.
4. Left-cue and right-cue decision-frame vectors are exactly equal.
5. Two independent runs with the same seed return identical result objects and
   digests.
6. At least three documented seeds pass criteria 1–5; default CI seeds are
   `7`, `17`, and `29`.
7. Phase 1 tests and benchmark JSON remain numerically unchanged.
8. No explicit cue memory, semantic hidden-state label, compound behavior
   state, behavior tree, or transition table is added to controller code.
9. `pytest -q`, `ruff check .`, the Phase 1 benchmark, and the Phase 2 memory
   benchmark pass on Python 3.10, 3.11, and 3.12.

If the recurrent accuracy threshold is not reached, the experiment reports a
failed hypothesis boundary. The threshold must not be weakened, and seeds must
not be replaced, merely to make CI green. A design revision is required before
changing the reservoir, learning rule, episode count, or task difficulty.

## Error handling

- Invalid sizes, radii, probabilities, rewards, and non-finite stimuli raise
  `ValueError`.
- Exploration without a supplied generator raises `ValueError`.
- Empty, duplicate, or out-of-range legal-action masks raise `ValueError`.
- `learn` without a preceding decision raises `RuntimeError`.
- Learning twice from one decision raises `RuntimeError`; one reward consumes
  the eligibility record.
- Invalid cue, delay, or episode counts raise `ValueError`.
- Evaluation never mutates learned weights; tests compare weight digests before
  and after evaluation.

## Files and dependency limits

Expected new modules:

```text
src/neural_state_machine/policy.py
src/neural_state_machine/memory_task.py
src/neural_state_machine/memory_benchmark.py
scripts/benchmark_memory.py
```

Expected modifications are limited to the Phase 1 controller adapter, public
exports, tests, README, and CI. NumPy remains the only runtime dependency.

## Interpretation boundary

Passing Phase 2 supports this narrow statement:

> A stimulus that is absent at decision time remains behaviorally available in
> recurrent neural activity, and erasing that activity removes the behavior.

It does not demonstrate semantic attractors, open-world game intelligence,
biological plausibility, or a complete replacement for all state machines.
Those require later experiments.

## Next gate

Only after Phase 2 passes should Phase 3 introduce multiple concurrent memories
and conflicting objectives such as attack, retreat, and healing. A `fly-brain`
adapter remains later still; it must be compared against the same delayed-cue
protocol before biological topology is credited with any improvement.
