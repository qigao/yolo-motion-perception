# Neural State Machine Game Experiment Design

## Decision

`experiment/neural-state-machine` is a standalone orphan branch. It has no
parent commit and shares no files, package names, runtime paths, tests, CI, or
documentation with `master`.

The branch contains one focused research prototype: a deterministic game loop
controlled by a recurrent neural dynamical system whose hidden activity carries
context without predefined compound behavior states.

## Research question

Can a small recurrent neural controller produce reproducible,
history-dependent action tendencies and respond to reward feedback without
enumerating states such as `Attack`, `Retreat`, `Heal`, or their combinations?

## Boundaries

### Included

- Python 3.10+ and NumPy.
- A normalized observation vector representing enemy, health, threat, healing,
  and map-boundary stimuli.
- Four atomic actions: move left, move right, attack, and wait.
- A seeded recurrent controller with observable neural activity.
- Deterministic action selection.
- Reward-modulated Hebbian plasticity on the action decoder.
- A deterministic one-dimensional combat environment.
- Hard environment invariants for bounds, health, attack range, item use, and
  terminal states.
- Exact replay, neural-context separation, and action-margin measurements.
- Unit tests, a machine-readable benchmark, Ruff, and GitHub Actions.

### Excluded

- Every file and interface from `master`.
- YOLO, FlyVis, fly-brain, ROS2, game engines, GPU code, model downloads, and
  network-dependent tests.
- A claim of biological fidelity, full-brain simulation, STDP, or dopamine
  neuron dynamics.
- RL frameworks such as PPO or DQN.
- Named compound behavior states, behavior trees, or manually authored state
  transition graphs.
- Automatic semantic labels for neural clusters.

## Package structure

```text
src/neural_state_machine/
  types.py         immutable observations, decisions, snapshots, traces
  encoding.py      observation-to-vector boundary
  controller.py    recurrent dynamics and reward plasticity
  environment.py   toy-game facts and hard rules
  experiment.py    closed-loop replay and measurements
```

Each module has one responsibility and communicates through public immutable
records. The environment never reads controller internals. The controller never
reads environment state directly.

## Data flow

```text
Game observation
      |
      v
fixed numeric encoder
      |
      v
h(t+1) = tanh(W_in x(t) + W_rec h(t) + bias)
      |
      +----> copied neural state for analysis
      |
      v
action = argmax(W_out h(t+1))
      |
      v
hard-rule game step ----> reward ----> selected decoder-row update
```

The controller receives sensor facts, not an `AttackState` or `RetreatState`.
Actions are atomic outputs, not states.

## Contracts

### Observation

`GameObservation` is frozen and contains:

- `enemy_distance` in `[0, 1]`; `1` means far away or absent.
- `enemy_direction` in `{-1, 0, 1}`.
- `health` in `[0, 1]`.
- `incoming_threat` in `[0, 1]`.
- `healing_distance` in `[0, 1]`; `1` means far away or absent.
- `healing_direction` in `{-1, 0, 1}`.
- `left_blocked` and `right_blocked` booleans.

The encoder returns a nine-element `float64` vector in the listed order with a
final constant `1.0` bias input.

### Controller

`RecurrentController(hidden_size, seed, learning_rate)` owns:

- seeded `W_in`, `W_rec`, and `W_out` matrices;
- a bounded recurrent hidden vector;
- the most recent immutable decision as its eligibility source.

`step(observation)` advances the neural dynamics once and returns
`NeuralDecision(action, logits, hidden_state)`. Arrays are read-only copies so a
caller cannot mutate controller state.

`learn(reward)` clips reward to `[-1, 1]` and updates only the selected action
row:

```text
W_out[selected] += learning_rate * clipped_reward * hidden_state
```

Calling `learn` without a preceding step is an error. Resetting recurrent state
does not erase learned output weights.

### Environment

The arena uses integer cells. The player acts first. A legal adjacent attack
damages the enemy. A living enemy then damages an adjacent player or moves one
cell toward the player. Entering the healing cell consumes the item and clamps
health to its configured maximum.

The environment owns all reward constants and hard rules. It clips movement at
boundaries, rejects post-terminal steps, and prevents negative or excessive
health. These rules are intentionally explicit; they are invariants rather than
complex behavioral decisions.

## Measurements

The benchmark reports JSON with:

- `replay_equal`: exact equality for identical seed and episode inputs;
- `context_separation`: centroid distance divided by pooled within-context
  spread for two observation histories;
- `positive_margin_delta`: change in selected-action margin after positive
  reward from the same reset neural state;
- `negative_margin_delta`: change after negative reward;
- deterministic step count, total reward, and terminal snapshot for one
  closed-loop episode.

Context separation only demonstrates distinguishable neural trajectories. It
does not prove that semantic attractors or biological game understanding have
emerged.

## Failure behavior

- Non-finite or out-of-range observations and configuration values raise
  `ValueError`.
- Shape mismatches raise `ValueError` before matrix multiplication.
- Learning without eligibility raises `RuntimeError`.
- Stepping a terminal game raises `RuntimeError`.
- Benchmark invariants are assertions in tests; the CLI exits non-zero on
  failure and emits JSON on success.

## Acceptance criteria

- The branch's first commit has no parents.
- No tracked file from `master` remains unless independently recreated for this
  project.
- Equal seeds and inputs produce exactly equal states, actions, rewards, and
  final snapshots.
- Positive reward raises, and negative reward lowers, the selected-action
  margin from the same reset context.
- Different observation histories produce a finite positive separation score.
- Tests cover every game invariant and all validation boundaries.
- `pytest -q`, `ruff check .`, and `python scripts/benchmark.py` all exit zero on
  Python 3.10, 3.11, and 3.12 in CI.
- The README states what the experiment demonstrates and what it does not.

## Later integration gate

Only after this baseline is stable should a separate design evaluate a
`fly-brain` adapter behind the same observation/action/reward boundary. FlyVis
remains outside this project because it is a visual subsystem, not the game
controller being tested here.
