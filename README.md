# Neural State Machine Game Experiment

This branch is a **standalone orphan project**. It does not inherit the files,
package, CI, or commit history of `master`.

The project tests one narrow idea:

> Can recurrent neural activity carry complex game context and change its
> action tendency through reward feedback without defining every compound
> behavior state in advance?

It is a deterministic research baseline, not a biologically complete fly brain
and not a replacement for YOLO perception.

## Model

```text
game facts
   ↓
normalized stimulus vector
   ↓
recurrent neural state h(t)
   ↓
atomic action: left / right / attack / wait
   ↓
hard-rule toy game
   ↓
reward or punishment
   ↓
reward-modulated output plasticity
```

The controller never receives labels such as `AttackState`, `RetreatState`, or
`LowHealthBossState`. It receives distances, directions, health, threat, and
boundary signals. The only enum is the atomic action vocabulary.

The environment still enforces deterministic invariants:

- map boundaries cannot be crossed;
- attacks only damage an adjacent enemy;
- health remains within configured bounds;
- healing items are single-use;
- terminal games cannot be stepped.

This is the intended split between a **soft neural state machine** for choices
and **hard rules** for legality.

## Install

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
```

On Windows, use `.venv\Scripts\python` instead of `.venv/bin/python`.

## Run

```bash
python scripts/benchmark.py
```

The command prints one JSON object containing:

- `replay_equal`: identical seeds and inputs produced identical neural states,
  actions, rewards, and final snapshots;
- `context_separation`: two stimulus histories occupied distinguishable regions
  of hidden-state space;
- `positive_margin_delta`: positive feedback increased the selected action's
  logit margin for the same reset context;
- `negative_margin_delta`: negative feedback decreased that margin;
- `episode`: one bounded deterministic closed-loop trace summary.

Run the verification suite:

```bash
pytest -q
ruff check .
python scripts/benchmark.py
```

Tests and benchmarks require no GPU, model weights, network access, FlyVis,
fly-brain, YOLO, ROS2, or game engine.

## Package layout

```text
src/neural_state_machine/
├── types.py         observations, actions, snapshots, decisions
├── encoding.py      observation → numeric stimulus
├── controller.py    recurrent dynamics + reward plasticity
├── environment.py   toy-game facts + hard invariants
├── experiment.py    closed-loop traces + measurements
└── benchmark.py     reproducible research benchmark
```

## What the result means

A positive separation score demonstrates that the recurrent system distinguishes
the two tested input histories. A repeatable margin change demonstrates that
reward can modify a later action tendency. Together they establish the minimum
mechanics needed to study a neural dynamical state machine.

They do **not** prove that semantic states, stable attractors, intelligence, or
biological game understanding have emerged. The V0 learning rule is a compact
reward-modulated Hebbian update on the action decoder, not dopamine-gated STDP.

## Next research gate

A future branch can place a `fly-brain` adapter behind the same boundary:

```text
observation/stimulus in → neural activity + action out → reward feedback in
```

That adapter should be compared against this deterministic baseline before
adding connectome-scale compute or stronger biological claims. FlyVis remains a
possible temporal visual subsystem, not the controller studied here.

## Design documents

- `docs/superpowers/specs/2026-09-14-neural-state-machine-game-design.md`
- `docs/superpowers/plans/2026-09-14-neural-state-machine-game.md`

## License

Apache-2.0.

