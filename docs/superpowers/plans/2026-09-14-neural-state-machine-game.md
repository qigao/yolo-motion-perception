# Neural State Machine Game Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone deterministic neural game controller and benchmark on an orphan branch, without any dependency on `master`.

**Architecture:** A fixed encoder converts immutable observations to numeric stimuli. A seeded recurrent controller selects atomic actions and applies reward-modulated output plasticity; a separate toy environment owns all hard rules, and experiment functions record replay and neural metrics.

**Tech Stack:** Python 3.10+, NumPy, pytest, Ruff, setuptools, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-14-neural-state-machine-game-design.md`

## Global Constraints

- The branch is orphaned and has no parent commits.
- No import, file, API, or runtime dependency on `master` is permitted.
- NumPy is the only runtime dependency.
- All initialization and episodes are deterministic for an explicit seed.
- Atomic actions are allowed; compound behavior states and transition tables are not.
- No YOLO, FlyVis, fly-brain, ROS2, game engine, GPU, or network access.

---

### Task 1: Bootstrap contracts and encoding with TDD

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `src/neural_state_machine/__init__.py`
- Create: `src/neural_state_machine/types.py`
- Create: `src/neural_state_machine/encoding.py`
- Create: `tests/test_encoding.py`

**Interfaces:**
- Produces: `Action`, frozen `GameObservation`, frozen `NeuralDecision`, and
  `encode_observation(observation: GameObservation) -> numpy.ndarray`.
- Encoding order is enemy distance/direction, health, threat, healing
  distance/direction, left blocked, right blocked, and bias.

- [ ] Write `tests/test_encoding.py` importing the wished-for public API and
  asserting the literal vector
  `[0.25, -1.0, 0.5, 0.75, 1.0, 0.0, 1.0, 0.0, 1.0]`.
- [ ] Run `.venv/bin/pytest tests/test_encoding.py -q` and confirm RED because
  `neural_state_machine` does not exist.
- [ ] Create the minimal setuptools project and immutable contracts; validate
  finite normalized scalars and exact direction values.
- [ ] Implement the nine-element `float64`, read-only encoded vector.
- [ ] Add parametrized invalid-observation tests where each literal invalid
  value raises `ValueError`; run focused tests after each minimal fix.
- [ ] Run the focused tests and Ruff; commit as
  `feat: add neural stimulus contracts`.

### Task 2: Implement recurrent dynamics and plasticity with TDD

**Files:**
- Create: `src/neural_state_machine/controller.py`
- Create: `tests/test_controller.py`
- Modify: `src/neural_state_machine/__init__.py`

**Interfaces:**
- Produces: `RecurrentController(hidden_size: int = 16, seed: int = 0,
  learning_rate: float = 0.1)`, `step(GameObservation) -> NeuralDecision`,
  `learn(reward: float) -> None`, and `reset_state() -> None`.

- [ ] Write a same-seed test that feeds three literal observations to two
  controllers and compares actions and arrays with `numpy.array_equal`.
- [ ] Run the focused test and confirm RED because the controller is absent.
- [ ] Implement seeded matrices, recurrent spectral scaling below `0.9`, tanh
  evolution, stable action-order argmax, and read-only copied outputs.
- [ ] Add RED-first validation tests for `hidden_size <= 0`, non-positive or
  non-finite learning rate, and attempts to mutate returned arrays.
- [ ] Add a RED test that measures the selected logit minus the largest other
  logit from the same reset context, learns `+1.0`, and observes a larger
  margin; implement the selected-row Hebbian update.
- [ ] Add RED tests for reward `-1.0` lowering the margin, reward clipping,
  non-finite reward rejection, and learning before a step raising
  `RuntimeError`; implement only those behaviors.
- [ ] Run all tests and Ruff; commit as
  `feat: add recurrent controller and plasticity`.

### Task 3: Implement the hard-rule toy environment with TDD

**Files:**
- Create: `src/neural_state_machine/environment.py`
- Create: `tests/test_environment.py`
- Modify: `src/neural_state_machine/types.py`
- Modify: `src/neural_state_machine/__init__.py`

**Interfaces:**
- Produces: frozen `GameConfig`, `GameSnapshot`, `StepResult`, and `ToyGame`
  with `reset() -> GameObservation`, `observe() -> GameObservation`,
  `snapshot() -> GameSnapshot`, and `step(action: Action) -> StepResult`.

- [ ] Write a literal initial snapshot/observation test and confirm RED.
- [ ] Implement validated configuration, reset, snapshot, and normalized
  observation generation.
- [ ] Write separate RED tests for left/right boundary clipping, attack outside
  range, adjacent hit, kill, enemy approach, adjacent player damage, healing
  consumption, health clamping, player death, and post-terminal rejection.
- [ ] Implement one rule at a time in player-then-enemy order and rerun each
  focused failing test before proceeding.
- [ ] Add a reset replay test comparing observations, rewards, and snapshots
  from the same literal action sequence.
- [ ] Run all tests and Ruff; commit as
  `feat: add deterministic combat environment`.

### Task 4: Add experiment traces and metrics with TDD

**Files:**
- Create: `src/neural_state_machine/experiment.py`
- Create: `tests/test_experiment.py`
- Modify: `src/neural_state_machine/__init__.py`

**Interfaces:**
- Produces: frozen `EpisodeTrace`,
  `run_episode(controller, game, max_steps: int) -> EpisodeTrace`,
  `replay_equal(left, right) -> bool`, and
  `context_separation(left_states, right_states) -> float`.

- [ ] Write RED tests for a bounded episode, copied neural arrays, and exact
  equality/inequality of two hand-built traces.
- [ ] Implement the closed loop `observe -> step controller -> step game ->
  learn reward`, stopping at terminal or `max_steps`.
- [ ] Write a RED metric test with hand-derived two-dimensional state arrays;
  require centroid distance divided by pooled RMS spread.
- [ ] Implement validation for rank, matching width, non-empty rows, finite
  values, and a finite zero-spread rule.
- [ ] Add a same-seed closed-loop replay test and a different-seed inequality
  test; implement explicit NumPy-safe comparison.
- [ ] Run all tests and Ruff; commit as
  `feat: add replay and neural context metrics`.

### Task 5: Add standalone benchmark, documentation, and CI

**Files:**
- Create: `scripts/benchmark.py`
- Create: `tests/test_benchmark.py`
- Create: `README.md`
- Create: `LICENSE`
- Create: `.github/workflows/ci.yml`
- Modify: `pyproject.toml`

**Interfaces:**
- Produces: `run_benchmark(seed: int = 7) -> dict[str, object]` and a CLI that
  prints one stable JSON object.

- [ ] Write a RED test importing the absent benchmark and requiring exact
  replay equality, positive finite context separation, positive reward margin
  delta `> 0`, negative reward margin delta `< 0`, and stable episode summary.
- [ ] Implement two literal stimulus histories, isolated positive/negative
  plasticity probes, and two same-seed closed-loop episodes.
- [ ] Add `scripts/__init__.py` only if import mechanics require it; do not add
  unused package structure.
- [ ] Write README commands, architecture, metric interpretation, limitations,
  and the future `fly-brain` adapter boundary.
- [ ] Add Apache-2.0 licensing and CI for Python 3.10/3.11/3.12 running install,
  pytest, Ruff, and the benchmark.
- [ ] Run `.venv/bin/python -m pip install -e '.[dev]'`, `.venv/bin/pytest -q`,
  `.venv/bin/ruff check .`, and `.venv/bin/python scripts/benchmark.py`.
- [ ] Commit as `feat: add standalone neural game benchmark`.

### Task 6: Audit orphan history and publish exact head

**Files:**
- Modify only files required by verified audit failures.

**Interfaces:**
- Produces: a remote orphan branch whose exact head matches verified local
  content.

- [ ] Check the spec line by line and search for excluded imports, compound
  behavior-state declarations, placeholders, and stale master file names.
- [ ] Run a clean-build verification plus `git diff --check`.
- [ ] Confirm the root commit has zero parents with `git rev-list --parents
  --max-count=1 HEAD`.
- [ ] Force-update only `origin/experiment/neural-state-machine`; do not update
  `master`.
- [ ] Inspect exact-head GitHub Actions and report job/step evidence or the
  precise remaining failure.
