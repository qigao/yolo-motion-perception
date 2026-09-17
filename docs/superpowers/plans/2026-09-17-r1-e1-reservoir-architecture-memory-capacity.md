# R1 Phase E1 Reservoir Architecture and Memory Capacity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the preregistered R1 E1 reservoir-architecture experiment so shallow, grouped, and deep ESNs can be compared under fixed neuron budgets on memory, history separability, and synthetic YOLO-like corruption without modifying or exercising R2 delayed-credit learning.

**Architecture:** Add an isolated `r1_e1_*` research subsystem beside the frozen Phase 2A memory code. A common deterministic reservoir factory supplies Shallow / Grouped-2 / Grouped-4 / Deep-2 / Deep-4 states; a separate SVD/least-squares Ridge module supplies binary and multiclass frozen readouts; E1-A/B/C own their fixtures and metrics; a protocol/evidence layer binds exact configuration, digests, deterministic controls, and write-once measurement outputs. Existing `memory_probe.py` remains unchanged and is used only as the Phase 2A compatibility oracle.

**Tech Stack:** Python 3.10+, NumPy only at runtime, pytest, Ruff, setuptools, GitHub Actions on Python 3.12.

**Spec:** `docs/superpowers/specs/2026-09-17-r1-e1-reservoir-architecture-memory-capacity-design.md`

## Global Constraints

- Work on `research/r1-e1-reservoir-architecture`; the approved spec checkpoint is commit `b241f0abc04b023590a798f84a1b36eddaa411fc`.
- R1 and R2 remain independent. Do not modify or import `delayed_credit.py`, `action_value.py`, `action_value_benchmark.py`, any `phase3*` module, any `phase_c4*` module, `reward_learning.py`, `reward_readout.py`, or frozen R2 evidence.
- Do not modify `src/neural_state_machine/memory_probe.py`, `memory_task.py`, `memory_benchmark.py`, or `policy.py` for E1. Phase 2A is a frozen compatibility oracle.
- NumPy remains the only runtime dependency. AutoESN is provenance/reference only at `Ro6ertWcislo/AutoESN@b3d2e287716176fc3e1312b5be2a7a0b91ba538e`; do not add it to dependencies and do not copy `GreedyESN` behavior.
- Registered architectures are exactly `shallow`, `grouped2`, `grouped4`, `deep2`, `deep4`; registered budgets are exactly `64` and `256`; registered seeds are exactly `[7, 17, 29, 43, 61]`.
- Every recurrent component uses `tanh`, spectral radius `0.9`, leak `1.0`, no recurrent bias, zero initial state, float64, input weights `N(0, 1/sqrt(input_dim))`, recurrent weights `N(0, 1/sqrt(hidden_dim))`, then the same eigenvalue-radius normalization policy as `RecurrentPolicy`.
- Reservoir RNG lineage is exactly `SeedSequence([base_seed, 0x52314531, budget_id, architecture_id, component_id])` with `budget_id={64:0,256:1}`, `architecture_id={shallow:0,grouped2:1,grouped4:2,deep2:3,deep4:4}`, and zero-based component IDs.
- Fixture/noise lineages are exactly: E1-A train `[seed,0x45314154]`, E1-A eval `[seed,0x45314145]`, E1-B train `[seed,0x45314254]`, E1-B eval `[seed,0x45314245]`, E1-C train `[seed,0x45314354]`, E1-C eval `[seed,0x45314345]`, E1-C corruption `[seed,0x45314343]`.
- Fixture/noise RNG state must never be consumed by reservoir initialization; architecture and budget arms for a seed must receive byte-identical semantic fixtures/corruption plans.
- Ridge regularization is exactly `1e-6`; bias is enabled and unpenalized; fitting uses `np.linalg.lstsq`/SVD-style least squares, never an explicit inverse and never `solve(X.T @ X, ...)`.
- E1-A binary targets are `-1/+1`; prediction is class 1 only for score `> 0`, so an exact tie maps to class 0. E1-B/C use one-hot multiclass targets and `np.argmax`, so an exact tie maps to the lowest class index.
- E1-A delays are exactly `[1,2,5,10,20,40,80]`, with `400` training episodes and `40` evaluation episodes per delay. One binary probe is fitted to the combined 2,800-state training matrix and reused for every delay/control evaluation in that arm.
- E1-B histories are exactly `AB, BA, AA, BB`; tail horizons are exactly `[1,5,20,40]`; training is `200 paired nuisance realizations × 4 classes` per horizon and evaluation is `50 × 4` per horizon.
- E1-C behaviors are exactly `approach, touch, pick_up, pass_by`; training is `200 paired nuisance realizations × 4 classes`; evaluation is `50 × 4`; the clean-trained probe is frozen and reused for `clean`, `drop10`, `wrong10`, `occlusion4`, `jitter`, and `mixed`.
- E1-C corruption laws, donor cycle, frame indices, jitter dimensions, clipping bounds, and missing-observation token must match the spec exactly; do not reject or resample inconvenient masks.
- E1-A reset control must be exactly `140/280` overall and `20/40` at each delay, with each paired reset state byte-identical. E1-B reset control must be exactly `50/200` at each horizon, with each four-class group byte-identical after reset.
- The N=64 shallow implementation must pass an explicit Phase 2A compatibility path on delays 1–5 before any registered E1 measurement is authorized.
- No post-observation hyperparameter selection, architecture selection, threshold tuning, refitting, or metric reweighting is permitted.
- Registered evidence JSON is canonical sorted compact UTF-8, exactly one trailing newline, `allow_nan=False`, and measurement outputs are write-once.
- Permanent CI may run unit tests, protocol tests, compatibility tests, deterministic small-fixture smoke tests, and prospective-manifest verification. It must not execute the full registered E1 measurement or create registered `result.json`.
- RED–GREEN–REFACTOR for each behavior change. Every task ends with focused tests, relevant full tests, Ruff, and its own commit.

## File and responsibility map

| File | Responsibility | Change |
|---|---|---|
| `src/neural_state_machine/r1_e1_reservoir.py` | Reservoir enums/configuration, deterministic component initialization, shallow/grouped/deep evolution, reset, state width, parameter digest | Create |
| `tests/test_r1_e1_reservoir.py` | Architecture sizes, lineage isolation, recurrence equivalence, immutability, digest stability, reset behavior | Create |
| `src/neural_state_machine/r1_e1_probe.py` | Stable binary/multiclass Ridge probes, diagnostics, coefficient/prediction digests | Create |
| `tests/test_r1_e1_probe.py` | SVD/lstsq fit law, tie rules, unpenalized bias, immutability, multiclass behavior, diagnostics | Create |
| `src/neural_state_machine/r1_e1_memory.py` | E1-A fixtures, state collection, reset control, memory metrics, Phase 2A compatibility adapter | Create |
| `tests/test_r1_e1_memory.py` | E1-A counts, pairing, lineages, horizon metric, reset exactness, Phase 2A compatibility | Create |
| `src/neural_state_machine/r1_e1_history.py` | E1-B four-history fixtures, collection, reset control, margin/cosine metrics | Create |
| `tests/test_r1_e1_history.py` | E1-B event positions, nuisance pairing, counts, reset exactness, metric definitions | Create |
| `src/neural_state_machine/r1_e1_yolo_like.py` | E1-C clean templates, deterministic corruption plans, state collection, corruption metrics | Create |
| `tests/test_r1_e1_yolo_like.py` | Exact 20-frame templates, donor mapping, mask counts, jitter/clipping, paired corruption | Create |
| `src/neural_state_machine/r1_e1_protocol.py` | Registered constants, arm orchestration, validity gates, result schema, deterministic protocol smoke path | Create |
| `tests/test_r1_e1_protocol.py` | Full manifest constants, same-fixture cross-arm checks, R2 import isolation, validity failure behavior | Create |
| `src/neural_state_machine/r1_e1_evidence.py` | Canonical JSON, prospective manifest/provenance, write-once result bundle, verifier | Create |
| `tests/test_r1_e1_evidence.py` | Canonical encoding, manifest binding, write-once behavior, verifier tamper tests | Create |
| `scripts/benchmark_r1_e1_reservoir.py` | `protocol`, `prepare`, dormant `measure`, and `verify` commands | Create |
| `tests/test_r1_e1_cli.py` | CLI exit contracts; verify that ordinary protocol/prepare never run registered measurement | Create |
| `.github/workflows/ci.yml` | Add permanent R1 E1 tests/protocol/preflight only; no registered measurement | Modify |

Do not export E1 symbols through `neural_state_machine.__init__` in this phase. The experiment remains an internal research subsystem until evidence is frozen.

## Locked interfaces

```python
# r1_e1_reservoir.py
class ReservoirArchitecture(IntEnum):
    SHALLOW = 0
    GROUPED2 = 1
    GROUPED4 = 2
    DEEP2 = 3
    DEEP4 = 4

@dataclass(frozen=True)
class ReservoirSpec:
    architecture: ReservoirArchitecture
    budget: int
    input_size: int
    seed: int

class Reservoir:
    @property
    def state_dim(self) -> int: ...
    def reset(self) -> None: ...
    def advance(self, observation: np.ndarray) -> np.ndarray: ...
    def parameter_digest(self) -> str: ...

def build_reservoir(spec: ReservoirSpec) -> Reservoir: ...
```

```python
# r1_e1_probe.py
@dataclass(frozen=True)
class RidgeDiagnostics:
    train_count: int
    design_shape: tuple[int, int]
    rank: int
    singular_values: tuple[float, ...]

@dataclass(frozen=True)
class BinaryRidgeProbe:
    coefficients: np.ndarray
    bias: float
    diagnostics: RidgeDiagnostics
    def predict_scores(self, states: np.ndarray) -> np.ndarray: ...
    def predict(self, states: np.ndarray) -> np.ndarray: ...
    def coefficient_digest(self) -> str: ...

@dataclass(frozen=True)
class MulticlassRidgeProbe:
    coefficients: np.ndarray  # shape (class_count, feature_count)
    bias: np.ndarray          # shape (class_count,)
    diagnostics: RidgeDiagnostics
    def predict_scores(self, states: np.ndarray) -> np.ndarray: ...
    def predict(self, states: np.ndarray) -> np.ndarray: ...
    def coefficient_digest(self) -> str: ...

def fit_binary_ridge(states: np.ndarray, labels: np.ndarray, *, regularization: float = 1e-6) -> BinaryRidgeProbe: ...
def fit_multiclass_ridge(states: np.ndarray, labels: np.ndarray, *, class_count: int, regularization: float = 1e-6) -> MulticlassRidgeProbe: ...
```

The experiment modules return frozen dataclasses internally and serialize only Python scalars/lists/dicts through `r1_e1_protocol.py`; no NumPy arrays or NumPy scalar objects may escape into canonical evidence JSON.

---

### Task 1: Implement the deterministic reservoir architecture core

**Files:**
- Create: `src/neural_state_machine/r1_e1_reservoir.py`
- Create: `tests/test_r1_e1_reservoir.py`

**Interfaces:**
- Consumes: NumPy and `ReservoirSpec` only.
- Produces: the locked `ReservoirArchitecture`, `ReservoirSpec`, `Reservoir`, and `build_reservoir()` interfaces.
- No dependency on `RecurrentPolicy`; a compatibility test may compare trajectories against it, but the registered implementation must not wrap policy/output-learning state.

- [ ] **Step 1: Write RED tests for enum IDs, budgets, component widths, and `state_dim`**

Use a table asserting exact component layouts:

```python
@pytest.mark.parametrize(
    ("architecture", "budget", "widths"),
    [
        (ReservoirArchitecture.SHALLOW, 64, (64,)),
        (ReservoirArchitecture.GROUPED2, 64, (32, 32)),
        (ReservoirArchitecture.GROUPED4, 64, (16, 16, 16, 16)),
        (ReservoirArchitecture.DEEP2, 256, (128, 128)),
        (ReservoirArchitecture.DEEP4, 256, (64, 64, 64, 64)),
    ],
)
def test_registered_layouts_preserve_total_budget(architecture, budget, widths):
    reservoir = build_reservoir(ReservoirSpec(architecture, budget, 4, 7))
    assert reservoir.component_widths == widths
    assert reservoir.state_dim == budget
```

Also reject every nonregistered budget, nonpositive input width, invalid seed, and architecture/budget combination that is not evenly divisible.

- [ ] **Step 2: Run the focused tests and verify RED**

```bash
pytest -q tests/test_r1_e1_reservoir.py
```

Expected: import/collection failure because the module does not exist.

- [ ] **Step 3: Implement one private recurrent component and the five topology compositions**

The private component update is exactly:

```python
next_state = np.tanh(input_weights @ observation + recurrent_weights @ state)
```

Grouped arms feed the same external observation to every component and concatenate in component order. Deep arms feed layer 1 the external observation, each later layer the *current* output of the prior layer, then concatenate every current layer state in layer order.

For component `i`, initialize only from:

```python
rng = np.random.default_rng(np.random.SeedSequence([
    spec.seed,
    0x52314531,
    budget_id(spec.budget),
    int(spec.architecture),
    i,
]))
```

Generate input weights first and recurrent weights second. Normalize recurrent weights by `max(abs(eigvals(W)))` to exactly `0.9`, matching current baseline policy. Returned states are defensive, C-contiguous, float64, and non-writeable.

- [ ] **Step 4: Add RED tests for exact RNG isolation and parameter digests**

Assert that changing only architecture or budget changes the parameter digest, rebuilding an identical spec produces the same digest/state trajectory, and fixture RNG use before/after reservoir construction cannot change reservoir parameters.

The digest must bind a version tag, architecture integer, budget, input size, seed, ordered component widths, and the float64 bytes/shapes of every input/recurrent matrix.

- [ ] **Step 5: Add reset and immutability tests**

Advance several frames, call `reset()`, feed the same next frame to two independently built identical reservoirs, and require byte-identical output. Mutating source observations or attempted mutation of returned states must not mutate reservoir state/parameters.

- [ ] **Step 6: Add recurrence-equivalence regression for one shallow component**

In the test only, reconstruct the baseline initialization law for a chosen seed/input/hidden width and compare one-step/multi-step outputs to the same recurrence used by `RecurrentPolicy`. This guards the mathematical cell law without making E1 depend on policy learning/output weights.

- [ ] **Step 7: Run focused tests and Ruff**

```bash
pytest -q tests/test_r1_e1_reservoir.py
ruff check src/neural_state_machine/r1_e1_reservoir.py tests/test_r1_e1_reservoir.py
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/neural_state_machine/r1_e1_reservoir.py tests/test_r1_e1_reservoir.py
git commit -m "feat: add R1 E1 reservoir architectures"
```

---

### Task 2: Implement stable frozen binary and multiclass Ridge probes

**Files:**
- Create: `src/neural_state_machine/r1_e1_probe.py`
- Create: `tests/test_r1_e1_probe.py`

**Interfaces:**
- Consumes: immutable float64 state matrices and integer labels.
- Produces: the locked binary/multiclass probe types and diagnostics.

- [ ] **Step 1: Write RED tests for validation, exact tie rules, defensive copies, and coefficient shapes**

Binary prediction must use:

```python
choices = np.where(scores > 0.0, 1, 0)
```

Multiclass prediction must use:

```python
choices = np.argmax(scores, axis=1)
```

Include exact-zero/tied score rows proving class 0 wins.

- [ ] **Step 2: Run focused tests and verify RED**

```bash
pytest -q tests/test_r1_e1_probe.py
```

- [ ] **Step 3: Implement Ridge with an augmented least-squares system**

Build `design = column_stack([states, ones])`. Penalize features but not bias by appending regularization rows:

```python
p = states.shape[1]
penalty = np.zeros((p, p + 1), dtype=np.float64)
penalty[:, :p] = np.sqrt(regularization) * np.eye(p)
augmented_design = np.vstack((design, penalty))
```

For binary targets use `-1/+1` plus `p` trailing zeros. For multiclass use one-hot targets plus a `(p, class_count)` zero block. Fit with `np.linalg.lstsq(augmented_design, augmented_target, rcond=None)`; never form `X.T @ X` and never call matrix inverse.

Compute diagnostic singular values/rank from the unregularized design matrix with deterministic NumPy operations, and copy all stored parameters into non-writeable float64 arrays.

- [ ] **Step 4: Add tests that bias is actually unpenalized**

Fit a constant-offset dataset with a large test-only regularization value and compare against a directly constructed augmented least-squares reference. Require coefficients/bias to match the reference, proving the final column is not penalized.

- [ ] **Step 5: Add coefficient/prediction digest tests**

Use SHA-256 over version tag, shape metadata, and canonical C-order float64/int64 bytes. The same fit/data must reproduce the same digest; one changed coefficient or prediction must change it.

- [ ] **Step 6: Run focused tests and Ruff**

```bash
pytest -q tests/test_r1_e1_probe.py
ruff check src/neural_state_machine/r1_e1_probe.py tests/test_r1_e1_probe.py
```

- [ ] **Step 7: Commit**

```bash
git add src/neural_state_machine/r1_e1_probe.py tests/test_r1_e1_probe.py
git commit -m "feat: add R1 E1 frozen ridge probes"
```

---

### Task 3: Implement E1-A memory curve and the Phase 2A compatibility gate

**Files:**
- Create: `src/neural_state_machine/r1_e1_memory.py`
- Create: `tests/test_r1_e1_memory.py`

**Interfaces:**
- Consumes: `build_reservoir`, `fit_binary_ridge`, fixed E1-A RNG lineages, and the frozen Phase 2A `run_memory_probe` only in compatibility code.
- Produces: deterministic E1-A datasets/results and `run_phase2a_compatibility()`.

- [ ] **Step 1: Write RED tests locking the E1-A fixture law and counts**

Use the Phase 2A observation semantics for E1-A: cue frame `[1,0,0,0]` or `[0,1,0,0]`; each delay frame `[0,0,u,0]` with `u ~ Uniform(-0.25, 0.25)`; decision frame `[0,0,0,1]`. Extend only the allowed delays to `[1,2,5,10,20,40,80]`.

Construct balanced paired blocks so each delay has exactly 200 LEFT + 200 RIGHT training episodes and 20 LEFT + 20 RIGHT evaluation episodes. Labels stay outside the reservoir input path.

- [ ] **Step 2: Implement immutable E1-A fixtures and fixture digest**

Training RNG is exactly `SeedSequence([seed,0x45314154])`; evaluation RNG is exactly `SeedSequence([seed,0x45314145])`. Build each seed's fixtures once, then reuse the same immutable objects for every architecture/budget arm.

The fixture digest binds episode order, cue label, delay, and all input-vector bytes; it must be identical across architecture/budget arms for a seed.

- [ ] **Step 3: Write RED tests for collection, single-probe fitting, and reset controls**

Collect terminal states from all 2,800 training fixtures, fit exactly one binary probe, and evaluate all 280 evaluation fixtures. For reset control, process cue/delay history, reset immediately before the shared decision frame, then advance only the decision frame. For every left/right pair at a delay, reset terminal states must be byte-identical.

- [ ] **Step 4: Implement memory metrics**

Return per-delay `correct/total/accuracy`, macro accuracy, delay-1 degradation, training accuracy, and contiguous 85% horizon. Define the horizon with the registered ordered delays; stop at the first delay below `0.85`, and return `None` if delay 1 is below threshold.

- [ ] **Step 5: Implement exact reset validity gate**

Reject the arm as protocol-invalid unless reset predictions equal `140/280` overall and exactly `20/40` at every delay, paired reset states are byte-identical, parameter digest is unchanged before/after collection/fitting, and all required digests exist.

- [ ] **Step 6: Add the separate Phase 2A compatibility adapter**

Do not change the registered E1 RNG lineage to mimic old seeds. Instead, make compatibility an explicit frozen-baseline path:

```python
def run_phase2a_compatibility(seed: int) -> Phase2ACompatibility:
    legacy = run_memory_probe(seed, MemoryProbeConfig())
    # Re-run the legacy fixture/readout law through a shallow compatibility
    # recurrence and compare per-delay/reset/prediction behavior.
    ...
```

The compatibility recurrence may reconstruct the legacy `default_rng(seed)` input/recurrent initialization *only inside this compatibility path*. Require hidden-state/prediction equivalence to the frozen Phase 2A law on delays 1–5. This path is not an E1 performance arm and its weights/results must never enter E1 architecture comparisons.

For the historical acceptance seeds `7,17,29`, also require the frozen Phase 2A acceptance result to remain passing. Seeds `43,61` may be exercised for deterministic equivalence but must not retroactively redefine Phase 2A's historical three-seed acceptance contract.

- [ ] **Step 7: Run focused and frozen-regression tests**

```bash
pytest -q tests/test_r1_e1_memory.py tests/test_memory_probe.py
ruff check src/neural_state_machine/r1_e1_memory.py tests/test_r1_e1_memory.py
```

- [ ] **Step 8: Commit**

```bash
git add src/neural_state_machine/r1_e1_memory.py tests/test_r1_e1_memory.py
git commit -m "feat: add R1 E1 memory curve protocol"
```

---

### Task 4: Implement E1-B multi-event history separability

**Files:**
- Create: `src/neural_state_machine/r1_e1_history.py`
- Create: `tests/test_r1_e1_history.py`

**Interfaces:**
- Consumes: `build_reservoir`, `fit_multiclass_ridge`, E1-B fixed RNG lineages.
- Produces: four-class terminal-state datasets, reset control, separability metrics.

- [ ] **Step 1: Write RED tests for exact history encoding**

Use six input channels: first two are the event channels and last four are nuisance. Event A is `[1,0]`, B is `[0,1]`, neutral is `[0,0]`; first event is step 0, step 1 is neutral, second event is step 2, then append exactly `H` neutral-tail frames for `H in [1,5,20,40]`.

For each paired realization, generate one nuisance stream shared by all four classes, each nuisance value independently chosen from `{-0.25,+0.25}`. Assert class order `0=AB,1=BA,2=AA,3=BB`.

- [ ] **Step 2: Implement train/evaluation fixture builders and digests**

For each horizon generate 200 paired training realizations (800 sequences) and 50 paired evaluation realizations (200 sequences) using only the registered E1-B lineages. The same seed/horizon fixture bytes must be reused across all architecture/budget arms.

- [ ] **Step 3: Implement terminal-state collection and one multiclass probe per horizon**

For a given architecture/budget/seed/horizon, fit the probe only on that horizon's 800 training states and score its 200 evaluation states. This preserves the spec's `per architecture/budget/seed/horizon` count contract and avoids adding an unregistered horizon feature to the readout.

- [ ] **Step 4: Implement the reset negative control**

For reset evaluation, process through the second event, reset immediately after it, then feed only the neutral/nuisance tail. Within each paired four-class group, require byte-identical terminal states. With deterministic lowest-index tie handling and balanced class order, require exactly `50/200` correct for every horizon.

- [ ] **Step 5: Implement separability metrics**

For each horizon compute accuracy, normalized correct-class margin, within-class cosine-distance summary, and between-class cosine-distance summary. Define normalized correct-class margin for sample `i` as:

```python
margin = (score_true - max(score_other)) / max(np.linalg.norm(scores), np.finfo(np.float64).eps)
```

Define cosine distance with zero-norm protection; summarize deterministic `count/min/median/mean/max`. Compute macro accuracy over four horizons and contiguous 80% horizon using `[1,5,20,40]` in order.

- [ ] **Step 6: Add invalidity tests**

Fail protocol validity, rather than returning a performance result, for reset inequality, wrong counts, missing digests, state width mismatch, changed reservoir parameters, or nonfinite metric values.

- [ ] **Step 7: Run focused tests and Ruff**

```bash
pytest -q tests/test_r1_e1_history.py
ruff check src/neural_state_machine/r1_e1_history.py tests/test_r1_e1_history.py
```

- [ ] **Step 8: Commit**

```bash
git add src/neural_state_machine/r1_e1_history.py tests/test_r1_e1_history.py
git commit -m "feat: add R1 E1 history separability protocol"
```

---

### Task 5: Implement E1-C synthetic YOLO-like corruption fixtures

**Files:**
- Create: `src/neural_state_machine/r1_e1_yolo_like.py`
- Create: `tests/test_r1_e1_yolo_like.py`

**Interfaces:**
- Consumes: `build_reservoir`, `fit_multiclass_ridge`, E1-C train/eval/corruption lineages.
- Produces: exact clean 20-frame templates, paired corruption plans, robustness metrics.

- [ ] **Step 1: Write RED golden-vector tests for all four clean templates**

Assert every frame/dimension against spec formulas for `approach`, `touch`, `pick_up`, and `pass_by`; assert frames 16–19 share the neutral suffix structure; assert no class index enters the 9-D feature vector; assert nuisance values are only `-0.05/+0.05` and shared across all four classes in a paired realization.

- [ ] **Step 2: Implement clean train/eval fixture builders**

Generate 200 paired clean training realizations (800 sequences) and 50 paired clean evaluation realizations (200 sequences). Keep train/evaluation nuisance RNG isolated using `[seed,0x45314354]` and `[seed,0x45314345]`.

- [ ] **Step 3: Write RED tests for every corruption arm**

For each evaluation paired group, use only `[seed,0x45314343]` to precompute the full corruption plan before any architecture executes:

- `drop10`: exactly two distinct indices in `0..15`, all-zero token;
- `wrong10`: exactly two distinct indices in `0..15`, fixed donor cycle `approach→touch→pick_up→pass_by→approach`, preserving paired nuisance values;
- `occlusion4`: exactly frames `8,9,10,11` replaced by all-zero token;
- `jitter`: Gaussian `sigma=0.05` only on dims `0,2,3,4,5,7,8`, clipping `0,2,3,4,5` to `[0,1]` and `7,8` to `[-0.1,0.1]`, never changing dims `1,6`;
- `mixed`: exactly two distinct dropped frames, one distinct wrong-donor frame from the remaining indices, and the same jitter law on nonmissing frames.

No corruption may touch frames 16–19 except jitter is also excluded there by the spec's behavior-bearing-frame boundary.

- [ ] **Step 4: Implement corruption-plan and fixture digests**

Digest the clean evaluation fixture separately from the corruption plan. The same digests for a seed must be observed by every architecture/budget arm.

- [ ] **Step 5: Implement clean-only fitting and frozen corrupted evaluation**

Fit exactly one four-class probe on 800 clean training terminal states for an architecture/budget/seed. Freeze it. Evaluate the same probe on 200 clean evaluation states and on all five corrupted versions of those same paired evaluation fixtures; never refit on corruption.

- [ ] **Step 6: Implement robustness metrics**

Return clean accuracy; per-corruption correct/total/accuracy; absolute drop `clean_accuracy - corrupted_accuracy`; macro corrupted accuracy over exactly five nonclean arms; worst registered corrupted-arm accuracy; and a 4×4 integer confusion matrix for each arm in class order `approach,touch,pick_up,pass_by`.

- [ ] **Step 7: Add protocol-invalid tests**

Reject wrong mask cardinality, donor mismatch, changed nuisance values in wrong-donor replacement, jitter on binary dimensions, out-of-bound clipping, changed reservoir parameters, state width mismatch, or architecture-dependent corruption digests.

- [ ] **Step 8: Run focused tests and Ruff**

```bash
pytest -q tests/test_r1_e1_yolo_like.py
ruff check src/neural_state_machine/r1_e1_yolo_like.py tests/test_r1_e1_yolo_like.py
```

- [ ] **Step 9: Commit**

```bash
git add src/neural_state_machine/r1_e1_yolo_like.py tests/test_r1_e1_yolo_like.py
git commit -m "feat: add R1 E1 YOLO-like corruption protocol"
```

---

### Task 6: Build the registered protocol orchestrator and hard R1/R2 isolation gates

**Files:**
- Create: `src/neural_state_machine/r1_e1_protocol.py`
- Create: `tests/test_r1_e1_protocol.py`

**Interfaces:**
- Consumes: E1 reservoir/probe/A/B/C modules only.
- Produces: registered arm specs, deterministic protocol-smoke results, full-measurement raw result objects, validity-gate decisions.

- [ ] **Step 1: Encode every registered constant in one immutable manifest source**

Define exact tuples/maps for architectures, budgets, seeds, delays, horizons, behavior classes, corruption names, lineage tags, radius/leak/activation/ridge, sample counts, and threshold definitions. Tests must compare the full public manifest payload to a literal expected dictionary so a changed constant cannot silently enter CI.

- [ ] **Step 2: Add a cross-arm fixture-isolation test**

For one small protocol-only seed, monkeypatch/spy collection so the orchestrator proves all ten architecture×budget arms consume the same E1-A/B/C fixture digests and E1-C corruption digest while having distinct registered reservoir parameter digests.

- [ ] **Step 3: Add an AST import-isolation gate**

Parse every `src/neural_state_machine/r1_e1_*.py` file and fail if it imports any R2/reward module, including module names containing `delayed_credit`, `action_value`, `phase3`, `phase_c4`, `reward_learning`, `reward_readout`, or `learning_diagnostics`. Allow only the explicit compatibility import of `memory_probe` from `r1_e1_memory.py`.

- [ ] **Step 4: Implement protocol validity as fail-closed status, not score substitution**

Create a frozen `ValidityIssue(code, detail)` and require the orchestrator to return/raise protocol-invalid status for any spec Section 13 gate failure. Never fill a missing/invalid result with zero accuracy and never include it in architecture comparison summaries.

- [ ] **Step 5: Implement deterministic `protocol_smoke()` using reduced counts only**

The smoke path may use tiny test-only counts injected through private fixture-builder arguments, but all scientific constants (architectures, topology law, RNG tags, feature schemas, corruption laws, tie rules) remain unchanged. Its purpose is deterministic CI coverage; its output must be tagged `registered_measurement: false` and can never be written to the evidence directory.

- [ ] **Step 6: Implement the full registered runner as callable but do not execute it**

`run_registered_measurement()` must hard-code registered counts/seeds/arms and reject count/config overrides. It must begin by running the Phase 2A compatibility gate and all manifest validity checks. This function is implementation only at this task; no CI/test may invoke the full five-seed measurement.

- [ ] **Step 7: Run focused tests and Ruff**

```bash
pytest -q tests/test_r1_e1_protocol.py
ruff check src/neural_state_machine/r1_e1_protocol.py tests/test_r1_e1_protocol.py
```

- [ ] **Step 8: Commit**

```bash
git add src/neural_state_machine/r1_e1_protocol.py tests/test_r1_e1_protocol.py
git commit -m "feat: add R1 E1 registered protocol gates"
```

---

### Task 7: Implement prospective sealing, write-once evidence, and verification

**Files:**
- Create: `src/neural_state_machine/r1_e1_evidence.py`
- Create: `tests/test_r1_e1_evidence.py`
- Create: `scripts/benchmark_r1_e1_reservoir.py`
- Create: `tests/test_r1_e1_cli.py`

**Interfaces:**
- Consumes: protocol manifest and raw registered result payload.
- Produces: `protocol`, `prepare`, dormant `measure`, `verify` commands and evidence bundle schema.

- [ ] **Step 1: Implement canonical JSON bytes and tests**

Use exactly:

```python
def canonical_json_bytes(payload: object) -> bytes:
    text = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return (text + "\n").encode("utf-8")
```

Reject NaN/Inf through `allow_nan=False`. SHA-256 all sealed JSON bytes.

- [ ] **Step 2: Implement `prepare` without measurement**

`prepare --output DIR --scientific-head SHA` writes only a prospective manifest/provenance set and its digests. The manifest binds at minimum exact scientific head, base integration head from the spec, architecture enum/definitions, budgets, seeds, all RNG tags/enums, reservoir dynamics, Ridge law, E1-A/B/C fixtures/counts/metrics, Python/NumPy versions, and AutoESN reference commit.

`prepare` must not import/call `run_registered_measurement`, fit a probe, collect reservoir states, or create `result.json`/`report.md`.

- [ ] **Step 3: Implement write-once measurement bundle API**

Only `measure` may create `result.json`, `provenance.json`, and `trace-index.json`, and only when a caller supplies the expected prospective manifest SHA-256 and the current scientific head matches the manifest. Use exclusive-create file semantics (`"xb"` or equivalent) and fail if any target result file already exists.

Large traces are optional artifacts; if produced, `trace-index.json` stores only artifact name, byte size, and SHA-256. Do not store a second mutable copy under the evidence root.

- [ ] **Step 4: Keep human interpretation separate**

Do not auto-generate a winner or Outcome A–E classification. `measure` produces raw result/provenance only. `report.md` remains absent until a later human interpretation/freeze step after registered measurement review.

- [ ] **Step 5: Implement a fail-closed verifier**

`verify --root DIR --no-result-ok` validates prospective manifest/provenance before measurement. Without `--no-result-ok`, require result/provenance/trace-index, canonical encoding, expected hashes, exact manifest linkage, complete arm counts, complete required digests, all validity gates, and no unregistered architecture/budget/seed/corruption names.

- [ ] **Step 6: Implement CLI command boundaries**

Commands:

```text
python scripts/benchmark_r1_e1_reservoir.py protocol
python scripts/benchmark_r1_e1_reservoir.py prepare --output <dir> --scientific-head <sha>
python scripts/benchmark_r1_e1_reservoir.py measure --root <dir> --manifest-sha256 <sha>
python scripts/benchmark_r1_e1_reservoir.py verify --root <dir> [--no-result-ok]
```

Tests monkeypatch `run_registered_measurement` to raise if `protocol`, `prepare`, or `verify --no-result-ok` attempts to call it. The `measure` command is tested with a tiny injected fake raw result, never the real full runner.

- [ ] **Step 7: Run focused tests and Ruff**

```bash
pytest -q tests/test_r1_e1_evidence.py tests/test_r1_e1_cli.py
ruff check src/neural_state_machine/r1_e1_evidence.py scripts/benchmark_r1_e1_reservoir.py tests/test_r1_e1_evidence.py tests/test_r1_e1_cli.py
```

- [ ] **Step 8: Commit**

```bash
git add src/neural_state_machine/r1_e1_evidence.py scripts/benchmark_r1_e1_reservoir.py tests/test_r1_e1_evidence.py tests/test_r1_e1_cli.py
git commit -m "feat: add R1 E1 evidence sealing"
```

---

### Task 8: Add permanent CI gates without authorizing registered measurement

**Files:**
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: Tasks 1–7 tests and CLI.
- Produces: permanent exact-head implementation/protocol evidence only.

- [ ] **Step 1: Add a focused R1 E1 CI test step**

Add after the existing delayed-cue linear memory probe:

```yaml
      - name: R1 E1 reservoir protocol tests
        run: >-
          pytest -q
          tests/test_r1_e1_reservoir.py
          tests/test_r1_e1_probe.py
          tests/test_r1_e1_memory.py
          tests/test_r1_e1_history.py
          tests/test_r1_e1_yolo_like.py
          tests/test_r1_e1_protocol.py
          tests/test_r1_e1_evidence.py
          tests/test_r1_e1_cli.py

      - name: R1 E1 protocol smoke
        run: python scripts/benchmark_r1_e1_reservoir.py protocol
```

Do not add a `measure` invocation.

- [ ] **Step 2: Add prospective-manifest creation/verification only if the head is exact**

Use a temporary directory and current `${{ github.sha }}`:

```yaml
      - name: Prepare prospective R1 E1 manifest
        shell: bash
        run: |
          set -euo pipefail
          rm -rf /tmp/r1-e1-prospective
          python scripts/benchmark_r1_e1_reservoir.py prepare \
            --output /tmp/r1-e1-prospective \
            --scientific-head "${GITHUB_SHA}"
          python scripts/benchmark_r1_e1_reservoir.py verify \
            --root /tmp/r1-e1-prospective \
            --no-result-ok
```

Do not persist this as the final registered manifest until implementation review explicitly authorizes the prospective sealing checkpoint.

- [ ] **Step 3: Run local workflow-equivalent commands**

```bash
pytest -q tests/test_r1_e1_*.py
python scripts/benchmark_r1_e1_reservoir.py protocol
ruff check src/neural_state_machine/r1_e1_*.py scripts/benchmark_r1_e1_reservoir.py tests/test_r1_e1_*.py
pytest -q
ruff check .
```

Expected: PASS. Confirm no `docs/experiments/r1-e1-reservoir-architecture/result.json` was created.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: gate R1 E1 reservoir protocol"
```

---

### Task 9: Exact-head implementation review gate and prospective measurement handoff

**Files:**
- No production/test/workflow changes unless review finds a defect.
- Later, only after explicit human authorization, create `docs/experiments/r1-e1-reservoir-architecture/manifest.json` and associated prospective provenance from the reviewed exact head.

**Interfaces:**
- Consumes: exact implementation head from Tasks 1–8 and green permanent CI.
- Produces: a review decision and, only after separate authorization, the prospective registered manifest. It does **not** produce registered measurements during this task.

- [ ] **Step 1: Verify the branch diff preserves frozen R2 files**

Run:

```bash
git diff b241f0abc04b023590a798f84a1b36eddaa411fc..HEAD --name-only
```

Expected changed implementation paths are only the new `r1_e1_*` modules/tests/script plus `.github/workflows/ci.yml`; the approved spec and all R2 production/evidence files remain unchanged.

- [ ] **Step 2: Re-run the complete local gate**

```bash
pytest -q
ruff check .
python scripts/benchmark_memory_probe.py
python scripts/benchmark_r1_e1_reservoir.py protocol
```

Expected: all pass. The historical Phase 2A benchmark remains green.

- [ ] **Step 3: Require exact-head GitHub CI green**

Do not proceed from an older green run. Record the implementation head SHA and require the permanent `ci` workflow for that exact SHA to pass.

- [ ] **Step 4: Review scientific invariants before sealing**

Confirm from tests/output that: all ten architecture×budget arms have `state_dim=N`; parameter digests are stable and unchanged; all fixture/corruption digests are arm-independent where required; E1-A/B reset controls are exact; Phase 2A compatibility passes; protocol smoke is deterministic; no R2 imports exist; no registered measurement has run.

- [ ] **Step 5: Stop at the mandatory human measurement-approval gate**

The next allowed sequence is:

```text
exact-head implementation CI green
→ prospective manifest/preflight at that exact head
→ explicit human approval to measure
→ one registered measurement
→ human interpretation review
→ no-refit evidence freeze
→ remove any one-shot measurement workflow
→ final exact-head CI
```

Do not create or trigger a registered measurement workflow in advance. Do not interpret protocol-smoke numbers as science.

## Plan self-review

- Spec coverage: architecture definitions, budgets, all RNG lineages, fixed dynamics, binary/multiclass Ridge laws, E1-A/B/C fixtures/counts/controls/metrics, Phase 2A compatibility, validity gates, evidence format, CI/measurement boundary, AutoESN reference-only status, and R2 freeze are each mapped to explicit tasks.
- Placeholder scan: no TBD/TODO/"implement later" steps remain. The only intentionally deferred action is the spec-mandated registered measurement itself, guarded by explicit human approval.
- Type consistency: all tasks use `ReservoirSpec`/`Reservoir`, `BinaryRidgeProbe`, and `MulticlassRidgeProbe` with the signatures locked above; experiment outputs remain internal frozen dataclasses and serialize through one protocol/evidence boundary.
- Scope check: E1-A, E1-B, and E1-C are separable modules but share the same reservoir/probe/evidence core and together constitute one preregistered scientific experiment. R2 and later R1 E2/E3 remain outside this plan.
