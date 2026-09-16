# Phase C4 Delay-Marginalized Anonymous Credit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and register Phase C4 as a two-stage experiment that first tests anonymous batch operator sufficiency and, only if that passes, tests a prospectively frozen online delay-marginalized credit learner without exposing realized reward-source metadata.

**Architecture:** Extend the proven Phase 3C anonymous-credit theory with a separate Lean `MarginalizedTemporalCredit` module, bind the exact passing theorem head into C4, then add additive Python modules for a fixed delay-law observation operator, the C4-A batch ridge probe, the C4-B online learner, fail-closed protocol controls, orchestration, and strict evidence handling. Both C4-A and C4-B implementations are completed and frozen **before the first C4-A registered measurement** so the C4-A result cannot tune C4-B. C4-A and C4-B then use separate sealed-environment measurement checkpoints and one-shot workflows.

**Tech Stack:** Lean 4.32.0, Mathlib v4.32.0, Lake, Python 3.12.14 for registered measurement, NumPy 2.5.3, pytest 9.1.1, Ruff 0.15.22, setuptools 84.0.0, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-16-phase-c4-delay-marginalized-anonymous-credit-design.md`

## Global Constraints

- Planning base is exact `qigao/yolo-motion-perception@f018384a6bdc25bd6b20dd393a7058df04912f08`; implementation starts on a new branch `experiment/phase-c4-delay-marginalized-credit` from that exact commit after execution approval.
- Formal work starts from exact proven Phase 3C formal head `qigao/lean@3de297cee2a94a7fc309531334f720f1b34467c9` on branch `formal/marginalized-temporal-credit-v1`.
- Do not modify `qigao/lean/NarrativeDynamics/Core/AnonymousTemporalCredit.lean`, `NarrativeDynamics/Core/TemporalCredit.lean`, or their existing tests. C4 adds a new module.
- Do not modify these C3 scientific files: `src/neural_state_machine/action_value.py`, `src/neural_state_machine/phase3c_schedule.py`, `src/neural_state_machine/phase3c_learners.py`, `src/neural_state_machine/phase3c_controls.py`, `src/neural_state_machine/phase3c_benchmark.py`, `src/neural_state_machine/reward_learning.py`, `src/neural_state_machine/action_value_benchmark.py`, or `scripts/verify_phase3c_anonymous_credit.py`.
- Do not modify any existing Phase 3C evidence or `docs/experiments/phase-3c-failure-attribution-v1/**`.
- Registered seeds remain exactly `(7, 17, 29)`; training decisions `2000`; evaluation blocks `20`; hidden size `64`; recurrent radius `0.9`; step size `0.1`; cue delays `(1,2,3,4,5)`; hidden feedback delays `(1,3,5)` with public prior `(1/3,1/3,1/3)`.
- Reuse the frozen training/evaluation fixture lineages, action RNG lineage `[seed, 0x33414354]`, hidden-delay lineage `[seed, 0x3343444C]`, and original shuffled-control lineage `[seed, 0x33534846]`.
- Primary behavior thresholds remain exactly: normal `>=180/200`, every cue delay `>=34/40`, reset exactly `100/200` and each reset delay exactly `20/40`, shuffled `<150/200`.
- The eight Task 12 additional evaluation sets use `PCG64(SeedSequence([seed, 0x33434641, 2, evaluation_id]))`, IDs `0..7`; they are secondary stability surfaces only and never select a model, parameter, threshold, or checkpoint.
- C4-A ridge penalty is exactly `1e-6`, including action-block bias coordinates. No scaling, search, feature selection, checkpoint selection, or evaluation-label fitting.
- C4-B must use current pre-update weights for `P_t=<W_t,Z_t>`; it must not store or use a prediction trace or stale decision-time prediction.
- The normal C4-B credit support is exactly lags `(1,3,5)`. The learner receives one finite scalar per delivery clock and no realized source ID, delay, due step, multiplicity, latent reward, or queue metadata.
- Terminal drain uses the same C4 observation/update equation, adds no synthetic action, and stops after the last possible registered delay.
- Protocol control `{0:1}` must match frozen Phase 3A in the same environment exactly: actions, scalar update fields, full weight bytes, and parameter digest. No tolerance is allowed for this check.
- For exact `{0:1}` Python continuity, preserve the Phase 3A arithmetic order `step_size * td_error * feature / denominator` on the selected row; do not substitute a mathematically equivalent matrix expression with different floating-point operation order.
- C4-B code, C4-A code, formal binding, thresholds, lineages, and registered config must all be frozen before the first C4-A measurement. After observing C4-A, no scientific C4-B code change is permitted inside C4 v1.
- Gate F failure stops Python scientific execution. Gate P failure means `harness invalid`; do not inspect or serialize C4-A/C4-B registered behavior.
- C4-A failure records `operator_passed=false` and terminates C4 v1 before C4-B registered measurement. Do not tune in place.
- C4-A success does not authorize C4-B automatically. A second explicit human approval is required after a separately sealed C4-B manifest is reviewed.
- Never claim Lean proves NumPy floating-point execution, batch identifiability in general, convergence, source reconstruction, or production readiness.

## File Ownership Map

### `qigao/lean`

| Path | Responsibility |
|---|---|
| `NarrativeDynamics/Core/MarginalizedTemporalCredit.lean` | Finite delay law, candidate support, weighted aggregate observation, current-weight prediction, bounded credit, boundary truncation, update equation, immediate reduction. |
| `NarrativeDynamics/Tests/MarginalizedTemporalCredit.lean` | F1-F6 examples, registered `{1,3,5}` law, `{0}` immediate law, boundary/drain examples, axiom audit. |
| `NarrativeDynamics.lean` | Export the new C4 module. |
| `.github/workflows/proof.yml` | Permanent focused C4 theorem test before full Lean build. |

### `qigao/yolo-motion-perception`

| Path | Responsibility |
|---|---|
| `docs/experiments/phase-c4-delay-marginalized-credit/formal-contract.json` | Exact passing C4 Lean SHA and theorem binding. |
| `src/neural_state_machine/phase_c4_delay_model.py` | Fixed delay laws, learner-owned decision rows, candidate indices, `Z_t`, `C_t`, current-weight `P_t`. |
| `src/neural_state_machine/phase_c4_batch_probe.py` | C4-A design matrix, fixed ridge solve, fit diagnostics; no evaluation labels/source metadata. |
| `src/neural_state_machine/phase_c4_learner.py` | C4-B online learner with five-step history plus one current pending decision. |
| `src/neural_state_machine/phase_c4_controls.py` | Formal-contract loader, frozen-input hashes, protocol audit, information-boundary checks, immediate continuity. |
| `src/neural_state_machine/phase_c4_benchmark.py` | Fixed training streams, C4-A/C4-B protocol-only orchestration, evaluation bundles, normal/shuffled execution, fixed gates. |
| `src/neural_state_machine/phase_c4_evidence.py` | Strict stage-specific schemas, canonical JSON, prospective manifest and result validation. |
| `scripts/benchmark_phase_c4_delay_marginalized_credit.py` | Protocol CLI and explicitly authorized stage measurement entry point. |
| `scripts/verify_phase_c4_delay_marginalized_credit.py` | Fail-closed formal/manifest/result verifier, including no-result mode. |
| `requirements/phase-c4.in` | Direct C4 scientific/test tool anchors. |
| `requirements/phase-c4-python312.lock` | Exact Python 3.12.14 registered measurement environment. |
| `.github/workflows/ci.yml` | Permanent C4 tests, Ruff, protocol-only gate, lock recreation, prospective manifest preparation. |
| `.github/workflows/phase-c4a-measurement-once.yml` | Created only after explicit C4-A measurement approval; removed after valid/invalid attempt is frozen. |
| `.github/workflows/phase-c4b-measurement-once.yml` | Created only after C4-A passes and explicit C4-B measurement approval; removed after measurement. |
| `tests/test_phase_c4_delay_model.py` | Pure operator and boundary tests. |
| `tests/test_phase_c4_batch_probe.py` | Fixed ridge and information-boundary tests. |
| `tests/test_phase_c4_learner.py` | Online arithmetic, current-weight prediction, bounded history, drain, exact immediate reduction. |
| `tests/test_phase_c4_controls.py` | Formal contract, frozen inputs, fail-closed Gate P mutations, non-interference. |
| `tests/test_phase_c4_benchmark.py` | Matched lineages, protocol anti-peeking, secondary evaluation lineage, fixed gate. |
| `tests/test_phase_c4_evidence.py` | Manifest/result/writer/verifier mutation tests. |

---

### Task 1: Formalize finite delay-law observation and bounded candidate support

**Repository:** `qigao/lean`

**Branch:** `formal/marginalized-temporal-credit-v1` from exact `3de297cee2a94a7fc309531334f720f1b34467c9`.

**Files:**
- Create: `NarrativeDynamics/Core/MarginalizedTemporalCredit.lean`
- Create: `NarrativeDynamics/Tests/MarginalizedTemporalCredit.lean`
- Frozen: `NarrativeDynamics/Core/AnonymousTemporalCredit.lean`

**Interfaces:**
- Imports: `NarrativeDynamics.Core.AnonymousTemporalCredit`
- Produces namespace: `NarrativeDynamics.MarginalizedTemporalCredit`
- Produces: `DelayLaw`, `registeredLaw`, `immediateLaw`, `validSource`, `expectedAggregate`, `candidateCredit`, `candidate_support_bounded`, `registered_law_valid`, `immediate_law_valid`, `expected_aggregate_decomposition`, `boundary_truncation`

- [ ] **Step 1: Write RED theorem tests for the public C4 formal surface.**

Create the test module first and require these exact theorem names:

```lean
import NarrativeDynamics.Core.MarginalizedTemporalCredit

open Finset
namespace NarrativeDynamics.MarginalizedTemporalCredit

example : registeredLaw.support = {1, 3, 5} := by
  native_decide

example : registeredLaw.weight 1 = (1 : ℝ) / 3 := by norm_num [registeredLaw]
example : registeredLaw.weight 3 = (1 : ℝ) / 3 := by norm_num [registeredLaw]
example : registeredLaw.weight 5 = (1 : ℝ) / 3 := by norm_num [registeredLaw]

example : registeredLaw.Valid := by exact registered_law_valid
example : immediateLaw.Valid := by exact immediate_law_valid

example (r : Nat → ℝ) (n t : Nat) :
    expectedAggregate registeredLaw r n t =
      ∑ d ∈ registeredLaw.support,
        if d ≤ t ∧ t - d < n then registeredLaw.weight d * r (t - d) else 0 := by
  exact expected_aggregate_decomposition registeredLaw r n t

example (law : DelayLaw) (r : Nat → ℝ) (n t j : Nat)
    (h : ∀ d ∈ law.support, ¬ (d ≤ t ∧ t - d = j ∧ j < n)) :
    candidateCredit law r n t j = 0 := by
  exact candidate_support_bounded law r n t j h

end NarrativeDynamics.MarginalizedTemporalCredit
```

- [ ] **Step 2: Run RED and confirm only the new module/theorem surface is missing.**

Run:

```bash
lake env lean NarrativeDynamics/Tests/MarginalizedTemporalCredit.lean
```

Expected: import/module or unknown-identifier failures for `MarginalizedTemporalCredit`; no failure in existing `AnonymousTemporalCredit`.

- [ ] **Step 3: Implement the minimum finite law and candidate equations.**

Use a finite support plus real weights, with validity carried as a predicate rather than embedding proof fields into every value:

```lean
import NarrativeDynamics.Core.AnonymousTemporalCredit

namespace NarrativeDynamics.MarginalizedTemporalCredit

structure DelayLaw where
  support : Finset Nat
  weight : Nat → ℝ

namespace DelayLaw

def Valid (law : DelayLaw) : Prop :=
  (∀ d, 0 ≤ law.weight d) ∧
  (∀ d, d ∉ law.support → law.weight d = 0) ∧
  (∑ d ∈ law.support, law.weight d) = 1
end DelayLaw

noncomputable def registeredLaw : DelayLaw :=
  { support := {1, 3, 5}
    weight := fun d => if d = 1 ∨ d = 3 ∨ d = 5 then (1 : ℝ) / 3 else 0 }

noncomputable def immediateLaw : DelayLaw :=
  { support := {0}
    weight := fun d => if d = 0 then 1 else 0 }

def validSource (n t d : Nat) : Prop := d ≤ t ∧ t - d < n

noncomputable def expectedAggregate
    (law : DelayLaw) (reward : Nat → ℝ) (n t : Nat) : ℝ :=
  ∑ d ∈ law.support,
    if validSource n t d then law.weight d * reward (t - d) else 0

noncomputable def candidateCredit
    (law : DelayLaw) (credit : Nat → ℝ) (n t j : Nat) : ℝ :=
  ∑ d ∈ law.support,
    if validSource n t d ∧ t - d = j then law.weight d * credit j else 0
```

Prove `registered_law_valid` and `immediate_law_valid` by finite-set simplification and `norm_num`. Prove `expected_aggregate_decomposition` by unfolding `expectedAggregate`. Prove `candidate_support_bounded` by `Finset.sum_eq_zero` using the supplied exclusion hypothesis. Add a boundary example at `t=0` for `registeredLaw` and a drain example with `n=4,t=8`, showing only valid historical indices can contribute.

The expectation theorem is deliberately an expectation **operator** over the fixed marginal delay law. The Python schedule separately audits independent pre-generation. Do not add an unnecessary probability monad to C4 v1.

- [ ] **Step 4: Run focused GREEN.**

```bash
lake build NarrativeDynamics.Core.MarginalizedTemporalCredit
lake env lean NarrativeDynamics/Tests/MarginalizedTemporalCredit.lean
```

Expected: exit 0.

- [ ] **Step 5: Commit Task 1 only.**

```bash
git add NarrativeDynamics/Core/MarginalizedTemporalCredit.lean \
        NarrativeDynamics/Tests/MarginalizedTemporalCredit.lean
git diff --cached --check
git commit -m "feat(lean): model marginalized temporal credit"
git push origin formal/marginalized-temporal-credit-v1
```

---

### Task 2: Prove current-weight update, immediate reduction, boundary/drain contract, and add permanent Lean CI

**Repository:** `qigao/lean`

**Files:**
- Modify: `NarrativeDynamics/Core/MarginalizedTemporalCredit.lean`
- Modify: `NarrativeDynamics/Tests/MarginalizedTemporalCredit.lean`
- Modify: `NarrativeDynamics.lean`
- Modify: `.github/workflows/proof.yml`

**Interfaces:**
- Produces: `Vector`, `dot`, `marginalizedPrediction`, `marginalizedUpdate`, `current_weight_observation`, `immediate_reduction_to_phase3a`, `invalid_candidates_zero`, `drain_uses_same_equation`

- [ ] **Step 1: Add RED tests for current-weight semantics and exact mathematical reduction.**

Append tests requiring:

```lean
example (w z : Vector n) :
    marginalizedPrediction w z = dot w z := by
  exact current_weight_observation w z

example (w feature : Vector n) (denominator alpha reward : ℝ) :
    marginalizedUpdate w alpha reward feature (normalizedCredit feature denominator) =
      AnonymousTemporalCredit.normalizedPhase3AUpdate
        w alpha reward (dot w feature) (normalizedCredit feature denominator) := by
  exact immediate_reduction_to_phase3a w feature denominator alpha reward

example (law : DelayLaw) (x : Nat → Vector n) (n t : Nat)
    (h : ∀ d ∈ law.support, ¬ validSource n t d) :
    marginalizedFeature law x n t = zeroVector := by
  exact invalid_candidates_zero law x n t h
```

- [ ] **Step 2: Run RED.**

```bash
lake env lean NarrativeDynamics/Tests/MarginalizedTemporalCredit.lean
```

Expected: unknown new update definitions/theorems.

- [ ] **Step 3: Implement the vector observation/update layer without prediction history.**

Use flattened finite vectors for the formal contract:

```lean
abbrev Vector (n : Nat) := Fin n → ℝ

def zeroVector : Vector n := fun _ => 0

noncomputable def dot (left right : Vector n) : ℝ :=
  ∑ i, left i * right i

noncomputable def marginalizedFeature
    (law : DelayLaw) (feature : Nat → Vector n) (count t : Nat) : Vector n :=
  fun i => ∑ d ∈ law.support,
    if validSource count t d then law.weight d * feature (t - d) i else 0

noncomputable def marginalizedPrediction (weights feature : Vector n) : ℝ :=
  dot weights feature

noncomputable def marginalizedUpdate
    (weights : Vector n) (alpha feedback : ℝ)
    (expectedFeature normalizedCredit : Vector n) : Vector n :=
  fun i => weights i +
    alpha * (feedback - marginalizedPrediction weights expectedFeature) *
      normalizedCredit i
```

`marginalizedUpdate` receives no historical prediction value. Prove `current_weight_observation` by `rfl`. Prove `immediate_reduction_to_phase3a` by `funext`, unfolding both updates and the existing Phase 3A update. Prove invalid candidates contribute zero by `Finset.sum_eq_zero`. State and prove `drain_uses_same_equation` as equality between the generic update at any post-training clock and the same `marginalizedUpdate`; do not introduce a second drain update definition.

- [ ] **Step 4: Add axiom audit and permanent focused CI.**

Append:

```lean
#print axioms NarrativeDynamics.MarginalizedTemporalCredit.registered_law_valid
#print axioms NarrativeDynamics.MarginalizedTemporalCredit.expected_aggregate_decomposition
#print axioms NarrativeDynamics.MarginalizedTemporalCredit.candidate_support_bounded
#print axioms NarrativeDynamics.MarginalizedTemporalCredit.immediate_reduction_to_phase3a
#print axioms NarrativeDynamics.MarginalizedTemporalCredit.invalid_candidates_zero
```

Export the module from `NarrativeDynamics.lean`. In `.github/workflows/proof.yml`, add a focused step before the full library build:

```yaml
      - name: Marginalized temporal credit theorem tests
        run: |
          export PATH="$HOME/.elan/bin:$PATH"
          lake build NarrativeDynamics.Core.MarginalizedTemporalCredit
          lake env lean NarrativeDynamics/Tests/MarginalizedTemporalCredit.lean
```

- [ ] **Step 5: Run full formal verification.**

```bash
lake build NarrativeDynamics.Core.MarginalizedTemporalCredit
lake env lean NarrativeDynamics/Tests/MarginalizedTemporalCredit.lean
lake build
git grep -nE '\b(sorry|admit)\b' -- \
  NarrativeDynamics/Core/MarginalizedTemporalCredit.lean \
  NarrativeDynamics/Tests/MarginalizedTemporalCredit.lean || true
git diff --check
```

Expected: all commands exit 0; grep has no proof holes.

- [ ] **Step 6: Commit and require exact-head GitHub proof CI.**

```bash
git add NarrativeDynamics/Core/MarginalizedTemporalCredit.lean \
        NarrativeDynamics/Tests/MarginalizedTemporalCredit.lean \
        NarrativeDynamics.lean .github/workflows/proof.yml
git commit -m "feat(lean): prove marginalized temporal credit gate"
git push origin formal/marginalized-temporal-credit-v1
git rev-parse HEAD
```

Record the exact 40-hex head. Do not start Python scientific implementation until its GitHub proof workflow passes and the `#print axioms` output is reviewed with no `sorryAx` or project-defined custom axiom.

---

### Task 3: Bind Gate F and implement the pure Python delay-marginalization operator

**Repository:** `qigao/yolo-motion-perception`

**Branch:** create `experiment/phase-c4-delay-marginalized-credit` from exact `f018384a6bdc25bd6b20dd393a7058df04912f08` only after Task 2 Gate F passes.

**Files:**
- Create: `docs/experiments/phase-c4-delay-marginalized-credit/formal-contract.json`
- Create: `src/neural_state_machine/phase_c4_delay_model.py`
- Create: `src/neural_state_machine/phase_c4_controls.py`
- Create: `tests/test_phase_c4_delay_model.py`
- Create: `tests/test_phase_c4_controls.py`

**Interfaces:**

```python
@dataclass(frozen=True, slots=True)
class DelayLaw:
    support: tuple[int, ...]
    probabilities: tuple[float, ...]

    @classmethod
    def registered(cls) -> "DelayLaw": ...

    @classmethod
    def immediate(cls) -> "DelayLaw": ...

@dataclass(frozen=True, slots=True)
class DecisionCreditRow:
    decision_index: int
    action_index: int
    feature: np.ndarray
    denominator: float

@dataclass(frozen=True, slots=True)
class MarginalizedFeatures:
    candidate_indices: tuple[int, ...]
    expected_feature: np.ndarray
    normalized_credit: np.ndarray


def candidate_indices(feedback_step: int, decision_count: int, law: DelayLaw) -> tuple[int, ...]: ...
def build_marginalized_features(rows: tuple[DecisionCreditRow, ...], feedback_step: int, action_count: int, law: DelayLaw) -> MarginalizedFeatures: ...
def current_weight_prediction(weights: np.ndarray, expected_feature: np.ndarray) -> float: ...
```

The literal `...` above denotes the signature in this plan only; production code must contain complete implementations and no placeholder bodies.

- [ ] **Step 1: Write fail-closed formal-contract and pure operator RED tests.**

Tests must reject a formal manifest with wrong repo/module/Lean/Mathlib version, missing theorem, non-40-hex SHA, false CI/axiom flags, or symlinked path. Delay-model tests must cover:

```python
def test_registered_candidate_indices_are_only_1_3_5_lags():
    law = DelayLaw.registered()
    assert candidate_indices(5, 10, law) == (4, 2, 0)


def test_pre_start_candidates_are_truncated():
    law = DelayLaw.registered()
    assert candidate_indices(2, 10, law) == (1,)


def test_current_weight_prediction_uses_supplied_weights():
    z = np.array([[1.0, 2.0], [0.0, 3.0]], dtype=np.float64)
    w1 = np.array([[2.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    w2 = np.array([[4.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    assert current_weight_prediction(w1, z) == 5.0
    assert current_weight_prediction(w2, z) == 7.0
```

Use tiny `DecisionCreditRow` fixtures to hand-check that `Z_t` uses unnormalized action-blocked features while `C_t` uses `feature/denominator`, each weighted by exactly `1/3`.

- [ ] **Step 2: Run RED.**

```bash
python -m pytest -q tests/test_phase_c4_delay_model.py tests/test_phase_c4_controls.py
```

Expected: missing C4 modules/manifest.

- [ ] **Step 3: Commit the exact Gate F manifest.**

Use this shape, replacing only the `commit` value with the literal exact Task 2 passing SHA captured from `git rev-parse HEAD`:

```json
{
  "schema_version": 1,
  "repository": "qigao/lean",
  "module": "NarrativeDynamics/Core/MarginalizedTemporalCredit.lean",
  "commit": "THE_EXACT_TASK_2_40_HEX_SHA_IS_WRITTEN_HERE_AT_EXECUTION",
  "lean_version": "4.32.0",
  "mathlib_version": "v4.32.0",
  "exact_head_ci_passed": true,
  "axiom_audit_reviewed": true,
  "sorry_ax_present": false,
  "custom_axiom_present": false,
  "theorems": [
    "NarrativeDynamics.MarginalizedTemporalCredit.registered_law_valid",
    "NarrativeDynamics.MarginalizedTemporalCredit.expected_aggregate_decomposition",
    "NarrativeDynamics.MarginalizedTemporalCredit.candidate_support_bounded",
    "NarrativeDynamics.MarginalizedTemporalCredit.current_weight_observation",
    "NarrativeDynamics.MarginalizedTemporalCredit.immediate_reduction_to_phase3a",
    "NarrativeDynamics.MarginalizedTemporalCredit.invalid_candidates_zero"
  ]
}
```

At execution time the uppercase sentence is not committed; it is replaced by the captured literal SHA before staging.

- [ ] **Step 4: Implement strict delay-law values and immutable decision rows.**

`DelayLaw.registered()` returns exactly `(1,3,5)` and three Python `1.0/3.0` probabilities. `DelayLaw.immediate()` returns `(0,)` and `(1.0,)`. Reject booleans, duplicate/negative support, non-finite/negative probabilities, length mismatch, and a probability sum not equal to `1.0` within absolute `1e-15` solely for validating the float representation of the fixed law.

`DecisionCreditRow` copies `feature` to a read-only contiguous float64 vector and validates `denominator == np.dot(feature, feature)` with exact same-value construction by callers; it stores no reward, delay, due step, multiplicity, source identity, prediction, or queue state.

`build_marginalized_features` creates `(action_count, feature_size)` float64 matrices. For every valid candidate `(delay,index)`, add `probability * feature` to `expected_feature[action_index]` and `probability * feature / denominator` to `normalized_credit[action_index]`. Return read-only copies and the candidate indices in law-support order.

- [ ] **Step 5: Implement the formal-contract loader in `phase_c4_controls.py`.**

Expose:

```python
@dataclass(frozen=True, slots=True)
class PhaseC4FormalContract:
    commit: str
    theorems: tuple[str, ...]


def load_phase_c4_formal_contract(root: Path | None = None) -> PhaseC4FormalContract:
    ...
```

Use stdlib `json`, `pathlib`, and a fixed expected theorem tuple. Resolve the repository root, reject symlinks and paths outside the root, hash the manifest bytes for later evidence binding, and never perform a network call at runtime.

- [ ] **Step 6: Run GREEN and freeze-file diff checks.**

```bash
python -m pytest -q tests/test_phase_c4_delay_model.py tests/test_phase_c4_controls.py
python -m ruff check src/neural_state_machine/phase_c4_delay_model.py \
  src/neural_state_machine/phase_c4_controls.py \
  tests/test_phase_c4_delay_model.py tests/test_phase_c4_controls.py
git diff -- src/neural_state_machine/action_value.py \
  src/neural_state_machine/phase3c_schedule.py \
  src/neural_state_machine/phase3c_learners.py \
  src/neural_state_machine/phase3c_controls.py \
  src/neural_state_machine/phase3c_benchmark.py
git diff --check
```

Frozen-file diff must be empty.

- [ ] **Step 7: Commit Task 3.**

```bash
git add docs/experiments/phase-c4-delay-marginalized-credit/formal-contract.json \
  src/neural_state_machine/phase_c4_delay_model.py \
  src/neural_state_machine/phase_c4_controls.py \
  tests/test_phase_c4_delay_model.py tests/test_phase_c4_controls.py
git commit -m "feat: bind C4 delay observation contract"
git push origin experiment/phase-c4-delay-marginalized-credit
```

---

### Task 4: Implement C4-B online learner before any C4-A measurement

**Files:**
- Create: `src/neural_state_machine/phase_c4_learner.py`
- Create: `tests/test_phase_c4_learner.py`
- Frozen/read-only reuse: `src/neural_state_machine/action_value.py`

**Interfaces:**

```python
@dataclass(frozen=True, slots=True)
class MarginalizedCreditUpdate:
    feedback_step: int
    reward: float
    prediction: float
    td_error: float
    applied: bool

class DelayMarginalizedAnonymousCredit(NormalizedActionValue):
    def __init__(self, hidden_size: int, action_count: int, step_size: float = 0.1, *, law: DelayLaw | None = None) -> None: ...
    def select_for_training(self, hidden_state: object, legal_action_indices: object, rng: np.random.Generator) -> ActionValueDecision: ...
    def learn(self, aggregate_reward: object) -> MarginalizedCreditUpdate: ...
    def learn_drain(self, aggregate_reward: object) -> MarginalizedCreditUpdate: ...
    def history_snapshot(self) -> tuple[DecisionCreditRow, ...]: ...
```

- [ ] **Step 1: Write RED tests for selection lifecycle and bounded history.**

Require uniform RNG selection with the same `[seed,0x33414354]` semantics as Phase 3A/C. Before real-step feedback, keep exactly one `_current` row plus at most five completed historical decision rows. The historical row contains only decision index/action/feature/denominator. Assert by dataclass fields and object `__dict__` that no prediction, reward, delay, due step, multiplicity, source ID, pending-source flag, eligibility, or prediction trace exists.

- [ ] **Step 2: Write RED hand-arithmetic test for current-weight prediction.**

Use two actions and one hidden coordinate. Force three historical rows at candidate lags and set known weights. Immediately before `learn`, mutate the test learner's weights through a controlled test helper/snapshot restoration so that current `W_t` differs from every decision-time `W`. Assert `prediction` equals `<W_t,Z_t>` and changes with `W_t`. The test must fail for an implementation that cached decision-time predictions.

- [ ] **Step 3: Write RED drain tests.**

After the final real decision, call `learn_drain` for clocks `N..N+5`. Assert no new `DecisionCreditRow` is appended, candidate indices shrink according to `(1,3,5)`, history becomes empty after the last useful candidate, and every call uses the same update function. A zero-source final clock with `Z=C=0` must validate the scalar and return `applied=False` without changing weights.

- [ ] **Step 4: Write exact `{0:1}` Phase 3A continuity RED test.**

For the same hidden states, forced action RNG, rewards, and `step_size=0.1`, compare a fresh `NormalizedActionValue` with `DelayMarginalizedAnonymousCredit(..., law=DelayLaw.immediate())` after every step:

```python
assert c4_action == baseline_action
assert c4_update.reward == baseline_update.reward
assert c4_update.prediction == baseline_update.prediction_before
assert c4_update.td_error == baseline_update.td_error
assert c4.parameter_snapshot().tobytes() == baseline.parameter_snapshot().tobytes()
assert c4.parameter_digest() == baseline.parameter_digest()
```

- [ ] **Step 5: Run RED.**

```bash
python -m pytest -q tests/test_phase_c4_learner.py
```

Expected: module/class missing.

- [ ] **Step 6: Implement real-step lifecycle.**

On `select_for_training`, compute the frozen Phase 3A feature, legal action set, random behavior action, denominator, and one current `DecisionCreditRow`. Do not append it to completed history yet. On `learn` at feedback clock `t`, build candidates from the completed history for registered law `(1,3,5)`; for immediate law include the current row as lag zero. Compute `P_t` from `self._weights` immediately before update.

For registered law, apply:

```python
candidate_weights = self._weights + self.step_size * td_error * normalized_credit
```

only after validating all intermediates are finite. Then append the current row to history, clear `_current`, evict completed rows older than five decisions, and increment the internal feedback clock.

For immediate law, preserve Phase 3A arithmetic order exactly:

```python
delta = self.step_size * td_error * current.feature / current.denominator
candidate = self._weights[current.action_index] + delta
```

Update only the selected row before appending/evicting history. This special arithmetic path is a Python continuity requirement, not a different scientific law.

- [ ] **Step 7: Implement atomic drain.**

`learn_drain` requires `_current is None`, validates the scalar first, builds `Z/C` from completed history at the internal feedback clock, computes current-weight prediction, validates candidate weights before mutation, evicts by age after the update, and increments the clock. It accepts no source/delay/count/timestamp argument.

- [ ] **Step 8: Run GREEN and regressions.**

```bash
python -m pytest -q tests/test_phase_c4_learner.py tests/test_action_value.py tests/test_phase3c_learners.py
python -m ruff check src/neural_state_machine/phase_c4_learner.py tests/test_phase_c4_learner.py
git diff -- src/neural_state_machine/action_value.py src/neural_state_machine/phase3c_learners.py
git diff --check
```

- [ ] **Step 9: Commit Task 4.**

```bash
git add src/neural_state_machine/phase_c4_learner.py tests/test_phase_c4_learner.py
git commit -m "feat: add C4 marginalized online learner"
git push origin experiment/phase-c4-delay-marginalized-credit
```

---

### Task 5: Implement C4-A anonymous batch operator probe

**Files:**
- Create: `src/neural_state_machine/phase_c4_batch_probe.py`
- Create: `tests/test_phase_c4_batch_probe.py`

**Interfaces:**

```python
@dataclass(frozen=True, slots=True)
class BatchProbeFit:
    weights: np.ndarray
    row_count: int
    column_count: int
    augmented_rank: int
    residual_norm: float
    singular_values: np.ndarray
    penalty: float


def build_batch_design(rows: tuple[DecisionCreditRow, ...], feedback: tuple[float, ...], *, action_count: int, law: DelayLaw) -> tuple[np.ndarray, np.ndarray]: ...
def fit_anonymous_batch_probe(design: np.ndarray, target: np.ndarray, *, action_count: int, feature_size: int, penalty: float = 1e-6) -> BatchProbeFit: ...
```

- [ ] **Step 1: Write RED analytic ridge tests.**

Use a tiny full-rank matrix where the augmented least-squares solution can be checked against `np.linalg.solve(Z.T @ Z + 1e-6*I, Z.T @ F)` **in the test only**. Production must use `np.linalg.lstsq` on the augmented system. Assert bias columns are included in the penalty by constructing a design where an unpenalized bias would produce a different solution.

- [ ] **Step 2: Write RED information-boundary tests.**

`build_batch_design` accepts only `DecisionCreditRow`, scalar feedback, action count, and `DelayLaw`. Its module must not import `LatentRewardRecord`, `AggregateFeedback.records`, diagnostic provenance, `DelayedCueEpisode.correct_action_index`, or Task 12 reference modules. Add a source-inspection guard that fails if forbidden identifier strings occur in `phase_c4_batch_probe.py`.

- [ ] **Step 3: Write RED drain-row test.**

For `N=4` and support `(1,3,5)`, provide `N+5` scalar feedback calls. Assert the design contains one row per delivery clock, including post-training drain clocks, and later rows contain only still-valid pre-`N` candidate decisions.

- [ ] **Step 4: Run RED.**

```bash
python -m pytest -q tests/test_phase_c4_batch_probe.py
```

- [ ] **Step 5: Implement canonical design construction and fixed ridge.**

Build each `Z_t` with `build_marginalized_features`, flatten in C-order from shape `(action_count, feature_size)` to `action_count*feature_size`, and pair it with the actual scalar learner-call stream. Fit exactly:

```python
sqrt_penalty = math.sqrt(1e-6)
regularizer = sqrt_penalty * np.eye(column_count, dtype=np.float64)
augmented_design = np.vstack((design, regularizer))
augmented_target = np.concatenate((target, np.zeros(column_count, dtype=np.float64)))
coef, residuals, rank, singular_values = np.linalg.lstsq(
    augmented_design,
    augmented_target,
    rcond=None,
)
```

Reject any penalty other than exactly `1e-6` in the registered fit API. Validate finite inputs/results and return read-only copies.

- [ ] **Step 6: Run GREEN and commit.**

```bash
python -m pytest -q tests/test_phase_c4_batch_probe.py tests/test_phase_c4_delay_model.py
python -m ruff check src/neural_state_machine/phase_c4_batch_probe.py tests/test_phase_c4_batch_probe.py
git diff --check
git add src/neural_state_machine/phase_c4_batch_probe.py tests/test_phase_c4_batch_probe.py
git commit -m "feat: add C4 anonymous batch operator probe"
git push origin experiment/phase-c4-delay-marginalized-credit
```

---

### Task 6: Build fail-closed C4 protocol controls and current-weight/non-interference audits

**Files:**
- Modify: `src/neural_state_machine/phase_c4_controls.py`
- Modify: `tests/test_phase_c4_controls.py`

**Interfaces:**

```python
@dataclass(frozen=True, slots=True)
class PhaseC4ProtocolAudit:
    action_count: int
    latent_record_count: int
    delivered_record_count: int
    real_feedback_count: int
    drain_feedback_count: int
    pending_final: int
    action_digest: str
    delay_digest: str
    due_step_digest: str
    aggregate_feedback_digest: str
    learner_call_digest: str
    candidate_feature_digest: str
    parameter_digest: str
    source_relabel_invariant: bool
    multiplicity_hidden_invariant: bool
    current_weight_probe_passed: bool
    bounded_history_passed: bool
    immediate_continuity_passed: bool
    repeated_run_equal: bool


def validate_phase_c4_protocol(audit: PhaseC4ProtocolAudit, *, expected_actions: int) -> None: ...
```

- [ ] **Step 1: Write one valid synthetic audit and mutate every field independently.**

Require `ValueError` for count mismatch, pending final nonzero, changed digest lengths/hex, false non-interference, false current-weight probe, false bounded-history check, false immediate continuity, or false repeatability. No behavioral score field exists in this dataclass.

- [ ] **Step 2: Add independent reconstruction tests for `Z_t/P_t/C_t`.**

Capture learner-owned decision rows externally and independently call `build_marginalized_features`. For each synthetic step compare reconstructed matrices and prediction with learner debug snapshots exposed only as read-only protocol data. Change weights before a synthetic feedback call and require the predicted change to equal `<delta_W,Z_t>` within exact NumPy operation ordering used by the helper; a stale prediction implementation must fail.

- [ ] **Step 3: Add source/multiplicity non-interference tests using the frozen C3 aggregator externally.**

Use `phase3c_schedule.AnonymousRewardAggregator` only in the harness. Create two observer histories with relabeled `records[*].source_step` and altered observer multiplicity metadata but identical `.value` scalar streams. Feed only scalar values into two fresh C4 learners and require action/parameter digests equal.

- [ ] **Step 4: Add frozen-input byte hashes.**

Create a constant tuple of required read-only paths including all C3 scientific modules/evidence named in the spec. A helper `frozen_input_hashes(root)` returns sorted `(path, sha256)` rows and rejects symlinks/missing files. Tests mutate a temporary copy and require fail-closed mismatch.

- [ ] **Step 5: Run GREEN and commit.**

```bash
python -m pytest -q tests/test_phase_c4_controls.py tests/test_phase_c4_learner.py tests/test_phase3c_schedule.py
python -m ruff check src/neural_state_machine/phase_c4_controls.py tests/test_phase_c4_controls.py
git diff --check
git add src/neural_state_machine/phase_c4_controls.py tests/test_phase_c4_controls.py
git commit -m "test: define C4 protocol gate"
git push origin experiment/phase-c4-delay-marginalized-credit
```

---

### Task 7: Build protocol-only C4 orchestration, exact lineage checks, and secondary evaluation manifest

**Files:**
- Create: `src/neural_state_machine/phase_c4_benchmark.py`
- Create: `tests/test_phase_c4_benchmark.py`
- Read-only reuse: `phase3c_schedule.py`, `action_value_benchmark.py`, `reward_learning.py`, `phase3c_diagnostics/evaluation.py` in tests only

**Interfaces:**

```python
@dataclass(frozen=True, slots=True)
class PhaseC4Config:
    hidden_size: int = 64
    recurrent_radius: float = 0.9
    step_size: float = 0.1
    training_decisions: int = 2000
    evaluation_blocks: int = 20
    checkpoint_interval: int = 100
    ridge_penalty: float = 1e-6

@dataclass(frozen=True, slots=True)
class PhaseC4ProtocolResult:
    seed: int
    training_fixture_digest: str
    evaluation_fixture_digest: str
    action_digest: str
    latent_reward_digest: str
    audit: PhaseC4ProtocolAudit


def run_phase_c4_protocol_gate(seeds: tuple[int, ...] = (7, 17, 29), config: PhaseC4Config | None = None) -> tuple[PhaseC4ProtocolResult, ...]: ...
```

- [ ] **Step 1: Write anti-peeking RED test.**

For a reduced test config, call `run_phase_c4_protocol_gate` and assert the result has no attributes named `post_training`, `state_reset`, `shuffled_control`, `operator_passed`, `behavior_passed`, `secondary_scores`, or `accuracy`.

- [ ] **Step 2: Write matched-lineage RED tests.**

Pre-generate expected actions from `[seed,0x33414354]`. Build the hidden delay schedule through the frozen `build_hidden_delay_schedule`. Execute the normal C4-B protocol twice and require identical action/latent reward/delay/due/aggregate-call/parameter digests. Build a C4-A protocol stream from the same decision rows and aggregate scalars without fitting it; require the same action, fixture, schedule, and call digests.

- [ ] **Step 3: Write exact immediate-continuity integration test.**

Run the full registered fixture lineage with `DelayLaw.immediate()` and compare against `run_action_value_experiment` for each registered seed: action tuple/digest, reward tuple/digest, training/evaluation fixture digests, final parameter bytes/digest. This integration test is stricter than the unit test and must remain behavior-score blind for the delayed C4 path.

- [ ] **Step 4: Implement Task 12 secondary evaluation bundles independently.**

Production C4 code generates the eight additional sets itself from:

```python
rng = np.random.Generator(
    np.random.PCG64(np.random.SeedSequence([seed, 0x33434641, 2, evaluation_id]))
)
```

using frozen `_build_fixtures(task, rng, evaluation_blocks)` and `_episode_digest`. In tests only, compare the 27 `(seed, evaluation_id, digest)` rows against `phase3c_diagnostics.evaluation.build_evaluation_bundles` to prove exact Task 12 lineage continuity. C4 production must not import the diagnostics package.

- [ ] **Step 5: Implement fixed behavior-gate helper without running registered behavior.**

Expose a pure function `registered_c4_gate(...)` that reuses the exact numeric surface from Phase 3C but is implemented in C4 code to avoid modifying C3. Test threshold boundaries `179/180`, `33/34`, reset `19/20/21`, shuffled `149/150` and exact totals.

- [ ] **Step 6: Implement protocol-only orchestration.**

Timeline for every real step remains:

```text
hidden -> random behavior action -> latent reward -> enqueue -> aggregate feedback_at(t) -> scalar-only C4 call
```

C4-A protocol collection records learner-owned decision rows and scalar calls but does **not** call the ridge fitter. C4-B protocol executes the online learner. Drain from `training_decisions` through `max(due_steps)` with no selection. Build the external audit and validate it before returning.

- [ ] **Step 7: Run GREEN and commit.**

```bash
python -m pytest -q tests/test_phase_c4_benchmark.py tests/test_phase_c4_controls.py \
  tests/test_phase_c4_learner.py tests/test_phase_c4_batch_probe.py
python -m ruff check src/neural_state_machine/phase_c4_benchmark.py tests/test_phase_c4_benchmark.py
git diff --check
git add src/neural_state_machine/phase_c4_benchmark.py tests/test_phase_c4_benchmark.py
git commit -m "feat: add C4 protocol-only orchestration"
git push origin experiment/phase-c4-delay-marginalized-credit
```

---

### Task 8: Add stage-specific behavior execution, evidence schemas, strict verifier, and locked environment

**Files:**
- Modify: `src/neural_state_machine/phase_c4_benchmark.py`
- Create: `src/neural_state_machine/phase_c4_evidence.py`
- Create: `scripts/benchmark_phase_c4_delay_marginalized_credit.py`
- Create: `scripts/verify_phase_c4_delay_marginalized_credit.py`
- Create: `tests/test_phase_c4_evidence.py`
- Extend: `tests/test_phase_c4_benchmark.py`
- Create: `requirements/phase-c4.in`
- Create: `requirements/phase-c4-python312.lock`

**Interfaces:**

```python
@dataclass(frozen=True, slots=True)
class EvaluationScore:
    overall_correct: int
    overall_total: int
    per_delay: tuple[tuple[int, int, int], ...]

@dataclass(frozen=True, slots=True)
class C4ASeedResult:
    seed: int
    normal: EvaluationScore
    reset: EvaluationScore
    shuffled: EvaluationScore
    secondary: tuple[dict[str, object], ...]

@dataclass(frozen=True, slots=True)
class C4BSeedResult:
    seed: int
    normal: EvaluationScore
    reset: EvaluationScore
    shuffled: EvaluationScore
    secondary: tuple[dict[str, object], ...]
```

- [ ] **Step 1: Write RED tests for C4-A measurement function on a tiny unregistered config.**

The function must first receive/validate a `PhaseC4ProtocolResult`, then fit normal and original-block-shuffled streams separately through `fit_anonymous_batch_probe`, install the resulting weights into a fresh evaluation-only action-value object, and score original + secondary bundles. Training labels and evaluation labels never enter the fit function. A deliberately label-leaking adapter in a negative test must be rejected by API/type boundaries.

- [ ] **Step 2: Write RED tests for C4-B measurement function on a tiny unregistered config.**

Replay the protocol-validated normal and shuffled online learners, require their normal protocol digest to equal the supplied protocol result, then evaluate. Shuffled rewards are created only with frozen `_permute_reward_blocks(..., SeedSequence([seed,0x33534846]), block_size=10)`; actions and hidden-delay schedule must remain identical.

- [ ] **Step 3: Write RED evidence-schema mutation tests.**

Prospective C4-A manifest must include exact scientific implementation reference, formal-contract hash/SHA, C3 frozen input hashes, config, seeds, lineages, evaluation manifest, environment fields, expected record keys, and `stage="c4a"`. Before measurement, `c4a-result.json` must be absent. C4-B manifest additionally binds the frozen C4-A result hash and requires `operator_passed=true`; before C4-B measurement, `c4b-result.json` must be absent.

Mutation tests reject extra/missing keys, wrong booleans, bool-as-int, wrong counts, NaN/Infinity, changed thresholds/lineages, changed `1e-6`, changed delay law, altered formal SHA, or a result file present in no-result mode.

- [ ] **Step 4: Implement canonical JSON and verifier.**

Canonical JSON is UTF-8, sorted keys, compact separators, `allow_nan=False`, exactly one terminal newline. The strict verifier recomputes all hashes from bytes, validates every expected row key, recalculates gate booleans from score counts, and refuses a `behavior_passed/operator_passed` value that disagrees with raw counts.

CLI modes:

```text
benchmark_phase_c4_delay_marginalized_credit.py protocol
benchmark_phase_c4_delay_marginalized_credit.py prepare-c4a --output DIR
benchmark_phase_c4_delay_marginalized_credit.py measure-c4a --manifest FILE --output DIR
benchmark_phase_c4_delay_marginalized_credit.py prepare-c4b --c4a-result FILE --output DIR
benchmark_phase_c4_delay_marginalized_credit.py measure-c4b --manifest FILE --output DIR
verify_phase_c4_delay_marginalized_credit.py --stage c4a --no-result-ok
verify_phase_c4_delay_marginalized_credit.py --stage c4a
verify_phase_c4_delay_marginalized_credit.py --stage c4b --no-result-ok
verify_phase_c4_delay_marginalized_credit.py --stage c4b
```

The measurement commands must fail unless the supplied manifest's scientific implementation, formal contract, input hashes, environment fields, and exact current scientific file hashes match before fitting/evaluation.

- [ ] **Step 5: Create the exact C4 Python lock.**

`requirements/phase-c4.in` contains direct anchors:

```text
numpy==2.5.3
pytest==9.1.1
ruff==0.15.22
```

`requirements/phase-c4-python312.lock` is the following exact environment, with a header identifying Python 3.12.14 and the C4 exact-head CI run once generated:

```text
iniconfig==2.3.0
numpy==2.5.3
packaging==26.3
pip==26.2.1
pluggy==1.6.0
Pygments==2.21.0
pytest==9.1.1
ruff==0.15.22
setuptools==84.0.0
wheel==0.48.0
```

Do not silently upgrade the lock after any scientific measurement.

- [ ] **Step 6: Run local tests without registered measurement.**

```bash
python -m pytest -q tests/test_phase_c4_*.py
python -m ruff check src/neural_state_machine/phase_c4_*.py \
  scripts/benchmark_phase_c4_delay_marginalized_credit.py \
  scripts/verify_phase_c4_delay_marginalized_credit.py tests/test_phase_c4_*.py
python scripts/benchmark_phase_c4_delay_marginalized_credit.py protocol
python scripts/verify_phase_c4_delay_marginalized_credit.py --stage c4a --no-result-ok
git diff --check
```

Do **not** run `measure-c4a` or `measure-c4b` here.

- [ ] **Step 7: Commit Task 8.**

```bash
git add src/neural_state_machine/phase_c4_benchmark.py \
  src/neural_state_machine/phase_c4_evidence.py \
  scripts/benchmark_phase_c4_delay_marginalized_credit.py \
  scripts/verify_phase_c4_delay_marginalized_credit.py \
  tests/test_phase_c4_benchmark.py tests/test_phase_c4_evidence.py \
  requirements/phase-c4.in requirements/phase-c4-python312.lock
git commit -m "feat: add C4 staged evidence pipeline"
git push origin experiment/phase-c4-delay-marginalized-credit
```

---

### Task 9: Make Gate P permanent in CI, seal C4-A prospectively, and STOP before measurement

**Files:**
- Modify: `.github/workflows/ci.yml`
- Tests: all `tests/test_phase_c4_*.py`
- Prospective artifact only: generated C4-A manifest outside committed result path

**Interfaces:**
- Produces exact reviewed **scientific implementation reference** `C4_SCIENCE_HEAD`
- Produces prospective C4-A manifest/hash from that head
- Does not create a registered result

- [ ] **Step 1: Add permanent C4 CI after existing C3 gates without weakening/removing them.**

Add steps equivalent to:

```yaml
      - name: Phase C4 tests
        run: pytest -q tests/test_phase_c4_*.py

      - name: Phase C4 Ruff
        run: >-
          ruff check src/neural_state_machine/phase_c4_*.py
          scripts/benchmark_phase_c4_delay_marginalized_credit.py
          scripts/verify_phase_c4_delay_marginalized_credit.py
          tests/test_phase_c4_*.py

      - name: Phase C4 protocol gate
        run: python scripts/benchmark_phase_c4_delay_marginalized_credit.py protocol
```

Add an exact Python 3.12.14 venv recreation step using `pip==26.2.1` and `requirements/phase-c4-python312.lock`, compare `pip freeze --all` byte-for-line after case-insensitive sorting, install the project `--no-deps --no-build-isolation -e .`, rerun C4 tests/protocol/verifier, then prepare the prospective C4-A manifest without fitting.

Upload only the prospective manifest artifact named `phase-c4a-prospective-${{ github.sha }}`.

- [ ] **Step 2: Run local regression and frozen-file audit.**

```bash
python -m pytest -q
python -m ruff check .
python scripts/benchmark_phase_c4_delay_marginalized_credit.py protocol
python scripts/verify_phase3c_anonymous_credit.py
python scripts/verify_phase_c4_delay_marginalized_credit.py --stage c4a --no-result-ok
git diff --check
```

Also compare SHA-256 of every frozen C3 path against the Task 6 manifest.

- [ ] **Step 3: Commit CI only.**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add Phase C4 protocol gate"
git push origin experiment/phase-c4-delay-marginalized-credit
git rev-parse HEAD
```

Record that exact head as `C4_SCIENCE_HEAD`. C4-A and C4-B scientific code is now frozen for C4 v1.

- [ ] **Step 4: Require exact-head CI and inspect evidence.**

The exact `C4_SCIENCE_HEAD` GitHub run must show all repository tests, Ruff, Phase 3C verifier, C4 protocol gate, locked-environment recreation, no-result C4 verifier, and prospective C4-A manifest generation green. Download/read the prospective manifest and confirm it binds the exact head and formal SHA.

- [ ] **Step 5: Mandatory STOP.**

Stop here. Report the exact head, exact CI run/job IDs, formal SHA, prospective C4-A manifest hash, environment hash, and that **no C4-A fit or behavioral score has been produced**. Obtain explicit human approval before Task 10.

---

### Task 10: After explicit approval, run exactly one sealed C4-A registered measurement and freeze its result

**Authorization gate:** Do not execute this task without an explicit post-Task-9 approval.

**Files:**
- Create temporarily: `.github/workflows/phase-c4a-measurement-once.yml`
- Create after valid run: `docs/experiments/phase-c4-delay-marginalized-credit/c4a-manifest.json`
- Create after valid run: `docs/experiments/phase-c4-delay-marginalized-credit/c4a-result.json`
- Create after valid run: `docs/experiments/phase-c4-delay-marginalized-credit/c4a-report.md`
- Create after valid run: `docs/experiments/phase-c4-delay-marginalized-credit/c4a-provenance.json`

- [ ] **Step 1: Create a one-shot workflow whose first scientific step is sealed-environment equality.**

It must checkout the approved execution head, recreate exact Python 3.12.14 lock, compare every environment field/hash in the sealed C4-A manifest, rerun Gate P/no-result verifier, and only then invoke:

```bash
python scripts/benchmark_phase_c4_delay_marginalized_credit.py \
  measure-c4a \
  --manifest /path/to/sealed/c4a-manifest.json \
  --output /tmp/phase-c4a-result
python scripts/verify_phase_c4_delay_marginalized_credit.py \
  --stage c4a \
  --root /tmp/phase-c4a-result
```

Any preflight mismatch exits before ridge fitting and produces no scientific result.

- [ ] **Step 2: Run exactly once for the approved sealed manifest.**

If preflight fails, record the attempt as invalid provenance, correct only the environment mismatch through a reviewed manifest/environment procedure, and require a new explicit measurement approval. Do not inspect partial fit/evaluation output.

- [ ] **Step 3: For a valid run, verify complete C4-A evidence.**

Require three registered seeds, normal and original shuffled fits, all primary original-set score fields, all fixed reset/per-delay fields, all eight secondary sets per seed, solver rank/residual diagnostics, exact frozen lineages, and recomputed `operator_passed`.

- [ ] **Step 4: Commit result/provenance without altering scientific code.**

```bash
git add docs/experiments/phase-c4-delay-marginalized-credit/c4a-manifest.json \
  docs/experiments/phase-c4-delay-marginalized-credit/c4a-result.json \
  docs/experiments/phase-c4-delay-marginalized-credit/c4a-report.md \
  docs/experiments/phase-c4-delay-marginalized-credit/c4a-provenance.json
git commit -m "docs: freeze Phase C4-A operator result"
```

- [ ] **Step 5: Delete the one-shot C4-A workflow in a separate cleanup commit.**

```bash
git rm .github/workflows/phase-c4a-measurement-once.yml
git commit -m "ci: remove completed C4-A measurement workflow"
```

- [ ] **Step 6: Mandatory interpretation stop.**

If any seed fails the fixed C4-A gate, record `operator_passed=false`, declare C4 v1 stopped before C4-B registered behavior, and do not create a C4-B manifest/workflow. If all seeds pass, proceed only to Task 11 preparation; this does not authorize C4-B measurement.

---

### Task 11: If C4-A passes, seal C4-B prospectively against the already frozen C4-B implementation and STOP

**Authorization precondition:** Valid C4-A result with verifier-recomputed `operator_passed=true`.

**Files:**
- Create: `docs/experiments/phase-c4-delay-marginalized-credit/c4b-manifest.json`
- Do not create `c4b-result.json` yet.

- [ ] **Step 1: Prove C4-B scientific files are byte-identical to `C4_SCIENCE_HEAD`.**

Hash the C4 delay model, learner, controls, benchmark, formal contract, config constants, verifier, lock, and all frozen C3 inputs. Compare against the Task 9 sealed scientific manifest. Any difference stops C4 v1; do not restage C4-B after seeing C4-A.

- [ ] **Step 2: Prepare C4-B manifest.**

It binds:

- original `C4_SCIENCE_HEAD` as the scientific implementation reference;
- current execution head separately;
- exact formal SHA and formal-contract hash;
- exact C4-A result/provenance hashes and `operator_passed=true`;
- registered config/thresholds/lineages;
- secondary evaluation manifest;
- exact Python 3.12.14 lock/environment;
- expected three-seed normal/reset/shuffled/secondary result keys;
- no `c4b-result.json` present.

- [ ] **Step 3: Run no-result verifier and exact-head CI.**

```bash
python scripts/verify_phase_c4_delay_marginalized_credit.py --stage c4b --no-result-ok
python scripts/benchmark_phase_c4_delay_marginalized_credit.py protocol
```

Require the current GitHub CI to pass while also proving the scientific file hashes still equal `C4_SCIENCE_HEAD`.

- [ ] **Step 4: Mandatory STOP.**

Report C4-A result hash, original frozen C4 science head, C4-B manifest hash, current CI run/job IDs, and explicitly state that no C4-B registered behavior has been evaluated. Obtain a second explicit human approval before Task 12.

---

### Task 12: After second explicit approval, run sealed C4-B measurement, freeze result, and remove one-shot workflow

**Authorization gate:** Do not execute without explicit post-Task-11 C4-B measurement approval.

**Files:**
- Create temporarily: `.github/workflows/phase-c4b-measurement-once.yml`
- Create after valid run: `docs/experiments/phase-c4-delay-marginalized-credit/c4b-result.json`
- Create after valid run: `docs/experiments/phase-c4-delay-marginalized-credit/c4b-report.md`
- Create after valid run: `docs/experiments/phase-c4-delay-marginalized-credit/c4b-provenance.json`

- [ ] **Step 1: Create one-shot C4-B workflow with sealed preflight before behavior.**

The workflow recreates the exact lock, verifies scientific file hashes against `C4_SCIENCE_HEAD`, validates C4-A pass binding and C4-B manifest bytes, reruns Gate P, and only then invokes `measure-c4b`.

- [ ] **Step 2: Execute one valid sealed measurement.**

A preflight failure is provenance only and produces no scientific result. A valid run must score all three seeds for normal/reset/original shuffled and the fixed secondary surfaces without parameter/threshold changes.

- [ ] **Step 3: Verify status from raw counts.**

The strict verifier computes:

```text
formal_valid = true only from the bound passing formal contract
protocol_valid = true only from exact Gate P evidence
operator_passed = true only from frozen verified C4-A result
behavior_passed = true only if every registered C4-B seed passes the unchanged primary gate
all_passed = formal_valid && protocol_valid && operator_passed && behavior_passed
```

- [ ] **Step 4: Commit evidence only.**

```bash
git add docs/experiments/phase-c4-delay-marginalized-credit/c4b-result.json \
  docs/experiments/phase-c4-delay-marginalized-credit/c4b-report.md \
  docs/experiments/phase-c4-delay-marginalized-credit/c4b-provenance.json
git commit -m "docs: freeze Phase C4-B behavioral result"
```

- [ ] **Step 5: Delete the one-shot workflow separately.**

```bash
git rm .github/workflows/phase-c4b-measurement-once.yml
git commit -m "ci: remove completed C4-B measurement workflow"
```

- [ ] **Step 6: Final audit.**

Rerun full CI/verifier, compare every frozen C3 scientific/evidence hash to its original value, compare every C4 scientific file to `C4_SCIENCE_HEAD`, and report the bounded interpretation required by the spec. Do not infer generic identifiability, unknown-delay inference, source reconstruction, convergence, or production readiness.

---

## Plan Self-Review Checklist

Before execution approval, review this committed plan against the spec:

1. **Spec coverage:** Tasks 1-2 cover Formal Gate F; Tasks 3-7 cover delay law, batch operator, online learner, current-weight semantics, bounded history, drain, Gate P, and exact Phase 3A continuity; Task 8 covers strict evidence/lock; Task 9 seals before behavior; Tasks 10-12 enforce the two independent measurement checkpoints.
2. **Anti-tuning:** C4-B implementation is completed and frozen before C4-A measurement; C4-A failure terminates C4 v1; C4-A success cannot modify C4-B scientific bytes.
3. **Information boundary:** batch and online learner APIs contain no realized source/delay/due/multiplicity/latent reward fields; observer metadata remains outside scalar learner calls.
4. **Current-weight contract:** both formal and Python paths use `P_t=<W_t,Z_t>` and no prediction trace.
5. **Immediate continuity:** `{0:1}` has both mathematical reduction and byte-for-byte Python arithmetic continuity.
6. **Drain:** no synthetic actions and no special trace state; the same marginalized equation is used until support expires.
7. **Frozen history:** all C3 scientific/evidence files remain read-only and are hashed in Gate P/measurement manifests.
8. **No placeholders in production:** any explanatory placeholder strings shown in this plan are replaced by exact captured values before a file is staged; production/test files contain no `TODO`, `TBD`, `pass`, `...`, `sorry`, or `admit` in required implementations/proofs.

## Execution Boundary

This plan commit itself authorizes **no production implementation and no C4 measurement**. After review approval, execution begins with Task 1 and follows the gates serially. Task 9 is the mandatory stop before C4-A measurement; Task 11 is the mandatory stop before C4-B measurement.
