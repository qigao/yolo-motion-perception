# Phase C4 Delay-Marginalized Anonymous Credit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and register Phase C4 as a two-stage experiment that first tests whether the fixed anonymous delay-mixture operator admits a successful batch fit and, only if that gate passes, tests a prospectively frozen online delay-marginalized credit learner.

**Architecture:** Extend the proven Phase 3C anonymous-credit theory with a separate Lean `MarginalizedTemporalCredit` module. Bind the exact passing theorem head into additive Python modules for the fixed delay-law operator, C4-A batch ridge, C4-B online learner, protocol controls, orchestration, sealed evidence, and stage-specific verification. Both C4-A and C4-B scientific implementations are completed and frozen before the first C4-A registered measurement, so the C4-A result cannot be used to tune C4-B.

**Tech Stack:** Lean 4.32.0, Mathlib v4.32.0, Lake, Python 3.12.14 for registered measurement, NumPy 2.5.3, pytest 9.1.1, Ruff 0.15.22, setuptools 84.0.0, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-16-phase-c4-delay-marginalized-anonymous-credit-design.md`

## Global Constraints

- Planning base is exact `qigao/yolo-motion-perception@f018384a6bdc25bd6b20dd393a7058df04912f08`.
- Implementation starts on `experiment/phase-c4-delay-marginalized-credit` from that exact commit only after execution approval.
- Formal work starts from exact proven Phase 3C formal head `qigao/lean@3de297cee2a94a7fc309531334f720f1b34467c9` on `formal/marginalized-temporal-credit-v1`.
- Do not modify `qigao/lean/NarrativeDynamics/Core/AnonymousTemporalCredit.lean`, `NarrativeDynamics/Core/TemporalCredit.lean`, or their existing tests.
- Do not modify these C3 scientific files: `src/neural_state_machine/action_value.py`, `src/neural_state_machine/phase3c_schedule.py`, `src/neural_state_machine/phase3c_learners.py`, `src/neural_state_machine/phase3c_controls.py`, `src/neural_state_machine/phase3c_benchmark.py`, `src/neural_state_machine/reward_learning.py`, `src/neural_state_machine/action_value_benchmark.py`, or `scripts/verify_phase3c_anonymous_credit.py`.
- Do not modify existing Phase 3C evidence or `docs/experiments/phase-3c-failure-attribution-v1/**`.
- Registered seeds are exactly `(7, 17, 29)`; training decisions `2000`; evaluation blocks `20`; hidden size `64`; recurrent radius `0.9`; step size `0.1`; cue delays `(1,2,3,4,5)`; hidden feedback delays `(1,3,5)`; public probabilities `(1/3,1/3,1/3)`.
- Reuse training/evaluation fixture lineages, action RNG `[seed, 0x33414354]`, hidden-delay lineage `[seed, 0x3343444C]`, and original shuffled-control lineage `[seed, 0x33534846]`.
- Primary behavior thresholds remain exactly: normal `>=180/200`, every cue delay `>=34/40`, reset exactly `100/200`, every reset delay exactly `20/40`, shuffled `<150/200`.
- The eight Task 12 additional evaluation sets use `PCG64(SeedSequence([seed, 0x33434641, 2, evaluation_id]))`, IDs `0..7`; they are secondary stability surfaces only.
- C4-A ridge penalty is exactly `1e-6`, including action-block bias coordinates. No scaling, hyperparameter search, feature selection, checkpoint selection, or evaluation-label fitting.
- C4-B uses `P_t=<W_t,Z_t>` with the current pre-update weights. It stores no prediction trace and uses no stale decision-time prediction.
- Normal C4-B credit support is exactly lags `(1,3,5)`. The learner receives one finite scalar per delivery clock and no realized source ID, delay, due step, multiplicity, latent reward, or environment queue metadata.
- Terminal drain uses the same observation/update equation, adds no synthetic action, and consumes every anonymous delivery clock through the **public support horizon** `N - 1 + max(delay_support)`, even when the frozen aggregator returns `0.0` for an empty bucket. It must not stop early at `max(actual_due_steps)`, because that would reveal realized hidden-schedule information through run length.
- For registered support `(1,3,5)`, drain clocks are exactly `N, N+1, N+2, N+3, N+4`: five scalar calls after the final real decision.
- Protocol control `{0:1}` must match frozen Phase 3A exactly in the same environment: action sequence, scalar update fields, full weight bytes, and parameter digest.
- For exact `{0:1}` continuity, preserve Phase 3A arithmetic order `step_size * td_error * feature / denominator` on the selected row. Do not substitute a mathematically equivalent matrix operation with a different floating-point order.
- C4-A code, C4-B code, formal binding, registered config, thresholds, and lineages are frozen before the first C4-A registered measurement. After observing C4-A, no C4 scientific code change is permitted inside C4 v1.
- Gate F failure stops Python scientific execution. Gate P failure means `harness invalid`; registered behavior must not be inspected or serialized.
- C4-A failure records `operator_passed=false` and terminates C4 v1 before C4-B measurement. Do not tune in place.
- C4-A success does not authorize C4-B. A second explicit human approval is required after a separately sealed C4-B manifest is reviewed.
- Lean proves the real-number contract only. It does not prove NumPy execution, behavioral success, general identifiability, convergence, source reconstruction, or production readiness.

## File Ownership Map

### `qigao/lean`

| Path | Responsibility |
|---|---|
| `NarrativeDynamics/Core/MarginalizedTemporalCredit.lean` | Finite delay law, candidate support, weighted observation, current-weight prediction, bounded credit, boundary truncation, update equation, immediate reduction. |
| `NarrativeDynamics/Tests/MarginalizedTemporalCredit.lean` | Registered/immediate laws, boundary/drain examples, update theorems, axiom audit. |
| `NarrativeDynamics.lean` | Export C4 formal module. |
| `.github/workflows/proof.yml` | Permanent focused C4 theorem gate. |

### `qigao/yolo-motion-perception`

| Path | Responsibility |
|---|---|
| `docs/experiments/phase-c4-delay-marginalized-credit/formal-contract.json` | Exact passing C4 Lean SHA and theorem names. |
| `src/neural_state_machine/phase_c4_delay_model.py` | Fixed delay laws, learner-owned decision rows, candidate indices, `Z_t`, `C_t`, current-weight `P_t`. |
| `src/neural_state_machine/phase_c4_batch_probe.py` | C4-A design matrix and fixed ridge solve. |
| `src/neural_state_machine/phase_c4_learner.py` | C4-B online learner. |
| `src/neural_state_machine/phase_c4_controls.py` | Formal-contract loader, frozen-input hashes, protocol audit, non-interference and continuity controls. |
| `src/neural_state_machine/phase_c4_benchmark.py` | Fixed streams, protocol-only orchestration, evaluation bundles, normal/shuffled execution, fixed gates. |
| `src/neural_state_machine/phase_c4_evidence.py` | Canonical manifests/results and stage-specific schema validation. |
| `scripts/benchmark_phase_c4_delay_marginalized_credit.py` | Protocol CLI and authorized stage measurement entry point. |
| `scripts/verify_phase_c4_delay_marginalized_credit.py` | Strict no-result/result verifier. |
| `requirements/phase-c4.in` | Direct C4 dependency anchors. |
| `requirements/phase-c4-python312.lock` | Exact registered Python 3.12.14 environment. |
| `.github/workflows/ci.yml` | Permanent C4 tests, Ruff, protocol gate, locked-environment recreation, prospective manifest preparation. |
| `.github/workflows/phase-c4a-measurement-once.yml` | Created only after explicit C4-A measurement approval; removed after the attempt is frozen. |
| `.github/workflows/phase-c4b-measurement-once.yml` | Created only after C4-A passes and explicit C4-B approval; removed after measurement. |
| `tests/test_phase_c4_delay_model.py` | Pure operator tests. |
| `tests/test_phase_c4_batch_probe.py` | Fixed ridge and information-boundary tests. |
| `tests/test_phase_c4_learner.py` | Online arithmetic, bounded history, current-weight prediction, drain, exact immediate reduction. |
| `tests/test_phase_c4_controls.py` | Formal contract, frozen inputs, Gate P mutations, non-interference. |
| `tests/test_phase_c4_benchmark.py` | Matched lineages, anti-peeking, secondary evaluation lineage, fixed gate. |
| `tests/test_phase_c4_evidence.py` | Manifest/result/writer/verifier mutation tests. |

---

### Task 1: Formalize finite delay-law observation and bounded candidate support

**Repository:** `qigao/lean`

**Branch:** `formal/marginalized-temporal-credit-v1` from exact `3de297cee2a94a7fc309531334f720f1b34467c9`.

**Files:**
- Create: `NarrativeDynamics/Core/MarginalizedTemporalCredit.lean`
- Create: `NarrativeDynamics/Tests/MarginalizedTemporalCredit.lean`
- Frozen: `NarrativeDynamics/Core/AnonymousTemporalCredit.lean`

**Interfaces:** `DelayLaw`, `DelayLaw.Valid`, `registeredLaw`, `immediateLaw`, `validSource`, `expectedAggregate`, `candidateCredit`, `registered_law_valid`, `immediate_law_valid`, `expected_aggregate_decomposition`, `candidate_support_bounded`.

- [ ] **Step 1: Write RED theorem tests before the module exists.**

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

- [ ] **Step 2: Run RED.**

```bash
lake env lean NarrativeDynamics/Tests/MarginalizedTemporalCredit.lean
```

Expected: missing new module or identifiers; existing anonymous-credit files still compile.

- [ ] **Step 3: Implement the minimum finite law and candidate equations.**

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

Prove the two law-validity theorems with finite-set simplification and `norm_num`. Prove `expected_aggregate_decomposition` by unfolding `expectedAggregate`. Prove `candidate_support_bounded` with `Finset.sum_eq_zero` and the supplied exclusion hypothesis. Add concrete boundary tests at `t=0` and a post-training clock where only one valid historical candidate remains.

The formal expectation is a finite marginal-delay expectation operator. Independent schedule generation is audited separately in Python; C4 v1 does not need a probability monad.

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

### Task 2: Prove current-weight update, immediate reduction, boundary/drain contract, and permanent Lean CI

**Repository:** `qigao/lean`

**Files:**
- Modify: `NarrativeDynamics/Core/MarginalizedTemporalCredit.lean`
- Modify: `NarrativeDynamics/Tests/MarginalizedTemporalCredit.lean`
- Modify: `NarrativeDynamics.lean`
- Modify: `.github/workflows/proof.yml`

**Interfaces:** `Vector`, `zeroVector`, `dot`, `marginalizedFeature`, `marginalizedPrediction`, `marginalizedUpdate`, `current_weight_observation`, `immediate_reduction_to_phase3a`, `invalid_candidates_zero`, `drain_uses_same_equation`.

- [ ] **Step 1: Add RED tests for current-weight semantics and immediate reduction.**

```lean
example (w z : Vector n) :
    marginalizedPrediction w z = dot w z := by
  exact current_weight_observation w z

example (w feature : Vector n) (denominator alpha reward : ℝ) :
    marginalizedUpdate w alpha reward feature
        (AnonymousTemporalCredit.normalizedCredit feature denominator) =
      AnonymousTemporalCredit.normalizedPhase3AUpdate
        w alpha reward (dot w feature)
        (AnonymousTemporalCredit.normalizedCredit feature denominator) := by
  exact immediate_reduction_to_phase3a w feature denominator alpha reward

example (law : DelayLaw) (x : Nat → Vector n) (count t : Nat)
    (h : ∀ d ∈ law.support, ¬ validSource count t d) :
    marginalizedFeature law x count t = zeroVector := by
  exact invalid_candidates_zero law x count t h
```

- [ ] **Step 2: Run RED.**

```bash
lake env lean NarrativeDynamics/Tests/MarginalizedTemporalCredit.lean
```

Expected: unknown update definitions/theorems.

- [ ] **Step 3: Implement flattened finite-vector observation and update.**

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

Prove `current_weight_observation` by reflexivity. Prove immediate reduction by function extensionality and unfolding both updates. Prove invalid candidates produce `zeroVector`. Define `drain_uses_same_equation` as a theorem about the same generic `marginalizedUpdate`; do not create a second drain update function.

- [ ] **Step 4: Add axiom audit and focused permanent CI.**

Append:

```lean
#print axioms NarrativeDynamics.MarginalizedTemporalCredit.registered_law_valid
#print axioms NarrativeDynamics.MarginalizedTemporalCredit.expected_aggregate_decomposition
#print axioms NarrativeDynamics.MarginalizedTemporalCredit.candidate_support_bounded
#print axioms NarrativeDynamics.MarginalizedTemporalCredit.immediate_reduction_to_phase3a
#print axioms NarrativeDynamics.MarginalizedTemporalCredit.invalid_candidates_zero
```

Export the module from `NarrativeDynamics.lean`. Add before the full Lean build:

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

Expected: build/test exit 0; grep returns no proof holes.

- [ ] **Step 6: Commit and require exact-head GitHub proof CI.**

```bash
git add NarrativeDynamics/Core/MarginalizedTemporalCredit.lean \
        NarrativeDynamics/Tests/MarginalizedTemporalCredit.lean \
        NarrativeDynamics.lean .github/workflows/proof.yml
git commit -m "feat(lean): prove marginalized temporal credit gate"
git push origin formal/marginalized-temporal-credit-v1
git rev-parse HEAD
```

Record the exact head. Python scientific implementation does not begin until that exact-head proof workflow passes and the axiom log is reviewed with no `sorryAx` or project-defined custom axiom.

---

### Task 3: Bind Gate F and implement the pure Python delay-marginalization operator

**Repository:** `qigao/yolo-motion-perception`

**Branch:** `experiment/phase-c4-delay-marginalized-credit` from exact `f018384a6bdc25bd6b20dd393a7058df04912f08` after Task 2 Gate F passes.

**Files:**
- Create: `docs/experiments/phase-c4-delay-marginalized-credit/formal-contract.json`
- Create: `src/neural_state_machine/phase_c4_delay_model.py`
- Create: `src/neural_state_machine/phase_c4_controls.py`
- Create: `tests/test_phase_c4_delay_model.py`
- Create: `tests/test_phase_c4_controls.py`

**Interfaces:**
- `DelayLaw.registered() -> DelayLaw`
- `DelayLaw.immediate() -> DelayLaw`
- `candidate_indices(feedback_step: int, decision_count: int, law: DelayLaw) -> tuple[int, ...]`
- `build_marginalized_features(rows: tuple[DecisionCreditRow, ...], feedback_step: int, action_count: int, law: DelayLaw) -> MarginalizedFeatures`
- `current_weight_prediction(weights: np.ndarray, expected_feature: np.ndarray) -> float`
- `load_phase_c4_formal_contract(root: Path | None = None) -> PhaseC4FormalContract`

- [ ] **Step 1: Write formal-contract and pure-operator RED tests.**

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

Also reject wrong formal repository/module/version/theorem set, non-40-hex commit, false CI/axiom flags, symlink path, duplicate/negative delay support, non-finite/negative probabilities, and invalid decision-row shapes.

- [ ] **Step 2: Run RED.**

```bash
python -m pytest -q tests/test_phase_c4_delay_model.py tests/test_phase_c4_controls.py
```

Expected: missing C4 modules/formal manifest.

- [ ] **Step 3: Generate the formal-contract JSON from the captured Task 2 SHA.**

From the C4 Python repo root, set `LEAN_HEAD` to the exact reviewed Task 2 head and run:

```bash
export LEAN_HEAD="$(git -C ../lean rev-parse HEAD)"
python - <<'PY'
import json
import os
from pathlib import Path

head = os.environ["LEAN_HEAD"]
if len(head) != 40 or any(ch not in "0123456789abcdef" for ch in head):
    raise SystemExit("LEAN_HEAD must be lowercase 40-hex")

payload = {
    "schema_version": 1,
    "repository": "qigao/lean",
    "module": "NarrativeDynamics/Core/MarginalizedTemporalCredit.lean",
    "commit": head,
    "lean_version": "4.32.0",
    "mathlib_version": "v4.32.0",
    "exact_head_ci_passed": True,
    "axiom_audit_reviewed": True,
    "sorry_ax_present": False,
    "custom_axiom_present": False,
    "theorems": [
        "NarrativeDynamics.MarginalizedTemporalCredit.registered_law_valid",
        "NarrativeDynamics.MarginalizedTemporalCredit.expected_aggregate_decomposition",
        "NarrativeDynamics.MarginalizedTemporalCredit.candidate_support_bounded",
        "NarrativeDynamics.MarginalizedTemporalCredit.current_weight_observation",
        "NarrativeDynamics.MarginalizedTemporalCredit.immediate_reduction_to_phase3a",
        "NarrativeDynamics.MarginalizedTemporalCredit.invalid_candidates_zero",
    ],
}
path = Path("docs/experiments/phase-c4-delay-marginalized-credit/formal-contract.json")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
```

Before staging, compare the generated `commit` value with the exact GitHub passing formal head recorded in Task 2.

- [ ] **Step 4: Implement strict delay laws and immutable decision rows.**

`DelayLaw.registered()` returns support `(1,3,5)` and three `1.0/3.0` probabilities. `DelayLaw.immediate()` returns `(0,)` and `(1.0,)`. Probability sum validation uses absolute `1e-15` only to validate the fixed float law.

`DecisionCreditRow` stores exactly `decision_index`, `action_index`, a read-only contiguous float64 `feature`, and finite positive `denominator`. It stores no reward, realized delay, due step, multiplicity, source ID, prediction, or pending-source flag.

`build_marginalized_features` returns read-only `(action_count, feature_size)` matrices. For each valid candidate, add `probability * feature` to `expected_feature[action_index]` and `probability * feature / denominator` to `normalized_credit[action_index]`.

- [ ] **Step 5: Implement strict formal-contract loading in `phase_c4_controls.py`.**

Define immutable `PhaseC4FormalContract(commit: str, theorems: tuple[str, ...])`. `load_phase_c4_formal_contract` uses stdlib `json/pathlib`, a fixed theorem tuple, root containment checks, and symlink rejection. It performs no network access.

- [ ] **Step 6: Run GREEN and frozen-file diff checks.**

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
- `DelayMarginalizedAnonymousCredit(hidden_size, action_count, step_size=0.1, law=None)`
- `select_for_training(hidden_state, legal_action_indices, rng) -> ActionValueDecision`
- `learn(aggregate_reward) -> MarginalizedCreditUpdate`
- `learn_drain(aggregate_reward) -> MarginalizedCreditUpdate`
- `history_snapshot() -> tuple[DecisionCreditRow, ...]`

`MarginalizedCreditUpdate` contains exactly `feedback_step`, `reward`, `prediction`, `td_error`, and `applied`.

- [ ] **Step 1: Write RED tests for selection lifecycle and bounded history.**

Require the same uniform behavior-action RNG semantics as Phase 3A/C. Before real-step feedback, the learner owns one current decision row plus at most five completed rows. Inspect private state in tests and reject prediction trace, eligibility, reward, realized-delay, due-step, multiplicity, source-ID, or environment pending-source fields.

- [ ] **Step 2: Write RED hand-arithmetic test for current-weight prediction.**

Use two actions and one hidden coordinate. Build three candidate historical rows with hand-computed `Z_t/C_t`, then set a known current weight matrix immediately before feedback. Assert returned `prediction` equals `<W_t,Z_t>`. Repeat from the same history with a different current matrix and assert the prediction changes exactly through the changed matrix, which would fail for cached decision-time predictions.

- [ ] **Step 3: Write RED drain tests.**

For `N=4` and registered support `(1,3,5)`, make real decisions at clocks `0..3`, then call `learn_drain` exactly at clocks `4,5,6,7,8`, corresponding to `N` through `N-1+max(support)`. Assert no new decision row is appended, candidate sets shrink only by public age, and completed history is empty after clock `8`. Do not call clock `9`; it is outside the registered public support horizon.

- [ ] **Step 4: Write exact `{0:1}` Phase 3A continuity RED test.**

For identical hidden states, RNG and rewards, compare a fresh `NormalizedActionValue` with C4 immediate-law learner after every step:

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

Expected: C4 learner module/class missing.

- [ ] **Step 6: Implement real-step lifecycle.**

On selection, reuse frozen `_feature` and `_legal_actions`, select uniformly with the supplied NumPy generator, compute denominator, and keep one current `DecisionCreditRow`. Do not append it to completed history before feedback.

For registered law, `learn` builds `Z/C` from completed history, computes `td_error = reward - <W_t,Z_t>`, validates the full candidate weight matrix, then mutates. After the update, append the current row, clear current state, evict completed rows older than five decisions, and advance the internal feedback clock.

For immediate law, include the current row at lag zero and update with exact Phase 3A arithmetic order:

```python
delta = self.step_size * td_error * current.feature / current.denominator
candidate_row = self._weights[current.action_index] + delta
```

Update only the selected row after finite validation.

- [ ] **Step 7: Implement atomic drain.**

`learn_drain` requires no current decision, accepts only the scalar, computes current-weight `P_t`, validates candidate weights before mutation, evicts history by age after feedback, and increments the internal clock. It accepts no source/delay/count/timestamp parameter. The orchestrator, not the learner, guarantees exactly five registered drain scalar calls for support `(1,3,5)`.

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
- `build_batch_design(rows, feedback, action_count, law) -> tuple[np.ndarray, np.ndarray]`
- `fit_anonymous_batch_probe(design, target, action_count, feature_size, penalty=1e-6) -> BatchProbeFit`
- `BatchProbeFit` contains read-only `weights`, row/column counts, augmented rank, residual norm, read-only singular values, and penalty.

- [ ] **Step 1: Write RED analytic ridge tests.**

Use a tiny full-rank matrix and compare the production result to this test-only reference:

```python
reference = np.linalg.solve(
    design.T @ design + 1e-6 * np.eye(design.shape[1]),
    design.T @ target,
)
np.testing.assert_allclose(fit.weights.reshape(-1), reference, rtol=0.0, atol=1e-12)
```

Add a case that distinguishes penalized versus unpenalized bias.

- [ ] **Step 2: Write RED information-boundary tests.**

`build_batch_design` accepts only learner-owned decision rows, scalar feedback, action count, and delay law. Add source-inspection assertions that `phase_c4_batch_probe.py` does not import or reference `LatentRewardRecord`, `AggregateFeedback.records`, correct-action labels, Task 12 provenance, or source-visible references.

- [ ] **Step 3: Write RED terminal-drain design-row test.**

For `N=4`, support `(1,3,5)`, and scalar feedback clocks `0..8`, assert exactly nine design rows. The final real drain row at clock `8` contains only source decision `3` as its valid lag-five candidate. Separately assert the pure helper reports no candidate at clock `9`; no learner/environment scalar call is made at clock `9`.

- [ ] **Step 4: Run RED.**

```bash
python -m pytest -q tests/test_phase_c4_batch_probe.py
```

- [ ] **Step 5: Implement canonical design and fixed ridge.**

Each design row is `expected_feature.reshape(-1, order="C")`. Fit exactly with augmented least squares:

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

The registered API rejects a penalty different from `1e-6`. Validate all inputs/results as finite; reshape coefficients to `(action_count, feature_size)` and return read-only arrays.

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

### Task 6: Build fail-closed C4 protocol controls and non-interference audits

**Files:**
- Modify: `src/neural_state_machine/phase_c4_controls.py`
- Modify: `tests/test_phase_c4_controls.py`

**Interfaces:**
- `PhaseC4ProtocolAudit` contains counts, action/schedule/call/candidate/parameter digests and booleans for source relabel invariance, hidden multiplicity, current-weight probe, bounded history, immediate continuity, repeatability.
- `validate_phase_c4_protocol(audit, expected_actions)` raises `ValueError` on any mismatch.
- `frozen_input_hashes(root) -> tuple[tuple[str, str], ...]` hashes all required C3 scientific/evidence inputs and rejects symlinks/missing paths.

- [ ] **Step 1: Write one valid synthetic audit and mutate every field independently.**

Reject action/latent/delivered count mismatch, `real_feedback_count != N`, registered `drain_feedback_count != 5`, nonzero pending final, malformed/changed digest, false non-interference, false current-weight probe, false bounded-history check, false immediate continuity, or false repeatability. The audit type has no behavioral score or behavior-gate field.

- [ ] **Step 2: Add independent reconstruction tests for `Z_t/P_t/C_t`.**

Capture learner-owned rows externally and reconstruct terms through `build_marginalized_features`. Compare to learner protocol snapshots. Perturb current weights immediately before feedback and require the prediction delta to match the same `Z_t`, proving no stale prediction state is used.

- [ ] **Step 3: Add source/multiplicity non-interference tests with C3 aggregator metadata kept external.**

Create two observer histories that differ in record source labels or multiplicity metadata but expose identical scalar `.value` streams. Feed only scalars to fresh C4 learners and require identical action and parameter digests.

- [ ] **Step 4: Add frozen-input hashes.**

Hash every C3 frozen source/evidence path listed by the spec. In temporary-copy tests, mutate one byte, delete one file, or replace one file with a symlink and require fail-closed validation.

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

### Task 7: Build protocol-only C4 orchestration, exact lineages, secondary evaluation manifest, and fixed gate helper

**Files:**
- Create: `src/neural_state_machine/phase_c4_benchmark.py`
- Create: `tests/test_phase_c4_benchmark.py`
- Read-only reuse: C3 schedule, action-value benchmark, reward-learning helpers
- Test-only read: `phase3c_diagnostics/evaluation.py`

**Interfaces:**
- `PhaseC4Config` defaults: hidden size `64`, radius `0.9`, step `0.1`, decisions `2000`, evaluation blocks `20`, checkpoint `100`, ridge `1e-6`.
- `PhaseC4ProtocolResult` contains seed, fixture/action/reward digests and `PhaseC4ProtocolAudit` only.
- `run_phase_c4_protocol_gate(seeds=(7,17,29), config=None) -> tuple[PhaseC4ProtocolResult, ...]`.
- `registered_c4_gate(normal, per_delay, reset, reset_per_delay, shuffled) -> bool`.

- [ ] **Step 1: Write anti-peeking RED test.**

For a reduced test config, assert protocol results have no attributes `post_training`, `state_reset`, `shuffled_control`, `operator_passed`, `behavior_passed`, `secondary_scores`, or `accuracy`.

- [ ] **Step 2: Write matched-lineage RED tests.**

Pre-generate expected actions from `[seed,0x33414354]`. Build the frozen hidden-delay schedule. Execute C4-B protocol twice and require identical action/reward/delay/due/call/parameter digests. Build C4-A protocol rows/scalars from the same fixed stream without fitting; require matching fixture/action/schedule/call digests.

- [ ] **Step 3: Write exact immediate-continuity integration test.**

For each registered seed, run full fixture lineage with immediate law and compare against `run_action_value_experiment`: action tuple/digest, reward tuple/digest, training/evaluation fixture digests, final weight bytes and parameter digest. Immediate law has no terminal drain calls because `max(support)=0`. Do not evaluate delayed C4 behavior in this test.

- [ ] **Step 4: Implement Task 12 secondary evaluation bundles independently.**

Production C4 uses:

```python
rng = np.random.Generator(
    np.random.PCG64(
        np.random.SeedSequence([seed, 0x33434641, 2, evaluation_id])
    )
)
```

Build fixtures with frozen `_build_fixtures` and digest them with `_episode_digest`. In tests only, compare all 27 `(seed,evaluation_id,digest)` rows with Task 12 `phase3c_diagnostics.evaluation.build_evaluation_bundles`. Production C4 must not import the diagnostics package.

- [ ] **Step 5: Implement fixed gate and boundary tests.**

Test overall `179/180`, per-delay `33/34`, reset `19/20/21`, shuffled `149/150`, and exact totals. Do not call the gate during protocol-only execution.

- [ ] **Step 6: Implement protocol-only orchestration with public-horizon drain.**

Real-step order is:

```text
hidden -> random behavior action -> latent reward -> enqueue -> feedback_at(t) -> scalar-only C4 call
```

C4-A protocol collection records rows/scalars but never calls the ridge fitter. C4-B protocol executes online updates. After decision `N-1`, continue calling `aggregator.feedback_at(t)` and the C4 scalar drain API for every public clock:

```python
last_delivery_clock = config.training_decisions - 1 + max(law.support)
for delivery_step in range(config.training_decisions, last_delivery_clock + 1):
    feedback = aggregator.feedback_at(delivery_step)
    learner.learn_drain(feedback.value)
```

This is exactly five drain calls for registered support `(1,3,5)`, including zero-valued empty buckets. Never stop at `max(schedule.due_steps)`. At the end require aggregator pending count zero and C4 completed history empty. Validate the external protocol audit before returning.

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

### Task 8: Add stage-specific measurement execution, evidence schemas, strict verifier, and locked environment

**Files:**
- Modify: `src/neural_state_machine/phase_c4_benchmark.py`
- Create: `src/neural_state_machine/phase_c4_evidence.py`
- Create: `scripts/benchmark_phase_c4_delay_marginalized_credit.py`
- Create: `scripts/verify_phase_c4_delay_marginalized_credit.py`
- Create: `tests/test_phase_c4_evidence.py`
- Modify: `tests/test_phase_c4_benchmark.py`
- Create: `requirements/phase-c4.in`
- Create: `requirements/phase-c4-python312.lock`

**Interfaces:**
- `measure_c4a_from_protocol(protocol, config) -> C4ASeedResult`
- `measure_c4b_from_protocol(protocol, config) -> C4BSeedResult`
- `prepare_c4a_manifest(root, output_dir) -> Path`
- `prepare_c4b_manifest(root, c4a_result_path, output_dir) -> Path`
- `verify_c4_stage(root, stage, allow_missing_result) -> None`

- [ ] **Step 1: Write C4-A tiny unregistered measurement RED test.**

Require a validated protocol result first. Fit normal and original-block-shuffled scalar streams separately with fixed ridge, install each fitted matrix into a fresh evaluation-only action-value object, and score original plus secondary bundles. The fit API receives no labels/source metadata. Assert registered C4-A design rows total `N + 5` under support `(1,3,5)`.

- [ ] **Step 2: Write C4-B tiny unregistered measurement RED test.**

Replay protocol-validated normal and shuffled online runs. Original shuffle uses only:

```python
shuffle_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x33534846]))
shuffled_rewards = _permute_reward_blocks(normal_rewards, shuffle_rng, block_size=10)
```

Require unchanged action/schedule lineages and exactly five drain scalar calls in both normal and shuffled registered paths.

- [ ] **Step 3: Write evidence-schema mutation tests.**

C4-A prospective manifest includes exact scientific implementation reference, formal contract SHA/hash, C3 frozen hashes, config, seeds, lineages, evaluation manifest, public drain horizon, expected `N+5` scalar/design rows, environment fields, expected result keys, thresholds, and `stage="c4a"`. No result exists before measurement.

C4-B manifest additionally binds frozen C4-A result/provenance hashes and requires verifier-recomputed `operator_passed=true`. Reject extra/missing keys, bool-as-int, count changes, non-finite values, changed thresholds, changed `1e-6`, changed delay law, changed drain horizon, altered formal SHA, altered lineages, or a result present in no-result mode.

- [ ] **Step 4: Implement canonical JSON and strict verifier.**

Canonical JSON uses UTF-8, sorted keys, compact separators, `allow_nan=False`, one terminal newline. Recompute all hashes from bytes and all gate booleans from raw counts. Reject serialized gate flags that disagree with recomputation.

CLI commands are exactly:

```text
python scripts/benchmark_phase_c4_delay_marginalized_credit.py protocol
python scripts/benchmark_phase_c4_delay_marginalized_credit.py prepare-c4a --output DIR
python scripts/benchmark_phase_c4_delay_marginalized_credit.py measure-c4a --manifest FILE --output DIR
python scripts/benchmark_phase_c4_delay_marginalized_credit.py prepare-c4b --c4a-result FILE --output DIR
python scripts/benchmark_phase_c4_delay_marginalized_credit.py measure-c4b --manifest FILE --output DIR
python scripts/verify_phase_c4_delay_marginalized_credit.py --stage c4a --no-result-ok
python scripts/verify_phase_c4_delay_marginalized_credit.py --stage c4a
python scripts/verify_phase_c4_delay_marginalized_credit.py --stage c4b --no-result-ok
python scripts/verify_phase_c4_delay_marginalized_credit.py --stage c4b
```

Measurement commands compare manifest scientific SHA/file hashes, formal binding, immutable inputs and environment before fitting/evaluation.

- [ ] **Step 5: Create the exact C4 Python lock.**

`requirements/phase-c4.in`:

```text
numpy==2.5.3
pytest==9.1.1
ruff==0.15.22
```

`requirements/phase-c4-python312.lock`:

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

The file header records Python 3.12.14 and the exact C4 CI run once the first exact-head environment recreation passes. Dependency versions are not changed after scientific measurement.

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

Do not run either measurement command.

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

### Task 9: Make Gate P permanent, seal C4-A prospectively, and STOP before measurement

**Files:**
- Modify: `.github/workflows/ci.yml`

**Produces:** reviewed `C4_SCIENCE_HEAD`, exact-head CI evidence, prospective C4-A manifest/hash, no registered fit/score.

- [ ] **Step 1: Add C4 CI after existing C3 gates without removing or weakening them.**

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

Add Python 3.12.14 venv recreation with `pip==26.2.1`, install `requirements/phase-c4-python312.lock`, compare sorted `pip freeze --all`, install project with `--no-deps --no-build-isolation -e .`, rerun C4 tests/protocol/no-result verifier, prepare C4-A manifest, and upload only `phase-c4a-prospective-${{ github.sha }}`.

- [ ] **Step 2: Run full local regression and frozen-file audit.**

```bash
python -m pytest -q
python -m ruff check .
python scripts/benchmark_phase_c4_delay_marginalized_credit.py protocol
python scripts/verify_phase3c_anonymous_credit.py
python scripts/verify_phase_c4_delay_marginalized_credit.py --stage c4a --no-result-ok
git diff --check
```

Compare all frozen C3 SHA-256 values against Task 6.

- [ ] **Step 3: Commit CI and freeze the scientific head.**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add Phase C4 protocol gate"
git push origin experiment/phase-c4-delay-marginalized-credit
git rev-parse HEAD
```

Record the literal output as `C4_SCIENCE_HEAD`. From this point C4-A and C4-B scientific files are immutable for C4 v1.

- [ ] **Step 4: Require exact-head CI and inspect prospective evidence.**

Require repository tests, Ruff, C3 verifier, C4 protocol, exact locked environment, C4 no-result verifier and prospective manifest preparation all green on the exact head. Read the uploaded manifest and confirm exact scientific/formal/environment/input hashes and five registered drain calls.

- [ ] **Step 5: Mandatory STOP.**

Report exact head, exact CI run/job IDs, formal SHA, prospective C4-A manifest hash and environment hash. State explicitly that no C4-A fit or behavioral score has been produced. Obtain explicit human approval before Task 10.

---

### Task 10: After explicit approval, run one sealed C4-A registered measurement and freeze its result

**Authorization:** execute only after explicit post-Task-9 approval.

**Files:**
- Create temporarily: `.github/workflows/phase-c4a-measurement-once.yml`
- Create after valid run: `docs/experiments/phase-c4-delay-marginalized-credit/c4a-manifest.json`
- Create after valid run: `docs/experiments/phase-c4-delay-marginalized-credit/c4a-result.json`
- Create after valid run: `docs/experiments/phase-c4-delay-marginalized-credit/c4a-report.md`
- Create after valid run: `docs/experiments/phase-c4-delay-marginalized-credit/c4a-provenance.json`

- [ ] **Step 1: Create one-shot workflow with sealed equality before fitting.**

The workflow checks out the approved execution head, recreates exact Python 3.12.14 environment, compares all sealed fields/hashes, reruns Gate P and no-result verifier, then and only then executes:

```bash
python scripts/benchmark_phase_c4_delay_marginalized_credit.py \
  measure-c4a --manifest sealed/c4a-manifest.json --output /tmp/phase-c4a
python scripts/verify_phase_c4_delay_marginalized_credit.py \
  --stage c4a --root /tmp/phase-c4a
```

Any mismatch exits before fitting and produces no scientific result.

- [ ] **Step 2: Run the approved sealed attempt.**

A preflight failure is provenance only. Do not inspect partial fit/evaluation data. Any corrected seal requires a reviewed new manifest and new explicit approval.

- [ ] **Step 3: Verify complete C4-A evidence.**

Require all three seeds, normal and original shuffled fits, primary original-set score fields, reset/per-delay fields, eight secondary sets per seed, fit rank/residual diagnostics, exactly `2005` scalar/design rows per registered training condition, frozen lineages and verifier-recomputed `operator_passed`.

- [ ] **Step 4: Commit result/provenance only.**

```bash
git add docs/experiments/phase-c4-delay-marginalized-credit/c4a-manifest.json \
  docs/experiments/phase-c4-delay-marginalized-credit/c4a-result.json \
  docs/experiments/phase-c4-delay-marginalized-credit/c4a-report.md \
  docs/experiments/phase-c4-delay-marginalized-credit/c4a-provenance.json
git commit -m "docs: freeze Phase C4-A operator result"
```

- [ ] **Step 5: Delete the completed one-shot workflow in a separate cleanup commit.**

```bash
git rm .github/workflows/phase-c4a-measurement-once.yml
git commit -m "ci: remove completed C4-A measurement workflow"
```

- [ ] **Step 6: Mandatory interpretation stop.**

If any seed fails, record `operator_passed=false`, stop C4 v1 before C4-B, and do not create C4-B manifest/workflow. If all seeds pass, only Task 11 preparation is authorized next.

---

### Task 11: If C4-A passes, seal C4-B against the already frozen implementation and STOP

**Precondition:** valid verified C4-A with `operator_passed=true`.

**Files:**
- Create: `docs/experiments/phase-c4-delay-marginalized-credit/c4b-manifest.json`
- Do not create C4-B result/report/provenance yet.

- [ ] **Step 1: Prove C4-B scientific files are byte-identical to `C4_SCIENCE_HEAD`.**

Hash C4 delay model, batch probe, learner, controls, benchmark, evidence code, scripts, formal contract, lock and all frozen C3 inputs. Compare to Task 9 scientific manifest. Any difference terminates C4 v1 rather than restaging C4-B after observing C4-A.

- [ ] **Step 2: Prepare C4-B manifest.**

Bind original `C4_SCIENCE_HEAD`, current execution head separately, exact formal SHA/hash, C4-A result/provenance hashes, verifier-recomputed `operator_passed=true`, registered config/thresholds/lineages, five-call public drain horizon, secondary evaluation manifest, exact environment, expected result keys, and absence of C4-B result.

- [ ] **Step 3: Run no-result verifier and exact-head CI.**

```bash
python scripts/verify_phase_c4_delay_marginalized_credit.py --stage c4b --no-result-ok
python scripts/benchmark_phase_c4_delay_marginalized_credit.py protocol
```

Require current CI green and scientific hashes equal to `C4_SCIENCE_HEAD`.

- [ ] **Step 4: Mandatory STOP.**

Report C4-A result hash, frozen C4 science head, C4-B manifest hash, exact CI run/job IDs, and explicitly state that no C4-B registered behavior has been evaluated. Obtain a second explicit human approval before Task 12.

---

### Task 12: After second explicit approval, run sealed C4-B measurement, freeze result, and remove workflow

**Authorization:** execute only after explicit post-Task-11 approval.

**Files:**
- Create temporarily: `.github/workflows/phase-c4b-measurement-once.yml`
- Create after valid run: `docs/experiments/phase-c4-delay-marginalized-credit/c4b-result.json`
- Create after valid run: `docs/experiments/phase-c4-delay-marginalized-credit/c4b-report.md`
- Create after valid run: `docs/experiments/phase-c4-delay-marginalized-credit/c4b-provenance.json`

- [ ] **Step 1: Create one-shot C4-B workflow with sealed preflight.**

Recreate the exact lock, verify scientific hashes against `C4_SCIENCE_HEAD`, validate C4-A pass binding and C4-B manifest bytes, rerun Gate P, then execute C4-B measurement only after all checks pass.

- [ ] **Step 2: Execute one valid sealed measurement.**

A preflight failure is provenance only. A valid run scores all three seeds for normal/reset/original shuffled plus the fixed secondary surfaces without changing any scientific parameter. Every registered normal/shuffled training path must contain `2000` real scalar calls plus exactly `5` drain scalar calls.

- [ ] **Step 3: Recompute status from raw evidence.**

```text
formal_valid = bound formal contract is valid
protocol_valid = exact Gate P evidence is valid
operator_passed = frozen C4-A result passes its fixed gate
behavior_passed = every registered C4-B seed passes unchanged Gate B
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

Rerun full CI/verifier, prove every frozen C3 path is unchanged and every C4 scientific path still matches `C4_SCIENCE_HEAD`, then write only the bounded interpretation permitted by the spec.

---

## Plan Self-Review

- **Spec coverage:** Tasks 1-2 cover Gate F; Tasks 3-7 cover the fixed delay operator, C4-A, C4-B, current-weight semantics, bounded history, public-horizon drain, Gate P and exact Phase 3A continuity; Task 8 covers evidence and lock; Task 9 seals before behavior; Tasks 10-12 enforce independent C4-A/C4-B measurement checkpoints.
- **Anti-tuning:** C4-B implementation is complete and frozen before C4-A measurement. A C4-A failure terminates C4 v1. A C4-A success cannot change C4-B scientific bytes.
- **Information boundary:** batch and online APIs contain no realized source/delay/due/multiplicity/latent-reward fields. Observer metadata remains outside learner calls. Public run length is fixed by support, not actual hidden due times.
- **Current-weight contract:** formal and Python paths use `P_t=<W_t,Z_t>` with no prediction trace.
- **Immediate continuity:** `{0:1}` has both formal reduction and exact Python arithmetic/byte continuity.
- **Drain:** no synthetic action or special trace state; registered support forces exactly five post-decision scalar clocks, including empty-bucket zeros, so hidden schedule realization is not exposed by early termination.
- **Frozen history:** C3 scientific/evidence files remain read-only and are hashed at protocol/measurement gates.
- **Concrete execution:** runtime-captured identifiers such as the Task 2 Lean head are generated directly from reviewed commands and written literally before staging; no unresolved implementation bodies or proof holes are allowed in committed production/test/formal files.

## Execution Boundary

This plan commit authorizes no production implementation and no C4 measurement. After plan review approval, execution begins with Task 1 and follows the gates serially. Task 9 is the mandatory stop before C4-A measurement; Task 11 is the mandatory stop before C4-B measurement.
