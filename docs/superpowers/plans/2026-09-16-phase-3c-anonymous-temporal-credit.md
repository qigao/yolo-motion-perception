# Phase 3C Anonymous Temporal Credit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a formally bounded, deterministic Phase 3C experiment in which per-action latent rewards arrive after hidden variable delays, may arrive out of source order, may collide into one aggregate scalar, and remain anonymous to the learner; prove the missing anonymous-credit properties in Lean before Python behavioral measurement, then validate the Python protocol before observing the first registered behavioral result.

**Architecture:** Treat Phase 3C as one serial cross-repository program. First extend `qigao/lean` with a new `AnonymousTemporalCredit` module that imports but does not modify the proven TemporalCredit v1 theory. Gate F proves aggregate-view non-interference, reward conservation, source-metadata hiding, eligibility-history coefficients, and immediate reduction to the Phase 3A normalized update. Bind the exact passing Lean SHA into a read-only formal-contract manifest in `qigao/yolo-motion-perception`. Then add Phase 3C-only Python modules for hidden-delay scheduling/aggregation, learner arms, protocol controls, and benchmark orchestration. Gate P must pass on the exact Python head before any registered post-training behavior is evaluated. A mandatory human checkpoint separates structural validation from the first preregistered behavioral measurement.

**Tech Stack:** Lean 4.32.0, Mathlib v4.32.0, Lake, Python 3.10–3.12, NumPy >=1.26, pytest >=8, Ruff, setuptools, GitHub Actions.

**Spec:** `qigao/yolo-motion-perception:docs/superpowers/specs/2026-09-16-phase-3c-anonymous-temporal-credit-design.md`

## Global Constraints

- Python work is based on `qigao/yolo-motion-perception` branch `experiment/phase3c-anonymous-temporal-credit`, whose approved scientific base is clean Phase 3B head `a5ab079d56fe569aded44348f9591d226ce83009`.
- Formal work starts from `qigao/lean@c59c1a8df9ece630a7f16747475554f62700ecb4` on a new execution branch named `formal/anonymous-temporal-credit-v2`.
- Do not modify the existing formal v1 files `NarrativeDynamics/Core/TemporalCredit.lean` or `NarrativeDynamics/Tests/TemporalCredit.lean`. Phase 3C must extend the theory in a new module.
- Do not modify `qigao/lean/lean-toolchain` or `qigao/lean/lakefile.toml`.
- Keep these Python files byte-for-byte frozen throughout Phase 3C implementation: `src/neural_state_machine/action_value.py`, `src/neural_state_machine/policy.py`, `src/neural_state_machine/reward_readout.py`, `src/neural_state_machine/reward_learning.py`, `src/neural_state_machine/learning_diagnostics.py`, `src/neural_state_machine/memory_task.py`, `src/neural_state_machine/memory_probe.py`, and `src/neural_state_machine/memory_benchmark.py`.
- Keep all existing Phase 2, Phase 3A, and Phase 3B evidence artifacts byte-for-byte frozen. In particular do not rewrite `docs/experiments/phase-3b-delayed-credit.json` or its report.
- Do not reuse `DelayedTD0Adapter`, `_PendingCredit`, or any FIFO unresolved-credit queue inside a Phase 3C learner.
- The Phase 3C learner receives only decision-time hidden vectors, legal action indices, the behavior-action RNG at selection, and one finite aggregate scalar at feedback. It must not receive source identity, source count, latent rewards, delay, due step, queue state, timestamps, a delivery-present flag, cue identity, correct action, or cue-to-decision delay.
- Registered hidden delays are independently pre-generated from `(1, 3, 5)` with equal probability from `np.random.SeedSequence([seed, 0x3343444C])` before action selection begins.
- Keep existing lineages: training fixture `[seed, 0x54524149]`, evaluation fixture `[seed, 0x4556414C]`, action RNG `[seed, 0x33414354]`, shuffled-control RNG `[seed, 0x33534846]`.
- Registered seeds are ordered `(7, 17, 29)`. Registered training decisions are `2000`; evaluation episodes are `200`; cue-to-decision delays are `1..5`; hidden size `64`; recurrent radius `0.9`; step size `0.1`; gamma `0.9`; lambda `0.8`; rho `0.72`; checkpoint interval `100` real decisions.
- Every real decision step must call the learner exactly once with one aggregate scalar, including `0.0`. Environment audit objects must never cross the learner API.
- Eligibility decay is decision-based. Terminal-drain steps add no trace contribution and cause no trace decay.
- Arm A drain feedback is a parameter no-op because no current decision exists. Arm B drain feedback updates against the persistent pre-drain trace without adding/decaying trace state.
- Both primary arms must exactly reproduce frozen Phase 3A when all hidden delays are zero and `rho=0`.
- Gate F failure stops all Python scientific execution. Gate P failure means `harness invalid`; do not inspect or serialize registered behavioral accuracy from that head.
- Do not run the first registered Phase 3C behavioral measurement until the human explicitly approves the Gate F + Gate P structural checkpoint in Task 8.
- Small unit/integration fixtures used before Task 8 are protocol tests, not registered Phase 3C behavioral measurement. They must not be interpreted scientifically.
- No parameter sweep, threshold change, delay-support change, seed change, task change, reservoir change, or post-result tuning is allowed.
- Never claim Lean proves NumPy floating-point execution. Lean proves the mathematical contract; Python conformance tests prove the implementation follows selected observable consequences of that contract.

## Cross-Repository File Map

### `qigao/lean`

| Path | Responsibility |
|---|---|
| `NarrativeDynamics/Core/AnonymousTemporalCredit.lean` | Anonymous delivery buckets, aggregate stream/view, conservation, source hiding, eligibility recurrence reuse, and normalized immediate-reduction theorem. |
| `NarrativeDynamics/Tests/AnonymousTemporalCredit.lean` | F1–F5 theorem examples, multiplicity/source-relabel examples, registered coefficient examples, and `#print axioms`. |
| `NarrativeDynamics.lean` | Export the new formal module after focused proof compiles. |
| `.github/workflows/proof.yml` | Permanent focused Anonymous Temporal Credit theorem step before the full Lean library build. |

### `qigao/yolo-motion-perception`

| Path | Responsibility |
|---|---|
| `docs/experiments/phase-3c-formal-contract.json` | Immutable binding to the exact passing Lean SHA and primary theorem names; not behavioral evidence. |
| `src/neural_state_machine/phase3c_formal_contract.py` | Fail-closed loader/schema validation for the formal contract. |
| `src/neural_state_machine/phase3c_schedule.py` | Hidden-delay schedule, internal latent reward records, aggregation, terminal drain, and structural digests. |
| `src/neural_state_machine/phase3c_learners.py` | Arm A anonymous current-step TD(0) and Arm B normalized anonymous eligibility credit; no FIFO history. |
| `src/neural_state_machine/phase3c_controls.py` | Gate P audits, non-interference controls, conservation, lineage checks, trace/reset/conformance checks. |
| `src/neural_state_machine/phase3c_benchmark.py` | Fixed fixtures, matched arms/controls, protocol-only execution, continuity comparison, and later behavioral evaluation. |
| `tests/test_phase3c_formal_contract.py` | Formal-contract schema/SHA/theorem fail-closed tests. |
| `tests/test_phase3c_schedule.py` | Schedule support, pre-generation, inversions, collisions, aggregation, exactly-once delivery, drain. |
| `tests/test_phase3c_learners.py` | Arm A/B update equations, anonymity, drain behavior, reset lifecycle, immediate boundary. |
| `tests/test_phase3c_controls.py` | Gate P structural validation, source relabeling, hidden multiplicity, conservation, conformance. |
| `tests/test_phase3c_benchmark.py` | Matched lineages, exact Phase 3A continuity, protocol-only path, controls, repeatability. |
| `scripts/benchmark_phase3c_anonymous_credit.py` | Gate F/Gate P CLI first; later the only approved registered measurement writer. |
| `scripts/verify_phase3c_anonymous_credit.py` | Fail-closed committed Phase 3C artifact verifier added before the first measurement. |
| `tests/test_phase3c_evidence.py` | Payload/writer/verifier mutation tests before any registered artifact exists. |
| `.github/workflows/ci.yml` | Python 3.10/3.11/3.12 focused Phase 3C tests/Ruff and protocol-only gate; later frozen-artifact verifier only. |
| `src/neural_state_machine/__init__.py` | Export stable Phase 3C public result/config types only after protocol interfaces settle. |
| `docs/experiments/phase-3c-anonymous-temporal-credit.json` | First valid registered behavioral measurement, created only in Task 10. |
| `docs/experiments/phase-3c-anonymous-temporal-credit-report.md` | Bounded interpretation written from the immutable Task 10 artifact. |
| `README.md` | Final measured Phase 3C result and reproducibility commands, added only after Task 10. |

---

## Task 1: Prove anonymous aggregation and source non-interference in Lean

**Repository:** `qigao/lean`

**Branch:** Create `formal/anonymous-temporal-credit-v2` from exact commit `c59c1a8df9ece630a7f16747475554f62700ecb4`.

**Files:**
- Create: `NarrativeDynamics/Core/AnonymousTemporalCredit.lean`
- Create: `NarrativeDynamics/Tests/AnonymousTemporalCredit.lean`
- Frozen: `NarrativeDynamics/Core/TemporalCredit.lean`

**Interfaces:**

Use a separate namespace and import v1:

```lean
import NarrativeDynamics.Core.TemporalCredit

namespace NarrativeDynamics.AnonymousTemporalCredit

structure Delivery where
  source : Nat
  reward : ℝ
  deriving DecidableEq

abbrev DeliveryBucket := List Delivery
abbrev DeliveryHistory := List DeliveryBucket

def aggregateBucket (bucket : DeliveryBucket) : ℝ :=
  (bucket.map Delivery.reward).sum

def aggregateStream (history : DeliveryHistory) : List ℝ :=
  history.map aggregateBucket

structure LearnerView (Obs : Type) where
  observations : List Obs
  feedback : List ℝ
  deriving DecidableEq

def learnerView (observations : List Obs) (history : DeliveryHistory) : LearnerView Obs :=
  { observations := observations, feedback := aggregateStream history }
```

Do not put source count, source ID, due step, or delay into `LearnerView`.

- [ ] **Step 1: Write RED theorem tests before the module exists.**

Create `NarrativeDynamics/Tests/AnonymousTemporalCredit.lean` importing the missing core module and reference the required theorem names:

```lean
example (obs : List Nat) (left right : DeliveryHistory)
    (h : aggregateStream left = aggregateStream right) :
    learnerView obs left = learnerView obs right := by
  exact aggregate_view_source_noninterference obs left right h

example (history : DeliveryHistory) :
    (aggregateStream history).sum =
      (history.flatten.map Delivery.reward).sum := by
  exact aggregate_conservation history
```

Add a concrete multiplicity example whose learner-visible stream is equal despite a different internal decomposition:

```lean
private def multiplicityA : DeliveryHistory :=
  [[{source := 0, reward := 1}, {source := 1, reward := -1}], []]

private def multiplicityB : DeliveryHistory :=
  [[], []]

example : aggregateStream multiplicityA = aggregateStream multiplicityB := by decide
```

If `by decide` is not accepted for real-valued equality, prove this concrete equality with `norm_num [aggregateStream, aggregateBucket, multiplicityA, multiplicityB]`.

Add a source-relabel theorem test:

```lean
example (obs : List Nat) (history : DeliveryHistory) (f : Nat → Nat) :
    learnerView obs (relabelSources f history) = learnerView obs history := by
  exact learner_view_source_relabel_invariant obs history f
```

- [ ] **Step 2: Run RED.**

```bash
lake env lean NarrativeDynamics/Tests/AnonymousTemporalCredit.lean
```

Expected: import/module or identifier failures because `AnonymousTemporalCredit` does not exist.

- [ ] **Step 3: Implement the minimum aggregation model.**

Add:

```lean
def relabelDelivery (f : Nat → Nat) (delivery : Delivery) : Delivery :=
  { delivery with source := f delivery.source }

def relabelSources (f : Nat → Nat) (history : DeliveryHistory) : DeliveryHistory :=
  history.map (List.map (relabelDelivery f))
```

Prove F1 by the learner-view definition:

```lean
theorem aggregate_view_source_noninterference
    (observations : List Obs)
    (left right : DeliveryHistory)
    (h : aggregateStream left = aggregateStream right) :
    learnerView observations left = learnerView observations right := by
  simp [learnerView, h]
```

Prove F2 by induction over delivery buckets. The target theorem is exactly:

```lean
theorem aggregate_conservation (history : DeliveryHistory) :
    (aggregateStream history).sum =
      (history.flatten.map Delivery.reward).sum := by
  ...
```

The right side is the formal “every listed latent record exactly once” representation. The Python scheduler must separately prove that its delivery buckets contain each generated record exactly once.

Prove source relabeling does not alter reward aggregation, then derive learner-view invariance:

```lean
theorem aggregate_stream_relabel_sources
    (f : Nat → Nat) (history : DeliveryHistory) :
    aggregateStream (relabelSources f history) = aggregateStream history := by
  ...

theorem learner_view_source_relabel_invariant
    (observations : List Obs) (history : DeliveryHistory) (f : Nat → Nat) :
    learnerView observations (relabelSources f history) = learnerView observations history := by
  exact aggregate_view_source_noninterference _ _ _
    (aggregate_stream_relabel_sources f history)
```

F3 is represented by the stronger F1 statement: any internal history change, including out-of-order source identity/decomposition, is unobservable if the aggregate feedback stream is unchanged. The concrete test module must include two histories with different source ordering but the same aggregate stream and instantiate F1.

- [ ] **Step 4: Verify focused GREEN.**

```bash
lake build NarrativeDynamics.Core.AnonymousTemporalCredit
lake env lean NarrativeDynamics/Tests/AnonymousTemporalCredit.lean
```

Expected: exit 0, no `sorry`/`admit`.

- [ ] **Step 5: Commit only Task 1 proof files.**

```bash
git add NarrativeDynamics/Core/AnonymousTemporalCredit.lean \
        NarrativeDynamics/Tests/AnonymousTemporalCredit.lean
git commit -m "feat(lean): model anonymous aggregate credit"
git push origin formal/anonymous-temporal-credit-v2
```

---

## Task 2: Prove trace reachability and exact immediate reduction; make Gate F permanent

**Repository:** `qigao/lean`

**Files:**
- Modify: `NarrativeDynamics/Core/AnonymousTemporalCredit.lean`
- Modify: `NarrativeDynamics/Tests/AnonymousTemporalCredit.lean`
- Modify: `NarrativeDynamics.lean`
- Modify: `.github/workflows/proof.yml`
- Frozen: `NarrativeDynamics/Core/TemporalCredit.lean`

**Interfaces:**

Reuse the proven v1 coefficient; do not duplicate a second recurrence under different semantics.

Model the normalized update with finite vectors:

```lean
abbrev Vector (n : Nat) := Fin n → ℝ

structure TraceState (n : Nat) where
  eligibility : Vector n
  prediction : ℝ

def zeroVector : Vector n := fun _ => 0

def normalizedCredit (feature : Vector n) (denominator : ℝ) : Vector n :=
  fun i => feature i / denominator

def advanceTrace
    (rho : ℝ) (state : TraceState n)
    (currentCredit : Vector n) (currentPrediction : ℝ) : TraceState n :=
  {
    eligibility := fun i => rho * state.eligibility i + currentCredit i
    prediction := rho * state.prediction + currentPrediction
  }

def normalizedPhase3AUpdate
    (weights : Vector n) (alpha reward prediction : ℝ)
    (credit : Vector n) : Vector n :=
  fun i => weights i + alpha * (reward - prediction) * credit i

def anonymousEligibilityUpdate
    (weights : Vector n) (alpha feedback : ℝ)
    (state : TraceState n) : Vector n :=
  fun i => weights i + alpha * (feedback - state.prediction) * state.eligibility i
```

- [ ] **Step 1: Add RED theorem tests.**

Require these public theorems before implementing them:

```lean
theorem eligibility_historical_coefficient
    (gamma lambda : ℝ) (d : Nat) :
    TemporalCredit.causalTraceCoeff gamma lambda d = (gamma * lambda) ^ d

theorem zero_rho_trace_is_current_credit
    (credit : Vector n) (prediction : ℝ) :
    advanceTrace 0 {eligibility := zeroVector, prediction := 0}
      credit prediction =
      {eligibility := credit, prediction := prediction}

theorem immediate_reduction_to_phase3a
    (weights feature : Vector n)
    (denominator alpha reward prediction : ℝ) :
    let credit := normalizedCredit feature denominator
    let state := advanceTrace 0
      {eligibility := zeroVector, prediction := 0}
      credit prediction
    anonymousEligibilityUpdate weights alpha reward state =
      normalizedPhase3AUpdate weights alpha reward prediction credit
```

Add registered coefficient examples at `d=0,1,3,5` with `gamma=0.9`, `lambda=0.8`. Use exact rationals in Lean, e.g. `(9 : ℝ) / 10` and `(8 : ℝ) / 10`, so no binary floating-point claim enters the theorem.

- [ ] **Step 2: Run RED.**

```bash
lake build NarrativeDynamics.Core.AnonymousTemporalCredit
lake env lean NarrativeDynamics/Tests/AnonymousTemporalCredit.lean
```

Expected: unknown theorem/definition failures.

- [ ] **Step 3: Implement F4 by explicit reuse of v1.**

The proof must visibly delegate to the already-proven theorem rather than re-asserting it:

```lean
theorem eligibility_historical_coefficient
    (gamma lambda : ℝ) (d : Nat) :
    TemporalCredit.causalTraceCoeff gamma lambda d = (gamma * lambda) ^ d := by
  exact TemporalCredit.causal_trace_coeff_closed_form gamma lambda d
```

- [ ] **Step 4: Implement F5.**

Prove `zero_rho_trace_is_current_credit` with structure equality/function extensionality and simplification. Then prove `immediate_reduction_to_phase3a` by unfolding `anonymousEligibilityUpdate`, `normalizedPhase3AUpdate`, and the zero-rho trace result; use `funext` if Lean does not close the function equality by simplification.

The theorem is intentionally about the real-number update equation. It does not claim NumPy byte identity; Task 7 separately checks exact same-environment Python continuity.

- [ ] **Step 5: Add axiom audits.**

At the end of `NarrativeDynamics/Tests/AnonymousTemporalCredit.lean`, print axioms for at least:

```lean
#print axioms NarrativeDynamics.AnonymousTemporalCredit.aggregate_view_source_noninterference
#print axioms NarrativeDynamics.AnonymousTemporalCredit.aggregate_conservation
#print axioms NarrativeDynamics.AnonymousTemporalCredit.learner_view_source_relabel_invariant
#print axioms NarrativeDynamics.AnonymousTemporalCredit.eligibility_historical_coefficient
#print axioms NarrativeDynamics.AnonymousTemporalCredit.immediate_reduction_to_phase3a
```

No theorem may depend on `sorryAx` or a project-defined custom axiom.

- [ ] **Step 6: Export and add a focused permanent CI step.**

Append to `NarrativeDynamics.lean`:

```lean
import NarrativeDynamics.Core.AnonymousTemporalCredit
```

Add before `Build Lean library` in `.github/workflows/proof.yml`:

```yaml
      - name: Anonymous temporal credit theorem tests
        run: |
          export PATH="$HOME/.elan/bin:$PATH"
          lake build NarrativeDynamics.Core.AnonymousTemporalCredit
          lake env lean NarrativeDynamics/Tests/AnonymousTemporalCredit.lean
```

Do not remove or weaken the existing TemporalCredit v1 focused gate.

- [ ] **Step 7: Run local/focused verification when available.**

```bash
lake build NarrativeDynamics.Core.AnonymousTemporalCredit
lake env lean NarrativeDynamics/Tests/AnonymousTemporalCredit.lean
lake build
git grep -nE '\b(sorry|admit)\b' -- \
  NarrativeDynamics/Core/AnonymousTemporalCredit.lean \
  NarrativeDynamics/Tests/AnonymousTemporalCredit.lean || true
git diff --check
```

The grep must produce no proof holes.

- [ ] **Step 8: Commit and require exact-head GitHub proof CI.**

```bash
git add NarrativeDynamics/Core/AnonymousTemporalCredit.lean \
        NarrativeDynamics/Tests/AnonymousTemporalCredit.lean \
        NarrativeDynamics.lean .github/workflows/proof.yml
git commit -m "feat(lean): prove anonymous temporal credit gate"
git push origin formal/anonymous-temporal-credit-v2
```

Record `LEAN_HEAD=$(git rev-parse HEAD)` and require the GitHub Actions proof run for that exact SHA to pass both `Lean proof` and repository Python regression jobs. Read the `#print axioms` log before proceeding. If Gate F fails, stop; do not start Task 3.

---

## Task 3: Bind the exact Gate F proof into the Python experiment

**Repository:** `qigao/yolo-motion-perception`

**Files:**
- Create: `docs/experiments/phase-3c-formal-contract.json`
- Create: `src/neural_state_machine/phase3c_formal_contract.py`
- Create: `tests/test_phase3c_formal_contract.py`

**Interfaces:**

The JSON contract is generated only after Task 2 exact-head CI succeeds. It contains the literal `LEAN_HEAD` captured in Task 2 and this fixed shape:

```json
{
  "schema_version": 1,
  "repository": "qigao/lean",
  "module": "NarrativeDynamics/Core/AnonymousTemporalCredit.lean",
  "commit": "<captured exact 40-hex Task 2 SHA written literally at execution time>",
  "lean_version": "4.32.0",
  "mathlib_version": "v4.32.0",
  "exact_head_ci_passed": true,
  "axiom_audit_reviewed": true,
  "sorry_ax_present": false,
  "custom_axiom_present": false,
  "theorems": [
    "NarrativeDynamics.AnonymousTemporalCredit.aggregate_view_source_noninterference",
    "NarrativeDynamics.AnonymousTemporalCredit.aggregate_conservation",
    "NarrativeDynamics.AnonymousTemporalCredit.learner_view_source_relabel_invariant",
    "NarrativeDynamics.AnonymousTemporalCredit.eligibility_historical_coefficient",
    "NarrativeDynamics.AnonymousTemporalCredit.immediate_reduction_to_phase3a"
  ]
}
```

The angle-bracket explanation above is not copied into the committed file: the executor writes the exact 40-hex SHA returned by `git rev-parse HEAD`.

`phase3c_formal_contract.py` exposes:

```python
@dataclass(frozen=True)
class Phase3CFormalContract:
    commit: str
    theorems: tuple[str, ...]
    exact_head_ci_passed: bool
    axiom_audit_reviewed: bool


def load_phase3c_formal_contract(root: Path | None = None) -> Phase3CFormalContract:
    ...
```

- [ ] **Step 1: Write fail-closed RED tests.**

Test the loader rejects:
- non-40-hex commit;
- wrong repository/module;
- wrong Lean/Mathlib versions;
- missing theorem;
- `exact_head_ci_passed != True`;
- `axiom_audit_reviewed != True`;
- `sorry_ax_present != False`;
- `custom_axiom_present != False`;
- unapproved/symlink contract path.

Also test the committed contract loads and returns the exact SHA captured from Task 2.

- [ ] **Step 2: Run RED.**

```bash
pytest -q tests/test_phase3c_formal_contract.py
```

Expected: module/loader missing.

- [ ] **Step 3: Implement the minimum loader and committed manifest.**

Use only stdlib `json`, `pathlib`, and a fixed theorem-name tuple. No network call is part of runtime conformance. The human/CI evidence from Task 2 establishes the remote proof status; this artifact binds Python evidence to that exact formal head.

- [ ] **Step 4: Verify and commit.**

```bash
pytest -q tests/test_phase3c_formal_contract.py
ruff check src/neural_state_machine/phase3c_formal_contract.py tests/test_phase3c_formal_contract.py
git diff --check
git add docs/experiments/phase-3c-formal-contract.json \
        src/neural_state_machine/phase3c_formal_contract.py \
        tests/test_phase3c_formal_contract.py
git commit -m "docs: bind phase 3c formal gate"
git push origin experiment/phase3c-anonymous-temporal-credit
```

---

## Task 4: Implement hidden variable-delay scheduling and anonymous aggregation

**Repository:** `qigao/yolo-motion-perception`

**Files:**
- Create: `src/neural_state_machine/phase3c_schedule.py`
- Create: `tests/test_phase3c_schedule.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class HiddenDelaySchedule:
    delays: tuple[int, ...]
    due_steps: tuple[int, ...]
    delay_digest: str
    due_step_digest: str
    inversion_count: int
    collision_step_count: int
    multiplicity_histogram: tuple[tuple[int, int], ...]

@dataclass(frozen=True)
class LatentRewardRecord:
    source_step: int
    source_action: int
    reward: float
    delay: int
    due_step: int
    delivery_step: int

@dataclass(frozen=True)
class AggregateFeedback:
    delivery_step: int
    value: float
    multiplicity: int
    records: tuple[LatentRewardRecord, ...]


def build_hidden_delay_schedule(
    seed: int,
    decision_count: int,
    *,
    support: tuple[int, ...] = (1, 3, 5),
) -> HiddenDelaySchedule:
    ...

class AnonymousRewardAggregator:
    def __init__(self, schedule: HiddenDelaySchedule) -> None: ...
    def enqueue(self, source_step: int, source_action: int, reward: float) -> None: ...
    def feedback_at(self, delivery_step: int) -> AggregateFeedback: ...
    @property
    def pending_count(self) -> int: ...
```

The benchmark, never the learner, may inspect `AggregateFeedback.records` and `.multiplicity`. The learner is passed only `.value`.

- [ ] **Step 1: Write RED schedule tests.**

Test that `build_hidden_delay_schedule(seed, 2000)`:
- is deterministic for the same seed;
- consumes only `(1,3,5)`;
- contains each support value at least once for registered seeds `7,17,29`;
- has `inversion_count > 0` and `collision_step_count > 0` for each registered seed;
- produces identical bytes/digests independent of any action RNG activity.

Add a direct explicit schedule fixture such as delays `(5,1,3,1)` and assert its exact due steps, inversion count, collision structure, and multiplicity histogram.

- [ ] **Step 2: Write RED aggregation tests.**

Using an explicit schedule, enqueue records with rewards that demonstrate:
- out-of-order source delivery;
- at least one collision;
- `feedback_at(t)` returns exactly one scalar even when no records are due (`0.0`);
- multiple rewards are summed and not individually surfaced through the scalar;
- a second `feedback_at(t)` cannot redeliver the same records;
- invalid/non-finite reward preserves state;
- terminal drain empties every record without synthetic enqueue.

- [ ] **Step 3: Run RED.**

```bash
pytest -q tests/test_phase3c_schedule.py
```

Expected: missing Phase 3C schedule module.

- [ ] **Step 4: Implement schedule pre-generation.**

Use exactly:

```python
rng = np.random.default_rng(np.random.SeedSequence([seed, 0x3343444C]))
indices = rng.integers(len(support), size=decision_count)
delays = tuple(support[int(index)] for index in indices)
due_steps = tuple(step + delay for step, delay in enumerate(delays))
```

Compute inversion count from source-order pairs whose due order is reversed. Compute collisions/multiplicity from `Counter(due_steps)`. Hash typed integer sequences canonically rather than `repr()`.

- [ ] **Step 5: Implement exactly-once aggregation.**

Internally own pending records keyed by due step. `enqueue` verifies that `source_step` matches the schedule index and derives the registered delay/due step itself; callers cannot override delay/due step. `feedback_at(step)` atomically removes all records due at that step, stamps `delivery_step=step`, sums rewards with Python/NumPy finite-value checks, and returns one immutable `AggregateFeedback`. For empty buckets return value `0.0`, multiplicity `0`, records `()`.

- [ ] **Step 6: Verify and commit.**

```bash
pytest -q tests/test_phase3c_schedule.py
ruff check src/neural_state_machine/phase3c_schedule.py tests/test_phase3c_schedule.py
git diff --check
git add src/neural_state_machine/phase3c_schedule.py tests/test_phase3c_schedule.py
git commit -m "feat: add anonymous reward schedule"
git push origin experiment/phase3c-anonymous-temporal-credit
```

---

## Task 5: Implement Phase 3C-only learner arms without FIFO credit

**Repository:** `qigao/yolo-motion-perception`

**Files:**
- Create: `src/neural_state_machine/phase3c_learners.py`
- Create: `tests/test_phase3c_learners.py`
- Frozen: `src/neural_state_machine/action_value.py`
- Frozen: `src/neural_state_machine/phase3b_learners.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class AnonymousCreditUpdate:
    reward: float
    prediction: float
    td_error: float
    applied: bool

class AnonymousCurrentStepTD0(NormalizedActionValue):
    def select_for_training(...): ...
    def learn(self, aggregate_reward: object) -> AnonymousCreditUpdate: ...
    def learn_drain(self, aggregate_reward: object) -> AnonymousCreditUpdate: ...

class NormalizedAnonymousEligibilityCredit(NormalizedActionValue):
    def __init__(..., discount=0.9, trace_decay=0.8): ...
    def select_for_training(...): ...
    def learn(self, aggregate_reward: object) -> AnonymousCreditUpdate: ...
    def learn_drain(self, aggregate_reward: object) -> AnonymousCreditUpdate: ...
    def eligibility_snapshot(self) -> np.ndarray: ...
    @property
    def prediction_trace(self) -> float: ...
    @property
    def trace_reset_count(self) -> int: ...
    def end_run(self) -> None: ...
```

`learn_drain(reward)` is a learner lifecycle method, not an environment metadata channel; it carries the same single scalar but signals that no current action was selected. It must not accept source/delay/count/timestamp. If implementation can use a single `learn(reward)` API with internal current-credit state and no extra metadata, prefer that; tests must still prove Arm A no-op and Arm B persistent-trace behavior during drain.

- [ ] **Step 1: Write Arm A RED tests.**

Prove:
- one selection + feedback matches `NormalizedActionValue` byte-for-byte on the same hidden/action RNG/reward;
- selection owns only one current decision credit and never exposes/stores a deque of historical credits;
- current-step `0.0` aggregate is still a real feedback update;
- drain feedback with no current decision leaves weights byte-identical and returns `applied=False`;
- invalid feedback is atomic.

Include an introspection guard that the Phase 3C class has no `_credits` deque and does not subclass `DelayedTD0Adapter`.

- [ ] **Step 2: Write Arm B RED tests.**

With tiny vectors and forced legal actions, verify exact arithmetic:

```text
rho=0.72
E0=0, P0=0
select feature f0/action0 -> E=f0/n0, P=q0
learn F0 -> delta=F0-P
select feature f1/action1 -> E=0.72*E + f1/n1, P=0.72*P + q1
```

Test:
- feedback does not clear `E` or `P`;
- a collision is indistinguishable from any other scalar of the same value;
- `learn_drain` changes weights using the same `E/P` but does not decay/add trace;
- trace is not reset at fixture-like boundaries because no such method exists;
- `end_run()` is legal only after the orchestrator marks the run fully drained, clears `E/P`, and increments reset count exactly once;
- non-finite state/update is rejected atomically.

- [ ] **Step 3: Write exact immediate-boundary RED test.**

For `rho=0`, same forced action, hidden vector, and immediate reward, compare each arm against a fresh `NormalizedActionValue`: action, update scalar fields, full weight matrix bytes, and parameter digest must match.

- [ ] **Step 4: Run RED.**

```bash
pytest -q tests/test_phase3c_learners.py
```

- [ ] **Step 5: Implement Arm A with Phase 3A arithmetic order.**

Reuse protected helper methods `_feature`, `_legal_actions`, `_weights`, `parameter_snapshot`, and `parameter_digest` from `NormalizedActionValue`, but do not call the parent `select_for_training` because its `_pending` lifecycle encodes identified feedback. Store at most one Phase 3C-local current credit containing action, normalized feature, denominator, and prediction. Consume it on the same real step. Never retain it into the next action selection.

- [ ] **Step 6: Implement Arm B global trace.**

At every real selection, compute decision-time feature/denominator/prediction, then:

```python
rho = self.discount * self.trace_decay
candidate_e = self._eligibility * rho
candidate_e[action_index] += feature / denominator
candidate_p = rho * self._prediction_trace + prediction
```

Validate both candidates before mutation. On feedback:

```python
td_error = reward_value - self._prediction_trace
candidate_weights = self._weights + self.step_size * td_error * self._eligibility
```

Do not clear trace after feedback. During drain use the identical update with unchanged trace state. `end_run()` is the only trace clear.

- [ ] **Step 7: Verify frozen files and commit.**

```bash
pytest -q tests/test_phase3c_learners.py tests/test_action_value.py tests/test_phase3b_learners.py
ruff check src/neural_state_machine/phase3c_learners.py tests/test_phase3c_learners.py
git diff -- src/neural_state_machine/action_value.py src/neural_state_machine/phase3b_learners.py
git diff --check
```

Frozen-file diff must be empty.

```bash
git add src/neural_state_machine/phase3c_learners.py tests/test_phase3c_learners.py
git commit -m "feat: add anonymous temporal credit learners"
git push origin experiment/phase3c-anonymous-temporal-credit
```

---

## Task 6: Define fail-closed Gate P controls and Python/formal conformance probes

**Repository:** `qigao/yolo-motion-perception`

**Files:**
- Create: `src/neural_state_machine/phase3c_controls.py`
- Create: `tests/test_phase3c_controls.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class AnonymousProtocolAudit:
    action_count: int
    latent_record_count: int
    delivered_record_count: int
    real_feedback_count: int
    drain_feedback_count: int
    queue_pending_final: int
    delay_histogram: tuple[tuple[int, int], ...]
    inversion_count: int
    collision_step_count: int
    multiplicity_histogram: tuple[tuple[int, int], ...]
    delay_digest: str
    due_step_digest: str
    multiplicity_digest: str
    aggregate_feedback_digest: str
    learner_call_digest: str
    latent_reward_sum: float
    aggregate_feedback_sum: float
    source_relabel_invariant: bool
    hidden_multiplicity_invariant: bool
    trace_reset_count: int
    trace_coefficients: tuple[tuple[int, float], ...]


def validate_anonymous_protocol(audit: AnonymousProtocolAudit, *, action_count: int) -> None:
    ...
```

Use an explicit Python conformance tolerance only for the registered trace coefficients:

```python
_TRACE_ATOL = 1e-15
```

This tolerance is for `0.72**d` Python representation at ages `0,1,3,5`; all discrete counts/digests/conservation on ±1 rewards remain exact.

- [ ] **Step 1: Write RED structural validator tests.**

Construct one valid synthetic audit, then mutate each Gate P field independently and require `ValueError` for:
- latent/delivered count mismatch;
- nonzero final pending count;
- missing support value in delay histogram;
- zero inversions;
- zero collisions;
- wrong real feedback count;
- malformed/changed digests;
- non-conservation;
- false source-relabel or hidden-multiplicity invariant;
- wrong trace reset count for Arm B;
- wrong trace coefficient.

- [ ] **Step 2: Write non-interference RED tests.**

Provide helpers that derive the learner-call scalar stream from environment audit histories. Demonstrate:
- arbitrary source relabeling changes source metadata digest but not learner-call digest;
- `[+1,-1]` collision and an empty delivery bucket both produce scalar `0.0` and identical learner-call encoding at that step;
- changing multiplicity metadata alone never enters the call digest.

- [ ] **Step 3: Write exact conservation RED test.**

Use only latent rewards `±1.0`, so both sums are exactly representable. Require equality with `==`, not tolerance.

- [ ] **Step 4: Run RED.**

```bash
pytest -q tests/test_phase3c_controls.py
```

- [ ] **Step 5: Implement canonical digests and fail-closed validation.**

Use typed canonical encodings (`to_bytes`, float64 bytes where appropriate), not `repr`. `learner_call_digest` hashes only the scalar sequence in call order; it deliberately cannot include multiplicity, source, due step, or delay.

Trace conformance expected values are computed from the registered formal recurrence:

```python
expected = ((0, 1.0), (1, 0.72), (3, 0.72**3), (5, 0.72**5))
```

- [ ] **Step 6: Verify and commit.**

```bash
pytest -q tests/test_phase3c_controls.py tests/test_phase3c_formal_contract.py
ruff check src/neural_state_machine/phase3c_controls.py tests/test_phase3c_controls.py
git diff --check
git add src/neural_state_machine/phase3c_controls.py tests/test_phase3c_controls.py
git commit -m "test: define phase 3c anonymous protocol gate"
git push origin experiment/phase3c-anonymous-temporal-credit
```

---

## Task 7: Build protocol-only orchestration and exact Phase 3A continuity

**Repository:** `qigao/yolo-motion-perception`

**Files:**
- Create: `src/neural_state_machine/phase3c_benchmark.py`
- Create: `tests/test_phase3c_benchmark.py`
- Reuse read-only helpers from: `action_value_benchmark.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class AnonymousCreditConfig:
    hidden_size: int = 64
    recurrent_radius: float = 0.9
    step_size: float = 0.1
    training_decisions: int = 2000
    evaluation_blocks: int = 20
    checkpoint_interval: int = 100
    delay_support: tuple[int, ...] = (1, 3, 5)
    discount: float = 0.9
    trace_decay: float = 0.8

@dataclass(frozen=True)
class Phase3CProtocolResult:
    seed: int
    arm: str
    training_fixture_digest: str
    evaluation_fixture_digest: str
    action_digest: str
    latent_reward_digest: str
    parameter_digest: str
    audit: AnonymousProtocolAudit
    immediate_continuity: bool
    repeatable: bool


def run_phase3c_protocol_gate(
    seeds: tuple[int, ...] = (7, 17, 29),
    config: AnonymousCreditConfig | None = None,
) -> tuple[Phase3CProtocolResult, ...]:
    ...
```

This function MUST NOT compute post-training, state-reset, shuffled accuracy, or `behavior_passed`. Its result type must have no such fields. This is the anti-peeking boundary before Task 8.

- [ ] **Step 1: Write RED API-boundary test.**

Call a reduced unregistered config and assert the protocol result exposes structural fields only. Explicitly assert `not hasattr(result, "post_training")` and `not hasattr(result, "behavior_passed")`.

- [ ] **Step 2: Write RED matched-lineage tests.**

For each small test run:
- pre-generate fixtures with existing `_build_fixture_bundle`;
- pre-generate expected actions from `[seed,0x33414354]` independently of both learners;
- pre-generate one hidden-delay schedule before learner construction;
- run Arm A and Arm B with separately reconstructed action RNGs;
- require identical action sequence/digest, fixture digests, delay/due/multiplicity digests.

- [ ] **Step 3: Write RED timeline/call tests.**

At every real decision:

```text
hidden -> select action -> compute latent reward -> enqueue -> feedback_at(t) -> learner learns scalar
```

Require exactly `training_decisions` real feedback calls. Then drain due steps without selecting actions. Require latent count == delivered count == action count and pending zero.

- [ ] **Step 4: Write RED exact immediate continuity test for both arms.**

Add an internal protocol-control mode that supplies an all-zero delay schedule and `rho=0`. For every registered seed compare with `run_action_value_experiment(seed, config.action_value_config)`:
- actions and action digest;
- latent rewards and reward digest;
- final parameter digest;
- training/evaluation fixture digests.

At this protocol-only stage, also invoke the existing Phase 3A evaluation result solely for continuity comparison. Do not evaluate anonymous `[1,3,5]` runs behaviorally.

- [ ] **Step 5: Run RED.**

```bash
pytest -q tests/test_phase3c_benchmark.py
```

- [ ] **Step 6: Implement select-before-aggregate orchestration.**

Normal anonymous runs must never inspect audit metadata when calling the learner. Use only:

```python
feedback = aggregator.feedback_at(decision_step)
update = learner.learn(feedback.value)
```

During drain call the learner with scalar values according to Task 5 semantics, but never pass the `AggregateFeedback` object.

After the queue empties, call `end_run()` exactly once for Arm B and record its trace reset audit. Do not let `end_run()` affect measured parameter digest; capture final trained parameter digest before trace teardown or prove teardown changes trace only.

- [ ] **Step 7: Build Gate P audit before any accuracy path.**

For each arm/seed compute schedule, reward, aggregate, call-stream and trace audits, then call `validate_anonymous_protocol`. Any exception aborts the run before a behavioral result object can be created.

- [ ] **Step 8: Verify protocol-only GREEN.**

```bash
pytest -q \
  tests/test_phase3c_formal_contract.py \
  tests/test_phase3c_schedule.py \
  tests/test_phase3c_learners.py \
  tests/test_phase3c_controls.py \
  tests/test_phase3c_benchmark.py
ruff check \
  src/neural_state_machine/phase3c_formal_contract.py \
  src/neural_state_machine/phase3c_schedule.py \
  src/neural_state_machine/phase3c_learners.py \
  src/neural_state_machine/phase3c_controls.py \
  src/neural_state_machine/phase3c_benchmark.py \
  tests/test_phase3c_*.py
git diff --check
```

- [ ] **Step 9: Commit.**

```bash
git add src/neural_state_machine/phase3c_benchmark.py tests/test_phase3c_benchmark.py
git commit -m "feat: build phase 3c protocol gate"
git push origin experiment/phase3c-anonymous-temporal-credit
```

---

## Task 8: Put Gate F + Gate P on exact-head CI, then STOP for human structural review

**Repository:** `qigao/yolo-motion-perception`

**Files:**
- Create: `scripts/benchmark_phase3c_anonymous_credit.py`
- Modify: `.github/workflows/ci.yml`
- Do not create: `docs/experiments/phase-3c-anonymous-temporal-credit.json`

**Interfaces:**

Initially the CLI supports only structural gating:

```bash
python scripts/benchmark_phase3c_anonymous_credit.py \
  --require-formal-valid \
  --require-protocol-valid
```

It may print structural JSON but must not calculate registered post/reset/shuffled accuracy and must not accept an evidence-writing flag yet.

- [ ] **Step 1: Write CLI RED tests in `tests/test_phase3c_benchmark.py`.**

Test that the protocol payload contains:
- `formal_valid`;
- exact Lean SHA from the committed formal contract;
- `protocol_valid`;
- structural rows for both arms and all ordered seeds;
- no `behavior_passed` key;
- no post/reset/shuffled fields.

- [ ] **Step 2: Run RED.**

```bash
pytest -q tests/test_phase3c_benchmark.py
```

- [ ] **Step 3: Implement protocol-only CLI.**

`--require-formal-valid` exits nonzero unless the bound formal contract is valid. `--require-protocol-valid` exits nonzero unless every registered structural row passes Gate P. There is no writer in this task.

- [ ] **Step 4: Add permanent Phase 3C structural CI steps.**

After existing Phase 3B verification add:

```yaml
      - name: Phase 3C anonymous credit tests
        run: pytest -q tests/test_phase3c_formal_contract.py tests/test_phase3c_schedule.py tests/test_phase3c_learners.py tests/test_phase3c_controls.py tests/test_phase3c_benchmark.py

      - name: Phase 3C Ruff
        run: ruff check src/neural_state_machine/phase3c_formal_contract.py src/neural_state_machine/phase3c_schedule.py src/neural_state_machine/phase3c_learners.py src/neural_state_machine/phase3c_controls.py src/neural_state_machine/phase3c_benchmark.py scripts/benchmark_phase3c_anonymous_credit.py tests/test_phase3c_formal_contract.py tests/test_phase3c_schedule.py tests/test_phase3c_learners.py tests/test_phase3c_controls.py tests/test_phase3c_benchmark.py

      - name: Phase 3C formal and protocol gate
        run: python scripts/benchmark_phase3c_anonymous_credit.py --require-formal-valid --require-protocol-valid
```

Do not add a measurement/evidence step.

- [ ] **Step 5: Run full local verification when available.**

```bash
pytest -q
ruff check .
python scripts/benchmark_phase3c_anonymous_credit.py \
  --require-formal-valid --require-protocol-valid
git diff --check
git diff a5ab079d56fe569aded44348f9591d226ce83009 -- \
  src/neural_state_machine/action_value.py \
  src/neural_state_machine/policy.py \
  src/neural_state_machine/reward_learning.py \
  src/neural_state_machine/memory_task.py \
  docs/experiments/phase-3b-delayed-credit.json
```

The frozen-path diff must be empty.

- [ ] **Step 6: Commit and require exact-head Python 3.10/3.11/3.12 CI.**

```bash
git add scripts/benchmark_phase3c_anonymous_credit.py .github/workflows/ci.yml
git commit -m "ci: enforce phase 3c structural gates"
git push origin experiment/phase3c-anonymous-temporal-credit
```

Require all three Python matrix jobs on the exact pushed SHA to pass the complete existing regression suite plus the new Phase 3C gate.

- [ ] **Step 7: Produce the mandatory structural review packet and STOP.**

Report without any anonymous-condition behavioral accuracy:
- exact Lean Gate F SHA and exact-head proof run;
- new theorem names and exact `#print axioms` output;
- exact Python head and CI run;
- formal-contract SHA binding;
- per-seed delay histogram/support coverage;
- per-seed inversion count;
- per-seed collision count/multiplicity histogram;
- action/latent/delivered counts;
- terminal drain counts;
- exact conservation status;
- source-relabel and hidden-multiplicity non-interference status;
- learner-call/delay/due/multiplicity digests shared across matched arms;
- immediate Phase 3A continuity digests for both arms;
- trace reset count and `d=0,1,3,5` coefficient conformance.

**HARD STOP:** Do not execute Task 9 or Task 10 until the human explicitly approves this structural packet with `go`/`批准` or equivalent. Do not inspect anonymous-condition post-training behavior as part of this checkpoint.

---

## Task 9: After structural approval, add the preregistered behavioral/evidence surface without running the registered measurement

**Repository:** `qigao/yolo-motion-perception`

**Precondition:** Human approval after Task 8.

**Files:**
- Modify: `src/neural_state_machine/phase3c_benchmark.py`
- Modify: `scripts/benchmark_phase3c_anonymous_credit.py`
- Create: `scripts/verify_phase3c_anonymous_credit.py`
- Create: `tests/test_phase3c_evidence.py`
- Modify: `tests/test_phase3c_benchmark.py`
- Modify: `src/neural_state_machine/__init__.py`
- Do not create yet: registered Phase 3C evidence JSON/report.

**Interfaces:**

Add a separate behavioral entry point so structural execution cannot accidentally peek:

```python
@dataclass(frozen=True)
class Phase3CMeasurementResult:
    seed: int
    arm: str
    protocol: Phase3CProtocolResult
    post_training: AccuracyCount
    state_reset: AccuracyCount
    shuffled_control: AccuracyCount
    per_delay: tuple[tuple[int, AccuracyCount], ...]
    reset_per_delay: tuple[tuple[int, AccuracyCount], ...]
    shuffled_per_delay: tuple[tuple[int, AccuracyCount], ...]
    behavior_passed: bool


def run_phase3c_measurement(...): ...
```

This function first obtains/validates Gate F + Gate P on the exact same run inputs, then performs `_evaluate` only if protocol validity is true.

- [ ] **Step 1: Write RED behavior-boundary tests with unregistered reduced configs.**

Use tiny configs solely to test plumbing. Prove that:
- a forced protocol-invalid row raises before `_evaluate` can run (spy/monkeypatch `_evaluate` to fail if called);
- valid small runs produce post/reset/shuffled count shapes;
- normal and shuffled controls use identical actions/delay/due/multiplicity structure;
- shuffled latent rewards preserve the registered block multiset before aggregation but alter reward assignment for a known non-identity test seed;
- behavioral threshold implementation is exactly `>=180/200`, every `>=34/40`, reset `100/200` + every `20/40`, shuffled `<150/200` when using full-size count fixtures.

- [ ] **Step 2: Implement matched shuffled/state-reset controls.**

Compute the normal latent reward sequence from fixed fixtures + pre-generated action lineage. Permute latent rewards with `[seed,0x33534846]` while reusing the same hidden-delay schedule. Normal and shuffled due-step and multiplicity digests must remain identical.

State-reset evaluation reuses the established `_evaluate(..., reset_before_decision=True)` and never changes the trained learner.

- [ ] **Step 3: Define fail-closed payload schema.**

Top-level fields must include:

```text
experiment = phase-3c-anonymous-temporal-credit
schema_version = 1
phase3b_base_sha = a5ab079d56fe569aded44348f9591d226ce83009
formal_contract = exact committed Gate F SHA/theorem audit
formal_valid
protocol_valid
behavior_passed
all_passed = formal_valid and protocol_valid and behavior_passed
```

Rows include all evidence fields pre-registered in the design. `behavior_passed` is forced false if either formal/protocol validity is false.

- [ ] **Step 4: Add an approved-path writer that still is not invoked.**

Only this path is legal:

`docs/experiments/phase-3c-anonymous-temporal-credit.json`

The writer refuses unless `formal_valid is True` and `protocol_valid is True`, writes canonical sorted JSON atomically, rejects symlinks, and permits `behavior_passed=False` because a valid negative result must be frozen.

- [ ] **Step 5: Add fail-closed verifier and mutation tests.**

The verifier rejects wrong base SHA, formal SHA, theorem list, schema, seeds/config, missing structural fields, duplicate arm/seed rows, invalid digests, inconsistent top-level booleans, protocol invalidity, or deterministic portable-projection mismatch. As with Phase 3B, environment-specific raw parameter digests may be excluded only from the portable projection, not from the committed artifact itself.

- [ ] **Step 6: Verify only tests and reduced non-registered fixtures.**

```bash
pytest -q tests/test_phase3c_*.py
ruff check src/neural_state_machine/phase3c_*.py scripts/benchmark_phase3c_anonymous_credit.py scripts/verify_phase3c_anonymous_credit.py tests/test_phase3c_*.py
git diff -- docs/experiments/phase-3c-anonymous-temporal-credit.json
git diff --check
```

The registered evidence diff must be empty. Do not run the CLI in full registered measurement mode yet.

- [ ] **Step 7: Commit.**

```bash
git add src/neural_state_machine/phase3c_benchmark.py \
        src/neural_state_machine/__init__.py \
        scripts/benchmark_phase3c_anonymous_credit.py \
        scripts/verify_phase3c_anonymous_credit.py \
        tests/test_phase3c_benchmark.py tests/test_phase3c_evidence.py
git commit -m "feat: prepare phase 3c registered measurement"
git push origin experiment/phase3c-anonymous-temporal-credit
```

Require exact-head CI to pass before Task 10. CI still runs only structural Gate F/Gate P plus tests; it must not auto-run or auto-write the registered behavioral measurement.

---

## Task 10: Run exactly one registered behavioral measurement and freeze it whether positive or negative

**Repository:** `qigao/yolo-motion-perception`

**Preconditions:**
- Task 8 structural packet explicitly approved by the human.
- Task 9 exact-head CI passed.
- No registered Phase 3C behavioral artifact exists yet.

**Files:**
- Create exactly once: `docs/experiments/phase-3c-anonymous-temporal-credit.json`
- Create: `docs/experiments/phase-3c-anonymous-temporal-credit-report.md`
- Modify: `README.md`
- Modify: `.github/workflows/ci.yml`

- [ ] **Step 1: Record the exact measurement implementation head before running.**

```bash
MEASURED_HEAD=$(git rev-parse HEAD)
git status --short
```

The working tree must be clean. Record the current formal-contract Lean SHA as well.

- [ ] **Step 2: Run the first registered measurement once.**

Use the fixed default configuration only:

```bash
python scripts/benchmark_phase3c_anonymous_credit.py \
  --measure-registered \
  --evidence docs/experiments/phase-3c-anonymous-temporal-credit.json
```

The command must internally re-check formal validity and protocol validity before behavior evaluation/writing. If either fails, it must refuse the write; classify only as harness/formal invalid and return to design review. Do not rerun with changed parameters.

- [ ] **Step 3: Verify candidate bytes without changing scientific parameters.**

```bash
python scripts/verify_phase3c_anonymous_credit.py \
  --evidence docs/experiments/phase-3c-anonymous-temporal-credit.json
python scripts/benchmark_phase3c_anonymous_credit.py \
  --require-formal-valid --require-protocol-valid
git diff --check
```

- [ ] **Step 4: Write the report from the frozen JSON only.**

The report must include:
- exact measured Python head and formal Lean head;
- Gate F and Gate P status;
- structural inversion/collision/anonymity/conservation evidence;
- exact immediate Phase 3A continuity;
- per-seed Arm A and Arm B post/reset/shuffled counts and cue-delay counts;
- the fixed behavioral thresholds;
- exactly one of the preregistered interpretation classes from the design;
- explicit limits: no general causal discovery, no standard-TD(lambda) claim, no NumPy proof by Lean, no YOLO/production claim.

No post-hoc parameter or threshold adjustment is allowed.

- [ ] **Step 5: Update README and switch CI to permanent artifact verification.**

Keep the structural Gate P command. Add only a read-only verifier step:

```yaml
      - name: Verify frozen Phase 3C anonymous-credit evidence
        run: python scripts/verify_phase3c_anonymous_credit.py
```

Do not put `--measure-registered` or evidence writing in permanent CI.

- [ ] **Step 6: Run final verification.**

```bash
pytest -q
ruff check .
python scripts/verify_reward_learning_failure.py
python scripts/verify_phase3b_delayed_credit.py
python scripts/verify_phase3c_anonymous_credit.py
python scripts/benchmark_phase3c_anonymous_credit.py \
  --require-formal-valid --require-protocol-valid
git diff --check
git diff a5ab079d56fe569aded44348f9591d226ce83009 -- \
  src/neural_state_machine/action_value.py \
  src/neural_state_machine/policy.py \
  src/neural_state_machine/reward_learning.py \
  src/neural_state_machine/memory_task.py \
  docs/experiments/phase-3b-delayed-credit.json
```

Frozen-path diff must remain empty.

- [ ] **Step 7: Commit immutable evidence and require final exact-head CI.**

```bash
git add docs/experiments/phase-3c-anonymous-temporal-credit.json \
        docs/experiments/phase-3c-anonymous-temporal-credit-report.md \
        README.md .github/workflows/ci.yml
git commit -m "docs: freeze phase 3c anonymous temporal credit evidence"
git push origin experiment/phase3c-anonymous-temporal-credit
```

Require Python 3.10/3.11/3.12 exact-head CI success with both Phase 3B verifier and Phase 3C verifier. Record the immutable Phase 3C artifact SHA-256.

---

## Plan Self-Review Checklist

Before Task 1 execution, verify all of the following:

- The plan preserves `qigao/lean@c59c1a8...` TemporalCredit v1 and adds a separate v2 module rather than rewriting established proofs.
- F1 covers arbitrary internal histories with identical aggregate streams, so multiplicity/source decomposition is formally hidden.
- F2 conservation proves aggregation over an exactly-once bucket partition; Python separately proves the scheduler really forms such a partition.
- F3 source identity/order is absent from formal learner view; Python source-relabel and multiplicity controls test the implementation boundary.
- F4 reuses the already-proven `(gamma*lambda)^d` theorem instead of cloning it.
- F5 proves a normalized real-number immediate-reduction equation; Python separately checks exact byte/digest Phase 3A continuity.
- No Lean theorem claims floating-point correctness, empirical convergence, or behavioral success.
- Phase 3C learners do not subclass/reuse Phase 3B FIFO credit behavior.
- Schedule generation happens before actions and uses only `[seed,0x3343444C]`.
- Every real decision generates one scalar learner call, including zero; multiplicity is never a learner argument.
- Out-of-order delivery and collisions are registered protocol requirements, not optional diagnostics.
- Terminal drain creates no action, causes no trace decay, gives Arm A no-op updates, and lets Arm B use persistent trace.
- Arm B trace resets exactly at run lifecycle boundaries, not delayed-cue fixture/reward boundaries.
- Immediate `delay=0,rho=0` continuity is checked for both arms against frozen Phase 3A.
- Gate F and Gate P are permanent read-only CI gates before behavior.
- `run_phase3c_protocol_gate` exposes no behavioral fields, preventing accidental preregistration peeking.
- There is an explicit human stop after Task 8 before any registered behavioral measurement.
- Task 9 may use tiny unregistered fixtures for tests but does not produce the registered artifact.
- Task 10 freezes the first valid registered result whether positive or negative and performs no retuning.
- Existing Phase 2/3A/3B evidence remains unchanged.

## Completion Handoff

At the end of Task 10, report:

- exact Lean Gate F head, workflow run, theorem names, and axiom audit;
- exact Python measured head and final frozen-evidence head;
- Task 1–10 commit subjects;
- formal-contract manifest SHA binding;
- Python 3.10/3.11/3.12 exact-head CI;
- per-seed hidden-delay histograms, inversion counts, collision counts, multiplicity histograms, and terminal drain counts;
- exact aggregate conservation and non-interference results;
- matched arm action/schedule/due/multiplicity digests;
- both arms' exact immediate Phase 3A continuity evidence;
- trace coefficient conformance and reset audit;
- per-seed Arm A/B post/reset/shuffled results and per-cue-delay counts;
- immutable Phase 3C artifact SHA-256;
- exactly one preregistered interpretation, with no stronger claim than the evidence supports.
