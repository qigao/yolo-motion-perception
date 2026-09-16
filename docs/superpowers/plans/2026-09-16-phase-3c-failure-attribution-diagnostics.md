# Phase 3C Failure-Attribution Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Status: committed-plan review.** The user approved the diagnostic spec and preparation of this plan, not implementation or new measurement. All execution checkboxes below are intentionally unchecked.
>
> Repository: `qigao/yolo-motion-perception`; branch: `experiment/phase3c-anonymous-temporal-credit`.
>
> Approved spec commit: `b510be9969be13b0ccc87b7323ceac9cb663d6e3`; unchanged scientific code reference: `55a7ba59ac301975f3df996636f9a596e90bc730`.

**Goal:** Explain, or explicitly leave unresolved, the frozen seed-17 shuffled TD(0) result of `197/200`, separating control structure, representation/readout limitations, anonymous update accounting, and terminal drain without changing the original experiment.

**Architecture:** Add an isolated diagnostic package around the existing concrete learners, fixtures, aggregator and evaluators. Keep source/donor metadata and scoring labels in observer-only records. Establish replay integrity before any new condition; then execute the fixed diagnostic grid, privileged references and update accounting under a sealed manifest. A successful diagnostic may retain both an unresolved cause and the original failed behavioral gate.

**Tech Stack:** Python 3.12 only, NumPy, pytest, Ruff, existing GitHub Actions; standard-library dataclasses, JSON, hashlib and subprocess. No new learning framework or Lean implementation.

---

## 1. Authority, immutable inputs and stop points

The controlling document is [the approved diagnostic spec](../specs/2026-09-16-phase-3c-failure-attribution-diagnostics-design.md) at the spec commit above. Where this plan supplies implementation detail, it must preserve that document's scientific constraints. Any change to seeds, conditions, tolerances, information access or acceptance requires a reviewed amendment, not an implementation convenience.

This commit adds only this plan, uses `[skip ci]`, and does not execute any command below. Test outputs described here are **expected future results**, not reported results. Prior run `35052025233` (#414) verified the scientific code reference, not this documentation commit.

Keep these boundaries separate:

| Gate | Authorized work after that gate | Mandatory stop |
|---|---|---|
| Plan review, current checkpoint | Read and revise this plan | Before production/test implementation |
| Execution approval | Tasks 1-10 implementation/verification, then Task 11 audit and prospective manifest preparation | End of Task 11, before new registered diagnostic measurement |
| Measurement approval of exact implementation and sealed manifest | Task 12: D0, complete D1/D2/D3 measurement and report | Before learner changes, new scientific phase or merge |

The frozen evidence path is `docs/experiments/phase-3c-anonymous-temporal-credit.json`; required SHA-256 is `53b52fb6716daaceeb68b4e5c78f33333c0076b0d727462aa7265b0887798263`. The original measurement run is `35046817410` at `4d13a55545e67aa97ce1e67fd40aaeef044b19a8`; the bound formal reference is `3de297cee2a94a7fc309531334f720f1b34467c9`. Validate actual bytes and binding, not just this prose.

Preserve all existing `docs/experiments` files, both Phase 3C specs, all existing scientific source files, and existing evidence-verifier scripts. Record their Git identities and SHA-256 values before and after every run. Future CI edits are confined to Task 10 and must preserve all original gates. No changes to original learner/scheduler/fixture/reward/verifier implementations, `requires-python`, Ruff's general compatibility target, old reports or old evidence are needed.

Original per-seed values, all with denominator 200:

| Seed | TD(0) normal | Eligibility normal | TD(0) original shuffle | Eligibility original shuffle |
|---|---:|---:|---:|---:|
| 7 | 83 | 104 | 100 | 100 |
| 17 | 121 | 170 | 197 | 100 |
| 29 | 100 | 100 | 100 | 100 |

All six primary rows remain behavior failures. Original thresholds remain normal `>=180/200`, each cue delay `>=34/40`, reset `100/200` and `20/40` per delay, shuffled `<150/200`. Keep the recorded `formal_valid=true`, `protocol_valid=true`, `behavior_passed=false`, `all_passed=false`; a later audit can qualify named claims without overwriting historical fields.

## 2. Read-first source map and proposed file ownership

Read these existing files at the scientific code reference before implementing their adapters:

- `src/neural_state_machine/phase3c_benchmark.py`: `_execute_training`, `_measurement_from_protocol`, `_new_arm`, `_build_audit`; selection precedes delivery, original shuffle is within blocks of ten, parameter digest precedes `end_run`.
- `src/neural_state_machine/phase3c_schedule.py`: `HiddenDelaySchedule`, `LatentRewardRecord`, `AggregateFeedback`, `AnonymousRewardAggregator`; scheduling follows the record slot, not a shuffled reward donor.
- `src/neural_state_machine/phase3c_learners.py`: concrete TD(0)/eligibility classes, public snapshots, `prediction_trace`, `learn_drain`, `end_run`. Do not use a proxy that breaks the existing `isinstance` lifecycle check.
- `src/neural_state_machine/action_value.py`: bias augmentation, zero weights, decision-time prediction, normalization and smaller-index greedy tie rule.
- `src/neural_state_machine/action_value_benchmark.py`: `_build_fixture_bundle`, `_new_policy`, `_evaluate`, `_collect_checkpoint`, `_permute_reward_blocks`, `_episode_digest`, `run_action_value_experiment`.
- `src/neural_state_machine/reward_learning.py`: `_build_fixtures`, `_decision_hidden`, frozen matrix helpers; recurrent state resets per fixture, eligibility does not.
- `scripts/verify_phase3c_anonymous_credit.py`, `.github/workflows/ci.yml`, the frozen Phase 3C report, and `docs/experiments/phase-3a-failure-analysis.md`.

All paths below are **future additions**, except the explicitly identified workflow modification. Use small modules rather than duplicating an experiment in one script.

| Path under `src/neural_state_machine/phase3c_diagnostics/` | Responsibility |
|---|---|
| `__init__.py` | Package description only; importing it never runs experiments |
| `contracts.py` | Typed row identities, fixed grid, config validation, nullable metrics |
| `manifest.py` | Immutable input/environment manifests, safe paths and byte hashes |
| `provenance.py` | Index permutations, slot/donor audit and descriptive association statistics |
| `replay.py` | Diagnostic replay driver and external snapshots; D0 parity and isolation |
| `evaluation.py` | Fixed evaluation bundles, independent snapshot scoring and ordering checks |
| `references.py` | Three information-privileged D2 references, never imported by original learners |
| `accounting.py` | D3 residual, history and drain reconstructions |
| `report.py` | Strict output validation, complete-grid accounting and Markdown report rendering |
| `runner.py` | Stage sequencing, explicit run profiles, attempt provenance and exit states |

Other future files:

```text
scripts/diagnose_phase3c_failure.py
requirements/phase3c-diagnostics.in
requirements/phase3c-diagnostics-python312.lock
.github/workflows/phase3c-diagnostics.yml          # manual measurement only
.github/workflows/ci.yml                         # retain existing gates; add light tests

tests/test_phase3c_diagnostics_contracts.py
tests/test_phase3c_diagnostics_manifest.py
tests/test_phase3c_diagnostics_provenance.py
tests/test_phase3c_diagnostics_replay.py
tests/test_phase3c_diagnostics_isolation.py
tests/test_phase3c_diagnostics_evaluation.py
tests/test_phase3c_diagnostics_statistics.py
tests/test_phase3c_diagnostics_references.py
tests/test_phase3c_diagnostics_accounting.py
tests/test_phase3c_diagnostics_report.py
tests/test_phase3c_diagnostics_runner.py
tests/test_phase3c_diagnostics_ci.py
```

Future retained evidence is confined to `docs/experiments/phase-3c-failure-attribution-v1/`: `manifest.json`, `execution-manifest.json`, `summary.json`, `report.md`. Large traces remain separately hashed artifacts. Work in a temporary attempt directory first; no command overwrites the original evidence or an already sealed attempt.

## 3. Fixed work grid and numerical contracts

Represent IDs as typed fields, not strings later parsed for scientific meaning. Stable ordering is seed `[7,17,29]`, family, mode, replicate, arm `[td0,eligibility]`; references have their own reference-kind field, not a fake anonymous arm.

| Family | Model count | Required evaluation sets |
|---|---:|---|
| Original normal and original frozen shuffle | 12 | Original plus eight new sets per seed |
| Additional block10/global permutations | `3 * 2 * 32 * 2 = 384` | Same nine sets |
| Supervised/immediate/source-visible references | `3 * 3 = 9` | Same nine sets |
| Main total | **405** | **3645 main model/set results** |

D0 twins, reset controls, reversed-order checks and intermediate drain snapshots are auxiliary records with separate IDs. They do not inflate, replace or satisfy these 405/3645 main counts. Track their own expected key sets. Require reset scoring for the 12 original models and nine references on all nine sets (189 auxiliary scores), and canonical-versus-reversed evaluation checks for the 12 original models on all nine sets (108 paired checks). For each original trajectory, require nine-set scoring before drain and after each actual drain call; additionally require after-`end_run` scoring for eligibility only. Arm A has no `end_run` method: do not invent that call or silently count it as an executed check. Derive the exact drain-snapshot key set from the sealed schedule. A score has overall and per-delay counts and margins.

Fixed scientific parameters: hidden size 64, radius 0.9, 2000 training decisions, 20 evaluation blocks, checkpoint interval 100, alpha 0.1, gamma 0.9, lambda 0.8. Cue delays `[1,2,3,4,5]` and hidden feedback delays `[1,3,5]` are different fields. Do not use a single ambiguous `delay` field in joined output.

New permutation generator: `Generator(PCG64(SeedSequence([seed, 0x33434641, 1, mode, replicate])))`, mode `0=block10`, `1=global`, replicate `0..31`. Original shuffle remains `[seed,0x33534846]`, a distinct condition. Additional evaluation generator: `Generator(PCG64(SeedSequence([seed,0x33434641,2,evaluation_id])))`, IDs `0..7`. No new reservoir, action, hidden-delay or training-fixture seeds.

Three comparison surfaces must remain distinct:

1. Same-environment diagnostic replay: exact values, types, order, RNG state and array bytes where paired; no rounding or tolerance.
2. Existing portable verifier: unchanged two parameter-digest exclusions and only seven checkpoint fields with absolute `1e-12`, relative zero.
3. Independent D3 algebra: componentwise `abs(actual-expected) <= 1e-10 + 1e-12*abs(expected)`; report maximum residual. Never reuse this tolerance for D0 or widen it after a failure.

Undefined statistics use an object such as `{"value":null,"reason":"zero_variance","sample_count":20}`. Non-finite required numbers are errors. Distinguish undefined correlation/cosine from missing work, zero effect and failed measurement.

## 4. Execution conventions

Every task follows RED -> smallest implementation -> GREEN -> review -> scoped commit. Run each named test before implementing it, check that the failure names the intended missing behavior, then rerun it and related existing tests. Create only the minimal import scaffold when needed to turn collection failures into meaningful assertion failures. Do not label an import/configuration mistake a scientific RED result.

Examples below define proposed APIs and acceptance tests; they are not existing callable code. Use `pytest`/NumPy assertions with explicit imports. New unit tests use hand-constructed arrays and short valid configurations (for replay, 20 decisions and checkpoint 10); they must never launch the 384-trajectory measurement. Use fresh pytest temporary directories for evidence tests. No fixed test-count promise is made.

All commands run from the repository root in an isolated Python 3.12 environment. Before execution, run `git status --short`, `git branch --show-current` and `git rev-parse HEAD`; verify the branch, clean worktree and spec/plan commits. Do not reset, rebase or discard unrelated changes. At each commit use `git add -- <the exact listed paths>`, then inspect `git diff --cached --stat` and `git diff --cached --check`, and commit using `git commit -m "<the task message>"`. Replace the bracketed command arguments with that task's explicit paths/message, never `git add .`. These are future execution instructions, not commands executed while drafting the plan.

### Task 1 — Fixed identities, immutable input manifest and strict contracts

**Files:** create package `__init__.py`, `contracts.py`, `manifest.py`; tests `test_phase3c_diagnostics_contracts.py`, `test_phase3c_diagnostics_manifest.py`.

- [ ] Read spec Sections 2, 5.3-5.4 and 8. Define `ModelId`, `EvaluationId`, `NullableMetric`, registered configuration and a separate `smoke` profile that can never be reported as registered evidence.
- [ ] Write RED grid tests, including duplicate IDs, bool-as-int, extra keys, altered parameters and accidental original-shuffle/replicate-0 collision. Core test:

```python
from neural_state_machine.phase3c_diagnostics.contracts import registered_model_ids


def test_registered_grid_is_complete_and_unique():
    rows = registered_model_ids()
    assert len(rows) == len(set(rows)) == 405
    assert sum(row.family == "permutation" for row in rows) == 384
    assert sum(row.family == "original" for row in rows) == 12
    assert sum(row.family == "reference" for row in rows) == 9
```

- [ ] Run `python -m pytest -q tests/test_phase3c_diagnostics_contracts.py tests/test_phase3c_diagnostics_manifest.py`; expect failures for the contract gaps, not a benchmark run.
- [ ] Implement deterministic IDs and exact key/type validation. Generate immutable input coverage from the reference Git tree, not from a glob that accidentally includes newly created outputs. Include every old experiment file, reused modules, verifier, original spec, approved diagnostic spec and this plan. Compute SHA-256 before parsing evidence; validate the formal binding and original configuration.
- [ ] Add manifest tests for changed input bytes, missing files, symlink/path traversal, mismatched implementation SHA, and attempted writes into existing evidence. Canonical JSON is UTF-8, sorted keys, compact separators, one terminal newline and `allow_nan=False`. Hash its exact bytes externally; do not put a manifest's own digest inside itself.
- [ ] Rerun both tests and `python -m ruff check src/neural_state_machine/phase3c_diagnostics tests/test_phase3c_diagnostics_contracts.py tests/test_phase3c_diagnostics_manifest.py`. Review staged paths and commit `feat: define immutable Phase 3C diagnostic contracts`.

### Task 2 — Slot/donor provenance and the minimal realignment witness

**Files:** create `provenance.py`; test `test_phase3c_diagnostics_provenance.py`.

- [ ] Write a pure `DonorAssignment(slot, donor, due_step)` contract. Test a swap explicitly: slot 0 borrows donor 1's reward and delivers at decision 1. A value-only match must not count as an index fixed point.

```python
from neural_state_machine.phase3c_diagnostics.provenance import donor_rows


def test_due_time_belongs_to_slot_not_donor():
    rows = donor_rows(permutation=(1, 0), due_steps=(1, 4))
    assert (rows[0].slot, rows[0].donor, rows[0].due_step) == (0, 1, 1)
    assert rows[0].donor == rows[0].due_step
    assert rows[1].due_step == 4
```

- [ ] Run `python -m pytest -q tests/test_phase3c_diagnostics_provenance.py` for RED. Include a deliberately wrong donor-time implementation in a negative test and show that it is detected.
- [ ] Implement index permutations using explicit PCG64 lineages. For each block call `rng.permutation(10)` in exactly the original order and offset indices by the block start. Applying the reconstructed original permutation must reproduce `_permute_reward_blocks` values with an independent, identically seeded RNG. Global mode calls `rng.permutation(N)` once. Identity mode represents normal rewards.
- [ ] Test full permutations, within-block constraints, whole/block reward multisets, independent RNG ownership, shared permutations between arms, legal future donors, collisions, cancellation and duplicate reward values. No rejection/resampling of fixed points or high scores.
- [ ] Implement donor-relative-to-delivery categories, displacement histograms, current-donor matches, and the two conditional expected-match formulas in spec Section 5.1. Test the expectation on a tiny enumerated permutation set, not on newly measured task accuracy.
- [ ] Rerun the test and targeted Ruff. Commit `feat: audit reward slots and permutation donors separately`.

### Task 3 — D0 replay driver with external, non-mutating observation

**Files:** create `replay.py`; test `test_phase3c_diagnostics_replay.py`.

- [ ] Read the exact `_execute_training` loop and public learner snapshots. Write tests parametrized over both concrete arms and normal/original-shuffle conditions, using a short configuration. Compare driver output with a fresh call to unchanged `_execute_training` on the same inputs.
- [ ] Run `python -m pytest -q tests/test_phase3c_diagnostics_replay.py` for RED. Assert full protocol/checkpoint/end-weight equality, not only action equality.
- [ ] Implement the adapter with the original concrete learner classes and frozen helper calls. Mirror orchestration, not numerical learner formulas: hidden -> selection -> enqueue slot -> feedback bucket -> scalar `learn` -> checkpoint. Preserve drain, the pre-`end_run` digest, and eligibility lifecycle. Do not globally monkeypatch the benchmark factory or replace a concrete learner with a proxy.
- [ ] At the call boundary capture actual finite scalar arguments, returned updates, selection-time action values, hidden copies, weight snapshots, E/P snapshots for eligibility and action-RNG state. Pass the learner only its old API arguments. Full observer metadata is assembled outside the learner from independent read-only copies; no callback result affects training.
- [ ] Use two complementary checks: untouched `_execute_training` provides endpoint/protocol/checkpoint parity; independently initialized light-capture/full-observer diagnostic executions provide exact per-step parity wherever both record a field. Do not claim the untouched driver exposes a per-step field that it does not return. Compare array dtype/shape/bytes and RNG-state structures exactly.
- [ ] Independently reconstruct delivered buckets and their sums from slot records, then compare with intercepted calls including zero/cancellation and drain. Verify each slot delivered once, no remaining queue, no fabricated drain action and exact aggregate conservation.
- [ ] Add deliberate corruptions of one scalar, one prediction, one checkpoint and one RNG state; require first mismatch path/step. Rerun new tests plus `tests/test_phase3c_benchmark.py` and `tests/test_phase3c_learners.py`. Commit `feat: add parity-checked Phase 3C diagnostic replay`.

### Task 4 — D0 information boundary, aliasing and scoring isolation

**Files:** extend `replay.py`; test `test_phase3c_diagnostics_isolation.py`.

- [ ] Write RED tests replacing observer source labels/multiplicity, deleting observer scoring labels, and reversing observer record order while preserving the anonymous input stream. Learned parameters and predictions must stay exact.
- [ ] Run `python -m pytest -q tests/test_phase3c_diagnostics_isolation.py`. Add a deliberately broken adapter that passes a metadata-derived scalar and ensure the gate rejects it; a no-op observer is not sufficient adversarial coverage.
- [ ] Ensure `np.shares_memory` is false for observer copies versus writable learner matrices, and across independent runs. Never attach the observer, labels, source IDs, donor IDs or environment records to a learner instance.
- [ ] Guard policy matrices, learner weights, E/P lifecycle and action RNG around scoring/checkpoint collection. Recurrent scoring state can advance/reset on its own policy copy; it must not alter the training instance. Verify the original checkpoint still scores its training block without parameter mutation.
- [ ] Introduce `D0Failure` with first field path, original/observed values or digests, step and attempt ID. On failure retain an invalid audit and stop; no fallback to partial behavioral interpretation or weakening the verifier.
- [ ] Rerun Tasks 3-4 tests with existing `tests/test_phase3c_controls.py`. Commit `test: enforce diagnostic information and mutation boundaries`.

### Task 5 — Independent evaluation manifest and snapshot scoring

**Files:** create `evaluation.py`; test `test_phase3c_diagnostics_evaluation.py`.

- [ ] Write RED tests for 27 evaluation bundles across the three seeds (nine per seed; three original plus 24 additional), distinct typed IDs, deterministic hashes, 200 cases per bundle and 40 per cue delay. Repeated cue/delay categories are legal; identity is bundle/index, not just category.
- [ ] Run `python -m pytest -q tests/test_phase3c_diagnostics_evaluation.py`. Instrument a fake fitter to prove all evaluation bundles/hashes are sealed before any fit call is allowed.
- [ ] Reuse `_build_fixtures(DelayedCueTask(), rng, 20)` and the existing fixture digest, with only the registered evaluation lineages. Keep the original bundle unchanged. Precompute/hash the complete manifest without training a new model.
- [ ] Score each snapshot with fresh policy/learner copies and `_evaluate`; never change the live training object. Verify a copied snapshot's greedy outputs and tie handling against the original class. Evaluation labels are scoring-only, never supplied to fit/training.
- [ ] Evaluate original and reversed fixture order on fresh copies, map results back by fixture index and require exact per-fixture agreement. Store overall/per-delay counts, action/hidden digests, margins and separately named reset controls. Test that mutating evaluation labels cannot affect a model's trained weights.
- [ ] Test end-run/pre-drain/intermediate snapshots without conflating training checkpoint accuracy with held-out accuracy. Rerun evaluation and isolation tests. Commit `feat: seal independent diagnostic evaluation fixtures`.

### Task 6 — D1 statistics and complete permutation execution rows

**Files:** extend `provenance.py`; test `test_phase3c_diagnostics_statistics.py`; extend provenance/replay tests as needed.

- [ ] Write RED tests for valid-pair lag alignment `0..10`, the final 100 real decisions, current action/label contingency cells, zero-variance correlations, no-arrival versus cancellation and separate drain statistics. Use small hand-computed sequences.
- [ ] Run `python -m pytest -q tests/test_phase3c_diagnostics_statistics.py tests/test_phase3c_diagnostics_provenance.py`.
- [ ] Implement counts, sums, means, `mean(F_t*r_(t-lag))`, centered Pearson correlations, and label/action products with `s(a)=2*a-1`. Never treat drain ticks as later real decisions; log all denominators and typed undefined reasons.
- [ ] Define nearest-rank p10/p90 as sorted index `ceil(q*32)-1`; test exact outputs on integers `0..31`. For every seed/arm/mode report mean/median/min/max, counts `>=150` and `>=197` on original evaluation, and every replicate. Original-shuffle rank is descriptive, not a calibrated p-value; use a rank interval when tied.
- [ ] Build an iterator for the 384 fixed permutation IDs, not a score-driven search. Both arms share the same donor permutation and all action/schedule/fixture lineages. Test the iterator with fake training so CI verifies completeness without running the full experiment.
- [ ] Require every additional model's nine main evaluations; keep original and additional rows distinct. A block/global comparison changes block structure as well as alignment and must be labeled accordingly in output metadata.
- [ ] Rerun statistics/provenance/replay tests and targeted Ruff. Commit `feat: define complete permutation diagnostics and statistics`.

### Task 7 — D2 separately typed privileged references

**Files:** create `references.py`; test `test_phase3c_diagnostics_references.py`.

- [ ] Write RED tests for ridge bias penalty and train-only labels; exact immediate reference reuse; delayed source association with stale decision-time predictions, collisions and terminal drain. Use two-action tiny matrices with hand-computed updates.
- [ ] Run `python -m pytest -q tests/test_phase3c_diagnostics_references.py`.
- [ ] Implement supervised ridge with `Phi=[h,1]`, two targets `+1/-1`, penalty `1e-6` including bias. Use an augmented least-squares solve, e.g. `np.linalg.lstsq(vstack([Phi,sqrt(1e-6)*I]), vstack([Y,zeros]), rcond=None)`, and record solver/rank. No feature scaling or hyperparameter search. Test the normal-equation residual and compare to a tiny analytic solution; do not invert a matrix explicitly.
- [ ] For immediate identified feedback create a fresh `NormalizedActionValue` and frozen policy, and reuse `_train_normal` with the original fixtures/task/action RNG/config; retain that trained learner for additional evaluations. Compare its outputs against `run_action_value_experiment` and the frozen per-seed original counts/lineage; that result object alone is not a fitted model that can score new fixtures. Existing reported counts are 181, 199 and 177, but read the frozen artifact as the authority. Do not require this reference to pass every original quality threshold.
- [ ] For source-visible delayed feedback create a distinct reference class with zero `(2,65)` weights. Store `a_i, phi_i, denominator_i, q_i` at selection. At delivery, in ascending source-index order, apply `0.1*(r_i-q_i)*phi_i/denominator_i` to row `a_i`; never refresh `q_i` at delivery. Reuse frozen normal action/delay schedules and drain every pending record.
- [ ] Verify feature streams across references/originals; evaluate all nine references on the same nine per-seed sets with separate reset controls. Assert `information_access` labels identify supervision or source/decomposition access. No donor-aware future-information shuffled oracle and no mixing these rows into anonymous Gate B.
- [ ] Rerun reference/evaluation tests plus relevant existing action-value tests. Commit `feat: add isolated diagnostic readout references`.

### Task 8 — D3 prediction, historical contribution and drain accounting

**Files:** create `accounting.py`; test `test_phase3c_diagnostics_accounting.py`.

- [ ] Write RED tests for residual cross terms, two-source collisions, source/other trace splitting, zero-norm cosines, and eligibility drain with zero-valued feedback. A representative identity test uses tiny finite arrays:

```python
import numpy as np
from neural_state_machine.phase3c_diagnostics.accounting import expected_drain_delta


def test_drain_subtracts_the_frozen_prediction_on_every_call():
    eligibility = np.array([[1.0, 0.0], [0.0, 2.0]])
    actual = expected_drain_delta(
        rewards=(1.0, 0.0), prediction=0.25,
        eligibility=eligibility, alpha=0.1,
    )
    np.testing.assert_allclose(actual, 0.05 * eligibility, rtol=0.0, atol=1e-15)
```

- [ ] Run `python -m pytest -q tests/test_phase3c_diagnostics_accounting.py` for RED. Also reject a fake Arm-A history trace and recomputed rather than stored historical predictions.
- [ ] For normal Arm B reconstruct `E_t=sum rho^(t-i) g_i`, `P_t=sum rho^(t-i) q_i`, `K_t=sum_{due_i=t}q_i`. Check `F-P=(F-K)+(K-P)` and retain `2*(F-K)*(K-P)` when comparing squared errors. For Arm A use only `g_t,q_t`, not a fictitious exponential state.
- [ ] Split E into delivered-source/other terms, reconstruct both update components, and calculate `J_t=sum_{due_i=t}(r_i-q_i)g_i` on the original trajectory. Report norms, inner products, cosines and typed undefined counts. Do not equate large other-history norm with harm or call K an available anonymous expectation.
- [ ] For shuffled paths keep donor and slot predictions separate, with future donors explicitly offline. Primary target-mismatch analysis remains on normal paths; no donor-derived value feeds training.
- [ ] Record pre-drain, each drain call and post-`end_run` snapshots. Arm A weights do not change. Arm B E/P stay fixed during drain, and `DeltaW=alpha*(sum(F)-n*P_before)*E_before`. After `end_run`, traces clear without a weight update. Score fresh snapshot copies; never subtract last-training-block checkpoint accuracy from held-out accuracy.
- [ ] Implement the independent algebra tolerance from Section 3, including non-finite/boundary tests and maximum residual reports. Full per-step accounting is only for the 12 original paths; permutation rows use lighter provenance captures.
- [ ] Rerun accounting/replay/reference tests and targeted Ruff. Commit `feat: account for anonymous prediction and drain updates`.

### Task 9 — Fail-closed orchestration, sealed manifests and reports

**Files:** create `runner.py`, `report.py`, `scripts/diagnose_phase3c_failure.py`; extend `manifest.py`; tests `test_phase3c_diagnostics_runner.py`, `test_phase3c_diagnostics_report.py`.

- [ ] Write RED tests with fake runners: no fitting in `prepare`, D0 failure prevents D1/D2/D3, missing/duplicate/unexpected rows invalidate completion, mismatched manifests reject resume, and a valid unresolved diagnosis leaves original `behavior_passed=false`.
- [ ] Run `python -m pytest -q tests/test_phase3c_diagnostics_runner.py tests/test_phase3c_diagnostics_report.py tests/test_phase3c_diagnostics_manifest.py`.
- [ ] Implement the thin CLI contract:

```text
prepare --profile registered-v1 --output DIR
run --manifest PATH --expected-manifest-sha256 HEX --attempt-id ID --approve-measurement
verify --manifest PATH --attempt-dir DIR
report --manifest PATH --attempt-dir DIR --output DIR
```

`prepare` only validates inputs/environment and constructs IDs, permutations/evaluation manifests; it never fits or scores a new model. Registered `run` requires an exact manifest digest and explicit measurement flag in addition to human workflow authorization. A smoke profile never emits registered-valid evidence. `verify` performs offline schema/hash/grid/accounting checks, not retraining.

- [ ] Seal manifest bytes before fitting: spec/plan/implementation SHAs, original input hashes, fixed params, exact environment, RNG schemes, all work IDs and evaluation hashes. Store the manifest digest in the separate execution manifest; do not rewrite the sealed input to add results or a self-hash.
- [ ] Sequence D0 -> D1 provenance/full grid -> D2/D3 -> complete validation. Persist typed per-row progress and raw outputs after each model, in deterministic ID order. Stream trace files rather than accumulating hundreds of matrices; do not attach traces to learners or change algorithmic state to reduce memory.
- [ ] Use a new immutable directory per attempt. On error/timeout preserve partial rows with `complete=false`, `diagnostic_valid=false`, nonzero exit and first failure. A technical retry uses the same sealed manifest and records predecessor/attempt provenance. Reusing completed rows requires matching config/input/output digests; conflicting deterministic outputs invalidate the retry, never select the favorable one.
- [ ] Report all 405 model identities and 3645 main scores, independently tracked auxiliary checks, original status, per-mode distributions, nine privileged references, accounting residuals and all undefined reasons. Require every registered auxiliary obligation, not just the main count.
- [ ] Render evidence-linked candidate conclusions with `supported`, `not_supported` or `unresolved`. Separate structural possibility, occurrence and demonstrated performance causation. Include block/global confounds and post-result selection; no causal verdict from correlation/rank alone. An unresolved valid report is allowed; at most one separately reviewed follow-up, not an automatic learner change.
- [ ] Test round-trip strict JSON, output path containment, old artifact overwrite rejection, wrong manifest/implementation identity and frozen input mutation at finalization. Rerun all diagnostic tests and Ruff. Commit `feat: orchestrate fail-closed Phase 3C diagnostic evidence`.

### Task 10 — Reproducible Python 3.12 environment and bounded CI

**Files:** create the two requirements files, manual workflow and `tests/test_phase3c_diagnostics_ci.py`; modify only `.github/workflows/ci.yml` among existing files.

- [ ] Write RED structural tests that normal push/PR jobs cannot request registered `run`, no Python matrix besides 3.12 exists, all original gates remain, manual measurement verifies exact checkout/manifest SHA, and artifact upload occurs even on failure. Run `python -m pytest -q tests/test_phase3c_diagnostics_ci.py`.
- [ ] Anchor the new diagnostic environment to the recorded passing #414 runtime: CPython `3.12.11`, NumPy `2.2.6`, pytest `8.4.1`, Ruff `0.12.10`. Put the three package pins in `requirements/phase3c-diagnostics.in`. These are deliberate replay anchors, not a claim about latest releases. Do not alter general package Python support.
- [ ] Resolve once in a clean 3.12.11 virtual environment without running behavioral measurements. Install build tooling, then project/dev dependencies constrained by the input pins with `--no-build-isolation`. Capture every installed package including pip/setuptools/wheel into the exact-version lock and installation report; exclude the editable project entry. Review the resolved lock rather than inventing unavailable historical build-tool versions.
- [ ] Recreate a second clean environment from that lock, then install the checkout with `--no-deps --no-build-isolation`. Record lock SHA-256, installation provenance, Python patch/NumPy versions, OS/architecture, NumPy BLAS configuration and thread-related environment. Refuse measurement if these differ from the sealed manifest. No environment/dependency sweep or post-result version selection.
- [ ] Keep ordinary CI on one Python 3.12 job; add only lightweight new tests/lint and retain every existing benchmark/protocol/frozen-verifier step. Do not add the new 384-trajectory grid or fit the nine new references on each push. `pytest` collection/imports never start a registered run.
- [ ] Add a `workflow_dispatch`-only measurement workflow with explicit experiment-branch and implementation/manifest identity checks, read-only repository permissions and a single bounded Python 3.12 job. Use the reviewed lock, checkout the requested exact implementation, no automatic commits/merges, and upload manifests/partial evidence on success or failure. Validate actual branch-qualified dispatch availability through GitHub before claiming it can run; if unavailable, stop for an approved runner route rather than modifying or merging the default branch.
- [ ] Set an explicit 120-minute job limit and include it in execution provenance. This is a resource ceiling, not a runtime promise. Timeout produces incomplete evidence and never authorizes fewer seeds/replicates/evaluations. Any later runtime-only revision requires a separately recorded approval and unchanged scientific grid.
- [ ] Run new CI tests, the complete diagnostic unit suite, full `python -m pytest -q`, `python -m ruff check .`, and all existing workflow commands. Obtain exact-head Python 3.12 CI after pushing the implementation. Record actual results, not this plan's expected outcomes. Commit `ci: verify diagnostic tooling on Python 3.12 only`.

### Task 11 — Implementation audit and sealed-manifest measurement gate

**Files:** no new scientific implementation; later create the prospective `manifest.json` only after review. This task contains a hard stop before measurement.

- [ ] Compare the implementation diff to this plan's file map. Require byte equality of every original scientific/evidence file and both specs. Confirm no new learner behavior in original modules, no evidence regeneration and no widening of original verifier tolerances.
- [ ] Review Tasks 1-10 tests for genuine failing counterexamples, especially donor versus slot, actual scalar interception, exact replay beyond action hashes, metadata leakage, stale reference predictions, E/P drain and missing-grid acceptance. Revisit any unclear diagnostic contract before proceeding.
- [ ] Verify the exact implementation commit has completed all original CI gates plus the diagnostic small-test suite on Python 3.12. A docs-only child is not silently substituted for the tested runtime commit. Report no new diagnostic performance claims at this gate.
- [ ] Run `prepare` in the reviewed locked environment to generate the complete prospective manifest/evaluation hashes without training new conditions. Review the final immutable manifest bytes, work counts, implementation identity, artifact retention and runner ceiling. Record their digest and human measurement approval separately; the command-line flag is not itself human approval.
- [ ] **STOP.** Present the exact code/CI reference, manifest digest, complete fixed grid and runtime route for measurement authorization. Do not execute Task 12 just because implementation CI is green.

### Task 12 — Explicitly approved measurement, evidence freeze and one decision

**Files:** later add only the four retained evidence files under `docs/experiments/phase-3c-failure-attribution-v1/`; artifact traces referenced by identity and digest. No learner, old evidence, spec or threshold changes.

- [ ] Confirm measurement approval covers the exact implementation SHA and sealed manifest digest. Launch one authorized attempt using the Task 9 `run` command in the Task 10 environment; retain run/attempt/checkout identities.
- [ ] Execute all 12 original D0 trajectories and fresh uninstrumented/twin checks before any additional condition. Invoke the unchanged portable verifier through its existing script with the same Python executable and retain its output/exit status. On the first integrity failure, retain the invalid audit and stop; do not repair/refreeze old evidence within this run.
- [ ] After D0 passes, execute all 384 permutation trajectories with shared schedules, all nine evaluation sets per model, the nine D2 references, and D3 accounting on the original 12 paths. Complete all declared reset/reversed-order/drain obligations. No adaptation to interim scores or selective early success stop.
- [ ] Run offline `verify` against the sealed manifest; require exact main and auxiliary key sets, valid input/output hashes, no original-file changes, all finite-or-explicitly-undefined fields, and bounded accounting residuals. Run `report` only after validity is established; otherwise emit an incomplete/invalid report clearly labeled as such.
- [ ] Freeze `execution-manifest.json` with complete/failed states, timings, output hashes, all attempt provenance and workflow artifact IDs/relative filenames/digests. Keep the input manifest unchanged. Record the execution manifest's final digest outside itself; do not construct circular digest dependencies.
- [ ] Retain large traces as immutable artifacts plus sufficient lineage and runner metadata for deterministic reconstruction. Archive/index evidence before ephemeral artifacts expire; the committed summary must not rely solely on a download URL.
- [ ] Commit only authorized evidence/report files after review, without changing original Phase 3C status. Verify the frozen evidence diff and read back the committed summary/manifest. Stop with supported/unresolved attribution and at most one recommended separate follow-up; no automatic tuning, new mechanism, proof claim or merge.

## 5. Review coverage and implementation exit checklist

| Spec requirement | Tasks |
|---|---|
| Immutable inputs, unchanged scientific claims, exact type/hash checks | 1, 9, 11, 12 |
| D0 original replay, actual calls, noninterference, no aliasing | 3, 4, 11, 12 |
| Slot/donor distinction, realignment witness and expected counts | 2, 6 |
| Fixed 384-trajectory grid and eight additional evaluation sets | 1, 5, 6, 9, 12 |
| No seed selection, pseudo-p-values or anonymous credit for privileged references | 6, 7, 9 |
| Ridge/immediate/source-visible reference definitions | 7 |
| F/P/K residuals, cross term, source/other contributions | 8 |
| Drain lifecycle, pre/post scoring and independent tolerance | 5, 8 |
| Sealed manifests, retry provenance, incomplete outcomes and artifact retention | 1, 9, 10, 12 |
| Python 3.12-only CI, no automatic full measurement | 10, 11 |
| Review gates and no premature learner changes | 11, 12 |

- [ ] All added functions have bounded responsibility, explicit type/shape validation and tests for invalid inputs.
- [ ] All RED/GREEN claims are backed by executed commands and observed failures/results, not plan prose.
- [ ] The tested runtime, prepared manifest and actual measurement checkout are explicitly distinguished and matched.
- [ ] A diagnostic can finish valid while the cause remains unresolved and original behavior remains false.
- [ ] This plan has been reviewed and execution has been explicitly approved before any implementation task starts.

**Current stopping point:** one committed implementation-plan document. No task above has been executed by the act of writing it. Approval starts Task 1; approval does not bypass Task 11's separate measurement gate.
