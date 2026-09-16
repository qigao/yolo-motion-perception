# Phase C4-A Failure-Attribution Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an additive, sealed diagnostic system that explains why the registered C4-A shuffled negative control remains predictive for seeds 17 and 29, without changing the frozen C4-A operator, evidence, seeds, thresholds, or ridge penalty.

**Architecture:** Keep the frozen C4-A code and evidence immutable. Add a separate `phase_c4a_diagnostics` package that reproduces registered C4-A exactly (D0), computes matrix geometry (D1), target/projection and supervised-reference evidence (D2), readout decomposition (D3), runs the fixed block10/global permutation grid (D4), and produces a mechanism-level attribution table (D5). Permanent CI, `Phase C4-A Attribution Preflight`, `Phase C4-A Attribution Measurement`, and `Phase C4-A Attribution Freeze` remain distinct; there is a mandatory stop before measurement and a second no-refit interpretation review before freeze.

**Tech Stack:** Python 3.12.14, NumPy 2.5.3, pytest 9.1.1, Ruff 0.15.22, GitHub Actions, canonical JSON, SHA-256.

**Spec:** `docs/superpowers/specs/2026-09-16-phase-c4a-failure-attribution-diagnostics-design.md`

## Global Constraints

- Execute from an isolated branch/worktree created from exact spec head `26cb9ccaeab03d7dd368337ba17af7318a3fca78`; suggested branch `diagnostics/phase-c4a-failure-attribution-v1`.
- Frozen C4 science head: `8ae3154950ed53c4d0a0f555463042ff72674d31`.
- Frozen C4-A evidence commit: `832c4a6874b87bba1b189e2d3604f66025730bb9`.
- Frozen C4-A manifest/result/provenance SHA-256 values are respectively `a990745cf6f0e8c5fd7ae5d9a7189737b30a40168a4db3df1a6d3043d6d5876a`, `7f398f474dba928d2a6aa993d0801563aa05af20faa8cd9a61ec94a6ca4697a4`, and `8fb47b5e670754bc44c9ea70dd4cedce40e6f52165923142114577b4aec00351`.
- Formal head remains `8b2180ed24b6ff03db4b927ec29cdd9903b0ccac`.
- Seeds `(7, 17, 29)`, hidden size `64`, recurrent radius `0.9`, `2000` real training decisions, hidden-delay support `(1,3,5)`, five drain clocks, `2005` design rows, ridge `1e-6`, and all registered gates remain unchanged.
- The eight secondary evaluation sets are diagnostic only; no evaluation label enters fitting or selection.
- Do not modify frozen C3/C4 scientific modules or frozen C4-A evidence files.
- C4-B must not be created, invoked, or represented in diagnostic result schemas.
- D0 is a hard prerequisite. D0 failure stops D1-D5 interpretation and prevents any diagnostic result from being called valid.
- Tasks 1-10 do not authorize registered diagnostic measurement. Task 10 ends at a mandatory human gate.

## File Map

Create only additive diagnostic files:

```text
src/neural_state_machine/phase_c4a_diagnostics/
  __init__.py
  model.py
  integrity.py
  geometry.py
  attribution.py
  decomposition.py
  permutations.py
  report.py
  evidence.py
scripts/
  diagnose_phase_c4a_failure.py
  verify_phase_c4a_failure_diagnostics.py
requirements/
  phase-c4a-diagnostics.in
  phase-c4a-diagnostics-python312.lock
tests/
  test_phase_c4a_diag_model.py
  test_phase_c4a_diag_integrity.py
  test_phase_c4a_diag_geometry.py
  test_phase_c4a_diag_attribution.py
  test_phase_c4a_diag_decomposition.py
  test_phase_c4a_diag_permutations.py
  test_phase_c4a_diag_report.py
  test_phase_c4a_diag_evidence.py
  test_phase_c4a_diag_cli.py
```

Permanent CI modifies `.github/workflows/ci.yml`; dedicated preflight is `.github/workflows/phase-c4a-attribution-preflight.yml`. Measurement/freeze workflow files are one-shot and are deleted after use.

---

### Task 1: Frozen Identity, Config, and Immutable Data Model

**Files:** create `phase_c4a_diagnostics/__init__.py`, `model.py`; test `test_phase_c4a_diag_model.py`.

**Interfaces:** `FrozenC4AIdentity.registered()`, `DiagnosticConfig.registered()`, `DiagnosticConfig.testing(replicates: int)`, `sha256_file(path: Path) -> str`, `freeze_float64(array) -> np.ndarray`.

- [ ] Write RED tests that assert exact frozen heads/hashes/seeds/`2005` rows/`1e-6`, and verify `freeze_float64()` detaches storage and returns C-contiguous read-only float64.
- [ ] Run `pytest -q tests/test_phase_c4a_diag_model.py`; require missing-module/API failure.
- [ ] Implement frozen/slots dataclasses. `DiagnosticConfig.registered()` returns exactly `permutation_replicates=32`, `permutation_lineage=0x43344144`, modes `("block10","global")`, `near_zero_margin=1e-9`. `DiagnosticConfig.testing()` accepts only positive small replicate counts and is never callable from the registered CLI.
- [ ] Run focused pytest + Ruff.
- [ ] Commit `feat: add C4-A diagnostic identity model`.

Key test:

```python
identity = FrozenC4AIdentity.registered()
assert identity.scientific_head == "8ae3154950ed53c4d0a0f555463042ff72674d31"
assert identity.seeds == (7, 17, 29)
assert identity.design_rows == 2005
assert identity.ridge_penalty == 1e-6
assert DiagnosticConfig.registered().permutation_replicates == 32
```

---

### Task 2: D0 Frozen Evidence Verification and Registered Replay

**Files:** create `integrity.py`; test `test_phase_c4a_diag_integrity.py`.

**Interfaces:**

```python
@dataclass(frozen=True, slots=True)
class RegisteredConditionReplay:
    seed: int
    condition: Literal["normal", "registered_shuffled"]
    design: np.ndarray
    target: np.ndarray
    rows: tuple[DecisionCreditRow, ...]
    actions: tuple[int, ...]
    rewards: tuple[float, ...]
    shuffled_rewards: tuple[float, ...] | None
    schedule_digest: str
    design_digest: str
    target_digest: str
    score: AccuracyCount

verify_frozen_c4a_evidence(root: Path) -> FrozenC4AIdentity
replay_registered_condition(seed: int, condition: str) -> RegisteredConditionReplay
run_d0_integrity(root: Path) -> tuple[RegisteredConditionReplay, ...]
```

Reuse read-only behavior from frozen `phase_c4_measurement` private helpers and `build_batch_design`/`fit_anonymous_batch_probe`; do not edit those modules.

- [ ] RED tests mutate one byte of copied manifest/result/provenance and require failure before replay; registered replay must reproduce `(normal, shuffled)=(200,147),(200,170),(200,160)` for seeds `7,17,29`.
- [ ] Require every design/target to have `2005` rows, five drain calls, matching schedule lineage, and normal/shuffled design byte identity per seed.
- [ ] Implement exact registered shuffle lineage `[seed, 0x33534846]`, unchanged hidden-delay aggregation, unchanged `1e-6` fit, and immutable returned arrays.
- [ ] Run `pytest -q tests/test_phase_c4a_diag_integrity.py tests/test_phase_c4_*.py` plus Ruff; compare diff to ensure no frozen file changed.
- [ ] Commit `feat: add C4-A diagnostic replay gate`.

Required assertion:

```python
for seed in (7, 17, 29):
    normal, shuffled = rows_for(seed)
    assert normal.design_digest == shuffled.design_digest
    assert normal.design.tobytes() == shuffled.design.tobytes()
```

---

### Task 3: D1 Design-Matrix Geometry

**Files:** create `geometry.py`; test `test_phase_c4a_diag_geometry.py`.

**Interfaces:** `GeometryMetrics`, `analyze_design_geometry(design: np.ndarray, *, penalty: float = 1e-6) -> GeometryMetrics`.

Metrics: shape, tolerance, rank, nullity, singular values, `sigma_max`, smallest retained singular value, condition number, Frobenius norm, stable rank, ridge effective DoF, leverage min/median/max/p10/p90, per-action-block column norms, cross-block Gram Frobenius norm, and the two bias-column norms.

- [ ] RED tests cover full-rank and rank-deficient matrices and exact tolerance `max(m,n) * eps * sigma_max`.
- [ ] Implement `np.linalg.svd(..., full_matrices=False)`; no explicit inverse.
- [ ] Ridge leverage uses `shrink=s*s/(s*s+1e-6)` and `sum((u*u)*shrink, axis=1)`; stable rank is `||Z||_F^2 / ||Z||_2^2`.
- [ ] Run focused pytest + D0 regression + Ruff.
- [ ] Commit `feat: add C4-A design geometry diagnostics`.

---

### Task 4: D2 Target Associations, Projection, and Supervised Reference

**Files:** create `attribution.py`; test `test_phase_c4a_diag_attribution.py`.

**Interfaces:**

```python
@dataclass(frozen=True, slots=True)
class CorrelationStat:
    count: int
    correlation: float | None
    reason: str | None

@dataclass(frozen=True, slots=True)
class TargetProjection:
    parallel: np.ndarray
    residual: np.ndarray
    parallel_ratio: float
    residual_ratio: float
    fit_residual_norm: float

@dataclass(frozen=True, slots=True)
class SupervisedReferenceFit:
    weights: np.ndarray           # shape (2, feature_size)
    row_count: int
    feature_size: int
    augmented_rank: int
    residual_norm: float
    singular_values: np.ndarray
    penalty: float

associate_target(...) -> dict[str, object]
project_target(design: np.ndarray, target: np.ndarray) -> TargetProjection
fit_supervised_reference(features: np.ndarray, correct_actions: np.ndarray) -> SupervisedReferenceFit
compare_weight_alignment(c4_weights: np.ndarray, supervised_weights: np.ndarray) -> WeightAlignment
```

- [ ] RED tests require zero-variance correlations to be `None` with a reason, and projection recomposition `parallel + residual == target` within `atol=1e-12`.
- [ ] Implement observer associations for current action/label/correctness/cue delay/realized hidden delay/block position, previous+future lags `1..10`, reward provenance, and separate drain/no-arrival/collision categories. Drain clocks never enter decision-label correlations.
- [ ] Implement supervised target `Y_i[a]=+1` for correct action, `-1` otherwise and solve the exact Frobenius ridge using augmented least squares on `Phi` for both outputs. Do **not** reuse `BatchProbeFit`; keep `SupervisedReferenceFit` semantically separate.
- [ ] Evaluate alignment only after fitting: global/per-action cosine, norm ratio, projection onto supervised direction, original + eight secondary margin correlations.
- [ ] Run focused pytest + D0/D1 regression + Ruff; commit `feat: add C4-A target attribution diagnostics`.

Stable supervised solve:

```python
aug_x = np.vstack((phi, np.sqrt(1e-6) * np.eye(phi.shape[1])))
aug_y = np.vstack((y, np.zeros((phi.shape[1], 2), dtype=np.float64)))
coef, residuals, rank, singular = np.linalg.lstsq(aug_x, aug_y, rcond=None)
weights = coef.T
```

---

### Task 5: D3 Weight and Margin Decomposition

**Files:** create `decomposition.py`; test `test_phase_c4a_diag_decomposition.py`.

**Interfaces:** `MarginComponents`, `WeightComparison`, `decompose_readout_margin(weights, hidden_with_bias, correct_action)`, `compare_weights(normal, shuffled)`, `summarize_margin_components(...)`.

- [ ] RED tests require hidden contribution + bias contribution to recompose the full correct-vs-other margin with `rtol=0, atol=1e-12`, and recomputed decisions to equal ordinary evaluation decisions for every tiny-fixture row.
- [ ] Split each action row into hidden weights and final bias; expose signed action-block-0/action-block-1 contributions. Bias-only/hidden-only values are post-hoc diagnostics only and are never retrained models.
- [ ] `compare_weights` reports `||Wn||`, `||Ws||`, `||Ws-Wn||`, global/per-action cosines, block norms, bias terms. Near-zero means `abs(margin) <= 1e-9`.
- [ ] Run focused pytest + Ruff.
- [ ] Commit `feat: add C4-A margin decomposition diagnostics`.

---

### Task 6: D4 Fixed Block10/Global Permutation Grid

**Files:** create `permutations.py`; test `test_phase_c4a_diag_permutations.py`.

**Interfaces:**

```python
PermutationMode = Literal["block10", "global"]
permutation_indices(seed: int, mode: PermutationMode, replicate: int, n: int) -> np.ndarray
permute_rewards(rewards: tuple[float, ...], indices: np.ndarray) -> tuple[float, ...]
run_permutation_grid_for_seed(replay, phase_config, diagnostic_config) -> tuple[PermutationRow, ...]
summarize_permutation_rows(rows: tuple[PermutationRow, ...]) -> PermutationSummary
```

`PermutationRow` stores seed/mode/replicate/permutation digest, original score, eight secondary scores, target-projection metrics and supervised-alignment metrics.

- [ ] RED tests assert exact RNG `PCG64(SeedSequence([seed,0x43344144,mode_id,replicate]))`, block-local permutation membership for `block10`, full-range membership for `global`, and no rejection/resampling of fixed points.
- [ ] Registered measurement always constructs `DiagnosticConfig.registered()` and therefore exactly 32 replicates/mode. Unit tests use `DiagnosticConfig.testing(2)` only through direct Python APIs.
- [ ] Reuse D0 rows/design/actions/schedule; permute reward values only, rebuild scalar feedback through all five drain clocks, refit unchanged C4-A ridge, evaluate original + eight secondary sets.
- [ ] Summary uses nearest-rank p10/p90 index `ceil(q*32)-1`, plus mean/median/min/max and counts `<150`, `>=150`, `>=170`; the registered shuffled score is a separately labeled post-observation reference, never a p-value.
- [ ] Run focused tests + Ruff; commit `feat: add C4-A fixed permutation diagnostics`.

---

### Task 7: D5 Mechanism Table and Interpretation-Safe Report Model

**Files:** create `report.py`; test `test_phase_c4a_diag_report.py`.

**Interfaces:**

```python
class DiagnosticOutcome(Enum):
    BLOCK_LOCAL_ASSOCIATION = "A"
    GLOBAL_PREDICTIVITY = "B"
    REGISTERED_OUTLIER_UNRESOLVED = "C"
    EVALUATION_FIXTURE_SENSITIVE = "D"
    MULTIPLE_MECHANISMS = "E"
    INTEGRITY_INVALID = "invalid"

build_attribution_table(...) -> tuple[MechanismEvidence, ...]
validate_outcome_rationale(outcome: DiagnosticOutcome, table, rationale: tuple[str, ...]) -> None
render_markdown_report(result, *, outcome: DiagnosticOutcome, rationale: tuple[str, ...]) -> str
```

Important: **do not implement an automatic A-E classifier.** The spec uses qualitative words such as “frequently” and “mostly” without new numeric thresholds; the plan must not invent a post-hoc machine gate. Measurement stores raw/summarized evidence. Outcome selection and rationale happen only in the separate no-refit freeze/review stage.

- [ ] RED tests ensure every allowed outcome renders separate `Observed association`, `Matched counterfactual evidence`, and `Interpretation` sections and never uses causal wording stronger than the spec.
- [ ] Required mechanism rows are geometry/effective rank, task-relevant target projection, bias/action shortcut, local block structure, evaluation-fixture sensitivity, unresolved/multiple mechanisms.
- [ ] `INTEGRITY_INVALID` is the only forced label: if D0 is false, report generation must reject any A-E label.
- [ ] Run pytest + Ruff.
- [ ] Commit `feat: add C4-A attribution report model`.

---

### Task 8: Canonical Evidence, Result Schema, and Strict Verifier

**Files:** create `evidence.py`; test `test_phase_c4a_diag_evidence.py`.

**Interfaces:**

```python
prepare_attribution_manifest(repository_root: Path, output_dir: Path) -> Path
validate_attribution_manifest(repository_root: Path, manifest_path: Path) -> dict[str, object]
write_attribution_bundle(repository_root, manifest_path, output_dir, result_payload, trace_index)
verify_attribution_stage(root: Path, allow_missing_result: bool) -> None
```

Canonical filenames are `manifest.json`, `result.json`, `provenance.json`, `trace-index.json`; `report.md` is created only at freeze after interpretation review.

- [ ] RED tests enforce sorted compact UTF-8 newline JSON, `allow_nan=False`, no overwrite, no symlinks, exact frozen C4-A hashes, exact science/formal binding, environment, lineage `0x43344144`, 32 replicates/mode, and no C4-B fields.
- [ ] Manifest keys are exactly `schema_version, stage, scientific_head, scientific_hashes, frozen_c4a, formal_head, environment, registered_seeds, registered_scores, diagnostic_config, permutation_lineage, secondary_evaluation_manifest, expected_result_keys`; `stage="c4a-failure-attribution"`.
- [ ] Measurement result keys are raw/scientific evidence only: `schema_version, stage, integrity_valid, geometry, registered_attribution, permutation_rows, permutation_summaries, mechanism_table`. **No `outcome` field** and no C4-A/C4-B pass booleans are permitted; interpretation belongs to `report.md` at freeze.
- [ ] Trace index records artifact filename, byte size, SHA-256 and semantic role; verifier recomputes all hashes and rejects missing/extra entries.
- [ ] Run all diagnostic tests + Ruff; commit `feat: seal C4-A attribution evidence`.

---

### Task 9: Diagnostic CLI, Exact Lock, and Tiny Integration

**Files:** create `scripts/diagnose_phase_c4a_failure.py`, `scripts/verify_phase_c4a_failure_diagnostics.py`, `requirements/phase-c4a-diagnostics.in`, `requirements/phase-c4a-diagnostics-python312.lock`; test `test_phase_c4a_diag_cli.py`.

**CLI:**

```text
protocol                         # full registered D0 reproduction only
prepare --output DIR             # prospective manifest only
measure --manifest FILE --output DIR   # explicit D0-D5 diagnostic measurement
```

Verifier accepts `--root DIR` and `--no-result-ok`.

- [ ] RED tests: `protocol` must reproduce registered D0 but create no result; `prepare` creates only `manifest.json`; `measure` rejects head/hash/environment mismatch before any permutation fit. Tiny unregistered integration uses direct APIs with `DiagnosticConfig.testing(2)`, never a CLI flag that weakens registered constants.
- [ ] `phase-c4a-diagnostics.in` contains exactly `numpy==2.5.3`, `pytest==9.1.1`, `ruff==0.15.22`.
- [ ] Lock pins the existing proven Python 3.12.14 set: iniconfig 2.3.0, numpy 2.5.3, packaging 26.3, pip 26.2.1, pluggy 1.6.0, Pygments 2.21.0, pytest 9.1.1, ruff 0.15.22, setuptools 84.0.0, wheel 0.48.0. Leave provenance header generic until Task 10.
- [ ] Run all diagnostic tests + existing C4 tests + Ruff.
- [ ] Commit `feat: add C4-A attribution diagnostic CLI`.

---

### Task 10: Permanent CI + Attribution Preflight + Science-Head Freeze — MANDATORY STOP

**Files:** modify `.github/workflows/ci.yml`; create `.github/workflows/phase-c4a-attribution-preflight.yml`; later update only lock provenance header.

**CI identities:** permanent `ci`; dedicated workflow name exactly `Phase C4-A Attribution Preflight`; artifact `phase-c4a-attribution-prospective-${{ github.sha }}`.

- [ ] Permanent CI adds `pytest -q tests/test_phase_c4a_diag_*.py` and Ruff for the new package/scripts; it never invokes `measure` or the 32x2x3 grid.
- [ ] Preflight recreates exact Python 3.12.14, diffs `pip freeze --all` against the diagnostic lock, runs all tests/Ruff, executes full registered `protocol` D0 reproduction, executes `prepare`, then strict verifier `--no-result-ok`. Upload prospective manifest only.
- [ ] First exact-head permanent CI + preflight must both succeed; audit that no `result.json`, `provenance.json` or permutation artifact exists and manifest records exactly 3 seeds, 32/mode, lineage `0x43344144`, frozen C4-A hashes.
- [ ] Update only the lock header with first passing preflight run number/id. Require a second exact-head permanent CI + preflight; this commit becomes `C4A_ATTRIBUTION_SCIENCE_HEAD`. Record prospective manifest SHA-256 and canonical environment SHA-256.
- [ ] **STOP.** Report head/run/job/hashes and explicit confirmation that D1-D5 registered diagnostic measurement has not run. Do not create `Phase C4-A Attribution Measurement` until the user explicitly approves.

---

### Task 11: One-Shot Registered Attribution Measurement — ONLY AFTER EXPLICIT APPROVAL

**Files:** create temporarily `.github/workflows/phase-c4a-attribution-measurement.yml`; no source edits.

Workflow name exactly `Phase C4-A Attribution Measurement`. Hard-code `C4A_ATTRIBUTION_SCIENCE_HEAD` and Task 10 prospective manifest SHA.

- [ ] Checkout frozen diagnostic head, recreate exact lock, regenerate prospective manifest byte-for-byte and verify SHA before fitting.
- [ ] Run exactly one registered `measure`; D0 executes first and must fail closed before D1-D5 if any frozen replay mismatch occurs.
- [ ] Strictly verify `manifest.json`, `result.json`, `provenance.json`, `trace-index.json` and every referenced large artifact hash/size. Upload sealed bundle. Scientific workflow success means valid execution, not a preferred attribution.
- [ ] Record run/job and manifest/result/provenance/trace-index/artifact SHA-256 values. Do not alter the registered C4-A verdict and do not create C4-B.
- [ ] **STOP for interpretation review before freeze.** Present D1-D5 evidence and the prospective Outcome A-E options from the spec. Because the spec defines no numeric “frequently/mostly” gate, do not auto-select an outcome in `result.json`.

Measurement command:

```bash
python scripts/diagnose_phase_c4a_failure.py measure \
  --manifest /tmp/c4a-attribution-prospective/manifest.json \
  --output /tmp/c4a-attribution-result
```

---

### Task 12: No-Refit Attribution Freeze, Interpretation Report, and Cleanup

**Files:** create temporarily `.github/workflows/phase-c4a-attribution-freeze.yml`; freeze only:

```text
docs/experiments/phase-c4a-failure-attribution/
  manifest.json
  result.json
  provenance.json
  report.md
  trace-index.json
```

Then delete measurement/freeze workflows separately.

- [ ] After Task 11 evidence review, freeze workflow downloads the exact measurement artifact by run ID, verifies all expected SHA-256 values, and **does not call `measure`**.
- [ ] Supply the reviewed `DiagnosticOutcome` A-E plus a concrete rationale tuple to `render_markdown_report`; verifier checks the label is legal, D0 is valid, required mechanism rows exist, and rationale references only sealed evidence. No new fit, permutation, threshold or score is generated during freeze.
- [ ] Commit exactly the five evidence/report files; compare parent→freeze commit and require no source/test/requirements/frozen C4-A changes.
- [ ] Delete measurement and freeze workflow files in separate cleanup commits; run final exact-head permanent CI + preflight.
- [ ] Final report records final repo head, frozen diagnostic science head, measurement/freeze/final CI run IDs, all evidence hashes, D0 validity, reviewed Outcome A-E + rationale, mechanism table, and explicit statements that registered C4-A remains `operator_passed=false` and C4-B remains unauthorized.

No mechanism-changing C4-A replacement experiment is part of this plan. Any replacement operator, representation, control, threshold, seed set, or C4-B work requires a new design/spec.
