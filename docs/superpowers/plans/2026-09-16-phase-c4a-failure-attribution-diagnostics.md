# Phase C4-A Failure-Attribution Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an additive, sealed diagnostic system that explains why the registered C4-A shuffled negative control remains predictive for seeds 17 and 29, without changing the frozen C4-A operator, evidence, seeds, thresholds, or ridge penalty.

**Architecture:** Keep the frozen C4-A code and evidence immutable. Add a separate `phase_c4a_diagnostics` package that first reproduces the registered replay exactly (D0), then computes geometry (D1), target/projection and supervised-reference attribution (D2), score decomposition (D3), a fixed block10/global permutation grid (D4), and a mechanism-level attribution report (D5). Permanent CI only tests deterministic code and no-result sealing; `Phase C4-A Attribution Preflight`, `Phase C4-A Attribution Measurement`, and `Phase C4-A Attribution Freeze` are distinct stages, with a mandatory human stop before measurement.

**Tech Stack:** Python 3.12.14, NumPy 2.5.3, pytest 9.1.1, Ruff 0.15.22, GitHub Actions, canonical JSON + SHA-256 sealing.

**Spec:** `docs/superpowers/specs/2026-09-16-phase-c4a-failure-attribution-diagnostics-design.md`

## Global Constraints

- Execute from an isolated branch/worktree created from exact spec head `26cb9ccaeab03d7dd368337ba17af7318a3fca78`; suggested branch: `diagnostics/phase-c4a-failure-attribution-v1`.
- Frozen C4 scientific head: `8ae3154950ed53c4d0a0f555463042ff72674d31`.
- Frozen C4-A evidence commit: `832c4a6874b87bba1b189e2d3604f66025730bb9`.
- Frozen C4-A manifest SHA-256: `a990745cf6f0e8c5fd7ae5d9a7189737b30a40168a4db3df1a6d3043d6d5876a`.
- Frozen C4-A result SHA-256: `7f398f474dba928d2a6aa993d0801563aa05af20faa8cd9a61ec94a6ca4697a4`.
- Frozen C4-A provenance SHA-256: `8fb47b5e670754bc44c9ea70dd4cedce40e6f52165923142114577b4aec00351`.
- Formal head: `8b2180ed24b6ff03db4b927ec29cdd9903b0ccac`.
- Registered seeds remain exactly `(7, 17, 29)`.
- Hidden size `64`, recurrent radius `0.9`, training decisions `2000`, public hidden-delay support `(1, 3, 5)`, exactly five drain clocks, and exactly `2005` C4-A design/feedback rows remain fixed.
- Ridge penalty remains exactly `1e-6`, with bias penalized; no regularization search is allowed.
- Registered shuffled threshold remains strictly `<150/200`; no gate or seed changes are allowed.
- The eight secondary evaluation sets remain diagnostic only and never enter fitting, selection, or acceptance.
- Never modify frozen C3 or C4-A scientific modules or any of the frozen C4-A manifest/result/provenance/report files.
- C4-B must not appear in any diagnostic workflow, result schema, or runner.
- D0 is a hard prerequisite. If D0 fails, no D1-D5 scientific interpretation or registered diagnostic measurement may proceed.
- Diagnostic measurement is not authorized by implementing Tasks 1-10. Task 10 ends with a mandatory stop and explicit human approval gate.

---

## File Map

Create these focused files; do not add diagnostic logic to the frozen `phase_c4_*` modules:

- `src/neural_state_machine/phase_c4a_diagnostics/__init__.py` — public diagnostic exports only.
- `src/neural_state_machine/phase_c4a_diagnostics/model.py` — immutable diagnostic dataclasses and constants.
- `src/neural_state_machine/phase_c4a_diagnostics/integrity.py` — frozen evidence verification, registered replay reconstruction, D0 digests.
- `src/neural_state_machine/phase_c4a_diagnostics/geometry.py` — D1 SVD/rank/leverage/ridge geometry.
- `src/neural_state_machine/phase_c4a_diagnostics/attribution.py` — D2 target associations/projections and supervised reference alignment.
- `src/neural_state_machine/phase_c4a_diagnostics/decomposition.py` — D3 margin/weight component decomposition.
- `src/neural_state_machine/phase_c4a_diagnostics/permutations.py` — D4 fixed block10/global reward permutation grid.
- `src/neural_state_machine/phase_c4a_diagnostics/report.py` — D5 mechanism table and prospective outcome classification.
- `src/neural_state_machine/phase_c4a_diagnostics/evidence.py` — canonical manifest/result/provenance/trace-index sealing and strict verification.
- `scripts/diagnose_phase_c4a_failure.py` — protocol/preflight/measurement CLI.
- `scripts/verify_phase_c4a_failure_diagnostics.py` — strict evidence verifier CLI.
- `requirements/phase-c4a-diagnostics.in` and `requirements/phase-c4a-diagnostics-python312.lock` — exact diagnostic environment, initially identical package versions to the passing C4 lock.
- Tests: `tests/test_phase_c4a_diag_{model,integrity,geometry,attribution,decomposition,permutations,report,evidence,cli}.py`.
- Permanent CI: modify `.github/workflows/ci.yml` only to add deterministic diagnostic tests/Ruff/no-result checks.
- Dedicated preflight: create `.github/workflows/phase-c4a-attribution-preflight.yml` as a permanent stage.
- Measurement/freeze workflows are one-shot files created only after the explicit Task 10 approval and deleted after use.

---

### Task 1: Frozen Diagnostic Identity and Immutable Model Types

**Files:**
- Create: `src/neural_state_machine/phase_c4a_diagnostics/__init__.py`
- Create: `src/neural_state_machine/phase_c4a_diagnostics/model.py`
- Test: `tests/test_phase_c4a_diag_model.py`

**Interfaces:**
- Produces: `FrozenC4AIdentity`, `DiagnosticConfig`, `FrozenArray`, `sha256_file(path)`, `freeze_float64(array)`.
- Consumes: no new scientific inputs; constants copied exactly from the spec.

- [ ] **Step 1: Write failing identity and immutability tests**

```python
from neural_state_machine.phase_c4a_diagnostics.model import (
    DiagnosticConfig,
    FrozenC4AIdentity,
    freeze_float64,
)


def test_identity_freezes_registered_c4a_inputs():
    identity = FrozenC4AIdentity.registered()
    assert identity.scientific_head == "8ae3154950ed53c4d0a0f555463042ff72674d31"
    assert identity.formal_head == "8b2180ed24b6ff03db4b927ec29cdd9903b0ccac"
    assert identity.seeds == (7, 17, 29)
    assert identity.design_rows == 2005
    assert identity.ridge_penalty == 1e-6


def test_frozen_arrays_are_detached_and_read_only():
    source = np.arange(6, dtype=np.float64).reshape(2, 3)
    frozen = freeze_float64(source)
    source[:] = -1
    assert frozen.flags.writeable is False
    np.testing.assert_array_equal(frozen, np.arange(6, dtype=np.float64).reshape(2, 3))
```

- [ ] **Step 2: Run the tests and require RED**

Run: `pytest -q tests/test_phase_c4a_diag_model.py`
Expected: import failure because `phase_c4a_diagnostics.model` does not exist.

- [ ] **Step 3: Implement exact immutable types**

Use frozen/slots dataclasses. `DiagnosticConfig.registered()` must expose exactly:

```python
DiagnosticConfig(
    permutation_replicates=32,
    permutation_lineage=0x43344144,
    modes=("block10", "global"),
    near_zero_margin=1e-9,
)
```

`freeze_float64()` must use `np.ascontiguousarray(..., dtype=np.float64).copy()` and set `flags.writeable = False`.

- [ ] **Step 4: Run focused tests and Ruff**

Run:
```bash
pytest -q tests/test_phase_c4a_diag_model.py
ruff check src/neural_state_machine/phase_c4a_diagnostics/model.py tests/test_phase_c4a_diag_model.py
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/neural_state_machine/phase_c4a_diagnostics tests/test_phase_c4a_diag_model.py
git commit -m "feat: add C4-A diagnostic identity model"
```

---

### Task 2: D0 Frozen Evidence Verification and Registered Replay Reconstruction

**Files:**
- Create: `src/neural_state_machine/phase_c4a_diagnostics/integrity.py`
- Test: `tests/test_phase_c4a_diag_integrity.py`

**Interfaces:**
- Produces:
  - `RegisteredConditionReplay(seed, condition, design, target, rows, actions, rewards, shuffled_rewards, schedule_digest, design_digest, target_digest, score)`.
  - `verify_frozen_c4a_evidence(root: Path) -> FrozenC4AIdentity`.
  - `replay_registered_condition(seed: int, condition: Literal["normal", "registered_shuffled"]) -> RegisteredConditionReplay`.
  - `run_d0_integrity(root: Path) -> tuple[RegisteredConditionReplay, ...]`.
- Reuse read-only behavior from `phase_c4_measurement._decision_rows_and_rewards`, `_aggregate_stream`, `_permute_reward_blocks`, `build_batch_design`, `fit_anonymous_batch_probe`, and the frozen evaluation helpers. Do not edit those source files.

- [ ] **Step 1: Write RED tests for frozen hashes and exact registered replay**

Tests must assert:

```python
replays = run_d0_integrity(repo_root)
assert [(r.seed, r.condition, r.score.correct) for r in replays] == [
    (7, "normal", 200), (7, "registered_shuffled", 147),
    (17, "normal", 200), (17, "registered_shuffled", 170),
    (29, "normal", 200), (29, "registered_shuffled", 160),
]
for replay in replays:
    assert replay.design.shape[0] == 2005
    assert replay.target.shape == (2005,)

for seed in (7, 17, 29):
    normal = next(r for r in replays if r.seed == seed and r.condition == "normal")
    shuffled = next(r for r in replays if r.seed == seed and r.condition == "registered_shuffled")
    assert normal.design_digest == shuffled.design_digest
    assert normal.design.tobytes() == shuffled.design.tobytes()
```

Add mutation tests that change one byte of copied manifest/result/provenance fixtures and require fail-closed before replay.

- [ ] **Step 2: Run focused RED**

Run: `pytest -q tests/test_phase_c4a_diag_integrity.py`
Expected: missing `integrity` module/API.

- [ ] **Step 3: Implement D0 without a new scientific equation**

Implementation rules:

```python
EXPECTED_FROZEN_HASHES = {
    "c4a-manifest.json": "a990745cf6f0e8c5fd7ae5d9a7189737b30a40168a4db3df1a6d3043d6d5876a",
    "c4a-result.json": "7f398f474dba928d2a6aa993d0801563aa05af20faa8cd9a61ec94a6ca4697a4",
    "c4a-provenance.json": "8fb47b5e670754bc44c9ea70dd4cedce40e6f52165923142114577b4aec00351",
}
```

Use registered shuffle lineage `[seed, 0x33534846]`; require five drain calls; require same schedule digest; require byte-identical design matrices between normal and registered shuffled; refit with the unchanged `fit_anonymous_batch_probe(..., penalty=1e-6)` and reproduce exact frozen counts. Return immutable arrays only.

- [ ] **Step 4: Run D0 tests, existing C4 regression tests, and Ruff**

```bash
pytest -q tests/test_phase_c4a_diag_integrity.py
pytest -q tests/test_phase_c4_*.py
ruff check src/neural_state_machine/phase_c4a_diagnostics/integrity.py tests/test_phase_c4a_diag_integrity.py
```
Expected: all pass; no frozen C4 file diff.

- [ ] **Step 5: Commit**

```bash
git add src/neural_state_machine/phase_c4a_diagnostics/integrity.py tests/test_phase_c4a_diag_integrity.py
git commit -m "feat: add C4-A diagnostic replay gate"
```

---

### Task 3: D1 Design-Matrix Geometry

**Files:**
- Create: `src/neural_state_machine/phase_c4a_diagnostics/geometry.py`
- Test: `tests/test_phase_c4a_diag_geometry.py`

**Interfaces:**
- Produces `GeometryMetrics` and `analyze_design_geometry(design: np.ndarray, *, penalty: float = 1e-6) -> GeometryMetrics`.
- `GeometryMetrics` must include shape, tolerance, rank, nullity, singular values, sigma max, smallest retained sigma, condition number, Frobenius norm, stable rank, ridge effective DoF, leverage summary, per-action-block norms, cross-block Gram norm, and bias-column norms.

- [ ] **Step 1: Write deterministic geometry tests**

Use hand-constructed full-rank and rank-deficient matrices. Assert the tolerance formula exactly:

```python
tol = max(m, n) * np.finfo(np.float64).eps * sigma_max
```

Require `rank == count(singular_values > tol)`, no explicit inverse, finite leverage values, and immutable singular-value output.

- [ ] **Step 2: Run RED**

Run: `pytest -q tests/test_phase_c4a_diag_geometry.py`
Expected: missing geometry API.

- [ ] **Step 3: Implement stable SVD geometry**

Use `np.linalg.svd(design, full_matrices=False)`. For ridge leverage use the compact SVD identity:

```python
shrink = s * s / (s * s + penalty)
leverage = np.sum((u * u) * shrink[np.newaxis, :], axis=1)
```

For the fixed two-action flattened design, derive `feature_size = n_columns // 2`; bias columns are indices `feature_size - 1` and `2 * feature_size - 1`. Cross-block Gram is `Z0.T @ Z1` and report its Frobenius norm only, not the full matrix in canonical JSON.

- [ ] **Step 4: Run focused + D0 tests and Ruff**

```bash
pytest -q tests/test_phase_c4a_diag_geometry.py tests/test_phase_c4a_diag_integrity.py
ruff check src/neural_state_machine/phase_c4a_diagnostics/geometry.py tests/test_phase_c4a_diag_geometry.py
```

- [ ] **Step 5: Commit**

```bash
git add src/neural_state_machine/phase_c4a_diagnostics/geometry.py tests/test_phase_c4a_diag_geometry.py
git commit -m "feat: add C4-A design geometry diagnostics"
```

---

### Task 4: D2 Target Associations and Design-Space Projection

**Files:**
- Create: `src/neural_state_machine/phase_c4a_diagnostics/attribution.py`
- Test: `tests/test_phase_c4a_diag_attribution.py`

**Interfaces:**
- Produces:
  - `CorrelationStat(count, correlation, reason)`.
  - `TargetProjection(parallel, residual, parallel_ratio, residual_ratio, fit_residual_norm)`.
  - `associate_target(...)` for the fixed observer variables/lags.
  - `project_target(design, target) -> TargetProjection`.
  - `fit_supervised_reference(seed, rows, labels, config) -> BatchProbeFit`.
  - `compare_weight_alignment(c4_weights, supervised_weights) -> WeightAlignment`.

- [ ] **Step 1: Write RED tests for null correlation and projection identities**

Require zero-variance inputs to return `correlation=None` plus a non-empty reason. For projection, require:

```python
np.testing.assert_allclose(projection.parallel + projection.residual, target, rtol=0, atol=1e-12)
assert abs(np.dot(projection.residual, design[:, 0])) < 1e-10
```

Add a supervised-reference test that constructs targets `+1/-1` for both action outputs and confirms penalty is exactly `1e-6`.

- [ ] **Step 2: Run RED**

Run: `pytest -q tests/test_phase_c4a_diag_attribution.py`
Expected: missing attribution API.

- [ ] **Step 3: Implement fixed association/projection formulas**

Projection must use SVD or `np.linalg.lstsq`, never `(Z.T @ Z)^-1`. Observer associations cover current action, correct label, action correctness, cue delay, realized hidden-delay category, block position, lags `1..10` in both directions, and reward provenance. Drain clocks must be excluded from decision-label correlations and reported separately.

Supervised target construction must be:

```python
y = np.full((n, 2), -1.0, dtype=np.float64)
y[np.arange(n), correct_action] = 1.0
```

Fit its flattened action-blocked weight matrix with the same `1e-6` ridge and no evaluation labels.

- [ ] **Step 4: Run focused tests, D0/D1 regression, and Ruff**

```bash
pytest -q tests/test_phase_c4a_diag_attribution.py tests/test_phase_c4a_diag_geometry.py tests/test_phase_c4a_diag_integrity.py
ruff check src/neural_state_machine/phase_c4a_diagnostics/attribution.py tests/test_phase_c4a_diag_attribution.py
```

- [ ] **Step 5: Commit**

```bash
git add src/neural_state_machine/phase_c4a_diagnostics/attribution.py tests/test_phase_c4a_diag_attribution.py
git commit -m "feat: add C4-A target attribution diagnostics"
```

---

### Task 5: D3 Weight and Evaluation-Margin Decomposition

**Files:**
- Create: `src/neural_state_machine/phase_c4a_diagnostics/decomposition.py`
- Test: `tests/test_phase_c4a_diag_decomposition.py`

**Interfaces:**
- Produces `MarginComponents`, `WeightComparison`, `decompose_readout_margin(weights, hidden_with_bias)`, `compare_weights(normal, shuffled)`, and `summarize_margin_components(...)`.

- [ ] **Step 1: Write exact recomposition tests**

For a two-action readout `W`, define decision margin as `score_correct - score_other`. Tests must require hidden contribution + bias contribution == full margin at each fixture to `rtol=0, atol=1e-12`; separately expose action-block-0 and action-block-1 signed contributions. Require recomputed decisions to match ordinary evaluation decisions for every fixture in a tiny deterministic replay.

- [ ] **Step 2: Run RED**

Run: `pytest -q tests/test_phase_c4a_diag_decomposition.py`
Expected: missing decomposition API.

- [ ] **Step 3: Implement decomposition without retraining**

Split each action row into hidden weights and final bias coefficient. Compute post-hoc bias-only/hidden-only scores only as diagnostics; never create a new fit. `compare_weights` reports norms, delta norm, global cosine, per-action cosines, block norms, and bias terms. Near-zero margins use fixed `abs(margin) <= 1e-9`.

- [ ] **Step 4: Run focused tests and Ruff**

```bash
pytest -q tests/test_phase_c4a_diag_decomposition.py tests/test_phase_c4a_diag_attribution.py
ruff check src/neural_state_machine/phase_c4a_diagnostics/decomposition.py tests/test_phase_c4a_diag_decomposition.py
```

- [ ] **Step 5: Commit**

```bash
git add src/neural_state_machine/phase_c4a_diagnostics/decomposition.py tests/test_phase_c4a_diag_decomposition.py
git commit -m "feat: add C4-A margin decomposition diagnostics"
```

---

### Task 6: D4 Fixed Block10/Global Permutation Grid

**Files:**
- Create: `src/neural_state_machine/phase_c4a_diagnostics/permutations.py`
- Test: `tests/test_phase_c4a_diag_permutations.py`

**Interfaces:**
- Produces:
  - `PermutationMode = Literal["block10", "global"]`.
  - `permutation_indices(seed, mode, replicate, n) -> np.ndarray`.
  - `permute_rewards(rewards, indices) -> tuple[float, ...]`.
  - `PermutationRow(seed, mode, replicate, permutation_digest, original_score, secondary_scores, projection_metrics, alignment_metrics)`.
  - `run_permutation_grid_for_seed(replay, config, diagnostic_config) -> tuple[PermutationRow, ...]`.
  - `summarize_permutation_rows(rows) -> PermutationSummary`.

- [ ] **Step 1: Write lineage and no-resampling RED tests**

Assert exact RNG construction:

```python
rng = np.random.Generator(
    np.random.PCG64(np.random.SeedSequence([seed, 0x43344144, mode_id, replicate]))
)
```

For `block10`, every group of ten output indices must be a permutation of the same source block. For `global`, the full output must be a permutation of `range(n)`. Require 32 unique replicate identifiers for each mode and never reject fixed points.

- [ ] **Step 2: Run RED**

Run: `pytest -q tests/test_phase_c4a_diag_permutations.py`
Expected: missing permutations API.

- [ ] **Step 3: Implement the fixed grid using the frozen design**

For each seed, reuse the D0 decision rows/design/action/schedule. Only reward values are permuted; aggregate feedback must be rebuilt through the unchanged hidden-delay schedule plus exactly five drain clocks. Refit via the unchanged `fit_anonymous_batch_probe`. Evaluate original + eight secondary sets. Record all 64 rows per seed; do not skip, retry, or resample any permutation.

`PermutationSummary` must report mean, median, min, max, nearest-rank p10/p90 using `ceil(q * 32) - 1`, and counts `<150`, `>=150`, `>=170`.

- [ ] **Step 4: Run tiny grid tests and Ruff**

Tests use `replicates=2` only for unit speed; the registered config constructor must reject any value other than 32 when `registered=True`.

```bash
pytest -q tests/test_phase_c4a_diag_permutations.py
ruff check src/neural_state_machine/phase_c4a_diagnostics/permutations.py tests/test_phase_c4a_diag_permutations.py
```

- [ ] **Step 5: Commit**

```bash
git add src/neural_state_machine/phase_c4a_diagnostics/permutations.py tests/test_phase_c4a_diag_permutations.py
git commit -m "feat: add C4-A fixed permutation diagnostics"
```

---

### Task 7: D5 Attribution Table and Prospective Outcome Classification

**Files:**
- Create: `src/neural_state_machine/phase_c4a_diagnostics/report.py`
- Test: `tests/test_phase_c4a_diag_report.py`

**Interfaces:**
- Produces `MechanismEvidence`, `DiagnosticOutcome`, `build_attribution_table(...)`, `classify_diagnostic_outcome(...)`, and `render_markdown_report(...)`.
- `DiagnosticOutcome` enum values: `BLOCK_LOCAL_ASSOCIATION`, `GLOBAL_PREDICTIVITY`, `REGISTERED_OUTLIER_UNRESOLVED`, `EVALUATION_FIXTURE_SENSITIVE`, `MULTIPLE_MECHANISMS`, `INTEGRITY_INVALID`.

- [ ] **Step 1: Write RED tests for each prospective outcome**

Use synthetic summaries to exercise Outcomes A-E exactly as described in the spec. The classifier must never return language claiming causality; the rendered report must contain separate headings for `Observed association`, `Matched counterfactual evidence`, and `Interpretation`.

- [ ] **Step 2: Run RED**

Run: `pytest -q tests/test_phase_c4a_diag_report.py`
Expected: missing report API.

- [ ] **Step 3: Implement explicit evidence table rules**

Required mechanism rows: design geometry, task-relevant target projection, bias/action shortcut, local block structure, evaluation-fixture sensitivity, unresolved/multiple mechanisms. `INTEGRITY_INVALID` short-circuits all other outcome classification if D0 is false.

- [ ] **Step 4: Run focused tests + Ruff**

```bash
pytest -q tests/test_phase_c4a_diag_report.py
ruff check src/neural_state_machine/phase_c4a_diagnostics/report.py tests/test_phase_c4a_diag_report.py
```

- [ ] **Step 5: Commit**

```bash
git add src/neural_state_machine/phase_c4a_diagnostics/report.py tests/test_phase_c4a_diag_report.py
git commit -m "feat: add C4-A attribution report model"
```

---

### Task 8: Canonical Diagnostic Evidence and Strict Verifier

**Files:**
- Create: `src/neural_state_machine/phase_c4a_diagnostics/evidence.py`
- Test: `tests/test_phase_c4a_diag_evidence.py`

**Interfaces:**
- Produces:
  - `prepare_attribution_manifest(repository_root, output_dir) -> Path`.
  - `validate_attribution_manifest(repository_root, manifest_path) -> dict[str, object]`.
  - `write_attribution_bundle(repository_root, manifest_path, output_dir, result_payload, trace_index) -> tuple[Path, Path, Path, Path]`.
  - `verify_attribution_stage(root, allow_missing_result: bool) -> None`.
- Canonical filenames: `manifest.json`, `result.json`, `provenance.json`, `trace-index.json`.

- [ ] **Step 1: Write schema/fail-closed RED tests**

Require canonical sorted compact UTF-8 newline JSON, `allow_nan=False`, regular-file/no-symlink checks, no overwrite, exact scientific/frozen-evidence hashes, exact Python environment, exact permutation lineage, exact replicate count 32, and no C4-B fields. Mutation of one key/hash/environment value must fail.

- [ ] **Step 2: Run RED**

Run: `pytest -q tests/test_phase_c4a_diag_evidence.py`
Expected: missing evidence API.

- [ ] **Step 3: Implement prospective manifest schema**

Manifest top-level keys must be exactly:

```text
schema_version
stage
scientific_head
scientific_hashes
frozen_c4a
formal_head
environment
registered_seeds
registered_scores
diagnostic_config
permutation_lineage
secondary_evaluation_manifest
expected_result_keys
```

`stage` is exactly `"c4a-failure-attribution"`. `registered_scores` records frozen normal/shuffled counts only as provenance, never recomputed acceptance. Result schema includes `integrity_valid`, `geometry`, `registered_attribution`, `permutation_rows`, `permutation_summaries`, `mechanism_table`, and `outcome`; no `operator_passed`, `behavior_passed`, or C4-B field is allowed.

- [ ] **Step 4: Run evidence tests + previous diagnostics + Ruff**

```bash
pytest -q tests/test_phase_c4a_diag_*.py
ruff check src/neural_state_machine/phase_c4a_diagnostics tests/test_phase_c4a_diag_*.py
```

- [ ] **Step 5: Commit**

```bash
git add src/neural_state_machine/phase_c4a_diagnostics/evidence.py tests/test_phase_c4a_diag_evidence.py
git commit -m "feat: seal C4-A attribution evidence"
```

---

### Task 9: Diagnostic CLI, Exact Lock, and Tiny Unregistered Integration

**Files:**
- Create: `scripts/diagnose_phase_c4a_failure.py`
- Create: `scripts/verify_phase_c4a_failure_diagnostics.py`
- Create: `requirements/phase-c4a-diagnostics.in`
- Create: `requirements/phase-c4a-diagnostics-python312.lock`
- Test: `tests/test_phase_c4a_diag_cli.py`

**Interfaces:**
- CLI commands:
  - `protocol` — D0 integrity only, no D1-D5 measurement artifact.
  - `prepare --output DIR` — prospective manifest only.
  - `measure --manifest FILE --output DIR` — explicit diagnostic D0-D5 execution.
- Verifier flags: `--root DIR` and `--no-result-ok`.

- [ ] **Step 1: Write RED CLI tests**

Require `protocol` to report `integrity_valid=true` and no score changes. `prepare` must create only `manifest.json`. `measure` must reject a manifest whose science head/environment/hash differs before any permutation fitting. Test a tiny unregistered diagnostic config through direct Python APIs rather than weakening registered CLI constants.

- [ ] **Step 2: Run RED**

Run: `pytest -q tests/test_phase_c4a_diag_cli.py`
Expected: missing scripts.

- [ ] **Step 3: Implement CLI and exact lock**

`phase-c4a-diagnostics.in` contains exactly:

```text
numpy==2.5.3
pytest==9.1.1
ruff==0.15.22
```

`phase-c4a-diagnostics-python312.lock` pins the same package set currently proven in `phase-c4-python312.lock`: iniconfig 2.3.0, numpy 2.5.3, packaging 26.3, pip 26.2.1, pluggy 1.6.0, Pygments 2.21.0, pytest 9.1.1, ruff 0.15.22, setuptools 84.0.0, wheel 0.48.0. Leave the provenance header generic until Task 10 obtains the first exact-head passing run.

- [ ] **Step 4: Run full diagnostic tests, existing C4 tests, Ruff**

```bash
pytest -q tests/test_phase_c4a_diag_*.py
pytest -q tests/test_phase_c4_*.py
ruff check src/neural_state_machine/phase_c4a_diagnostics scripts/diagnose_phase_c4a_failure.py scripts/verify_phase_c4a_failure_diagnostics.py tests/test_phase_c4a_diag_*.py
```

- [ ] **Step 5: Commit**

```bash
git add scripts/diagnose_phase_c4a_failure.py scripts/verify_phase_c4a_failure_diagnostics.py requirements/phase-c4a-diagnostics.in requirements/phase-c4a-diagnostics-python312.lock tests/test_phase_c4a_diag_cli.py
git commit -m "feat: add C4-A attribution diagnostic CLI"
```

---

### Task 10: Permanent CI, Attribution Preflight, Science-Head Freeze — MANDATORY STOP

**Files:**
- Modify: `.github/workflows/ci.yml`
- Create: `.github/workflows/phase-c4a-attribution-preflight.yml`
- Modify once after first passing preflight: `requirements/phase-c4a-diagnostics-python312.lock` provenance header only.

**Interfaces:**
- Permanent CI adds deterministic diagnostic tests/Ruff and verifies frozen C4-A evidence; it never runs the full 32x2x3 grid.
- Dedicated workflow name must be exactly `Phase C4-A Attribution Preflight`.
- Preflight artifact name: `phase-c4a-attribution-prospective-${{ github.sha }}`.

- [ ] **Step 1: Add permanent CI diagnostic gates**

Add steps after permanent C4 gates:

```yaml
- name: Phase C4-A attribution diagnostic tests
  run: pytest -q tests/test_phase_c4a_diag_*.py

- name: Phase C4-A attribution Ruff
  run: ruff check src/neural_state_machine/phase_c4a_diagnostics scripts/diagnose_phase_c4a_failure.py scripts/verify_phase_c4a_failure_diagnostics.py tests/test_phase_c4a_diag_*.py
```

Do not add `measure` to permanent CI.

- [ ] **Step 2: Add dedicated preflight workflow**

The workflow must checkout its exact head, recreate Python 3.12.14 from `phase-c4a-diagnostics-python312.lock`, `diff` `pip freeze --all` against the lock, run all diagnostic tests/Ruff, run `diagnose_phase_c4a_failure.py protocol`, run `prepare`, then run the verifier with `--no-result-ok`. Upload only the prospective directory.

- [ ] **Step 3: Verify first exact-head preflight**

Required evidence:
- permanent `ci` success;
- `Phase C4-A Attribution Preflight` success;
- no `result.json`, `provenance.json`, or permutation measurement artifact;
- prospective manifest records exactly 3 seeds, 32 replicates/mode, new lineage `0x43344144`, and frozen C4-A hashes.

- [ ] **Step 4: Bind environment provenance and rerun final exact head**

Update only the lock header with the first passing preflight run number/id. The resulting commit becomes the final candidate `C4A_ATTRIBUTION_SCIENCE_HEAD`. Require a second exact-head permanent CI + preflight success and audit the new prospective manifest SHA-256.

- [ ] **Step 5: Mandatory stop**

Report:
- `C4A_ATTRIBUTION_SCIENCE_HEAD`;
- permanent CI run/job;
- preflight run/job;
- frozen C4 scientific/formal heads;
- prospective attribution manifest SHA-256;
- canonical environment SHA-256;
- confirmation that `result.json` does not exist and no registered diagnostic grid has run.

**STOP. Do not create or run `Phase C4-A Attribution Measurement` until the user explicitly approves measurement after reviewing this evidence.**

---

### Task 11: One-Shot Registered Attribution Measurement — ONLY AFTER EXPLICIT APPROVAL

**Files:**
- Create temporarily: `.github/workflows/phase-c4a-attribution-measurement.yml`
- No production/scientific source edits.

**Interfaces:**
- Workflow name exactly `Phase C4-A Attribution Measurement`.
- Inputs are hard-coded `C4A_ATTRIBUTION_SCIENCE_HEAD` and the prospective manifest SHA from Task 10.

- [ ] **Step 1: Create one-shot workflow only after approval**

The job must checkout the frozen diagnostic science head, recreate the exact locked environment, regenerate the prospective manifest byte-for-byte, verify its SHA, then and only then invoke:

```bash
python scripts/diagnose_phase_c4a_failure.py measure \
  --manifest /tmp/c4a-attribution-prospective/manifest.json \
  --output /tmp/c4a-attribution-result
```

- [ ] **Step 2: Run D0-D5 once**

D0 failure must exit non-zero before D1-D5 interpretation. A successful workflow may contain any prospective Outcome A-E; scientific success means valid execution, not a preferred outcome.

- [ ] **Step 3: Strictly verify result/provenance/traces**

Run:

```bash
python scripts/verify_phase_c4a_failure_diagnostics.py --root /tmp/c4a-attribution-result
```

Require `manifest.json`, `result.json`, `provenance.json`, `trace-index.json`; require every artifact referenced by `trace-index.json` to match filename, byte size, and SHA-256. Upload the complete sealed bundle.

- [ ] **Step 4: Record measurement evidence**

Record run/job IDs plus manifest/result/provenance/trace-index SHA-256 values. Do not reinterpret the registered C4-A verdict and do not create a C4-B workflow.

- [ ] **Step 5: Commit no result yet**

Do not commit evidence from the measurement workflow itself. Proceed to Task 12 for a separate no-refit freeze stage.

---

### Task 12: Attribution Freeze, Report, Workflow Cleanup, Final Verification

**Files:**
- Create temporarily: `.github/workflows/phase-c4a-attribution-freeze.yml`
- Create through freeze workflow only:
  - `docs/experiments/phase-c4a-failure-attribution/manifest.json`
  - `docs/experiments/phase-c4a-failure-attribution/result.json`
  - `docs/experiments/phase-c4a-failure-attribution/provenance.json`
  - `docs/experiments/phase-c4a-failure-attribution/report.md`
  - `docs/experiments/phase-c4a-failure-attribution/trace-index.json`
- Delete separately after successful freeze:
  - `.github/workflows/phase-c4a-attribution-measurement.yml`
  - `.github/workflows/phase-c4a-attribution-freeze.yml`

**Interfaces:**
- Workflow name exactly `Phase C4-A Attribution Freeze`.
- Freeze workflow downloads Task 11 artifact by exact run ID and verifies exact hashes before committing evidence; it never imports the diagnostic runner's `measure` command.

- [ ] **Step 1: Run no-refit freeze workflow**

Require exact expected hashes and require the committed file set to be exactly the five evidence files above. Large trace artifacts remain workflow artifacts and are referenced through `trace-index.json`; do not commit matrices merely for convenience.

- [ ] **Step 2: Verify evidence-only commit diff**

Compare freeze parent to freeze commit and require no source, test, requirements, or frozen C4-A file changes.

- [ ] **Step 3: Delete one-shot workflows in separate commits**

Delete measurement and freeze workflow files after evidence is safely committed. Cleanup commit messages must not match their one-shot execution guards.

- [ ] **Step 4: Run final exact-head permanent CI + preflight**

Permanent CI and `Phase C4-A Attribution Preflight` must remain green. Preflight remains no-result with respect to a new diagnostic run; existing frozen diagnostic evidence is read-only historical output.

- [ ] **Step 5: Final report**

Report:
- final repository head;
- frozen diagnostic science head;
- measurement/freeze/final CI run IDs;
- manifest/result/provenance/trace-index hashes;
- D0 validity;
- the selected prospective Outcome A-E;
- mechanism table with observed association vs matched counterfactual evidence vs interpretation;
- explicit statement that the registered C4-A verdict remains `operator_passed=false` and C4-B remains unauthorized.

Do not begin a mechanism-changing C4-A replacement experiment in this plan. Any replacement operator, shuffle control, representation, threshold, or C4-B work requires a new design/spec.
