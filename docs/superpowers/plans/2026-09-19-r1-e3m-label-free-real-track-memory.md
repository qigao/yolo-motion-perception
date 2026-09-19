# R1-E3M Label-Free Real-Track ESN Memory Validation Implementation Plan

**Issue:** #15  
**Spec:** `docs/superpowers/specs/2026-09-19-r1-e3m-label-free-real-track-memory-design.md`

## Goal

Implement a label-free real-track ESN mechanism experiment that measures
delayed-observation reconstruction on frozen YOLO11 + BoT-SORT trajectories,
with no human behavior annotations and no semantic action claim.

## Global constraints

- Reuse the frozen 11-video Batch-1 source set.
- Reuse the existing R1-E3 extraction contract.
- Never read #11 semantic annotations, semantic episode bounds, or MEVA KPF
  activity labels as mechanism targets.
- Keep train/eval split at whole-video level.
- Window duration is exactly 2.0 seconds.
- Window tensor shape is exactly `20 x 6`.
- Presence gate is exactly >=16/20 bins.
- Interpolation gap limit is exactly 0.5 seconds.
- Delays are exactly `[1,2,5,10,15]`.
- Long-delay aggregate uses exactly `[5,10,15]`.
- Architectures/seeds remain exactly 4 x 5 = 20 arms.
- Ridge regularization remains `1e-6`.
- Ordinary CI never extracts YOLO/BoT-SORT and never runs registered mechanism
  measurement.
- Exactly one registered mechanism measurement is allowed after prospective
  sealing and explicit approval.

---

## Task 1 — Mechanism window schema and canonical hashing

**Create**

- `src/neural_state_machine/r1_e3m_artifact.py`
- `tests/test_r1_e3m_artifact.py`

### RED tests

Cover:

- `TrackWindow` requires a `20 x 6` float64 tensor;
- tensor is finite and immutable;
- split is only `train|eval`;
- window ID is deterministic from video/track/source bounds;
- duplicate window IDs fail closed;
- malformed SHA-256 fails closed;
- train/eval source-video overlap fails closed;
- semantic label fields are absent from the schema.

### GREEN implementation

Define immutable types for:

- `MechanismVideoRecord`;
- `RawTrackRow` reuse/adaptation;
- `TrackWindow`;
- `MechanismArtifactManifest`.

Implement canonical JSON/hash helpers.

### Verify

```bash
pytest -q tests/test_r1_e3m_artifact.py
ruff check src/neural_state_machine/r1_e3m_artifact.py tests/test_r1_e3m_artifact.py
```

---

## Task 2 — Deterministic 2-second / 20-bin track-window builder

**Create**

- `src/neural_state_machine/r1_e3m_window.py`
- `tests/test_r1_e3m_window.py`

### RED tests

Prove:

- windows are anchored to source-video time `[0,2),[2,4),...`;
- no track ID is bridged;
- output shape is exactly `20 x 6`;
- interpolation never crosses a >0.5s bracketing gap;
- missing bins are zero geometry/confidence with presence=0;
- <16 present bins rejects window;
- exactly 16 present bins is accepted;
- no non-finite tensor can be emitted;
- repeated construction yields identical window IDs and tensor digests;
- semantic annotations are not accepted as inputs.

### GREEN implementation

Provide:

- `build_track_windows(raw_rows, video_manifest)`;
- deterministic interpolation;
- deterministic sort order:
  `video_id, track_id, source_start_seconds`.

---

## Task 3 — Window artifact freeze and verifier

**Create**

- `src/neural_state_machine/r1_e3m_artifact_evidence.py`
- `scripts/prepare_r1_e3m_artifact.py`
- `tests/test_r1_e3m_artifact_evidence.py`
- `tests/test_r1_e3m_artifact_cli.py`

### Gates

Freeze only if:

- source manifest digest matches frozen Batch-1 identity;
- raw-track digest is valid;
- window manifest/tensor digests are valid;
- no video split overlap;
- both train and eval contain eligible windows;
- no duplicate IDs;
- all tensors satisfy schema;
- semantic annotation paths/fields are absent.

Freeze is write-once.

The CLI packages/verifies artifacts only; it never imports Ultralytics,
BoT-SORT, Torch inference, or semantic annotation tooling.

---

## Task 4 — Label-free mechanism dataset loader

**Create**

- `src/neural_state_machine/r1_e3m_dataset.py`
- `tests/test_r1_e3m_dataset.py`

Load frozen windows into immutable arrays:

- train tensors;
- eval tensors;
- window IDs/video IDs;
- frozen artifact root digest.

Reject any dataset mutation or split mismatch.

No labels exist in this loader.

---

## Task 5 — M0/M1 delayed-reconstruction probes

**Create**

- `src/neural_state_machine/r1_e3m_probe.py`
- `tests/test_r1_e3m_probe.py`

### Frozen delays

`[1,2,5,10,15]`.

### M0

Current input at bin 19 -> Ridge -> geometry at bin `19-d`.

### M1

Final reservoir state after bins 0..19 -> same Ridge law -> same target.

### Tests

- M0/M1 receive identical train/eval windows;
- target bin is exactly `19-d`;
- target channels are exactly `cx,cy,w,h`;
- regularization exactly `1e-6`;
- no exposed delay/window hyperparameter in registered public runner;
- R², per-channel R², MSE, prediction/coefficient digests deterministic.

---

## Task 6 — No-refit history-destruction controls

**Create**

- `src/neural_state_machine/r1_e3m_controls.py`
- `tests/test_r1_e3m_controls.py`

### H1 reset

Reset immediately before bin 16 and process bins 16..19.

### H2 prefix permutation

Keep bins 16..19 identical. Permute bins 0..15 deterministically using
SHA-256(window_id).

### Tests

- control readout coefficient digest exactly equals normal M1 digest;
- no refit is reachable from control API;
- suffix is byte-identical;
- H2 permutation repeatable;
- target values remain original/unpermuted.

---

## Task 7 — 20-arm benchmark and outcome classifier

**Create**

- `src/neural_state_machine/r1_e3m_benchmark.py`
- `tests/test_r1_e3m_benchmark.py`

Run exactly:

- four inherited architectures;
- seeds `[7,17,29,43,61]`.

Per arm report:

- M0/M1 metrics at all five delays;
- `delta_R2(d)`;
- `long_delay_delta = mean(d=5,10,15)`;
- H1/H2 metrics and drops;
- reservoir/window/probe digests.

Outcome:

- M-A: median long-delay delta >0, >=16/20 positive arms, and median H1
  long-delay drop >0;
- M-B: median long-delay delta >0 but M-A not met;
- M-C: median long-delay delta <=0.

No architecture winner is emitted.

---

## Task 8 — Same-suffix / different-history diagnostic

**Create**

- `src/neural_state_machine/r1_e3m_pairs.py`
- `tests/test_r1_e3m_pairs.py`

Pairing is frozen before reservoir scoring and uses input space only.

- suffix = bins 16..19;
- prefix = bins 0..15;
- deterministic nearest-neighbor candidate ordering;
- no same window;
- no semantic labels;
- pair list digest frozen.

Report normal vs reset reservoir-state distances.

Diagnostic only; never feeds selection or outcome classification.

---

## Task 9 — Prospective evidence lifecycle and CLI

**Create**

- `src/neural_state_machine/r1_e3m_evidence.py`
- `scripts/benchmark_r1_e3m_memory.py`
- `tests/test_r1_e3m_evidence.py`
- `tests/test_r1_e3m_cli.py`

Commands:

- `protocol --artifact-root DIR`
- `prepare --artifact-root DIR --output DIR --scientific-head SHA`
- `verify --root DIR [--no-result-ok]`
- `measure --artifact-root DIR --root DIR --manifest-sha256 SHA`

Measure must fail before scientific execution on:

- head mismatch;
- artifact digest mismatch;
- runtime mismatch;
- result already exists;
- wrong delays;
- wrong 20-arm identity.

---

## Task 10 — Permanent CI

Modify:

- `.github/workflows/ci.yml`

CI may run:

- focused E3M tests;
- artifact schema/verifier tests;
- protocol smoke;
- prospective prepare/verify tests.

CI must not:

- download source video;
- run YOLO/BoT-SORT;
- access semantic review artifacts;
- execute registered E3M measurement.

Add a source scan test enforcing these invariants.

---

## Task 11 — Real raw-track/window artifact generation gate

No mechanism scoring yet.

- [ ] download exact frozen 11-video artifact;
- [ ] run one sealed YOLO11 + BoT-SORT extraction under the existing extraction
  contract;
- [ ] record detector/tracker/runtime digests;
- [ ] freeze immutable raw-track rows;
- [ ] build deterministic 2-second windows;
- [ ] freeze window manifest/tensors;
- [ ] verify artifact twice with identical root digest;
- [ ] record train/eval window counts and source-video coverage;
- [ ] stop if extraction replay/digest gate fails.

No human annotations are read.

---

## Task 12 — Prospective mechanism measurement gate

- [ ] exact-head tests + Ruff + CI green;
- [ ] frozen E3M artifact verified;
- [ ] prepare prospective manifest bound to exact science head + artifact root;
- [ ] record runtime, delays, arm count=20, window counts;
- [ ] prove result file absent;
- [ ] STOP for explicit approval before measurement.

---

## Task 13 — Exactly-one mechanism measurement / no-refit freeze

- [ ] execute one approved detached-head measurement;
- [ ] verify 20 arms and all five delays;
- [ ] compute M-A/B/C without refit;
- [ ] retain H1/H2 and pair diagnostics;
- [ ] freeze raw result bytes and digests;
- [ ] remove one-shot measurement workflow/tooling if any;
- [ ] final CI verifies frozen evidence only.

## Claims boundary

This plan can establish only label-free temporal memory on frozen real tracking
trajectories. It cannot establish behavior recognition or replace #11.
