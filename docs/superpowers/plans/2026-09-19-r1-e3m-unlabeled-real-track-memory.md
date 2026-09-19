# R1-E3M Unlabeled Real-Track Temporal-Memory Validation Implementation Plan

**Issue:** #13  
**Spec:** `docs/superpowers/specs/2026-09-19-r1-e3m-unlabeled-real-track-memory-design.md`

## Goal

Implement the preregistered R1-E3M mechanism experiment without using human
semantic labels. The registered question is whether fixed ESN/reservoir states
retain linearly recoverable information about earlier real YOLO11 + BoT-SORT
observations beyond current/suffix-only baselines, with fail-closed
history-destruction controls.

## Global constraints

- Preserve issue #8 / PR #9 semantic R1-E3 unchanged.
- Do not consume #11 semantic annotations or review outcomes.
- Use frozen Batch-1 source manifest SHA-256
  `1db8cc200ba9eaab2c14ed260b06ed582ed75d54f57c3036e1312cd2f184f4cf`.
- Source split remains whole-video and source-window-component isolated.
- Window construction is fixed at 4.0 s, non-overlapping, 20 bins at 5 Hz.
- Actor/target presence gate is fixed at 16/20 bins.
- Delays are exactly `[1,2,5,10,15]`; delay 10 is primary.
- Reservoir architectures/seeds/Ridge regularization are inherited unchanged.
- Registered measurement is not allowed in ordinary CI.
- No semantic labels, event boundaries, topology search, delay search, window
  search, threshold tuning, or post-result seed expansion.

---

## Task 1 — Unlabeled track artifact schema

**Create**

- `src/neural_state_machine/r1_e3m_artifact.py`
- `tests/test_r1_e3m_artifact.py`

**Responsibilities**

- load canonical source-video metadata and raw `tracks.jsonl`;
- bind exact Batch-1 source-manifest SHA;
- validate track ordering, frame bounds, finite geometry/confidence;
- validate zero source-video/source-window-component overlap;
- reject semantic episode/label payloads from the mechanism artifact;
- compute canonical artifact root digest.

**RED gates**

- wrong source-manifest SHA fails;
- duplicate video SHA fails;
- train/eval source-window component overlap fails;
- unsorted track rows fail;
- semantic `episodes.json` / behavior labels are rejected;
- valid minimal artifact has stable root digest.

---

## Task 2 — Deterministic 4-second / 20-bin pair windows

**Create**

- `src/neural_state_machine/r1_e3m_windows.py`
- `tests/test_r1_e3m_windows.py`

**Responsibilities**

- absolute windows `[0,4),[4,8),...`;
- sample times `s + (i+0.5)*0.2`;
- interpolation only across raw gaps <=0.5 s;
- actor modal class must be person class 0;
- actor/target each present >=16/20;
- retain every eligible directed pair;
- output finite float64 `20 x 14`;
- stable sequence IDs from source/artifact/window/track identities;
- no track/class IDs in numeric features.

**RED gates**

- windows never overlap;
- no partial end window;
- interpolation gap >0.5 yields missing;
- 15/20 presence rejects, 16/20 accepts;
- modal-class tie uses lower class ID;
- repeated build is byte/digest deterministic.

---

## Task 3 — Multi-output Ridge regression law

**Create**

- `src/neural_state_machine/r1_e3m_regression.py`
- `tests/test_r1_e3m_regression.py`

**Responsibilities**

- multi-output float64 Ridge with explicit unpenalized bias;
- regularization fixed to `1e-6`;
- deterministic coefficient digest;
- `R²` per channel and macro over non-degenerate channels;
- report degenerate `SST<=1e-12` channels.

**RED gates**

- exact affine synthetic target reconstructs;
- bias is not penalized;
- wrong/non-finite shapes fail closed;
- degenerate channels excluded, not silently assigned zero.

---

## Task 4 — Reservoir representations and frozen arms

**Create**

- `src/neural_state_machine/r1_e3m_representation.py`
- `tests/test_r1_e3m_representation.py`

**Responsibilities**

- reuse existing R1-E2/R1-E3 architecture builders;
- exactly four architectures x five seeds;
- C0 = raw bin 19;
- S4 = flattened bins 16..19;
- R1 = reservoir state at bin 19;
- R4 = mean reservoir states 16..19;
- same reservoir trajectory for R1/R4;
- parameter digest must stay fixed within an arm.

**RED gates**

- arm count exactly 20;
- S4 exact shape 56;
- R1/R4 shape 256;
- no architecture selection/ranking helper.

---

## Task 5 — M1 delay reconstruction benchmark

**Create**

- `src/neural_state_machine/r1_e3m_benchmark.py`
- `tests/test_r1_e3m_benchmark.py`

**Responsibilities**

- train on frozen training-video windows only;
- evaluate frozen evaluation-video windows only;
- delays exactly `1,2,5,10,15`;
- fit C0/S4 and per-arm R1/R4 Ridge probes;
- report per-channel and macro R²;
- compute `delta_R2 = R1 - S4`;
- delay 10 is the primary registered contrast.

**RED gates**

- delay outside registry fails;
- target bin is exactly `19-delay`;
- no eval sample enters fitting;
- delay 1/2 sanity fixture lets S4 directly reconstruct target;
- synthetic memory fixture gives positive long-delay R1 delta.

---

## Task 6 — M2 history-destruction controls

**Extend**

- `src/neural_state_machine/r1_e3m_benchmark.py`
- `tests/test_r1_e3m_history_destruction.py`

**Responsibilities**

- reset immediately before bin 16;
- reverse bins 0..15 only;
- deterministic SHA-256 shuffle of bins 0..15 only;
- preserve bins 16..19 byte-identically;
- reuse already fitted probes with no refit;
- compute reset/reverse/shuffle R² drops.

**RED gates**

- suffix bytes are identical for all controls;
- shuffle is stable per sequence ID;
- changing sequence ID changes deterministic shuffle where possible;
- refit API unavailable from control runner;
- known-memory fixture degrades after reset.

---

## Task 7 — M3 suffix-matched separability

**Create**

- `src/neural_state_machine/r1_e3m_separability.py`
- `tests/test_r1_e3m_separability.py`

**Responsibilities**

- standardize with training-only mean/std;
- std floor `1e-12`;
- eval neighbor must come from a different source video;
- nearest S4 neighbor with lexical sequence-ID tie break;
- deduplicate unordered pairs;
- report suffix/prefix/R1/R4 distances;
- deterministic Spearman correlation.

This diagnostic is secondary and cannot change Outcome A/B/C.

---

## Task 8 — Registered outcome classifier and protocol

**Create**

- `src/neural_state_machine/r1_e3m_protocol.py`
- `tests/test_r1_e3m_protocol.py`

**Responsibilities**

- exactly 20 reservoir arms;
- calculate per-arm `delta10` and `reset10`;
- Outcome A:
  - median delta10 >=0.05;
  - >=16/20 delta10 >0;
  - median reset10 >=0.05;
  - >=16/20 reset10 >0;
- Outcome B: both medians >0 without A;
- Outcome C: either median <=0;
- secondary diagnostics cannot upgrade outcome.

---

## Task 9 — Prospective evidence lifecycle

**Create**

- `src/neural_state_machine/r1_e3m_evidence.py`
- `scripts/benchmark_r1_e3m_memory.py`
- `tests/test_r1_e3m_evidence.py`
- `tests/test_r1_e3m_cli.py`

**Commands**

- `protocol`
- `prepare`
- `measure`
- `verify`

**Responsibilities**

- bind exact science head;
- bind source-manifest SHA;
- bind track/window artifact root digest;
- bind sequence counts/splits;
- bind exact architectures/seeds/delays/runtime;
- write-once prospective manifest;
- require explicit approval marker before `measure`;
- exactly-one registered result;
- no-refit evidence freeze.

Ordinary CI may run `protocol/prepare/verify` on synthetic fixtures but may not
run a registered real-data `measure`.

---

## Task 10 — Permanent CI gate

**Modify**

- `.github/workflows/ci.yml`

**Requirements**

- focused R1-E3M unit tests;
- Ruff;
- no network/model download;
- no YOLO/BoT-SORT execution;
- no registered measurement;
- scope test proving R1-E3M mechanism modules do not import semantic annotation
  artifacts.

---

## Task 11 — External R1-E3M extraction tool

**Create on isolated extraction surface**

- `tools/r1_e3m_extract_tracks.py`
- `tests/test_r1_e3m_extract_contract.py`
- dedicated one-shot/manual workflow

**Requirements**

- consume only frozen Batch-1 source manifest/video artifact;
- fixed YOLO11 + BoT-SORT configuration frozen before run;
- output canonical raw tracks and exact provenance;
- never read #11 semantic review files;
- no reservoir code/model score in source/video selection;
- retain rejected extraction attempts as audit evidence.

This task freezes a real-track artifact but does **not** run the registered ESN
measurement.

---

## Task 12 — Real window artifact freeze

Using the accepted Task-11 extraction:

- build all eligible 4 s windows;
- freeze `20 x 14` tensors and sequence manifest;
- report train/eval video/component/sequence counts;
- verify all hashes and zero split overlap;
- record artifact root digest;
- do not compute registered reservoir results.

If construction yields an implementation/resource blocker, stop and revise the
design before measurement. Do not sample based on convenience or scores.

---

## Task 13 — Prospective registration gate

Prepare the exact R1-E3M manifest containing:

- science head;
- source-manifest SHA;
- real-track/window artifact root;
- train/eval sequence counts;
- four architectures;
- five seeds;
- delays `[1,2,5,10,15]`;
- primary delay 10;
- Ridge `1e-6`;
- runtime/environment digest;
- expected 20-arm count.

No registered mechanism score is computed here.

---

## Task 14 — Exactly-one registered mechanism measurement

Only after the prospective manifest is explicitly approved:

- run once;
- freeze raw per-arm M1/M2/M3 evidence;
- classify Outcome A/B/C without refit;
- preserve coefficient/prediction/control digests;
- remove/disable one-shot measurement path if required by the evidence policy;
- run final exact-head CI.

No semantic claim is permitted from this task.
