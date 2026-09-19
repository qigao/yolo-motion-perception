# R1-E3M Label-Free Real-Track ESN Memory Validation

Status: design for issue #15.

## Motivation

R1-E3 semantic transfer asks whether a fixed temporal reservoir representation
improves classification of `approach/touch/pick_up/pass_by` on real tracked
video. That claim correctly requires independent human semantic labels.

R1-E3M asks a narrower mechanism question that can be answered earlier:

> On immutable real YOLO11 + BoT-SORT track trajectories, does a fixed ESN
> reservoir state retain linearly decodable information about earlier
> observations beyond what is available from the current observation alone?

R1-E3M does not make a behavior-recognition claim. Human semantic annotation is
therefore not an input to this experiment.

## Relationship to R1-E3

R1-E3M is a parallel, pre-semantic mechanism experiment.

- #8 / PR #9 remain the registered semantic transfer experiment.
- #11 remains mandatory for semantic behavior claims.
- R1-E3M may proceed while #11 human review is incomplete.
- R1-E3M may reuse frozen source-video bytes, the extraction contract, and
  reservoir implementations, but it must not read semantic annotation files,
  MEVA activity labels as targets, or semantic episode boundaries.

A positive R1-E3M result cannot upgrade or substitute for a semantic R1-E3
result.

## Frozen real-video source set

Use the already-frozen Batch-1 MEVA source set:

- 11 source videos;
- 7 train / 4 eval;
- 4 train / 2 eval guarded source-window components;
- source manifest SHA-256:
  `1db8cc200ba9eaab2c14ed260b06ed582ed75d54f57c3036e1312cd2f184f4cf`;
- source-video artifact digest:
  `sha256:211e41cf0d6100b13b8194dc81d2faf7dcb492984459de674f0eefe5fcacc838`.

Whole-video train/eval assignment is inherited unchanged.

## Artifact boundary

```text
frozen real videos
      ↓
fixed YOLO11 + fixed BoT-SORT
      ↓
immutable raw per-track rows
      ↓
deterministic 2.0 s track windows / 20 bins
      ↓
window manifest + hashes
================ measurement boundary ================
fixed ESN reservoirs
      ↓
M0 current-observation Ridge
M1 reservoir-state Ridge
      ↓
delay reconstruction + no-refit controls
```

Detector/tracker execution is outside the mechanism measurement. The registered
measurement consumes only the frozen raw-track/window artifact.

## Extraction contract

Reuse the existing R1-E3 extraction contract.

The extraction manifest must freeze at minimum:

- source video SHA-256 and metadata;
- YOLO implementation/version;
- YOLO weights SHA-256;
- detector thresholds/class filter;
- BoT-SORT implementation/version;
- tracker configuration SHA-256;
- extraction code commit;
- Python/runtime identity;
- raw-track artifact SHA-256.

Frozen R1-E3M extraction settings:

- Ultralytics package: `8.4.155`;
- detector: `yolo11n.pt`;
- detector class filter: person only (COCO class 0);
- detector confidence threshold: `0.25`;
- detector IoU/NMS threshold: `0.70`;
- inference size: `640`;
- device: CPU;
- half precision: disabled;
- source sampling: every 6th native frame;
- current Batch-1 sources are 30 FPS, therefore effective detector/tracker rate
  is 5 FPS;
- timestamp and `frame_index` remain in native source-video coordinates;
- BoT-SORT configuration is repository-owned
  `configs/r1_e3m_botsort.yaml`;
- ReID is disabled;
- no detector/tracker parameter may be changed after extraction results are
  observed.

No ESN score may influence extraction or split membership.

## Frozen extraction runtime

R1-E3M freezes one detector/tracker configuration before mechanism scoring:

- detector: Ultralytics YOLO11n;
- weights logical name: `yolo11n.pt`, exact bytes recorded by SHA-256;
- class filter: person only, COCO class `0`;
- detector confidence threshold: `0.25`;
- detector NMS/IoU threshold: `0.70`;
- detector image size: `640`;
- native source-frame stride: `6`;
- for the frozen 30 fps Batch-1 videos this yields a 5 fps detector/tracker
  observation cadence;
- tracker: BoT-SORT;
- tracker configuration path: `configs/r1_e3m_botsort.yaml`;
- tracker configuration bytes and parsed values are recorded in extraction
  provenance and bound by SHA-256;
- no detector/tracker parameter may be selected or changed using an ESN
  mechanism score.

The frozen BoT-SORT configuration is:

- `track_high_thresh=0.25`;
- `track_low_thresh=0.1`;
- `new_track_thresh=0.25`;
- `track_buffer=30`;
- `match_thresh=0.8`;
- `fuse_score=true`;
- `gmc_method=sparseOptFlow`;
- `proximity_thresh=0.5`;
- `appearance_thresh=0.8`;
- `with_reid=false`.

Any change to these extraction choices creates a new raw-track artifact revision
and requires a new prospective mechanism seal.

## Raw track representation

Each raw row retains:

- `video_id`;
- `frame_index`;
- `timestamp_seconds`;
- `track_id`;
- detector `class_id`;
- detector confidence;
- normalized `cx, cy, w, h`.

Track IDs are local provenance identifiers only.

R1-E3M uses one track at a time. No actor/target semantic assignment is needed.

## Deterministic 2-second window construction

Window construction is frozen before ESN scoring.

For every track in every frozen video:

1. use source-video time as the global anchor;
2. define fixed half-open windows
   `[0,2), [2,4), [4,6), ...` seconds;
3. a track may contribute to a window only if it has observations within that
   exact source-time interval;
4. resample that track onto exactly 20 equally spaced bin centers inside the
   2.0-second interval;
5. for each bin center, use only the latest observation from the same track at
   or before that bin center;
6. accept that observation only when its age is at most `0.5` seconds;
7. never interpolate from a future detector/tracker observation;
8. bins without a valid causal observation use zero geometry/confidence and
   `presence=0`;
9. require at least 16 of 20 bins with `presence=1`;
10. reject windows with non-finite values;
11. never bridge a track-ID change;
12. never cross a source-video boundary.

This yields one immutable `20 x 6` float64 tensor per eligible window:

1. normalized `cx`;
2. normalized `cy`;
3. normalized `w`;
4. normalized `h`;
5. detector confidence;
6. presence bit.

No channel is interpreted as physical depth.

The window ID is content-addressed from
`video_id | track_id | source_start_time | source_end_time`.

Train/eval isolation remains whole-video level.

## Frozen reservoir configuration

Reuse without search:

- Flat-256;
- Grouped-4x64;
- Hierarchical-2x128;
- Hierarchical-4x64;
- total neurons: 256;
- activation: tanh;
- spectral radius: 0.9 per component;
- leak: 1.0;
- no recurrent bias;
- float64;
- Ridge regularization: `1e-6`;
- seeds: `[7,17,29,43,61]`;
- no learned recurrent weights.

Architecture is a blocking factor, never a selection target.

## Frozen delays

The delayed-observation probe uses exactly:

`[1, 2, 5, 10, 15]`

At final bin 19, delay `d` reconstructs geometry from bin `19-d`.

Targets are exactly:

`[cx, cy, w, h]`

Confidence and presence are reservoir inputs but are not reconstruction targets.

## M0 and M1 probes

For each architecture/seed/delay, fit two Ridge probes on the same training
windows and evaluate on the same evaluation windows.

### M0 — instantaneous control

Input:

- current 6-channel observation at bin 19.

Target:

- four geometry channels from bin `19-d`.

### M1 — reservoir memory

Input:

- reservoir state after processing bins 0..19 causally.

Target:

- the exact same four-channel delayed target.

Both use Ridge regularization `1e-6`.

No window/delay-specific hyperparameter tuning is allowed.

## Primary metrics

Per delay:

- mean evaluation R² across `cx,cy,w,h`;
- per-channel R²;
- mean squared error;
- sample count;
- prediction digest;
- coefficient digest.

Primary contrast:

```text
delta_R2(d) = R2(M1,d) - R2(M0,d)
```

The long-delay aggregate is frozen as:

```text
long_delay_delta = mean(delta_R2(5), delta_R2(10), delta_R2(15))
```

Short delays 1 and 2 are reported separately.

## History-destruction controls

Controls use the already-fitted M1 Ridge probe. They never refit.

### H1 — suffix reset

For each evaluation window:

1. process bins 0..15;
2. reset reservoir immediately before bin 16;
3. process bins 16..19;
4. use the resulting final state with the already-fitted M1 probe.

Expected diagnostic structure:

- delays 1/2 may remain recoverable because their targets lie in the preserved
  suffix;
- delays 5/10/15 should lose information if the normal M1 advantage truly
  depends on older recurrent history.

### H2 — deterministic prefix permutation

For each evaluation window:

- keep bins 16..19 exactly unchanged;
- deterministically permute bins 0..15 using SHA-256(window_id) as the frozen
  permutation seed;
- evaluate with the already-fitted M1 probe.

The delayed target remains the original unpermuted target. This destroys
historical order without changing the current suffix.

## Same-suffix / different-history diagnostic

Before any reservoir scoring, create an immutable audit-pair list using only
input-space distances.

Distance is RMS over all six input channels.

Freeze the prefix-separation threshold from training inputs only:

1. for each training window, compute its suffix vector from bins 16..19;
2. find the nearest suffix neighbor with a different `(video_id, track_id)`;
3. break equal-distance ties by lexical `window_id`;
4. record that pair's prefix RMS distance over bins 0..15;
5. set the frozen threshold to the median of those training prefix distances.

Then construct evaluation audit pairs:

1. for each evaluation window, find the deterministic nearest suffix neighbor
   with a different `(video_id, track_id)`;
2. canonicalize each pair by lexical window ID and remove duplicates;
3. retain the pair only when its prefix RMS distance is strictly greater than
   the frozen training-derived threshold.

No reservoir state, ESN score, label, or control result may influence pairing.

Report for each retained pair:

- suffix input distance;
- prefix input distance;
- normal final reservoir-state distance;
- H1 reset final-state distance.

This is secondary diagnostic evidence and is not used to select architectures or
windows.

## Leakage and validity gates

Fail closed on:

- source-video overlap across train/eval;
- semantic annotation file access;
- MEVA activity labels used as targets;
- duplicate window IDs;
- non-finite tensors;
- wrong tensor shape;
- track-ID bridging;
- window construction changed after scoring;
- delay set other than `[1,2,5,10,15]`;
- architecture/seed count other than 4 x 5 = 20;
- M0/M1 train/eval window mismatch;
- H1/H2 readout refit;
- artifact digest mismatch;
- exact-head/runtime mismatch.

## Registered interpretation

The mechanism result is summarized without architecture selection.

For each of 20 architecture/seed arms compute `long_delay_delta`.

### Outcome M-A — robust real-track memory benefit

Both hold:

- median `long_delay_delta > 0`;
- at least 16 of 20 arms have `long_delay_delta > 0`;
- median M1 long-delay R² drops under H1 relative to normal evaluation.

### Outcome M-B — positive but inconsistent

- median `long_delay_delta > 0`;
- M-A is not satisfied.

### Outcome M-C — no demonstrated memory advantage

- median `long_delay_delta <= 0`.

H2 is supporting evidence and cannot upgrade M-B/M-C to M-A.

No magnitude threshold is tuned from observed results.

## Evidence lifecycle

1. committed design;
2. committed TDD implementation plan;
3. code/tests only;
4. frozen extraction manifest;
5. one accepted YOLO11 + BoT-SORT extraction over the already-frozen videos;
6. raw-track artifact freeze;
7. deterministic track-window artifact freeze;
8. artifact verifier proves split/window/digest invariants;
9. prospective mechanism manifest bound to exact code head + artifact root;
10. explicit approval before registered mechanism measurement;
11. exactly one registered measurement;
12. no-refit outcome classification;
13. raw evidence freeze;
14. final CI.

Ordinary CI never runs detector/tracker extraction and never runs the registered
mechanism measurement.

## Claims boundary

A positive result supports only:

> Under the frozen real-track corpus and preregistered controls, fixed ESN
> reservoir states retain linearly decodable information about earlier
> YOLO/BoT-SORT track geometry beyond the instantaneous current observation.

It does not establish semantic understanding, action recognition, physical
depth, persistent identity, or superiority of any reservoir topology.
