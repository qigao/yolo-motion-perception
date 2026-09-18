# R1-E3 Real-Track Temporal Decoder Validation

Status: committed design for issue #8.

## Motivation

R1-E2 is frozen and merged. Its registered result established a large decoder effect on deterministic synthetic temporal-composition fixtures: a fixed causal temporal-mean decoder recovered information that an instantaneous linear readout failed to extract, while architecture effects were smaller and not stable enough to support a topology winner.

R1-E3 tests the external-validity question left open by R1-E2:

> Does the fixed causal temporal decoder still improve episode classification when the reservoir consumes immutable trajectories extracted from real video by YOLO11 + BoT-SORT?

R1-E3 is not a detector benchmark and not a tracker benchmark. Detector/tracker execution is moved outside the registered measurement so their runtime variance cannot be confused with the decoder comparison.

## Scientific questions

Primary question:

1. Within a fixed reservoir architecture and seed, does a fixed causal temporal-mean representation improve real-track episode classification relative to the instantaneous final reservoir state?

Secondary questions:

2. Is any temporal-decoder advantage consistent across the four already-registered R1-E2 reservoir architectures?
3. Does destroying pre-suffix recurrent history reduce B1/B2 performance, supporting a temporal-memory interpretation rather than a final-frame shortcut?
4. Which behavior classes benefit or fail under real tracking noise?

R1-E3 does **not** rank architectures and does not select a topology.

## Frozen inheritance from R1-E2

R1-E3 inherits without search:

- architectures: Flat-256, Grouped-4x64, Hierarchical-2x128, Hierarchical-4x64;
- total neuron budget: 256;
- reservoir activation: tanh;
- recurrent spectral radius: 0.9 per component;
- leak: 1.0;
- no recurrent bias;
- float64 reservoir execution;
- Ridge regularization: `1e-6`;
- reservoir seeds: `[7, 17, 29, 43, 61]`;
- no learned recurrent weights;
- no attention, RNN, GRU, LSTM, transformer, or learned temporal pooling.

The R1-E2 frozen evidence remains unchanged.

## Two-stage architecture

R1-E3 has two deliberately separated stages.

### Stage A — real-video extraction and annotation

Stage A may use YOLO11 + BoT-SORT and video tooling. It produces a content-addressed immutable artifact. Stage A is **not** the registered scientific measurement.

### Stage B — registered NumPy-only validation

Stage B consumes only the frozen Stage-A artifact. It does not load video, YOLO weights, Ultralytics, OpenCV tracking state, or GPU kernels. All registered decoder comparisons therefore see exactly the same frozen observations.

The boundary is:

```text
real videos
   ↓
fixed YOLO11 + fixed BoT-SORT
   ↓
immutable raw track rows
   ↓
frozen manual episode annotations
   ↓
deterministic 20-bin episode tensors
   ↓
artifact hashes + manifest
================ registration boundary ================
NumPy-only reservoir trajectories
   ↓
instantaneous vs fixed causal temporal-mean Ridge
   ↓
registered metrics + frozen evidence
```

## Stage-A reproducibility contract

Before any registered R1-E3 measurement, the extraction manifest must freeze:

- source video logical ID;
- source video SHA-256;
- source/license or ownership note;
- native FPS, frame count, duration, width, and height;
- exact YOLO implementation/package version;
- exact YOLO model/weights identity and SHA-256;
- detector confidence threshold;
- detector NMS/IoU threshold;
- class filter;
- exact BoT-SORT implementation/version;
- complete tracker configuration and configuration digest;
- extraction code commit;
- Python/runtime versions used for extraction;
- raw-track artifact SHA-256;
- annotation artifact SHA-256;
- normalized episode tensor artifact SHA-256.

Changing any of these after registration creates a new experiment and cannot replace the registered R1-E3 result.

The extraction process must be deterministic at the artifact boundary: a validation rerun using the sealed extraction environment must reproduce the same raw-track and normalized-episode digests before registration. If the chosen detector runtime cannot satisfy that requirement, extraction must be moved to a deterministic runtime or the registration must fail closed.

## Real-track artifact schema

### Raw track rows

Each raw track row represents one detector/tracker observation and contains:

- `video_id`;
- `frame_index`;
- `timestamp_seconds`;
- `track_id`;
- detector `class_id`;
- detector confidence;
- absolute `x1/y1/x2/y2`;
- normalized `cx/cy/w/h`;
- frame width and height.

Rows are ordered by `video_id, frame_index, track_id`.

Track IDs are local to a video and are never treated as semantic features.

### Episode annotations

Each registered episode annotation contains:

- `episode_id`;
- `video_id`;
- `start_time_seconds`;
- `end_time_seconds`;
- `label`;
- `actor_track_id`;
- `target_track_id`;
- annotation revision;
- annotation provenance/reviewer record.

Registered labels remain:

1. `approach`
2. `touch`
3. `pick_up`
4. `pass_by`

Annotations are frozen before any registered model score is computed. Episode boundaries or labels cannot be revised in response to R1-E3 results.

The target may be a tracked object or another tracked person, but every episode must identify exactly one actor and one target.

## Eligibility and split contract

An episode is eligible only if:

- it has a frozen label and actor/target track IDs;
- duration is positive;
- the actor and target each have at least four valid tracked observations within the episode;
- the deterministic normalizer can produce all 20 bins without non-finite values.

No model score may influence eligibility.

Train/evaluation isolation is at the **whole-video level**. A source video can contribute to only one split. Frames, track IDs, or episodes from the same source video must never cross splits.

The dataset manifest freezes the exact video-level split and all eligible episode IDs before registered measurement.

No random frame-level split is permitted.

## Deterministic episode normalization

Every eligible real episode is mapped to exactly 20 temporal bins spanning its annotated start/end interval. The normalized representation is content-addressed and frozen in Stage A.

For each bin, actor and target track geometry are sampled deterministically from the frozen raw-track rows using timestamp-based linear interpolation only when the nearest bracketing observations for that track are separated by at most `0.5` seconds. Otherwise that track is represented as missing in the bin.

Interpolation applies only to normalized box geometry and confidence. Track IDs, class IDs, and labels are never interpolated.

The registered measurement does not perform this interpolation; it reads the frozen normalized tensors.

## Frozen per-bin feature vector

Each normalized bin has 14 float64 channels:

Actor channels:

1. normalized actor `cx`
2. normalized actor `cy`
3. normalized actor `w`
4. normalized actor `h`
5. actor detector confidence
6. actor presence bit

Target channels:

7. normalized target `cx`
8. normalized target `cy`
9. normalized target `w`
10. normalized target `h`
11. target detector confidence
12. target presence bit

Pair channels:

13. normalized center distance `sqrt((dx)^2 + (dy)^2) / sqrt(2)`
14. box IoU

When a track is missing in a bin, its geometry/confidence channels are zero and its presence bit is zero. If either track is missing, pair distance and IoU are zero.

No channel is called or interpreted as depth. Bounding-box size is an image-space measurement only.

Coordinates and confidences are clipped to `[0, 1]` after normalization. The complete `20 x 14` episode tensor must be finite float64.

## Fixed decoder arms

All fitted arms use the frozen multiclass Ridge law with regularization `1e-6`.

For a given architecture/seed, B1 and B2 consume the **same reservoir trajectory**.

- **B0 frame-only control:** raw 14-channel observation at normalized bin 19 -> Ridge.
- **B1 reservoir instantaneous:** reservoir state after normalized bin 19 -> Ridge.
- **B2 reservoir temporal mean:** arithmetic mean of reservoir states produced by normalized bins 16, 17, 18, and 19 -> the same Ridge law.

The B2 window is fixed at four bins because R1-E3 externalizes the already-frozen R1-E2 E2-C decoder. The window is not searched or tuned.

## History-destruction control

For each evaluation episode:

1. process bins 0..15;
2. reset the reservoir immediately before bin 16;
3. process bins 16..19;
4. score B1 and B2 using the already-fitted probes.

This preserves the final four real observations while destroying recurrent history from bins 0..15.

Unlike the balanced synthetic E2-C fixture, real data may contain label information in the final suffix, so this control has **no preregistered chance-accuracy requirement**. It is reported as a diagnostic paired with the normal evaluation.

The control may support a historical-memory interpretation only when normal B1/B2 performance exceeds the corresponding reset-control performance on the same frozen evaluation episodes.

## Metrics

Primary metric:

- macro-F1 over the four behavior classes.

Secondary metrics:

- accuracy;
- per-class precision/recall/F1;
- full `4 x 4` confusion-count matrix;
- B0/B1/B2 prediction digests;
- B0/B1/B2 coefficient digests;
- B1/B2 reset-control metrics and prediction digests;
- reservoir parameter digest;
- train/evaluation episode artifact digests;
- raw-track, annotation, normalization, and split-manifest digests.

All metrics are computed per architecture and reservoir seed.

## Primary registered contrast

For each of the 20 architecture/seed arms:

```text
delta_macro_f1 = B2 temporal-mean macro-F1 - B1 instantaneous macro-F1
```

Architecture is a blocking factor, not a selection variable.

The primary aggregate reports:

- median `delta_macro_f1` over all 20 arms;
- count of arms with `delta_macro_f1 > 0`;
- per-architecture median delta across five seeds.

No architecture is selected after measurement.

## Preregistered interpretation classes

The registered outcome is classified without refit:

### Outcome A — robust external temporal-decoder benefit

Both conditions hold:

- median `delta_macro_f1 >= 0.05`;
- at least 16 of 20 architecture/seed arms have `delta_macro_f1 > 0`.

### Outcome B — positive but inconsistent/modest transfer

- median `delta_macro_f1 > 0`;
- Outcome A is not satisfied.

### Outcome C — no registered transfer benefit

- median `delta_macro_f1 <= 0`.

These thresholds classify the registered observation; they are not tuning targets.

History-destruction results are secondary evidence and cannot upgrade Outcome B/C to Outcome A.

## Dataset-size registration gate

Before prospective measurement registration, the frozen dataset must contain at least:

- 20 training episodes per class;
- 10 evaluation episodes per class;
- at least two distinct training source videos;
- at least two distinct evaluation source videos;
- no source-video overlap between splits.

All eligible frozen episodes are used. The experiment must not downsample or rebalance episodes after observing model results.

If this gate is not met, R1-E3 does not measure; the dataset must be expanded under a new pre-measurement artifact revision.

## Leakage and validity gates

Registered measurement fails closed if any of the following occurs:

- source-video overlap between train and evaluation;
- duplicate `episode_id`;
- unregistered label;
- unknown actor/target track reference;
- non-finite or wrong-shape normalized tensor;
- artifact digest mismatch;
- extraction/annotation/split manifest mismatch;
- reservoir parameter digest changes during an arm;
- B1 and B2 are not derived from the same trajectory;
- temporal pooling uses anything other than bins 16..19;
- any future observation enters a decision;
- registered arm count differs from `4 architectures x 5 seeds = 20`;
- exact runtime/head/provenance checks fail.

Invalid arms are never replaced with zero scores.

## Evidence lifecycle

The mandatory sequence is:

1. committed R1-E3 design;
2. committed implementation plan;
3. TDD implementation;
4. frozen source-video corpus manifest;
5. deterministic YOLO11 + BoT-SORT extraction;
6. raw-track artifact freeze;
7. annotation freeze and review;
8. deterministic 20-bin episode normalization;
9. artifact/split validation and dataset-size gate;
10. permanent CI validating code and frozen-artifact structure without registered measurement;
11. prospective manifest bound to exact science head, dataset artifact digests, runtime, and arm count;
12. explicit human approval for that exact manifest;
13. exactly one registered measurement;
14. no-refit interpretation;
15. raw evidence freeze;
16. removal of one-shot measurement tooling;
17. final exact-head CI.

Ordinary CI must never invoke the registered measurement.

## Explicit exclusions

R1-E3 does not include:

- detector training or fine-tuning;
- tracker tuning;
- selecting YOLO weights from R1-E3 model scores;
- topology search;
- temporal-window search;
- reservoir hyperparameter search;
- learned recurrent weights;
- monocular-depth claims;
- using bbox growth as a depth estimate;
- SLAM;
- ROS2;
- delayed reward credit;
- reinforcement learning;
- online adaptation;
- attention/RNN/GRU/LSTM/transformer decoders;
- post-result seed expansion, threshold changes, episode relabeling, split changes, or refit.

Any such change requires a new experiment/design review.

## Claims boundary

A positive R1-E3 result would support only this statement:

> On the frozen registered real-track corpus, a fixed causal temporal reservoir readout transfers better than the instantaneous reservoir readout under the preregistered comparison.

It would not prove semantic understanding, physical depth estimation, general action recognition, generalization to arbitrary videos, or superiority of one reservoir topology.
