# R1-E3M Unlabeled Real-Track Temporal-Memory Validation

Status: **committed design for issue #13**.

## 1. Purpose

R1-E3M is a new experiment that is scientifically independent from the existing
R1-E3 semantic-transfer preregistration in issue #8 / PR #9.

R1-E3 asks:

> Can a fixed temporal reservoir decoder improve human-defined
> `approach/touch/pick_up/pass_by` classification on real tracked video?

That question requires human semantic labels.

R1-E3M asks a narrower mechanism question:

> Does the frozen ESN/reservoir retain linearly recoverable information about
> earlier real YOLO11 + BoT-SORT observations beyond what is available in the
> current observation or the preserved final suffix?

R1-E3M therefore uses **no human behavior labels and no human event
boundaries**. Its regression targets are earlier observations from the same
frozen input sequence.

A positive R1-E3M result does not imply semantic behavior understanding. It
establishes only a real-input temporal-memory property.

## 2. Non-interference with semantic R1-E3

R1-E3M must not modify or reinterpret the existing R1-E3 registration.

The following remain true:

- issue #8 / PR #9 remain the semantic-transfer experiment;
- issue #11 remains the human semantic-review gate for issue #8;
- R1-E3M is not a substitute for issue #11;
- R1-E3M extraction/model outputs must not be used by #11 annotators;
- R1-E3M track overlays, model scores, sequence diagnostics, or reservoir results
  must not be added to the human review pack;
- an R1-E3M track artifact is **not automatically an accepted R1-E3 Stage-A
  artifact**.

This separation prevents the unlabeled mechanism experiment from influencing
the independent semantic ground truth.

## 3. Frozen source corpus

R1-E3M uses only the already-frozen MEVA Batch-1 source-video set from the
acquisition work supporting issues #10/#11.

Frozen source manifest SHA-256:

`1db8cc200ba9eaab2c14ed260b06ed582ed75d54f57c3036e1312cd2f184f4cf`

The source set contains:

- 11 videos;
- 7 training videos;
- 4 evaluation videos;
- 4 training guarded source-window components;
- 2 evaluation guarded source-window components;
- zero train/eval source-window-component overlap.

The source split is inherited exactly. R1-E3M may not move a source video,
re-encode it into the opposite split, or select a different subset after
observing reservoir results.

## 4. Frozen reservoir inheritance

R1-E3M reuses R1-E2/R1-E3 reservoir choices without search:

- Flat-256;
- Grouped-4x64;
- Hierarchical-2x128;
- Hierarchical-4x64;
- total neuron budget: 256;
- tanh activation;
- recurrent spectral radius: 0.9 per component;
- leak: 1.0;
- no recurrent bias;
- float64 execution;
- Ridge regularization: `1e-6`;
- seeds: `[7, 17, 29, 43, 61]`.

There are exactly 20 architecture/seed reservoir arms.

No architecture winner is selected.

## 5. External track extraction boundary

One R1-E3M extraction is frozen before any registered mechanism measurement.

The extraction runtime is outside the NumPy-only registered measurement and may
use YOLO11, Ultralytics, BoT-SORT, OpenCV/video decoding, Torch, and required
runtime dependencies.

The frozen provenance must record:

- exact source-manifest SHA-256;
- extraction code commit;
- YOLO implementation/package version;
- model name and weights SHA-256;
- detector confidence threshold;
- detector IoU/NMS threshold;
- class filter;
- BoT-SORT implementation/version;
- complete tracker config and config SHA-256;
- frame sampling policy;
- resize/letterbox policy;
- Python/runtime/package-lock identity;
- platform/hardware provenance;
- canonical raw-track artifact SHA-256.

No detector/tracker setting may be selected from R1-E3M reservoir results.

The raw track schema follows the existing R1-E3 handoff:

- `video_id`;
- `frame_index`;
- `timestamp_seconds`;
- `track_id`;
- `class_id`;
- detector confidence;
- absolute `x1/y1/x2/y2`;
- normalized `cx/cy/w/h`;
- source frame geometry.

Track IDs are identifiers only and never enter the reservoir feature vector.

## 6. Deterministic unlabeled window construction

### 6.1 Absolute source-time grid

Each source video is partitioned into **non-overlapping 4.0-second windows**
aligned to absolute source time:

`[0,4), [4,8), [8,12), ...`

A window is retained only if its full 4.0-second interval lies inside the source
video.

There is no event-triggered windowing and no human boundary.

### 6.2 Registered 20-bin clock

Each 4.0-second window contains exactly 20 registered sample times at 5 Hz.

For window start `s`, bin `i` in `0..19` is sampled at:

`s + (i + 0.5) * 0.2 seconds`

This gives centers:

`s+0.1, s+0.3, ..., s+3.9`

The 20-bin count matches the existing R1-E3 sequence length while using a fixed
physical-time clock rather than human event normalization.

### 6.3 Track interpolation

For each track and registered bin:

- use an exact frozen observation when present;
- otherwise use timestamp-linear interpolation only when the nearest bracketing
  observations are separated by at most 0.5 seconds;
- otherwise mark the track missing for that bin.

Only normalized geometry and confidence are interpolated. Track ID and class ID
are never interpolated.

### 6.4 Actor/target pair eligibility

For every fixed window, enumerate deterministic directed pairs
`(actor_track_id, target_track_id)`.

Requirements:

- actor and target are distinct track IDs;
- actor's modal detector class within the window is class 0 (person);
- modal-class ties are resolved by the numerically smaller class ID;
- target may be any detector class, including another person;
- actor is present in at least 16 of 20 registered bins;
- target is present in at least 16 of 20 registered bins.

Every eligible pair is retained.

There is no nearest-object selection, semantic interaction filter, behavior
filter, confidence-based reranking, or result-driven downsampling.

If the resulting artifact is too large for the registered implementation, the
experiment fails closed and requires a new design revision before measurement;
the implementation may not silently sample easier windows.

### 6.5 Sequence identity

Each sequence receives a stable ID derived from:

- source-video ID;
- absolute window start time;
- actor track ID;
- target track ID;
- frozen raw-track artifact root digest.

Sequence IDs and track IDs are audit metadata only.

## 7. Frozen 14-channel input

Each registered bin uses the same image-space channel definition as semantic
R1-E3:

Actor:

1. normalized `cx`
2. normalized `cy`
3. normalized `w`
4. normalized `h`
5. detector confidence
6. presence bit

Target:

7. normalized `cx`
8. normalized `cy`
9. normalized `w`
10. normalized `h`
11. detector confidence
12. presence bit

Pair:

13. normalized center distance
14. box IoU

Missing-track geometry/confidence is zero with presence bit zero. Pair channels
are zero if either track is missing.

No channel is called or interpreted as depth.

The complete sequence tensor shape is exactly `20 x 14`, finite float64.

## 8. Train/evaluation isolation

Training/evaluation assignment comes only from the frozen source-video split.

- no source video crosses splits;
- no sequence crosses splits;
- no source-window component crosses splits;
- no random frame/window split;
- all eligible frozen sequences are used.

The evaluation set is never used to select window construction, delays,
reservoir parameters, or transformation rules.

## 9. Registered representations

For each sequence define:

### C0 — current observation baseline

Raw bin 19 only:

shape `14`.

### S4 — preserved suffix baseline

Flatten raw bins 16..19:

shape `56`.

This is deliberately strong: it has direct access to the same final four raw
observations preserved in the history-destruction controls.

### R1 — reservoir final state

Reservoir state after bin 19:

shape `256`.

### R4 — reservoir final-four temporal mean

Arithmetic mean of reservoir states at bins 16..19:

shape `256`.

R1 is the primary reservoir representation. R4 is secondary and reuses the
existing R1-E3 temporal-mean convention without searching the window.

## 10. M1 — real-input delay reconstruction

### 10.1 Automatically generated targets

For delay `d`, the target is the original raw 14-channel observation at:

`target_bin = 19 - d`

Registered delays are exactly:

`[1, 2, 5, 10, 15]`

Therefore target bins are:

- delay 1 -> bin 18;
- delay 2 -> bin 17;
- delay 5 -> bin 14;
- delay 10 -> bin 9;
- delay 15 -> bin 4.

Delays 1 and 2 are sanity controls because S4 directly contains those target
bins.

Delay 5 is the shortest target outside S4.

**Delay 10 is the primary mechanism delay.**

Delay 15 is the long-history secondary diagnostic.

### 10.2 Probe law

For each representation and delay, fit a multi-output linear Ridge regressor on
training videos only.

The fit law must match the existing repository Ridge convention:

- float64;
- explicit bias term;
- feature coefficients penalized;
- bias unpenalized;
- regularization exactly `1e-6`;
- least-squares solution;
- no hyperparameter search.

Targets are 14-dimensional real observations rather than class one-hot vectors.

### 10.3 Evaluation metric

For each output channel on evaluation videos:

`R² = 1 - SSE / SST`

Channels with evaluation `SST <= 1e-12` are marked degenerate and excluded
from the macro average for that delay; their count is reported.

Primary score:

`macro_R2(d)`

= arithmetic mean over all non-degenerate registered channels.

Also report per-channel and grouped actor/target/pair values.

For each reservoir arm:

`delta_R2(d) = macro_R2_R1(d) - macro_R2_S4(d)`

C0 and R4 are reported controls.

## 11. M2 — preregistered history destruction

The fitted probes from M1 are reused unchanged. No transformed data is used for
refitting.

For every evaluation sequence create three transformations while preserving raw
bins 16..19 exactly.

### Reset

Process bins 0..15 normally, then reset the reservoir immediately before bin 16,
then process bins 16..19.

### Reverse-prefix

Feed bins 15..0, then unchanged bins 16..19.

### Hash-shuffled prefix

Permute bins 0..15 with a deterministic permutation seeded only from:

`SHA256("r1-e3m-prefix-shuffle-v1|" + sequence_id)`

Then append unchanged bins 16..19.

The suffix itself is byte-identical across normal/reset/reverse/shuffle
conditions.

Primary destruction quantity at delay 10:

`reset_drop_R2 = macro_R2_full_R1(10) - macro_R2_reset_R1(10)`

Reverse/shuffle drops are secondary order-sensitivity diagnostics.

## 12. M3 — suffix-matched history separability

This is a secondary representation diagnostic and cannot upgrade the primary
outcome classification.

Using training-set channel means/standard deviations, standardize raw
representations with standard-deviation floor `1e-12`.

For each evaluation sequence:

1. flatten bins 16..19 to its S4 vector;
2. search evaluation sequences from a **different source video**;
3. select the nearest suffix neighbor by Euclidean S4 distance;
4. break ties by lexical sequence ID;
5. deduplicate unordered pairs.

For each resulting pair report:

- suffix distance;
- prefix distance using flattened bins 0..15;
- R1 final-state distance;
- R4 distance.

Report Spearman correlation between prefix distance and reservoir-state distance
and the same correlation for suffix distance.

No threshold from this diagnostic is part of the primary registered outcome.

## 13. Registered outcome classes

The primary registered quantities are evaluated over the 20 architecture/seed
arms.

For each arm:

- `delta10 = macro_R2_R1(10) - macro_R2_S4(10)`;
- `reset10 = macro_R2_full_R1(10) - macro_R2_reset_R1(10)`.

### Outcome A — robust real-input temporal memory

All conditions hold:

- median `delta10 >= 0.05`;
- at least 16 of 20 arms have `delta10 > 0`;
- median `reset10 >= 0.05`;
- at least 16 of 20 arms have `reset10 > 0`.

### Outcome B — positive but modest/inconsistent memory

Outcome A does not hold, but:

- median `delta10 > 0`;
- median `reset10 > 0`.

### Outcome C — no registered mechanism evidence

Either:

- median `delta10 <= 0`; or
- median `reset10 <= 0`.

Delay 15, R4, reverse-prefix, hash-shuffle, and M3 are secondary diagnostics and
cannot upgrade B/C to A.

## 14. Artifact and prospective evidence lifecycle

Mandatory order:

1. this committed design;
2. written implementation plan;
3. TDD implementation on synthetic fixtures;
4. freeze exact extraction configuration;
5. one real-track extraction over the frozen Batch-1 videos;
6. freeze canonical raw tracks + window tensors + sequence manifest;
7. verify source/split/content digests;
8. permanent CI validates code and fixture logic only;
9. prepare prospective measurement manifest bound to exact science head,
   real-track artifact root, source-manifest SHA, sequence counts, architectures,
   seeds, delays, and runtime;
10. explicit approval of that prospective manifest;
11. exactly one registered R1-E3M mechanism measurement;
12. no-refit Outcome A/B/C interpretation;
13. freeze raw measurement evidence and digests.

Ordinary CI must not regenerate the registered real-track artifact or rerun the
registered mechanism measurement.

## 15. Fail-closed conditions

The experiment fails closed if:

- source-manifest SHA does not match the frozen Batch-1 manifest;
- source-video or source-window-component split overlap is detected;
- semantic annotation files are consumed;
- sequence construction differs from 4.0 seconds / 20 bins / 5 Hz;
- actor/target presence gate differs from 16/20;
- an unregistered delay is used;
- any window/seed/topology is selected from measured results;
- track IDs or class IDs enter the numeric reservoir feature vector;
- transformed controls alter bins 16..19;
- transformed controls are refit;
- artifact/provenance/runtime digest mismatches;
- arm count differs from 20;
- prospective exact-head/artifact binding fails.

Invalid arms are not replaced by zero scores.

## 16. Explicit exclusions

R1-E3M does not include:

- human behavior labels;
- human event boundaries;
- semantic behavior accuracy/F1;
- detector/tracker tuning from reservoir results;
- topology search;
- delay search;
- window-duration search;
- seed expansion;
- learned recurrent weights;
- online adaptation;
- attention/RNN/GRU/LSTM/transformer decoders;
- metric depth claims;
- SLAM/ROS2;
- reinforcement learning;
- delayed reward credit.

Any such change requires a new design review.

## 17. Claims boundary

Outcome A would support only:

> On the frozen real YOLO11 + BoT-SORT trajectory artifact, the fixed reservoir
> retains linearly recoverable information about observations approximately two
> seconds earlier beyond a strong final-four-observation baseline, and this
> advantage degrades when earlier recurrent history is removed while the final
> suffix is held fixed.

It would not establish:

- `approach/touch/pick_up/pass_by` recognition;
- semantic understanding;
- general action recognition;
- physical depth or velocity;
- generalization beyond the frozen real-track corpus;
- superiority of any one reservoir architecture.
