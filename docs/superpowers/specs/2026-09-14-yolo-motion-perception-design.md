# YOLO Motion Perception Design

**Date:** 2026-09-14

## 1. Goal

Build a pure-Python research repository that uses YOLO detections/tracks to estimate whether each tracked object is stationary, moving laterally, approaching, or receding over time. The first milestone is a fixed-camera baseline; later milestones may add monocular depth and TTC/CPA without changing the core track-history interfaces.

## 2. Hard constraints

- Pure Python project.
- No ROS2, ROS messages, robot middleware, DDS, or platform-specific adapters.
- YOLO/BoT-SORT is the perception source; motion semantics are computed outside Ultralytics internals.
- Core motion estimation must be testable without loading a neural-network model.
- V1 assumes a fixed camera.
- V1 must not report metric distance from bbox scale.
- Bbox growth is evidence of approach, not proof of physical depth change.
- Thresholds must be explicit configuration, not hidden constants.

## 3. Scope

### V1

Input:
- RGB video/webcam frames through Ultralytics YOLO tracking, or
- synthetic/pre-recorded `TrackObservation` sequences for tests and benchmarks.

Output per track:
- normalized center position,
- image-plane velocity,
- normalized speed,
- log-area expansion rate,
- trend consistency,
- fit quality,
- lateral state (`stationary`, `moving`),
- radial state (`unknown`, `stable`, `approaching`, `receding`),
- approach/recede confidence.

### Explicit non-goals for V1

- metric depth,
- 3D world coordinates,
- camera ego-motion estimation,
- robot control,
- path planning,
- SLAM,
- ROS2 integration,
- collision avoidance commands.

## 4. Architecture

```text
RGB frame
   |
   v
Ultralytics YOLO + BoT-SORT
   |
   v
TrackObservation
   |
   v
TrackHistory (per track_id)
   |
   v
MotionEstimator
  / \
 /   \
center slope   log(area) slope
 |             |
 v             v
lateral evidence   radial evidence
       \       /
        \     /
         v   v
      MotionEvidence
           |
           v
      StateClassifier
           |
           v
       MotionState
           |
           v
 JSON / overlay / benchmark
```

The Ultralytics adapter is an edge module. `TrackHistory`, `MotionEstimator`, and `StateClassifier` operate only on repository-defined types and NumPy arrays.

## 5. Data model

### `TrackObservation`

Fields:
- `track_id: int`
- `timestamp: float`
- `class_id: int`
- `confidence: float`
- `cx: float` normalized to `[0, 1]`
- `cy: float` normalized to `[0, 1]`
- `width: float` normalized to `[0, 1]`
- `height: float` normalized to `[0, 1]`

Derived properties:
- `area = width * height`
- `log_area = log(area)`

Invalid dimensions, non-finite values, and non-monotonic timestamps are rejected at the history boundary.

### `MotionEvidence`

Fields:
- `vx`, `vy`: normalized center velocity per second
- `speed`: `hypot(vx, vy)`
- `expansion_rate`: slope of `log(area)` per second
- `trend_consistency`: fraction of consecutive area deltas agreeing with the fitted trend sign
- `fit_quality`: bounded quality from log-area linear fit
- `sample_count`
- `duration`
- `approach_confidence`
- `recede_confidence`

### `MotionState`

Orthogonal state, not one overloaded enum:
- `lateral`: `stationary | moving`
- `radial`: `unknown | stable | approaching | receding`

This allows a person to be both laterally moving and approaching.

## 6. Motion estimation

For observations in a configurable trailing time window:

```text
x(t) = normalized bbox center x
y(t) = normalized bbox center y
s(t) = log(width * height)
```

Fit independent first-order least-squares lines:

```text
x(t) ~= ax * t + bx
y(t) ~= ay * t + by
s(t) ~= as * t + bs
```

Then:

```text
vx = ax
vy = ay
speed = sqrt(vx^2 + vy^2)
expansion_rate = as
```

Using log area makes expansion approximately scale-relative rather than dependent on absolute bbox pixels.

## 7. Confidence and false-positive rejection

Approach/recede classification must not use a single frame pair.

Radial confidence combines:
- sample sufficiency,
- temporal duration,
- monotonic trend consistency,
- linear fit quality,
- expansion magnitude relative to configured threshold.

A temporary pose-induced bbox spike should therefore have poor consistency and/or fit quality and stay `stable` or `unknown` rather than become a high-confidence approach event.

V1 does not claim this rejects every articulation/pose change; that limitation must remain explicit in benchmark reports.

## 8. Baseline configuration

`configs/baseline.yaml` exposes:
- `history_seconds`
- `min_samples`
- `stationary_speed_threshold`
- `radial_rate_threshold`
- `min_radial_confidence`
- `min_trend_consistency`

Defaults are research starting points, not safety thresholds.

## 9. Ultralytics boundary

The runtime adapter uses the public tracking API:

```python
model.track(frame, persist=True, tracker="botsort.yaml")
```

`persist=True` is used only for consecutive frames from the same stream. Track IDs and XYWH boxes are converted immediately into `TrackObservation`; the core package does not retain Ultralytics `Results` objects.

The model weights are configurable. The repository documentation uses `yolo11n.pt` as the baseline detector name but does not download weights during tests or CI.

## 10. CLI

`python -m yolo_motion.cli --source <video-or-camera> --model yolo11n.pt`

Options include:
- model path/name,
- source,
- tracker yaml,
- confidence threshold,
- baseline config,
- optional JSONL output,
- optional visualization window.

The CLI is optional-runtime functionality and imports Ultralytics/OpenCV lazily so core unit tests do not require them.

## 11. Testing

Deterministic unit tests use synthetic track histories:
- stationary bbox,
- lateral motion with constant scale,
- monotonic approach,
- monotonic recede,
- approach then stop,
- one-frame bbox scale spike,
- invalid/non-monotonic observations,
- insufficient history.

No unit test requires YOLO weights, video downloads, GPU, or network access.

## 12. Benchmark

A synthetic benchmark runner produces JSON summaries for known trajectories and reports:
- predicted lateral/radial state,
- expansion rate,
- confidence,
- latency in samples to stable classification.

Real-video evaluation is a follow-up dataset task, not required to prove the core temporal estimator works.

## 13. V2 extension: depth evidence

V2 may add a `DepthEvidenceSource` that attaches per-track depth measurements and compares:
- bbox-only radial state,
- depth-only radial state,
- fused state,
- `TTC_bbox` vs `TTC_depth`.

This extension must not require changing `TrackObservation` consumers that do not use depth.

## 14. V3 extension: relative motion

A later pure-Python module may accept externally supplied ego-motion/odometry and estimate relative motion, TTC, and CPA. That remains a numeric/library interface only; no ROS2 integration is planned.
