# yolo-motion-perception

A **pure-Python** research baseline for track-centric image-motion perception from YOLO multi-object tracks.

The repository now contains two orthogonal temporal estimators:

```text
RGB video
   ↓
YOLO11 + BoT-SORT
   ↓
TrackObservation ───────────────→ V1 MotionPipeline
   │                              center velocity + log-area expansion
   │                              ↓
   │                              stationary/moving + stable/approaching/receding
   │
   └→ optional RTMW 133-point pose
        + optical flow
        + camera compensation
        + torso-motion subtraction
        ↓
      articulated leg residuals
        ↓
      V2 OpticalGaitPipeline
        ↓
      unknown / standing / walking / running
```

There is **no ROS2 dependency or integration**. Neither bbox scale nor optical/image motion is metric depth, physical velocity, or 3D motion.

## Why this structure?

YOLO tells us *what* was detected and BoT-SORT supplies the tracker ID used by both temporal paths. Motion and gait are properties of a sequence, so they are estimated from bounded histories rather than from a single frame or from Ultralytics internals.

V1 fits linear trends to:

- normalized bbox center `x(t)` and `y(t)`,
- `log(width(t) * height(t))`.

The log-area slope is used as **approach/recede evidence**. A sustained positive slope supports `approaching`; a sustained negative slope supports `receding`. Trend consistency and fit quality reduce sensitivity to one-frame bbox spikes.

V2 is deliberately independent from those labels. It uses full-body pose geometry and compensated optical flow to remove camera/whole-person translation before estimating scale-normalized left/right leg articulation. A person may therefore be, for example, `walking + stationary`, `walking + approaching`, or `walking + receding`.

## Scope

### V1 implemented here

- YOLO/BoT-SORT adapter
- normalized `TrackObservation`
- bounded per-track history
- normalized image-plane velocity
- log-area expansion rate
- temporal confidence/fit measures
- orthogonal lateral/radial state
- deterministic synthetic benchmark
- optional video/webcam CLI

### V2 optical gait implemented here

- immutable COCO-WholeBody-133 pose observations
- deterministic torso/thigh/calf/foot regions
- optional OpenCV dense optical-flow backend
- fixed-camera or affine camera-motion compensation
- explicit rejection of invalid affine compensation
- torso translation subtraction
- person-height scale normalization
- pose/flow agreement quality
- bilateral temporal gait evidence
- independent per-track gait histories
- fail-closed `unknown | standing | walking | running` classification
- optional lazy MMPose RTMW adapter
- deterministic gait benchmark covering rigid motion, approach/recede orthogonality, size gating, and camera compensation

### Explicit non-goals

- metric monocular depth
- physical 3D coordinates or velocity
- persistent identity after tracker-ID expiry
- general action recognition
- TTC/CPA in the current implementation
- robot control/path planning
- ROS2 or robotics middleware

Image-plane displacement, optical flow, and bbox expansion are evidence in image coordinates only. They must not be reported as metric distance, depth, or physical speed without a separate calibrated 3D estimator.

## Install

Core development install:

```bash
python -m pip install -e '.[dev]'
```

OpenCV optical-flow/camera-compensation runtime:

```bash
python -m pip install -e '.[dev,optical]'
```

Video runtime with Ultralytics/OpenCV:

```bash
python -m pip install -e '.[vision,dev]'
```

Optional RTMW/MMPose runtime:

```bash
python -m pip install -e '.[rtmw]'
```

Install both optical flow and RTMW when using the complete V2 runtime path:

```bash
python -m pip install -e '.[optical,rtmw]'
```

MMPose is imported lazily by `load_mmpose_rtmw()`. Importing the core package or `yolo_motion.rtmw_adapter` does not require MMPose. Model weights are not downloaded by tests or CI.

## Deterministic benchmarks

Run the original V1 motion benchmark:

```bash
python scripts/benchmark_synthetic.py
```

Expected V1 scenario classes include:

- stationary → `stationary / stable`
- lateral crossing → `moving / stable`
- monotonic bbox expansion → `stationary / approaching`
- monotonic bbox shrink → `stationary / receding`
- one-frame bbox scale spike → `stationary / stable`

Run the V2 optical-gait benchmark:

```bash
python scripts/benchmark_gait_synthetic.py
```

It emits JSON and fails non-zero if the expected contract is violated. Repository-owned deterministic scenarios include:

- `standing`
- `walking_in_place`
- `walking_transverse`
- `walking_approaching`
- `walking_receding`
- `running`
- `rigid_translation_control`
- `too_small_unknown`
- `camera_translation_compensated`

Approaching/receding labels in this benchmark come from the unchanged V1 `MotionPipeline`; gait classification does not infer radial motion from scale heuristics. The camera-translation case executes the compensation path rather than treating raw camera flow as gait.

## V2 `unknown` semantics

`LocomotionState.UNKNOWN` is a deliberate fail-closed result, not a miscellaneous class. It is used when the gait path lacks sufficient evidence, including insufficient history/quality, insufficient bilateral leg support, or upstream pose/flow support that cannot satisfy the configured gates.

A low-energy but otherwise well-supported bilateral sequence may classify as `standing`. Missing/poor evidence must not be promoted to walking or running.

The research defaults live in `configs/gait.yaml`; they are configuration values, not safety thresholds.

## Optional RTMW adapter

`RtmwPoseAdapter` consumes an already-tracked person crop. It does not create a second person detector. `load_mmpose_rtmw()` constructs `MMPoseInferencer` lazily with caller-supplied RTMW config/weights and `det_model="whole_image"`, then converts the 133 crop-relative keypoints immediately back to normalized full-frame coordinates tied to the original tracker ID.

No result or low-quality pose returns `None`. A malformed non-133-point result is rejected rather than silently mapped to another pose layout.

## Capture the first fixed-camera benchmark clips

Use the optional OpenCV runtime to record the four highest-value V1 scenes with consistent names and metadata:

```bash
python scripts/capture_scenario.py --scenario stationary
python scripts/capture_scenario.py --scenario lateral_crossing
python scripts/capture_scenario.py --scenario approaching
python scripts/capture_scenario.py --scenario pose_change
```

Each command performs a visible 3-second countdown, records 8 seconds from camera `0`, and writes:

```text
benchmarks/videos/<scenario>.mp4
benchmarks/videos/<scenario>.json
```

The JSON sidecar records the camera index, requested/actual duration, FPS, frame size, frame count, and UTC start time. Existing outputs are refused by default; use `--overwrite` deliberately when replacing a take. Override capture parameters when needed, for example:

```bash
python scripts/capture_scenario.py \
  --scenario approaching \
  --camera 1 \
  --duration 10 \
  --countdown 3
```

Video files remain ignored by git. Keep the sidecar metadata when comparing capture conditions across scenes. Follow `benchmarks/PROTOCOL.md` before recording.

## Run V1 on a video or camera

```bash
python -m yolo_motion.cli \
  --source path/to/video.mp4 \
  --model yolo11n.pt \
  --tracker botsort.yaml \
  --config configs/baseline.yaml \
  --show
```

For a webcam:

```bash
python -m yolo_motion.cli --source 0 --show
```

Write per-track motion states as JSONL:

```bash
python -m yolo_motion.cli \
  --source path/to/video.mp4 \
  --jsonl runs/motion.jsonl
```

The CLI uses `persist=True` only for consecutive frames from the same stream and converts Ultralytics results immediately into local data types.

## Evaluate a real-video V1 run

V1 evaluation is intentionally tied to a specific tracker run: first generate JSONL, then annotate the resulting `track_id` over labeled time intervals. This avoids adding a second object-matching algorithm to the benchmark.

Example annotation file (`benchmarks/example_annotations.yaml`):

```yaml
video: path/to/video.mp4
intervals:
  - track_id: 1
    start: 1.0
    end: 3.0
    lateral: stationary
    radial: approaching
  - track_id: 1
    start: 3.0
    end: 5.0
    lateral: moving
    radial: stable
```

Either dimension may be omitted when it is not part of the scene label. Score the run with:

```bash
python scripts/evaluate_jsonl.py \
  --predictions runs/motion.jsonl \
  --annotations benchmarks/example_annotations.yaml
```

The report contains per-dimension sample count, accuracy, confusion counts, and first-correct latency for each labeled interval. Predictions outside labeled intervals are ignored.

## Run the real-video V1 benchmark suite

`benchmarks/suite.yaml` defines eight fixed-camera V1 scenario slots. They are disabled by default so the repository does not pretend to ship video data. Put your clips under `benchmarks/videos/`, add matching interval labels under `benchmarks/annotations/`, and set `enabled: true` for the scenarios you want to measure.

Run inference and evaluation for every enabled scenario:

```bash
python scripts/run_benchmark_suite.py \
  --suite benchmarks/suite.yaml \
  --output-dir runs/benchmark
```

The suite runner reuses the existing video CLI and writes:

```text
runs/benchmark/
├── approaching/
│   ├── motion.jsonl
│   └── report.json
├── receding/
│   ├── motion.jsonl
│   └── report.json
├── summary.json
└── summary.md
```

After tuning annotations or reporting logic, recompute metrics without rerunning YOLO:

```bash
python scripts/run_benchmark_suite.py \
  --suite benchmarks/suite.yaml \
  --output-dir runs/benchmark \
  --evaluate-only
```

`summary.md` reports scenario-level lateral/radial accuracy and mean first-correct latency, plus sample-weighted overall accuracy. The eight initial slots are `stationary`, `lateral_crossing`, `approaching`, `receding`, `diagonal_approaching`, `approach_then_stop`, `pose_change`, and `partial_occlusion`.

Before running real videos, follow `benchmarks/PROTOCOL.md` and validate dataset readiness:

```bash
python scripts/check_benchmark_dataset.py --suite benchmarks/suite.yaml
```

The preflight exits with code `2` when no scenarios are enabled or when any enabled scenario is missing a non-empty video, valid annotations, or at least one labeled interval. Use `--format json` for machine-readable output. The repository includes empty annotation stubs under `benchmarks/annotations/`; fill them only after an inference run has produced the actual `track_id` values.

## Configuration

V1 `configs/baseline.yaml`:

```yaml
history_seconds: 1.0
min_samples: 5
min_duration: 0.3
stationary_speed_threshold: 0.02
radial_rate_threshold: 0.15
min_radial_confidence: 0.60
min_trend_consistency: 0.75
```

V2 gait parameters are in `configs/gait.yaml` and include history length, minimum history/quality/support, standing energy, periodicity/correlation, and cadence bounds.

All values are **research defaults**, not safety thresholds.

## Core V1 API

```python
from yolo_motion.motion import MotionConfig
from yolo_motion.pipeline import MotionPipeline
from yolo_motion.types import TrackObservation

pipeline = MotionPipeline(MotionConfig())

result = pipeline.update(
    TrackObservation(
        track_id=7,
        timestamp=0.4,
        class_id=0,
        confidence=0.94,
        cx=0.50,
        cy=0.52,
        width=0.18,
        height=0.34,
    )
)

if result is not None:
    print(result.state.lateral, result.state.radial)
    print(result.evidence.expansion_rate)
```

## V2 gait API

The V2 temporal classifier consumes `ArticulatedFlowEvidence`, which is produced only after pose geometry, camera compensation, and torso subtraction have been applied:

```python
from yolo_motion.gait import GaitConfig
from yolo_motion.gait_pipeline import OpticalGaitPipeline

pipeline = OpticalGaitPipeline(GaitConfig())
result = pipeline.update(articulated_flow_observation)

if result is not None:
    print(result.state)
    print(result.evidence.cadence_hz, result.evidence.quality)
```

The V1 and V2 pipelines intentionally keep separate histories and may be run side by side using the same tracker ID.

## Tests and CI

```bash
python -m pip install -e '.[dev,optical]'
pytest -q
ruff check .
python scripts/benchmark_synthetic.py
python scripts/benchmark_gait_synthetic.py
```

CI runs the Python 3.10/3.11/3.12 matrix with `.[dev,optical]`, the full pytest suite, Ruff, the V1 synthetic benchmark, and the V2 gait benchmark. It does **not** install MMPose/RTMW or download model weights.

The deterministic test suite does not require a GPU, camera, or network access.

## Real-video V2 acceptance gate

Synthetic/code completion is not sufficient to claim real-video gait validation. Before the V2 PR is ready to merge, record at least one real-video locomotion run on the exact branch head containing `standing`, `walking`, and `running` or `jogging`.

The evidence must retain `unknown` predictions and record at least:

- tracker model,
- RTMW config and weights identity,
- flow backend,
- camera mode,
- gait config,
- labeled `track_id` time intervals.

Until that evidence exists, the V2 PR remains Draft.

## Research roadmap

**Next — real-video V2 validation:** measure standing/walking/running behavior under tracker continuity, pose occlusion, subject scale, and camera-motion conditions without tuning away `unknown` cases.

**Future — depth evidence:** compare bbox-only radial estimates against an explicitly separate monocular-depth estimator; do not relabel image expansion as metric depth.

**Future — relative motion:** accept externally supplied ego-motion as numeric Python input before estimating any TTC/CPA-like quantities. This remains a library concern, not robotics middleware.

## Design documents

- `docs/superpowers/specs/2026-09-14-yolo-motion-perception-design.md`
- `docs/superpowers/plans/2026-09-14-v1-motion-baseline.md`
- `docs/superpowers/specs/2026-09-15-rtmw-optical-gait-v2-design.md`
- `docs/superpowers/plans/2026-09-15-rtmw-optical-gait-v2.md`

## License

Apache-2.0.
