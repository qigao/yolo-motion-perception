# yolo-motion-perception

A **pure-Python** research baseline for estimating object motion and approach/recede state from YOLO multi-object tracks.

The project deliberately separates **object perception** from **temporal motion semantics**:

```text
RGB video
   ↓
YOLO11 + BoT-SORT
   ↓
TrackObservation
   ↓
per-track history
   ↓
center velocity + log-area expansion
   ↓
stationary / moving + stable / approaching / receding
```

There is **no ROS2 dependency or integration**. The V1 baseline assumes a fixed camera and does not claim metric depth from bbox scale.

## Why this structure?

YOLO tells us *what* was detected and BoT-SORT gives a persistent track ID. Motion state is a temporal property, so it is estimated from a window of repository-owned observations rather than from a single frame or Ultralytics internals.

For each track, V1 fits linear trends to:

- normalized bbox center `x(t)` and `y(t)`,
- `log(width(t) * height(t))`.

The log-area slope is used as **approach/recede evidence**. A sustained positive slope supports `approaching`; a sustained negative slope supports `receding`. Trend consistency and fit quality reduce sensitivity to one-frame bbox spikes.

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

### Not in V1

- metric monocular depth
- 3D coordinates
- camera ego-motion
- TTC/CPA
- robot control/path planning
- ROS2 or robotics middleware

Depth/TTC can be added later as independent evidence without changing the core temporal estimator.

## Install

Core development install:

```bash
python -m pip install -e '.[dev]'
```

Video runtime with Ultralytics/OpenCV:

```bash
python -m pip install -e '.[vision,dev]'
```

Model weights are not downloaded by tests or CI.

## Run the deterministic benchmark

```bash
python scripts/benchmark_synthetic.py
```

Expected scenario classes include:

- stationary → `stationary / stable`
- lateral crossing → `moving / stable`
- monotonic bbox expansion → `stationary / approaching`
- monotonic bbox shrink → `stationary / receding`
- one-frame bbox scale spike → `stationary / stable`

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

## Run on a video or camera

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

## Evaluate a real-video run

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

## Run a real-video benchmark suite

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

`configs/baseline.yaml`:

```yaml
history_seconds: 1.0
min_samples: 5
min_duration: 0.3
stationary_speed_threshold: 0.02
radial_rate_threshold: 0.15
min_radial_confidence: 0.60
min_trend_consistency: 0.75
```

These values are **research defaults**, not safety thresholds.

## Core API

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

## Tests

```bash
PYTHONPATH=src pytest -q
ruff check .
```

The core test suite uses only deterministic synthetic tracks. It does not require a GPU, YOLO weights, camera, or network access.

## Research roadmap

**V2 — depth evidence:** compare bbox-only radial estimates against monocular depth change and fuse confidence; add `TTC_bbox` and `TTC_depth` as explicitly different estimators.

**V3 — relative motion:** accept externally supplied ego-motion as numeric Python input, estimate object-relative motion, TTC, and closest-point-of-approach. This remains a library concern, not a robotics-middleware integration.

## Design documents

- `docs/superpowers/specs/2026-09-14-yolo-motion-perception-design.md`
- `docs/superpowers/plans/2026-09-14-v1-motion-baseline.md`

## License

Apache-2.0.
