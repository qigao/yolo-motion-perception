# V1 Motion Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a pure-Python fixed-camera YOLO track temporal baseline that classifies lateral motion and approach/recede state from normalized bbox history.

**Architecture:** Ultralytics/BoT-SORT is isolated in an adapter that emits repository-owned `TrackObservation` values. Core history, regression, evidence, and state classification are deterministic NumPy/Python code and can be fully tested without model weights or a GPU.

**Tech Stack:** Python >=3.10, NumPy, PyYAML; optional runtime extras: Ultralytics, OpenCV; pytest and Ruff for development.

**Spec:** `docs/superpowers/specs/2026-09-14-yolo-motion-perception-design.md`

## Global Constraints

- Pure Python project.
- No ROS2, ROS messages, robot middleware, DDS, or platform-specific adapters.
- V1 assumes a fixed camera.
- Core unit tests must not load a YOLO model or download weights.
- Bbox expansion is approach evidence, never metric depth.
- Runtime Ultralytics/OpenCV imports are lazy/optional.

---

### Task 1: Core observation and history contracts

**Files:**
- Create: `src/yolo_motion/types.py`
- Create: `src/yolo_motion/history.py`
- Create: `tests/test_types.py`
- Create: `tests/test_history.py`

**Interfaces:**
- Produces `TrackObservation`, `TrackHistory`, `LateralState`, `RadialState`, `MotionEvidence`, `MotionState`.

- [ ] **Step 1: Write failing tests** proving normalized bbox validation, derived area/log-area, timestamp monotonicity, and time-window pruning.
- [ ] **Step 2: Run** `PYTHONPATH=src pytest tests/test_types.py tests/test_history.py -v` and verify failures are due to missing implementation.
- [ ] **Step 3: Implement minimal dataclasses/enums and `TrackHistory.add()` / `observations()` behavior.**
- [ ] **Step 4: Re-run** the same pytest command and require all tests to pass.
- [ ] **Step 5: Commit** `feat: add track observation history contracts`.

### Task 2: Deterministic motion estimator

**Files:**
- Create: `src/yolo_motion/motion.py`
- Create: `tests/test_motion.py`

**Interfaces:**
- Consumes `Sequence[TrackObservation]`.
- Produces `MotionEvidence estimate_motion(observations, config)`.

- [ ] **Step 1: Write failing tests** for stationary, lateral movement, monotonic approach, monotonic recede, and insufficient history.
- [ ] **Step 2: Run** `PYTHONPATH=src pytest tests/test_motion.py -v` and verify RED.
- [ ] **Step 3: Implement least-squares slopes over normalized center and `log(area)`, trend consistency, fit quality, and confidence.**
- [ ] **Step 4: Re-run tests and require GREEN.**
- [ ] **Step 5: Commit** `feat: estimate motion from track history`.

### Task 3: State classification and transient scale rejection

**Files:**
- Create: `src/yolo_motion/state.py`
- Extend: `tests/test_motion.py`
- Create: `tests/test_state.py`

**Interfaces:**
- Consumes `MotionEvidence` and `MotionConfig`.
- Produces `MotionState classify_motion(evidence, config)`.

- [ ] **Step 1: Write failing tests** showing an isolated bbox area spike does not become high-confidence `APPROACHING`, while sustained monotonic growth does.
- [ ] **Step 2: Run** `PYTHONPATH=src pytest tests/test_state.py tests/test_motion.py -v` and verify RED.
- [ ] **Step 3: Implement orthogonal lateral/radial classification using configured speed/radial/confidence/consistency thresholds.**
- [ ] **Step 4: Re-run tests and require GREEN.**
- [ ] **Step 5: Commit** `feat: classify lateral and radial motion states`.

### Task 4: Per-track pipeline

**Files:**
- Create: `src/yolo_motion/pipeline.py`
- Create: `tests/test_pipeline.py`

**Interfaces:**
- `MotionPipeline.update(observation) -> TrackMotionResult | None`
- Maintains one `TrackHistory` per `track_id` and drops stale histories by explicit API.

- [ ] **Step 1: Write failing tests** for independent histories of two track IDs and correct state output after sufficient samples.
- [ ] **Step 2: Run** `PYTHONPATH=src pytest tests/test_pipeline.py -v` and verify RED.
- [ ] **Step 3: Implement the minimal pipeline orchestration.**
- [ ] **Step 4: Re-run tests and require GREEN.**
- [ ] **Step 5: Commit** `feat: add per-track motion pipeline`.

### Task 5: Optional Ultralytics runtime adapter and CLI

**Files:**
- Create: `src/yolo_motion/ultralytics_adapter.py`
- Create: `src/yolo_motion/cli.py`
- Create: `tests/test_ultralytics_adapter.py`
- Create: `configs/baseline.yaml`

**Interfaces:**
- `observations_from_result(result, timestamp, frame_shape) -> list[TrackObservation]`
- CLI calls `YOLO.track(frame, persist=True, tracker="botsort.yaml")` for consecutive frames.

- [ ] **Step 1: Write failing adapter tests** using small fake result/box objects; do not import Ultralytics in tests.
- [ ] **Step 2: Run** `PYTHONPATH=src pytest tests/test_ultralytics_adapter.py -v` and verify RED.
- [ ] **Step 3: Implement duck-typed result conversion and lazy imports in CLI.**
- [ ] **Step 4: Re-run adapter and full core tests.**
- [ ] **Step 5: Commit** `feat: add optional YOLO tracking runtime adapter`.

### Task 6: Synthetic benchmark, packaging, docs, and CI

**Files:**
- Create: `scripts/benchmark_synthetic.py`
- Create: `pyproject.toml`
- Create: `.github/workflows/ci.yml`
- Create: `.gitignore`
- Update: `README.md`

**Interfaces:**
- `python scripts/benchmark_synthetic.py` prints JSON and exits non-zero if expected baseline states are not recovered.

- [ ] **Step 1: Add a benchmark smoke test** in `tests/test_benchmark.py` around deterministic synthetic scenarios.
- [ ] **Step 2: Run it RED before benchmark implementation.**
- [ ] **Step 3: Implement benchmark, package metadata, install extras, and CI for Python 3.10/3.11/3.12.**
- [ ] **Step 4: Run** `PYTHONPATH=src pytest -q`, `ruff check .`, and `python scripts/benchmark_synthetic.py`; all must exit 0.
- [ ] **Step 5: Commit** `chore: add benchmark packaging docs and ci`.

## Final verification

- `PYTHONPATH=src pytest -q`
- `ruff check .`
- `python scripts/benchmark_synthetic.py`
- `git status --short`
- Scan repository for forbidden dependency/scope terms: `grep -RniE 'rclpy|ament|ros2|sensor_msgs|geometry_msgs' src tests pyproject.toml || true`; expected no matches.
