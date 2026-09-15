# RTMW-Guided Optical Gait V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a track-centric optical gait path that uses RTMW body geometry plus optical flow to distinguish rigid person motion from articulated leg motion and classify `unknown | standing | walking | running` without changing the existing bbox-only lateral/radial estimator.

**Architecture:** V1 remains intact: YOLO/BoT-SORT emits `TrackObservation`, and `MotionPipeline` owns lateral/radial motion. V2 adds repository-owned pose/flow contracts, deterministic body-region geometry, camera compensation, torso-translation removal, scale-normalized limb residuals, temporal gait estimation, and a separate `OpticalGaitPipeline` keyed by the same tracker ID.

**Tech Stack:** Python >=3.10, NumPy, PyYAML, pytest, Ruff; optional OpenCV optical-flow runtime; optional MMPose RTMW runtime. Default tests require no model weights, GPU, camera, or network.

**Spec:** `docs/superpowers/specs/2026-09-15-rtmw-optical-gait-v2-design.md`

## Global Constraints

- Preserve current `TrackObservation`, `MotionPipeline`, `MotionEvidence`, `MotionState`, `LateralState`, and `RadialState` semantics.
- Track identity remains a tracker responsibility; V2 does not perform ReID after track expiry.
- Scale change is not identity evidence and is not a gait label.
- Camera motion and whole-person translation must be removed before articulated gait estimation.
- Rigid bbox/person translation alone must never be sufficient evidence for walking.
- Low pose/flow/camera/temporal quality fails closed to `LocomotionState.UNKNOWN`; no silent raw-flow or one-leg fallback.
- Core tests require no downloaded YOLO/RTMW weights, GPU, camera, or network.
- V2 locomotion is orthogonal to V1 lateral/radial state.
- Thresholds are explicit configuration and are research defaults, not safety thresholds.
- No SNN, FlyWire, graph-recurrent model, metric depth, ROS2, or action-classification neural network is introduced.

---

## File Structure

```text
src/yolo_motion/
    pose_types.py         # pose layouts and immutable pose observations
    pose_regions.py       # deterministic torso/leg region geometry
    flow_types.py         # flow/camera/articulated evidence records
    flow_backend.py       # FlowBackend protocol + OpenCV Farneback backend
    camera_motion.py      # identity/affine camera model and flow compensation
    articulated_flow.py   # torso translation, regional residual flow, scale normalization
    gait.py               # GaitConfig + temporal gait evidence estimation
    gait_state.py         # LocomotionState classification
    gait_pipeline.py      # bounded per-track gait histories
    rtmw_adapter.py       # optional lazy MMPose RTMW adapter

tests/
    test_pose_types.py
    test_pose_regions.py
    test_flow_types.py
    test_flow_backend.py
    test_camera_motion.py
    test_articulated_flow.py
    test_gait.py
    test_gait_state.py
    test_gait_pipeline.py
    test_rtmw_adapter.py
    test_v2_regression.py

configs/
    gait.yaml

scripts/
    benchmark_gait_synthetic.py
```

The existing V1 modules remain unchanged unless a final integration step needs additive exports or documentation.

---

### Task 1: Freeze pose, flow, articulated-motion, and gait contracts

**Files:**
- Create: `src/yolo_motion/pose_types.py`
- Create: `src/yolo_motion/flow_types.py`
- Create: `tests/test_pose_types.py`
- Create: `tests/test_flow_types.py`

**Interfaces:**

```python
class PoseLayout(str, Enum):
    COCO_WHOLEBODY_133 = "coco_wholebody_133"

@dataclass(frozen=True)
class PoseObservation:
    track_id: int
    timestamp: float
    frame_width: int
    frame_height: int
    xy: np.ndarray          # shape (K, 2), normalized full-frame coordinates
    confidence: np.ndarray  # shape (K,)
    layout: PoseLayout

@dataclass(frozen=True)
class FlowObservation:
    start_timestamp: float
    end_timestamp: float
    dx: np.ndarray
    dy: np.ndarray
    valid: np.ndarray | None
    backend: str

@dataclass(frozen=True)
class CameraMotionEstimate:
    model: str              # "identity" | "affine"
    matrix: np.ndarray      # 2x3
    matched_count: int
    inlier_count: int
    inlier_ratio: float
    residual_error: float
    quality: float
    valid: bool

@dataclass(frozen=True)
class RegionFlowEvidence:
    dx: float
    dy: float
    energy: float
    valid_fraction: float

@dataclass(frozen=True)
class ArticulatedFlowEvidence:
    track_id: int
    start_timestamp: float
    end_timestamp: float
    torso_dx: float
    torso_dy: float
    normalized_torso_dx: float
    normalized_torso_dy: float
    region_flow: dict[str, RegionFlowEvidence]
    pose_flow_agreement: float
    person_height_px: float
    quality: float
```

Validation must reject non-finite values, mismatched array shapes, non-positive frame dimensions, confidence outside `[0, 1]`, `end_timestamp <= start_timestamp`, flow fields with unequal shapes, and non-positive `person_height_px`.

- [ ] **Step 1: Add RED contract tests.** Test immutable construction, shape/finite validation, timestamp ordering, and one valid COCO-WholeBody-133 observation.
- [ ] **Step 2: Run** `PYTHONPATH=src pytest tests/test_pose_types.py tests/test_flow_types.py -v`. Expected: import/module failures only.
- [ ] **Step 3: Implement the exact records and validation above.** Copy arrays defensively and mark stored arrays read-only so adapter-side mutation cannot alter history.
- [ ] **Step 4: Re-run the focused tests and require GREEN.**
- [ ] **Step 5: Run V1 regression:** `PYTHONPATH=src pytest tests/test_types.py tests/test_motion.py tests/test_pipeline.py tests/test_state.py -q`.
- [ ] **Step 6: Commit:** `feat: add optical gait observation contracts`.

### Task 2: Build deterministic RTMW/COCO-WholeBody locomotion regions

**Files:**
- Create: `src/yolo_motion/pose_regions.py`
- Create: `tests/test_pose_regions.py`

**Interfaces:**

```python
class BodyPart(str, Enum):
    TORSO = "torso"
    LEFT_THIGH = "left_thigh"
    RIGHT_THIGH = "right_thigh"
    LEFT_CALF = "left_calf"
    RIGHT_CALF = "right_calf"
    LEFT_FOOT = "left_foot"
    RIGHT_FOOT = "right_foot"

@dataclass(frozen=True)
class BodyRegion:
    part: BodyPart
    segments: tuple[tuple[np.ndarray, np.ndarray], ...]
    radius_px: float
    confidence: float

@dataclass(frozen=True)
class PoseRegionConfig:
    min_keypoint_confidence: float = 0.35
    capsule_width_ratio: float = 0.06


def build_body_regions(
    pose: PoseObservation,
    person_height_px: float,
    config: PoseRegionConfig,
) -> dict[BodyPart, BodyRegion]: ...


def rasterize_region(region: BodyRegion, frame_shape: tuple[int, int]) -> np.ndarray: ...
```

Use COCO WholeBody indices: shoulders `5,6`, hips `11,12`, knees `13,14`, ankles `15,16`, left foot `17,18,19`, right foot `20,21,22`. Region radius is `max(1.0, capsule_width_ratio * person_height_px)`.

- [ ] **Step 1: Write RED tests** for all seven region names, confidence rejection, scale-relative capsule width, left/right separation, and deterministic raster masks.
- [ ] **Step 2: Run** `PYTHONPATH=src pytest tests/test_pose_regions.py -v` and verify RED.
- [ ] **Step 3: Implement COCO-WholeBody mapping and NumPy-only capsule rasterization** using point-to-segment distance; do not require OpenCV in geometry code.
- [ ] **Step 4: Re-run and require GREEN.**
- [ ] **Step 5: Add a bilateral-support assertion:** if the right ankle/knee confidence is below threshold, right calf/foot are absent rather than fabricated.
- [ ] **Step 6: Commit:** `feat: derive locomotion regions from whole-body pose`.

### Task 3: Add reference dense flow and explicit camera-motion compensation

**Files:**
- Create: `src/yolo_motion/flow_backend.py`
- Create: `src/yolo_motion/camera_motion.py`
- Create: `tests/test_flow_backend.py`
- Create: `tests/test_camera_motion.py`
- Modify: `pyproject.toml`

**Interfaces:**

```python
class FlowBackend(Protocol):
    def compute(
        self,
        previous_frame: np.ndarray,
        current_frame: np.ndarray,
        start_timestamp: float,
        end_timestamp: float,
    ) -> FlowObservation: ...

class OpenCvFarnebackBackend:
    def compute(...) -> FlowObservation: ...

@dataclass(frozen=True)
class CameraMotionConfig:
    mode: str = "fixed"          # "fixed" | "affine"
    min_background_features: int = 20
    min_inlier_ratio: float = 0.60
    max_residual_error: float = 2.0


def estimate_camera_motion(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    person_masks: Sequence[np.ndarray],
    config: CameraMotionConfig,
) -> CameraMotionEstimate: ...


def compensate_flow(
    flow: FlowObservation,
    camera: CameraMotionEstimate,
) -> FlowObservation: ...
```

`fixed` mode always returns a valid identity transform. `affine` mode masks people, uses background features + pyramidal LK tracking + robust partial-affine estimation, and returns `valid=False` when support/quality gates fail. `compensate_flow` must raise `ValueError` for an invalid estimate; it must never silently pass raw flow through.

Add an `optical` extra containing `opencv-python>=4.9`; CI installs it explicitly later, while the base package remains NumPy/PyYAML only.

- [ ] **Step 1: Write RED tests** with synthetic translated images/flow fields: fixed identity, affine translation recovery, person-mask exclusion, invalid-low-feature estimate, and invalid-compensation rejection.
- [ ] **Step 2: Run** `PYTHONPATH=src pytest tests/test_flow_backend.py tests/test_camera_motion.py -v` and verify RED.
- [ ] **Step 3: Implement Farneback forward flow and an optional backward-consistency validity map.** Store backend identity as `opencv-farneback`.
- [ ] **Step 4: Implement identity/affine compensation.** Convert the 2x3 affine transform into a dense camera-flow field and subtract it from raw `dx/dy`.
- [ ] **Step 5: Re-run focused tests with** `python -m pip install -e '.[dev,optical]'` and require GREEN.
- [ ] **Step 6: Commit:** `feat: add optical flow and camera compensation`.

### Task 4: Decompose torso translation from articulated limb flow

**Files:**
- Create: `src/yolo_motion/articulated_flow.py`
- Create: `tests/test_articulated_flow.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class ArticulatedFlowConfig:
    min_person_height_px: float = 40.0
    min_region_valid_fraction: float = 0.50
    min_torso_valid_fraction: float = 0.60
    max_pose_flow_error_norm: float = 0.08


def estimate_articulated_flow(
    track: TrackObservation,
    previous_pose: PoseObservation,
    current_pose: PoseObservation,
    regions: Mapping[BodyPart, BodyRegion],
    compensated_flow: FlowObservation,
    config: ArticulatedFlowConfig,
) -> ArticulatedFlowEvidence | None: ...
```

Algorithm order is fixed:

1. Compute `person_height_px = track.height * current_pose.frame_height`; return `None` below the configured minimum.
2. Rasterize torso and region masks.
3. Compute robust median compensated torso flow over valid torso pixels.
4. Subtract torso flow from each region's vectors.
5. Normalize residual displacement by `person_height_px`.
6. Compare pose joint displacement against local compensated flow; convert disagreement into `pose_flow_agreement` and aggregate `quality`.
7. Return `None` if torso or bilateral support does not meet validity gates.

- [ ] **Step 1: Write RED fixtures** for rigid translation, walking-in-place, translation+alternating legs, identical gait at 50px and 250px person heights, pose-jump disagreement, and too-small targets.
- [ ] **Step 2: Run** `PYTHONPATH=src pytest tests/test_articulated_flow.py -v` and verify RED.
- [ ] **Step 3: Implement robust regional sampling and torso subtraction.** Pure rigid translation must produce near-zero normalized limb residual energy.
- [ ] **Step 4: Implement scale normalization and pose/flow agreement.** Equivalent synthetic gait at 50px/250px must agree within an explicit test tolerance of `abs(delta_energy) <= 0.01`.
- [ ] **Step 5: Re-run focused tests and require GREEN.**
- [ ] **Step 6: Commit:** `feat: decompose articulated body motion`.

### Task 5: Estimate bilateral temporal gait evidence and classify locomotion

**Files:**
- Create: `src/yolo_motion/gait.py`
- Create: `src/yolo_motion/gait_state.py`
- Create: `tests/test_gait.py`
- Create: `tests/test_gait_state.py`
- Create: `configs/gait.yaml`

**Interfaces:**

```python
class LocomotionState(str, Enum):
    UNKNOWN = "unknown"
    STANDING = "standing"
    WALKING = "walking"
    RUNNING = "running"

@dataclass(frozen=True)
class GaitConfig:
    history_seconds: float = 2.0
    min_samples: int = 12
    min_duration: float = 0.8
    min_quality: float = 0.55
    standing_energy_threshold: float = 0.008
    min_periodicity: float = 0.55
    min_bilateral_correlation: float = 0.45
    walking_cadence_min_hz: float = 0.7
    walking_cadence_max_hz: float = 2.4
    running_cadence_min_hz: float = 2.2

@dataclass(frozen=True)
class GaitEvidence:
    sample_count: int
    duration: float
    left_energy: float
    right_energy: float
    periodicity: float
    bilateral_correlation: float
    phase_lag_seconds: float
    cadence_hz: float
    articulated_amplitude: float
    temporal_consistency: float
    quality: float


def estimate_gait(
    observations: Sequence[ArticulatedFlowEvidence],
    config: GaitConfig,
) -> GaitEvidence: ...


def classify_gait(evidence: GaitEvidence, config: GaitConfig) -> LocomotionState: ...
```

Build left/right signals from thigh/calf/foot residual vectors. Derive a dominant 2D motion axis from the window, project each leg signal onto it, then use normalized autocorrelation for periodicity and left/right cross-correlation for phase. Search gait periods only inside the broad configured cadence range so zero-lag drift does not become a gait period.

- [ ] **Step 1: Write RED deterministic signal tests** at 20Hz sample rate for: standing noise, 1.5Hz antiphase walking, 3.0Hz running, translation-only zero residual, low-quality history, irregular/non-periodic leg motion, and one-leg missing support.
- [ ] **Step 2: Run** `PYTHONPATH=src pytest tests/test_gait.py tests/test_gait_state.py -v` and verify RED.
- [ ] **Step 3: Implement evidence extraction** with NumPy autocorrelation/cross-correlation and explicit bounded lag search.
- [ ] **Step 4: Implement fail-closed classification.** Any insufficient history/quality/bilateral support is `UNKNOWN`; low valid articulated energy is `STANDING`; periodic bilateral signals choose `WALKING`/`RUNNING` from cadence/amplitude rules.
- [ ] **Step 5: Re-run and require GREEN.** Assert 1.5Hz fixture cadence is within `±0.15Hz`; 3.0Hz within `±0.20Hz`.
- [ ] **Step 6: Commit:** `feat: classify locomotion from bilateral gait signals`.

### Task 6: Add per-track gait history and prove V1/V2 orthogonality

**Files:**
- Create: `src/yolo_motion/gait_pipeline.py`
- Create: `tests/test_gait_pipeline.py`
- Create: `tests/test_v2_regression.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class OpticalGaitResult:
    track_id: int
    evidence: GaitEvidence
    state: LocomotionState

class OpticalGaitPipeline:
    def __init__(self, config: GaitConfig): ...
    def update(self, observation: ArticulatedFlowEvidence) -> OpticalGaitResult | None: ...
    def drop_stale(self, active_track_ids: set[int]) -> set[int]: ...
    def track_ids(self) -> set[int]: ...
```

The pipeline owns one bounded articulated-flow history per tracker ID. It never imports or modifies `MotionPipeline`. Tests may instantiate both pipelines side by side to prove orthogonality.

- [ ] **Step 1: Write RED tests** for independent gait histories, stale removal, track-ID reuse isolation, walking-in-place with V1 `LateralState.STATIONARY`, and walking combined separately with `APPROACHING`, `RECEDING`, and `STABLE` V1 radial results.
- [ ] **Step 2: Run** `PYTHONPATH=src pytest tests/test_gait_pipeline.py tests/test_v2_regression.py -v` and verify RED.
- [ ] **Step 3: Implement bounded history/update/drop behavior** mirroring V1 semantics without sharing mutable history.
- [ ] **Step 4: Re-run focused tests and then full V1 suite.** Require `PYTHONPATH=src pytest -q` GREEN.
- [ ] **Step 5: Commit:** `feat: add per-track optical gait pipeline`.

### Task 7: Add an optional RTMW adapter without coupling core logic to MMPose

**Files:**
- Create: `src/yolo_motion/rtmw_adapter.py`
- Create: `tests/test_rtmw_adapter.py`
- Modify: `pyproject.toml`

**Interfaces:**

```python
@dataclass(frozen=True)
class RtmwAdapterConfig:
    model_config: str
    weights: str
    device: str = "cpu"

class RtmwPoseAdapter:
    def __init__(self, inferencer: object): ...
    def infer_track(
        self,
        frame: np.ndarray,
        track: TrackObservation,
        timestamp: float,
    ) -> PoseObservation | None: ...


def load_mmpose_rtmw(config: RtmwAdapterConfig) -> RtmwPoseAdapter: ...
```

`load_mmpose_rtmw()` lazily imports `MMPoseInferencer`, uses the caller-supplied RTMW config/weights, and runs top-down on the tracked crop/whole image without creating a second person detector. The adapter converts output immediately to full-frame normalized COCO-WholeBody-133 coordinates. A no-detection/low-quality inference returns `None`; malformed 133-point output raises a contextual `ValueError`.

Add an `rtmw` optional dependency containing `mmpose>=1.3`; default CI does not install it.

- [ ] **Step 1: Write RED adapter tests** with a fake inferencer returning one 133-point sample, no sample, malformed 17-point output, and crop-relative coordinates.
- [ ] **Step 2: Run** `PYTHONPATH=src pytest tests/test_rtmw_adapter.py -v` and verify RED without MMPose installed.
- [ ] **Step 3: Implement the duck-typed adapter and lazy loader.** Importing `yolo_motion.rtmw_adapter` must succeed without MMPose; only `load_mmpose_rtmw()` may require it.
- [ ] **Step 4: Re-run adapter tests and full core suite.**
- [ ] **Step 5: Commit:** `feat: add optional RTMW pose adapter`.

### Task 8: Add deterministic gait benchmark, CI gate, and real-video handoff

**Files:**
- Create: `scripts/benchmark_gait_synthetic.py`
- Create: `tests/test_gait_benchmark.py`
- Modify: `.github/workflows/ci.yml`
- Modify: `README.md`
- Update: `docs/superpowers/plans/2026-09-15-rtmw-optical-gait-v2.md` only to record completed evidence after execution

**Interfaces:**

`python scripts/benchmark_gait_synthetic.py` emits JSON for at least these scenarios and exits non-zero on mismatch:

```text
standing
walking_in_place
walking_transverse
walking_approaching
walking_receding
running
rigid_translation_control
too_small_unknown
camera_translation_compensated
```

CI keeps the existing Python 3.10/3.11/3.12 matrix, installs `.[dev,optical]`, runs the full pytest suite, Ruff, the existing V1 synthetic benchmark, and the new gait benchmark. It does not install RTMW/MMPose or download model weights.

- [ ] **Step 1: Add a RED benchmark smoke test** requiring all scenario names and expected states.
- [ ] **Step 2: Implement the deterministic benchmark from repository-owned synthetic pose/flow fixtures.** Ensure approaching/receding scenarios obtain those labels from V1 `MotionPipeline`, not from gait scale heuristics.
- [ ] **Step 3: Update CI** to install `.[dev,optical]` and execute both benchmark scripts.
- [ ] **Step 4: Update README** with the V2 architecture, `unknown` semantics, optional `.[optical]` and `.[rtmw]` installs, and the explicit limitation that image motion is not metric 3D motion.
- [ ] **Step 5: Run final local gates:**

```bash
python -m pip install -e '.[dev,optical]'
pytest -q
ruff check .
python scripts/benchmark_synthetic.py
python scripts/benchmark_gait_synthetic.py
```

All commands must exit 0.

- [ ] **Step 6: Run scope scans:**

```bash
grep -RniE 'flywire|\bsnn\b|rclpy|sensor_msgs|geometry_msgs' src tests scripts configs pyproject.toml || true
```

Expected: no production/test dependency matches.

- [ ] **Step 7: Commit:** `test: gate optical gait v2 behavior`.

## Final real-video evidence gate

The implementation can be code-complete before RTMW weights are provisioned, but PR #2 must remain Draft until one real-video locomotion run is recorded against the exact branch head. The run must contain at least `standing`, `walking`, and `running` or `jogging`, retain `unknown` predictions in metrics, and record tracker model, RTMW config/weights identity, flow backend, camera mode, gait config, and labeled `track_id` time intervals.

No benchmark result may be reported as metric depth, physical velocity, persistent identity, or general action recognition.

## Final verification checklist

- `pytest -q` — all V1 + V2 tests GREEN.
- `ruff check .` — GREEN.
- Existing `scripts/benchmark_synthetic.py` — GREEN and unchanged semantics.
- New `scripts/benchmark_gait_synthetic.py` — GREEN.
- PR diff contains no weakening of V1 thresholds/tests.
- `MotionPipeline` public behavior remains unchanged.
- Affine-camera failure cannot fall back to uncompensated flow.
- Missing bilateral pose/flow support cannot become walking/running.
- Far/near synthetic gait invariance and camera-translation invariance are explicitly tested.
- PR remains Draft until real-video evidence is attached to #1 / PR #2.
