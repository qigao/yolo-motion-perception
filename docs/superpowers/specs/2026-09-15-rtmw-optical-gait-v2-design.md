# RTMW-Guided Optical Gait V2 Design

**Date:** 2026-09-15  
**Repository:** `qigao/yolo-motion-perception`  
**Tracking issue:** #1

## 1. Goal

Extend the existing YOLO/BoT-SORT motion-perception baseline with a track-centric articulated-motion layer that can distinguish rigid person translation from internal body motion and classify basic locomotion as `unknown`, `standing`, `walking`, or `running`.

The V2 result must remain orthogonal to the existing lateral/radial state. A single track may therefore be represented as, for example:

- `walking + approaching`,
- `walking + receding`,
- `walking + stable`,
- `walking + laterally moving`, or
- `moving + gait unknown`.

The implementation is deliberately optical and geometric. RTMW supplies body geometry and confidence; optical flow supplies image motion; deterministic temporal signal processing supplies gait evidence. V2 does not introduce an SNN, FlyWire model, recurrent graph, or neural action-classification network.

## 2. Existing V1 boundary

V1 already owns a useful stable contract:

```text
YOLO + BoT-SORT
      |
      v
TrackObservation
      |
      v
TrackHistory
      |
      v
bbox center velocity + log-area trend
      |
      v
LateralState + RadialState
```

V2 must preserve this behavior. In particular:

- `TrackObservation` remains the bbox/tracker observation type and is not expanded with model-specific pose or flow payloads.
- `MotionPipeline`, `MotionEvidence`, `MotionState`, `LateralState`, and `RadialState` retain their existing semantics.
- Current synthetic and real-video V1 evaluation remain valid.
- Existing bbox log-area evidence continues to estimate `approaching | receding | stable | unknown`; gait does not replace radial evidence.

This separation is important because gait quality can legitimately be unknown while bbox motion remains useful.

## 3. Hard constraints

- Pure Python repository architecture remains intact.
- Core unit tests must require no downloaded YOLO or RTMW weights, network, GPU, or camera.
- Ultralytics and MMPose/RTMW objects are edge-adapter concerns; temporal logic consumes repository-owned immutable types.
- Track identity remains a tracker responsibility. V2 does not perform person re-identification after track expiry.
- Scale change is not identity evidence.
- A rigidly translating person-shaped region must not be sufficient evidence for walking.
- Camera motion and whole-person translation must be separated from articulated body motion before gait classification.
- Gait must fail closed to `unknown` when pose, flow, temporal support, or camera compensation is unreliable.
- Thresholds and quality gates are explicit configuration.
- Synthetic tests establish algorithm contracts only; real behavior claims require real-video evidence.

## 4. Architecture

```text
RGB frame t-1 --------------------------- RGB frame t
     |                                         |
     |                                         +--> YOLO + BoT-SORT
     |                                                  |
     |                                                  v
     |                                           TrackObservation
     |                                                  |
     |                                                  +--> existing MotionPipeline
     |                                                  |       |
     |                                                  |       v
     |                                                  |   lateral/radial state
     |                                                  |
     |                                                  +--> RTMW adapter
     |                                                          |
     |                                                          v
     |                                                   PoseObservation
     |                                                          |
     +---------------- Optical Flow Backend ---------------------+
                            |
                            v
                       FlowObservation
                            |
                            v
                 CameraMotionCompensator
                            |
                            v
                    compensated flow
                            |
                 +----------+----------+
                 |                     |
                 v                     v
           torso/body flow       body-part ROIs
                 |                     |
                 +----------+----------+
                            v
                 ArticulatedFlowEstimator
                            |
                            v
                 ArticulatedFlowEvidence
                            |
                            v
                    per-track gait history
                            |
                            v
                      GaitEstimator
                            |
                            v
                 GaitEvidence + GaitState
```

V2 adds a parallel evidence path rather than modifying the V1 bbox estimator.

## 5. New repository-owned data contracts

### 5.1 `PoseObservation`

A pose observation belongs to one `track_id` at one `timestamp` and contains whole-body landmarks in full-frame normalized coordinates.

Required fields:

- `track_id: int`
- `timestamp: float`
- `frame_width: int`
- `frame_height: int`
- `xy: ndarray[float]` with shape `(K, 2)`
- `confidence: ndarray[float]` with shape `(K,)`
- `layout: PoseLayout`

`PoseLayout` identifies the semantic landmark map. The first production adapter targets RTMW whole-body 133-point output, but core geometry tests use synthetic layouts and do not import MMPose.

The adapter converts model output immediately. Temporal/core modules never retain RTMW/MMPose result objects.

### 5.2 `BodyRegion`

V2 initially needs only locomotion-relevant geometry:

- torso,
- left thigh,
- right thigh,
- left calf,
- right calf,
- left foot,
- right foot.

A `BodyRegion` is represented as one or more line/capsule primitives generated from pose landmarks plus a confidence value. Rasterization into a flow mask occurs at the flow-analysis boundary.

The gait core does not require face or hand regions.

### 5.3 `FlowObservation`

`FlowObservation` represents dense frame-to-frame image motion and its validity metadata:

- `start_timestamp`
- `end_timestamp`
- dense `dx`, `dy` fields in frame pixels,
- optional validity/confidence map,
- backend identifier,
- frame shape.

Raw backend-specific objects are not retained.

### 5.4 `CameraMotionEstimate`

Represents the image transform attributable to the camera/background model:

- transform model (`identity` or `affine` in V2),
- affine matrix when applicable,
- matched background feature count,
- inlier count and inlier ratio,
- residual error,
- quality in `[0, 1]`,
- validity flag.

### 5.5 `ArticulatedFlowEvidence`

Per track and frame interval:

- torso translation vector,
- normalized torso translation vector,
- per-region residual flow vector,
- per-region residual motion energy,
- per-region valid-pixel fraction,
- pose/flow agreement score,
- person scale used for normalization,
- combined quality.

### 5.6 `GaitEvidence` and `LocomotionState`

`LocomotionState` is independent of `MotionState`:

```text
unknown | standing | walking | running
```

`GaitEvidence` contains:

- sample count and duration,
- left/right leg signal energy,
- dominant motion axis,
- periodicity score,
- left/right cross-correlation strength,
- phase/lag estimate,
- cadence in cycles/second,
- normalized articulated-motion amplitude,
- temporal consistency,
- aggregate quality.

No single scalar is treated as an identity signal.

## 6. RTMW geometry boundary

The RTMW adapter runs top-down on the tracked person box and converts returned landmarks into full-frame normalized coordinates associated with that `track_id`.

For locomotion V2, the geometry layer uses only the body/foot landmarks needed to build torso and leg regions. A landmark or region is invalid when its required points do not satisfy configured confidence thresholds.

Region construction rules are deterministic:

- torso: polygon/capsules spanning shoulders and hips,
- thigh: hip-to-knee capsule,
- calf: knee-to-ankle capsule,
- foot: ankle-to-foot landmark capsule/polygon when valid.

Capsule width is proportional to tracked-person height, not a fixed pixel radius. This keeps sampling geometry approximately scale-relative.

If one leg is not observable, the system may still report per-leg optical evidence, but two-leg gait phase classification fails closed unless the configured minimum bilateral support is met.

## 7. Flow backend abstraction

Define a small backend protocol:

```python
class FlowBackend(Protocol):
    def compute(self, previous_frame, current_frame, start_timestamp, end_timestamp) -> FlowObservation:
        ...
```

The first implementation is a CPU/reference OpenCV backend suitable for reproducible local experiments. It must not require model downloads.

The gait and decomposition layers depend only on `FlowObservation`. A future NVOFA, RAFT, GMFlow, or other backend can be added without changing their public contracts.

V2 does not require equivalence across flow backends. Each backend must expose its identity in benchmark output.

## 8. Camera-motion compensation

Camera motion must be explicit rather than implicitly mixed into person motion.

V2 supports two configured modes:

### `fixed`

The repository's existing fixed-camera assumption is declared explicitly. The camera transform is identity and no ego-motion claim is made.

### `affine`

For non-static or slightly vibrating cameras:

1. mask active person bounding boxes with a configurable margin,
2. select background features outside those masks,
3. track background features between frames,
4. estimate a robust partial affine transform with outlier rejection,
5. derive camera-induced pixel flow from the transform,
6. subtract camera-induced flow from raw optical flow.

If background feature count, inlier ratio, or residual quality is below configuration thresholds, compensated gait classification for that interval is invalid. The system returns `unknown` rather than silently using uncompensated flow.

A homography is deliberately deferred. V2 starts with identity/affine because they are testable and sufficient for fixed-camera plus small pan/translation/vibration experiments.

## 9. Whole-body translation removal

After camera compensation, a walking person still has a large rigid component because the entire body translates through the image.

V2 estimates body translation from robust torso flow:

```text
T_body = robust median of valid compensated torso flow vectors
```

For every body region:

```text
F_articulated(region) = F_compensated(region) - T_body
```

The torso is used because it is typically more rigid than distal limbs. When torso support is insufficient, articulated evidence is invalid for that interval.

This decomposition is the key rejection gate for the false rule `moving bbox => walking`.

## 10. Scale normalization

Raw optical-flow magnitude changes strongly with image scale. V2 normalizes articulated pixel displacement using tracked-person bbox height in pixels:

```text
F_norm = F_articulated / person_height_px
```

The selected scale must be positive and above a configured minimum pixel height.

Person height below that minimum does not force a gait label. The correct result is `unknown` while V1 may still report useful lateral/radial motion.

The existing V1 log-area estimator remains the radial evidence source; V2 does not infer metric depth from scale normalization.

## 11. Pose/flow consistency

RTMW geometry and optical flow provide independent observations of image motion.

For valid joints/regions, V2 compares pose displacement with local optical flow. Large disagreement reduces interval quality instead of being interpreted as high articulated motion.

This specifically protects against one-frame pose landmark jumps.

The consistency check is a quality gate only. RTMW landmark velocity is not substituted for optical flow as the primary gait-motion signal.

## 12. Temporal gait representation

For every track, V2 stores a bounded trailing history of `ArticulatedFlowEvidence`.

From left/right thigh/calf/foot residual vectors it constructs bilateral leg motion signals. Because viewing direction changes which image axis carries the largest gait motion, the estimator derives one dominant 2D motion axis from the current temporal window and projects both leg residual vectors onto that axis.

The estimator then computes:

- per-leg normalized energy,
- autocorrelation-based periodicity,
- left/right cross-correlation,
- lag/phase relationship,
- cadence from the dominant temporal period,
- consistency across overlapping subwindows.

Walking evidence requires both periodicity and a stable bilateral phase relationship. High track translation without these signals is not walking evidence.

## 13. Locomotion classification

Classification is deterministic and threshold-driven.

### `unknown`

Returned when any required support is insufficient, including:

- person too small,
- insufficient temporal duration or samples,
- unreliable pose geometry,
- unreliable flow,
- invalid camera compensation in `affine` mode,
- insufficient bilateral leg support,
- temporally inconsistent signals.

### `standing`

Requires sufficient quality/history and articulated leg motion below the configured standing-energy threshold.

### `walking`

Requires sufficient quality/history plus:

- articulated motion above standing threshold,
- periodicity above threshold,
- bilateral cross-correlation/phase within configured acceptance,
- cadence within the configured broad walking research band.

### `running`

Uses the same periodic evidence but requires configured higher cadence and/or normalized articulated-motion amplitude. These thresholds are research heuristics and are not claimed to generalize across every camera/viewpoint without benchmark evidence.

The classifier never changes `RadialState` or `LateralState`.

## 14. Pipeline integration

V1 `MotionPipeline` remains unchanged.

V2 adds a separate `OpticalGaitPipeline` whose state is keyed by existing `track_id`. It receives repository-owned track, pose, and articulated-flow observations and produces an `OpticalGaitResult`.

A thin orchestration layer may expose a combined person result:

```text
track_id
current TrackObservation
V1 MotionEvidence / MotionState
V2 GaitEvidence / LocomotionState
```

Composition happens above both estimators. Neither estimator imports the other.

`drop_stale(active_track_ids)` semantics mirror V1. Once a track expires, V2 history is discarded. Reappearing people receive whatever new tracker ID BoT-SORT assigns; cross-track re-identification is outside scope.

## 15. Proposed module boundaries

Production changes are expected to remain small and single-purpose:

```text
src/yolo_motion/
    types.py                 # existing V1 types remain; only generic shared enums if appropriate
    pose_types.py            # PoseObservation, PoseLayout, body geometry records
    pose_regions.py          # RTMW landmark mapping and deterministic region construction
    flow_types.py            # FlowObservation, CameraMotionEstimate, region flow records
    flow_backend.py          # backend protocol + CPU/reference implementation boundary
    camera_motion.py         # identity/affine estimation and compensation
    articulated_flow.py      # torso translation and scale-normalized body-part residuals
    gait.py                  # GaitConfig, GaitEvidence, temporal estimation
    gait_state.py            # LocomotionState classification
    gait_pipeline.py         # per-track bounded history/orchestration
    rtmw_adapter.py          # optional runtime adapter; lazy third-party imports
```

Tests mirror module responsibilities instead of concentrating all scenarios in one integration test.

## 16. Configuration

V2 adds a separate gait configuration rather than overloading `MotionConfig`.

Configuration families include:

- gait history duration and minimum samples,
- minimum bbox/person height,
- pose point/region confidence,
- body-region capsule width ratio,
- flow valid-pixel fraction,
- camera-motion mode and affine quality thresholds,
- pose/flow agreement threshold,
- standing articulated-energy threshold,
- minimum periodicity,
- bilateral correlation/phase tolerance,
- walking cadence band,
- running cadence/amplitude thresholds.

Defaults are research starting points and are documented as such.

## 17. Error and quality handling

Core dataclasses reject non-finite values, inconsistent shapes, invalid time order, and invalid dimensions at construction/boundary time.

Recoverable visual uncertainty is not an exception. It is represented through explicit quality/validity fields and ultimately `LocomotionState.UNKNOWN`.

Third-party runtime failures remain adapter/CLI errors and must include the relevant track/frame context.

No fallback may silently replace:

- failed affine camera compensation with raw flow,
- missing RTMW output with fabricated keypoints,
- insufficient bilateral support with one-leg walking classification,
- expired track history with a previous track's history.

## 18. Testing strategy

### Contract/unit tests

No model weights, video, GPU, or network.

Required deterministic fixtures include:

1. pure rigid person translation,
2. walking-in-place bilateral alternation,
3. translation plus walking articulation,
4. approaching walking with increasing bbox scale,
5. receding walking with decreasing bbox scale,
6. identical gait synthesized at multiple person scales,
7. global camera translation added to the full scene,
8. one-frame pose landmark jump with stable optical flow,
9. partial leg occlusion,
10. insufficient person pixel height,
11. cadence transition between walking-like and running-like signals,
12. stale track removal and a new independent track ID.

### Required invariants

- rigid translation alone does not produce walking,
- walking-in-place can produce walking while V1 lateral state is stationary,
- scale normalization keeps comparable synthetic gait evidence within declared tolerances,
- compensated gait evidence is stable under synthetic affine/global camera translation within declared tolerances,
- approach/recede remains determined by V1 and is compatible with any V2 gait state,
- low quality produces `unknown`.

### Integration tests

Optional-runtime tests verify adapters with local/fake model outputs first. Real RTMW/YOLO model execution is not part of default CI unless weights are explicitly provisioned outside the repository.

## 19. Real-video evaluation

Extend the existing benchmark discipline rather than invent a second reporting system.

Initial locomotion scenarios should include:

- standing,
- transverse walking,
- walking toward camera,
- walking away from camera,
- walking in place,
- rigid/non-articulated translation control where feasible,
- jogging/running,
- partial occlusion.

Reports must record:

- tracker/model identifiers,
- pose adapter/model identifier,
- flow backend identifier,
- camera-motion mode,
- gait configuration,
- labeled track/time intervals,
- per-state accuracy/confusion,
- unknown rate,
- first-correct latency,
- quality-rejection counts.

An `unknown` prediction is retained as a real output, not removed from evaluation.

## 20. Acceptance gates

V2 is implementation-complete only when all of the following hold:

1. all existing V1 tests remain green without weakening expectations;
2. production gait code depends only on repository-owned observation/evidence types;
3. rigid translation rejection passes;
4. walking-in-place passes;
5. far/near scale-normalization invariance passes within a predeclared tolerance;
6. affine/global camera-motion invariance passes within a predeclared tolerance;
7. pose-jump/flow-disagreement reduces quality instead of creating false gait energy;
8. occlusion and too-small targets fail closed to `unknown`;
9. walking can coexist with each existing radial state without altering V1 classification;
10. core CI requires no model download/network/GPU;
11. at least one real-video locomotion benchmark path is exercised and reports `unknown` explicitly;
12. documentation preserves the limits: image motion is not metric 3D motion, RTMW geometry is not identity, and heuristic walk/run thresholds are research results rather than safety claims.

## 21. Explicit non-goals

- person ReID after tracker loss,
- cross-camera identity,
- metric depth or 3D reconstruction,
- face or hand gesture semantics,
- arbitrary action recognition,
- neural action-classification models,
- SNN/FlyWire models,
- SLAM,
- ROS2 integration,
- control/path-planning decisions.

## 22. Delivery order

Implementation is intentionally staged:

1. freeze contracts, synthetic signals, and quality semantics;
2. implement optical/camera/body-motion decomposition with deterministic RED/GREEN tests;
3. add RTMW geometry adapter and pose/flow consistency;
4. add temporal gait estimation/state classification;
5. extend real-video benchmark/evidence and document measured limits.

The detailed RED/GREEN implementation plan is written only after this design is reviewed and approved.