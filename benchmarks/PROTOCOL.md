# Fixed-Camera V1 Real-Video Protocol

This protocol exists to measure the bbox-temporal baseline before adding depth or ego-motion. Keep the camera fixed for every V1 clip.

## Capture setup

- Mount the RGB camera on a tripod or rigid surface; do not pan, tilt, zoom, or hand-hold it during a clip.
- Prefer 1920×1080 or 1280×720 at 25–30 FPS. Keep the same resolution and frame rate across the first benchmark batch when practical.
- Record one primary person per clip for the first baseline. Background people should be avoided when possible.
- Keep the whole body visible for ordinary scenarios. `partial_occlusion` is the deliberate exception.
- Use roughly 6–12 seconds per clip with about 1 second of lead-in and tail time.
- Do not stage unsafe near-collisions; “approaching” only needs a clear monotonic scale increase.
- Do not use digital stabilization, digital zoom, or post-capture cropping that changes scale over time.

## Scenario definitions

| Scenario | Required behavior | Expected state |
| --- | --- | --- |
| `stationary` | Person remains at approximately constant image position and distance. | stationary / stable |
| `lateral_crossing` | Person walks left↔right at roughly constant depth. | moving / stable |
| `approaching` | Person walks mostly toward the camera with limited lateral drift. | stationary / approaching |
| `receding` | Person walks mostly away from the camera with limited lateral drift. | stationary / receding |
| `diagonal_approaching` | Person moves laterally while also approaching. | moving / approaching |
| `approach_then_stop` | Person approaches, then stops at a stable distance. | approaching → stable transition |
| `pose_change` | Person stays at fixed distance but changes silhouette (turn, open arms, crouch mildly). | stationary / stable; negative control for bbox expansion |
| `partial_occlusion` | Person passes behind a foreground occluder and re-emerges. | preserve the physically correct state through/after occlusion when track continuity allows |

For the first baseline, one good clip per scenario is enough to expose gross errors. Use three or more independent clips per scenario before treating percentages as stable performance estimates.

## File layout

Put local clips here (video files are ignored by git):

```text
benchmarks/
├── videos/
│   ├── stationary.mp4
│   ├── lateral_crossing.mp4
│   └── ...
├── annotations/
│   ├── stationary.yaml
│   ├── lateral_crossing.yaml
│   └── ...
└── suite.yaml
```

The repository ships empty annotation stubs. Do not invent `track_id` values in advance.

## Annotation workflow

1. Record/copy a clip into `benchmarks/videos/`.
2. Temporarily enable that scenario in `benchmarks/suite.yaml` only after the clip exists.
3. Run the video pipeline to obtain the exact tracker IDs for that run:

```bash
python -m yolo_motion.cli \
  --source benchmarks/videos/approaching.mp4 \
  --jsonl runs/approaching-preview.jsonl \
  --show
```

4. Fill the corresponding annotation file using the observed `track_id` and only the time interval where the intended behavior is unambiguous:

```yaml
video: videos/approaching.mp4
intervals:
  - track_id: 1
    start: 1.2
    end: 5.8
    lateral: stationary
    radial: approaching
```

5. Run the dataset preflight. It fails when no scenarios are enabled or when an enabled scenario has a missing/empty video, missing/invalid annotations, or zero annotation intervals:

```bash
python scripts/check_benchmark_dataset.py --suite benchmarks/suite.yaml
```

6. Once preflight returns success, run the suite:

```bash
python scripts/run_benchmark_suite.py \
  --suite benchmarks/suite.yaml \
  --output-dir runs/benchmark
```

## Labeling rules

- Label only what is visually clear; shorten an interval around ambiguous transitions rather than forcing a label.
- `lateral` and `radial` are orthogonal. A diagonal approach should label both.
- For `approach_then_stop`, use separate intervals for `approaching` and `stable`.
- For `pose_change`, keep the subject's physical distance fixed. The scene is specifically intended to expose false radial positives caused by silhouette growth.
- For `partial_occlusion`, do not label frames where the intended track is completely absent unless evaluating re-acquisition behavior separately.
- Re-run inference before final annotation if tracker settings, detector model, confidence threshold, or video content changes, because `track_id` is run-specific.
