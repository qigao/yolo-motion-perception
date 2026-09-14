# Track annotation workflow

This V1 benchmark deliberately keeps prediction and ground truth separate. YOLO/BoT-SORT predictions may help you find the correct `track_id`, but predicted motion states must **not** be copied into ground-truth labels automatically.

## 1. Run the tracker and write JSONL

```bash
python -m yolo_motion.cli \
  --source benchmarks/videos/approaching.mp4 \
  --jsonl runs/approaching.jsonl
```

## 2. Summarize tracks and create a draft

```bash
python scripts/summarize_tracks.py \
  --predictions runs/approaching.jsonl \
  --annotation-draft benchmarks/annotations/approaching.yaml
```

The terminal summary shows each `track_id`, class, first/last timestamp, sample count, and the distribution of predicted lateral/radial states. Those distributions are diagnostic only.

The generated YAML contains only candidate track/time ranges:

```yaml
video: benchmarks/videos/approaching.mp4
intervals:
  - track_id: 1
    start: 0.42
    end: 7.91
    # lateral:
    # radial:
```

## 3. Human-label the interval

Review the video, keep only the intended subject and valid time range, then enter the true labels manually. For example:

```yaml
video: benchmarks/videos/approaching.mp4
intervals:
  - track_id: 1
    start: 1.10
    end: 6.80
    lateral: stationary
    radial: approaching
```

Do not infer ground truth from the summary's predicted state percentages.

## 4. Preflight and benchmark

```bash
python scripts/check_benchmark_dataset.py --suite benchmarks/suite.yaml
python scripts/run_benchmark_suite.py \
  --suite benchmarks/suite.yaml \
  --output-dir runs/benchmark
```

See `PROTOCOL.md` for scene definitions and capture constraints.
