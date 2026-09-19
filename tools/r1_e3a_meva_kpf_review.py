#!/usr/bin/env python3
"""Build deterministic KPF-guided review crops for one MEVA feasibility pilot.

Acquisition-only: uses MEVA human KPF activity/geometry annotations plus source
video pixels. It does not run a detector, tracker, reservoir, or registered score.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import yaml
from PIL import Image, ImageDraw


def parse_pickups(path: Path) -> list[dict]:
    doc = yaml.safe_load(path.read_text(encoding="utf-8", errors="ignore")) or []
    events = []
    for ordinal, item in enumerate(doc):
        act = item.get("act") if isinstance(item, dict) else None
        if not isinstance(act, dict):
            continue
        act2 = act.get("act2") or {}
        name = str(next(iter(act2), "")) if isinstance(act2, dict) else ""
        if name != "person_picks_up_object":
            continue
        ids, spans = [], []
        for actor in act.get("actors", []):
            if not isinstance(actor, dict):
                continue
            if actor.get("id1") is not None:
                ids.append(int(actor["id1"]))
            for ts in actor.get("timespan", []):
                tsr = ts.get("tsr0") if isinstance(ts, dict) else None
                if isinstance(tsr, list) and len(tsr) == 2:
                    spans.append((int(tsr[0]), int(tsr[1])))
        if spans:
            events.append({
                "ordinal": ordinal,
                "actor_ids": sorted(set(ids)),
                "start_frame": min(a for a, _ in spans),
                "end_frame": max(b for _, b in spans),
            })
    return events


def parse_geom(path: Path) -> dict[int, dict[int, tuple[int, int, int, int]]]:
    doc = yaml.safe_load(path.read_text(encoding="utf-8", errors="ignore")) or []
    out: dict[int, dict[int, tuple[int, int, int, int]]] = defaultdict(dict)
    for item in doc:
        geom = item.get("geom") if isinstance(item, dict) else None
        if not isinstance(geom, dict):
            continue
        aid, frame, box = geom.get("id1"), geom.get("ts0"), geom.get("g0")
        if aid is None or frame is None or not box:
            continue
        parts = str(box).split()
        if len(parts) != 4:
            continue
        x1, y1, x2, y2 = (int(float(v)) for v in parts)
        out[int(aid)][int(frame)] = (x1, y1, x2, y2)
    return out


def nearest_box(track: dict[int, tuple[int, int, int, int]], frame: int):
    if not track:
        return None, None
    nearest = min(track, key=lambda candidate: abs(candidate - frame))
    if abs(nearest - frame) > 3:
        return None, None
    return nearest, track[nearest]


def extract_frame(video: Path, frame: int, fps: float, out: Path) -> None:
    seconds = frame / fps
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{seconds:.6f}", "-i", str(video),
        "-frames:v", "1", "-q:v", "2", str(out),
    ], check=True)


def padded_crop(boxes, width, height):
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    target_w = max(480, bw * 2.4)
    target_h = max(270, bh * 2.4)
    target_w = min(width, target_w)
    target_h = min(height, target_h)
    left = max(0, min(width - target_w, cx - target_w / 2))
    top = max(0, min(height - target_h, cy - target_h / 2))
    return tuple(map(int, (left, top, left + target_w, top + target_h)))


def evenly_spaced_indices(count: int, n: int) -> list[int]:
    if n <= 1:
        return [0]
    return sorted({round(i * (count - 1) / (n - 1)) for i in range(n)})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--activities", type=Path, required=True)
    parser.add_argument("--geom", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--expected-pickups", type=int, default=129)
    parser.add_argument("--sample-count", type=int, default=12)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    events = parse_pickups(args.activities)
    if len(events) != args.expected_pickups:
        raise SystemExit(
            f"pickup-count mismatch: expected={args.expected_pickups} parsed={len(events)}"
        )
    geom = parse_geom(args.geom)
    selected = evenly_spaced_indices(len(events), args.sample_count)

    panels = []
    evidence = []
    tmp = args.out_dir / "frames"
    tmp.mkdir(exist_ok=True)

    for sample_no, event_index in enumerate(selected, 1):
        event = events[event_index]
        start, end = event["start_frame"], event["end_frame"]
        frame_numbers = [start, round((start + end) / 2), end]
        sample_panels = []
        sample_frames = []
        for phase, frame in zip(("start", "mid", "end"), frame_numbers):
            raw = tmp / f"s{sample_no:02d}-{phase}-raw.jpg"
            extract_frame(args.video, frame, args.fps, raw)
            image = Image.open(raw).convert("RGB")
            found = []
            boxes_for_crop = []
            for actor_id in event["actor_ids"]:
                matched_frame, box = nearest_box(geom.get(actor_id, {}), frame)
                if box is not None:
                    boxes_for_crop.append(box)
                    found.append({
                        "actor_id": actor_id,
                        "geometry_frame": matched_frame,
                        "box": list(box),
                        "width_px": box[2] - box[0],
                        "height_px": box[3] - box[1],
                        "area_px": max(0, box[2] - box[0]) * max(0, box[3] - box[1]),
                    })
            if len(boxes_for_crop) != len(event["actor_ids"]):
                raise SystemExit(
                    f"missing KPF geometry sample={sample_no} phase={phase} "
                    f"frame={frame} actor_ids={event['actor_ids']} found={len(found)}"
                )
            crop_box = padded_crop(boxes_for_crop, image.width, image.height)
            cropped = image.crop(crop_box)
            draw = ImageDraw.Draw(cropped)
            left, top, _, _ = crop_box
            for rec in found:
                x1, y1, x2, y2 = rec["box"]
                relative = (x1 - left, y1 - top, x2 - left, y2 - top)
                draw.rectangle(relative, outline="white", width=3)
                draw.rectangle(
                    (relative[0], relative[1], relative[0] + 90, relative[1] + 22),
                    fill="black",
                )
                draw.text((relative[0] + 4, relative[1] + 3), f"id {rec['actor_id']}", fill="white")
            draw.rectangle((0, 0, cropped.width, 30), fill="black")
            draw.text(
                (6, 6),
                f"sample {sample_no} | event-index {event_index} | {phase} | frame {frame}",
                fill="white",
            )
            crop_path = args.out_dir / f"sample-{sample_no:02d}-{phase}.jpg"
            cropped.save(crop_path, quality=92)
            sample_panels.append(cropped)
            areas = sorted(rec["area_px"] for rec in found)
            sample_frames.append({
                "phase": phase,
                "frame": frame,
                "actors": found,
                "smaller_actor_area_px": areas[0],
                "larger_actor_area_px": areas[-1],
                "smaller_to_larger_area_ratio": areas[0] / areas[-1] if areas[-1] else 0,
                "crop": list(crop_box),
            })

        panel_h = 360
        panel_w = 640
        strip = Image.new("RGB", (panel_w * 3, panel_h), "white")
        for col, image in enumerate(sample_panels):
            copy = image.copy()
            copy.thumbnail((panel_w, panel_h))
            x = col * panel_w + (panel_w - copy.width) // 2
            y = (panel_h - copy.height) // 2
            strip.paste(copy, (x, y))
        panels.append(strip)
        evidence.append({
            "sample_no": sample_no,
            "event_index": event_index,
            **event,
            "frames": sample_frames,
        })

    sheet = Image.new("RGB", (1920, 360 * len(panels)), "white")
    for row, panel in enumerate(panels):
        sheet.paste(panel, (0, row * 360))
    sheet.save(args.out_dir / "kpf-guided-contact-sheet.jpg", quality=88)

    (args.out_dir / "kpf-guided-review.json").write_text(
        json.dumps({
            "status": "pilot_screening_only_not_frozen_annotation",
            "selection": "12 evenly spaced pickup events from the 129 KPF events",
            "model_inference_used": False,
            "events": evidence,
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"review samples: {len(evidence)}")
    print(args.out_dir / "kpf-guided-contact-sheet.jpg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
