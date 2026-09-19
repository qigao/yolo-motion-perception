# R1-E3 External YOLO11 + BoT-SORT Extraction Contract

This document defines the only supported handoff from **R1-E3A external extraction** into the standalone NumPy research branch. The extractor is outside the registered R1-E3B measurement. It may use YOLO11, Ultralytics, BoT-SORT, video decoding, GPU kernels, and their required dependencies, but only the frozen content-addressed files defined here cross the registration boundary.

The registered measurement must not run YOLO.
The registered measurement must not run BoT-SORT.

## 1. Accepted extraction run

One extraction run is accepted before E3B prospective sealing. Its provenance record must identify all of the following exactly:

- extraction code commit;
- source-video manifest SHA-256;
- YOLO11 implementation and package version;
- YOLO model name and exact weights SHA-256;
- detector confidence threshold;
- detector IoU/NMS threshold;
- detector class filter;
- BoT-SORT implementation and version;
- complete tracker configuration and tracker configuration SHA-256;
- frame sampling policy;
- image resize / letterbox policy;
- Python runtime;
- package lock SHA-256;
- platform and hardware provenance;
- raw tracks SHA-256;
- annotation SHA-256;
- normalized episode SHA-256.

Detector or tracker settings cannot be selected from R1-E3B decoder scores.

## 2. Source-video manifest

Every source video has a stable logical ID plus:

- exact video SHA-256;
- source/license or ownership note;
- byte size;
- native FPS;
- decoded frame count;
- duration;
- width and height;
- frozen whole-video split assignment: training or evaluation.

A video SHA-256 may occur only once. The whole-video split is immutable after artifact freeze. Re-encoded duplicates, clips, crops, or overlapping segments from the same source are not allowed to cross training/evaluation.

## 3. Raw track output

The accepted extractor writes canonical `tracks.jsonl`. Each observation row contains:

- `video_id`
- `frame_index`
- `timestamp_seconds`
- `track_id`
- detector `class_id`
- detector confidence
- absolute `x1/y1/x2/y2`
- normalized `cx/cy/w/h`
- source frame width and height in extraction provenance or video manifest

Rows are sorted by `video_id, frame_index, track_id`.

Track IDs are local identifiers, not semantic features. Predicted-only tracker rows must not masquerade as detector observations. If retained for audit, they must be separately marked and excluded from the primary registered observation stream.

## 4. Annotation handoff

Episode annotations are created and reviewed before E3B scores are computed. Each row freezes:

- episode ID;
- video ID;
- start/end timestamp;
- label in `approach/touch/pick_up/pass_by`;
- actor track ID;
- target track ID;
- annotation revision;
- reviewer/provenance.

Ambiguous episodes are excluded before freeze. Labels or episode boundaries cannot be changed after registered results.

## 5. Frame sampling

The frame sampling policy is fixed in extraction provenance. If source videos have different native FPS values, the external pipeline applies the registered sampling rule before artifact freeze.

R1-E3B consumes the frozen rows and does not resample video itself.

## 6. Deterministic replay gate

Before accepting a Stage-A artifact, a deterministic replay using the sealed extraction environment must reproduce the same content digests at the artifact boundary:

- source manifest;
- `tracks.jsonl`;
- annotations;
- normalized episode artifact.

If exact replay is not reproducible, the artifact is not eligible for E3B registration.

The replay requirement is about accepted bytes, not bitwise identity of hidden GPU intermediates.

## 7. Rejected extraction runs

Every rejected extraction run remains an audit record and must state:

- run identifier;
- environment/config digest;
- reason for rejection;
- whether any output bytes differed from the accepted run.

A rejected extraction run cannot later replace the accepted artifact after E3B results have been observed.

## 8. Provenance example

The repository fixture `tests/fixtures/r1_e3/extraction-provenance.example.json` is the schema example used by contract tests. Real extraction provenance replaces the example values with exact accepted identities and hashes.

## 9. Registration boundary

Only frozen files, checksums, normalized episode tensors, and provenance cross from E3A into E3B.

The registered NumPy runtime must not depend on Ultralytics, Torch, OpenCV inference, detector weights, BoT-SORT, video files, or mutable tracker state.

Any detector/tracker rerun after prospective sealing creates a new experiment and invalidates the prior approval checkpoint.
