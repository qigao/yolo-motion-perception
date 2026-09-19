# R1-E3A Batch-1 Semantic Review Submission

This directory is the canonical handoff point from human semantic review to the
fail-closed R1-E3A annotation-freeze gate.

## Authoritative inputs

Batch-1 review is bound to:

- science head: `90cf4b55398935786de1c5e24c80c1a7597ac43b`;
- source workflow run: `35434084445`;
- source manifest SHA-256:
  `1db8cc200ba9eaab2c14ed260b06ed582ed75d54f57c3036e1312cd2f184f4cf`;
- frozen handbook:
  `docs/r1-e3/acquisition-annotation-handbook.md`;
- frozen sampling protocol:
  `docs/r1-e3/annotation-sampling-protocol.md`.

Use the newest `r1-e3a-human-review-pack` artifact whose handbook and sampling
digests match the files above. Do **not** use the older batch-1 packet bound to
sampling SHA `2a77df781c65f4e9781aeb9fa8d20f61129a130a3699f230705c0b8bf92f438f`;
that packet is superseded by the current sampling protocol.

## Human work

The human reviewer must:

1. review the full source-frame range for every activated Batch-1 video;
2. enumerate every plausible actor-target event;
3. apply maximal-outcome labeling and frozen rejection codes;
4. de-duplicate synchronized multiview events with one `physical_event_id`;
5. choose exactly one registered source view per accepted physical event;
6. perform the second review pass;
7. leave no `needs_resolution` record;
8. set every activated video's `full_range_reviewed` to `true`;
9. reconcile `candidate_count` with registered/context appearances;
10. change top-level `status` to `reviewed`.

The canonical JSON is:

`docs/r1-e3/reviews/batch1-semantic-review.json`

Start from the canonical `annotation-review-template.json` in the current
human-review-pack artifact. The source AVI bytes remain authoritative for frame
bounds; proxies are navigation aids only.

Reviewer/annotator fields may use stable pseudonymous reviewer IDs if direct
personal names are not desired, but the IDs must remain auditable and must not
be `UNASSIGNED`.

## Automatic freeze gate

A push that adds or changes `batch1-semantic-review.json` triggers the
annotation-freeze workflow. It:

- downloads only the frozen Batch-1 metadata artifact;
- verifies the exact source-manifest SHA-256;
- recomputes handbook and sampling-protocol SHA-256 values;
- requires the submitted JSON to match all frozen bindings;
- validates full-video review completion, frame bounds, rejection codes,
  multiview consistency, `physical_event_id` uniqueness, and candidate counts;
- reports accepted train/eval counts for all four registered labels;
- reports whether the 20-train / 10-eval per-class minima are met;
- freezes a summary artifact.

The workflow never runs YOLO11, BoT-SORT, a reservoir/readout, prospective
sealing, or a registered R1-E3 measurement.
