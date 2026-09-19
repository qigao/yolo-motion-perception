# R1-E3A Semantic Annotation Sampling Protocol

Status: **pre-annotation freeze addendum**

This document is an acquisition/annotation addendum to
`docs/r1-e3/acquisition-annotation-handbook.md`. It does not change the frozen
source-video selection or train/eval split, and it does not authorize detector,
tracker, decoder, prospective-sealing, or registered-measurement work.

The purpose of this addendum is to freeze **which source pixels are reviewed and
when review may stop** before semantic annotation begins.

## 1. Frozen inputs

Annotation is bound to:

- science head `90cf4b55398935786de1c5e24c80c1a7597ac43b`;
- source manifest SHA-256
  `1db8cc200ba9eaab2c14ed260b06ed582ed75d54f57c3036e1312cd2f184f4cf`;
- source-pool plan SHA-256
  `e192c9228c9f440b078db595e916300108e28c96fb005c2f183750986c3aaf58`;
- handbook SHA-256
  `44fbcabf57a6cc8f39871068ed8e4539ae6cc03f80e222338b0e4c15569b086e`;
- source batch-1 workflow run `35434084445`.

No source video may be added, removed, moved between splits, or reordered because
of detector/tracker behavior or an R1-E3 score.

## 2. Activated-video exhaustive review

Once a source video is activated by the frozen source-pool ladder, its entire
source-frame range MUST be reviewed in chronological order.

For an activated video:

1. begin at source frame 0;
2. continue through the final source frame;
3. log every visually plausible actor-target event that could resolve to one of
   `approach/touch/pick_up/pass_by`;
4. apply the handbook maximal-outcome, boundary, ambiguity, and target rules;
5. do not stop scanning the video when a class quota has been reached.

A candidate that is reviewed and rejected remains in the audit record with the
frozen exclusion reason. Ambiguous candidates may not disappear silently.

## 3. Navigation aids

Public MEVA KPF annotations may be used only as navigation aids, for example to
jump near a likely pickup interaction.

Navigation metadata does not define:

- the registered R1-E3 label;
- source-frame boundaries;
- actor/target semantic identity;
- whether an event is accepted;
- the registered multiview source view.

A full chronological scan is still required. KPF-guided review cannot replace
the full-video pass and cannot justify omitting unhinted events.

## 4. Multiview de-duplication

For simultaneous views of the same physical event:

- assign one stable `physical_event_id`;
- choose exactly one registered source view using source-pixel visibility,
  occlusion, and event continuity only;
- retain other synchronized selected views as `context_video_ids`;
- never register the same physical event twice.

The registered-view choice is frozen before YOLO11 + BoT-SORT extraction.

## 5. Batch activation and stopping

The source-pool plan already freezes an activation ladder.

The only allowed expansion rule is:

1. fully annotate and review **all videos in the currently activated batch**;
2. compute semantic counts only;
3. if any registered class has fewer than:
   - 20 accepted training episodes, or
   - 10 accepted evaluation episodes,
   activate the next pre-ranked batch;
4. fully annotate every video in that newly activated batch;
5. repeat until all count minima are met or the frozen ladder is exhausted.

Prohibited:

- activating a later video because it looks easier for YOLO/BoT-SORT;
- skipping an activated video after enough examples have been found;
- stopping halfway through a video because a class quota is reached;
- selecting only the easiest accepted episodes to hit exactly 20/10;
- discarding additional valid episodes after the minima are exceeded.

All accepted episodes from activated, fully reviewed videos remain in the frozen
semantic annotation set.

## 6. Review order

To make the audit deterministic:

- process activation batches in ascending integer order;
- within a batch, process split as `train` then `eval`;
- within each split, process `source_video_id` in lexical order;
- within a video, process candidate events by
  `start_frame_inclusive`, then `end_frame_exclusive`, then
  `physical_event_id`.

This order is administrative only and does not change labels or boundaries.

## 7. Two-pass completion rule

The first pass enumerates and annotates candidates from source pixels.

The second pass reviews every logged candidate under the frozen handbook.

An activated batch is complete only when:

- every activated video has a recorded full-range review completion marker;
- every logged candidate is `accepted` or `rejected:<reason_code>`;
- no `needs_resolution` candidate remains;
- multiview physical-event de-duplication is complete.

For each source video, the completion record freezes
`reviewed_candidate_count`. This count is the number of unique reviewed
physical-event rows in which that video appears either as the
`registered_video_id` **or** in `context_video_ids`. This definition closes
the audit loop without double-registering synchronized views of one physical
event.

Only after every activated video is complete may the semantic-count gate decide
whether another frozen batch is activated.

## 8. Freeze evidence

Before accepted YOLO11 + BoT-SORT extraction, retain:

- this sampling protocol SHA-256;
- source manifest SHA-256;
- handbook SHA-256;
- source-pool plan SHA-256;
- activated batch numbers;
- full-review completion marker for every activated video;
- reviewed-candidate count per video, reconciled against registered/context
  appearances in the reviewed physical-event rows;
- accepted/rejected counts;
- per-label train/eval accepted counts;
- distinct accepted source-video counts per split;
- distinct accepted guarded source-window-component counts per split;
- zero unresolved-candidate assertion;
- zero duplicated registered `physical_event_id` assertion;
- statement that no detector/tracker/decoder output influenced candidate
  enumeration, annotation, view choice, or stopping.

No R1-E3 registered measurement may be computed in this stage.
