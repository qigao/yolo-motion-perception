# R1-E3A Real-Video Acquisition and Annotation Handbook

Status: **pre-freeze acquisition draft**

This handbook is Stage-A protocol material only. It does not authorize R1-E3B
prospective sealing or any registered R1-E3 measurement.

The verified science head remains external to this acquisition branch. This
document must be reviewed and frozen before accepted episode annotation begins.

## 1. Purpose

R1-E3A turns real fixed-camera video into a frozen real-track artifact without
allowing detector behavior or downstream decoder scores to redefine the corpus.

This handbook closes two practical gaps that appear only once real MEVA data is
used:

1. simultaneous MEVA camera views can show the same physical event, so a
   whole-video split must also be capture-group aware;
2. the four registered class names need operational human definitions and
   boundary rules before annotations are frozen.

The registered class order remains unchanged:

1. `approach`
2. `touch`
3. `pick_up`
4. `pass_by`

## 2. Separation of acquisition, semantic annotation, extraction, and scoring

The allowed order is:

1. discover source-video candidates from public metadata / KPF annotations;
2. choose the source corpus and freeze capture-group-aware train/eval split;
3. record exact source bytes and source metadata;
4. annotate event semantics from source-video pixels under this handbook;
5. run the one accepted sealed YOLO11 + BoT-SORT extraction;
6. bind already-frozen semantic episodes to actor/target track IDs;
7. apply the registered deterministic eligibility and normalization gates;
8. freeze the E3A artifact;
9. only after that, prepare the E3B prospective gate.

Prohibited:

- choosing source videos because YOLO tracks them well;
- choosing or moving train/eval videos because a decoder score is good or bad;
- changing label or time bounds because a track is fragmented;
- dropping a difficult but semantically valid annotation by hand after seeing
  detector/tracker output;
- revising annotations after any registered E3B result is observed.

A semantic episode that cannot be bound to eligible frozen tracks is rejected by
the predeclared Stage-A eligibility gate. Its semantic annotation remains in the
audit record unchanged.

## 3. MEVA source-window components and split isolation

### 3.1 Exact capture-group identity

For MEVA ground-camera filenames, retain the descriptive exact group:

```text
capture_group_id =
  <date>.<start-time>.<end-time>.<site>
```

The camera token is deliberately omitted. This catches synchronized multi-view
files whose source window strings are identical.

### 3.2 Stronger source-window component

The first real inventory showed that exact equality is not sufficient. MEVA
contains same-date/site source windows whose timestamps differ by one second or
whose nominal windows touch even though they may originate from the same
continuous recording boundary.

Therefore the split unit is the stronger `source_window_component_id`:

1. group exact capture groups by `date + site`;
2. sort by source start time;
3. transitively merge the next source window when
   `next_start <= current_end + 1 second`;
4. assign the whole connected source-window component to one split.

The one-second guard band is a fail-closed protection against timestamp
rounding/off-by-one clip boundaries. It is defined before source-video download,
YOLO/BoT-SORT extraction, or any R1-E3 score.

### 3.3 Split rule

All selected videos sharing one `source_window_component_id` MUST receive the
same split.

The stronger Stage-A split requirements are therefore:

- zero video SHA-256 overlap across train/eval;
- zero re-encoded/cropped/overlapping-source overlap across train/eval;
- zero `source_window_component_id` overlap across train/eval;
- at least two distinct training source-window components;
- at least two distinct evaluation source-window components;
- retain the registered minimum of at least two source videos in each split;
- where practical, use the same site/camera families on both sides while using
  different source-window components, so background or camera identity is not a
  trivial split cue.

The split is frozen before accepted extraction.

### 3.4 Multiview event de-duplication

Multiple selected cameras from one source-window component may be used as
source-pixel context during semantic review, but simultaneous views of the same
physical actor-target event MUST NOT become multiple registered episodes.

Each semantic event receives a stable `physical_event_id`. Exactly one source
view is designated the registered episode view before accepted detector/tracker
extraction. Other synchronized views are `context_only` and remain audit
evidence.

The registered view may be chosen from source pixels for semantic visibility,
occlusion, and event continuity only. Detector confidence, track quality, or
decoder score must not influence view choice.

## 4. Candidate discovery is not a registered label

Public MEVA KPF activities may be used to find promising source recordings.
For example, `person_picks_up_object` may be used to locate videos likely to
contain movable targets.

Those KPF activity labels are discovery metadata only. They are not imported as
the registered `approach/touch/pick_up/pass_by` labels.

No candidate KPF count, detector confidence, track quality, reservoir state, or
decoder score is a registered outcome.

## 5. Annotation unit

One annotation represents one complete reviewed actor-target event.

The annotator freezes:

- `episode_id`;
- `physical_event_id`;
- `video_id`;
- `capture_group_id`;
- `source_window_component_id`;
- `start_frame_inclusive`;
- `end_frame_exclusive`;
- derived `start_time_seconds` / `end_time_seconds`;
- one registered label;
- stable actor description for audit;
- stable target description for audit;
- annotation revision;
- annotator identity/provenance;
- review status and reviewer/provenance.

The frame interval is authoritative. Time-in-seconds fields are derived from
the frozen source FPS and are convenience metadata only.

After the accepted extraction, the frozen event is bound to exactly one
`actor_track_id` and one `target_track_id`.

The semantic label, registered source view, and frame bounds are not changed
during track binding.

## 6. Maximal-outcome rule

The four labels are mutually exclusive at the complete-event outcome level.

Do not split one physical event into overlapping class prefixes.

Examples:

- if an actor approaches, touches, then picks up the target, annotate the event
  once as `pick_up`; do not also create `approach` or `touch` episodes;
- if an actor approaches and touches but never picks up the target, annotate
  once as `touch`;
- if an actor approaches but the reviewed event ends without contact, it may be
  `approach`;
- if the actor comes near and then recedes without contact, it is `pass_by`,
  not `approach`.

This rule preserves the temporal-history interpretation used by the earlier
synthetic E1-C templates: the label describes what the observed history
culminates in, not merely the first phase that occurred.

## 7. Operational label definitions

### 7.1 `approach`

Required:

- one actor and one target are continuously identifiable for the reviewed
  event;
- the actor shows sustained closing motion toward that target;
- the target is not physically contacted or manipulated during the complete
  reviewed event;
- the target does not transition into actor-coupled carried motion.

Exclude or relabel:

- if clear contact occurs in the same uninterrupted event, use `touch` or
  `pick_up`;
- if the actor passes the target and separation clearly increases again, use
  `pass_by`;
- if intent/target identity cannot be determined, exclude as ambiguous.

### 7.2 `touch`

Required:

- the event contains an approach/closing phase or equivalent transition into
  contact;
- visible source pixels support physical contact or manipulation between the
  actor and target;
- the target does not clearly leave its support and does not enter sustained
  actor-coupled carried motion during the complete event.

If the same uninterrupted interaction progresses into a clear pickup, the
maximal outcome is `pick_up`.

Box overlap alone is not sufficient proof of physical contact.

### 7.3 `pick_up`

Required:

- the actor establishes contact/manipulation of the target;
- the target clearly leaves its prior support OR otherwise undergoes a clear
  transition into sustained displacement controlled by the actor;
- source pixels support a lift/carry transition rather than a detector-box
  artifact;
- the target remains the same physical target through the event.

Where visible, actor-target co-motion after pickup is supporting evidence but is
not substituted for the human semantic decision.

### 7.4 `pass_by`

Required:

- actor-target separation first decreases and later increases within the same
  reviewed event;
- no physical contact/manipulation occurs;
- the actor continues past / away from the target rather than terminating in a
  near-target approach state.

If contact occurs, the event is not `pass_by`.

## 8. Event-boundary rules

The purpose of the window is to include the informative history that culminates
in the maximal outcome. It is not a short crop around the final frame.

Freeze boundaries as a half-open source-frame interval
`[start_frame_inclusive, end_frame_exclusive)`. Do not freeze floating-point
seconds as the primary boundary representation.

### 8.1 Start

Set the start at the earliest source-video time at which all of the following
are true:

- actor and target are visually resolvable;
- the event can be followed without a camera cut;
- sustained event-relevant actor motion toward / around the target begins.

Do not start only at first contact for `touch` or `pick_up`; their approach
history belongs inside the episode when visible.

### 8.2 End

- `approach`: end once closing motion has resolved into a stable near-target
  state without contact, or the distinct approach event otherwise clearly
  terminates.
- `touch`: end when contact/manipulation clearly terminates or settles, after
  enough visual context to determine that the same interaction did not become
  a pickup.
- `pick_up`: end after the pickup outcome is visually established and the
  target is clearly off its prior support / in actor-controlled motion.
- `pass_by`: end after closest approach, once increasing separation is
  visually established.

The annotator reviews at least 1.0 second of source context after the provisional
end when available. That review context is not automatically part of the
registered episode; it is used only to decide whether the event actually
continues into a stronger maximal outcome.

If the event continues without a meaningful break, extend the window and apply
the maximal-outcome rule.

## 9. Actor and target policy

- exactly one actor and one target are frozen per episode;
- the actor is normally a person track;
- the target must be a physical entity expected to be trackable by the sealed
  extraction policy;
- for `pick_up`, the target must be a movable object;
- target identity must remain stable across the full event;
- prefer target categories that are standard YOLO-detectable classes where
  practical;
- prefer the same target-category family across labels where practical;
- if multiple target categories are required, freeze and report per-label
  target-category counts before E3B sealing.

The detector class ID is not itself a registered decoder input.

## 10. Ambiguity and exclusion codes

Ambiguous semantic candidates are excluded before track binding and before any
registered score.

Use one of these frozen reason codes:

- `ambiguous_actor_identity`
- `ambiguous_target_identity`
- `ambiguous_contact`
- `ambiguous_pickup_transition`
- `compound_or_overlapping_event`
- `insufficient_pre_event_context`
- `insufficient_post_event_context`
- `camera_cut_or_view_change`
- `target_not_visually_resolvable`
- `preexisting_target_motion_confounds_outcome`
- `other_predeclared_semantic_ambiguity`

Do not use detector failure, low detector confidence, bad downstream accuracy, or
an inconvenient confusion matrix as a semantic exclusion reason.

## 11. Two-pass review

Every candidate semantic annotation receives:

1. an initial annotation pass;
2. a separate review pass before freeze.

The reviewer may inspect source-video pixels and the frozen source metadata.
The reviewer must not inspect R1-E3 decoder outputs because none may exist yet.

Review outcomes:

- `accepted`
- `rejected:<reason_code>`
- `needs_resolution`

All `needs_resolution` rows must be resolved or excluded before annotation
freeze. The final frozen annotation set contains no unresolved rows.

## 12. Post-extraction track binding

After one accepted sealed extraction:

1. map each frozen semantic event to one actor track and one target track;
2. do not alter the semantic label or source-time bounds;
3. apply the registered deterministic Stage-A eligibility checks;
4. if track binding is impossible or ambiguous, record an extraction/binding
   rejection reason rather than relabeling the event;
5. keep all rejected semantic rows in the audit evidence.

This separates human event semantics from detector/tracker convenience.

## 13. Freeze evidence

Before accepted E3A artifact generation, record:

- this handbook SHA-256;
- source-video candidate inventory digest;
- final source-video manifest SHA-256;
- source-window component / split manifest SHA-256;
- one-second source-window guard rule/version;
- zero source-window-component overlap assertion;
- semantic annotation file SHA-256;
- multiview `physical_event_id` de-duplication assertion;
- registered-view/context-view audit fields;
- review file SHA-256;
- rejected/ambiguous candidate counts by reason;
- per-label train/eval counts;
- distinct video and capture-group counts per split;
- per-label target-category counts;
- statement that no detector/tracker/decoder score influenced source selection,
  split, semantic labels, or source-time bounds.

Only after these are frozen may the accepted normalized artifact proceed to the
existing Task 11 replay and dataset-size gates.
