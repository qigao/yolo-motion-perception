# R1-E2 Reservoir Representation Beyond Fading Memory

Status: committed design for issue #6.

## Motivation

R1/E1 is frozen. Its registered result showed that increasing reservoir budget can extend single-cue fading memory and improve corruption robustness, while long multi-event history separability remains a distinct bottleneck. R1-E2 therefore studies representation mechanisms rather than searching for a globally winning topology.

## Research questions

1. At fixed total neuron budget, does hierarchical structure improve temporal composition relative to flat or grouped reservoirs?
2. Is long-history failure caused by information loss in the reservoir state or by an instantaneous linear readout that cannot extract retained information?
3. Can reservoir temporal state improve synthetic YOLO-like episode recognition relative to a frame-only baseline?

## Frozen scientific scope

Primary total neuron budget is `N=256` for all architecture comparisons.

Architectures:

- `flat-256`: one 256-neuron recurrent component;
- `grouped-4x64`: four parallel 64-neuron components receiving the same observation;
- `hierarchical-2x128`: two serial 128-neuron recurrent levels;
- `hierarchical-4x64`: four serial 64-neuron recurrent levels.

Dynamics inherit R1/E1:

- activation: tanh;
- recurrent spectral radius: 0.9 per component;
- leak: 1.0;
- no recurrent bias;
- float64 state evolution;
- Ridge regularization: `1e-6`;
- seeds: `[7, 17, 29, 43, 61]`.

R1-E2 may reuse the already verified R1/E1 reservoir implementation as the low-level execution engine. E2 must not modify frozen R1/E1 science or evidence.

## Parameter-accounting constraint

Every registered architecture must expose, per input dimension:

- neuron count;
- recurrent-edge count;
- input-parameter count;
- total reservoir parameter count.

The primary comparison fixes neuron count, not dense-edge count. Dense recurrent parameter count is therefore an audited explanatory variable, not a hidden equality assumption: flat, grouped, and hierarchical layouts naturally induce different dense matrix sizes. Claims must not attribute any observed effect solely to topology without reporting this difference.

## E2-A: memory preservation

Use the R1/E1 delayed-cue task at delays `1/2/5/10/20/40/80` as a compatibility and fading-memory baseline. Report per-delay accuracy and contiguous 85% horizon. This task does not select an architecture.

## E2-B: temporal composition

Primitive events are `A`, `B`, and `C`. Registered order classes are `ABC`, `ACB`, `BAC`, `BCA`, `CAB`, `CBA`. Class labels are fixed in that order as `0..5`.

### Frozen fixture contract

Each frame has seven float64 channels:

- channels `0..2`: one-hot `A/B/C` event channels;
- channels `3..6`: four nuisance channels, each taking only `-0.25` or `+0.25`.

For every sequence the three event frames occur at absolute indices `0`, `2`, and `4`; frames `1` and `3` have neutral event channels. After the third event, the sequence contains exactly `H` neutral-event tail frames, where registered history span `H` is one of `5/10/20/40`. Therefore sequence length is exactly `5 + H`, and the decision state is the state after the final tail frame.

For each history span and nuisance group, all six order classes reuse the **same complete nuisance trajectory**. Only event ordering differs. This paired design prevents nuisance realization from acting as a class cue.

Registered counts are fixed prospectively:

- training: 100 nuisance groups per history span, giving 600 sequences per span and exactly 100 samples per class;
- evaluation: 25 nuisance groups per history span, giving 150 sequences per span and exactly 25 samples per class.

Training and evaluation RNG lineages are independent and must use distinct E2-B lineage tags together with the registered seed. Fixture arrays are immutable and each fixture set carries a deterministic digest.

### Readout arms

Both arms use the same reservoir trajectories and the frozen R1/E1 multiclass Ridge law with regularization `1e-6`.

- instantaneous: final reservoir state -> Ridge;
- temporal mean: mean of exactly the final `H` **post-event tail states** -> Ridge.

The temporal window therefore ends at the decision frame and excludes the reservoir state immediately after the third event. No future state is available. Learned attention, RNN/GRU/LSTM, transformer, learned pooling, or result-selected windowing are out of scope.

### Reset control

For the reset control, process frames `0..4`, reset the reservoir immediately after the third event, then process only the `H` tail frames. Within each nuisance group the resulting reset trajectory must be identical for all six class labels.

Because evaluation is class-balanced with six labels per nuisance group, both instantaneous and temporal-mean reset predictions must score **exactly 25/150 = 1/6** regardless of which label the readout chooses for a group. Any violation invalidates the arm.

### Metrics

For each history span and readout arm report:

- accuracy;
- macro-F1 over the six classes;
- the full `6 x 6` confusion-count matrix;
- all 15 pairwise Euclidean distances between class centroids;
- six within-class dispersions, each defined as the mean Euclidean distance from samples to that class centroid;
- separation ratio = mean pairwise centroid distance / max(mean within-class dispersion, float64 epsilon);
- prediction and coefficient digests;
- train/evaluation fixture digests;
- reset prediction digest and exact reset-control counts.

Geometry is computed on the exact representation consumed by the readout: final states for the instantaneous arm and pooled tail states for the temporal-mean arm. Geometry metrics describe representation; they are not replacement acceptance thresholds.

## E2-C: synthetic YOLO-like episodes

E2-C reuses the frozen E1-C episode generator and corruption implementation exactly; it does not introduce a second YOLO-like data model.

### Frozen episode contract

Registered behavior labels remain, in order, `approach`, `touch`, `pick_up`, `pass_by` as class indices `0..3`.

Every clean episode is exactly the E1-C `20 x 9` float64 sequence:

- frames `0..15`: the frozen E1-C active behavior template;
- frames `16..19`: the frozen four-frame neutral suffix;
- the nine channel meanings, nuisance values, template formulas, clipping rules, and paired-nuisance construction are inherited byte-for-byte from E1-C.

Registered counts are inherited exactly:

- training: 200 paired nuisance groups = 800 clean episodes = 200 samples per class;
- evaluation: 50 paired nuisance groups = 200 clean episodes = 50 samples per class.

All four labels inside a paired group share the same nuisance trajectory. E2-C uses the same registered R1-E2 seed set `[7, 17, 29, 43, 61]`.

### Frozen corruption contract

Evaluation corruption reuses the E1-C `CorruptionPlan` and `corrupt_sequence` implementation exactly. Registered arms remain:

`clean`, `drop10`, `wrong10`, `occlusion4`, `jitter`, `mixed`.

Training remains clean only. Corruption lineage is generated from the registered seed and frozen evaluation fixture exactly as in E1-C. The terminal neutral suffix `16..19` is never modified by any corruption arm.

### Readout arms

All fitted arms use the frozen multiclass Ridge law with regularization `1e-6`. No result-selected windowing is permitted.

- **B0 frame-only:** raw observation at frame index `19` -> Ridge.
- **B1 reservoir instantaneous:** reservoir state after frame index `19` -> Ridge.
- **B2 reservoir temporal mean:** arithmetic mean of exactly the four reservoir states produced by frames `16..19` -> the same Ridge law.

B1 and B2 must consume the same reservoir trajectory for a given episode. B2's window is fixed at four and is causal. It contains only the neutral suffix states; no active-frame state and no future state is included.

Because the semantic suffix is identical across labels and nuisance is shared within each paired group, B0 is a registered negative control: its predictions must be group-constant and its accuracy must be exactly `50/200 = 1/4` on every corruption arm.

### Reset control

For the reservoir reset control, process active frames `0..15`, reset immediately before frame `16`, then process only neutral suffix frames `16..19`.

Within each paired nuisance group the reset suffix trajectory must be identical for all four labels. Therefore both B1 and B2 reset predictions must score exactly `50/200 = 1/4`. Any violation invalidates the arm.

### Metrics

For each baseline and corruption arm report:

- accuracy;
- macro-F1 over four classes;
- full `4 x 4` confusion-count matrix;
- prediction digest.

Also report each fitted baseline's coefficient digest, B1/B2 reservoir parameter digest, train/evaluation fixture digests, corruption-plan digest, and for the five non-clean corruption arms macro corrupted accuracy and worst corrupted accuracy.

E2-C compares representations only. It does not select an architecture, tune a temporal window, alter corruption severity, or refit from registered results.

## Evidence lifecycle

The scientific sequence is mandatory:

1. committed design;
2. committed implementation plan;
3. TDD implementation and permanent CI;
4. prospective manifest and preflight verification;
5. explicit human approval for the exact manifest/science head;
6. exactly one registered measurement;
7. human interpretation review;
8. no-refit evidence freeze;
9. removal of any one-shot measurement workflow;
10. final exact-head CI.

Ordinary CI must never invoke the registered measurement command.

## Frozen exclusions

R2 delayed credit, Phase3 reward learning, online adaptation, learned recurrent weights, AutoESN runtime dependency, PyTorch, and external deep-learning frameworks remain outside R1-E2. No post-result architecture selection, seed expansion, threshold tuning, hyperparameter tuning, or refit is permitted.