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

Primitive events are `A`, `B`, and `C`. Registered order classes are `ABC`, `ACB`, `BAC`, `BCA`, `CAB`, `CBA`. Evaluate history spans `5/10/20/40` with class-balanced frozen fixtures.

Readout arms:

- instantaneous: final reservoir state -> frozen Ridge;
- temporal mean: causal state window ending at the decision frame -> deterministic mean pooling -> the same Ridge law.

Temporal pooling is causal only. Future states are forbidden. Learned attention, RNN/GRU/LSTM, transformer, learned pooling, or result-selected windowing are out of scope.

Metrics:

- accuracy;
- macro-F1;
- confusion counts;
- inter-class centroid distance;
- within-class dispersion;
- between/within separation ratio.

Geometry metrics describe representation; they are not replacement acceptance thresholds.

## E2-C: synthetic YOLO-like episodes

Extend the frozen E1-C observation semantics into complete episodes for `approach`, `touch`, `pick_up`, and `pass_by`. Compare:

- B0 frame-only baseline;
- B1 reservoir + instantaneous Ridge;
- B2 reservoir + causal temporal-mean readout.

Corruption families inherit R1/E1: `clean`, `drop10`, `wrong10`, `occlusion4`, `jitter`, `mixed`. The exact episode templates, counts, corruption lineage, clipping rules, and window sizes must be frozen in the prospective manifest before measurement.

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