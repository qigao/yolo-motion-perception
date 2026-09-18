# R1-E2 Registered Measurement Report

This report records the single registered R1-E2 measurement without refit or post-result tuning.

## Registration

- scientific head: `6b72cb6442e724c8d888aa934169a13e84168b21`
- manifest SHA-256: `0b50234fbb826c1bda43c4cf63b51a13265ebc0aa2111a8eafed525842473110`
- result SHA-256: `fbec12cdbe0cfda5750bc092c2106a4a385839bf84a0426925d5e4f9ef80f063`
- provenance SHA-256: `994c066aa86cfc46dd5f26a43f3f44327f752187b31c609c76ad255d19378327`
- trace-index SHA-256: `645e366c192128d332e7d6232f383430abb518257f81e2ce1d5307479fb3cc3c`
- registered measurement run: `35334134978`
- registered arms: 20 = 4 architectures x 5 seeds
- exactly one registered measurement run was executed.

## Facts

### E2-A memory preservation

All four architectures were perfect through delay 20. At delay 80 every arm was exactly 0.50.

Mean delay-40 accuracy across the five registered seeds:

| architecture | delay-40 mean | observed 85% horizon |
| --- | ---: | --- |
| Flat-256 | 0.605 | 20 for all 5 seeds |
| Grouped-4x64 | 0.755 | 40 for seeds 7/17; 20 otherwise |
| Hierarchical-2x128 | 0.570 | 20 for all 5 seeds |
| Hierarchical-4x64 | 0.705 | 40 for seed 43; 20 otherwise |

### E2-B temporal composition

Mean accuracy across histories 5/10/20/40:

| architecture | instantaneous | causal temporal mean |
| --- | ---: | ---: |
| Flat-256 | 0.4877 | 0.9967 |
| Grouped-4x64 | 0.4807 | 0.9953 |
| Hierarchical-2x128 | 0.4897 | 0.9967 |
| Hierarchical-4x64 | 0.4530 | 0.9817 |

At history 40, instantaneous readout was approximately chance (about 1/6), while causal temporal mean remained 0.948-0.992 by architecture. All reset controls were exactly 1/6.

### E2-C YOLO-like episodes

The frame-only negative control was exactly 0.25 on every corruption arm for every registered architecture/seed. B1 and B2 clean accuracy were 1.00 throughout.

Mean corrupted accuracy:

| architecture | reservoir instantaneous | reservoir temporal mean |
| --- | ---: | ---: |
| Flat-256 | 0.9418 | 0.9468 |
| Grouped-4x64 | 0.9398 | 0.9448 |
| Hierarchical-2x128 | 0.9442 | 0.9460 |
| Hierarchical-4x64 | 0.9420 | 0.9516 |

All reservoir reset controls were exactly 0.25.

## No-refit interpretation

1. The long-history composition bottleneck is primarily an extraction/readout problem in this registered synthetic task, not complete loss of temporal information from reservoir trajectories. The fixed causal temporal-mean decoder changes E2-B from roughly 0.45-0.49 macro accuracy across histories to roughly 0.98-1.00.
2. Architecture effects are much smaller and less stable than the decoder effect. The registered result does not establish a stable topology winner.
3. E2-A preserves the E1 observation: topology can shift delay-40 behavior on some seeds, but none of the registered structures preserves useful signal to delay 80.
4. In E2-C, temporal mean gives a modest robustness improvement over instantaneous reservoir readout, while the frame-only and reset controls behave exactly as preregistered.
5. No seeds, windows, thresholds, corruption severities, architectures, or Ridge settings were changed after observing results.

## Scope

These conclusions apply to the registered deterministic synthetic fixtures. They do not establish the same effect on real YOLO video, learned recurrent weights, delayed reward credit, or online adaptation.
