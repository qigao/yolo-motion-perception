# R1 Phase E1 — Registered Measurement Report

> Status: **registered measurement complete; human interpretation approved; no-refit evidence freeze**.
>
> Scientific head: `5371b80638ff429ad37e73fedb8c9357c34d5f87`
>
> Prospective preflight run: `35182093605`
>
> Registered measurement run: `35182685692`
>
> Manifest SHA-256: `148451614e84ba4da1dbd43e0f3894bd06288a6a12bf309b4dc56872feb2efed`
>
> Result SHA-256: `b7ff7d8ebff53433fc0611d1d6697a9368e07848c51403fc1273713c98e5e261`
>
> Environment: Python `3.12.14`, NumPy `2.5.3`

## Validity

The registered verifier accepted the bundle with `registered_arm_count = 50` and `valid = true`. All five Phase 2A compatibility seeds passed (5/5). The measurement reused the sealed prospective manifest, ran on the sealed scientific head, and did not refit or alter the preregistered architecture set, budgets, seeds, thresholds, fixture lineages, or corruption laws.

E1-A and E1-B reset controls remained exact. Reservoir parameter digests were unchanged before/after state collection. The registered result therefore passes the preregistered validity gates and is interpreted as scientific evidence rather than a protocol failure.

## Cross-seed summary

| Budget | Architecture | E1-A memory macro | 85% horizon by seed 7/17/29/43/61 | E1-B history macro | 80% history horizon by seed | E1-C corrupted macro |
| ---: | --- | ---: | --- | ---: | --- | ---: |
| 64 | Shallow | 85.14% | 20/20/20/20/20 | 62.25% | 5/5/5/5/5 | 93.36% |
| 64 | Grouped-2 | 84.64% | 20/10/20/10/20 | 62.78% | 5/5/5/5/5 | 93.38% |
| 64 | Grouped-4 | 85.71% | 20/20/20/20/20 | 62.65% | 5/5/5/5/5 | 93.12% |
| 64 | Deep-2 | 83.29% | 10/20/10/20/20 | 61.15% | 5/5/5/5/5 | 91.68% |
| 64 | Deep-4 | 84.36% | 10/20/20/20/20 | 60.55% | 5/5/5/1/5 | 91.46% |
| 256 | Shallow | 87.21% | 20/20/20/20/20 | 63.07% | 5/5/5/5/5 | 94.18% |
| 256 | Grouped-2 | 87.86% | 20/20/20/20/20 | 62.62% | 5/5/5/5/5 | 94.88% |
| 256 | Grouped-4 | 89.36% | 40/40/20/20/20 | 62.48% | 5/5/5/5/5 | 93.98% |
| 256 | Deep-2 | 86.71% | 20/20/20/20/20 | 63.18% | 5/5/5/5/5 | 94.42% |
| 256 | Deep-4 | 88.64% | 20/20/20/40/20 | 62.38% | 5/5/5/5/5 | 94.20% |

## E1-A — Memory capacity

Across all architectures and seeds, raising the neuron budget from 64 to 256 increased mean memory macro accuracy from 84.63% to 87.96%. Delays 1, 2 and 5 remained essentially perfect across arms; the main architecture/budget differences appear at delay 20 and especially delay 40. Delay 80 is at approximately binary chance for every registered topology.

At N=256, Grouped-4 has the highest cross-seed memory macro (89.36%) and the strongest mean delay-40 accuracy (75.50%), but only seeds 7 and 17 attain the preregistered contiguous 85% horizon of 40. Deep-4 attains horizon 40 for seed 43 only. Because this advantage is not stable across all five seeds and does not align with E1-B/E1-C into one common ordering, it is not interpreted as a general architecture winner.

## E1-B — Multi-event history separability

The long-history result is the clearest negative finding. Horizon 1 is effectively perfect and horizon 5 remains high for nearly every arm, but horizon 20 falls to roughly 27–30% accuracy and horizon 40 converges to approximately the four-class chance level of 25%. Increasing the budget from 64 to 256 changes overall E1-B macro accuracy only from 61.88% to 62.74%.

Thus, the registered experiment does not show that a larger reservoir or any tested shallow/grouped/deep topology preserves linearly separable multi-event history beyond the short horizon under the fixed dynamics. This is a representation result only: it does not imply that the underlying nonlinear state contains no information under any possible decoder, and it does not establish semantic attractors.

## E1-C — Synthetic YOLO-like corruption

All clean evaluations are 100% accurate. Jitter is effectively harmless under the registered sigma and most `occlusion4` arms are also unaffected. The meaningful degradation comes from `drop10`, `wrong10` and `mixed` corruption. Increasing the budget from 64 to 256 raises mean macro corrupted accuracy from 92.60% to 94.33%.

Architecture differences are smaller than this budget effect and do not form a stable cross-task ordering. Grouped-2/256 has the highest registered cross-seed corrupted macro (94.88%), but the preregistration explicitly forbids selecting an architecture from one corruption metric alone.

## Interpretation

The approved interpretation is **Outcome D as the primary category, with Outcome E as a secondary category**.

**Outcome D — architecture effect is small relative to seed or budget.** The 64→256 neuron-budget increase produces a consistent improvement in E1-A memory and E1-C corruption robustness, while architecture ordering changes by task and seed.

**Outcome E — mixed or unresolved topology effect.** Local topology effects exist—for example Grouped-4/256 extends delay-40 memory for some seeds—but E1-A, E1-B and E1-C do not agree on a single stable topology ordering. No post-hoc metric weighting is used to manufacture one.

The registered evidence therefore does **not** support a general claim that Shallow, Grouped, or Deep is globally superior under E1. It does support two narrower conclusions:

1. A larger fixed reservoir budget materially extends single-cue fading memory and improves synthetic detector-corruption robustness under the registered dynamics.
2. The tested architectures all lose long multi-event linear separability by H=20–40, so preserving a single vanished cue is not sufficient evidence of robust long behavior-history representation.

## Scope boundary and next research question

No R2 delayed-credit result is changed by this experiment. E1 contains no reward learning, TD update, eligibility trace, or credit assignment.

A later R1 E2 may study reservoir dynamics—spectral radius, leak, input scaling, or activation—while treating this E1 evidence as frozen. Any E2 design must be prospective and must not reinterpret E1 by refitting these registered results.

## Frozen evidence

The directory contains the exact registered files emitted by the measurement workflow:

- `manifest.json` and `manifest.sha256`
- `prospective-provenance.json` and `prospective-provenance.sha256`
- `provenance.json` and `provenance.sha256`
- `result.json` and `result.sha256`
- `trace-index.json` and `trace-index.sha256`

`report.md` is a human interpretation of those frozen files. It introduces no new fitting, seeds, thresholds, or measurements.
