# Phase 3A failure attribution

Status: design-review artifact. This document explains the committed Phase 3A
failure; it does not replace or modify the measured evidence JSON.

## Inputs and fixed facts

- Evidence: `docs/experiments/phase-3a-action-value.json`
- Evidence SHA-256: `897f1917d3d950f64439ffc0c80aa83aa4c5b940cc5bf41dfbbb89f422da0295`
- Final implementation head at analysis time: `dad2e2ad72eececaa2dbad5e2a5d3ae7f270cfd3`
- Fixed configuration: hidden size 64, recurrent radius 0.9, step size 0.1,
  2,000 training episodes, 20 evaluation blocks, checkpoint interval 100.
- The scientific gate is false: seeds 7 and 29 fail; seed 17 passes.
- Frozen Phase 2C geometry is a relevant control: all three seeds have a
  supervised evaluation of 200/200, every delay is 40/40, and the delay-5 to
  delay-1 median-margin ratios are 0.99715 (seed 7), 1.00250 (seed 17), and
  1.00196 (seed 29). Phase 2C classifies seeds 7 and 29 as
  `REWARD_CREDIT_FAILURE` and seed 17 as `NO_FAILURE_REPRODUCED`.

The frozen experiment preserved equal normal/shuffled action schedules,
equal per-block reward multisets, unchanged policy matrices, changed value
parameters, and no pending feedback. Reset evaluation produces identical
hidden states and 100/200 accuracy. These are controls, not acceptance claims.

## Observed failure pattern

The failure is concentrated at the longest delays rather than being a uniform
failure of reward learning.

| Seed | Post-training | Delay 1 | Delay 2 | Delay 3 | Delay 4 | Delay 5 | Shuffled |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | 181/200 | 40/40 | 40/40 | 39/40 | **28/40** | 34/40 | 100/200 |
| 17 | 199/200 | 40/40 | 40/40 | 40/40 | 40/40 | 39/40 | 100/200 |
| 29 | 177/200 | 40/40 | 40/40 | 40/40 | 37/40 | **20/40** | 138/200 |

The read-only probe reconstructed the frozen fixtures and normal training
path through the existing APIs, then measured the post-training decision
hidden states by delay. It did not modify tracked files, regenerate evidence,
or change the fixed algorithm. `hidden_between` is the L2 distance between
the two cue-class hidden means; `hidden_within` is the mean within-class
distance to those means.

| Seed | Delay | Accuracy | hidden_between | hidden_within | Q-margin mean | Q-margin p10 | Q-margin min |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | 1 | 40/40 | 2.322 | 0.357 | 1.222 | 0.845 | 0.815 |
| 7 | 2 | 40/40 | 1.586 | 0.462 | 0.454 | 0.274 | 0.185 |
| 7 | 3 | 39/40 | 1.252 | 0.569 | 0.236 | 0.128 | -0.017 |
| 7 | 4 | **28/40** | 1.026 | 0.686 | 0.132 | -0.231 | -0.334 |
| 7 | 5 | 34/40 | 0.963 | 0.679 | 0.261 | -0.042 | -0.228 |
| 17 | 1 | 40/40 | 2.635 | 0.259 | 1.905 | 1.727 | 1.604 |
| 17 | 2 | 40/40 | 2.353 | 0.440 | 1.890 | 1.554 | 1.424 |
| 17 | 3 | 40/40 | 1.963 | 0.469 | 1.976 | 1.395 | 1.036 |
| 17 | 4 | 40/40 | 1.965 | 0.474 | 1.717 | 1.268 | 0.574 |
| 17 | 5 | 39/40 | 1.390 | 0.546 | 1.371 | 0.619 | -0.044 |
| 29 | 1 | 40/40 | 1.682 | 0.237 | 1.076 | 0.920 | 0.887 |
| 29 | 2 | 40/40 | 1.318 | 0.302 | 0.853 | 0.569 | 0.519 |
| 29 | 3 | 40/40 | 0.997 | 0.330 | 0.868 | 0.390 | 0.318 |
| 29 | 4 | 37/40 | 0.687 | 0.383 | 0.271 | 0.090 | -0.082 |
| 29 | 5 | **20/40** | 0.421 | 0.411 | 0.055 | -0.400 | -0.639 |

## Attribution

### Proven

1. The recurrent policy is the memory substrate. Its input, recurrent, and
   output matrices are frozen during the action-value experiment; only the
   action-value table changes. The probe table uses the frozen policy's
   decision-time hidden states, so the same fixture is not being adapted by
   the value learner.
2. The raw hidden-state centroid geometry becomes weaker and more variable at
   longer delays. The most severe scientific miss, seed 29/delay 5, has a
   between-class mean distance (`0.421`) only about equal to its within-class
   spread (`0.411`). This descriptive weakening is not itself an information
   ceiling: the independent Phase 2C supervised linear probe remains 40/40 at
   delay 5 for every seed.
3. The action-value learner is receiving a valid immediate scalar signal. The
   normal and shuffled paths have identical sampled actions, while the
   shuffled reward control disrupts and cannot reliably preserve the cue
   mapping (seed 29 reaches 138/200, below the preregistered `<150` control
   gate). This rules out action-schedule divergence and makes a pure “no reward
   reached the learner” explanation insufficient.

### Best-supported inference

The Phase 2C supervised geometry control rules out a pure “the cue has
disappeared from memory” explanation. The fixed recurrent state still contains
linearly usable cue information at delay 5 for all three seeds. The Phase 3A
failures therefore arise later: the action-local normalized TD(0) readout does
not reliably acquire that mapping from sampled scalar feedback for the
preregistered reservoir/fixture realizations. Auxiliary probes indicate that
the severity is sensitive to the realization rather than universal. Seed 29 is
the clearest case: its Phase 2C
supervised readout is 200/200 with a stable delay-margin ratio, while its Phase
3A action-value readout reaches only 20/40 at delay 5.

The hidden-centroid distances in the probe are still useful descriptive
statistics: they show that the raw geometry becomes less separated and more
variable at longer delays. They are not an information ceiling, because the
Phase 2C supervised readout remains perfect. The failure is best described as
seed-dependent readout/credit acquisition on a weakly separated representation,
not irreversible memory loss.

This does not prove that TD(0) is the only cause, nor that an eligibility trace
will solve the problem. It does establish that a credit/readout experiment is
justified before changing the recurrent memory substrate.

## Executed diagnostic arms

The diagnostic-only variance decomposition was then run with the same fixed
configuration. It produced the independent artifact
`docs/experiments/phase-3a-failure-attribution.json` (SHA-256
`84b9a85efc5022411ad3e8ee886419d8ad47393186d3d31148bdd3f50ce3452b`). The
baseline reproduces the frozen seed-7 result, and every arm retains a 200/200
supervised reference, including delay 5 at 40/40.

| Arm | Changed lineage | TD(0) post-training | TD(0) delay 4 | TD(0) delay 5 | Supervised reference |
|---|---|---:|---:|---:|---:|
| baseline | none (seed 7) | 181/200 | 28/40 | 34/40 | 200/200 |
| reservoir-17 | reservoir | 200/200 | 40/40 | 40/40 | 200/200 |
| reservoir-29 | reservoir | 177/200 | 37/40 | 20/40 | 200/200 |
| action-RNG-17 | action sampling/reward trajectory | 171/200 | 25/40 | 27/40 | 200/200 |
| action-RNG-29 | action sampling/reward trajectory | 181/200 | 29/40 | 33/40 | 200/200 |
| fixture-17 | fixture realization | 177/200 | 31/40 | 28/40 | 200/200 |
| fixture-29 | fixture realization | 177/200 | 31/40 | 36/40 | 200/200 |

The arms preserve the intended controls: reservoir and fixture changes retain
the baseline action sequence, action-RNG changes alter both sampled actions and
the resulting reward trajectory, and fixture changes regenerate the hidden
states. Thus the probe does not support a pure recurrent-memory explanation;
it supports a realization- and sampling-sensitive interaction between weak raw
geometry and action-local TD(0) credit acquisition. The action-RNG arm measures
the joint effect of behavior sampling and its induced rewards, not an isolated
reward-only perturbation.

## Follow-up experiment

The pre-registered TD(0) versus persistent eligibility-trace comparison was
also run on the same three lineages. It is recorded in
`docs/experiments/phase-3a-credit-comparison.json` (SHA-256
`563987000e8214a18a7e1b967aa3503cf0a6d25722273b3f8706c8d6ee6d474b`). The two
learners consume identical sampled actions and fixtures in every seed.

| Seed | TD(0) post | TD(0) d4 / d5 | TD(lambda) post | TD(lambda) d4 / d5 |
|---:|---:|---:|---:|---:|
| 7 | 181/200 | 28/40 / 34/40 | 175/200 | 29/40 / 39/40 |
| 17 | 199/200 | 40/40 / 39/40 | 199/200 | 40/40 / 39/40 |
| 29 | 177/200 | 37/40 / 20/40 | 179/200 | 35/40 / 24/40 |

The trace improves seed 29/delay 5 by four cases but remains below the fixed
gate, and it lowers seed 7's overall result while introducing a delay-3 miss.
Therefore TD(lambda) is not a general fix under this protocol. The evidence
supports the narrower attribution—credit/readout dynamics interact with the
realization—but does not justify promoting a new learner or changing the
frozen Phase 3A acceptance threshold.

The oracle must remain outside the acceptance benchmark: it is a causal
diagnostic, not a label-free agent policy. No current threshold, seed, reward,
or evidence byte should be changed before that diagnostic is reviewed.

## Scope boundary

Tasks 7–9 remain intentionally unexecuted because the fixed Phase 3A
scientific gate is false. The separate public-API overflow observation for
extreme finite inputs is deferred to a later numerical-contract task and is
not part of this attribution result.
