# Phase 3B corrected delayed-credit evidence

## Decision

**Protocol validity: PASS. Behavioral Arm A gate: FAIL.**

This is the first frozen measurement produced by the corrected overlapping
Phase 3B protocol. The measured artifact is
`docs/experiments/phase-3b-delayed-credit.json` with SHA-256 `dbe263aa916ca0e2bbf3281b20d56de7d6cee70c38f08146c96eb9987e024074`.
The scientific implementation measured here is commit `42a4b26f80969a0abd6f878248ec860365b699c1`; that
exact head passed the full Python 3.10/3.11/3.12 CI matrix and Gate P before
this evidence was frozen.

`protocol_valid=true` means the measurement is interpretable as delayed credit.
`behavior_passed=false` is therefore a valid negative behavioral result, not a
harness failure. The top-level behavior gate aggregates only registered delayed
conditions `d_r in {1, 3, 5}`; `d_r=0` is the Phase 3A continuity control.

## Registered protocol

- seeds: `[7, 17, 29]`
- training decisions: `2000`
- evaluation episodes: `200`
- cue-to-decision delays: `[1, 2, 3, 4, 5]`
- action-to-reward delays: `[0, 1, 3, 5]`
- hidden size: `64`
- recurrent radius: `0.9`
- TD(0) step size: `0.1`
- checkpoint interval: `100`
- frozen Phase 3A evidence SHA-256: `897f1917d3d950f64439ffc0c80aa83aa4c5b940cc5bf41dfbbb89f422da0295`

Corrected Phase 3B contains **only Arm A / TD(0)**. It makes no TD(lambda)
comparison; overlapping eligibility-trace/reset semantics require a separate
design before such an arm can be registered.

## Gate P and exact zero-delay continuity

Every registered row delivered exactly `2000/2000` rewards, finished with an
empty environment queue and zero unresolved learner credits, and reproduced
independently. Normal and shuffled controls used identical action lineages and
identical delivery timelines.

For `d_r=0`, the harness exactly reproduces frozen Phase 3A TD(0): action
sequence and digest, reward sequence and digest, final parameter digest,
post-training counts, per-cue-delay counts, state-reset counts, and training and
evaluation fixture digests all match. This exact continuity is part of
`protocol_valid=true`, not a tolerance-based comparison.

The structural delayed-reward audit is:

| `d_r` | exact lag histogram | max pending before delivery | max pending after delivery | decisions with prior feedback pending | terminal drain |
|---:|---|---:|---:|---:|---:|
| 0 | `0 × 2000` | 1 | 0 | 0 | 0 |
| 1 | `1 × 2000` | 2 | 1 | 1999 | 1 |
| 3 | `3 × 2000` | 4 | 3 | 1999 | 3 |
| 5 | `5 × 2000` | 6 | 5 | 1999 | 5 |

Thus non-zero `d_r` now necessarily overlaps later real decisions; the original
serialized-harness failure is no longer present.

## Arm A results

The fixed per-row behavioral gate is: post-training at least `180/200`, every
cue-to-decision delay at least `34/40`, state reset exactly `100/200` and
`20/40` per cue delay, and shuffled control below `150/200`.

| seed | `d_r` | post | reset | shuffled | row gate | lag | queue high-water | terminal drain |
|---:|---:|---:|---:|---:|:---:|---:|---:|---:|
| 7 | 0 | 181/200 | 100/200 | 100/200 | FAIL | 0 | 1 | 0 |
| 7 | 1 | 181/200 | 100/200 | 100/200 | FAIL | 1 | 2 | 1 |
| 7 | 3 | 181/200 | 100/200 | 100/200 | FAIL | 3 | 4 | 3 |
| 7 | 5 | 182/200 | 100/200 | 100/200 | FAIL | 5 | 6 | 5 |
| 17 | 0 | 199/200 | 100/200 | 100/200 | PASS | 0 | 1 | 0 |
| 17 | 1 | 199/200 | 100/200 | 100/200 | PASS | 1 | 2 | 1 |
| 17 | 3 | 199/200 | 100/200 | 100/200 | PASS | 3 | 4 | 3 |
| 17 | 5 | 199/200 | 100/200 | 100/200 | PASS | 5 | 6 | 5 |
| 29 | 0 | 177/200 | 100/200 | 138/200 | FAIL | 0 | 1 | 0 |
| 29 | 1 | 177/200 | 100/200 | 136/200 | FAIL | 1 | 2 | 1 |
| 29 | 3 | 177/200 | 100/200 | 151/200 | FAIL | 3 | 4 | 3 |
| 29 | 5 | 177/200 | 100/200 | 135/200 | FAIL | 5 | 6 | 5 |

Detailed cue-delay counts (`1 / 2 / 3 / 4 / 5`):

| seed | `d_r` | post-training | shuffled control |
|---:|---:|---|---|
| 7 | 0 | 40/40 / 40/40 / 39/40 / 28/40 / 34/40 | 20/40 / 20/40 / 20/40 / 20/40 / 20/40 |
| 7 | 1 | 40/40 / 40/40 / 39/40 / 28/40 / 34/40 | 20/40 / 20/40 / 20/40 / 20/40 / 20/40 |
| 7 | 3 | 40/40 / 40/40 / 39/40 / 27/40 / 35/40 | 20/40 / 20/40 / 20/40 / 20/40 / 20/40 |
| 7 | 5 | 40/40 / 40/40 / 39/40 / 28/40 / 35/40 | 20/40 / 20/40 / 20/40 / 20/40 / 20/40 |
| 17 | 0 | 40/40 / 40/40 / 40/40 / 40/40 / 39/40 | 20/40 / 20/40 / 20/40 / 20/40 / 20/40 |
| 17 | 1 | 40/40 / 40/40 / 40/40 / 40/40 / 39/40 | 20/40 / 20/40 / 20/40 / 20/40 / 20/40 |
| 17 | 3 | 40/40 / 40/40 / 40/40 / 40/40 / 39/40 | 20/40 / 20/40 / 20/40 / 20/40 / 20/40 |
| 17 | 5 | 40/40 / 40/40 / 40/40 / 40/40 / 39/40 | 20/40 / 20/40 / 20/40 / 20/40 / 20/40 |
| 29 | 0 | 40/40 / 40/40 / 40/40 / 37/40 / 20/40 | 21/40 / 27/40 / 40/40 / 30/40 / 20/40 |
| 29 | 1 | 40/40 / 40/40 / 40/40 / 37/40 / 20/40 | 22/40 / 27/40 / 40/40 / 27/40 / 20/40 |
| 29 | 3 | 40/40 / 40/40 / 40/40 / 37/40 / 20/40 | 21/40 / 34/40 / 40/40 / 36/40 / 20/40 |
| 29 | 5 | 40/40 / 40/40 / 40/40 / 37/40 / 20/40 | 33/40 / 23/40 / 37/40 / 22/40 / 20/40 |

Seed 17 passes every registered delay. Seed 7 misses the per-cue threshold at
longer cue delays despite overall `181–182/200`. Seed 29 remains `177/200` at
all four reward delays and misses cue delay 5 (`20/40`); its `d_r=3` shuffled
control is also `151/200`, just above the fixed `<150/200` control threshold.
Because seeds 7 and 29 fail registered delayed conditions, the pre-registered
Arm A aggregate is false.

## Interpretation boundary

The corrected experiment establishes that reward delay is now real and
auditable. It does **not** establish a successful delayed-credit learner across
the registered seeds. Small changes across `d_r` (for example seed 7 moving
from `181/200` to `182/200`) are observations only; this evidence does not
attribute them to a mechanism.

Action, reward, and fixture lineages are intentionally allowed to remain equal
across `d_r`. Delay is proven by the structural timeline audit above, not by
forcing behavioral or digest divergence. Environment-local floating-point
parameter digests are retained in the artifact for same-environment integrity;
the committed-artifact verifier compares the portable protocol/behavior
projection across supported Python/NumPy environments.

No tuning, alternative step size, reward shaping, TD(lambda), replay, or other
repair was attempted after observing this result.
