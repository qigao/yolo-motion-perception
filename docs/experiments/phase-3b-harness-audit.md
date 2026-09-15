# Phase 3B delayed-credit harness audit

Status: **harness invalid for delayed-credit interpretation**. No Phase 3B acceptance artifact is frozen.

## Measured run

- Branch: `experiment/neural-state-machine`
- Source commit: `edb46d1429441d0fe1f546fb9411f22647a2ebfd`
- GitHub Actions run: `34950439392`
- Measurement job: `104320974927`
- Diagnostic payload: 24 records (3 seeds × 2 arms × 4 reward delays)
- Uncommitted diagnostic blob: `3966a4df4901d88b660de435e263eb94fdfb3448`

The full default configuration ran successfully in Python 3.11. The payload was deliberately marked diagnostic-only and `all_passed=false`.

| arm | seed | post-training correct for d_r = 0,1,3,5 | distinct action digests | distinct reward digests | distinct parameter digests |
|---|---:|---:|---:|---:|---:|
| TD(0) | 7 | 181, 181, 181, 181 | 1 | 1 | 1 |
| TD(0) | 17 | 199, 199, 199, 199 | 1 | 1 | 1 |
| TD(0) | 29 | 177, 177, 177, 177 | 1 | 1 | 1 |
| TD(lambda) | 7 | 181, 181, 181, 181 | 1 | 1 | 1 |
| TD(lambda) | 17 | 199, 199, 199, 199 | 1 | 1 | 1 |
| TD(lambda) | 29 | 177, 177, 177, 177 | 1 | 1 | 1 |

All 24 records reported 2,000 queue deliveries, no pending feedback, and repeatability=true.

## Interpretation

The equality across all reward delays is not evidence that delayed credit is solved. The current harness enqueues one reward, advances the queue exactly the selected delay, delivers that reward, and only then selects the next action. No later decision overlaps a pending delivery. Therefore changing d_r only inserts an unobserved wait before the same update; it cannot change action attribution or learning dynamics.

The next implementation must redesign the timeline so delayed deliveries overlap subsequent decisions while preserving the action-to-reward association. Until that protocol is implemented and re-tested, the correct classification is **harness invalid**, not “TD(0) succeeds” or “TD(lambda) adds benefit”.
