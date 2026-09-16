# Phase 3C — Anonymous Temporal Credit: Frozen Registered Result

## Status

This is the first registered Phase 3C behavioral result. It is **formal/protocol valid but behavior failed**. Frozen evidence: `phase-3c-anonymous-temporal-credit.json`, SHA-256 `53b52fb6716daaceeb68b4e5c78f33333c0076b0d727462aa7265b0887798263`.

No scientific parameter, threshold, seed, delay support, fixture lineage, or learner rule was changed after observing the result.

## Provenance

- measured Python implementation head: `4d13a55545e67aa97ce1e67fd40aaeef044b19a8`;
- one-shot registered measurement workflow run: `35046817410`;
- pre-measurement Task 9 exact-head CI: `35046158170` (#393);
- formal Lean Gate F head: `3de297cee2a94a7fc309531334f720f1b34467c9`;
- Lean exact-head proof run: `35037145100` (#1927);
- Phase 3B scientific base: `a5ab079d56fe569aded44348f9591d226ce83009`;
- registered seeds: `[7, 17, 29]`;
- hidden reward-delay support: `[1, 3, 5]`;
- `formal_valid=true`, `protocol_valid=true`, `behavior_passed=false`, `all_passed=false`.

Bound formal theorem surface:

- `NarrativeDynamics.AnonymousTemporalCredit.aggregate_view_source_noninterference`
- `NarrativeDynamics.AnonymousTemporalCredit.aggregate_conservation`
- `NarrativeDynamics.AnonymousTemporalCredit.learner_view_source_relabel_invariant`
- `NarrativeDynamics.AnonymousTemporalCredit.eligibility_historical_coefficient`
- `NarrativeDynamics.AnonymousTemporalCredit.immediate_reduction_to_phase3a`

Lean establishes the mathematical contract, not NumPy floating-point execution.

## Fixed Gate B

A row passes only with post-training `>=180/200`, every cue-delay `>=34/40`, reset exactly `100/200` with every reset delay exactly `20/40`, and shuffled control `<150/200`.

## Behavioral result

| Seed | Arm | Post | Delay 1 / 2 / 3 / 4 / 5 | Reset | Shuffled | Gate B |
|---:|---|---:|---|---:|---:|---|
| 7 | `td0` | 83/200 | 20/40 / 5/40 / 20/40 / 11/40 / 27/40 | 100/200 | 100/200 | `false` |
| 7 | `eligibility` | 104/200 | 20/40 / 24/40 / 20/40 / 20/40 / 20/40 | 100/200 | 100/200 | `false` |
| 17 | `td0` | 121/200 | 40/40 / 20/40 / 20/40 / 21/40 / 20/40 | 100/200 | 197/200 | `false` |
| 17 | `eligibility` | 170/200 | 40/40 / 39/40 / 40/40 / 23/40 / 28/40 | 100/200 | 100/200 | `false` |
| 29 | `td0` | 100/200 | 20/40 / 20/40 / 20/40 / 20/40 / 20/40 | 100/200 | 100/200 | `false` |
| 29 | `eligibility` | 100/200 | 20/40 / 20/40 / 20/40 / 20/40 / 20/40 | 100/200 | 100/200 | `false` |

TD(0) totals `304/600`; eligibility totals `374/600`, a descriptive +70 correct decisions (+11.67 percentage points). Eligibility is higher at seeds 7 and 17 and tied at seed 29, but **no registered arm/seed row passes Gate B**. Seed 17 TD(0) shuffled control is `197/200`, above the preregistered rejection boundary; eligibility shuffled control is `100/200` for all three seeds.

Preregistered interpretation:

> **Arm B improves over Arm A but both fail the fixed quality gate.** Historical eligibility credit helped descriptively but was insufficient to solve the registered task. Phase 3C is not successful under the registered mechanisms.

## Gate P structural evidence

Both arms use the same fixtures, action lineage, latent rewards, hidden-delay schedule, due-step schedule, multiplicity structure, aggregate feedback stream, and learner scalar-call stream for each seed.

| Seed | Delay histogram | Inversions | Collision steps | Multiplicity histogram | Drain calls | Latent/aggregate sum |
|---:|---|---:|---:|---|---:|---:|
| 7 | 1:619, 3:685, 5:696 | 1082 | 516 | 1:904, 2:452, 3:64 | 5 | -54.0 |
| 17 | 1:637, 3:691, 5:672 | 1124 | 506 | 1:924, 2:442, 3:64 | 5 | 16.0 |
| 29 | 1:673, 3:656, 5:671 | 1109 | 504 | 1:926, 2:438, 3:66 | 1 | -4.0 |

For every registered seed and arm: action count = latent count = delivered count = 2000; final pending count = 0; aggregate conservation is exact; source relabel and hidden multiplicity invariants are true; immediate Phase 3A continuity and repeatability are true; Arm A trace reset count is 0 and Arm B is 1. Trace coefficient probes are ages 0/1/3/5 = `1`, `0.72`, `0.373248`, `0.1934917632` modulo the preregistered Python representation tolerance.

| Seed | Delay digest | Due-step digest | Multiplicity digest | Learner-call digest |
|---:|---|---|---|---|
| 7 | `93b7ac757e1f6318ebdbe88435e526b36ecad9aa8f8860f31314127123d7a4cb` | `f6b90bd77e4f663012676315a013e073fc0dea04a5ab2a425d6939097572734e` | `592fb10c22325cad51ac5e7c88e38ac4be6d03175fa9a1576e2cf30e08813693` | `4d5114edaae437ecc979de083f2a70d34a372f5603331ce700614db3861cb653` |
| 17 | `6a42e21c7f692c35ee77e9f73560f3825c83d5321d29adb71e0c507c3e63cb22` | `9d14991f7e3b94f3b3fc0f35774811b774dae2d83d681094b697e16fbbe3a7a5` | `c2f89105ecd49ef122ec74e6766f5ba9f99749fed57063b57185db5c9d1dbb4e` | `5761dff3dec9134378c8fc56e9afaa84b2c41efc263599c1333fddab4159f283` |
| 29 | `d25058004b12b75c52ba08927949429bea140b23822d9983ed387a145385a7d2` | `9cb6b8fd3888ba50a0494fe9f5f2976e65a6f088dc1d54cac25590cca6535228` | `336da4e3803d840fe5d4965ac216a0750e3a0404170ead943403bd880e6b0f19` | `60b6bbaec5602c3b5467a18d0829f1f23454ee9f2ae93a4867804c32bb438e0e` |

## Interpretation boundary

The bounded conclusion is that the registered normalized eligibility mechanism produced higher aggregate post-training accuracy than current-step TD(0), but was insufficient to satisfy the fixed behavioral quality gate. This does not establish general causal discovery, standard TD(lambda) equivalence, convergence guarantees, general reinforcement learning, a Lean proof of NumPy, a YOLO result, or production readiness. The negative result is frozen rather than tuned away.
