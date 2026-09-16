# Phase 3C Failure-Attribution Diagnostic — Registered Result

> Status: **registered diagnostic valid; causal attribution remains bounded and multi-factor.**
> This evidence does not replace or refreeze the original Phase 3C behavioral result.

## Provenance

- Reviewed scientific implementation: `3934360384483a50f3132badf709bec2e598d453`
- Registered manifest SHA-256: `7e4df48a0d73fcdad7de9dc0c8e8a15e64f041388a5c6044dd9fa79b938ebc9b`
- Valid measurement workflow run: `35078403826`
- Attempt ID: `task12-20260916-a`
- Evidence artifact: `phase3c-diagnostics-task12-20260916-a-env-audit` (`10439985654`)
- Evidence artifact SHA-256: `939178582c3c997bbf4ddd72942e26200f10e98cb42bb9c8b7ab1bff3dea015c`
- Models: `405`; main scores: `3645`; reset scores: `189`; reverse checks: `108`; drain scores: `558`.
- All `108` reverse-order checks are invariant. Registered verifier passed after measurement.

An earlier execution attempt (`35078223151`) stopped before fitting because the runtime environment did not equal the sealed environment. The valid run above reported no sealed/current environment-field differences and then completed the registered measurement.

## 1. The seed-17 TD(0) shuffled anomaly is real across held-out fixtures

Original and eight additional evaluation-set scores for the frozen seed-17 TD(0) shuffled model:

```text
197, 199, 196, 196, 200, 200, 197, 197, 196
```

The mean is `197.56/200`, range `196..200`. Therefore the original `197/200` observation is not an artifact of the one original evaluation fixture.

For comparison, the seed-17 TD(0) normal model is:

```text
121, 120, 121, 122, 121, 123, 121, 120, 122
```

with mean `121.22/200`.

## 2. Local block structure matters descriptively, but does not fully explain 197/200

For seed 17 / TD(0), 32 fixed permutation replicates per mode on the original evaluation set:

| Mode | Mean | Median | Min | Max | >=150 | >=197 |
|---|---:|---:|---:|---:|---:|---:|
| `block10` | 130.75 | 122.5 | 93 | 189 | 10/32 | 0/32 |
| `global` | 105.56 | 100.0 | 46 | 194 | 3/32 | 0/32 |

`block10` therefore retains substantially more high-score structure than `global` in this fixed diagnostic grid. The registered original `197/200` exceeds every one of the 32 block-10 replicates (`max=189`) and every one of the 32 global replicates (`max=194`). This rank is descriptive, not a calibrated p-value, because the diagnostic was selected after observing the anomaly.

## 3. Exact donor realignment is not a sufficient explanation

Original shuffled TD(0) provenance:

| Seed | Fixed points `pi(j)=j` | Current-decision donor matches `pi(j)=due_j` | Past / current / future donor vs delivery |
|---:|---:|---:|---|
| 7 | 197 | 131 | 1412 / 131 / 457 |
| 17 | 196 | 111 | 1431 / 111 / 458 |
| 29 | 201 | 146 | 1401 / 146 / 453 |

Notably, seed 17 has *fewer* exact current-decision donor matches than seeds 7 and 29, despite its much higher frozen shuffled score. Also, global replicate 28 scores `194/200` with `0` fixed points and only `2` current-decision donor matches. Thus `pi(j)=due_j` realignment/fixed points cannot be the whole mechanism.

## 4. Representation remains usable; source identity alone is not a complete repair

Across the original plus eight held-out evaluation sets:

| Seed | Supervised ridge | Immediate identified | Source-visible delayed |
|---:|---:|---:|---:|
| 7 | 200.00 (range 200..200) | 183.22 (range 179..188) | 182.33 (range 177..188) |
| 17 | 200.00 (range 200..200) | 199.78 (range 199..200) | 199.78 (range 199..200) |
| 29 | 200.00 (range 200..200) | 178.00 (range 175..181) | 174.11 (range 170..177) |

The supervised ridge reference is exactly `200/200` on all 27 seed/evaluation combinations, so the recurrent representation contains a linearly usable solution for this fixed task. However, source-visible delayed credit does not uniformly outperform the immediate identified readout and remains below the ridge reference at seeds 7 and 29. Therefore source identity is not by itself a complete solution; readout/credit-acquisition limitations coexist.

## 5. Anonymous eligibility carries a large non-source history component

D3 decomposes each eligibility update into history positions corresponding to delivered reward sources versus all other retained history. Summing the per-step update-norm magnitudes:

| Seed | Condition | Source norm sum | Other-history norm sum | Other/source |
|---:|---|---:|---:|---:|
| 7 | `normal` | 22.17 | 110.38 | 4.98x |
| 7 | `original_shuffle` | 22.64 | 111.30 | 4.92x |
| 17 | `normal` | 25.01 | 115.00 | 4.60x |
| 17 | `original_shuffle` | 26.79 | 123.34 | 4.60x |
| 29 | `normal` | 23.84 | 113.16 | 4.75x |
| 29 | `original_shuffle` | 23.75 | 113.26 | 4.77x |

The other-history component is about `4.60x–4.98x` the source-history magnitude on these registered trajectories. This is direct accounting evidence of substantial history interference; it is not, by itself, proof that this interference causes every behavioral miss.

The accounting algebra itself closes numerically: maximum eligibility reconstruction residual `1.11e-16`, maximum update reconstruction residual `2.78e-17`, maximum drain residual `4.16e-17`.

## 6. Terminal drain materially changes eligibility models

Nine-evaluation-set mean score before terminal drain versus after `end_run`:

| Seed | Condition | Pre-drain mean | Final mean | Delta |
|---:|---|---:|---:|---:|
| 7 | `normal` | 121.22 | 105.67 | -15.56 |
| 7 | `original_shuffle` | 89.44 | 100.00 | +10.56 |
| 17 | `normal` | 180.11 | 169.67 | -10.44 |
| 17 | `original_shuffle` | 99.56 | 100.00 | +0.44 |
| 29 | `normal` | 100.00 | 100.00 | +0.00 |
| 29 | `original_shuffle` | 100.00 | 100.00 | +0.00 |

At seed 17 normal, the original evaluation trajectory moves `179 -> 146 -> 188 -> 173 -> 186 -> 170` across drain updates before ending at `170`. At seed 7 normal the final drain effect is also strongly negative, while seed 7 shuffled is strongly positive. Terminal reward delivery therefore has a material effect on the final eligibility readout for some seeds/conditions.

## Bounded attribution

The registered evidence supports the following bounded interpretation:

1. **Control residual structure exists.** Local block-10 permutation preserves more high-score behavior than global permutation at seed 17 TD(0).
2. **The 197/200 anomaly is not explained by simple exact donor alignment.** Exact current-donor matches are not elevated at seed 17, and near-perfect global-shuffle behavior can occur with essentially no exact matches.
3. **Representation is not the primary missing ingredient on this task.** A fixed supervised ridge readout solves every registered evaluation set.
4. **Source identity alone is insufficient.** Identified/source-visible normalized action-value references still show seed-dependent readout limitations.
5. **Anonymous eligibility introduces substantial history interaction and drain sensitivity.** These are concrete candidate failure mechanisms.

These findings support a **multi-factor** failure picture: local control structure, realization-sensitive normalized readout/credit acquisition, and anonymous-history/prediction-target interaction can coexist. The experiment does **not** establish a single causal leak or prove that one mechanism alone explains the original seed-17 result.

## Original Phase 3C status is unchanged

`formal_valid=true`, `protocol_valid=true`, `behavior_passed=false`, `all_passed=false`. This diagnostic does not tune thresholds, replace the learner, or refreeze the original behavioral artifact.
