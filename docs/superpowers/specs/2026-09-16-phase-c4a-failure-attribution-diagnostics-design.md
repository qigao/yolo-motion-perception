# Phase C4-A Failure-Attribution Diagnostics Design

> Status: **diagnostic design committed for review; implementation and diagnostic measurement are not authorized by this document's existence**.
>
> Repository: `qigao/yolo-motion-perception`
>
> Branch: `experiment/phase-c4-delay-marginalized-credit`
>
> Frozen C4 scientific head: `8ae3154950ed53c4d0a0f555463042ff72674d31`
>
> Frozen C4-A evidence commit: `832c4a6874b87bba1b189e2d3604f66025730bb9`
>
> Current cleanup/reference head: `fe91501ad2baa00b47ee40464aa60f352c3d646d`
>
> Formal head: `8b2180ed24b6ff03db4b927ec29cdd9903b0ccac`
>
> Registered C4-A measurement run: `35105578412`
>
> Date: 2026-09-16
>
> Review boundary: spec only. After approval, write a separate implementation plan. Do not modify frozen C4-A evidence, change the ridge penalty, alter seeds or gates, run C4-B, or perform new diagnostic measurements at this checkpoint.

## 1. Decision and diagnostic question

The registered C4-A result is a valid negative result:

- `formal_valid = true`
- `protocol_valid = true`
- `operator_passed = false`

All three normal rows score `200/200`; every normal cue-delay cell is `40/40`; every reset row is exactly `100/200` with `20/40` in each delay cell. The failure is the shuffled negative control: seed 7 scores `147/200`, seed 17 scores `170/200`, and seed 29 scores `160/200`, while the registered requirement is `<150/200` for every seed.

The diagnostic question is therefore not “why can C4-A solve the task?” It demonstrably can on the registered normal rows. The question is:

> **Why does the fixed anonymous delay-marginalized batch operator retain task-predictive structure after the registered reward shuffle, especially for seeds 17 and 29?**

The objective is an auditable attribution, or an explicit unresolved attribution, of the shuffled-control specificity failure. This diagnostic phase must not be used to tune a replacement C4-A trial or to authorize C4-B.

## 2. Approaches considered

Three approaches were considered.

### 2.1 Retune the operator or gate

Examples include changing ridge `1e-6`, increasing regularization, changing the shuffle threshold, choosing different seeds, altering the delay prior, or changing the representation. This could make the control score lower without explaining the registered failure and would destroy the preregistered interpretation. **Rejected.**

### 2.2 Jump directly to C4-B online behavior

The registered design explicitly requires C4-A to pass before C4-B. Running C4-B after `operator_passed=false` would violate the stop rule and make the experimental sequence uninterpretable. **Rejected.**

### 2.3 Frozen-operator attribution with matched counterfactual diagnostics

Keep the C4-A operator, representation, ridge, seeds, schedules, training length, evaluation fixtures and thresholds fixed. Reconstruct the registered design/target pairs, quantify matrix geometry and target projections, then run a prospectively fixed permutation grid that changes only reward reassignment. **Selected.**

This approach separates four questions that must not be conflated:

1. Is the design matrix itself ill-conditioned or unusually low-rank?
2. Does the shuffled aggregate target still project strongly into task-relevant design directions?
3. Does the fitted shuffled weight vector align with a privileged label-predictive direction even though labels were never used by C4-A fitting?
4. Is the high shuffled score specifically tied to the registered block-10 shuffle structure, or does it persist under global reward permutations?

## 3. Immutable inputs and claim boundary

The following are frozen inputs, not editable diagnostic parameters:

- C4 scientific head: `8ae3154950ed53c4d0a0f555463042ff72674d31`
- C4 formal head: `8b2180ed24b6ff03db4b927ec29cdd9903b0ccac`
- C4-A manifest SHA-256: `a990745cf6f0e8c5fd7ae5d9a7189737b30a40168a4db3df1a6d3043d6d5876a`
- C4-A result SHA-256: `7f398f474dba928d2a6aa993d0801563aa05af20faa8cd9a61ec94a6ca4697a4`
- C4-A provenance SHA-256: `8fb47b5e670754bc44c9ea70dd4cedce40e6f52165923142114577b4aec00351`
- registered seeds: `7, 17, 29`
- hidden size: `64`
- recurrent radius: `0.9`
- real training decisions: `2000`
- public hidden-feedback delay support: `{1,3,5}` with probability `1/3` each
- public drain horizon: exactly 5 clocks
- C4-A design/feedback rows: exactly `2005`
- ridge penalty: exactly `1e-6`, bias penalized
- original evaluation: 200 decisions
- eight secondary evaluation fixture sets per seed: diagnostic only
- normal threshold: at least `180/200`
- each cue-delay threshold: at least `34/40`
- reset threshold: exactly `100/200` and `20/40` per cue delay
- shuffled threshold: strictly below `150/200`

The diagnostic phase may observe privileged labels, source identities and realized delays **only in external observers after the frozen anonymous design/target data have been reconstructed**. Privileged data must never flow into C4-A fitting or alter any C4 learner call.

No diagnostic output changes the recorded C4-A verdict. `operator_passed` remains `false` even if diagnostics reveal a narrow explanation for the failed negative control.

## 4. Diagnostic architecture

Implement later as an additive package, preferably under `src/neural_state_machine/phase_c4a_diagnostics/`, plus a thin driver script. Do not edit the frozen C4-A scientific modules or result files.

Separate five responsibilities:

1. **Replay/integrity driver** — replays frozen C4 protocol inputs and reconstructs registered normal/shuffled C4-A design matrices and aggregate targets.
2. **Geometry observer** — computes rank, singular spectrum, condition measures, leverage and ridge-transfer statistics from the frozen design without changing it.
3. **Association observer** — uses external labels/actions/reward provenance to quantify which target components retain task-relevant structure.
4. **Counterfactual permutation runner** — generates a fixed block-10/global permutation grid, refits the unchanged C4-A ridge operator and evaluates on frozen evaluation fixtures.
5. **Reporter/sealer** — joins diagnostic records after fitting/evaluation ends, writes canonical evidence and prevents observer data from feeding back into any fit.

All NumPy arrays stored by observers must be immutable copies or detached digests. No observer is allowed to share writable storage with learner/protocol state.

## 5. D0 — replay and integrity gate

Before any new diagnostic condition, reproduce the registered C4-A rows for all three seeds.

Required integrity checks:

- verify the frozen C4-A manifest/result/provenance SHA-256 values above;
- verify the formal contract and all frozen C3 hashes carried by the C4-A manifest;
- reproduce the exact registered action, fixture, hidden-delay and evaluation lineages;
- require exactly `2000` real scalar calls plus `5` public-horizon drain calls;
- independently reconstruct the `2005 x P` C4-A design matrix, where `P` is the flattened action-blocked feature dimension;
- independently reconstruct the `2005` aggregate scalar target values;
- prove registered normal/shuffled action and schedule lineages are identical where required;
- record exact digests for normal and shuffled design matrices and aggregate targets;
- verify the reproduced registered scores are exactly the frozen scores: `200/147`, `200/170`, `200/160` for normal/shuffled across seeds 7/17/29, with reset exactly `100/200`;
- verify the diagnostic replay does not change any frozen C4-A file byte.

A D0 mismatch invalidates the diagnostic run. Stop without geometry or behavioral interpretation.

A central D0 question is whether normal and shuffled C4-A fits use byte-identical design matrices. The current protocol suggests they should because reward reassignment does not define the action/hidden-state lineages, but this must be established from replay evidence rather than assumed. If the design matrices differ, all later attribution must explicitly condition on that fact.

## 6. D1 — design-matrix geometry

For each seed and each registered design matrix `Z`, record:

- matrix shape;
- exact rank using NumPy/LAPACK rank with a prospectively fixed tolerance `tol = max(m,n) * eps * sigma_max`;
- nullity `n - rank`;
- singular values in descending order;
- `sigma_max`, smallest retained singular value and condition number;
- Frobenius norm;
- stable rank `||Z||_F^2 / ||Z||_2^2`;
- ridge effective degrees of freedom `sum(sigma_i^2 / (sigma_i^2 + 1e-6))`;
- row leverage diagonal for the ridge hat matrix, summarized by min/median/max and nearest-rank p10/p90;
- per-action-block column norms and cross-block Gram terms;
- bias-column contribution separately from hidden-feature columns.

Do not call a large condition number “the cause” by itself. Geometry becomes explanatory only when connected to the target and fitted weights in D2/D3.

If normal and shuffled `Z` are byte-identical, compute the geometry once and bind both target conditions to the same matrix digest. If they differ, compute and report both without averaging them together.

## 7. D2 — target structure and projection diagnostics

Let `F_normal` and `F_shuffled` be the registered `2005` aggregate targets for one seed and let `Z` be the corresponding frozen design.

### 7.1 Direct target association

For real-decision clocks only, report associations between each target and observer-only variables:

- current action;
- correct action label;
- action correctness;
- cue delay;
- hidden feedback delay category;
- block position `t mod 10`;
- previous and future correct-action labels for lags `1..10` where valid;
- previous and future action correctness for lags `1..10`;
- normal latent reward and shuffled donor reward where provenance exists.

For scalar/binary variables report counts, means, centered Pearson correlation when variance is non-zero, and `null` with a reason when variance is zero. For categorical cells report counts and means; do not silently convert absent cells to zeros.

Separate real-decision calls from drain calls. Separate no-arrival zeros, one-source buckets and collision buckets in observer output. These categories never alter C4-A fitting.

### 7.2 Projection into the C4-A design space

Using a stable SVD/least-squares formulation, decompose each target into:

```text
F = F_parallel + F_perp
F_parallel = projection onto col(Z)
F_perp     = orthogonal residual
```

Report:

- `||F_parallel|| / ||F||`;
- `||F_perp|| / ||F||`;
- centered correlation between `F_parallel` and `F`;
- ridge-fitted residual norm;
- normal-vs-shuffled cosine/correlation for `F`, `F_parallel`, and residuals.

Do not use an explicit matrix inverse.

### 7.3 Task-relevant privileged direction

Construct a separate supervised ridge reference from the same decision-time features and training labels, with the same fixed `1e-6` penalty and no evaluation-label fitting. This reference is diagnostic-only.

Compare C4-A normal/shuffled fitted weights with the supervised reference using:

- global cosine similarity;
- per-action-block cosine similarity;
- norm ratios;
- projection magnitude onto the supervised weight direction;
- evaluation margin correlation on the original plus eight secondary evaluation sets.

A strong shuffled alignment with the supervised direction shows that the shuffled aggregate target still induces a task-relevant readout under this design. It does **not** prove data leakage or that the supervised direction caused the alignment.

## 8. D3 — weight and score decomposition

For every registered normal/shuffled C4-A fit, decompose the final evaluation margin into:

- hidden-feature contribution;
- bias contribution;
- action-block 0 contribution;
- action-block 1 contribution.

Report contribution distributions for correct and incorrect examples and by cue delay. Recompute the final score from the summed contributions and require exact agreement with the ordinary evaluation decision for every fixture.

Also report:

- `||W_normal||`, `||W_shuffled||`, `||W_shuffled - W_normal||`;
- cosine between normal and shuffled weights;
- per-action-block weight norm and bias term;
- original and secondary-set scores for both fits;
- margin quantiles and number of margins near zero under a prospectively fixed absolute threshold `1e-9`.

No ablated component is promoted as a new model. If a diagnostic reports “bias-only score” or “hidden-only score”, it must be clearly labeled as a post-hoc readout decomposition, not a retrained policy.

## 9. D4 — fixed permutation grid

After D0 passes, run a fixed diagnostic permutation grid. This is new diagnostic evidence, not a rerun of the registered C4-A trial.

For each seed, generate exactly 32 `block10` permutations and 32 `global` permutations. Use a new isolated RNG lineage:

```text
PCG64(SeedSequence([seed, 0x43344144, mode, replicate]))
mode 0 = block10
mode 1 = global
replicate = 0..31
```

Do not consume the registered shuffle RNG. Do not reject or resample permutations with fixed points, inconvenient scores or unusual target correlations.

For every replicate:

- keep the frozen reservoir, fixtures, actions, hidden-delay schedule, design matrix, ridge penalty and training length fixed;
- permute latent reward values only;
- rebuild aggregate scalar feedback through the same public delivery clocks including all five drain clocks;
- fit the unchanged C4-A ridge operator;
- evaluate on the original evaluation set plus all eight frozen secondary evaluation sets;
- record the target/design projection metrics from D2 and weight-alignment metrics from D3.

For each seed/mode report all 32 rows plus descriptive summaries: mean, median, min, max, nearest-rank p10/p90, count `<150`, count `>=150`, count `>=170`, and the registered shuffled score as a separately marked post-observation reference. Do not present the registered score's empirical rank as a calibrated p-value.

The principal comparison is block10 versus global. A difference implicates local exchangeability/block structure as a candidate source of residual predictive structure, but it is not a pure causal intervention because block10 and global permutations change multiple temporal statistics together.

## 10. D5 — matched counterfactual attribution table

The final diagnostic report must classify evidence by mechanism, not simply list scores.

Required rows:

| Candidate mechanism | Required evidence |
|---|---|
| Design ill-conditioning / low effective rank | D1 spectrum, condition, leverage, effective dof |
| Shuffled target still lies in task-relevant design space | D2 target projection + supervised-direction alignment |
| Bias/action-block shortcut | D3 component decomposition |
| Local block structure | D4 block10 vs global distributions |
| Simple overfit to one evaluation fixture | original vs eight secondary evaluation sets |
| Multiple mechanisms / unresolved | conflicting or insufficient diagnostics explicitly retained |

No single numeric threshold automatically proves causality. The report must distinguish `observed association`, `matched counterfactual evidence`, and `interpretation`.

## 11. Fixed diagnostic outcomes and interpretation

The following interpretations are prospective.

### Outcome A — block-local structure is dominant

If block10 replicates frequently remain `>=150` while global permutations mostly fall below `150`, and this change tracks a loss of target/design or supervised-direction alignment, conclude only:

> residual negative-control predictivity is strongly associated with local block-preserving reward structure under the tested operator.

Do not claim that block structure is the only cause.

### Outcome B — global permutations also remain predictive

If global permutations frequently remain `>=150`, investigate D1-D3 as the primary attribution. A high target projection and high supervised-direction alignment would indicate that the fixed design/operator can turn broad reward permutations into task-relevant readouts under this frozen feature/action structure.

This is not information-theoretic evidence that anonymity is impossible.

### Outcome C — registered shuffled rows are unusual but not structurally explained

If both permutation modes usually fall below `150` and the registered seed 17/29 rows remain outliers without a matching geometry/association signature, freeze the attribution as unresolved. Do not tune a new permutation seed or reinterpret the original control post hoc.

### Outcome D — score is evaluation-realization-specific

If the registered shuffled fit is high on the original evaluation but collapses across the eight secondary sets, classify the result as fixture-sensitive specificity failure. This does not retroactively pass C4-A because the registered gate used the original frozen evaluation.

### Outcome E — multiple mechanisms

If block structure, design geometry and bias/action-block contributions all contribute, report them jointly. Do not force a single-cause narrative.

## 12. CI and evidence lifecycle

Use distinct CI identities, matching the established C4-A lifecycle.

### Stage 1 — permanent `ci`

Add only deterministic diagnostic unit tests, Ruff and a no-result preflight. Existing permanent C3/C4 gates remain unchanged.

### Stage 2 — `Phase C4-A Attribution Preflight`

A dedicated CI verifies:

- frozen C4-A evidence hashes;
- frozen C4 scientific head and formal head;
- locked Python 3.12.14 environment;
- D0 deterministic replay on a tiny/unregistered fixture and registered no-fit manifest construction;
- no diagnostic result exists.

It uploads a prospective attribution manifest only.

### Stage 3 — `Phase C4-A Attribution Measurement`

A one-shot workflow checks out the frozen diagnostic science head, recreates the exact environment, regenerates the prospective manifest byte-for-byte, then runs D0-D5. It uploads canonical result/provenance/traces. Large matrices/traces may be workflow artifacts with SHA-256 references rather than committed JSON.

### Stage 4 — `Phase C4-A Attribution Freeze`

A separate no-refit workflow downloads the already verified measurement artifact by run ID, verifies exact hashes, commits only the result/report/provenance/manifest files, then is deleted separately.

Measurement and freeze workflows must be removed after use. Their cleanup commits run only permanent CI and may not refit any diagnostic row.

## 13. Evidence layout

Use:

```text
docs/experiments/phase-c4a-failure-attribution/
  manifest.json
  result.json
  provenance.json
  report.md
  trace-index.json
```

Large per-replicate tables, SVD arrays and observer traces may remain GitHub artifacts if `trace-index.json` records filenames, sizes and SHA-256 values.

Canonical JSON rules remain the same as C4 evidence: sorted keys, compact separators, UTF-8, newline terminated, `allow_nan=false`, no overwrite, no symlinks.

## 14. Non-goals

This phase does not:

- modify or rerun the registered C4-A verdict;
- authorize C4-B;
- change ridge `1e-6`;
- choose a new representation;
- tune thresholds or seeds;
- search for a shuffle that produces a desired score;
- fit evaluation labels;
- claim information-theoretic impossibility;
- claim that any single association proves leakage;
- modify frozen C3 or C4-A scientific files.

A future mechanism change requires a new design and a new prospective experimental phase.

## 15. Acceptance criteria for the diagnostic implementation

Before any registered diagnostic measurement:

1. frozen C4-A manifest/result/provenance hashes verify exactly;
2. D0 reproduces registered C4-A scores and `2005` design rows exactly;
3. no frozen C3/C4-A scientific file changes;
4. normal/shuffled design identity is explicitly tested rather than assumed;
5. D1 geometry metrics have fixed formulas/tolerances;
6. D2 target projections use stable decompositions, no explicit inverse;
7. supervised labels remain observer/reference-only;
8. D3 margin decomposition recomposes every prediction exactly;
9. D4 uses exactly 32 block10 + 32 global permutations per seed with the registered new RNG lineage above;
10. all eight secondary evaluation sets remain diagnostic only;
11. permanent/preflight/measurement/freeze CI stages have distinct names;
12. a preflight environment or integrity mismatch produces no diagnostic scientific result;
13. C4-B remains absent from all workflows and result schemas.

## 16. Review gate

Approval of this spec authorizes only creation of a separate implementation plan. It does **not** authorize diagnostic implementation or measurement.

The implementation plan must preserve the staged CI lifecycle and include an explicit mandatory stop before `Phase C4-A Attribution Measurement`.