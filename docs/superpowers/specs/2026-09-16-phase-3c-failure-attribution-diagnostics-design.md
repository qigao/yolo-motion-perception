# Phase 3C Failure-Attribution Diagnostics Design

> Status: **diagnostic design committed for review; implementation and measurement are not authorized by this document's existence**.
>
> Repository: `qigao/yolo-motion-perception`
>
> Branch: `experiment/phase3c-anonymous-temporal-credit`
>
> Scientific/code reference: `55a7ba59ac301975f3df996636f9a596e90bc730`
>
> Date: 2026-09-16
>
> Review boundary: spec only. After approval, write a separate implementation plan and obtain its execution approval. Do not implement a new learner, run new diagnostics, merge branches, or overwrite evidence at this checkpoint.

## 1. Decision and question

Investigate the registered Phase 3C failure before changing its mechanism. The first question is why seed 17's current-step TD(0) achieves `197/200` after latent rewards are shuffled, compared with `121/200` under normal rewards. The objective is **an auditable explanation or an explicit unresolved attribution**, not a lower control score or a green behavioral gate.

This is a post-result diagnostic design. It prospectively fixes the additional measurements below, but it is not a blind preregistration of the already observed Phase 3C result. New results remain diagnostic evidence, not a replacement Phase 3C trial.

Three approaches were considered. Retuning decay/step size or adding a stronger learner could change performance without explaining the anomalous control. More trace-coefficient proofs alone would not establish that the prediction target matches delivered feedback. The selected approach is a bounded, non-mutating audit followed by fixed permutation diagnostics, information-privileged references, and update accounting. No mechanism improvement is selected in advance.

## 2. Immutable inputs and claim boundaries

The source documents and code are listed in Section 13. At the reference commit:

| Input | Recorded identity |
|---|---|
| Phase 3C evidence | `docs/experiments/phase-3c-anonymous-temporal-credit.json` |
| Evidence Git blob | `61e7d1e3253d552665e9b36d1af8248521ec2fc5` |
| Evidence SHA-256 reported by its frozen report | `53b52fb6716daaceeb68b4e5c78f33333c0076b0d727462aa7265b0887798263` |
| Original measurement implementation | `4d13a55545e67aa97ce1e67fd40aaeef044b19a8` |
| Original measurement run | `35046817410` |
| Replay-verifier repair / verified code base | `55a7ba59ac301975f3df996636f9a596e90bc730` |
| Existing Python 3.12 CI for that code base | run `35052025233` (#414) |
| Formal contract head recorded by the report | `3de297cee2a94a7fc309531334f720f1b34467c9` |

The diagnostic implementation must compute the actual evidence SHA-256 before consuming it; the table is provenance, not a substitute for that check. Record a before/after manifest for every existing `docs/experiments` file, the original Phase 3C spec, and all reused scientific modules. Nothing in that manifest may change during diagnostics.

The frozen facts are:

| Seed | TD(0) normal | Eligibility normal | TD(0) shuffled | Eligibility shuffled |
|---:|---:|---:|---:|---:|
| 7 | 83/200 | 104/200 | 100/200 | 100/200 |
| 17 | 121/200 | 170/200 | 197/200 | 100/200 |
| 29 | 100/200 | 100/200 | 100/200 | 100/200 |

All six primary arm/seed rows fail Gate B. Keep `formal_valid=true`, `protocol_valid=true`, `behavior_passed=false`, and `all_passed=false` as the recorded Phase 3C status. Preserve the original thresholds: normal overall at least `180/200`, each cue delay at least `34/40`, reset exactly `100/200` and `20/40` per delay, shuffled below `150/200`.

The unchanged configuration is hidden size 64, recurrent radius 0.9, N=2000 real training decisions, 20 original evaluation blocks, checkpoint interval 100, step size 0.1, discount 0.9 and trace decay 0.8.

Two clocks must not be conflated: cue-to-decision delays are `[1,2,3,4,5]`; action-to-feedback hidden delays are `[1,3,5]`. Training action choices are sampled independently of learned values. Consequently, a replay's matching action digest alone says nothing about whether the learner's weights or evaluation policy are correct.

Prior Phase 3A evidence already contains failures with immediate identified feedback. Source identity is not assumed sufficient for every seed, and privileged references are not guaranteed performance upper bounds.

## 3. Scope and information architecture

Implement later as a **diagnostic-only driver and observers**, reusing the frozen fixtures, policies, learners, aggregator, and evaluation contracts. Do not edit the existing learner, reward, scheduling, acceptance, or frozen-verifier implementations. If the implementation cannot observe a required quantity through snapshots/wrappers without altering those contracts, return for a scoped design amendment rather than changing them silently.

Separate four responsibilities:

1. A fixture/replay driver reproduces frozen calls and prepares immutable sequences.
2. An anonymous learner receives only the existing hidden vector, legal actions, action RNG, and scalar feedback/drain API.
3. An external observer records immutable copies and computes provenance/geometry/update diagnostics. Its results never flow back into selection, feedback generation, or learning.
4. Privileged reference learners operate in separate runs with separately typed inputs. A reporter joins records only after learning/evaluation ends.

Do not attach source identifiers, reward donors, correct labels, multiplicities, delay distributions, timestamps, or observer objects to the anonymous learner. No monkeypatch may change its numerical update or feed observer data back. Read-only copies must not share writable NumPy storage with learner state.

Policy recurrent state resets at each fixture through the existing hidden-state helper. Eligibility state is different: its registered trace persists across those fixtures. The observer must preserve that distinction. Trace-reset counters must be interpreted from the existing lifecycle, not invented from a prose description.

## 4. D0: replay and integrity gate

Before new diagnostic conditions, reproduce all `3 seeds x 2 arms x 2 original reward conditions = 12` training trajectories. The conditions are normal rewards and the **original frozen block-shuffle lineage**, not a newly sampled shuffle. Compare instrumented execution with a fresh, uninstrumented same-environment replay; use independent learner/policy/RNG instances.

Required checks:

- Validate the frozen file hash, registered configuration, formal contract binding, and original portable evidence verifier. Retain its existing comparison surface: only its seven registered checkpoint diagnostic fields allow absolute `1e-12`, with no relative tolerance; its existing parameter-digest exclusions are unchanged.
- Compare original counts, booleans, schemas, sequence order, fixture/action/reward/schedule/aggregate-call digests and checkpoint structure exactly. Do not insert missing fields or coerce types.
- Require same-environment instrumented/uninstrumented weights, hidden-state streams, predictions, eligibility/prediction traces, and RNG states to match exactly wherever recorded on both paths. Cross-environment raw parameter bytes are not required to match the old artifact.
- Independently reconstruct each due bucket and scalar feedback. Every slot creates one record, is consumed once, and the drain leaves no pending records. Read the **actual intercepted scalar calls**, not merely a reconstruction assumed to equal them.
- Verify distinct objects/no writable array aliasing across normal/shuffled runs. Freeze policy matrices and readout snapshots around evaluation/checkpoint collection. Evaluation may reset/advance its policy state, but must not update parameters, consume training randomness, or alter training traces.
- Repeat evaluation in reversed fixture order on fresh copies and restore results to canonical order. Check agreement for the same fixture. Inspect training/evaluation RNG lineage separation; legitimately repeated discrete cue/delay cases are not by themselves data leakage.
- Replay unchanged anonymous inputs while relabeling observer-only sources, hiding/replacing multiplicity metadata, and removing scoring labels from the observer. Anonymous outputs and parameters must not change. Include a counterexample test that would fail if metadata influenced a learner call.

A hash, lineage, replay, mutation, or information-boundary failure stops subsequent behavioral interpretation. Produce an invalid-diagnostic audit with the first mismatch and supporting trace; do not repair the learner or refreeze old evidence in the same run. Existing Gate P success is historical evidence, not permission to skip this stronger audit.

## 5. D1: shuffle provenance and possible temporal realignment

### 5.1 Distinguish slot from reward donor

The code permutes rewards **within consecutive blocks of ten**, and fixtures are also constructed in balanced ten-case blocks. It does not guarantee independence between every delivered value and every action/label.

Define `j` as the latent-record slot, `d_j` its hidden delay, and `pi(j)` the original decision donating its reward after permutation:

```text
normal:    r'_j = r_j
shuffled:  r'_j = r_pi(j)
delivery:  due_j = j + d_j
feedback:  F_t = sum(r'_j for j with due_j = t)
```

The due time belongs to **slot j**, not to donor `pi(j)`. Store both identities, even when two reward values happen to be numerically equal. Reconstruct the original index permutation from its frozen RNG lineage and verify that applying it reproduces the original reward sequence/digest.

A specific diagnostic hypothesis is **temporal realignment**: `pi(j) = due_j = t` makes an older slot deliver the reward originally generated by the current decision. For example, a permitted within-block swap `pi(0)=1`, `pi(1)=0` and `d_0=1` delivers `r_1` at decision 1. This algebraic witness shows why local shuffling does not guarantee destruction of current-step correspondence. It does not establish how often this happened in the frozen run or that it caused `197/200`.

For a uniform independent permutation, conditioning on the fixed schedule gives a useful structural check:

```text
E[number of current-decision donor matches | block-10 shuffle]
  = count(j: due_j < N and floor(j/10) = floor(due_j/10)) / 10

E[number of current-decision donor matches | global shuffle]
  = count(j: due_j < N) / N
```

These are index-matching expectations, not accuracy predictions. Neither global shuffling nor the absence of exact donor matches proves full statistical independence. The precomputed shuffled condition can contain future-donor rewards; log this explicitly as an offline control property rather than silently calling it online causal feedback.

### 5.2 Mandatory provenance and association output

For the original normal/shuffled paths of both arms and every registered seed, record source slot, donor index, due step, actual feedback call, latent value, action, label, cue delay, and delivery multiplicity in the **observer-only** trace. Produce:

- Index fixed-point counts `pi(j)=j`, same-block donor counts, donor/slot displacement histograms, and donor time relative to delivery (past/current/future). Report current-decision donor matches separately from same-valued rewards and from multiple-match delivery buckets.
- Counts/sums/means of `F_t` grouped by current selected action and correct-action label; repeat at lags `0..10` and for the final 100 real decisions. Define lag as the number of later real decisions, not cue delay or drain ticks.
- For each lag, report `mean(F_t * r_(t-lag))` and centered Pearson correlation over valid real-decision pairs. For label/action encodings use `s(a)=2*a-1`; report cell counts and the corresponding products. Zero variance yields `null` plus an explicit reason, never correlation zero.
- Separate no-arrival zero feedback from cancellation-to-zero, one-source buckets, collisions, real-decision calls, and drain calls. These categories are audit-only and must not change what the learner receives.

Associations and donor matches identify candidate channels. They do not by themselves prove leakage or a causal explanation of final accuracy.

### 5.3 Fixed additional permutation grid

After D0 passes, run exactly 32 permutation replicates for each of two conditions: `block10` and `global`. Run both frozen arms on all three original seeds. This is `32 x 2 modes x 2 arms x 3 seeds = 384` diagnostic training trajectories, in addition to the original replay/reference runs.

For each seed keep reservoir, training fixtures, action schedule, hidden-delay schedule, step size, training length and trace parameters fixed. The two arms share each permutation byte-for-byte. `block10` preserves each reward block's multiset; `global` preserves only the full training reward multiset. Schedule, multiplicity and action lineages remain fixed in both. A block/global comparison changes local exchangeability and block statistics as well as donor alignment; it is not a one-variable causal intervention isolating alignment.

Use explicit `PCG64(SeedSequence([seed, 0x33434641, 1, mode, replicate]))`, with mode `0=block10`, `1=global` and replicate `0..31`. Do not consume any original RNG. The original control still uses `[seed, 0x33534846]` and is a separate row, not replicate 0. No resampling to remove fixed points, high scores, or structurally inconvenient outcomes is permitted.

Report every replicate and per-seed/per-arm distributions: mean, median, minimum, maximum, nearest-rank p10/p90, number at or above `150/200`, and number at or above `197/200` on the original evaluation fixture. Nearest-rank quantile q uses sorted index `ceil(q*32)-1`. Report the original control's rank descriptively. Do not report that rank as a calibrated p-value: this investigation was selected after observing an anomalous result. Do not pool different reservoirs or correlated evaluation episodes as independent experimental replicates.

### 5.4 Fixed held-out evaluation extension

Generate eight additional evaluation fixture sets per original seed, each with 20 balanced blocks/200 decisions, using `PCG64(SeedSequence([seed, 0x33434641, 2, evaluation_id]))`, with evaluation IDs `0..7`. Generate and hash the entire evaluation manifest before fitting any new diagnostic/reference row. Do not create new reservoir or training-fixture seeds.

Evaluate every original final model, every permutation replicate, and every D2 reference on the original evaluation set plus all eight additional sets. Reuse the same sets across matched rows; report each set separately and their descriptive mean. Training uses none of their labels. Additional evaluation determines whether an observation persists across fixture realizations conditional on a frozen reservoir/model; it does not establish generalization across reservoirs or environments. Do not select an evaluation set, checkpoint, permutation, or model based on its score.

## 6. D2: representation and source-identified references

Use the identical decision-time hidden-state sequence and `phi_i = [h_i, 1]`. Verify hidden-stream equality before interpreting comparisons. Construct three references for each original seed; all are diagnostic-only and evaluated as specified in Section 5.4.

**Supervised ridge reference.** Fit both action outputs from training labels only, with target `Y_i[a]=+1` for the correct action and `-1` for the other. Solve `min_B ||Phi B - Y||_F^2 + 1e-6 ||B||_F^2`, penalizing all coefficients including bias. No feature scaling, regularization search, reservoir training, or evaluation-label fitting is allowed. Use a stable linear solve, not explicit matrix inversion. Report overall/per-cue-delay counts and margins, including reset evaluation. A high score demonstrates usable information for this privileged fit; a low score does not prove an information-theoretic impossibility.

**Immediate identified reference.** Reuse frozen Phase 3A normalized action-value training on the same normal rewards/actions/features and verify its original per-seed counts. This retains the known immediate-readout limitation instead of assuming identified feedback always passes.

**Source-visible delayed reference.** For normal rewards only, use a separate privileged learner storing `(a_i, phi_i, ||phi_i||^2, q_i)` at each real decision, where `q_i` is that reference learner's decision-time prediction. At delivery update the actual source:

```text
W[a_i] += 0.1 * (r_i - q_i) * phi_i / ||phi_i||^2
```

Use the unchanged action/delay schedules. Process collision records by ascending source index; drain every remaining record without synthetic actions. Predictions remain frozen at decision time; do not silently recompute them on delivery. Initial weights are zero, as in the frozen action-value learner. This reference receives individual source rewards and identities, so both source association and aggregate decomposition are privileged. It is not an anonymous algorithm, a strict upper bound, or proof that source identity alone is the cause. Do not run a donor-aware shuffled oracle that imports future hidden states before their decisions.

Report representation, immediate readout, delayed source-visible readout and original anonymous results side by side. Multiple failure mechanisms may coexist. None of these privileged scores can satisfy Phase 3C's original Gate B or be promoted as the production policy.

## 7. D3: prediction-target and update accounting

Instrument only the 12 original normal/frozen-shuffled trajectories for full per-step accounting; the large permutation grid needs the lighter D1 audit, not full matrix dumps. Use actual decision-time predictions and weight snapshots, not a later recomputation substituted for history.

### 7.1 Separate historical prediction from delivered reward

For Arm B with normal rewards during real-decision steps define `g_i` as a zero matrix except `g_i[a_i] = phi_i / ||phi_i||^2`. Let `q_i` be that original learner's frozen decision-time prediction, `rho=0.9*0.8`, and `S_t={i: due_i=t}`. Its registered eligibility mechanism has:

```text
E_t = sum(i<=t, rho^(t-i) * g_i)
P_t = sum(i<=t, rho^(t-i) * q_i)
F_t = sum(i in S_t, r_i)
K_t = sum(i in S_t, q_i)                 # observer-only source-matched predictor
F_t - P_t = (F_t - K_t) + (K_t - P_t)
```

Arm A has no persistent E/P trace: its effective update direction is `g_t` and its prediction is `q_t`. Use those quantities for its current-step residual and update checks; do not reconstruct an exponential trace and attribute it to Arm A.

`K_t` uses realized hidden source information and is never supplied to a learner. It is not an available anonymous predictor or asserted conditional expectation. Compare means/RMSE of `F-P`, `F-K`, and `K-P`, retaining their cross term so squared errors are not incorrectly treated as additive. Report all-time, checkpoint-block, zero/collision and tail summaries. Calibration is descriptive; no posterior model or learned delay kernel is introduced.

For shuffled rewards, retain slot predictions and reward-donor predictions under different field names; a donor may be in the future at delivery time. Any future-donor calculation is offline audit only. Do not label a slot-matched predictor a true causal target after shuffling. The normal-reward decomposition is the primary mechanism analysis.

### 7.2 Decompose history without equating magnitude with correctness

On Arm B normal paths split `E_t = E_source,t + E_other,t`, where `E_source,t` includes exactly the members of `S_t` with their registered coefficients. Account for `alpha*(F-P)*E` as the sum of its two components. Separately record the observer reference direction `J_t=sum(i in S_t, (r_i-q_i)*g_i)` along the original trajectory. This is not the independently trained privileged reference's trajectory.

Report Frobenius norms, inner products, and cosines between the actual update and `J_t`, as well as the source/other components. Undefined zero-norm cosines are `null` with counts. Large other-history norm is not sufficient to conclude harmful interference: shared features can make that component helpful. Do not map the coefficient `rho^d` directly to expected accuracy.

Analytic/small hand-constructed examples may establish target mismatch or a counterexample under explicit assumptions. Existing Lean results establish anonymity/conservation/trace coefficients/immediate reduction, not convergence or correct NumPy execution. Any new Lean theorem or learner correction requires a separate approved scope; it is not a prerequisite to reporting unresolved diagnostics.

### 7.3 Isolate terminal drain and scoring surfaces

Snapshot weights and persistent traces immediately before drain, after every drain call, and after `end_run`. Run evaluation on fresh copies at each snapshot, never on the training instance. Label pre-drain and intermediate scores diagnostic; the registered endpoint remains post-drain.

In Arm A drain must be a parameter no-op. In Arm B no new real decision means both `E` and `P` stay fixed during drain, so its aggregate increment satisfies:

```text
W_after - W_before = alpha * (sum_drain(F) - n_drain * P_before) * E_before
```

Account for the repeated prediction subtraction explicitly, including zero feedback calls. Do not alter decay/reset semantics to improve the endpoint. The last existing checkpoint scores the last training block, not the held-out evaluation set; never infer a drain loss by subtracting those two incompatible accuracy counts.

For independently reconstructed algebra only, require componentwise `abs(actual-expected) <= 1e-10 + 1e-12*abs(expected)` and report maximum residual. This does not relax D0 exact same-environment replay or the original frozen verifier. Non-finite values or residuals beyond this bound invalidate the diagnostic accounting; do not increase tolerances after inspecting results.

## 8. Evidence and failure handling

Later outputs use an isolated directory `docs/experiments/phase-3c-failure-attribution-v1/` with `manifest.json`, `execution-manifest.json`, `summary.json`, and `report.md`. Large immutable trace files may be retained as CI artifacts referenced by exact workflow run, artifact identity, byte digest and relative filename. Do not rely solely on an expiring URL; retain the summary/manifest and enough fixed lineage information to regenerate the trace.

Before new measurements, freeze `manifest.json`: spec revision/commit, implementation SHA, frozen input hashes, Python patch version, NumPy version, BLAS/runtime platform, exact RNG scheme, ordered work grid, evaluation IDs/hashes and learner parameters. Record output file digests, timings, attempt provenance and completion/error states separately in `execution-manifest.json`, then seal it when the run ends. Do not mutate the preregistered manifest to add outcomes. Timeouts never authorize reducing the grid.

The summary separates:

- `diagnostic_valid` and integrity outcomes;
- unchanged original Phase 3C status;
- original replay results and D1 replicate/evaluation rows;
- privileged-reference results with an explicit information-access label;
- D3 accounting residuals, undefined-metric reasons and trace references;
- attribution candidates, supporting/contradicting evidence, and limitations.

`diagnostic_valid=true` requires a complete registered grid, D0 success, finite/explicitly undefined metrics as specified, and all accounting/immutability checks passing. It does not require a unique root cause or high behavioral accuracy.

Candidate explanations are not exclusive: implementation/information-boundary defect, control residual structure/temporal realignment, fixture-sensitive control outcome, representation/readout limitation, prediction-target mismatch, history interaction, and drain contribution. Mark each `supported`, `not_supported`, or `unresolved` and cite actual output fields. A rare rank does not prove chance; a correlation does not prove causation; a better privileged reference does not prove anonymity is the only problem.

The report must explicitly distinguish a mathematical possibility from its occurrence in the registered trace and from a demonstrated performance cause. Without a discriminating intervention or adequate accounting, retain `unresolved` for causation and propose at most one separately reviewed follow-up. Do not force a single root-cause verdict.

Missing/duplicate rows, mismatched configurations, unexpected keys/types, non-finite required metrics, unauthorized file mutation or provenance mismatch fail closed. Represent genuinely undefined ratios/correlations with a typed reason, not zero/NaN. Preserve partial output as incomplete audit, never a complete experiment. Technical retries must reuse identical manifests/seeds and preserve all attempt provenance; they must not select a favorable result.

## 9. Execution gates after review

The ordered boundaries are:

```text
review this committed spec
    -> write/review a separate implementation plan
    -> approve diagnostic-only implementation
    -> Python 3.12 tests, static checks, adversarial controls and replay parity
    -> freeze the diagnostic manifest
    -> D0 integrity gate
    -> D1 provenance + complete permutation/evaluation grid
    -> D2 references and D3 accounting
    -> freeze diagnostic report, including negative/unresolved outcomes
    -> review one next action; no automatic learner change
```

No D1/D2/D3 behavioral interpretation proceeds after D0 failure. An unexpected implementation defect is reported separately; fixing it requires review and a new evidence version. The original artifact remains unchanged even if an audit qualifies its scientific interpretation. Any attribution report must name the affected claims instead of silently changing the historical Gate P fields.

## 10. Testing and Python 3.12-only CI contract

Future tests must fail when a permutation donor is confused with its slot, feedback is delivered at donor time, fixed points are inferred from equal reward values, labels leak across the observer boundary, arrays alias, evaluation mutates parameters, RNG lineages interfere, the grid is incomplete, or drain introduces an action/decay. Include the hand-constructed temporal-realignment witness, collisions/cancellations, source-visible collision order and decision-time prediction staleness.

Test exact same-environment instrumented parity, frozen evidence immutability, independent held-out manifests, accounting reconstruction, zero-norm diagnostics and explicit negative/unresolved classifications. Diagnostics must be able to complete successfully while original `behavior_passed` stays false.

CI remains **Python 3.12 only**. Do not restore a 3.10/3.11/3.13 matrix. The implementation plan must pin and record a reproducible Python 3.12/NumPy environment before diagnostic measurement; no dependency sweep is part of this design. Retain all existing replay and protocol gates.

Do not run the full 384-trajectory grid automatically on every push/PR. The later plan may specify one explicitly approved bounded/manual measurement job with Python 3.12 and complete artifact retention. Small deterministic test fixtures are CI gates, not new registered behavioral evidence.

This spec-only commit changes no workflow and requests `[skip ci]` to avoid rerunning experiments for a document. Existing run #414 belongs to the immutable code reference, not to the new documentation commit. No new exact-head test claim is made here.

## 11. Acceptance of this specification

The checkpoint is complete when this single spec is committed on the stated branch, source references and original counts are checked, the diagnostic grid/lineages/information boundaries/failure rules are explicit, and the commit diff confirms no code, workflow, original evidence, original spec or implementation-plan change.

User approval of this document is required before creating the implementation plan. A further execution approval is required before diagnostic implementation/measurement. This checkpoint ends at **committed-spec review**.

## 12. Explicit non-goals

No learner promotion or tuning, new production mechanism, alternate seeds chosen after results, new acceptance threshold, refreezing Phase 3C, rewriting Phase 2/3A/3B artifacts or silently changing their conclusions, general causal-discovery claim, automatic merge, YOLO/FlyVis/ROS2 integration, actor-critic/BPTT/attention/replay-buffer subsystem, or claim that Lean proves the Python runtime. Mechanism improvement remains a later decision informed by these diagnostics.

## 13. Source map

All code/document observations above refer to `55a7ba59ac301975f3df996636f9a596e90bc730`; future readers must use that immutable revision, not a moving default branch.

- [Frozen Phase 3C report](https://github.com/qigao/yolo-motion-perception/blob/55a7ba59ac301975f3df996636f9a596e90bc730/docs/experiments/phase-3c-anonymous-temporal-credit-report.md): registered outcomes, input digest and formal/empirical interpretation boundary.
- [Original Phase 3C design](https://github.com/qigao/yolo-motion-perception/blob/55a7ba59ac301975f3df996636f9a596e90bc730/docs/superpowers/specs/2026-09-16-phase-3c-anonymous-temporal-credit-design.md): clocks, learner-visible inputs, trace/drain semantics and unchanged Gates F/P/B.
- [Phase 3C benchmark](https://github.com/qigao/yolo-motion-perception/blob/55a7ba59ac301975f3df996636f9a596e90bc730/src/neural_state_machine/phase3c_benchmark.py): `_execute_training`, `_measurement_from_protocol`, reward override, block size ten, original RNG and post-drain evaluation.
- [Action-value benchmark](https://github.com/qigao/yolo-motion-perception/blob/55a7ba59ac301975f3df996636f9a596e90bc730/src/neural_state_machine/action_value_benchmark.py): `_permute_reward_blocks`, fixture lineage, evaluation and training-checkpoint definitions.
- [Fixture/hidden-state helper](https://github.com/qigao/yolo-motion-perception/blob/55a7ba59ac301975f3df996636f9a596e90bc730/src/neural_state_machine/reward_learning.py): `_balanced_cases`, `_build_fixtures`, `_decision_hidden` and per-fixture recurrent reset.
- [Anonymous learners](https://github.com/qigao/yolo-motion-perception/blob/55a7ba59ac301975f3df996636f9a596e90bc730/src/neural_state_machine/phase3c_learners.py): persistent E/P, decision-time predictions, drain and `end_run` lifecycle.
- [Normalized action value](https://github.com/qigao/yolo-motion-perception/blob/55a7ba59ac301975f3df996636f9a596e90bc730/src/neural_state_machine/action_value.py): bias feature, normalization, zero initialization and deterministic greedy tie rule.
- [Existing Phase 3A attribution](https://github.com/qigao/yolo-motion-perception/blob/55a7ba59ac301975f3df996636f9a596e90bc730/docs/experiments/phase-3a-failure-analysis.md): identified-feedback and privileged-readout limitations motivating D2.
- [Frozen verifier](https://github.com/qigao/yolo-motion-perception/blob/55a7ba59ac301975f3df996636f9a596e90bc730/scripts/verify_phase3c_anonymous_credit.py) and [CI](https://github.com/qigao/yolo-motion-perception/blob/55a7ba59ac301975f3df996636f9a596e90bc730/.github/workflows/ci.yml): replay comparison boundary and Python 3.12-only configuration.

The realignment witness, conditional index-matching expectations and accounting identities are deductions/design requirements in this document. They are not claimed measurements of a newly executed experiment.
