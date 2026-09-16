# Phase C4 — Delay-Marginalized Anonymous Credit Design

> Status: **design committed for review; implementation and new scientific measurement are not authorized by this document's existence**.
>
> Repository: `qigao/yolo-motion-perception`
>
> Design branch: `experiment/phase3c-anonymous-temporal-credit`
>
> Clean scientific base after one-shot workflow removal: `470f84d54364a1bc42d5c0ddedf7ca398d4f9d7d`
>
> Frozen Phase 3C registered status: `formal_valid=true`, `protocol_valid=true`, `behavior_passed=false`, `all_passed=false`.
>
> Registered Task 12 diagnostic run: `35078403826`; evidence SHA-256: `939178582c3c997bbf4ddd72942e26200f10e98cb42bb9c8b7ab1bff3dea015c`.
>
> Date: 2026-09-16
>
> Review boundary: spec only. After approval, write and commit a separate implementation plan. Do not implement the C4 learner, change frozen C3 evidence, create a measurement workflow, or run C4 behavioral measurement at this checkpoint.

## 1. Decision

C4 will test a new anonymous temporal-credit mechanism based on the **known delay distribution**, not on realized source identity and not on a global exponentially decayed eligibility trace.

The mechanism is called **delay-marginalized anonymous credit**. For the registered hidden feedback law, the learner knows only the public support and probabilities:

```text
D = {1, 3, 5}
p(d) = 1/3 for d in D
```

It does **not** know which delay was realized for any source action, which source(s) generated a delivered aggregate scalar, delivery multiplicity, latent individual rewards, queue state, due-step metadata, or source identifiers.

C4 is designed around the strongest bounded conclusion from Task 12:

1. local `block10` control structure exists, but exact donor/time realignment is not sufficient to explain the seed-17 `197/200` result;
2. the recurrent representation contains a linearly usable solution on the frozen task because the privileged supervised ridge reference scored `200/200` on all 27 registered seed/evaluation combinations;
3. source identity alone is not a complete repair because identified/source-visible normalized readouts remain realization-sensitive at seeds 7 and 29;
4. anonymous eligibility carries a large non-source history component, about `4.60x–4.98x` the source-history magnitude in the registered accounting;
5. terminal drain materially changes final eligibility behavior for some seeds and conditions.

These observations motivate a mechanism that limits credit support to **only the finite set of source times that could have produced the current feedback under the public delay law**. They do not establish that this change must solve the task.

C4 therefore separates two questions that Phase 3C combined:

- **C4-A — anonymous operator identifiability:** if we fit the same linear readout in batch mode using only anonymous aggregate feedback, the learner's own decision history, and the public delay law, is a high-quality solution recoverable at all registered seeds?
- **C4-B — online credit acquisition:** if C4-A succeeds, can a fixed online normalized update using the same marginalized observation model satisfy the unchanged behavioral gate?

A failure of C4-A stops the program before registered C4-B measurement. A success of C4-A does not guarantee C4-B; it only establishes that the remaining failure is not explained by the tested batch anonymous observation model.

## 2. Frozen inputs and claim boundary

C4 is a successor experiment. It must not rewrite, reinterpret, or refreeze Phase 3C.

The following inputs are immutable scientific references:

| Item | Frozen identity / rule |
|---|---|
| Phase 3C behavioral evidence | `docs/experiments/phase-3c-anonymous-temporal-credit.json` |
| Original Phase 3C evidence SHA-256 | `53b52fb6716daaceeb68b4e5c78f33333c0076b0d727462aa7265b0887798263` |
| Phase 3C failure-attribution result | `docs/experiments/phase-3c-failure-attribution-v1/registered-result.md` |
| Task 12 provenance commit | `aacbb99559628db8012cd4bec9e1293a4e41fdb3` |
| Valid Task 12 run | `35078403826` |
| Task 12 evidence SHA-256 | `939178582c3c997bbf4ddd72942e26200f10e98cb42bb9c8b7ab1bff3dea015c` |
| C4 clean base | `470f84d54364a1bc42d5c0ddedf7ca398d4f9d7d` |
| Registered seeds | `(7, 17, 29)` |
| Training decisions | `2000` |
| Evaluation decisions | `200` in 20 balanced blocks |
| Cue-to-decision delays | `(1, 2, 3, 4, 5)` |
| Hidden action-to-feedback delay support | `(1, 3, 5)` |
| Delay probabilities | exactly `(1/3, 1/3, 1/3)` |
| Hidden size | `64` |
| Recurrent radius | `0.9` |
| Step size | `0.1` |
| Existing Phase 3A/C action-value normalization | unchanged |
| Training fixture lineage | unchanged from Phase 3C |
| Evaluation fixture lineage | unchanged from Phase 3C |
| Behavior-action RNG lineage | unchanged from Phase 3C |
| Hidden-delay schedule lineage | unchanged from Phase 3C |
| Original shuffled-control lineage | unchanged from Phase 3C |

The original Phase 3C behavioral gate remains unchanged for comparability:

```text
normal overall >= 180/200
each cue delay >= 34/40
reset overall == 100/200
each reset cue delay == 20/40
original frozen shuffled control < 150/200
```

No C4 result may change the historical Phase 3C status fields. C4 receives its own evidence and status.

Task 12's additional eight fixed evaluation sets per seed are retained as **secondary stability surfaces**. They are not used for fitting, parameter selection, checkpoint selection, or changing the primary acceptance threshold. Their results are reported separately and in a descriptive mean.

## 3. Alternatives considered

### 3.1 Selected: delay-marginalized observation and credit

Use the public delay distribution to form the expected contribution of the only source times that could generate feedback now. This removes the global eligibility tail and does not require realized source metadata.

Advantages:

- directly targets the history-interference finding from Task 12;
- uses only learner-owned past decisions and a fixed public environment law;
- has a bounded history of five real decisions;
- has a clean batch identifiability test before online optimization;
- can be required to reduce exactly to Phase 3A at the immediate-delay boundary;
- terminal drain becomes the same observation equation evaluated after the final action, rather than a special update against one persistent global trace.

### 3.2 Rejected for C4: lag-separated eligibility banks

Maintain one eligibility bank per candidate lag, such as `E1`, `E3`, and `E5`. This reduces some cross-history interference but leaves an arbitrary question when one anonymous scalar arrives: how should that scalar be assigned among the three banks? Without an explicit observation model, the mechanism mainly relocates the ambiguity.

### 3.3 Deferred: learned responsibility / gating

Train a model to infer posterior source responsibility from history and feedback. This is more expressive but introduces a second learned subsystem, new optimization dynamics, and a larger information surface. It would make a C4 failure harder to attribute. C4 keeps the responsibility weights fixed to the known delay prior.

## 4. Core observation model

### 4.1 Decision-time quantities

For real decision `i`, let:

```text
h_i       recurrent hidden state at the decision
phi_i     existing Phase 3A augmented feature [h_i, 1]
a_i       selected action
n_i       exactly the existing Phase 3A normalization denominator for phi_i
W         action-value weight matrix
q_i       W[a_i] dot phi_i
```

C4 does not alter reservoir dynamics, feature construction, legal-action handling, action RNG, tie rules, or the definition of the existing normalized credit denominator.

Define an action-blocked prediction feature `X_i` with the same shape as `W`:

```text
X_i[a_i] = phi_i
X_i[a != a_i] = 0
```

and an action-blocked normalized credit feature `G_i`:

```text
G_i[a_i] = phi_i / n_i
G_i[a != a_i] = 0
```

Then `q_i = <W, X_i>_F`, where `<.,.>_F` is the Frobenius inner product.

### 4.2 Realized aggregate feedback

The environment still uses the hidden realized delay schedule. If source `j` draws delay `d_j`, its latent reward is delivered at `j + d_j`. At delivery time `t` the learner receives exactly one scalar:

```text
F_t = sum(r_j for j such that j + d_j = t)
```

The learner never receives the set of contributing `j` values.

### 4.3 Marginalized expected prediction

For every delivery clock `t`, define valid candidate sources:

```text
S_t = { t-d | d in {1,3,5}, 0 <= t-d < N }
```

Using the fixed public prior `p(d)=1/3`, C4 predicts the anonymous aggregate as:

```text
P_t = sum_d p(d) * q_(t-d)
```

for valid candidate sources only.

Equivalently define:

```text
Z_t = sum_d p(d) * X_(t-d)
P_t = <W, Z_t>_F
```

Under the registered independently generated delay law, `P_t` is the model's conditional expectation of the aggregate scalar when each candidate source reward is represented by its decision-time prediction. It is not a claim that the realized source set equals the three candidates.

### 4.4 Marginalized normalized credit

Define:

```text
C_t = sum_d p(d) * G_(t-d)
delta_t = F_t - P_t
W <- W + alpha * delta_t * C_t
```

with fixed `alpha=0.1`.

This is a normalized residual update, not a claim of standard TD(lambda), an exact Bayesian posterior, or convergence to a unique optimum.

The credit support is bounded. At any `t`, at most the source decisions `t-1`, `t-3`, and `t-5` can contribute to `C_t`. There is no exponentially decayed contribution from every retained historical decision.

### 4.5 Immediate boundary

C4 must expose a protocol-control delay law:

```text
D = {0}
p(0) = 1
```

For that boundary:

```text
P_t = q_t
C_t = G_t
W <- W + 0.1 * (F_t - q_t) * G_t
```

The update must reduce to the frozen Phase 3A normalized action-value update with exact same-environment action, scalar-update, weight-byte, and parameter-digest continuity.

No tolerance-based substitute is accepted for this same-environment immediate-boundary check.

### 4.6 Finite candidate-history buffer

The C4 learner may retain only the decision-time quantities necessary to evaluate the public support `{1,3,5}`. A ring buffer of the last five real decisions is sufficient.

This buffer is **not** an unresolved-credit queue:

- entries are inserted on every real decision independently of future feedback;
- entries are evicted deterministically by age, not when a reward claims them;
- there is no pending/resolved flag;
- no realized delay, due step, source identity, multiplicity, latent reward, or queue metadata is stored;
- one feedback scalar never selects or removes a source entry.

## 5. Terminal drain semantics

Terminal drain uses the same delivery-clock equation as training.

After decision `N-1`, no new decision history is appended. For each remaining delivery clock `t`, compute `P_t` and `C_t` from still-valid candidate source indices in `[0, N-1]`, receive the one anonymous aggregate scalar `F_t`, and apply the same residual update.

Therefore drain:

- does not add a synthetic action;
- does not decay or extend a global eligibility trace;
- does not use a special persistent pre-drain trace state;
- does not reveal which pending source was delivered;
- naturally reaches `C_t=0` once no valid candidate source remains.

The environment still audits exactly-once latent delivery externally. That audit never enters the learner.

## 6. C4-A — Anonymous batch operator probe

C4-A is a diagnostic gate, not the production learner.

### 6.1 Purpose

Test whether the anonymous aggregate observation operator itself permits recovery of a useful linear readout from the frozen recurrent representation when online optimization is removed.

A C4-A success supports this bounded statement:

> For the fixed task, seeds, representation, action sequence, delay law, and training length, a regularized batch linear fit using only anonymous aggregate feedback plus learner-owned history can recover a readout that satisfies the prospectively fixed C4-A behavioral gate.

It does not prove general identifiability, uniqueness, online learnability, or production readiness.

### 6.2 Fit

For every delivery clock, flatten `Z_t` into one row of the design matrix `Z` and use the actual learner-visible scalar `F_t` as the target.

Fit:

```text
min_w ||Z w - F||_2^2 + 1e-6 ||w||_2^2
```

where the existing action-block bias coordinates are included and penalized. The regularization value is fixed at `1e-6`, matching the already registered diagnostic ridge convention; there is no search, scaling, feature selection, checkpoint selection, or evaluation-label fitting.

Use a stable least-squares construction with an augmented regularization system. Do not invert a normal-equation matrix explicitly.

The fitted vector is reshaped into the normal action-value weight matrix and evaluated through the same frozen evaluation policy/evaluator used by Phase 3C.

### 6.3 Information boundary

The batch fitter may consume only:

- `phi_i` from the learner's own decision history;
- `a_i` selected during the fixed training run;
- public `D={1,3,5}` and `p(d)=1/3`;
- actual scalar learner calls `F_t` including terminal drain.

It may not consume realized delay assignments, source IDs, individual latent rewards, multiplicity, due buckets, correct labels, cue labels, evaluation labels, or Task 12 observer-only provenance.

The offline use of the complete training sequence is privileged in time but **not in source metadata**. C4-A is therefore an optimization/identifiability reference, not an online deployable learner.

### 6.4 C4-A fixed gate

Run the batch probe for the same three seeds under:

1. normal rewards;
2. the original frozen Phase 3C shuffled-control lineage.

The primary C4-A gate is exactly the historical Gate B surface:

```text
normal overall >= 180/200
normal each cue delay >= 34/40
reset overall == 100/200
reset each cue delay == 20/40
original frozen shuffled control < 150/200
```

All three seeds must pass every required primary field.

The eight Task 12 additional evaluation sets per seed are scored after fitting and reported separately as stability evidence. They do not change the primary gate and are not used to select the fit.

If any registered seed fails the C4-A primary gate, C4-B registered behavioral measurement is not authorized. Record `operator_passed=false`; do not tune the regularizer, prior, step size, representation, support, thresholds, training length, or seeds.

## 7. C4-B — Online delay-marginalized learner

C4-B is fully specified prospectively in this design so C4-A cannot be used to tune it.

The online learner:

1. records the current decision's `X_i`, `G_i`, and decision-time `q_i` in the finite candidate-history buffer;
2. receives exactly one aggregate scalar call for the current delivery clock, including `0.0`;
3. forms `P_t` and `C_t` using fixed prior weights `(1/3,1/3,1/3)` over lags `(1,3,5)`;
4. applies `W <- W + 0.1 * (F_t-P_t) * C_t`;
5. evicts history only by deterministic age;
6. uses the identical equation during terminal drain without adding decisions.

No adaptive responsibility weights, learned delay model, importance weights, eligibility decay parameter, source-visible fallback, or per-seed branch is permitted.

### 7.1 Behavioral gate

Only after C4-A passes and a separate human measurement checkpoint is approved may the first registered C4-B behavioral measurement be run.

C4-B uses the unchanged primary Gate B:

```text
normal overall >= 180/200
normal each cue delay >= 34/40
reset overall == 100/200
reset each cue delay == 20/40
original frozen shuffled control < 150/200
```

All registered seeds must pass.

Report the eight fixed additional evaluation sets as secondary stability evidence. Do not change thresholds after observing them.

C4 status is recorded independently as:

```text
formal_valid
protocol_valid
operator_passed
behavior_passed
all_passed = formal_valid && protocol_valid && operator_passed && behavior_passed
```

Before C4-B is measured, `behavior_passed` is unmeasured rather than implicitly false.

## 8. Formal gate

C4 adds a new Lean module rather than modifying the proven Phase 3C theorem files.

Proposed module boundary:

```text
NarrativeDynamics/Core/MarginalizedTemporalCredit.lean
NarrativeDynamics/Tests/MarginalizedTemporalCredit.lean
```

It imports the existing anonymous temporal-credit theory and proves the mathematical contract required by C4.

The formal gate must cover at least:

1. **Finite delay-law validity.** Delay weights are nonnegative and sum to one on the registered finite support.
2. **Expected aggregate decomposition.** Under independent draws from the fixed finite delay law, the expectation of the delivered scalar at `t` equals the weighted sum of valid candidate source rewards.
3. **Bounded credit support.** The marginalized credit at `t` depends only on candidate source indices `t-d` for support elements `d`; no other history index contributes.
4. **Immediate reduction.** Support `{0}` with probability one reduces the marginalized normalized update to the previously bound Phase 3A normalized update equation.
5. **Boundary truncation.** Invalid negative/pre-start and post-training candidate indices contribute zero; the same finite equation covers terminal drain.

Axiom audit is required. No `sorryAx` or project-defined custom axiom is allowed.

The formal gate proves the real-number mathematical contract. It does not prove NumPy floating-point execution, behavioral success, statistical identifiability, or convergence.

The exact passing Lean SHA must be bound into a C4 formal-contract manifest before Python registered measurement.

## 9. Python architecture and isolation

C4 implementation must be additive. Existing C3 scientific modules and evidence remain byte-for-byte frozen.

Proposed ownership:

| Path | Responsibility |
|---|---|
| `src/neural_state_machine/phase_c4_delay_model.py` | Fixed delay-law object, candidate-index generation, `Z_t/P_t/C_t` construction, finite history primitives. |
| `src/neural_state_machine/phase_c4_batch_probe.py` | C4-A anonymous batch ridge fit and fit diagnostics. No evaluation labels in fitting. |
| `src/neural_state_machine/phase_c4_learner.py` | C4-B online delay-marginalized learner only. |
| `src/neural_state_machine/phase_c4_controls.py` | Formal-contract loading, information-boundary checks, immediate continuity, call/accounting digests, fail-closed Gate P. |
| `src/neural_state_machine/phase_c4_benchmark.py` | Frozen fixtures, paired normal/shuffled runs, C4-A/C4-B orchestration, drain, evaluation, primary gate calculation. |
| `scripts/benchmark_phase_c4_delay_marginalized_credit.py` | Protocol gate and, only after authorization, registered measurement writer. |
| `scripts/verify_phase_c4_delay_marginalized_credit.py` | Strict committed evidence verifier. |
| `tests/test_phase_c4_delay_model.py` | Candidate support, boundary truncation, scalar equations. |
| `tests/test_phase_c4_batch_probe.py` | Anonymous-only fit, fixed ridge, no-label/no-source guards. |
| `tests/test_phase_c4_learner.py` | Exact online arithmetic, bounded history, drain, atomic failures. |
| `tests/test_phase_c4_controls.py` | Fail-closed protocol mutations, immediate continuity, information non-interference. |
| `tests/test_phase_c4_benchmark.py` | Matched lineages, unchanged gates, no behavior fields in protocol-only mode. |
| `tests/test_phase_c4_evidence.py` | Schema/writer/verifier mutation tests. |

Naming may be mechanically shortened during the implementation plan only if every path and public import is updated consistently before implementation begins. Scientific semantics may not change through renaming.

### 9.1 Frozen modules

At minimum, do not modify these existing scientific implementations for C4:

```text
src/neural_state_machine/action_value.py
src/neural_state_machine/phase3c_schedule.py
src/neural_state_machine/phase3c_learners.py
src/neural_state_machine/phase3c_controls.py
src/neural_state_machine/phase3c_benchmark.py
src/neural_state_machine/reward_learning.py
src/neural_state_machine/action_value_benchmark.py
scripts/verify_phase3c_anonymous_credit.py
docs/experiments/phase-3c-anonymous-temporal-credit.json
docs/experiments/phase-3c-anonymous-temporal-credit-report.md
docs/experiments/phase-3c-failure-attribution-v1/**
```

If implementation requires changing one of these files, stop and amend this design before proceeding.

## 10. Protocol gate and anti-peeking boundary

Before any C4-A or C4-B registered behavior is inspected, Gate P must prove structural validity on the exact implementation head.

Required checks include:

- exact frozen fixture/action/delay/control lineages;
- same action and hidden-state streams for paired mechanisms where required;
- learner-call stream contains only one finite scalar per delivery clock;
- candidate-history buffer contains no realized source/delay/due/multiplicity metadata;
- source relabeling and observer-only multiplicity changes cannot alter learner calls or parameters;
- `Z_t`, `P_t`, and `C_t` reconstructed independently from learner-owned history match the implementation;
- decision count, latent count, delivered count, and final pending count satisfy the frozen environment contract;
- terminal drain adds no synthetic decisions and ends after the last possible registered delay;
- immediate `{0:1}` boundary is exact same-environment Phase 3A continuity;
- repeated protocol-only execution is deterministic;
- protocol-only result types expose no C4-A/C4-B post-training score, threshold result, or `behavior_passed` field.

Any Gate P failure means `harness invalid`. Do not inspect or serialize registered behavioral accuracy from that head.

## 11. Measurement lifecycle and sealed environment

C4 uses the same fail-closed philosophy that made Task 12 valid.

Before each registered measurement stage:

1. freeze the reviewed scientific implementation SHA;
2. freeze the formal-contract SHA;
3. freeze the experiment/config manifest and every immutable input hash;
4. freeze the exact Python/dependency environment using the repository's C4 lock and sealed-environment manifest;
5. run exact-head CI and the strict evidence verifier in no-result/pre-measurement mode;
6. create a one-shot measurement workflow only for the approved stage;
7. at workflow start, compare every sealed environment field before fitting or evaluation;
8. fail closed on any difference, producing no scientific result;
9. after a valid run and evidence verification, commit the result/provenance and remove the one-shot workflow in a separate non-scientific cleanup commit.

A failed preflight attempt is provenance, not a scientific run.

C4-A and C4-B are separate measurement checkpoints. A valid C4-A run cannot automatically authorize C4-B.

## 12. Evidence layout

C4 evidence is isolated from C3:

```text
docs/experiments/phase-c4-delay-marginalized-credit/
  formal-contract.json
  c4a-manifest.json
  c4a-result.json
  c4a-report.md
  c4a-provenance.json
  c4b-manifest.json        # created only if C4-A passes and C4-B is approved
  c4b-result.json          # created only after valid registered C4-B measurement
  c4b-report.md
  c4b-provenance.json
```

Large traces remain workflow artifacts with recorded SHA-256 values rather than being silently substituted for committed summaries.

Each result records exact code SHA, formal SHA, environment manifest hash, input hashes, run ID, artifact ID/hash, counts, primary gate fields, and secondary evaluation-set scores.

## 13. Stop rules

The following outcomes are intentional terminal states, not invitations to tune in place.

### 13.1 Formal failure

If the required C4 mathematical contract cannot be proved as stated, stop before C4 Python scientific execution. Revise the design explicitly rather than weakening the theorem after seeing behavior.

### 13.2 Protocol failure

If anonymity, candidate support, lineages, immediate continuity, deterministic replay, or drain accounting fails, record `protocol_valid=false` / `harness invalid` and stop. Do not interpret scores.

### 13.3 C4-A failure

If the anonymous batch probe fails its fixed primary gate at any registered seed, record `operator_passed=false` and stop before C4-B registered measurement.

Permitted conclusion: the tested fixed anonymous expectation operator plus frozen representation/training data did not demonstrate a sufficient batch solution under the precommitted gate.

Not permitted: claim information-theoretic impossibility or tune ridge/prior/support until it passes.

### 13.4 C4-A success, C4-B failure

This is a scientifically useful separation. It supports the bounded conclusion that the tested anonymous operator admits a successful batch fit while the fixed online normalized acquisition rule does not satisfy the registered behavioral gate.

Do not respond by changing `alpha`, probabilities, normalization, training length, or thresholds inside the same registered phase.

### 13.5 C4-B success

A pass supports only the bounded claim that this fixed delay-marginalized mechanism satisfies the registered task/gate for the frozen seeds, lineages, representation, and known delay law.

It does not establish generic reinforcement learning, unknown-delay inference, source identification, causal discovery, convergence guarantees, production readiness, or a theorem about NumPy execution.

## 14. Non-goals

C4 does not:

- change YOLO perception;
- train the recurrent reservoir;
- infer the realized hidden delay;
- reconstruct individual latent rewards from a collision;
- learn a delay distribution;
- use source-visible supervision;
- add a Transformer/RNN credit network;
- retune Phase 3C thresholds;
- replace or amend the frozen Phase 3C result;
- treat Task 12's privileged references as production algorithms;
- claim that the `block10` shuffled control is a perfect independence test.

## 15. Review decision

Approval of this document authorizes only the next planning step: write a separate C4 implementation plan with TDD tasks, formal/Python gates, two measurement checkpoints, exact file ownership, and scoped commits.

It does **not** authorize production implementation or C4-A/C4-B measurement by itself.
