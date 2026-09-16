# Phase 3B Delayed Action-to-Reward Credit Design

> Status: **protocol correction committed for review**. The original single-pending serialized protocol is invalid for delayed-credit interpretation and is superseded by this revision. No Phase 3B production implementation is included in this change.
>
> Branch: `experiment/neural-state-machine`
>
> Date: 2026-09-15
>
> Audit: `docs/experiments/phase-3b-harness-audit.md`

## 1. Purpose

Phase 3A established a narrow result: a normalized action-value readout can be trained with an immediate scalar terminal reward on the frozen delayed-cue task, but the fixed acceptance gate was not met on all pre-registered seeds. Its cue-to-decision delay is not an action-to-reward delay: the reward is delivered immediately after the selected action.

The first Phase 3B harness attempted to add action-to-reward delay by selecting one action, waiting `d_r` queue steps, delivering its reward, learning, and only then selecting the next action. Full diagnostics showed that `d_r ∈ {0,1,3,5}` produced identical action, reward, and parameter digests for every measured seed and arm. This equality was not evidence that delayed credit had been solved. The harness never allowed a later decision to occur while an earlier reward remained pending.

The audit therefore classifies the measured harness as **invalid for delayed-credit interpretation**. No Phase 3B acceptance artifact is frozen from that run.

This revision corrects the protocol itself.

Phase 3B now asks:

> Can the same neural state and action-value boundary learn when rewards for earlier actions remain pending while the agent continues to make later decisions?

The experiment must separate three effects:

1. the recurrent substrate retaining cue information until each decision;
2. the learner retaining multiple unresolved action credits while later decisions occur;
3. the learner assigning each later scalar reward to the correct earlier decision without being told the reward delay.

Phase 3B remains an experiment about delayed credit assignment. It is not an upgrade to Phase 3A evidence, and it does not introduce game integration, FlyVis, fly-brain, YOLO, recurrent-weight training, or a new recurrent substrate.

## 2. Non-negotiable boundaries

The following remain frozen byte-for-byte:

- `src/neural_state_machine/policy.py`
- `src/neural_state_machine/reward_readout.py`
- `src/neural_state_machine/reward_learning.py`
- `src/neural_state_machine/learning_diagnostics.py`
- `src/neural_state_machine/memory_task.py`
- `src/neural_state_machine/memory_probe.py`
- `src/neural_state_machine/memory_benchmark.py`
- `src/neural_state_machine/action_value.py`
- `docs/experiments/phase-2b-failure.json`
- `docs/experiments/phase-2c-diagnostics.json`
- `docs/experiments/phase-3a-action-value.json`

In particular, `NormalizedActionValue` remains the frozen Phase 3A single-pending learner. Phase 3B must not change its behavior to support multiple unresolved actions. Any multi-inflight behavior belongs in new Phase 3B-only learner adapters.

The learner receives only:

- a finite decision-time hidden vector;
- legal numeric action indices;
- a seeded NumPy generator for training selection;
- one finite scalar reward when an environment delivery occurs.

The learner must not receive cue identity, correct action, cue-to-decision delay, action-to-reward delay, task phase, probe output, episode objects, environment timestamps, or a reward-schedule label.

The environment may retain action identity internally only to audit exactly-once action-to-reward association. That action identity is not passed back through the learner reward API.

The term `cue-to-decision delay` remains reserved for the existing task. `action-to-reward delay d_r` now means the number of **later action selections that occur before the reward for an earlier action is delivered** for interior training decisions.

## 3. Corrected experimental protocol

### 3.1 Decision-step timeline

Training proceeds on a global decision-step clock `t = 0, 1, 2, ...`.

At each decision step:

```text
1. compute the frozen hidden state for training item t
2. select action a_t
3. compute terminal scalar reward r_t internally
4. enqueue credit C_t = (a_t, r_t) with due step t + d_r
5. deliver every reward whose due step equals t
6. apply each delivered scalar reward to the oldest matching unresolved learner credit
7. advance to decision step t + 1
```

The critical ordering is **select first, deliver second**. Therefore a due reward at step `t` is delivered only after the action for step `t` has already been selected.

The observable timelines are:

```text
d_r = 0:
  a0 -> r0 -> a1 -> r1 -> a2 -> r2

d_r = 1:
  a0 -> a1 -> r0 -> a2 -> r1 -> a3 -> r2

d_r = 3:
  a0 -> a1 -> a2 -> a3 -> r0 -> a4 -> r1 -> ...

d_r = 5:
  a0 -> a1 -> a2 -> a3 -> a4 -> a5 -> r0 -> ...
```

Thus, for every interior decision `t` with at least `d_r` later training decisions available, exactly `d_r` later actions are selected before `r_t` is delivered.

This overlap is the defining validity property of Phase 3B. Merely waiting for wall-clock time or advancing an otherwise unobserved queue is not delayed credit.

### 3.2 Terminal drain

After the final training action is selected, no synthetic decisions are invented merely to drain the queue.

Instead, the environment advances empty delivery steps until all remaining rewards are delivered exactly once. Tail deliveries are bookkeeping only and are excluded from the interior overlap-count assertion because no later real action exists after the final training decision.

The final state must satisfy:

```text
queue.pending_count == 0
learner.unresolved_credit_count == 0
delivery_count == action_count
```

A missing, duplicate, stale, out-of-order, or extra reward is a hard protocol failure.

### 3.3 Fixed-delay runs

Each measured run uses one fixed `d_r` from:

```text
[0, 1, 3, 5]
```

The protocol does not randomize reward delay within a run. This preserves FIFO delivery order and keeps the experiment focused on delayed credit rather than schedule inference.

Runs for different `d_r` values use byte-identical training and evaluation fixtures and independently reconstructed action RNGs from the same lineage.

### 3.4 Multi-inflight credit boundary

A valid Phase 3B learner must be able to retain more than one unresolved decision.

Conceptually, each decision captures immutable credit state equivalent to:

```text
PendingCredit
  action_index
  frozen feature vector
  normalization denominator
  prediction at decision time
  decision sequence number
```

The Phase 3B learner stores these credits in decision order. Because `d_r` is fixed inside a run, reward delivery order is also decision order. Therefore the learner may consume the oldest unresolved credit whenever it receives the next scalar reward.

The learner API remains reward-only:

```python
selection = learner.select_for_training(hidden, legal_actions, action_rng)
# later decisions may happen here
update = learner.learn(delivered_reward)
```

`learn()` must not accept `action_index`, `delay`, `due_step`, or any environment identifier.

The environment separately verifies that the action identity attached to each delivered queue record matches the learner credit consumed in FIFO order. Association mismatches invalidate the run.

## 4. Comparison arms

### Arm A: multi-inflight delayed TD(0) baseline

Arm A preserves the Phase 3A normalized update equation while allowing multiple unresolved decisions.

For each action, the learner freezes the same decision-time feature, denominator, and prediction that Phase 3A would have stored in its single `_PendingCredit`. When that action's scalar reward is eventually delivered, the learner applies exactly one TD(0)-style normalized update using the frozen decision-time credit.

No later hidden state, later action value, delay value, or bootstrapped target is introduced.

For `d_r=0`, Arm A must be numerically equivalent to the frozen Phase 3A learner on the same fixture and RNG lineages. This is an exact continuity control, not merely a similar-accuracy check.

### Arm B: eligibility-trace comparison is temporarily blocked

The original design specified an episode-reset accumulating eligibility trace. The corrected overlapping protocol exposes an unresolved semantic problem: rewards can arrive after later episode decisions have already occurred, so resetting a trace at the original episode boundary can erase credit before its reward is delivered, while retaining it can mix unrelated later decisions.

Therefore the existing `EpisodeResetEligibilityTrace` is **not approved for corrected Phase 3B measurement yet**.

Arm B remains a planned comparison, but its overlapping-trial trace/reset semantics require a separate design amendment after Arm A proves the corrected timeline is valid. No Phase 3B scientific comparison may claim TD(0) versus TD(λ) benefit until that amendment is reviewed and committed before measurement.

This restriction prevents a second protocol ambiguity from being hidden inside the harness correction.

### Arm C: supervised reference

The existing supervised reference remains a descriptive upper baseline for the frozen feature family. It is not a reward source, target source, delayed learner, or acceptance threshold.

### Explicitly excluded arms

The corrected protocol still excludes actor-critic, BPTT, STDP, recurrent-weight training, replay buffers, prioritized replay, reward shaping, adaptive delay-dependent step sizes, automatic hyperparameter search, and variable per-action reward delays.

## 5. Lineages and controls

For each seed in the ordered set `[7, 17, 29]`:

- training fixture lineage remains `[seed, 0x54524149]`;
- evaluation fixture lineage remains `[seed, 0x4556414C]`;
- each delay run reconstructs its behavior action generator from `[seed, 0x33414354]`;
- shuffled-reward controls use `[seed, 0x33534846]` only for the reward permutation itself.

### 5.1 Action and reward digests are controls, not delay-validity evidence

The training policy currently samples actions from the seeded behavior RNG rather than choosing from learned values. Therefore action sequences may legitimately be identical across `d_r` values.

Because the task reward is a deterministic function of fixture plus action, training reward sequences may also legitimately be identical across `d_r` values.

Accordingly:

- equality of `action_digest` across delays is allowed and expected;
- equality of `training_reward_digest` across delays is allowed and expected;
- equality or inequality of `parameter_digest` across delays is an observed scientific result, not a structural validity condition.

The invalid first harness was diagnosed not because the digests were equal by themselves, but because direct timeline inspection proved that no later decision occurred before reward delivery.

### 5.2 Required timeline evidence

Every measured run must record enough information to prove that the delay was actually exercised. At minimum record:

- `action_count`;
- `delivery_count`;
- `terminal_drain_count`;
- `max_pending_before_delivery`;
- `max_pending_after_delivery`;
- `decisions_with_prior_feedback_pending`;
- a histogram of `delivery_step - decision_step`;
- an immutable `delivery_timeline_digest` derived from decision sequence, decision step, due step, and delivery step;
- queue and learner unresolved counts after terminal drain.

For a run with `N` training decisions and fixed `d_r`, the structural assertions are:

```text
d_r = 0:
  terminal_drain_count == 0
  max_pending_after_delivery == 0
  every delivery_step - decision_step == 0

0 < d_r < N:
  exactly N - d_r rewards are delivered during real decision steps
  exactly d_r rewards are delivered during terminal drain
  every interior delivery_step - decision_step == d_r
  max_pending_after_delivery == d_r
  max_pending_before_delivery == d_r + 1
  decisions_with_prior_feedback_pending > 0
```

With the registered `N=2000` and `d_r ∈ {1,3,5}`, these assertions must hold exactly.

A run that fails any timeline assertion is classified **harness invalid** and stops before scientific acceptance logic.

### 5.3 Immediate continuity control

For every registered seed, `d_r=0` under the new multi-inflight Arm A adapter must reproduce the frozen immediate Phase 3A update behavior on the same training fixtures and action RNG lineage.

Continuity checks include:

- identical action sequence;
- identical reward sequence;
- identical final parameter digest;
- identical post-training evaluation counts;
- identical reset evaluation counts where the same evaluation protocol applies.

Failure means the new adapter changed the Phase 3A learning equation and Phase 3B measurement must stop.

### 5.4 Shuffled-reward control

The shuffled control preserves action sequence and the reward multiset inside each registered block while changing which earlier action receives which later scalar reward.

The permutation changes reward values only. It does not change decision steps, due steps, queue occupancy, or delivery timing.

Normal and shuffled runs therefore must have identical action and delivery-timeline digests while differing in the reward-assignment digest when the permutation is non-identity.

### 5.5 Queue integrity control

For every run:

- each selected action creates exactly one queue entry;
- each queue entry is delivered exactly once;
- FIFO association must match fixed-delay decision order;
- no delivery may become stale;
- no queue entry may be silently discarded during terminal drain;
- queue count and learner unresolved-credit count must agree at every audited boundary.

## 6. Fixed configuration

Unless amended before measurement, retain:

- seeds: ordered `[7, 17, 29]`;
- training decisions: `2000`;
- evaluation episodes: `200` per reward-delay setting;
- cue-to-decision delays: `[1, 2, 3, 4, 5]`;
- action-to-reward delays: `[0, 1, 3, 5]`;
- hidden size: `64`;
- recurrent radius: `0.9`;
- step size: `0.1`;
- checkpoint interval: `100` training decisions.

The previous TD(λ) parameters `discount=0.9` and `trace_decay=0.8` remain historical pre-registration data but are not active Phase 3B measurement parameters until Arm B's corrected semantics are separately approved.

## 7. Acceptance sequencing

Phase 3B now has two gates that must occur in order.

### Gate P: protocol validity

Before any delayed-credit scientific result is interpreted, all of the following must pass:

1. `d_r=0` exact Phase 3A continuity;
2. `d_r>0` structural overlap assertions;
3. exact decision-to-delivery lag assertions;
4. queue/learner occupancy agreement;
5. exactly-once delivery and zero unresolved terminal state;
6. repeatability on identical seed/configuration;
7. finite-value checks;
8. fixture lineage checks;
9. shuffled-control timing identity.

Gate P failure means only **harness invalid**. Accuracy and parameter results from that run are not Phase 3B evidence.

### Gate A: Arm A scientific measurement

Only after Gate P passes may Arm A be interpreted with the existing Phase 3A behavioral thresholds:

- post-training overall at least `180/200`;
- every cue-to-decision delay at least `34/40`;
- reset exactly `100/200` overall and `20/40` per cue-to-decision delay where the reset protocol is unchanged;
- shuffled reward below `150/200`;
- all protocol, finite-value, fixture, and repeatability checks pass.

Results are reported separately for each `(seed, d_r)` and as a pooled descriptive summary. Thresholds, seeds, delays, episode counts, and fixtures are not changed after observing results.

Arm B receives no acceptance interpretation until its follow-up semantics are committed.

## 8. Evidence rules

Do not create or freeze `docs/experiments/phase-3b-delayed-credit.json` until Gate P has passed on the exact implementation intended for measurement.

Before that point, diagnostic output may be emitted locally or in CI logs but is explicitly non-acceptance evidence.

The eventual immutable artifact must record:

- protocol version;
- ordered seeds;
- both delay axes;
- arm identity;
- fixture/action/reward/timeline/parameter digests;
- queue and learner occupancy statistics;
- delivery-lag histogram;
- terminal drain count;
- post/reset/shuffled counts;
- checkpoint summaries;
- repeatability and finite-value checks;
- separate `protocol_valid` and `behavior_passed` fields;
- final `all_passed` derived fail-closed from both.

A one-off CI measurement job must not be left permanently in the workflow merely to capture evidence. Measurement workflow changes must either be part of the approved reusable verification path or removed after audit, as done for the invalid first harness.

## 9. Interpretation rules

The corrected interpretation is pre-registered:

- If Gate P fails, conclude **harness invalid** and make no delayed-credit claim.
- If Gate P passes and Arm A passes at nonzero `d_r`, immediate reward was not necessary for useful action-value acquisition under this fixed-delay overlapping protocol.
- If Gate P passes and Arm A degrades as `d_r` increases, record the degradation without changing thresholds or immediately attributing it to a specific algorithmic remedy.
- If Gate P passes and Arm A fails while `d_r=0` reproduces Phase 3A, the evidence supports a limitation associated with overlapping delayed feedback under this protocol.
- No claim about eligibility traces is permitted until Arm B's semantics are separately approved and measured.
- Equality of action or reward digests across delays is not evidence of harness invalidity.
- Equality of parameter or accuracy results across delays is scientifically allowed; timeline evidence, not outcome divergence, proves that delay was exercised.

A passing result still does not establish emergent state machines, general game competence, general reinforcement learning ability, or a replacement for explicit safety rules.

## 10. Implementation order after this correction is approved

1. Add RED protocol tests that fail against the current serialized harness and prove that `d_r>0` must overlap later decisions.
2. Add timeline/occupancy instrumentation and exact structural assertions before changing learner behavior.
3. Add a new Phase 3B multi-inflight TD(0) adapter without modifying `NormalizedActionValue`.
4. Rewrite the training loop to select first, deliver due rewards second, and terminal-drain only after all real training decisions.
5. Prove `d_r=0` exact continuity against the frozen Phase 3A learner.
6. Prove `d_r={1,3,5}` exact lag, queue high-water, and terminal-drain invariants.
7. Rebuild shuffled and reset controls on the corrected timeline.
8. Run Gate P only. Do not freeze behavioral evidence yet if protocol validity is not exact.
9. If Gate P passes, run and review Arm A measurement.
10. Separately design and approve overlapping TD(λ) trace/reset semantics before implementing Arm B.
11. Only after the approved measured arms are valid, generate and verify the immutable Phase 3B artifact.

No production implementation begins until this corrected design is reviewed and approved. The existing implementation plan predates this protocol correction and must not be executed as written; a replacement implementation plan is required after approval.
