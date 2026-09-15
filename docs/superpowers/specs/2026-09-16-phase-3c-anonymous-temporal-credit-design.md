# Phase 3C Anonymous Temporal Credit Design

> Status: **pre-registered design committed for review**. No Phase 3C production implementation or behavioral measurement is included in this change.
>
> Repository: `qigao/yolo-motion-perception`
>
> Branch: `experiment/phase3c-anonymous-temporal-credit`
>
> Base: clean Phase 3B evidence head `a5ab079d56fe569aded44348f9591d226ce83009`
>
> Date: 2026-09-16
>
> Formal reference: `qigao/lean@c59c1a8df9ece630a7f16747475554f62700ecb4`, module `NarrativeDynamics/Core/TemporalCredit.lean`

## 1. Purpose

Phase 3B established a corrected delayed-feedback protocol: rewards for earlier actions remain pending while later real decisions occur, the delay is structurally audited, and the frozen evidence is protocol-valid. Its Arm A result is a valid negative behavioral result across the registered seeds.

Phase 3B nevertheless retains a decisive simplification. Fixed delay preserves delivery order, and the Phase 3B learner stores unresolved credits in FIFO order. Although `learn()` receives only a scalar reward, the next reward is mechanically associated with the oldest unresolved action. The learner therefore does not need to infer which part of recent history is responsible for the feedback.

Phase 3C removes that ordering oracle.

Phase 3C asks:

> Can a learner acquire useful action values when each action produces a latent reward after an independently hidden delay, later rewards may arrive before earlier rewards, multiple latent rewards may collide at the same delivery step, and the learner observes only the aggregate scalar feedback stream rather than source identities or delivery multiplicity?

This phase is about **anonymous temporal credit assignment**, not merely delayed update.

The intended progression is:

```text
Phase 3A
immediate identified feedback
        ↓
Phase 3B
delayed feedback + FIFO-identifiable source order
        ↓
Phase 3C
delayed + out-of-order + colliding + anonymous aggregate feedback
```

A positive Phase 3C result would remain narrow: it would show useful learning under this registered anonymous delayed-feedback protocol. It would not establish general causal discovery, general reinforcement learning, emergent intelligence, or correctness of a production perception system.

## 2. Frozen scientific base

Phase 3C starts from the clean Phase 3B head `a5ab079d56fe569aded44348f9591d226ce83009`.

The following Phase 3B facts are frozen inputs to this design:

- scientific implementation head: `42a4b26f80969a0abd6f878248ec860365b699c1`;
- frozen corrected Phase 3B evidence SHA-256: `dbe263aa916ca0e2bbf3281b20d56de7d6cee70c38f08146c96eb9987e024074`;
- Phase 3B Gate P passes;
- Phase 3B Arm A aggregate behavioral gate fails;
- `d_r=0` exactly reproduces the frozen Phase 3A normalized TD(0) boundary;
- Phase 3B delivery identity is still recoverable by FIFO ordering and is therefore not the target capability of Phase 3C.

Phase 3C must not rewrite or reinterpret frozen Phase 2/3A/3B evidence.

The following existing components remain frozen unless a later reviewed amendment explicitly says otherwise:

- recurrent policy / reservoir implementation;
- delayed-cue task fixture generation;
- evaluation fixture generation;
- Phase 3A `NormalizedActionValue`;
- Phase 3B evidence and verifier;
- existing Phase 2/3A evidence artifacts.

New Phase 3C behavior belongs in Phase 3C-only modules and tests.

## 3. Formal foundation and its boundary

The current formal reference is:

```text
qigao/lean@c59c1a8df9ece630a7f16747475554f62700ecb4
NarrativeDynamics/Core/TemporalCredit.lean
```

That model machine-checks the following reusable properties:

1. `reward_invariant_under_distractor_substitution` — terminal reward can be defined independently of distractor actions;
2. `learner_view_independent_of_causal_label` — internal causal-label metadata can be excluded from the learner-visible projection;
3. `terminal_td0_zero_direct_causal_credit` — terminal TD(0) gives zero direct credit to an earlier causal position when at least one later decision exists;
4. `terminal_td0_terminal_credit` — terminal TD(0) directly credits the terminal position;
5. `causal_trace_coeff_closed_form` — eligibility contribution decays as `(gamma * lambda)^d`;
6. `causal_trace_coeff_ne_zero` — that direct temporal credit path remains non-zero for finite `d` when both factors are non-zero.

Those theorems are relevant but **insufficient to prove Phase 3C**. They do not model:

- multiple latent rewards in flight;
- per-action variable hidden delays;
- out-of-order delivery;
- collision of multiple rewards into one aggregate scalar;
- equivalence of learner view under different source decompositions with the same aggregate stream;
- the proposed normalized anonymous eligibility update;
- exact reduction of that update to Phase 3A at the immediate zero-trace boundary.

Therefore Phase 3C has a formal gate, Gate F. Before behavioral measurement, the Lean model must be extended and exact-head checked for the missing anonymous-aggregation properties described in Section 10. Current Lean v1 must never be cited as proving the Python Phase 3C implementation or the anonymous aggregate protocol.

## 4. Anonymous feedback protocol

### 4.1 Global decision clock

Training uses a global decision clock `t = 0, 1, 2, ...`.

For each real training decision `t`:

```text
1. obtain frozen delayed-cue hidden state h_t
2. select action a_t
3. compute latent scalar reward r_t internally
4. read pre-generated hidden delay d_t
5. enqueue internal delivery record (source=t, reward=r_t, due=t+d_t)
6. collect all records due at t
7. aggregate them into exactly one scalar F_t = sum(reward_i)
8. pass only F_t to the learner
9. advance to t+1
```

The ordering remains **select first, aggregate feedback second**. A reward delivered at step `t` is therefore observed only after the action for step `t` has already been selected.

The learner receives one scalar feedback value at every real decision step, including `0.0` when no latent rewards are due or when simultaneous rewards cancel numerically.

This is intentional. The learner is not told whether `F_t = 0` means:

- no latent reward arrived;
- one zero-valued reward arrived;
- multiple rewards arrived and summed to zero.

Delivery multiplicity is environment-internal audit data only.

### 4.2 Hidden delay schedule

The registered anonymous condition uses per-action delays drawn independently from:

```text
[1, 3, 5]
```

with equal probability from a dedicated delay generator.

For ordered seed `seed`, the delay generator lineage is:

```text
[seed, 0x3343444C]
```

The complete delay schedule is generated **before action selection begins**. Delay generation must not depend on hidden state, selected action, latent reward, learner parameters, or observed behavior.

The delay generator is environment-only. It is never passed to the learner.

The same pre-generated schedule for a seed is reused byte-for-byte across all registered learner arms and their matched controls.

The implementation must record the schedule digest and structural schedule statistics before any behavioral interpretation.

No delay schedule may be regenerated because an arm performs poorly. A structurally invalid registered schedule invalidates the run rather than causing resampling.

### 4.3 Out-of-order delivery

For actions `i < j`, define an inversion when:

```text
due(j) < due(i)
```

At least one inversion must occur in every registered seed schedule. The exact inversion count and digest are evidence fields.

The learner is not told which latent reward arrived first, which source it came from, or that an inversion occurred.

### 4.4 Colliding delivery

A collision occurs when two or more source actions have the same due step.

For each delivery step:

```text
F_t = sum(r_i for every source i with due(i)=t)
```

At least one collision must occur in every registered seed schedule. The internal multiplicity histogram is recorded for protocol audit but is not learner-visible.

The learner receives only `F_t`.

### 4.5 Terminal drain

After the final real training action, no synthetic action is created to drain pending rewards.

The environment advances delivery-only steps until every latent reward has been consumed exactly once. During drain:

- no action is selected;
- no new eligibility contribution is added;
- the learner still receives exactly one aggregate scalar for every drain step;
- eligibility decay is **decision-based**, not wall-clock based, so no additional trace decay occurs when no real decision occurs;
- Arm A has no current decision credit during drain, so drain feedback is an explicit no-op for its parameters while still being recorded as observed feedback;
- Arm B applies drain feedback to its persistent pre-drain eligibility/prediction trace without adding or decaying trace state.

The final audit state must satisfy:

```text
pending latent records == 0
latent rewards delivered == real actions selected
sum(all aggregate feedback) == sum(all latent rewards)
```

The tail is reported separately because the final actions cannot have their full registered number of later real decisions before delivery. Arm A's no-op drain behavior is part of its registered absence of historical credit, not an environment omission.

## 5. Learner information boundary

### 5.1 Selection API

At action selection the learner may receive only:

- finite decision-time hidden vector;
- legal numeric action indices;
- the registered behavior-action RNG.

### 5.2 Feedback API

At feedback time the learner may receive only:

```python
learn(aggregate_reward)
```

or an equivalent API carrying exactly one finite scalar.

The learner must not receive:

- source action index;
- source decision sequence;
- source hidden state;
- source count / collision multiplicity;
- due step;
- selected delay;
- delay distribution label;
- global environment timestamp;
- queue contents;
- a delivery-present boolean distinct from the scalar itself;
- latent individual rewards;
- causal label;
- correct action;
- cue identity or cue-to-decision delay.

The Phase 3B `_PendingCredit` FIFO model is explicitly forbidden in the Phase 3C anonymous arms. A learner that pops the oldest unresolved action when scalar feedback arrives is a Phase 3B learner, not a Phase 3C learner.

### 5.3 Environment audit data

The environment may retain full source metadata solely for protocol verification:

```text
source decision
source action
latent reward
hidden delay
due step
delivery step
```

This data may appear in immutable evidence only as audit metadata/digests. It must never be passed into action selection or learning.

## 6. Registered learner arms

Phase 3C registers two primary arms on identical fixtures, action RNGs, and hidden delay schedules.

### Arm A — anonymous current-step TD(0) baseline

Arm A intentionally has no historical credit mechanism.

At decision `t` it freezes only the current Phase 3A-style normalized decision credit. After the environment computes anonymous aggregate `F_t`, Arm A applies the Phase 3A normalized update to the **current** selected action only:

```text
prediction_t = Q(h_t, a_t)
delta_t      = F_t - prediction_t
W[a_t]      += alpha * delta_t * phi_t / ||phi_t||^2
```

It stores no queue of earlier decisions. At terminal drain, where no current decision exists, aggregate feedback is observed/audited but produces no parameter update.

This arm is the mechanism-negative baseline. Under anonymous delayed feedback, the current action is generally not the source of the observed feedback.

### Arm B — normalized anonymous eligibility credit

Arm B distributes each anonymous aggregate error across recent action-state history without receiving source identity.

Let:

```text
rho = gamma * lambda
```

At every **real action selection**:

```text
E <- rho * E
P <- rho * P

E[a_t] <- E[a_t] + phi_t / ||phi_t||^2
P      <- P + Q_decision_time(h_t, a_t)
```

where:

- `E` has the same shape as the action-value weight matrix;
- `P` is a scalar prediction trace;
- each `Q_decision_time` contribution is frozen at the moment its action was selected.

When aggregate scalar `F_t` is observed:

```text
delta_t = F_t - P
W       <- W + alpha * delta_t * E
```

Feedback does not identify or remove a particular historical action.

Trace reset semantics are fixed as follows:

- initialize `E = 0` and `P = 0` exactly once at the start of each independent training run/arm/control replay;
- do **not** reset at delayed-cue fixture boundaries;
- do **not** reset when a latent reward is generated;
- do **not** reset when aggregate feedback is received;
- do **not** reset on a collision or out-of-order delivery;
- do **not** decay or reset during terminal-drain steps because no real decision occurs;
- clear the trace only after the run has fully drained and ended.

Thus the trace is global across the registered training stream. This explicitly avoids the episode-reset ambiguity that blocked eligibility-trace interpretation in corrected Phase 3B.

This mechanism is called **Normalized Anonymous Eligibility Credit** in Phase 3C. It must not be described as a full standard TD(lambda) implementation unless a later design proves semantic equivalence.

### 6.1 Fixed eligibility parameters

Register before measurement:

```text
gamma       = 0.9
lambda      = 0.8
rho         = 0.72
step size   = 0.1
```

No parameter sweep, delay-dependent parameter, adaptive decay, or post-measurement tuning is permitted in Phase 3C-A.

For reference, the formal scalar trace prediction is:

```text
age 0 -> 1
age 1 -> 0.72
age 3 -> 0.373248
age 5 -> 0.1934917632
```

These are mechanism coefficients, not pre-registered accuracy predictions.

## 7. Exact immediate continuity control

Phase 3C must retain a zero-delay, zero-history boundary that reduces exactly to frozen Phase 3A.

For this control only:

```text
all delays = 0
rho = 0
feedback for a_t is delivered immediately after selecting a_t
```

Then Arm B has:

```text
E[a_t] = phi_t / ||phi_t||^2
P = Q(h_t, a_t)
delta = r_t - Q(h_t, a_t)
```

and therefore its update is exactly the Phase 3A normalized action-value update.

The control must compare, for every registered seed:

- action sequence and action digest;
- latent reward sequence and digest;
- final parameter digest in the same environment;
- post-training counts;
- per-cue-delay counts;
- state-reset counts;
- training/evaluation fixture digests.

Any mismatch stops Phase 3C before anonymous measurement.

Arm A must satisfy the same immediate continuity boundary.

## 8. Randomness and fixture lineages

For ordered seeds:

```text
[7, 17, 29]
```

retain the existing lineages where already registered:

```text
training fixture     [seed, 0x54524149]
evaluation fixture   [seed, 0x4556414C]
action RNG            [seed, 0x33414354]
shuffled control RNG  [seed, 0x33534846]
```

Add only:

```text
anonymous delay RNG   [seed, 0x3343444C]
```

Action selection remains generated independently of learner values under the existing registered behavior policy. Therefore both Phase 3C arms must see identical action lineages for a given seed and condition.

The delay RNG is independent of action RNG and shuffled-control RNG.

Changing anonymous delay consumes no action RNG state.

## 9. Controls

### 9.1 Aggregate-source relabeling control

Take a fully generated internal delivery history and arbitrarily relabel source identifiers while preserving:

- action-visible input sequence;
- aggregate scalar feedback stream;
- timing of every aggregate scalar.

The learner output/update trace must remain identical.

This is an implementation-level non-interference check: source labels are audit metadata only.

### 9.2 Hidden multiplicity control

Two environment histories that produce the same aggregate scalar feedback stream but different internal source multiplicities must induce identical learner calls.

Example conceptual equivalence:

```text
internal history A: [+1, -1] arrive together -> F_t = 0
internal history B: no reward arrives       -> F_t = 0
```

The learner receives `0` in both cases and no separate delivery-count signal.

This control proves API anonymity, not that arbitrary observation histories are causally indistinguishable.

### 9.3 Shuffled latent-reward control

The existing action sequence, hidden delay schedule, due steps, and delivery multiplicities are held fixed.

Latent reward values are permuted across source decisions using only the registered shuffled-control RNG before aggregation.

The control therefore changes causal action/reward assignment while preserving the schedule structure.

Required invariants:

- normal and shuffled action digests are identical;
- delay schedule digests are identical;
- due-step and multiplicity digests are identical;
- only reward assignment / aggregate feedback values may change.

### 9.4 State-reset evaluation

Retain the established delayed-cue state-reset evaluation. It remains an evaluation control for reliance on recurrent state and is not part of anonymous source assignment.

## 10. Gate F — formal conformance prerequisite

Before Phase 3C behavioral measurement, `qigao/lean` must extend the formal model beyond `c59c1a8...` and exact-head CI must machine-check the following Phase 3C properties.

### F1. Aggregate-view source non-interference

For two internal delivery histories with the same learner-visible observations and identical aggregate scalar feedback stream, changing source labels or source decomposition must not change the formal learner view.

The theorem must cover different internal multiplicities that yield the same aggregate scalar.

### F2. Aggregate conservation

The formal aggregation operator must prove that summing aggregate feedback over all delivery steps equals summing every latent reward exactly once, assuming an exactly-once delivery schedule.

### F3. Out-of-order source identity remains hidden

Permuting internal source order while preserving the same aggregate stream must not alter learner view. No formal learner input may contain source sequence number, delay, due step, or multiplicity.

### F4. Eligibility historical coefficient

The Phase 3C trace model must retain the existing result that a decision separated by `d` later real decisions contributes coefficient:

```text
rho^d = (gamma * lambda)^d
```

No feedback-source identity may be required to derive this coefficient.

### F5. Immediate reduction

At the registered boundary:

```text
rho = 0
hidden delay = 0
one immediate reward
```

prove that the formal normalized anonymous eligibility update reduces to the Phase 3A normalized update equation.

### F6. Axiom audit

The new primary theorems must compile with:

- no `sorry`;
- no `admit`;
- no new custom axioms;
- explicit `#print axioms` evidence reviewed in CI.

Gate F proves mathematical properties of the formal model. Python conformance tests are still required; Gate F must never be presented as automatic proof of NumPy implementation correctness.

## 11. Gate P — Python protocol validity

Behavioral results are uninterpretable until every registered seed and arm passes Gate P.

Gate P requires:

1. exact immediate Phase 3A continuity for both arms at `delay=0, rho=0`;
2. delay schedule generated before actions and independent of actions/rewards;
3. every anonymous delay is in `[1,3,5]` and each registered delay value occurs at least once per seed schedule;
4. at least one genuine out-of-order delivery inversion per registered seed;
5. at least one genuine multi-source collision per registered seed;
6. one scalar learner feedback value per real decision step, including zero;
7. no source ID, delay, due step, source count, queue object, or delivery boolean crosses the learner API;
8. source-relabeling non-interference test passes;
9. hidden-multiplicity non-interference test passes;
10. every selected action creates exactly one latent reward record;
11. every latent reward record is delivered exactly once;
12. aggregate conservation holds;
13. terminal drain empties the internal queue without synthetic actions;
14. terminal-drain Arm A no-op and Arm B persistent-trace semantics match Section 4.5 exactly;
15. action lineage is identical between matched arms;
16. delay schedule/due-step/multiplicity lineage is identical between matched arms;
17. repeated run with identical seed/configuration is portable-repeatable under the established evidence rules;
18. all learner state and updates remain finite;
19. trace reset occurs only at registered run boundaries and never at fixture/reward/collision boundaries;
20. trace coefficients at registered ages match the formal recurrence within the explicitly specified floating-point tolerance used only for Python conformance, never as a replacement for the Lean theorem.

A Gate P failure means **harness invalid**. Accuracy from that run is not Phase 3C evidence.

## 12. Registered empirical configuration

Unless amended before any behavioral measurement:

```text
seeds                    [7,17,29]
training decisions       2000
evaluation episodes      200
cue-to-decision delays   [1,2,3,4,5]
anonymous hidden delays  [1,3,5], equal-probability pre-generated schedule
hidden size              64
recurrent radius         0.9
step size                0.1
gamma                     0.9
lambda                    0.8
rho                       0.72
checkpoint interval      100 real training decisions
```

No seed, threshold, delay support, fixture lineage, step size, gamma, lambda, schedule lineage, or episode count may change after behavioral results are observed.

## 13. Gate B — behavioral interpretation

Only after Gate F and Gate P pass may behavioral outcomes be interpreted.

Retain the existing per-seed quality gate for each primary arm:

- post-training overall at least `180/200`;
- every cue-to-decision delay at least `34/40`;
- state-reset exactly `100/200` overall and `20/40` per cue delay;
- shuffled latent-reward control below `150/200`;
- protocol, finite-value, fixture, formal-conformance, and repeatability checks all pass.

The primary comparison is Arm B versus Arm A on the same seed, action lineage, fixture lineage, and anonymous delay schedule.

Pre-registered interpretations are:

```text
Arm B passes, Arm A fails
  -> strongest evidence in this phase that historical eligibility credit is useful
     under the registered anonymous aggregate feedback protocol.

Arm B improves over Arm A but both fail the fixed quality gate
  -> historical credit helped descriptively but was insufficient to solve the
     registered task. Do not call Phase 3C successful.

Arm A and Arm B both pass
  -> the task does not discriminate the historical-credit mechanisms strongly
     enough; do not claim eligibility was necessary.

Arm A and Arm B both fail
  -> anonymous temporal credit remains unsolved under the registered mechanisms.

Shuffled control meets or exceeds its rejection boundary
  -> causal-learning interpretation is blocked even if post-training accuracy is high.
```

No accuracy law such as `accuracy ∝ rho^d` is pre-registered. The formal coefficient is a direct-credit mechanism property, not a prediction of final task accuracy.

## 14. Evidence schema

The eventual immutable Phase 3C evidence must include at least:

- schema/protocol version;
- Phase 3B clean-base SHA;
- formal Lean exact-head SHA used by Gate F;
- formal theorem names and axiom-audit status;
- ordered seeds and all fixed configuration;
- fixture/action/delay/reward/aggregate-feedback digests;
- latent record count and exactly-once delivery status;
- delay histogram;
- inversion count;
- collision-step count and internal multiplicity histogram;
- terminal drain count;
- aggregate-conservation result;
- learner-visible call-stream digest;
- source-relabeling and hidden-multiplicity non-interference results;
- trace-reset audit;
- trace-coefficient conformance probes;
- same-environment parameter digests;
- post/reset/shuffled counts;
- checkpoints;
- repeatability and finite-value fields;
- separate `formal_valid`, `protocol_valid`, and `behavior_passed` fields;
- fail-closed top-level status derived from all three.

A behavioral artifact must not be frozen if Gate F or Gate P fails.

Cross-Python/NumPy verification must continue to distinguish portable semantic evidence from environment-specific raw floating-point parameter bytes.

## 15. Required implementation boundaries

Phase 3C implementation must use new, narrowly scoped modules. The exact filenames are chosen in the implementation plan, but responsibilities must remain separated:

```text
anonymous schedule / aggregation
        ↓
protocol audit + controls
        ↓
Phase 3C-only learner arms
        ↓
benchmark orchestration
        ↓
evidence writer/verifier
```

Do not modify `NormalizedActionValue` to make it anonymous/multi-history capable.

Do not reuse Phase 3B FIFO unresolved-credit state inside either Phase 3C primary arm.

Do not combine protocol scheduling, learner state, evidence serialization, and acceptance logic into one module.

## 16. Implementation order after design approval

After this design is reviewed and approved:

1. write and commit the detailed Phase 3C implementation plan spanning the Lean Gate F work and the Python protocol/learner work;
2. execute the Lean TemporalCredit extension for Gate F and obtain exact-head proof CI;
3. add RED Python tests for anonymous API non-interference, out-of-order delivery, collisions, conservation, trace-reset semantics, drain semantics, and exact immediate continuity;
4. implement the hidden schedule and aggregate-feedback protocol without learner changes;
5. make Gate P protocol mechanics pass before adding behavioral interpretation;
6. implement Arm A current-step baseline;
7. implement Arm B normalized anonymous eligibility credit;
8. prove Python/formal conformance probes and immediate Phase 3A continuity;
9. add matched shuffled/state-reset controls;
10. run Gate F + Gate P only and review structural evidence;
11. only then perform the first registered behavioral measurement;
12. freeze that first valid measurement whether positive or negative;
13. do not retune parameters or thresholds after seeing the result.

## 17. Out of scope

Phase 3C-A excludes:

- random/learned recurrent weights beyond the frozen substrate;
- BPTT or recurrent-network gradient training;
- actor-critic;
- replay buffers or prioritized replay;
- attention-based credit assignment;
- learned delay models;
- explicit Bayesian source inference;
- multi-causal reward decomposition beyond anonymous summation;
- continuous-time credit;
- YOLO integration or production motion perception;
- FlyVis/fly-brain integration;
- ROS2;
- general causal-discovery claims;
- claims that Lean proves NumPy floating-point execution correct.

Those require separate designs after Phase 3C-A is measured and interpreted.

## 18. Acceptance for this design phase

This design phase is complete when:

1. this spec is committed on a branch derived directly from clean Phase 3B head `a5ab079d...`;
2. no Phase 3C production Python is added;
3. no behavioral Phase 3C measurement is run;
4. the existing Phase 3B branch/evidence remains unchanged;
5. the formal insufficiency of Lean v1 is explicit;
6. Gate F, Gate P, and Gate B are separated and fail closed;
7. randomness, learner-visible data, delay semantics, aggregation, trace reset/drain semantics, controls, thresholds, and interpretation are pre-registered before implementation.
