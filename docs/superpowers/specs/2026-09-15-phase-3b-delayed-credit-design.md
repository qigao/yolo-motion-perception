# Phase 3B Delayed Action-to-Reward Credit Design

> Status: design proposal, committed for review. No Phase 3B production implementation is included in this change.
>
> Branch: `experiment/neural-state-machine`
>
> Date: 2026-09-15

## 1. Purpose

Phase 3A established a narrow result: a normalized action-value readout can be trained with an immediate scalar terminal reward on the frozen delayed-cue task, but the fixed acceptance gate was not met on all pre-registered seeds. Its cue-to-decision delay is not an action-to-reward delay: the reward is delivered immediately after the selected action.

Phase 3B tests the missing capability directly:

> Can the same neural state and action-value boundary acquire a useful action credit when the environment delivers the scalar reward only after a true action-to-reward delay?

The experiment must separate three effects:

1. the recurrent substrate retaining information until the decision;
2. the learner assigning a later scalar reward to the earlier action;
3. the readout's ability to preserve a useful policy under delayed feedback.

Phase 3B is an experiment about delayed credit assignment. It is not an upgrade to Phase 3A evidence, and it does not introduce game integration, FlyVis, fly-brain, YOLO, or a new recurrent substrate.

## 2. Non-negotiable boundaries

The following remain frozen byte-for-byte:

- `src/neural_state_machine/policy.py`
- `src/neural_state_machine/reward_readout.py`
- `src/neural_state_machine/reward_learning.py`
- `src/neural_state_machine/learning_diagnostics.py`
- `src/neural_state_machine/memory_task.py`
- `src/neural_state_machine/memory_probe.py`
- `src/neural_state_machine/memory_benchmark.py`
- `docs/experiments/phase-2b-failure.json`
- `docs/experiments/phase-2c-diagnostics.json`
- `docs/experiments/phase-3a-action-value.json`

Phase 3B may add new modules, tests, scripts, documentation, and a new evidence artifact. It must not alter the Phase 3A learner's scientific behavior, thresholds, seed lineages, or committed hashes.

The learner receives only a finite decision-time hidden vector, legal numeric action indices, a seeded NumPy generator for training selection, and a finite scalar reward when that reward becomes available. It does not receive cue identity, correct action, delay labels, task phase, probe output, episode objects, or a timestamp that reveals the reward schedule.

The term `cue-to-decision delay` remains reserved for the existing task. `action-to-reward delay` means the number of environment steps between the selected action and delivery of the scalar reward.

## 3. Experimental protocol

### 3.1 Episode timeline

Each episode has two independently specified delays:

```text
cue appears
   -> recurrent settling for cue-to-decision delay d_c
   -> agent selects action a_t
   -> environment advances for action-to-reward delay d_r
   -> environment delivers one scalar terminal reward r
   -> learner applies credit to the stored action
```

The first implementation uses `d_c ∈ {1, 2, 3, 4, 5}` from Phase 3A and pre-registered `d_r ∈ {1, 3, 5}`. The reward remains terminal and scalar. No intermediate reward, counterfactual reward, or supervised target is introduced.

The environment owns the delayed delivery queue. The learner must not infer the reward from the action or from the delay value. A delayed reward is delivered exactly once; a missing or duplicate delivery is a protocol error.

### 3.2 Action and reward lineages

For each seed in the ordered set `[7, 17, 29]`:

- training fixture lineage remains `[seed, 0x54524149]`;
- evaluation fixture lineage remains `[seed, 0x4556414C]`;
- normal and control behavior action generators are independently reconstructed from `[seed, 0x33414354]`;
- reward-delay scheduling uses a separate lineage `[seed, 0x3352444C]`;
- shuffled-reward controls use `[seed, 0x33534846]` and preserve the exact reward multiset inside every fixed block.

Normal and shuffled runs must have byte-identical action sequences. Any action divergence invalidates the comparison rather than being interpreted as learning.

### 3.3 Learner boundary

The delayed environment must expose a small protocol equivalent to:

```python
selection = learner.select_for_training(hidden, legal_actions, action_rng)
# environment advances without exposing reward to learner
reward = environment.deliver_terminal_reward()
update = learner.learn(reward)
```

The learner has exactly one pending action credit. A second selection before feedback is an error. A second reward after feedback is an error. Invalid rewards must leave weights and pending state unchanged.

For a trace learner, trace reset at terminal episode boundaries is the primary protocol. A persistent-across-episode trace is a diagnostic arm only; it is not allowed to become the acceptance learner through post-hoc selection.

## 4. Pre-registered comparison arms

### Arm A: delayed TD(0) baseline

Use the existing normalized action-value update, with the reward delivered after `d_r` environment steps but applied once to the stored pending action. This arm measures how far the existing immediate-reward mechanism generalizes without temporal eligibility.

### Arm B: accumulating eligibility trace

Use a separate delayed-credit learner with an accumulating trace. The trace is updated at each selected-action decision, decays according to fixed `(discount, trace_decay)` parameters, and is reset at terminal episode boundaries. The initial parameter values are `discount=0.9` and `trace_decay=0.8`, inherited from the diagnostic comparison and frozen before measurement.

The trace arm must be evaluated on the same fixtures, action schedules, reward schedules, and delayed-reward queue as Arm A. It is a comparison, not a replacement for the Phase 3A learner.

### Arm C: supervised reference

Retain the existing supervised reference only as a descriptive upper baseline for the frozen feature family. It must not be used as a target, reward source, or acceptance threshold for the delayed learner.

### Explicitly excluded arms

The first Phase 3B implementation will not include actor-critic, BPTT, STDP, recurrent-weight training, replay buffers, prioritized replay, reward shaping, adaptive delay-dependent step sizes, or automatic hyperparameter search. Each would confound delayed credit with a new learning architecture.

## 5. Controls and integrity checks

Every measured seed and reward delay includes these controls:

1. **Immediate-reward continuity control**: `d_r=0`, proving that the Phase 3B harness reproduces the known immediate-feedback boundary.
2. **Within-block shuffled-reward control**: rewards are permuted within fixed blocks while preserving each block's multiset; this must not pass.
3. **Reset control**: evaluate with the learner's value parameters reset to exact zero while preserving the same frozen hidden trajectories.
4. **Action-lineage control**: normal and shuffled runs use independently reconstructed identical behavior RNG lineages; action digests must match exactly.
5. **Fixture-digest control**: normal and trace arms share byte-identical training and evaluation fixtures.
6. **Queue integrity control**: each action receives exactly one reward after the registered delay; queue underflow, overflow, duplicate, and stale delivery are hard failures.
7. **Numerical integrity control**: hidden states, traces, rewards, candidate parameters, and committed parameters must remain finite; overflow is rejected without mutating pending state or weights.

The reset and shuffled controls are diagnostic checks, not alternate opportunities to tune thresholds.

## 6. Fixed configuration and acceptance gate

Unless this design is amended before measurement, use:

- seeds: ordered `[7, 17, 29]`;
- training episodes: `2000`;
- evaluation episodes: `200` per reward-delay setting;
- cue-to-decision delays: `[1, 2, 3, 4, 5]`;
- action-to-reward delays: `[0, 1, 3, 5]`, where `0` is the continuity control;
- hidden size: `64`;
- recurrent radius: `0.9`;
- step size: `0.1`;
- TD(λ) parameters: `discount=0.9`, `trace_decay=0.8`;
- checkpoint interval: `100` training episodes.

For each non-control reward delay and each seed, reuse the Phase 3A gate without relaxation:

- post-training overall at least `180/200`;
- every cue-to-decision delay at least `34/40`;
- reset exactly `100/200` overall and `20/40` per cue-to-decision delay;
- shuffled reward below `150/200`;
- normal and shuffled action sequences equal;
- all queue, finite-value, fixture, and repeatability checks pass.

The Phase 3B result is reported separately for each `(seed, d_r, arm)` and as a pooled summary. No threshold, seed, delay, episode count, or fixture is changed after observing results. A failed gate is retained as failure evidence.

## 7. Evidence and decision rules

Create a new immutable artifact:

```text
docs/experiments/phase-3b-delayed-credit.json
```

The artifact records protocol version, ordered seeds, both delay axes, arm identity, fixture and action digests, queue statistics, post/reset/shuffled counts, checkpoint summaries, parameter digests, repeatability, finite-value checks, and `all_passed`. It must be generated only by an approved-path writer and verified by a fail-closed verifier. The Phase 3A artifact is never rewritten.

Interpretation is pre-registered:

- If Arm A passes at nonzero `d_r`, immediate reward was not necessary for this task's action-value acquisition.
- If Arm A fails but Arm B passes, the evidence supports a delayed-credit limitation of TD(0) and a benefit from eligibility traces under this protocol; it does not prove general intelligence or universal trace superiority.
- If both fail while the immediate control reproduces Phase 3A behavior, the main failure is associated with delayed credit rather than the harness itself.
- If the immediate control also changes, stop and debug the harness before interpreting delayed-credit results.
- If normal and shuffled action sequences diverge, the comparison is invalid.

A passing Phase 3B result still does not establish emergent state machines, game competence, or a replacement for explicit safety rules. It only establishes the measured delayed-reward capability under this small deterministic protocol.

## 8. Implementation order after design approval

1. Add the delayed-reward queue as a standalone testable protocol object; no learner changes.
2. Add RED tests for queue timing, exactly-once delivery, stale/duplicate rewards, and immediate-control equivalence.
3. Add the delayed benchmark harness around the existing learner boundary.
4. Add Arm A and verify the `d_r=0` continuity control against the frozen Phase 3A behavior boundary.
5. Add the episode-reset trace learner as Arm B; keep persistent traces diagnostic-only.
6. Add shuffled, reset, action-lineage, fixture, and numerical-integrity tests.
7. Add the evidence writer/verifier and immutable artifact.
8. Run full local and exact-head CI verification.
9. Record one decision; do not proceed to game integration or neural-state interpretation until the delayed-credit result is reviewed.

No implementation begins until this design is reviewed and approved. The next implementation plan must be a separate committed artifact after design approval.
