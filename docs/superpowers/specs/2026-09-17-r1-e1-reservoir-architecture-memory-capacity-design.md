# R1 Phase E1 — Reservoir Architecture and Memory Capacity Design

> Status: **design committed for review; implementation and measurement are not authorized by this document's existence**.
>
> Repository: `qigao/yolo-motion-perception`
>
> Branch: `research/r1-e1-reservoir-architecture`
>
> Base integration head: `d3888b305e659e9f99f40f2c73b11a705622136a`
>
> AutoESN reference: `Ro6ertWcislo/AutoESN@b3d2e287716176fc3e1312b5be2a7a0b91ba538e`
>
> Date: 2026-09-17
>
> Review boundary: spec only. After approval, write a separate implementation plan. Do not implement reservoir backends, run registered E1 measurements, alter R2 delayed-credit code/evidence, or tune hyperparameters at this checkpoint.

## 1. Research split and decision

The project now separates two research questions that must not be conflated:

- **R1 — Temporal State Representation:** how a recurrent reservoir converts observation history into a useful current latent state.
- **R2 — Delayed Credit Assignment:** how delayed, possibly anonymous outcomes are attributed to earlier actions.

R1 Phase E1 studies only the first question:

> **Under a fixed total reservoir-neuron budget and a fixed linear readout, how do shallow, grouped, and deep ESN architectures differ in memory horizon, history separability, and robustness to YOLO-like temporal corruption?**

The study deliberately contains no reward learning. Labels are used only by frozen supervised linear probes after reservoir states have been collected. Existing Phase 2B/2C/3A/3B/3C/C4 learners and evidence remain unchanged.

## 2. Why E1 is needed

The current baseline uses a single fixed random recurrent reservoir with `tanh`, hidden size `64`, recurrent spectral radius `0.9`, and a learned/fitted output layer. Phase 2A established that this representation retains the registered binary cue across delays 1–5, while later phases showed that reward-credit learning can fail even when the cue remains linearly decodable.

That result isolates representation from credit acquisition, but it does not answer whether the single shallow reservoir is a good architecture for longer memory, multi-event history separation, or noisy detector sequences.

AutoESN is a useful scientific reference because it compares shallow, deep, grouped, and grouped-deep ESNs under fixed neuron budgets and explicitly measures memory capacity. E1 adopts the architecture-comparison idea and fixed-budget discipline, but not the package as a runtime dependency and not its `GreedyESN` search logic.

## 3. Approaches considered

### 3.1 Import AutoESN and use its automatic model search

This would provide several reservoir architectures quickly, but it introduces an old dependency stack, changes numerical/runtime assumptions, and mixes architecture selection with validation-driven hyperparameter search. It would also make exact lineage and evidence control harder. **Rejected.**

### 3.2 Tune one larger shallow reservoir

This would measure capacity scaling but not architecture effects. It cannot answer whether grouping or depth changes memory/representation at equal neuron count. **Rejected as the primary E1 design.**

### 3.3 Implement minimal native shallow/grouped/deep reservoir backends and compare them under fixed budgets

Keep activation, spectral radius, input initialization law, readout, fixtures, labels, seeds, regularization, and scoring fixed. Change only reservoir topology and the registered total-neuron budget. **Selected.**

Grouped-deep ESN is deferred. E1 first establishes interpretable effects for the three simplest topology families.

## 4. Non-goals

E1 must not:

- modify or rerun any frozen R2 delayed-credit result;
- use reward, TD updates, eligibility traces, C4 marginalization, or online reinforcement learning;
- introduce LSTM, GRU, Transformer, S4, Mamba, or other recurrent/sequence models;
- tune spectral radius, leaking rate, input scaling, activation, ridge penalty, architecture count, or thresholds after seeing registered results;
- use AutoESN as a package dependency;
- copy or rely on `GreedyESN` model-selection behavior;
- claim that the best synthetic E1 arm is automatically the best NPC or YOLO production model;
- claim semantic latent states or causal behavior understanding from linear decodability alone.

Hyperparameter studies belong to a later **R1 E2 — Reservoir Dynamics** phase. Real YOLO/video integration belongs to a later phase after E1 is frozen.

## 5. Common reservoir contract

All E1 reservoir implementations must expose the same conceptual interface:

```text
reset()
advance(observation) -> state
state_dim
parameter_digest()
```

Required properties:

1. Reservoir weights are initialized once and remain fixed for a run.
2. No reservoir parameter has a gradient/update path.
3. `reset()` returns every recurrent subreservoir to its canonical zero state.
4. `advance()` consumes exactly one observation step and returns an immutable float64 state.
5. `state_dim` equals the registered total-neuron budget for every architecture arm.
6. `parameter_digest()` binds architecture, initialization and all fixed reservoir parameters.
7. Labels are never supplied to a reservoir.
8. Readout fitting happens only after state collection ends.

## 6. Fixed dynamics

The shallow cell remains the current baseline family:

```text
h_t = tanh(W_in x_t + W_rec h_{t-1})
```

For every recurrent subreservoir:

- activation: `tanh`;
- recurrent spectral radius: exactly `0.9`;
- leaking rate: exactly `1.0` (no extra leak term; continuity with the current recurrence);
- recurrent bias: disabled;
- input weights: zero-mean Gaussian with standard deviation `1/sqrt(input_dim)`;
- recurrent weights before radius normalization: zero-mean Gaussian with standard deviation `1/sqrt(hidden_dim)`;
- recurrent matrix is scaled to spectral radius `0.9` using the same policy as the current baseline;
- initial hidden state: all zeros;
- dtype: float64 for registered scientific measurements.

No input scaling multiplier is introduced in E1.

## 7. Registered architecture arms

Run two total-neuron budgets:

- **continuity/small budget:** `N = 64`;
- **primary architecture budget:** `N = 256`.

For each budget, register exactly five architectures.

### 7.1 Shallow

One reservoir of size `N`. Readout state is that reservoir state.

### 7.2 Grouped-2

Two independent reservoirs of size `N/2`, each receiving the same external observation. Concatenate both states in fixed group order.

### 7.3 Grouped-4

Four independent reservoirs of size `N/4`, each receiving the same external observation. Concatenate states in fixed group order.

### 7.4 Deep-2

Two serial reservoir layers of size `N/2` each. Layer 1 receives the external observation. Layer 2 receives the current output of layer 1. Concatenate both layer states in layer order, so final `state_dim = N`.

### 7.5 Deep-4

Four serial reservoir layers of size `N/4` each. Layer `k+1` receives the current state of layer `k`. Concatenate all layer states in layer order, so final `state_dim = N`.

No architecture may receive more trainable readout dimensions than another architecture at the same budget.

## 8. Deterministic lineages

Registered base seeds are:

```text
[7, 17, 29, 43, 61]
```

Seeds 7/17/29 preserve continuity with earlier project measurements; 43/61 add prospective reservoir realizations.

Every architecture/subreservoir gets an isolated deterministic RNG lineage derived from:

```text
SeedSequence([base_seed, 0x52314531, budget_id, architecture_id, component_id])
```

where identifiers are fixed integer enums committed before measurement. Fixture/noise RNGs use separate lineage tags and must never consume reservoir RNG state.

The same semantic fixture realization and corruption mask must be presented to every architecture arm for a given base seed.

## 9. Fixed readout

All R1 E1 supervised measurements use the same linear Ridge probe:

```text
regularization = 1e-6
bias = enabled
```

The probe is fitted only from frozen collected reservoir states and training labels. It must not alter reservoir parameters. Use a stable SVD/least-squares implementation; do not form an explicit matrix inverse.

For every fitted probe record:

- train/evaluation counts;
- state/design shape;
- rank and singular spectrum summary;
- coefficient digest;
- reservoir parameter digest;
- fixture digest;
- prediction digest.

No validation-driven architecture or hyperparameter selection occurs before registered evaluation.

## 10. E1-A — Binary memory-capacity curve

E1-A extends the existing delayed-cue idea into a longer fixed memory curve.

Registered delays:

```text
D = [1, 2, 5, 10, 20, 40, 80]
```

For every architecture/budget/seed:

- training: exactly `400` balanced episodes per delay, total `2800`;
- evaluation: exactly `40` balanced episodes per delay, total `280`;
- cue classes: binary and exactly balanced;
- decision-time/current observation is class-independent;
- labels are attached only after reservoir states are collected;
- the same Ridge probe law is used for every arm.

### 10.1 Reset negative control

For every paired left/right evaluation fixture, reset the reservoir immediately before the class-independent decision observation. Require paired reset states to be byte-identical and balanced reset classification to be exactly `140/280` overall and `20/40` for every delay.

A reset mismatch invalidates the memory measurement for that arm; it is not interpreted as poor memory.

### 10.2 Memory metrics

Report, without post-hoc threshold tuning:

- correct/total and accuracy for every delay;
- macro accuracy across the seven delays;
- **contiguous 85% memory horizon:** largest registered delay `d` such that every tested delay `<= d` is at least `85%` accurate;
- accuracy degradation from delay 1 to each later delay;
- train accuracy as an overfit/context metric, not as the primary outcome.

Do not collapse all evidence into a single winner score.

### 10.3 Phase 2A continuity check

The `N=64` shallow arm must additionally reproduce the existing Phase 2A behavior on the original delays 1–5 using an explicit compatibility path. A continuity mismatch is a protocol blocker and must be resolved before registered E1 measurement.

## 11. E1-B — Multi-event history separability

Memory of one binary cue is insufficient to characterize behavior context. E1-B measures whether multiple earlier events remain jointly linearly separable after the current observation becomes uninformative.

### 11.1 Event fixture

Use event channels:

```text
A = [1, 0]
B = [0, 1]
neutral = [0, 0]
```

The four history classes are:

```text
AB
BA
AA
BB
```

The first event occurs at step 0 and the second at step 2. After the second event, append a neutral tail of registered length:

```text
H = [1, 5, 20, 40]
```

Add four class-independent nuisance channels. For each paired fixture group, all four history classes receive the same deterministic nuisance stream; nuisance values are zero-mean Rademacher values scaled by `0.25`. Training and evaluation use disjoint fixture lineages.

The readout sees only the terminal reservoir state and predicts one of the four history classes.

### 11.2 Counts

For every architecture/budget/seed/horizon:

- training: `200` paired nuisance realizations × 4 classes = `800` states;
- evaluation: `50` paired nuisance realizations × 4 classes = `200` states.

### 11.3 Reset control

Reset immediately after the second event and before the class-independent neutral/nuisance tail. Because each four-class group then receives identical post-reset inputs, the terminal states within that group must be byte-identical. With balanced labels, reset classification must be exactly `25%`.

### 11.4 Separability metrics

Report:

- four-class terminal-state Ridge accuracy by horizon;
- macro accuracy;
- normalized correct-class margin distributions;
- within-class and between-class terminal-state cosine-distance summaries;
- contiguous 80% separability horizon using the same all-shorter-horizons rule.

These are representation measurements. They do not establish semantic attractors.

## 12. E1-C — YOLO-like temporal corruption robustness

E1-C is synthetic and detector-like; it does not run YOLO or use video. Its purpose is to test whether reservoir topology changes robustness when primitive temporal observations are missing, wrong, occluded or jittered.

### 12.1 Clean primitive sequences

Use 20-step normalized primitive feature sequences with four behavior classes:

1. `approach` — hand/object distance decreases; no overlap; object remains static.
2. `touch` — approach followed by overlap; object remains static.
3. `pick_up` — approach, overlap, then object motion follows hand motion.
4. `pass_by` — distance decreases then increases; no overlap; object remains static.

Primitive features are numeric detector/tracker-style signals only: relative distance, overlap/contact indicator, hand motion, object motion, relative motion consistency, and visibility/missingness indicators. No class label is encoded in any input field.

Training uses clean sequences only. Evaluation applies the same trained probe to clean and corrupted sequences.

### 12.2 Registered corruption arms

For each evaluation sequence, generate fixed paired corruption masks shared across architecture arms:

- `clean`;
- `drop10`: exactly 2 of 20 frames replaced by the explicit missing-observation token;
- `wrong10`: exactly 2 of 20 frames replaced by the same-time primitive observation from a fixed incorrect donor class;
- `occlusion4`: one contiguous four-frame missing window crossing the behavior's interaction region;
- `jitter`: zero-mean Gaussian noise with `sigma = 0.05` on continuous features only, clipped to the registered normalized feature bounds;
- `mixed`: exactly 2 dropped frames, exactly 1 wrong donor frame, plus `sigma = 0.05` continuous jitter.

Corruption locations/donor identities come from an isolated deterministic fixture lineage. Do not resample inconvenient corruptions.

### 12.3 Robustness metrics

Report for every architecture/budget/seed:

- clean classification accuracy;
- accuracy for every corruption arm;
- absolute accuracy drop from clean for each corruption;
- macro corrupted accuracy;
- worst registered corrupted-arm accuracy;
- confusion matrix by behavior and corruption.

No architecture is declared superior from one corruption arm alone.

## 13. Measurement validity gates

A registered E1 result is scientifically valid only if all of the following hold:

1. exact architecture definitions and neuron budgets match the sealed manifest;
2. every arm returns `state_dim = N`;
3. reservoir weights are unchanged before/after all state collection and probe fitting;
4. all required parameter/fixture/prediction digests are present;
5. fixture and corruption lineages are identical across architecture arms where required;
6. reset negative controls are exact for E1-A and E1-B;
7. the shallow-64 Phase 2A compatibility gate passes;
8. repeated protocol-only runs are deterministic;
9. no reward-learning or R2 module is imported into the E1 measurement path;
10. no post-observation hyperparameter or architecture selection is performed.

A validity failure stops interpretation. It must not be converted into a performance result.

## 14. Prospective interpretation categories

The report may select one or more categories after measurement; no numeric auto-classifier chooses the scientific interpretation.

### Outcome A — shallow architecture preserves memory best

Long-delay E1-A performance is consistently stronger for shallow reservoirs at fixed `N`, while grouped/deep arms do not compensate through E1-B/C. Interpretation: fixed-budget splitting reduces useful fading-memory capacity under the tested dynamics.

### Outcome B — grouped reservoirs trade some memory for stronger separability/robustness

Grouped arms lose some long-delay E1-A capacity but improve E1-B history separation and/or E1-C corruption robustness across seeds. Interpretation: parallel random dynamical projections provide representational diversity at a measurable memory cost.

### Outcome C — deep reservoirs improve abstraction but shorten memory

Deep arms improve E1-B/C while degrading long-delay E1-A more strongly than grouped arms. Interpretation: serial nonlinear transformation changes the memory/representation trade-off under fixed neuron budget.

### Outcome D — architecture effect is small relative to seed or budget

Differences between architectures are inconsistent across seeds or substantially smaller than the `N=64` to `N=256` scaling effect. Interpretation: reservoir size/realization dominates topology under the registered task family.

### Outcome E — mixed or unresolved

If the three gates disagree without a stable pattern, freeze the result as mixed. Do not choose a preferred architecture by changing metric weights after observation.

## 15. Evidence and sealing

Registered E1 evidence should eventually be frozen under:

```text
docs/experiments/r1-e1-reservoir-architecture/
├── manifest.json
├── result.json
├── provenance.json
├── report.md
└── trace-index.json
```

The prospective manifest must bind at minimum:

- exact scientific head;
- base integration head;
- architecture enum and definitions;
- budgets `64/256`;
- seeds `[7,17,29,43,61]`;
- all RNG lineage tags;
- `tanh`, radius `0.9`, leak `1.0`, no recurrent bias;
- input/recurrent initialization laws;
- Ridge `1e-6`;
- E1-A delays/counts/threshold definitions;
- E1-B templates/horizons/counts/nuisance law;
- E1-C sequence schema and corruption laws;
- exact Python/NumPy environment;
- reference AutoESN commit for provenance only.

Canonical evidence JSON must be sorted, compact UTF-8 with one trailing newline and `allow_nan=false`. Measurement files must be write-once in the registered workflow. Large per-state traces, if retained, belong in an artifact and are referenced from `trace-index.json` by name, size and SHA-256.

## 16. CI and measurement workflow boundary

Implementation may later add permanent deterministic unit/protocol CI for E1. Registered scientific measurement must use a separately authorized, sealed exact-head workflow after all implementation tests pass.

The required sequence is:

```text
design approval
→ implementation plan approval
→ implementation/tests
→ exact-head permanent CI
→ prospective manifest/preflight
→ mandatory human measurement approval
→ one registered measurement
→ human interpretation review
→ no-refit evidence freeze
→ remove one-shot workflows
→ final exact-head CI
```

Do not run the registered measurement during ordinary implementation CI.

## 17. Relationship to later work

E1 answers which fixed reservoir topologies provide useful temporal representation under controlled tasks. It does not select a production NPC or YOLO architecture by itself.

After E1 is frozen:

- **R1 E2** may study spectral radius, leaking rate, input scaling and activation while holding the selected topology set fixed.
- **R1 E3** may use actual YOLO/tracker outputs and real noisy sequences.
- **R2** remains an independent delayed-credit research line.
- **R3** may later integrate a frozen R1 representation with a frozen R2 credit mechanism into an adaptive agent/NPC.

The central separation must remain explicit:

```text
R1: observation history → latent state
R2: delayed outcome → past-action credit
R3: integrate state representation and learning only after each line is independently characterized
```
