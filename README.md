# Neural State Machine Game Experiment

This branch is a **standalone orphan project**. It does not inherit the files,
package, CI, or commit history of `master`.

The project tests one narrow idea:

> Can recurrent neural activity carry complex game context and change its
> action tendency through reward feedback without defining every compound
> behavior state in advance?

It is a deterministic research baseline, not a biologically complete fly brain
and not a replacement for YOLO perception.

## Model

```text
game facts
   ↓
normalized stimulus vector
   ↓
recurrent neural state h(t)
   ↓
atomic action: left / right / attack / wait
   ↓
hard-rule toy game
   ↓
reward or punishment
   ↓
reward-modulated output plasticity
```

The controller never receives labels such as `AttackState`, `RetreatState`, or
`LowHealthBossState`. It receives distances, directions, health, threat, and
boundary signals. The only enum is the atomic action vocabulary.

The environment still enforces deterministic invariants:

- map boundaries cannot be crossed;
- attacks only damage an adjacent enemy;
- health remains within configured bounds;
- healing items are single-use;
- terminal games cannot be stepped.

This is the intended split between a **soft neural state machine** for choices
and **hard rules** for legality.

## Install

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
```

On Windows, use `.venv\Scripts\python` instead of `.venv/bin/python`.

## Run

```bash
python scripts/benchmark.py
```

The command prints one JSON object containing:

- `replay_equal`: identical seeds and inputs produced identical neural states,
  actions, rewards, and final snapshots;
- `context_separation`: two stimulus histories occupied distinguishable regions
  of hidden-state space;
- `positive_margin_delta`: positive feedback increased the selected action's
  logit margin for the same reset context;
- `negative_margin_delta`: negative feedback decreased that margin;
- `episode`: one bounded deterministic closed-loop trace summary.

Run the verification suite:

```bash
pytest -q
ruff check .
python scripts/benchmark.py
python scripts/benchmark_memory_probe.py
```

Tests and benchmarks require no GPU, model weights, network access, FlyVis,
fly-brain, YOLO, ROS2, or game engine.

## Delayed-cue memory gates

Phase 2 separates two claims that the first experiment combined:

- **Phase 2A — memory decodability:** one frozen Ridge linear probe measures
  whether the current recurrent hidden vector retains a vanished left/right cue
  across delays 1–5.
- **Phase 2B — reward learning:** a separate online reward-modulated readout
  must learn to use that information. Its current fixed-seed baseline fails and
  does not determine the Phase 2A result.

Run the Phase 2A scientific gate:

```bash
python scripts/benchmark_memory_probe.py
```

The command prints one compact sorted JSON object and exits nonzero if any of
seeds 7, 17, or 29 misses the fixed gate: at least 90% overall, at least 85%
for every delay, exact 50% after resetting state immediately before the shared
decision input, identical reset features, unchanged policy output weights, and
exact independent-run repeatability.

| Seed | Training | Phase 2A overall | Reset | Delay 1 / 2 / 3 / 4 / 5 |
|---:|---:|---:|---:|---|
| 7 | 2000/2000 | 200/200 | 100/200 | 40/40 / 40/40 / 40/40 / 40/40 / 40/40 |
| 17 | 2000/2000 | 200/200 | 100/200 | 40/40 / 40/40 / 40/40 / 40/40 / 40/40 |
| 29 | 2000/2000 | 200/200 | 100/200 | 40/40 / 40/40 / 40/40 / 40/40 / 40/40 |

The earlier delayed-cue reward-learning baseline remains:

| Seed | Reward learner | Reset | Delay 1 / 2 / 3 / 4 / 5 |
|---:|---:|---:|---|
| 7 | 0.61 | 0.50 | 1.00 / 0.50 / 0.50 / 0.50 / 0.55 |
| 17 | 0.84 | 0.50 | 1.00 / 0.85 / 0.65 / 0.90 / 0.80 |
| 29 | 0.80 | 0.50 | 1.00 / 1.00 / 1.00 / 0.50 / 0.50 |

A passing Phase 2A result supports only linear availability of cue information
in recurrent activity. It does not show that the current reward rule learns
the readout, that semantic attractors emerged, or that the system has general
game intelligence or biological plausibility.

## Phase 2C — learning failure diagnostics

Phase 2A proves linear decodability on this fixed delayed-cue task. The
pre-registered Phase 2B reward-learning result remains a failed scientific
gate, frozen in [its original evidence](docs/experiments/phase-2b-failure.json).
Phase 2C diagnoses that failure; it does not repair the learner, tune its
configuration, or replace the production controller.

The three diagnostic branches share frozen recurrent hidden vectors:

1. Offline Ridge geometry measures normalized signed margins.
2. A private online supervised softmax readout measures learnability with
   explicit labels, solely as a diagnostic instrument.
3. Read-only instrumentation reproduces the unchanged Phase 2B reward learner
   and measures its checkpoints and sampled-versus-supervised gradients.

All numeric hidden rows are collected before labels and delay metadata are
associated with them. Labels enter only diagnostic fitting, supervised updates,
and scoring; they never enter `RecurrentPolicy`, `RewardModulatedReadout`, the
game controller, or action selection. The policy receives only numeric task
stimuli, and the reward learner receives reward feedback, not labels.

Run the diagnostics:

```bash
python scripts/verify_reward_learning_failure.py
python scripts/benchmark_learning_diagnostics.py
```

The fixed configuration is hidden size `64`, recurrent radius `0.9`, learning
rate `0.05`, temperature `1.0`, `2000` training episodes, `20` evaluation
blocks, checkpoint interval `100`, and ordered seeds `[7, 17, 29]`.
The following values are copied without rounding from
[Phase 2C measured evidence](docs/experiments/phase-2c-diagnostics.json),
SHA-256 `19aab5746703f16b4b408bb98215cf47bd8316979c57e986977356fce4f3a8ce`.

| Seed | Classification | Online supervised | Supervised delay 1 / 2 / 3 / 4 / 5 | Frozen reward replay | Reward delay 1 / 2 / 3 / 4 / 5 |
|---:|---|---:|---|---:|---|
| 7 | `REWARD_CREDIT_FAILURE` | 197/200 | 40/40 / 40/40 / 40/40 / 37/40 / 40/40 | 169/200 | 40/40 / 40/40 / 40/40 / 20/40 / 29/40 |
| 17 | `NO_FAILURE_REPRODUCED` | 200/200 | 40/40 / 40/40 / 40/40 / 40/40 / 40/40 | 200/200 | 40/40 / 40/40 / 40/40 / 40/40 / 40/40 |
| 29 | `REWARD_CREDIT_FAILURE` | 199/200 | 40/40 / 40/40 / 40/40 / 40/40 / 39/40 | 171/200 | 40/40 / 40/40 / 40/40 / 31/40 / 20/40 |

The Ridge branch records `2000/2000` training and `200/200` evaluation for
every seed, with `40/40` at each delay. Its normalized signed margin summaries
are:

| Seed | Dataset | Minimum | 10th percentile | Median |
|---:|---|---:|---:|---:|
| 7 | Training | 0.05792864914751649 | 0.0634919085024498 | 0.06556740148006272 |
| 7 | Evaluation | 0.05521994836230297 | 0.0629331546706889 | 0.065425997621153 |
| 17 | Training | 0.09875603395780524 | 0.10368553250732623 | 0.10531321695116778 |
| 17 | Evaluation | 0.09883210681142476 | 0.10371485063634409 | 0.10533604948096265 |
| 29 | Training | 0.05486630199547468 | 0.05858430282800228 | 0.05959179529259498 |
| 29 | Evaluation | 0.056223039055082795 | 0.05864464244785909 | 0.05959969626107961 |

| Seed | Evaluation delay | Minimum | 10th percentile | Median |
|---:|---:|---:|---:|---:|
| 7 | 1 | 0.06484444853655802 | 0.06492690400466729 | 0.0655061104962121 |
| 7 | 2 | 0.06305818528133311 | 0.06431199767984763 | 0.06549240299175743 |
| 7 | 3 | 0.06250139013790612 | 0.06408996385341723 | 0.06539255408137636 |
| 7 | 4 | 0.061619241177164163 | 0.06222613795956641 | 0.06500445756847126 |
| 7 | 5 | 0.05521994836230297 | 0.06203999271821903 | 0.06531941528699553 |
| 17 | 1 | 0.10498424710898284 | 0.10502501466660795 | 0.10523184882202087 |
| 17 | 2 | 0.10338913155265715 | 0.10456375381398911 | 0.1053608871862295 |
| 17 | 3 | 0.09902640873175636 | 0.10389451511031914 | 0.10546393299001658 |
| 17 | 4 | 0.10057426250305962 | 0.1031050501682606 | 0.10559536407333767 |
| 17 | 5 | 0.09883210681142476 | 0.10250158243677265 | 0.10549453806259443 |
| 29 | 1 | 0.05870326189911872 | 0.059320133743332386 | 0.05954874886128013 |
| 29 | 2 | 0.05778089641394101 | 0.05893259442652749 | 0.05951593417276527 |
| 29 | 3 | 0.056777085229734764 | 0.058216538972940046 | 0.059845394227938935 |
| 29 | 4 | 0.056223039055082795 | 0.05860506263491486 | 0.059703764816361526 |
| 29 | 5 | 0.05708406387276004 | 0.05845156653352602 | 0.059665227802459905 |

The recorded delay-five/delay-one median ratios are `0.9971499573428746`
(seed 7), `1.0024962902725185` (seed 17), and `1.0019560266740635` (seed 29).
All three geometry and supervised gates pass. The reward-credit diagnosis
applies to seeds 7 and 29; `NO_FAILURE_REPRODUCED` describes seed 17 only,
not a successful Phase 2B gate across the pre-registered seeds.

The evidence includes every reward and supervised checkpoint, correct-action
probability summaries, expected-gradient scale, block cosine measurements,
and parameter, fixture, hidden-state, reservoir, and legacy-output digests.
All seeds record `diagnostic_valid=true`, `repeatable=true`, and
`phase_2b_portable_evidence_match=true`; `all_valid=true` means these diagnostics
are valid, not that reward learning succeeded.

Green CI on Python 3.10, 3.11, and 3.12 means **portable semantic failure
reproduction plus same-environment float/matrix integrity** and valid Phase 2C
diagnostics, alongside the unchanged Phase 1 and Phase 2A gates. It does not
mean Phase 2B behavior passed or that floating-point JSON bytes match across
platforms. The frozen Phase 2B evidence is not rewritten. Any next phase or
new learning mechanism requires a new design review.

## Phase 3A — normalized action-value credit

Phase 3A replaces the supervised diagnostic readout with a zero-initialized
normalized action-value table. Feedback is immediate after the sampled action;
the delays below are cue-to-decision delays, not delayed rewards. The frozen
acceptance artifact is
[phase-3a-action-value.json](docs/experiments/phase-3a-action-value.json),
SHA-256
`897f1917d3d950f64439ffc0c80aa83aa4c5b940cc5bf41dfbbb89f422da0295`.

The fixed gate is false for seeds 7 and 29 and passes for seed 17: the misses
are concentrated at longer cue delays (`28/40` at seed 7/delay 4 and `20/40`
at seed 29/delay 5). The frozen Phase 2C supervised reference remains `40/40`
at every delay for all three seeds, so the result is not evidence that the
recurrent cue memory disappeared.

The read-only attribution probe is recorded in
[phase-3a-failure-attribution.json](docs/experiments/phase-3a-failure-attribution.json)
and explained in
[phase-3a-failure-analysis.md](docs/experiments/phase-3a-failure-analysis.md).
Its reservoir, action-RNG, and fixture arms all retain a `200/200` supervised
reference while TD(0) varies, supporting a realization- and sampling-sensitive
readout/credit-acquisition explanation. The subsequent
[TD(0) versus TD(lambda) comparison](docs/experiments/phase-3a-credit-comparison.json)
keeps the action lineages identical: it improves seed 29/delay 5 modestly but
does not pass the fixed gate and worsens seed 7 overall. It is evidence for
the attribution, not a replacement learner.

## Package layout

```text
src/neural_state_machine/
├── types.py         observations, actions, snapshots, decisions
├── encoding.py      observation → numeric stimulus
├── controller.py    recurrent dynamics + reward plasticity
├── environment.py   toy-game facts + hard invariants
├── experiment.py    closed-loop traces + measurements
├── benchmark.py     reproducible research benchmark
├── memory_task.py   immutable delayed-cue protocol
├── memory_benchmark.py  Phase 2B reward-learning baseline
├── memory_probe.py  Phase 2A frozen linear measurement
├── phase3a_failure_attribution.py  Phase 3A diagnostic variance arms
└── phase3a_credit_compare.py  TD(0)/TD(lambda) diagnostic comparison
```

## What the result means

A positive separation score demonstrates that the recurrent system distinguishes
the two tested input histories. A repeatable margin change demonstrates that
reward can modify a later action tendency. Together they establish the minimum
mechanics needed to study a neural dynamical state machine.

They do **not** prove that semantic states, stable attractors, intelligence, or
biological game understanding have emerged. The V0 learning rule is a compact
reward-modulated Hebbian update on the action decoder, not dopamine-gated STDP.

## Next research gate

A future branch can place a `fly-brain` adapter behind the same boundary:

```text
observation/stimulus in → neural activity + action out → reward feedback in
```

That adapter should be compared against this deterministic baseline before
adding connectome-scale compute or stronger biological claims. FlyVis remains a
possible temporal visual subsystem, not the controller studied here.

## Design documents

- `docs/superpowers/specs/2026-09-14-neural-state-machine-game-design.md`
- `docs/superpowers/plans/2026-09-14-neural-state-machine-game.md`
- [Delayed-cue memory design](docs/superpowers/specs/2026-09-14-delayed-cue-memory-design.md)
- [Delayed-cue memory implementation plan](docs/superpowers/plans/2026-09-14-delayed-cue-memory.md)
- [Phase 2A memory probe plan](docs/superpowers/plans/2026-09-14-phase-2a-memory-probe.md)
- [Phase 2B reward-learning design](docs/superpowers/specs/2026-09-14-phase-2b-reward-learning-design.md)
- [Phase 2C learning diagnostics design](docs/superpowers/specs/2026-09-14-phase-2c-learning-diagnostics-design.md)
- [Phase 2C learning diagnostics plan](docs/superpowers/plans/2026-09-14-phase-2c-learning-diagnostics.md)

## License

Apache-2.0.
