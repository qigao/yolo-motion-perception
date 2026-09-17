# R1-E2 Reservoir Representation Implementation Plan

Tracks issue #6 and design `docs/superpowers/specs/2026-09-18-r1-e2-reservoir-representation-design.md`.

## Constraints

- keep R1/E1 source and frozen evidence unchanged;
- keep R2/Phase3/C4 code and evidence unchanged;
- NumPy-only execution;
- ordinary CI must not run registered measurement;
- preserve prospective single-measurement lifecycle.

## Task 1 — E2 architecture registry and accounting

Add `src/neural_state_machine/r1_e2_reservoir.py` as an E2 registry over the already verified R1/E1 reservoir engine. Registered architectures are Flat-256, Grouped-4x64, Hierarchical-2x128, Hierarchical-4x64. Expose deterministic topology metadata including neuron count, recurrent edges, input parameters, total reservoir parameters, component widths, and parameter digest.

TDD gate: tests first prove the E2 module/registry is absent, then implementation makes deterministic construction/accounting tests green.

## Task 2 — causal temporal readout

Add `src/neural_state_machine/r1_e2_readout.py`.

Implement:

- instantaneous Ridge adapter using the frozen R1/E1 Ridge law;
- deterministic causal mean pooling over an explicitly supplied trailing state window;
- no learned pooling and no future-state access.

Tests cover shape, causality, deterministic pooling, regularization identity, and multiclass tie behavior inherited from R1/E1.

## Task 3 — E2-A compatibility memory baseline

Add `src/neural_state_machine/r1_e2_memory.py` using the frozen R1/E1 delayed-cue fixture and compatibility path. Registered delays remain `1/2/5/10/20/40/80`. Report per-delay accuracy and contiguous 85% horizon. Do not modify `r1_e1_memory.py`, `memory_probe.py`, `memory_task.py`, or `policy.py`.

## Task 4 — E2-B temporal composition

Add `src/neural_state_machine/r1_e2_composition.py`.

Freeze deterministic fixtures for six order classes `ABC/ACB/BAC/BCA/CAB/CBA`, history spans `5/10/20/40`, balanced train/eval counts, reset controls, RNG lineage, and causal decision time. Evaluate instantaneous and temporal-mean readouts on the same reservoir trajectories.

Report accuracy, macro-F1, confusion counts, centroid distances, within-class dispersion, and between/within separation ratio.

## Task 5 — E2-C YOLO-like episode benchmark

Add `src/neural_state_machine/r1_e2_yolo_episode.py` by extending frozen E1-C feature semantics without changing E1. Compare frame-only, reservoir-instantaneous, and reservoir-temporal-mean baselines under clean/drop10/wrong10/occlusion4/jitter/mixed conditions. Freeze exact episode templates and corruption lineage before measurement.

## Task 6 — protocol and evidence

Add `src/neural_state_machine/r1_e2_protocol.py`, `src/neural_state_machine/r1_e2_evidence.py`, and `scripts/benchmark_r1_e2_reservoir.py`.

Required CLI lifecycle:

- `protocol`
- `prepare --output DIR --scientific-head SHA`
- `measure --root DIR --manifest-sha256 SHA`
- `verify --root DIR [--no-result-ok]`

`prepare` must not measure. `measure` is write-once and exact-head/runtime bound. `verify` fails closed on manifest, provenance, runtime, hash, count, or schema mismatch.

## Task 7 — permanent CI

Extend `.github/workflows/ci.yml` with focused R1-E2 tests and a protocol smoke/preflight only. CI must not call `measure`. Before measurement, CI may prepare and verify a prospective manifest; after evidence freeze, replace prospective generation with frozen-evidence verification.

## Task 8 — prospective measurement gate

After all implementation/tests and exact-head CI are green:

1. prepare prospective evidence on the exact science head;
2. retain the manifest artifact;
3. review manifest/runtime/counts;
4. stop for explicit human measurement approval;
5. execute exactly one registered measurement;
6. review interpretation without refit;
7. freeze evidence and remove one-shot workflow;
8. run final exact-head CI.

## Commit discipline

Use small TDD commits with explicit RED then GREEN evidence where practical. Never report completion from a prior head; every completion claim requires fresh verification of the current exact head.