# Task 2 report: isolated hidden-state datasets

## Changes

- Added frozen, validated `MemoryProbeConfig` with locked protocol defaults.
- Added local balanced fixture generation (`_balanced_cases`, `_build_fixtures`) independent of reward-learning code.
- Added immutable `_StateDataset` and literal `reset_state()`/`advance()` collectors (`_collect_hidden`, `_collect_dataset`).
- Added tests covering configuration validation, fresh balanced blocks, exact hidden collection and reset ablation, immutable metadata, and policy/eligibility invariants.

## TDD evidence

- RED: `.venv/bin/pytest tests/test_memory_probe.py -q` failed during collection with `ImportError: cannot import name 'MemoryProbeConfig'` before implementation.
- GREEN: after implementation, `.venv/bin/pytest tests/test_memory_probe.py -q` reported `52 passed in 0.12s`.

## Verification commands and output

- `.venv/bin/pytest -q` — `331 passed in 1.45s`.
- `.venv/bin/ruff check src/neural_state_machine/memory_probe.py tests/test_memory_probe.py` — `All checks passed!`.
- `git diff --check` — clean.

## Self-review

The collector only invokes `reset_state()` and `advance()`; it does not call `decide()` or `learn()`. Fixture blocks are freshly shuffled and balanced over both cues and delays. Dataset arrays are defensive read-only copies. Policy matrices and output digest are asserted unchanged, and reset clears decision eligibility.

## Concerns

No known concerns. Empty fixture tuples are not a supported collector input because the resulting dataset would have no samples.
