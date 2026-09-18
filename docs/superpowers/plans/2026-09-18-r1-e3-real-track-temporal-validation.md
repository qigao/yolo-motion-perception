# R1-E3 Real-Track Temporal Decoder Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Build a two-stage R1-E3 pipeline that freezes real YOLO11 + BoT-SORT tracks and annotations into an immutable artifact, then runs a NumPy-only registered comparison of instantaneous versus fixed causal temporal-mean reservoir decoding over that artifact.

**Architecture:** R1-E3A owns extraction-facing schemas, annotation/split validation, deterministic 20-bin normalization, artifact hashing, and artifact verification. R1-E3B consumes only frozen E3A files, reuses the frozen R1-E2 reservoir/Ridge implementation, evaluates B0/B1/B2 plus history-destruction controls, and follows the prospective → explicit approval → exactly-one measurement → no-refit freeze lifecycle.

**Tech Stack:** Python 3.12, NumPy, pytest, Ruff, canonical JSON/JSONL, SHA-256, existing R1-E2 reservoir/readout/evidence patterns. Stage-A extraction may use external YOLO11/Ultralytics/BoT-SORT tooling, but registered E3B runtime remains NumPy-only.

**Spec:** docs/superpowers/specs/2026-09-18-r1-e3-real-track-temporal-validation-design.md

## Global Constraints

- Preserve frozen R1-E1 and R1-E2 source/evidence unchanged.
- Keep R1-E3 split into independently reviewable E3A Artifact Freeze and E3B Registered Transfer Validation subprojects.
- Registered E3B must never import or execute Ultralytics, PyTorch, OpenCV inference, or BoT-SORT.
- Architectures remain Flat-256, Grouped-4x64, Hierarchical-2x128, Hierarchical-4x64.
- Seeds remain exactly [7, 17, 29, 43, 61].
- Ridge regularization remains exactly 1e-6.
- B2 temporal pooling remains the arithmetic mean of normalized bins 16..19.
- Train/evaluation isolation is whole-source-video level; frame-level splitting is forbidden.
- Bounding-box size/growth is image-space only and must never be described as metric depth.
- No topology/window/seed/threshold search after prospective sealing.
- Ordinary CI must never execute registered measurement or Stage-A detector/tracker extraction.
- Exactly one registered E3B measurement is allowed after explicit human approval of exact science head + frozen artifact root digest + runtime + counts.

---

## File Structure

### New E3A files

- src/neural_state_machine/r1_e3_artifact.py — schemas, canonical hashing, loader, structural validation.
- src/neural_state_machine/r1_e3_normalize.py — deterministic 20-bin interpolation and 14-channel tensor construction.
- src/neural_state_machine/r1_e3_artifact_evidence.py — artifact freeze verifier and root-digest computation.
- scripts/prepare_r1_e3_artifact.py — Stage-A packaging/verify CLI; no detector invocation inside registered branch.
- tests/test_r1_e3_artifact.py
- tests/test_r1_e3_normalize.py
- tests/test_r1_e3_artifact_evidence.py
- tests/test_r1_e3_artifact_cli.py

### New E3B files

- src/neural_state_machine/r1_e3_dataset.py — frozen artifact → registered episode tensors/splits.
- src/neural_state_machine/r1_e3_readout.py — B0/B1/B2 adapters over frozen Ridge/readout primitives.
- src/neural_state_machine/r1_e3_benchmark.py — architecture/seed evaluation + history-destruction control.
- src/neural_state_machine/r1_e3_protocol.py — fixed registration, smoke gate, 20-arm registered runner.
- src/neural_state_machine/r1_e3_evidence.py — prospective/write-once/frozen evidence lifecycle.
- scripts/benchmark_r1_e3_real_track.py — protocol/prepare/measure/verify CLI.
- tests/test_r1_e3_dataset.py
- tests/test_r1_e3_readout.py
- tests/test_r1_e3_benchmark.py
- tests/test_r1_e3_protocol.py
- tests/test_r1_e3_evidence.py
- tests/test_r1_e3_cli.py

### Existing files modified

- .github/workflows/ci.yml — focused E3A/E3B tests, frozen-artifact verification, protocol smoke/prospective verification only.
- docs/experiments/r1-e3-real-track-temporal-validation/ — created only after registered measurement freeze.

---

## Task 1: E3A Artifact Schema and Canonical Hashing

**Files:**
- Create: src/neural_state_machine/r1_e3_artifact.py
- Test: tests/test_r1_e3_artifact.py

**Interfaces:**
- Consumes canonical JSON/hash conventions from r1_e2_evidence.py.
- Produces SourceVideoRecord, RawTrackRow, EpisodeAnnotation, FrozenTrackArtifact, validate_artifact_structure(root), artifact_root_digest(root).

- [ ] **Step 1: Write the failing schema/hash tests**

~~~python
def test_artifact_rejects_video_hash_overlap_between_splits(tmp_path: Path) -> None:
    root = build_minimal_artifact(tmp_path)
    rewrite_eval_video_sha(root, training_video_sha(root))
    resign_file(root / "videos.json")
    with pytest.raises(ArtifactInvalid, match="video hash overlap"):
        validate_artifact_structure(root)

def test_tracks_must_be_sorted_by_video_frame_track(tmp_path: Path) -> None:
    root = build_minimal_artifact(tmp_path)
    reverse_first_two_track_rows(root / "tracks.jsonl")
    resign_file(root / "tracks.jsonl")
    with pytest.raises(ArtifactInvalid, match="sorted"):
        validate_artifact_structure(root)

def test_artifact_root_digest_is_deterministic(tmp_path: Path) -> None:
    root = build_minimal_artifact(tmp_path)
    assert artifact_root_digest(root) == artifact_root_digest(root)
    assert len(artifact_root_digest(root)) == 64
~~~

- [ ] **Step 2: Run RED**

Run: pytest -q tests/test_r1_e3_artifact.py

Expected: FAIL because neural_state_machine.r1_e3_artifact does not exist.

- [ ] **Step 3: Implement minimal immutable schema/validator**

~~~python
@dataclass(frozen=True)
class SourceVideoRecord:
    video_id: str
    sha256: str
    split: str
    fps: float
    frame_count: int
    width: int
    height: int

@dataclass(frozen=True)
class RawTrackRow:
    video_id: str
    frame_index: int
    timestamp_seconds: float
    track_id: int
    class_id: int
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    cx_norm: float
    cy_norm: float
    w_norm: float
    h_norm: float
~~~

Reject bool-as-int, malformed SHA-256, duplicate episode IDs, unsorted track rows, non-finite values, and source-video hash overlap.

- [ ] **Step 4: Verify GREEN**

Run:
- pytest -q tests/test_r1_e3_artifact.py
- ruff check src/neural_state_machine/r1_e3_artifact.py tests/test_r1_e3_artifact.py

Expected: PASS.

- [ ] **Step 5: Commit**

~~~bash
git add src/neural_state_machine/r1_e3_artifact.py tests/test_r1_e3_artifact.py
git commit -m "feat: add R1 E3 artifact schema and validation"
~~~

---

## Task 2: E3A Deterministic 20-Bin Normalizer

**Files:**
- Create: src/neural_state_machine/r1_e3_normalize.py
- Test: tests/test_r1_e3_normalize.py

**Interfaces:**
- Consumes FrozenTrackArtifact and EpisodeAnnotation.
- Produces NormalizedEpisode and normalize_episode(artifact, annotation).

- [ ] **Step 1: Write RED tests**

~~~python
def test_normalizer_returns_exact_20_by_14_float64_tensor() -> None:
    artifact, episode = fixture_with_actor_and_target_tracks()
    normalized = normalize_episode(artifact, episode)
    assert normalized.tensor.shape == (20, 14)
    assert normalized.tensor.dtype == np.float64
    assert not normalized.tensor.flags.writeable

def test_interpolation_stops_when_bracketing_gap_exceeds_half_second() -> None:
    artifact, episode = fixture_with_gap(seconds=0.6)
    normalized = normalize_episode(artifact, episode)
    assert normalized.tensor[10, 5] == 0.0
    assert np.all(normalized.tensor[10, 0:5] == 0.0)

def test_pair_features_zero_when_either_track_missing() -> None:
    artifact, episode = fixture_with_missing_target_bin()
    normalized = normalize_episode(artifact, episode)
    assert normalized.tensor[7, 12] == 0.0
    assert normalized.tensor[7, 13] == 0.0
~~~

- [ ] **Step 2: Run RED**

Run: pytest -q tests/test_r1_e3_normalize.py

Expected: FAIL due missing module/API.

- [ ] **Step 3: Implement exact normalization**

Generate exactly 20 timestamps over frozen episode bounds. Interpolate normalized geometry/confidence only when bracketing observations are <=0.5s apart. Missing track bins are zero geometry/confidence + zero presence bit. Pair distance is normalized Euclidean center distance / sqrt(2); IoU is normalized box IoU.

- [ ] **Step 4: Add digest repeatability test**

~~~python
def test_normalized_episode_digest_is_repeatable() -> None:
    artifact, episode = fixture_with_actor_and_target_tracks()
    assert normalize_episode(artifact, episode).tensor_digest == normalize_episode(artifact, episode).tensor_digest
~~~

- [ ] **Step 5: Verify + commit**

~~~bash
pytest -q tests/test_r1_e3_normalize.py
ruff check src/neural_state_machine/r1_e3_normalize.py tests/test_r1_e3_normalize.py
git add src/neural_state_machine/r1_e3_normalize.py tests/test_r1_e3_normalize.py
git commit -m "feat: add deterministic R1 E3 episode normalization"
~~~

---

## Task 3: E3A Artifact Freeze Evidence

**Files:**
- Create: src/neural_state_machine/r1_e3_artifact_evidence.py
- Create: scripts/prepare_r1_e3_artifact.py
- Test: tests/test_r1_e3_artifact_evidence.py
- Test: tests/test_r1_e3_artifact_cli.py

**Interfaces:**
- Produces freeze_artifact(candidate_root, frozen_root), verify_frozen_artifact(root), and a CLI with verify/freeze commands.

- [ ] **Step 1: RED tests for size gate/write-once**

~~~python
def test_freeze_rejects_dataset_below_registered_class_counts(tmp_path: Path) -> None:
    candidate = build_candidate_artifact(tmp_path, train_per_class=19, eval_per_class=10)
    with pytest.raises(ArtifactInvalid, match="20 training episodes per class"):
        freeze_artifact(candidate, tmp_path / "frozen")

def test_freeze_is_write_once(tmp_path: Path) -> None:
    candidate = build_registered_size_candidate(tmp_path)
    frozen = tmp_path / "frozen"
    freeze_artifact(candidate, frozen)
    with pytest.raises(FileExistsError):
        freeze_artifact(candidate, frozen)
~~~

- [ ] **Step 2: RED run**

Run: pytest -q tests/test_r1_e3_artifact_evidence.py tests/test_r1_e3_artifact_cli.py

- [ ] **Step 3: Implement freeze contract**

Require train >=20/class, eval >=10/class, >=2 training videos, >=2 evaluation videos, no source-video overlap, all checksums valid, and exactly one root digest.

- [ ] **Step 4: Prove artifact CLI never extracts**

~~~python
def test_artifact_cli_has_no_detector_runtime_imports() -> None:
    source = Path("scripts/prepare_r1_e3_artifact.py").read_text()
    assert "ultralytics" not in source
    assert "torch" not in source
    assert "botsort" not in source.lower()
~~~

- [ ] **Step 5: Verify + commit**

~~~bash
pytest -q tests/test_r1_e3_artifact_evidence.py tests/test_r1_e3_artifact_cli.py
ruff check src/neural_state_machine/r1_e3_artifact_evidence.py scripts/prepare_r1_e3_artifact.py tests/test_r1_e3_artifact_evidence.py tests/test_r1_e3_artifact_cli.py
git add src/neural_state_machine/r1_e3_artifact_evidence.py scripts/prepare_r1_e3_artifact.py tests/test_r1_e3_artifact_evidence.py tests/test_r1_e3_artifact_cli.py
git commit -m "feat: add R1 E3 artifact freeze lifecycle"
~~~

---

## Task 4: E3A External Extraction Contract

**Files:**
- Create: docs/r1-e3/extraction-contract.md
- Create: tests/fixtures/r1_e3/extraction-provenance.example.json
- Test: tests/test_r1_e3_extraction_contract.py

**Interfaces:**
- Produces the exact handoff schema an external YOLO11 + BoT-SORT extractor must satisfy.

- [ ] **Step 1: RED provenance completeness test**

~~~python
def test_extraction_provenance_requires_detector_tracker_and_runtime_identity() -> None:
    payload = load_example_provenance()
    assert payload["yolo"]["weights_sha256"]
    assert payload["yolo"]["package_version"]
    assert payload["botsort"]["config_sha256"]
    assert payload["runtime"]["python"]
    assert payload["runtime"]["lock_sha256"]
~~~

- [ ] **Step 2: Freeze the external handoff contract**

Document exact output columns, sorting, deterministic replay requirement, detector weights/config hashes, thresholds, frame sampling, runtime lock, extraction code commit, and rejected-run audit.

Do not implement detector inference in the registered research branch.

- [ ] **Step 3: Verify + commit**

~~~bash
pytest -q tests/test_r1_e3_extraction_contract.py
git add docs/r1-e3/extraction-contract.md tests/fixtures/r1_e3/extraction-provenance.example.json tests/test_r1_e3_extraction_contract.py
git commit -m "docs: freeze R1 E3 extraction handoff contract"
~~~

---

## Task 5: E3B Frozen Dataset Loader

**Files:**
- Create: src/neural_state_machine/r1_e3_dataset.py
- Test: tests/test_r1_e3_dataset.py

**Interfaces:**
- Consumes only frozen E3A artifact files.
- Produces RegisteredEpisode, RegisteredDataset, load_registered_dataset(root).

- [ ] **Step 1: RED tests**

~~~python
def test_registered_dataset_uses_only_frozen_normalized_tensors(tmp_path: Path) -> None:
    dataset = load_registered_dataset(frozen_artifact_fixture(tmp_path))
    assert all(item.tensor.shape == (20, 14) for item in dataset.training + dataset.evaluation)

def test_dataset_rejects_source_video_overlap(tmp_path: Path) -> None:
    root = frozen_artifact_fixture(tmp_path)
    corrupt_split_overlap(root)
    with pytest.raises(DatasetInvalid, match="source-video"):
        load_registered_dataset(root)
~~~

- [ ] **Step 2: RED run**

Run: pytest -q tests/test_r1_e3_dataset.py

- [ ] **Step 3: Implement loader**

No video decoding, YOLO, Torch, OpenCV inference, or tracker imports.

- [ ] **Step 4: Verify + commit**

~~~bash
pytest -q tests/test_r1_e3_dataset.py
ruff check src/neural_state_machine/r1_e3_dataset.py tests/test_r1_e3_dataset.py
git add src/neural_state_machine/r1_e3_dataset.py tests/test_r1_e3_dataset.py
git commit -m "feat: add R1 E3 frozen dataset loader"
~~~

---

## Task 6: E3B Fixed Readout Adapters

**Files:**
- Create: src/neural_state_machine/r1_e3_readout.py
- Test: tests/test_r1_e3_readout.py

**Interfaces:**
- Consumes frozen R1-E2/R1-E1 Ridge law.
- Produces frame-only, instantaneous reservoir, and fixed temporal-mean adapters.

- [ ] **Step 1: RED tests**

~~~python
def test_temporal_mean_is_exact_mean_of_bins_16_to_19() -> None:
    trajectories = np.arange(2 * 20 * 3, dtype=np.float64).reshape(2, 20, 3)
    np.testing.assert_array_equal(
        temporal_mean_last_four(trajectories),
        trajectories[:, 16:20, :].mean(axis=1),
    )

def test_temporal_readout_uses_frozen_ridge_regularization() -> None:
    probe = fit_reservoir_temporal_mean(trajectories, labels, class_count=4)
    reference = fit_multiclass_ridge(
        trajectories[:, 16:20].mean(axis=1),
        labels,
        class_count=4,
        regularization=1e-6,
    )
    assert probe.coefficient_digest() == reference.coefficient_digest()
~~~

- [ ] **Step 2: RED run**

Run: pytest -q tests/test_r1_e3_readout.py

- [ ] **Step 3: Implement minimal adapters**

Do not expose any temporal-window parameter.

- [ ] **Step 4: Verify + commit**

~~~bash
pytest -q tests/test_r1_e3_readout.py
ruff check src/neural_state_machine/r1_e3_readout.py tests/test_r1_e3_readout.py
git add src/neural_state_machine/r1_e3_readout.py tests/test_r1_e3_readout.py
git commit -m "feat: add fixed R1 E3 readout adapters"
~~~

---

## Task 7: E3B Benchmark and History-Destruction Control

**Files:**
- Create: src/neural_state_machine/r1_e3_benchmark.py
- Test: tests/test_r1_e3_benchmark.py

**Interfaces:**
- Consumes RegisteredDataset, frozen R1-E2 reservoir registry, E3 readout adapters.
- Produces evaluate_real_track_arm(spec, dataset) and RealTrackArmResult.

- [ ] **Step 1: RED tests proving B1/B2 trajectory identity**

~~~python
def test_b1_and_b2_share_exact_same_reservoir_trajectory() -> None:
    result = evaluate_real_track_arm(spec, tiny_registered_dataset())
    assert result.trajectory_digest_b1 == result.trajectory_digest_b2
~~~

- [ ] **Step 2: RED reset-timing test**

~~~python
def test_history_destruction_resets_immediately_before_bin_16() -> None:
    result = evaluate_real_track_arm(spec, tiny_registered_dataset())
    assert result.history_destruction.reset_before_bin == 16
    assert result.history_destruction.suffix_bins == (16, 17, 18, 19)
~~~

- [ ] **Step 3: Implement one architecture/seed evaluator**

~~~python
reservoir.reset()
states = [reservoir.advance(row) for row in tensor]
instant = states[19]
temporal = np.mean(states[16:20], axis=0)
~~~

History destruction:

~~~python
reservoir.reset()
for row in tensor[:16]:
    reservoir.advance(row)
reservoir.reset()
suffix_states = [reservoir.advance(row) for row in tensor[16:20]]
~~~

- [ ] **Step 4: Report required metrics**

Macro-F1 primary; accuracy, per-class precision/recall/F1, 4x4 confusion, prediction/coefficient/parameter/episode digests secondary.

- [ ] **Step 5: Verify + commit**

~~~bash
pytest -q tests/test_r1_e3_benchmark.py
ruff check src/neural_state_machine/r1_e3_benchmark.py tests/test_r1_e3_benchmark.py
git add src/neural_state_machine/r1_e3_benchmark.py tests/test_r1_e3_benchmark.py
git commit -m "feat: add R1 E3 real-track benchmark"
~~~

---

## Task 8: E3B Registered Protocol and Outcome Classification

**Files:**
- Create: src/neural_state_machine/r1_e3_protocol.py
- Test: tests/test_r1_e3_protocol.py

**Interfaces:**
- Produces registered_manifest_payload(artifact_root_digest), protocol_smoke, run_registered_measurement, classify_registered_outcome.

- [ ] **Step 1: RED manifest tests**

~~~python
assert manifest["architectures"] == {
    "flat": 0,
    "grouped4": 1,
    "hierarchical2": 2,
    "hierarchical4": 3,
}
assert manifest["seeds"] == [7, 17, 29, 43, 61]
assert manifest["temporal_window_bins"] == [16, 17, 18, 19]
assert manifest["arm_count"] == 20
assert manifest["primary_metric"] == "macro_f1"
~~~

- [ ] **Step 2: RED classification tests**

~~~python
def test_outcome_a_requires_both_registered_gates() -> None:
    assert classify_registered_outcome([0.06] * 16 + [0.0] * 4) == "A"

def test_outcome_b_is_positive_but_below_robust_gate() -> None:
    assert classify_registered_outcome([0.02] * 20) == "B"

def test_outcome_c_when_median_not_positive() -> None:
    assert classify_registered_outcome([-0.01] * 20) == "C"
~~~

- [ ] **Step 3: Implement exact 20-arm runner**

Loop exactly 5 seeds × 4 architectures. Public registered runner exposes no seed/architecture overrides.

- [ ] **Step 4: protocol_smoke must never measure**

Monkeypatch run_registered_measurement to fail if invoked.

- [ ] **Step 5: Verify + commit**

~~~bash
pytest -q tests/test_r1_e3_protocol.py
ruff check src/neural_state_machine/r1_e3_protocol.py tests/test_r1_e3_protocol.py
git add src/neural_state_machine/r1_e3_protocol.py tests/test_r1_e3_protocol.py
git commit -m "feat: add R1 E3 registered protocol"
~~~

---

## Task 9: E3B Evidence Lifecycle

**Files:**
- Create: src/neural_state_machine/r1_e3_evidence.py
- Test: tests/test_r1_e3_evidence.py

**Interfaces:**
- Produces prepare_prospective(root, scientific_head, artifact_root_digest), write_measurement, verify_evidence.
- Manifest binds both exact science head and frozen E3A root digest.

- [ ] **Step 1: RED artifact-binding test**

~~~python
def test_measurement_rejects_artifact_root_digest_mismatch(tmp_path: Path) -> None:
    root, prepared = prepare(tmp_path, artifact_digest="a" * 64)
    with pytest.raises(EvidenceInvalid, match="artifact"):
        write_measurement(
            root,
            manifest_sha256=prepared["manifest_sha256"],
            scientific_head=HEAD,
            artifact_root_digest="b" * 64,
            raw_result=fake_result(),
        )
~~~

- [ ] **Step 2: RED verifier schema tests**

Reject wrong 20-arm identity, wrong temporal window, malformed confusion matrices, missing artifact digest, missing B0/B1/B2 metrics, or checksum/runtime mismatch.

- [ ] **Step 3: Implement by adapting R1-E2**

Keep canonical JSON and write-once semantics identical where possible.

- [ ] **Step 4: Verify + commit**

~~~bash
pytest -q tests/test_r1_e3_evidence.py
ruff check src/neural_state_machine/r1_e3_evidence.py tests/test_r1_e3_evidence.py
git add src/neural_state_machine/r1_e3_evidence.py tests/test_r1_e3_evidence.py
git commit -m "feat: add R1 E3 evidence lifecycle"
~~~

---

## Task 10: E3 CLI and Permanent CI

**Files:**
- Create: scripts/benchmark_r1_e3_real_track.py
- Test: tests/test_r1_e3_cli.py
- Modify: .github/workflows/ci.yml

**Interfaces:**
- protocol --artifact-root DIR
- prepare --artifact-root DIR --output DIR --scientific-head SHA
- verify --root DIR [--no-result-ok]
- measure --artifact-root DIR --root DIR --manifest-sha256 SHA

- [ ] **Step 1: RED CLI fail-closed tests**

protocol/prepare/verify must never call measurement. measure must fail before science execution on head/hash/artifact/runtime mismatch or existing result.

- [ ] **Step 2: RED permanent-CI invariant**

~~~python
def test_permanent_ci_never_extracts_or_measures() -> None:
    workflow = Path(".github/workflows/ci.yml").read_text()
    assert "R1 E3 focused tests" in workflow
    assert "R1 E3 artifact verify" in workflow
    assert "R1 E3 protocol smoke" in workflow
    assert "benchmark_r1_e3_real_track.py measure" not in workflow
    assert "ultralytics" not in workflow.lower()
    assert "botsort" not in workflow.lower()
~~~

- [ ] **Step 3: Implement CLI preflight ordering**

1. read sealed manifest;
2. verify manifest hash;
3. verify current head;
4. verify frozen artifact digest;
5. verify prospective evidence;
6. verify runtime;
7. only then run measurement.

- [ ] **Step 4: Extend CI**

CI runs focused E3 tests, frozen-artifact verifier, protocol smoke, and prospective prepare/verify only before measurement. After evidence freeze it switches to frozen-evidence verification. CI never extracts or measures.

- [ ] **Step 5: Verify + commit**

~~~bash
pytest -q tests/test_r1_e3_*.py
ruff check src/neural_state_machine/r1_e3_*.py scripts/prepare_r1_e3_artifact.py scripts/benchmark_r1_e3_real_track.py tests/test_r1_e3_*.py
git add .github/workflows/ci.yml scripts/benchmark_r1_e3_real_track.py tests/test_r1_e3_cli.py
git commit -m "ci: add R1 E3 permanent protocol gates"
~~~

---

## Task 11: E3A Real Artifact Generation Gate

**Files:** no science-code changes unless validator defects are first reproduced with RED tests.

- [ ] Freeze source-video corpus manifest before extraction acceptance.
- [ ] Run external YOLO11 + BoT-SORT under sealed extraction environment.
- [ ] Record weights hash, package versions, tracker config hash, thresholds, runtime lock, extraction code commit.
- [ ] Complete human episode annotation/review before E3B scoring.
- [ ] Build deterministic normalized artifact.
- [ ] Run E3A verifier twice and require identical root digest.
- [ ] Require train >=20/class, eval >=10/class, >=2 training videos, >=2 evaluation videos, no video overlap.
- [ ] Freeze artifact manifest/checksums/provenance and review counts.
- [ ] STOP if dataset-size gate fails; expansion is permitted only before E3B prospective sealing.

---

## Task 12: Prospective E3B Measurement Gate

- [ ] Verify all E3 tests, Ruff, frozen-artifact verifier, protocol smoke, push/PR CI on one exact science head.
- [ ] Prepare prospective evidence bound to exact head + artifact root digest.
- [ ] Verify prospective-only state and absence of result files.
- [ ] Present exact science head, artifact root digest, manifest SHA-256, Python/NumPy runtime, source-video counts, per-class train/eval counts, arm count=20, temporal bins=16..19.
- [ ] STOP for explicit human approval. Do not measure before approval.

---

## Task 13: Single Registered Measurement and No-Refit Freeze

- [ ] Execute exactly one approved measurement from detached approved science head.
- [ ] Assert head, artifact digest, manifest hash, runtime before measure.
- [ ] Verify full evidence: valid=true, 20 arms, complete B0/B1/B2/reset metrics, registered Outcome A/B/C.
- [ ] Write no-refit report separating facts from interpretation.
- [ ] Freeze raw evidence without rewriting result bytes.
- [ ] Remove one-shot measurement tooling.
- [ ] Switch permanent CI to frozen artifact/evidence verification.
- [ ] Require final push/PR CI success and zero one-shot measurement workflows.
- [ ] Merge only after explicit human approval.

---

## Self-Review Checklist

- Spec coverage: E3A extraction boundary, schemas, annotations, normalization, split isolation, E3B decoders, history-destruction, outcome classification, evidence lifecycle, and no-refit rules all map to tasks.
- Placeholder scan: no TBD/TODO/FIXME or "implement later" placeholders.
- Type consistency: E3A produces FrozenTrackArtifact + root digest; E3B consumes only frozen artifact/tensors; B2 always uses bins 16..19; registered arm count is always 20; evidence binds science head + artifact digest.
- Scope decomposition: E3A and E3B are independently reviewable task groups joined only by immutable artifact files/checksums.
