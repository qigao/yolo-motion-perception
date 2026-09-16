from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from neural_state_machine.phase_c4a_diagnostics.integrity import (
    run_d0_integrity,
    verify_frozen_c4a_evidence,
)


_REPO_ROOT = Path(__file__).resolve().parents[1]
_EVIDENCE_RELATIVE = Path("docs/experiments/phase-c4-delay-marginalized-credit")
_FROZEN_FILES = (
    "c4a-manifest.json",
    "c4a-result.json",
    "c4a-provenance.json",
)


def _copy_frozen_evidence(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    destination = root / _EVIDENCE_RELATIVE
    destination.mkdir(parents=True)
    source = _REPO_ROOT / _EVIDENCE_RELATIVE
    for name in _FROZEN_FILES:
        shutil.copyfile(source / name, destination / name)
    return root


def test_verify_frozen_c4a_evidence_accepts_registered_bytes():
    identity = verify_frozen_c4a_evidence(_REPO_ROOT)

    assert identity.scientific_head == "8ae3154950ed53c4d0a0f555463042ff72674d31"
    assert identity.seeds == (7, 17, 29)
    assert identity.design_rows == 2_005


@pytest.mark.parametrize("name", _FROZEN_FILES)
def test_verify_frozen_c4a_evidence_fails_closed_on_single_byte_mutation(
    tmp_path: Path,
    name: str,
):
    root = _copy_frozen_evidence(tmp_path)
    path = root / _EVIDENCE_RELATIVE / name
    raw = bytearray(path.read_bytes())
    raw[len(raw) // 2] ^= 1
    path.write_bytes(bytes(raw))

    with pytest.raises(RuntimeError, match="frozen C4-A evidence hash mismatch"):
        verify_frozen_c4a_evidence(root)


def test_d0_replays_registered_scores_and_anonymous_design_identity():
    replays = run_d0_integrity(_REPO_ROOT)

    assert [
        (replay.seed, replay.condition, replay.score.correct, replay.score.total)
        for replay in replays
    ] == [
        (7, "normal", 200, 200),
        (7, "registered_shuffled", 147, 200),
        (17, "normal", 200, 200),
        (17, "registered_shuffled", 170, 200),
        (29, "normal", 200, 200),
        (29, "registered_shuffled", 160, 200),
    ]

    for replay in replays:
        assert replay.design.shape[0] == 2_005
        assert replay.target.shape == (2_005,)
        assert replay.design.flags.writeable is False
        assert replay.target.flags.writeable is False
        assert len(replay.actions) == 2_000
        assert len(replay.rewards) == 2_000
        assert len(replay.design_digest) == 64
        assert len(replay.target_digest) == 64

    for seed in (7, 17, 29):
        normal = next(
            replay
            for replay in replays
            if replay.seed == seed and replay.condition == "normal"
        )
        shuffled = next(
            replay
            for replay in replays
            if replay.seed == seed and replay.condition == "registered_shuffled"
        )
        assert normal.design_digest == shuffled.design_digest
        assert normal.design.tobytes(order="C") == shuffled.design.tobytes(order="C")
        assert normal.schedule_digest == shuffled.schedule_digest
        assert normal.shuffled_rewards is None
        assert shuffled.shuffled_rewards is not None
        assert shuffled.shuffled_rewards != normal.rewards
