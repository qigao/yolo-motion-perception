from __future__ import annotations

from pathlib import Path

import pytest

from neural_state_machine.phase3c_diagnostics.measurement_gate import seal_auxiliary_artifact


def _manifest() -> dict[str, object]:
    return {
        "auxiliary_keys": {
            "reset_scores": ["reset/a"],
            "reverse_checks": ["reverse/a"],
            "drain_scores": ["drain/a"],
        }
    }


def _payload() -> dict[str, object]:
    return {
        "reset_scores": [{"key": "reset/a", "overall": {"correct": 1, "total": 2}}],
        "reverse_checks": [{"key": "reverse/a", "order_invariant": True}],
        "drain_scores": [{"key": "drain/a", "overall": {"correct": 1, "total": 2}}],
    }


def test_seal_auxiliary_artifact_writes_exact_hash_bound_file(tmp_path: Path) -> None:
    metadata = seal_auxiliary_artifact(tmp_path, _manifest(), _payload())
    assert metadata["path"] == "auxiliary.json"
    assert len(metadata["sha256"]) == 64
    assert (tmp_path / "auxiliary.json").is_file()


def test_seal_auxiliary_artifact_rejects_missing_key(tmp_path: Path) -> None:
    payload = _payload()
    payload["drain_scores"] = []
    with pytest.raises(RuntimeError, match="auxiliary"):
        seal_auxiliary_artifact(tmp_path, _manifest(), payload)
    assert not (tmp_path / "auxiliary.json").exists()


def test_seal_auxiliary_artifact_refuses_overwrite(tmp_path: Path) -> None:
    (tmp_path / "auxiliary.json").write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError):
        seal_auxiliary_artifact(tmp_path, _manifest(), _payload())
