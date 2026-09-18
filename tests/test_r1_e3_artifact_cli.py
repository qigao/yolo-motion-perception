from __future__ import annotations

from pathlib import Path

import pytest


def _cli():
    from scripts import prepare_r1_e3_artifact as cli

    return cli


def test_artifact_cli_verify_delegates_to_frozen_verifier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = _cli()
    root = tmp_path / "artifact"
    root.mkdir()
    monkeypatch.setattr(
        cli,
        "verify_frozen_artifact",
        lambda value: {
            "valid": True,
            "artifact_root_digest": "a" * 64,
            "root": str(value),
        },
    )

    assert cli.main(["verify", "--root", str(root)]) == 0
    output = capsys.readouterr().out
    assert '"valid":true' in output


def test_artifact_cli_freeze_delegates_without_extraction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = _cli()
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    output = tmp_path / "frozen"
    calls: list[tuple[Path, Path]] = []

    def fake_freeze(left: Path, right: Path):
        calls.append((Path(left), Path(right)))
        return {"valid": True, "artifact_root_digest": "b" * 64}

    monkeypatch.setattr(cli, "freeze_artifact", fake_freeze)

    assert (
        cli.main(
            [
                "freeze",
                "--candidate",
                str(candidate),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert calls == [(candidate, output)]


def test_artifact_cli_has_no_detector_runtime_imports() -> None:
    _cli()
    source = Path("scripts/prepare_r1_e3_artifact.py").read_text(encoding="utf-8").lower()

    for forbidden in ("ultralytics", "import torch", "from torch", "botsort", "boT-sort".lower()):
        assert forbidden not in source
