from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest


_HEAD = "1" * 40
_ARTIFACT_DIGEST = "a" * 64
_OTHER_ARTIFACT_DIGEST = "b" * 64


def _cli():
    from scripts import benchmark_r1_e3m_memory as cli

    return cli


def _dataset(digest: str = _ARTIFACT_DIGEST):
    return SimpleNamespace(artifact_root_digest=digest)


def _forbidden_measurement(*args, **kwargs):
    pytest.fail(
        "registered E3M measurement was invoked before preflight completed"
    )


def _prepare_root(
    cli,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, str]:
    monkeypatch.setattr(
        cli,
        "load_mechanism_dataset",
        lambda root: _dataset(),
    )
    root = tmp_path / "prospective"
    assert (
        cli.main(
            [
                "prepare",
                "--artifact-root",
                str(tmp_path / "artifact"),
                "--output",
                str(root),
                "--scientific-head",
                _HEAD,
            ]
        )
        == 0
    )
    return root, (root / "manifest.sha256").read_text().strip()


def test_protocol_command_does_not_measure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    cli = _cli()
    monkeypatch.setattr(
        cli,
        "run_registered_memory_benchmark",
        _forbidden_measurement,
    )
    monkeypatch.setattr(
        cli,
        "load_mechanism_dataset",
        lambda root: _dataset(),
    )

    assert (
        cli.main(
            [
                "protocol",
                "--artifact-root",
                str(tmp_path / "artifact"),
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["registered_measurement"] is False
    assert output["artifact_root_digest"] == _ARTIFACT_DIGEST
    assert output["arm_count"] == 20
    assert output["delays"] == [1, 2, 5, 10, 15]


def test_prepare_and_verify_no_result_do_not_measure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = _cli()
    monkeypatch.setattr(
        cli,
        "run_registered_memory_benchmark",
        _forbidden_measurement,
    )

    root, _ = _prepare_root(cli, tmp_path, monkeypatch)

    assert not (root / "result.json").exists()
    assert (
        cli.main(
            [
                "verify",
                "--root",
                str(root),
                "--no-result-ok",
            ]
        )
        == 0
    )


def test_measure_rejects_manifest_hash_before_measurement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = _cli()
    root, _ = _prepare_root(cli, tmp_path, monkeypatch)
    monkeypatch.setattr(
        cli,
        "run_registered_memory_benchmark",
        _forbidden_measurement,
    )
    monkeypatch.setattr(cli, "_current_head", lambda: _HEAD)

    with pytest.raises(SystemExit, match="manifest sha256"):
        cli.main(
            [
                "measure",
                "--artifact-root",
                str(tmp_path / "artifact"),
                "--root",
                str(root),
                "--manifest-sha256",
                "0" * 64,
            ]
        )


def test_measure_rejects_head_mismatch_before_measurement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = _cli()
    root, manifest_sha = _prepare_root(cli, tmp_path, monkeypatch)
    monkeypatch.setattr(
        cli,
        "run_registered_memory_benchmark",
        _forbidden_measurement,
    )
    monkeypatch.setattr(cli, "_current_head", lambda: "0" * 40)

    with pytest.raises(SystemExit, match="scientific head"):
        cli.main(
            [
                "measure",
                "--artifact-root",
                str(tmp_path / "artifact"),
                "--root",
                str(root),
                "--manifest-sha256",
                manifest_sha,
            ]
        )


def test_measure_rejects_artifact_mismatch_before_measurement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = _cli()
    root, manifest_sha = _prepare_root(cli, tmp_path, monkeypatch)
    monkeypatch.setattr(
        cli,
        "run_registered_memory_benchmark",
        _forbidden_measurement,
    )
    monkeypatch.setattr(cli, "_current_head", lambda: _HEAD)
    monkeypatch.setattr(
        cli,
        "load_mechanism_dataset",
        lambda root: _dataset(_OTHER_ARTIFACT_DIGEST),
    )

    with pytest.raises(SystemExit, match="artifact root digest"):
        cli.main(
            [
                "measure",
                "--artifact-root",
                str(tmp_path / "artifact"),
                "--root",
                str(root),
                "--manifest-sha256",
                manifest_sha,
            ]
        )


def test_measure_preflights_prospective_evidence_before_measurement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = _cli()
    root, manifest_sha = _prepare_root(cli, tmp_path, monkeypatch)
    monkeypatch.setattr(
        cli,
        "run_registered_memory_benchmark",
        _forbidden_measurement,
    )
    monkeypatch.setattr(cli, "_current_head", lambda: _HEAD)

    def invalid_preflight(*args, **kwargs):
        raise cli.EvidenceInvalid("prospective evidence invalid")

    monkeypatch.setattr(cli, "verify_evidence", invalid_preflight)

    with pytest.raises(SystemExit, match="prospective evidence invalid"):
        cli.main(
            [
                "measure",
                "--artifact-root",
                str(tmp_path / "artifact"),
                "--root",
                str(root),
                "--manifest-sha256",
                manifest_sha,
            ]
        )


def test_measure_rejects_runtime_mismatch_before_measurement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = _cli()
    root, manifest_sha = _prepare_root(cli, tmp_path, monkeypatch)
    monkeypatch.setattr(
        cli,
        "run_registered_memory_benchmark",
        _forbidden_measurement,
    )
    monkeypatch.setattr(cli, "_current_head", lambda: _HEAD)
    monkeypatch.setattr(cli.platform, "python_version", lambda: "0.0.0")

    with pytest.raises(SystemExit, match="runtime"):
        cli.main(
            [
                "measure",
                "--artifact-root",
                str(tmp_path / "artifact"),
                "--root",
                str(root),
                "--manifest-sha256",
                manifest_sha,
            ]
        )


def test_measure_calls_registered_runner_only_after_all_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = _cli()
    root, manifest_sha = _prepare_root(cli, tmp_path, monkeypatch)
    monkeypatch.setattr(cli, "_current_head", lambda: _HEAD)

    calls: list[object] = []
    fake_measurement = {"registered_measurement": True}

    def run_measurement(dataset):
        calls.append(("measure", dataset.artifact_root_digest))
        return fake_measurement

    def write_measurement(
        received_root,
        *,
        manifest_sha256,
        scientific_head,
        artifact_root_digest,
        raw_result,
    ):
        calls.append(
            (
                "write",
                Path(received_root),
                manifest_sha256,
                scientific_head,
                artifact_root_digest,
                raw_result,
            )
        )
        return {
            "manifest_sha256": manifest_sha256,
            "result_sha256": "c" * 64,
        }

    monkeypatch.setattr(
        cli,
        "run_registered_memory_benchmark",
        run_measurement,
    )
    monkeypatch.setattr(cli, "write_measurement", write_measurement)

    assert (
        cli.main(
            [
                "measure",
                "--artifact-root",
                str(tmp_path / "artifact"),
                "--root",
                str(root),
                "--manifest-sha256",
                manifest_sha,
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert output["result_sha256"] == "c" * 64
    assert calls[0] == ("measure", _ARTIFACT_DIGEST)
    assert calls[1][0] == "write"
    assert calls[1][3] == _HEAD
    assert calls[1][4] == _ARTIFACT_DIGEST
    wrapped = calls[1][5]
    assert wrapped["registered_measurement"] is True
    assert (
        wrapped["manifest"]["artifact_root_digest"]
        == _ARTIFACT_DIGEST
    )


def test_focused_ci_never_invokes_registered_measurement() -> None:
    workflow = Path(
        ".github/workflows/r1-e3m-mechanism.yml"
    ).read_text(encoding="utf-8")

    assert "benchmark_r1_e3m_memory.py measure" not in workflow
