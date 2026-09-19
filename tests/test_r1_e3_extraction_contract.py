from __future__ import annotations

import json
from pathlib import Path


CONTRACT = Path("docs/r1-e3/extraction-contract.md")
EXAMPLE = Path("tests/fixtures/r1_e3/extraction-provenance.example.json")


def test_extraction_contract_files_exist() -> None:
    assert CONTRACT.is_file()
    assert EXAMPLE.is_file()


def test_extraction_provenance_requires_detector_tracker_and_runtime_identity() -> None:
    payload = json.loads(EXAMPLE.read_text(encoding="utf-8"))

    assert payload["schema"] == "r1-e3-extraction-provenance-v1"
    assert payload["extractor_commit"]
    assert payload["source_manifest_sha256"]
    assert payload["raw_tracks_sha256"]
    assert payload["annotations_sha256"]
    assert payload["normalized_episodes_sha256"]

    yolo = payload["yolo"]
    assert yolo["implementation"]
    assert yolo["package_version"]
    assert yolo["model"]
    assert yolo["weights_sha256"]
    assert isinstance(yolo["confidence_threshold"], float)
    assert isinstance(yolo["iou_threshold"], float)
    assert yolo["class_filter"]

    botsort = payload["botsort"]
    assert botsort["implementation"]
    assert botsort["version"]
    assert botsort["config_sha256"]
    assert botsort["config"]

    runtime = payload["runtime"]
    assert runtime["python"]
    assert runtime["lock_sha256"]
    assert runtime["platform"]
    assert runtime["hardware_provenance"]


def test_extraction_contract_freezes_handoff_and_forbids_registered_inference() -> None:
    text = CONTRACT.read_text(encoding="utf-8")

    for required in (
        "YOLO11",
        "BoT-SORT",
        "weights SHA-256",
        "tracker configuration",
        "confidence threshold",
        "IoU/NMS threshold",
        "frame sampling",
        "video SHA-256",
        "tracks.jsonl",
        "whole-video",
        "deterministic replay",
        "rejected extraction run",
    ):
        assert required in text

    assert "registered measurement must not run YOLO" in text
    assert "registered measurement must not run BoT-SORT" in text


def test_example_digest_fields_are_lowercase_sha256() -> None:
    payload = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    digests = (
        payload["source_manifest_sha256"],
        payload["raw_tracks_sha256"],
        payload["annotations_sha256"],
        payload["normalized_episodes_sha256"],
        payload["yolo"]["weights_sha256"],
        payload["botsort"]["config_sha256"],
        payload["runtime"]["lock_sha256"],
    )

    for digest in digests:
        assert len(digest) == 64
        assert all(character in "0123456789abcdef" for character in digest)
