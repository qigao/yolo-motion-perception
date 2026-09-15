import json
import subprocess
import sys
from pathlib import Path


EXPECTED_LOCOMOTION = {
    "standing": "standing",
    "walking_in_place": "walking",
    "walking_transverse": "walking",
    "walking_approaching": "walking",
    "walking_receding": "walking",
    "running": "running",
    "rigid_translation_control": "standing",
    "too_small_unknown": "unknown",
    "camera_translation_compensated": "walking",
}


def test_gait_synthetic_benchmark_emits_required_scenarios_and_states():
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "scripts/benchmark_gait_synthetic.py"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    summary = json.loads(completed.stdout)
    assert set(summary) == set(EXPECTED_LOCOMOTION)
    for name, expected in EXPECTED_LOCOMOTION.items():
        assert summary[name]["locomotion"] == expected

    assert summary["walking_in_place"]["lateral"] == "stationary"
    assert summary["walking_transverse"]["lateral"] == "moving"
    assert summary["walking_approaching"]["radial"] == "approaching"
    assert summary["walking_receding"]["radial"] == "receding"
    assert summary["camera_translation_compensated"]["camera_compensated"] is True
