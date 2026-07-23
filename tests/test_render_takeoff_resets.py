from pathlib import Path


def test_reset_video_overlay_contains_required_telemetry():
    text = Path("cli/render_takeoff_resets.py").read_text()
    for field in ("hip=", "knee=", "root z=", "vz=", "phase=", "wheel contacts="):
        assert field in text
    assert "--action-search" in text
    assert "bounded_action_success" in text
