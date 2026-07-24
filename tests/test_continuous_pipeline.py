from types import SimpleNamespace
from pathlib import Path

import numpy as np

from dvgc.continuous import (
    CONTINUOUS_STAGES, ContinuousPhaseTracker, load_trajectory, save_trajectory,
)


def _state(phase, airborne, landing, vz):
    return SimpleNamespace(
        info={
            "phase": np.asarray(phase),
            "had_airborne": np.asarray(airborne),
            "had_valid_landing": np.asarray(landing),
        },
        data=SimpleNamespace(qvel=np.asarray([0.0, 0.0, vz])),
    )


def test_continuous_phase_tracker_is_monotonic():
    tracker = ContinuousPhaseTracker()
    tracker.observe(_state(1, 0, 0, 0.0), descent_entry=False, tick=1)
    tracker.observe(_state(2, 1, 0, 1.0), descent_entry=False, tick=2)
    tracker.observe(_state(2, 1, 0, -0.1), descent_entry=False, tick=3)
    tracker.observe(_state(2, 1, 0, -0.2), descent_entry=True, tick=4)
    tracker.observe(_state(3, 1, 1, 0.0), descent_entry=False, tick=5)
    assert tracker.stage == "landing"
    assert [row["to"] for row in tracker.switches] == list(
        CONTINUOUS_STAGES[1:]
    )


def test_physical_descent_event_can_drive_smoke_only_handoff():
    tracker = ContinuousPhaseTracker(stage="apex", previous_vz=0.1)
    tracker.observe(
        _state(2, 1, 0, -0.1),
        descent_entry=False,
        physical_descent=True,
        tick=1,
    )
    assert tracker.stage == "descent"
    assert not tracker.switches[0]["descent_support_entry"]
    assert tracker.switches[0]["physical_descent_event"]


def test_trajectory_round_trip(tmp_path):
    arrays = {
        "qpos": np.arange(12, dtype=np.float32).reshape(3, 4),
        "phase": np.arange(3, dtype=np.int32),
    }
    path = tmp_path / "trace.npz"
    digest = save_trajectory(path, arrays)
    restored, actual = load_trajectory(path, digest)
    assert actual == digest
    np.testing.assert_array_equal(restored["qpos"], arrays["qpos"])


def test_continuous_runner_fixes_dynamics_and_never_restores_physics():
    text = Path("cli/run_mjx_continuous_smoke.py").read_text()
    assert '"domain_randomization": False' in text
    assert "restore_snapshot" not in text
    assert "reset_from_snapshot" not in text
    assert "--upstream-action-sequence" in text
