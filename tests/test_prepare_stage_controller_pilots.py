import numpy as np

from cli.prepare_stage_controller_pilots import aligned_reference_anchors


class _Named:
    def __init__(self, value):
        self.id = value


class _Model:
    jnt_qposadr = np.array([7, 8])
    jnt_dofadr = np.array([6, 7])
    jnt_range = np.array([[-1.3, 0.5], [-1.5, 2.5]])

    def joint(self, name):
        return _Named({"hip_joint": 0, "knee_joint": 1}[name])


def test_stage_input_rejects_clipped_or_cross_phase_reference_state():
    import pandas as pd

    frame = pd.DataFrame({
        "hip_position": [-1.2, -1.2],
        "knee_position": [2.4, 2.6],
        "hip_velocity": [0.1, 0.1],
        "knee_velocity": [-0.2, -0.2],
        "vel_x": [2.0, 2.0], "vel_y": [0.0, 0.0], "vel_z": [0.5, 0.5],
    })
    base = {
        "candidate_kind": "reference_anchor", "flight_subinterval": "ascent",
        "oracle_phase": 2, "qpos": np.array([0, 0, 0, 1, 0, 0, 0, -1.2, 2.4]),
        "qvel": np.array([2.0, 0.0, 0.5, 0, 0, 0, 0.1, -0.2]),
        "policy_state": {"filter_phase": 2, "obs_history": [[0.0]], "last_action": [0.0]},
    }
    legal = {**base, "reference_index": 0}
    out_of_range = {**base, "reference_index": 1}
    rows, rejected = aligned_reference_anchors(
        [legal, out_of_range], frame, _Model(), "ascent"
    )
    assert rows == [legal]
    assert rejected == {"reference_joint_outside_authoritative_xml_range": 1}


def test_takeoff_pilot_source_requires_balanced_reset_classes():
    from pathlib import Path
    text = Path("cli/prepare_stage_controller_pilots.py").read_text()
    assert 'evenly(canonical,3)+evenly(aligned,3)' in text
