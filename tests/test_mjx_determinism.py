import numpy as np

from cli.diagnose_mjx_determinism import _first_divergence


def _trace(value=0.0):
    out = {}
    for field in (
        "qpos", "qvel", "act", "ctrl", "time", "observation",
        "observation_history", "last_action", "delay_buffer",
        "controller_state", "prng_key", "phase", "estimated_phase",
        "physical_failure", "end_code", "contact_count", "contact_geom",
        "contact_position", "contact_normal", "contact_distance",
        "contact_force", "subtree_angmom",
    ):
        out[field] = np.full((2, 3, 1), value, np.float32)
    return out


def test_first_divergence_reports_physics_step():
    trace = _trace()
    trace["qpos"][1, 2, 0] = 0.25
    assert _first_divergence(trace) == {
        "tick": 2,
        "field": "qpos",
        "max_abs_error": 0.25,
        "location": "physics_step",
    }


def test_exact_trace_has_no_divergence():
    assert _first_divergence(_trace()) is None


def test_default_runtime_selects_repeatable_mjx_solver():
    text = open("configs/default.json").read()
    env_text = open("dvgc/env.py").read()
    assert '"mjx_solver": "CG"' in text
    assert "mujoco.mjtSolver.mjSOL_CG" in env_text
