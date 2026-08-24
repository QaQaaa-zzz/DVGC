from __future__ import annotations


def test_preflight_uses_fixed_interpreter_and_successful_closed_run(jit_root):
    script = (jit_root / "scripts/local_preflight.sh").read_text(encoding="utf-8")
    assert "/home/qy/mujoco_playground/.venv/bin/python" in script
    assert "phase_u_1024_one_block_20260824_seed820001_retry2" in script
    assert "-m \"not gpu\"" in script
    assert "-m gpu" in script
    assert "jit_dvgc.provenance verify-run" in script


def test_preflight_does_not_launch_training(jit_root):
    script = (jit_root / "scripts/local_preflight.sh").read_text(encoding="utf-8")
    assert "train_phase_expert.py" not in script
    assert "998400" not in script
