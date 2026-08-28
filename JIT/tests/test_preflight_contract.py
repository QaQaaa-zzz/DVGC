from __future__ import annotations


def test_preflight_uses_fixed_interpreter_and_current_source_contract(jit_root):
    script = (jit_root / "scripts/local_preflight.sh").read_text(encoding="utf-8")
    assert "/home/qy/mujoco_playground/.venv/bin/python" in script
    assert "phase_u_continuation_10m.json" in script
    assert "-m \"not gpu\"" in script
    assert "-m gpu" in script
    assert "jit_dvgc.provenance verify-run" not in script


def test_preflight_does_not_launch_training(jit_root):
    script = (jit_root / "scripts/local_preflight.sh").read_text(encoding="utf-8")
    assert "train_phase_expert.py" not in script
    assert "998400" not in script


def test_preflight_statically_validates_the_formal_config(jit_root):
    script = (jit_root / "scripts/local_preflight.sh").read_text(encoding="utf-8")
    assert "phase_u_formal.json" in script
    assert "load_config" in script


def test_preflight_statically_validates_the_unified_pilot_without_launching_it(jit_root):
    script = (jit_root / "scripts/local_preflight.sh").read_text(encoding="utf-8")
    assert "pi_unified_pilot.json" in script
    assert "load_unified_pilot_config" in script
    assert "run_unified_pilot" not in script


def test_readme_documents_persistent_formal_launch_and_claim_boundary(jit_root):
    readme = (jit_root / "README.md").read_text(encoding="utf-8")
    assert "--formal" in readme
    assert "nohup setsid" in readme
    assert "phase_u_formal.json" in readme
    assert "does not by itself establish a trained expert" in readme
