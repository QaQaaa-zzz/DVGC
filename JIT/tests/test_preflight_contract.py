from __future__ import annotations


def test_preflight_uses_fixed_interpreter_and_current_source_contract(jit_root):
    script = (jit_root / "scripts/local_preflight.sh").read_text(encoding="utf-8")
    assert "/home/qy/mujoco_playground/.venv/bin/python" in script
    assert "-m \"not gpu\"" in script
    assert "-m gpu" in script
    assert "jit_dvgc.provenance verify-run" not in script


def test_preflight_does_not_launch_training(jit_root):
    script = (jit_root / "scripts/local_preflight.sh").read_text(encoding="utf-8")
    assert "train_phase_expert.py" not in script
    assert "998400" not in script


def test_preflight_statically_validates_every_unified_formal_config(jit_root):
    script = (jit_root / "scripts/local_preflight.sh").read_text(encoding="utf-8")
    assert 'Path("JIT/configs").glob("*.json")' in script
    assert "training.FORMAL_SCHEMA" in script
    assert "training.load_unified_formal_config" in script
    assert "tube.normalize_core_replay_contract" in script
    assert "unified formal config" in script
    assert "replay contract" in script
    assert "pi_unified_iter1_tube1_natural10_retry01.json" not in script


def test_tube_package_exports_the_replay_contract_normalizer():
    from jit_dvgc import tube_rsi
    import jit_dvgc.tube as tube

    assert tube.normalize_core_replay_contract is tube_rsi.normalize_core_replay_contract


def test_readme_documents_iteration_workflow_and_final_test_claim_boundary(jit_root):
    readme = (jit_root / "README.md").read_text(encoding="utf-8")
    assert "run_iteration_workflow.py --config <workflow.json> --execute" in readme
    assert "final TEST/JCE/JEL: untouched" in readme
