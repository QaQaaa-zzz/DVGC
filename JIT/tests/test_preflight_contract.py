from __future__ import annotations

import json


def _formal_config_basenames(jit_root):
    from jit_dvgc.training import FORMAL_SCHEMA

    return [
        path.name
        for path in sorted((jit_root / "configs").glob("*.json"))
        if json.loads(path.read_text(encoding="utf-8")).get("schema") == FORMAL_SCHEMA
    ]


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


def test_preflight_has_no_unified_formal_config_filename_hardcodes(jit_root):
    script = (jit_root / "scripts/local_preflight.sh").read_text(encoding="utf-8")
    formal_config_basenames = _formal_config_basenames(jit_root)

    assert formal_config_basenames
    assert all(basename not in script for basename in formal_config_basenames)


def test_preflight_statically_validates_every_unified_formal_config(jit_root):
    script = (jit_root / "scripts/local_preflight.sh").read_text(encoding="utf-8")
    loop_header = 'for path in sorted(Path("JIT/configs").glob("*.json")):'
    loop_start = script.index(loop_header)
    loop_end = script.index("\n\nif formal_config_count < 1:", loop_start)
    selected_config_loop = script[loop_start:loop_end]

    assert "raw.get(\"schema\") != training.FORMAL_SCHEMA" in selected_config_loop
    assert "continue" in selected_config_loop
    assert "training.load_unified_formal_config(path)" in selected_config_loop
    assert "normalized_contract = tube.normalize_core_replay_contract(raw[\"tube_sampling\"])" in selected_config_loop
    assert "if normalized_contract is not None:\n            replay_contract_count += 1" in selected_config_loop
    assert "if formal_config_count < 1:\n    raise AssertionError" in script
    assert "if replay_contract_count < 1:\n    raise AssertionError" in script


def test_tube_package_exports_the_replay_contract_normalizer():
    from jit_dvgc import tube_rsi
    import jit_dvgc.tube as tube

    assert tube.normalize_core_replay_contract is tube_rsi.normalize_core_replay_contract


def test_readme_documents_iteration_workflow_and_final_test_claim_boundary(jit_root):
    readme = (jit_root / "README.md").read_text(encoding="utf-8")
    assert "run_iteration_workflow.py --config <workflow.json> --execute" in readme
    assert "final TEST/JCE/JEL: untouched" in readme
