from __future__ import annotations

import hashlib
import importlib.util
import json

import pytest


def _formal_config_basenames(jit_root):
    from jit_dvgc.training import FORMAL_SCHEMA

    return [
        path.name
        for path in sorted((jit_root / "configs").glob("*.json"))
        if json.loads(path.read_text(encoding="utf-8")).get("schema") == FORMAL_SCHEMA
    ]


def _load_collect_boundary_cli(jit_root):
    path = jit_root / "cli/collect_unified_boundary_candidates.py"
    spec = importlib.util.spec_from_file_location("collect_unified_boundary_candidates_cli", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _acceptance_predeclaration(source_iteration: int, candidate_iteration: int):
    protocol = {
        "schema": "jit_repair_acceptance_boundary_acquisition_protocol_v1",
        "status": "predeclared_before_repair_training",
        "source_iteration": source_iteration,
        "candidate_iteration": candidate_iteration,
        "acquisition": {
            "anchors_per_phase": 4,
            "minimum_anchors_per_phase": 2,
            "frontier_score_ceiling": 1.0,
            "strengths": [0.1],
            "durations": [2],
            "action_names": ["steer", "rear_wheel_drive", "hip", "knee"],
            "signs": [-1, 1],
            "active_action_dimensions": 2,
            "protocol_seed": 123,
        },
        "isolation": {
            "exclude_consumed_boundary_states": True,
            "exclude_consumed_boundary_parent_groups_by_phase": True,
            "validation_data_used": False,
            "test_data_used": False,
            "final_evaluation_data_used": False,
        },
    }
    protocol_sha = hashlib.sha256(
        json.dumps(
            protocol,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schema": "jit_repair_acceptance_boundary_acquisition_v1",
        "expected_protocol_sha256": protocol_sha,
        "protocol": protocol,
    }


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


def test_acceptance_predeclaration_supports_generic_k_to_kplus1(jit_root, tmp_path):
    module = _load_collect_boundary_cli(jit_root)

    valid = _acceptance_predeclaration(2, 3)
    valid_path = tmp_path / "valid.json"
    valid_path.write_text(json.dumps(valid), encoding="utf-8")
    loaded = module._load_repair_predeclaration(valid_path)
    assert loaded["protocol"]["source_iteration"] == 2
    assert loaded["protocol"]["candidate_iteration"] == 3
    assert loaded["protocol"]["acquisition"]["active_action_dimensions"] == 2

    invalid = _acceptance_predeclaration(2, 4)
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(ValueError, match="k -> k\+1"):
        module._load_repair_predeclaration(invalid_path)


def test_label_cli_maintains_warp_pool_without_changing_scientific_budget(jit_root):
    script = (jit_root / "cli/label_unified_continuations.py").read_text(
        encoding="utf-8"
    )

    assert "compiled_step = jax.jit(env.step)" in script
    assert "wp.set_mempool_release_threshold(device, 0)" in script
    assert "wp.synchronize_device(device)" in script
    assert "compiled_step_fn=memory_stable_step" in script
    assert "WARP_MEMORY_MAINTENANCE_STEP_INTERVAL = 64" in script
    assert "max_ticks=args.max_ticks" in script
    assert "protocol_seed=args.protocol_seed" in script
    assert "candidate" not in script.split("WARP_MEMORY_MAINTENANCE_STEP_INTERVAL = 64", 1)[0]


def test_tube_package_exports_the_replay_contract_normalizer():
    from jit_dvgc import tube_rsi
    import jit_dvgc.tube as tube

    assert tube.normalize_core_replay_contract is tube_rsi.normalize_core_replay_contract


def test_readme_documents_iteration_workflow_and_final_test_claim_boundary(jit_root):
    readme = (jit_root / "README.md").read_text(encoding="utf-8")
    assert "run_iteration_workflow.py --config <workflow.json> --execute" in readme
    assert "final TEST/JCE/JEL: untouched" in readme
