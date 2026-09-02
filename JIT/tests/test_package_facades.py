"""Compatibility tests for the categorized JIT package API and workflow."""

from __future__ import annotations

import json
import sys


def test_package_roots_preserve_legacy_authorities():
    from jit_dvgc import core_retaining_tube_iteration as legacy_iteration
    from jit_dvgc import policy_conditioned_continuation_field as legacy_field
    from jit_dvgc import snapshot_pool as legacy_pool
    from jit_dvgc import soft_tube as legacy_soft
    from jit_dvgc import tube_rsi as legacy_tube_rsi
    from jit_dvgc import tube_rsi_smoke as legacy_smoke
    from jit_dvgc import unified_boundary as legacy_boundary
    from jit_dvgc import unified_continuation_labels as legacy_labels
    from jit_dvgc import unified_diagnostic as legacy_diagnostic
    from jit_dvgc import unified_policy_freeze as legacy_freeze
    from jit_dvgc import unified_training as legacy_training

    import jit_dvgc.acquisition as acquisition
    import jit_dvgc.analysis as analysis
    import jit_dvgc.continuation as continuation
    import jit_dvgc.snapshots as snapshots
    import jit_dvgc.training as training
    import jit_dvgc.tube as tube

    assert tube.load_soft_tube is legacy_soft.load_soft_tube
    assert tube.run_tube_rsi_smoke is legacy_smoke.run_tube_rsi_smoke
    assert tube.build_core_retaining_tube is legacy_iteration.build_core_retaining_tube
    assert tube.normalize_core_replay_contract is legacy_tube_rsi.normalize_core_replay_contract
    assert snapshots.SnapshotPool is legacy_pool.SnapshotPool
    assert training.checkpoint_identity is legacy_training.checkpoint_identity
    assert training.freeze_unified_policy is legacy_freeze.freeze_unified_policy
    assert acquisition.collect_unified_boundary_candidates is legacy_boundary.collect_unified_boundary_candidates
    assert continuation.label_unified_continuations is legacy_labels.label_unified_continuations
    assert continuation.fit_policy_conditioned_continuation_fields is legacy_field.fit_policy_conditioned_continuation_fields
    assert analysis.run_unified_fixed_panel is legacy_diagnostic.run_unified_fixed_panel


def test_formal_api_adds_zero_interaction_preflight_without_redefining_contract():
    from jit_dvgc import unified_formal as legacy_formal
    import jit_dvgc.training as training

    assert training.FORMAL_SCHEMA == legacy_formal.FORMAL_SCHEMA
    assert training.FORMAL_TARGET == legacy_formal.FORMAL_TARGET
    assert training.load_unified_formal_config is legacy_formal.load_unified_formal_config
    assert training.run_unified_formal is not legacy_formal.run_unified_formal
    assert callable(training.preflight_unified_formal_tube)


def test_workflow_is_plan_first_and_resumable(tmp_path):
    from jit_dvgc.workflow import run_workflow

    config_path = tmp_path / "workflow.json"
    done = tmp_path / "done.txt"
    state_dir = tmp_path / "state"
    config = {
        "schema": "jit_iteration_workflow_v1",
        "workflow_name": "unit",
        "state_dir": str(state_dir),
        "variables": {"ROOT": str(tmp_path)},
        "environment": {},
        "stages": [
            {
                "name": "one",
                "command": [
                    sys.executable,
                    "-c",
                    "from pathlib import Path; Path('done.txt').write_text('ok', encoding='utf-8')",
                ],
                "cwd": "{ROOT}",
                "requires": [],
                "completion": {
                    "path": "{ROOT}/done.txt",
                    "kind": "file",
                    "assertions": [],
                    "exports": {},
                },
            }
        ],
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")

    plan = run_workflow(config_path, execute=False)
    assert plan["plan"][0]["status"] == "pending"
    assert not done.exists()

    first = run_workflow(config_path, execute=True)
    assert first["status"] == "completed"
    assert done.read_text(encoding="utf-8") == "ok"

    second = run_workflow(config_path, execute=True)
    assert second["status"] == "completed"
    state = json.loads((state_dir / "state.json").read_text(encoding="utf-8"))
    assert state["completed_stages"] == ["one"]
