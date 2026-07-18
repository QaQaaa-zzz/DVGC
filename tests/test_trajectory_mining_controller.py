import json
from pathlib import Path

import numpy as np

from cli.prepare_trajectory_mining_resume import prepare
from cli.trajectory_mining_controller import TrajectoryMiningController
from dvgc.bank import SnapshotBank


def test_new_route_is_bounded_and_does_not_repeat_old_ppo():
    source=Path("cli/trajectory_mining_controller.py").read_text()
    assert "trajectory_mining" in source and "roll_controllability" in source
    assert "single_roll_targeted_ppo" in source
    assert source.count("25600")==1
    assert "descent_acquisition_max_rounds" not in source


def test_persistent_start_script_uses_new_unit_and_active_pointer():
    source=Path("scripts/start_trajectory_mining_controller.sh").read_text()
    assert "dvgc-trajectory-mining-controller" in source
    assert "ACTIVE_PIPELINE.json" in source


def test_controller_preflights_corrected_bank_before_construction():
    source=Path("cli/trajectory_mining_controller.py").read_text()
    assert "candidate_preflight.json" in source
    assert source.index("validate_unique_candidates")<source.index("super().stage_a()")
    assert "physical_audit_pass" in source and "source_policy_records_complete" in source
    assert "runtime_gate_pass_current" in source and "tracked_worktree_clean" in source


def _record(identifier,state,parent):
    value=np.array([state],np.float32)
    return {"id":identifier,"source_phase":"flight","qpos":value,"qvel":value+1,"ctrl":value+2,
        "qacc_warmstart":value+3,"physical_feature":np.full(16,state,np.float32),"policy_state":{"last_action":value+4},
        "oracle_phase":2,"had_airborne":1,"had_valid_landing":0,"contact_age":0,"airborne_count":1,"recovery_count":0,
        "trajectory_parent_id":parent,"descent_layer":"middle","bootstrap_eligible":True,"training_only":False}


def test_prepare_resume_freezes_invalid_bank_and_reports_unique_shortfall(tmp_path):
    invalid=tmp_path/"invalid";mining=invalid/"trajectory_mining";stable=invalid/"cycle_3/stable"
    mining.mkdir(parents=True);stable.mkdir(parents=True)
    base_rows=[_record(f"base-{i}",i,f"base-parent-{i}") for i in range(2)]
    additions=[_record(f"new-{i}",100+i,f"parent-{i}") for i in range(7)]
    base_path=tmp_path/"base.pkl";SnapshotBank(base_rows,{"policy_hash":"policy"}).save(base_path)
    metadata={"policy_hash":"policy","xml_sha256":"xml","landing_entry_set_sha256":"entry",
        "landing_policy_hash":"landing","candidate_config_hash":"config","snapshot_source_policy_hashes":["policy"]}
    invalid_base=[dict(row,snapshot_source_policy_hash="policy") for row in base_rows]
    invalid_additions=[dict(row,snapshot_source_policy_hash="policy") for row in additions]
    SnapshotBank(invalid_base+[row for row in invalid_additions for _ in range(4)],metadata).save(mining/"candidate_pool.pkl")
    (mining/"report.json").write_text('{"selected_snapshots":28}')
    state={"current_stage":"stable_analyze","stop_reason":"duplicate snapshots","failure_signature":"sig",
        "consecutive_failure_count":3,"provenance":{},"current_policy":"policy-dir","current_checkpoint":"checkpoint",
        "current_cumulative_steps":76800,"policy_history":[],"source_stable_report":"source-report"}
    (invalid/"controller_state.json").write_text(json.dumps(state));(invalid/"controller.lock").write_text('{"pid":0}')
    (invalid/"seed_registry.json").write_text('{"claims":[]}');(stable/"analyze.log").write_text("duplicate snapshots\n")
    output=tmp_path/"corrected";report=prepare(invalid_run=invalid,output_run=output,base_bank_path=base_path)
    assert report["unique_additions"]==7 and report["quota_target"]==28 and report["quota_shortfall"]==21
    assert report["corrected_states"]==9 and report["analyzer_preflight"]["status"]=="PASS"
    corrected=SnapshotBank.load(output/"trajectory_mining_corrected/candidate_pool.pkl")
    assert all(row["snapshot_source_policy_hash"]=="policy" for row in corrected.records_for_phase("flight",include_training_only=False))
    frozen=json.loads((output/"trajectory_mining_corrected/invalid_engineering_manifest.json").read_text())
    assert frozen["status"]=="INVALID_ENGINEERING_DUPLICATE_SELECTION" and len(frozen["duplicate_groups"])==7


def test_resume_controller_preserves_prepared_stage(tmp_path):
    payload={"controller_type":"trajectory_mining","controller_version":2,"current_stage":"stable_stage_a",
        "current_cycle":4,"controller_unit":"unit","controller_module":"cli.trajectory_mining_controller",
        "provenance":{},"history":[]}
    (tmp_path/"controller_state.json").write_text(json.dumps(payload))
    first=TrajectoryMiningController(tmp_path)
    assert first.state["current_stage"]=="stable_stage_a" and first.state["current_cycle"]==4
    first.lock.close()
