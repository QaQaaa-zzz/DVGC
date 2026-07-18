"""Freeze a duplicate-selection failure and prepare a corrected, non-overwriting resume run."""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import time
from collections import Counter,defaultdict
from pathlib import Path

from cli.analyze_stable_descent_construction import validate_unique_candidates
from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256
from dvgc.runtime import save_json
from dvgc.seed_registry import save_registry
from dvgc.trajectory_mining import canonical_state_byte_hash,declared_snapshot_hash,select_parent_balanced_with_report


def hashed_files(root:Path,patterns):
    paths=sorted({path for pattern in patterns for path in root.glob(pattern) if path.is_file()})
    return [{"path":str(path),"sha256":file_sha256(path)} for path in paths]


def prepare(*,invalid_run:Path,output_run:Path,base_bank_path:Path,configured_target:int=64,parent_cap:int=4):
    if output_run.exists():raise SystemExit(f"Prepared resume run already exists: {output_run}")
    invalid_bank_path=invalid_run/"trajectory_mining/candidate_pool.pkl"
    invalid_report_path=invalid_run/"trajectory_mining/report.json"
    state_path=invalid_run/"controller_state.json";lock_path=invalid_run/"controller.lock"
    required=(invalid_bank_path,invalid_report_path,state_path,lock_path,base_bank_path)
    missing=[str(path) for path in required if not path.exists()]
    if missing:raise SystemExit(f"Resume inputs missing: {missing}")
    base=SnapshotBank.load(base_bank_path);invalid=SnapshotBank.load(invalid_bank_path)
    base_rows=base.records_for_phase("flight",include_training_only=False)
    invalid_rows=invalid.records_for_phase("flight",include_training_only=False)
    if len(invalid_rows)<len(base_rows):raise SystemExit("Invalid bank is smaller than its base")
    for old,new in zip(base_rows,invalid_rows[:len(base_rows)]):
        if canonical_state_byte_hash(old)!=canonical_state_byte_hash(new) or str(old["id"])!=str(new["id"]):
            raise SystemExit("Invalid bank base prefix changed")
    bad_selected=invalid_rows[len(base_rows):]
    masses={"middle":.40,"late":.40,"early":.20}
    selected,selection=select_parent_balanced_with_report(bad_selected,target=configured_target,masses=masses,parent_cap=parent_cap)
    if not selected:raise SystemExit("Corrected selection has no unique additions")
    corrected_metadata=copy.deepcopy(invalid.metadata)
    corrected_metadata.update({"bank_role":"successful_trajectory_mined_candidates_corrected",
        "invalid_engineering_source_bank_sha256":file_sha256(invalid_bank_path),
        "corrected_from_run":str(invalid_run),"selection_configured_target":configured_target,
        "selection_quota_target":selection["quota_target"],"selection_quota_shortfall":selection["quota_shortfall"],
        "selection_exhausted_unique_support":selection["exhausted_unique_support"]})
    corrected=SnapshotBank(copy.deepcopy(base_rows)+copy.deepcopy(selected),corrected_metadata)
    uniqueness=validate_unique_candidates(corrected.records_for_phase("flight",include_training_only=False))

    output_run.mkdir(parents=True)
    corrected_root=output_run/"trajectory_mining_corrected";corrected_root.mkdir()
    additions_path=corrected_root/"unique_additions.pkl";bank_path=corrected_root/"candidate_pool.pkl"
    SnapshotBank(copy.deepcopy(selected),{"bank_role":"trajectory_mining_unique_additions",
        "source_invalid_bank_sha256":file_sha256(invalid_bank_path),"policy_hash":invalid.metadata.get("policy_hash")}).save(additions_path)
    corrected.save(bank_path)
    invalid_report=json.loads(invalid_report_path.read_text());invalid_state=json.loads(state_path.read_text())
    groups=defaultdict(list)
    for offset,row in enumerate(bad_selected):groups[canonical_state_byte_hash(row)].append({"record_offset":offset,"id":str(row["id"])})
    duplicate_groups=[{"state_byte_hash":key,"records":value} for key,value in sorted(groups.items()) if len(value)>1]
    invalid_manifest={"status":"INVALID_ENGINEERING_DUPLICATE_SELECTION","eligible_for_formal_evidence":False,
        "invalid_run":str(invalid_run),"invalid_cycle":3,"base_states":len(base_rows),"invalid_output_records":len(bad_selected),
        "invalid_unique_records":len(groups),"invalid_candidate_bank":{"path":str(invalid_bank_path),"sha256":file_sha256(invalid_bank_path)},
        "invalid_report":{"path":str(invalid_report_path),"sha256":file_sha256(invalid_report_path)},
        "selected_order":[{"record_offset":i,"id":str(row["id"]),"trajectory_parent_id":str(row["trajectory_parent_id"]),
            "descent_layer":str(row["descent_layer"]),"snapshot_hash":declared_snapshot_hash(row),
            "state_byte_hash":canonical_state_byte_hash(row)} for i,row in enumerate(bad_selected)],
        "parent_quotas":dict(Counter(str(row["trajectory_parent_id"]) for row in bad_selected)),
        "duplicate_groups":duplicate_groups,"analyzer_error":(invalid_run/"cycle_3/stable/analyze.log").read_text().strip().splitlines()[-1],
        "controller_exit":{"current_stage":invalid_state.get("current_stage"),"stop_reason":invalid_state.get("stop_reason"),
            "failure_signature":invalid_state.get("failure_signature"),"consecutive_failure_count":invalid_state.get("consecutive_failure_count")},
        "archived_lock":{"path":str(lock_path),"sha256":file_sha256(lock_path),"payload":json.loads(lock_path.read_text())},
        "stage_manifests":hashed_files(invalid_run/"cycle_3",("stage_*/*.completed.json","stage_*/merged.json","adaptive/*.completed.json","adaptive/merged.json")),
        "provenance":{"source_bank_sha256":file_sha256(base_bank_path),"candidate_config_hash":invalid.metadata.get("candidate_config_hash"),
            "policy_hash":invalid.metadata.get("policy_hash"),"xml_sha256":invalid.metadata.get("xml_sha256"),
            "c_l_hash":invalid.metadata.get("landing_entry_set_sha256"),"pi_l_hash":invalid.metadata.get("landing_policy_hash")}}
    save_json(corrected_root/"invalid_engineering_manifest.json",invalid_manifest)
    report={"status":"PASS","artifact_role":"corrected_trajectory_mining_selection","base_states":len(base_rows),
        "corrected_states":len(corrected.records_for_phase("flight",include_training_only=False)),"unique_additions":len(selected),
        **selection,"candidate_bank_sha256":file_sha256(bank_path),"unique_additions_sha256":file_sha256(additions_path),
        "analyzer_preflight":uniqueness,"physical_gate":"PENDING","evidence_reuse":{"allowed":False,
            "reason":"Existing construction reports are whole-bank-hash bound and contain no per-state snapshot hash; invalid duplicate-cycle evidence is prohibited."}}
    save_json(corrected_root/"report.json",report)
    registry=json.loads((invalid_run/"seed_registry.json").read_text());claims=[]
    for claim in registry["claims"]:
        item=dict(claim)
        if str(item.get("name","")).startswith("stable_cycle_3_"):item["status"]="invalid_engineering_duplicate_selection"
        claims.append(item)
    save_registry(output_run/"seed_registry.json",claims,status="ACTIVE",source_registry=str(invalid_run/"seed_registry.json"),
        source_registry_sha256=file_sha256(invalid_run/"seed_registry.json"))
    state={"controller_type":"trajectory_mining","controller_version":2,"controller_unit":"dvgc-trajectory-mining-controller.service",
        "controller_module":"cli.trajectory_mining_controller","run_id":output_run.name,"current_stage":"stable_stage_a","current_cycle":4,
        "acquisition_round":2,"last_completed_action":"corrected_candidate_bank_prepared","in_progress_action":None,"expected_outputs":[],
        "next_decision":"stable_stage_b","retry_count":0,"heartbeat":time.time(),"stop_reason":None,"active_worker_unit":None,"history":[],
        "provenance":{**invalid_state.get("provenance",{}),"head":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),"candidate_bank_sha256":file_sha256(bank_path),
            "invalid_engineering_manifest_sha256":file_sha256(corrected_root/"invalid_engineering_manifest.json")},
        "failure_signature":None,"consecutive_failure_count":0,"current_candidate":str(bank_path),
        "current_policy":invalid_state["current_policy"],"current_checkpoint":invalid_state["current_checkpoint"],
        "current_cumulative_steps":invalid_state["current_cumulative_steps"],"policy_history":invalid_state.get("policy_history",[]),
        "route_phase":"trajectory_mining","source_stable_report":invalid_state["source_stable_report"],
        "corrected_selection_report":str(corrected_root/"report.json"),
        "candidate_physical_audit":str(corrected_root/"physical_audit.json"),
        "invalid_source_run":str(invalid_run),"evidence_reuse":False}
    save_json(output_run/"controller_state.json",state)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--invalid-run",type=Path,required=True)
    p.add_argument("--output-run",type=Path,required=True);p.add_argument("--base-bank",type=Path,required=True)
    p.add_argument("--configured-target",type=int,default=64);p.add_argument("--parent-cap",type=int,default=4);a=p.parse_args()
    print(json.dumps(prepare(invalid_run=a.invalid_run,output_run=a.output_run,base_bank_path=a.base_bank,
        configured_target=a.configured_target,parent_cap=a.parent_cap),indent=2))


if __name__=="__main__":main()
