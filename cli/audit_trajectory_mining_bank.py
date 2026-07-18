"""Run the physical/provenance gate for a corrected trajectory-mining bank."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import jax

from cli.analyze_stable_descent_construction import validate_unique_candidates
from cli.mine_success_trajectories import validate_candidate
from dvgc.bank import SnapshotBank
from dvgc.candidate_geometry import TerrainClearanceSolver
from dvgc.config import STAGE_ID,config_hash,file_sha256,load_config
from dvgc.env import OrangeBikeDVGC
from dvgc.policy import load_bundle
from dvgc.runtime import build_inference,save_json
from dvgc.snapshot_provenance import validate_snapshot_source_records
from dvgc.trajectory_mining import canonical_state_byte_hash


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--base-bank",required=True);p.add_argument("--candidate-bank",required=True)
    p.add_argument("--policy",required=True);p.add_argument("--landing-entry-set",required=True);p.add_argument("--landing-policy",required=True)
    p.add_argument("--output",required=True);p.add_argument("--seed",type=int,default=1700000001);p.add_argument("--config",default="configs/default.json");a=p.parse_args()
    output=Path(a.output)
    if output.exists():raise SystemExit(f"Output exists: {output}")
    params,policy_cfg,_=load_bundle(a.policy,verify_files=True);base=SnapshotBank.load(a.base_bank);bank=SnapshotBank.load(a.candidate_bank)
    base_rows=base.records_for_phase("flight",include_training_only=False);rows=bank.records_for_phase("flight",include_training_only=False)
    if len(rows)<len(base_rows):raise SystemExit("Corrected bank is smaller than base")
    prefix_ok=all(str(x["id"])==str(y["id"]) and canonical_state_byte_hash(x)==canonical_state_byte_hash(y) for x,y in zip(base_rows,rows))
    additions=rows[len(base_rows):];cfg=load_config(a.config,{**policy_cfg,"training_stage":"flight","expert_chain_termination":False,
        "domain_randomization":False,"obs_noise_enable":False,"use_bank_resets":False})
    env=OrangeBikeDVGC(cfg,snapshot_bank=SnapshotBank(),cert_bank=SnapshotBank.load(a.landing_entry_set));step=jax.jit(env.step)
    inference=build_inference(env,params,deterministic=True);solver=TerrainClearanceSolver(cfg.xml_path,margin=0.0,
        max_correction=cfg.flight_candidate_max_root_z_correction);reasons=Counter();accepted=[]
    for index,row in enumerate(additions):
        ok,reason,placement=validate_candidate(row,env,step,inference,solver,cfg,jax.random.PRNGKey(a.seed+index))
        reasons[reason]+=1
        if ok:accepted.append({"id":str(row["id"]),"state_byte_hash":canonical_state_byte_hash(row),
            "terrain_clearance_m":placement.clearance,"wheel_clearance_m":placement.wheel_clearance,
            "nonwheel_clearance_m":placement.nonwheel_clearance,"root_z_shift_m":placement.root_z_shift})
    uniqueness=validate_unique_candidates(rows);source_hashes=validate_snapshot_source_records(rows,bank.metadata)
    provenance={"policy_hash":bank.metadata.get("policy_hash")==file_sha256(Path(a.policy)/"params.pkl"),
        "xml_hash":bank.metadata.get("xml_sha256")==file_sha256(cfg.xml_path),
        "c_l_hash":bank.metadata.get("landing_entry_set_sha256")==file_sha256(a.landing_entry_set),
        "pi_l_hash":bank.metadata.get("landing_policy_hash")==file_sha256(Path(a.landing_policy)/"params.pkl"),
        "config_hash":bank.metadata.get("candidate_config_hash")==config_hash(cfg)}
    checks={"base_prefix_unchanged":prefix_ok,"all_additions_accepted":len(accepted)==len(additions),
        "seven_unique_additions":len(additions)==7,"flight_semantic":all(int(row.get("oracle_phase",-1))==STAGE_ID["flight"] for row in additions),
        "analyzer_preflight":uniqueness["status"]=="PASS","source_policy_records_complete":bool(source_hashes),
        "provenance_current":all(provenance.values())}
    report={"status":"PASS" if all(checks.values()) else "FAIL","artifact_role":"corrected_trajectory_mining_physical_audit",
        "seed":a.seed,"base_states":len(base_rows),"candidate_states":len(rows),"unique_additions":len(additions),
        "accepted_additions":len(accepted),"rejection_counts":dict(reasons),"checks":checks,"provenance_checks":provenance,
        "analyzer_preflight":uniqueness,"snapshot_source_policy_hashes":list(source_hashes),
        "candidate_bank_sha256":file_sha256(a.candidate_bank),"accepted":accepted}
    save_json(output,report);print(json.dumps({k:v for k,v in report.items() if k!="accepted"},indent=2))
    if report["status"]!="PASS":raise SystemExit(2)


if __name__=="__main__":main()
