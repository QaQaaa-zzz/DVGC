"""Zero-training evaluation of physical Apex bridge states under compact pi_D."""
from __future__ import annotations

import argparse
import copy
import json
import pickle
import subprocess
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from cli.run_backward_descent_nominal_pilot import C_L, PI_D, PI_L
from cli.run_backward_descent_rsi_pilot import certify_policy
from cli.run_descent_localized_consolidation_v1 import verified_assets_allowing_runtime_gate_refresh
from cli.runtime_gate import source_fingerprint
from dvgc.backward_search import compact_observation_command_adapter
from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256, load_config
from dvgc.descent_entry import descent_entry_feature
from dvgc.entry import normalized_nearest, robust_normalization
from dvgc.env import OrangeBikeDVGC
from dvgc.policy import load_bundle
from dvgc.runtime import save_json


SOURCE = Path("runs/stage_next_reset_v3_seed0_20260723/apex/feedback_bridge_v1/fresh_validation/stable_physical_descent.pkl")
TUBE = Path("runs/descent_diverse_p1_predecessor_recovery_v5_p1_core_adapter/independent_audit_v1/descent_tube_v2.pkl")
EXPERT = Path("runs/descent_diverse_p1_predecessor_recovery_v5_p1_core_adapter")
DEFAULT_RUN = Path("runs/apex_bridge_compact_descent_v1/zero_training_8x_p1")


def bridge_nodes(records, tube_features, center, scale, cfg):
    nodes=[]
    for row in records:
        feature=descent_entry_feature(row["physical_feature"],cfg)
        distance,index,_=normalized_nearest(feature,tube_features,center,scale)
        parent=str(row.get("trajectory_parent_id",row.get("bridge_parent",row["id"])))
        nodes.append({"node_id":row["id"],"candidate_id":parent,"layer":1,"region":"late",
            "source_state_hash":row.get("physical_state_hash",row["id"]),"physical_state":row,
            "parent_node_id":str(index),"tube_distance":distance,"nearest_tube_index":index})
    return nodes


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--run",default=str(DEFAULT_RUN));args=parser.parse_args()
    root=Path(args.run)
    if root.exists():raise SystemExit(f"refusing overwrite {root}")
    valid,failed,raw=verified_assets_allowing_runtime_gate_refresh()
    if not valid:raise SystemExit(f"frozen asset mismatch: {failed}; raw={raw}")
    gate=json.loads(Path("docs/RUNTIME_GATE.json").read_text())
    if gate.get("status")!="PASS" or gate.get("source_fingerprint")!=source_fingerprint(Path.cwd()):raise SystemExit("runtime gate stale")
    source=SnapshotBank.load(SOURCE);tube=SnapshotBank.load(TUBE);artifact=pickle.loads((EXPERT/"adapter.pkl").read_bytes())
    cfg=load_config("configs/backward_descent_rsi_pilot_v1.json",{"use_bank_resets":False,"expert_chain_termination":False,
        "domain_randomization":False,"obs_noise_enable":False})
    tube_features=np.asarray([row["entry_feature"] for row in tube.records],float)
    center,scale=robust_normalization(tube_features,cfg.descent_entry_scale_floors)
    nodes=bridge_nodes(source.records,tube_features,center,scale,cfg);root.mkdir(parents=True)
    frozen={"source_bank_sha256":file_sha256(SOURCE),"tube_sha256":file_sha256(TUBE),
        "adapter_sha256":file_sha256(EXPERT/"adapter.pkl"),"policy_identity_hash":artifact["policy_identity_hash"],
        "C_L_sha256":file_sha256(C_L),"selection":"all eight pre-existing stable physical Descent bridge snapshots",
        "outcome_blind":True,"nodes":[{k:node[k] for k in ("node_id","candidate_id","tube_distance","nearest_tube_index")} for node in nodes]}
    save_json(root/"manifest.json",frozen);save_json(root/"cost_estimate.json",{"estimated_seconds":900,"states":len(nodes),
        "rollouts_per_state":"2 exact + 4 fixed P1 micro","PPO_steps":0,"new_search":False})
    dparams,_,_=load_bundle(PI_D,verify_files=True);lparams,_,_=load_bundle(PI_L,verify_files=True)
    env=OrangeBikeDVGC(cfg,snapshot_bank=SnapshotBank(),cert_bank=SnapshotBank.load(C_L))
    adapter=compact_observation_command_adapter(jnp.asarray(artifact["prototypes"]),jnp.asarray(artifact["targets"]),
        jnp.asarray(artifact["normalizer_mean"]),jnp.asarray(artifact["normalizer_std"]),float(artifact["radius"]),float(artifact["core_radius"]))
    cert=certify_policy(env,dparams,lparams,nodes,3_400_000_000,record_loader=lambda node:node["physical_state"],
        descent_action_adapter=adapter,policy_identity_hash=artifact["policy_identity_hash"])
    save_json(root/"certification.json",cert)
    p1_ids={row["node_id"] for row in cert["rows"] if row["P1"]["pass"]};p0_ids={row["node_id"] for row in cert["rows"] if row["P0"]["pass"]}
    accepted=[]
    for node in nodes:
        if node["node_id"] not in p1_ids:continue
        item=copy.deepcopy(node["physical_state"]);item.update({"id":node["node_id"],"origin_phase":"apex_bridge",
            "source_phase":"flight","artifact_role":"expert_conditioned_provisional_envelope","safe_claim_allowed":False,
            "formal_jel_eligible":False,"bootstrap_eligible":True,"next_stage":"descent","descent_tube_distance":node["tube_distance"],
            "policy_identity_hash":artifact["policy_identity_hash"],"descent_tube_sha256":file_sha256(TUBE)})
        accepted.append(item)
    proposal_path=root/"apex_to_descent_predecessor_support.pkl"
    SnapshotBank(accepted,{"artifact_role":"expert_conditioned_provisional_envelope","safe_claim_allowed":False,
        "source_bank_sha256":file_sha256(SOURCE),"descent_tube_sha256":file_sha256(TUBE),
        "policy_identity_hash":artifact["policy_identity_hash"],"PPO_authorization":False}).save(proposal_path)
    unique_parents=len({node["candidate_id"] for node in nodes if node["node_id"] in p1_ids})
    status="PASS" if len(p1_ids)>=2 and unique_parents>=2 else "FAIL"
    report={"status":status,"head":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        "states":len(nodes),"P0":len(p0_ids),"P1":len(p1_ids),"unique_P1_parents":unique_parents,
        "tube_distance":{"min":min(n["tube_distance"] for n in nodes),"median":float(np.median([n["tube_distance"] for n in nodes])),
            "max":max(n["tube_distance"] for n in nodes)},"proposal_bank":str(proposal_path),"proposal_bank_sha256":file_sha256(proposal_path),
        "PPO_authorization":False,"next":"apex_predecessor_local_expansion" if status=="PASS" else "apex_bridge_support_gap_diagnosis",
        "rows":[{"id":row["node_id"],"candidate_id":row["candidate_id"],"tube_distance":row["tube_distance"],
            "P0":row["node_id"] in p0_ids,"P1":row["node_id"] in p1_ids} for row in nodes]}
    save_json(root/"APEX_BRIDGE_COMPACT_DESCENT_V1_REPORT.json",report);save_json(root/"completed.json",{"status":status,"next":report["next"]})
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
