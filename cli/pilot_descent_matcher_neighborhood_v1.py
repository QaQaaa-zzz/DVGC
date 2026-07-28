"""Small construction-only neighborhood pilot around the certified Descent Tube."""
from __future__ import annotations

import argparse
import json
import pickle
import subprocess
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from cli.run_backward_descent_nominal_pilot import C_L, PI_D, PI_L, _load_record
from cli.run_backward_descent_rsi_pilot import certify_policy
from cli.run_descent_localized_consolidation_v1 import verified_assets_allowing_runtime_gate_refresh
from dvgc.backward_search import compact_observation_command_adapter
from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256, load_config
from dvgc.descent_entry import descent_entry_feature
from dvgc.entry import robust_normalization
from dvgc.env import OrangeBikeDVGC
from dvgc.policy import load_bundle
from dvgc.runtime import save_json


TUBE=Path("runs/descent_diverse_p1_predecessor_recovery_v5_p1_core_adapter/independent_audit_v1/descent_tube_v2.pkl")
EXPERT=Path("runs/descent_diverse_p1_predecessor_recovery_v5_p1_core_adapter")
INDEX=Path("runs/backward_recovery_tube_fast_track_v1/proposal_state_index.json")
DEFAULT_RUN=Path("runs/descent_compact_matcher_neighborhood_v1/pilot_6x4")


def select_stratified(rows,per_region=2):
    selected=[]
    for region in ("early","middle","late"):
        candidates=sorted((row for row in rows if row["region"]==region),key=lambda row:(row["tube_distance"],row["candidate_id"],row["proposal_id"]))
        used=set()
        for row in candidates:
            if row["candidate_id"] in used:continue
            selected.append(row);used.add(row["candidate_id"])
            if len(used)==per_region:break
        if len(used)<per_region:raise ValueError(f"insufficient {region} candidate diversity")
    return selected


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--run",default=str(DEFAULT_RUN));args=parser.parse_args();root=Path(args.run)
    if root.exists():raise SystemExit(f"refusing overwrite {root}")
    valid,failed,raw=verified_assets_allowing_runtime_gate_refresh()
    if not valid:raise SystemExit(f"frozen asset mismatch: {failed}; raw={raw}")
    tube=SnapshotBank.load(TUBE);artifact=pickle.loads((EXPERT/"adapter.pkl").read_bytes());cfg=load_config("configs/backward_descent_rsi_pilot_v1.json",
        {"use_bank_resets":False,"expert_chain_termination":False,"domain_randomization":False,"obs_noise_enable":False})
    dparams,_,_=load_bundle(PI_D,verify_files=True);lparams,_,_=load_bundle(PI_L,verify_files=True)
    env=OrangeBikeDVGC(cfg,snapshot_bank=SnapshotBank(),cert_bank=SnapshotBank.load(C_L))
    features=np.asarray([row["entry_feature"] for row in tube.records],float);center,scale=robust_normalization(features,cfg.descent_entry_scale_floors)
    excluded={row["state_byte_hash"] for row in tube.records if row.get("state_byte_hash")}
    pool=[]
    for proposal in json.loads(INDEX.read_text())["rows"]:
        if proposal["physical_state_sha256"] in excluded:continue
        record=_load_record(proposal);feature=descent_entry_feature(record["physical_feature"],cfg)
        distance=float(np.min(np.linalg.norm((features-feature[None,:])/scale,axis=1)))
        pool.append({**proposal,"tube_distance":distance})
    selected=select_stratified(pool);root.mkdir(parents=True)
    save_json(root/"cost_estimate.json",{"states":6,"branches_per_state":"2 exact + 4 P1 micro","fraction_of_pool":6/len(pool),
        "estimated_seconds":600,"PPO_steps":0,"matcher_activation":False})
    save_json(root/"selection_manifest.json",{"status":"FROZEN_BEFORE_OUTCOMES","tube_sha256":file_sha256(TUBE),"expert_adapter_sha256":file_sha256(EXPERT/"adapter.pkl"),
        "proposal_index_sha256":file_sha256(INDEX),"selection":"two nearest distinct candidates per early/middle/late in robust-normalized Tube feature space",
        "center":center.tolist(),"scale":scale.tolist(),"rows":[{k:row[k] for k in ("proposal_id","candidate_id","region","shell_layer","tube_distance","physical_state_sha256")} for row in selected]})
    nodes=[{"node_id":row["proposal_id"],"candidate_id":row["candidate_id"],"layer":row["shell_layer"],"region":row["region"],
        "source_state_hash":row["physical_state_sha256"],"physical_state":row,"parent_node_id":row["nearest_downstream_node_id"]} for row in selected]
    adapter=compact_observation_command_adapter(jnp.asarray(artifact["prototypes"]),jnp.asarray(artifact["targets"]),
        jnp.asarray(artifact["normalizer_mean"]),jnp.asarray(artifact["normalizer_std"]),float(artifact["radius"]),float(artifact["core_radius"]))
    cert=certify_policy(env,dparams,lparams,nodes,3_200_000_000,record_loader=lambda node:_load_record(node["physical_state"]),
        descent_action_adapter=adapter,policy_identity_hash=artifact["policy_identity_hash"])
    save_json(root/"construction_pilot.json",cert)
    p0,p1=cert["P0"],cert["P1"];mixed=p1>0 and p1<len(nodes)
    report={"status":"PASS" if mixed else "FAIL","head":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        "states":len(nodes),"P0":p0,"P1":p1,"mixed_construction_support":mixed,"matcher_activated":False,
        "PPO_authorization":False,"next":"expand_construction_only_neighborhood" if mixed else "neighborhood_support_diagnosis",
        "rows":[{"id":row["node_id"],"candidate_id":row["candidate_id"],"region":row["region"],"P0":row["P0"]["pass"],"P1":row["P1"]["pass"]} for row in cert["rows"]]}
    save_json(root/"DESCENT_MATCHER_NEIGHBORHOOD_PILOT_V1_REPORT.json",report);save_json(root/"completed.json",{"status":report["status"],"next":report["next"]})
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
