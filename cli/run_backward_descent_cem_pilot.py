"""Run a fixed-budget first-tier CEM on the nearest unresolved Descent proposals."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from cli.run_backward_descent_nominal_pilot import (
    C_L, EXPECTED, PI_D, PI_L, PERTURBATIONS, _batched, _load_record,
    _micro_states, _outcome,
)
from cli.runtime_gate import source_fingerprint
from dvgc.backward_search import active_prefix_exact, bounded_cem, make_descent_landing_rollout
from dvgc.backward_tube import (
    BackwardTubeNode, canonical_hash, p0_decision, p1_decision,
    summarize_tube_nodes, tube_gate, validate_parent_lineage,
)
from dvgc.bank import SnapshotBank
from dvgc.config import ACTION_MAPPING_VERSION, file_sha256, load_config
from dvgc.env import OrangeBikeDVGC
from dvgc.policy import load_bundle
from dvgc.runtime import save_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",required=True);parser.add_argument("--prior-report",required=True)
    parser.add_argument("--limit",type=int,default=3);args=parser.parse_args()
    root=Path(args.run);root.mkdir(parents=True,exist_ok=True)
    if file_sha256(C_L)!=EXPECTED["C_L"] or file_sha256(PI_D/"params.pkl")!=EXPECTED["pi_D"] or file_sha256(PI_L/"params.pkl")!=EXPECTED["pi_L"]:raise SystemExit("frozen asset mismatch")
    dparams,dcfg,_=load_bundle(PI_D,verify_files=True);lparams,_,_=load_bundle(PI_L,verify_files=True)
    cfg=load_config("configs/default.json",{**dcfg,"episode_length":750,"use_bank_resets":False,"domain_randomization":False,"obs_noise_enable":False,"expert_chain_termination":False,"training_stage":"flight"})
    if file_sha256(cfg.xml_path)!=EXPECTED["xml"] or cfg.action_mapping_version!=ACTION_MAPPING_VERSION:raise SystemExit("runtime model mismatch")
    gate=json.loads((root.parent/"RUNTIME_GATE.json").read_text())
    if gate.get("status")!="PASS" or gate.get("source_fingerprint")!=source_fingerprint(Path.cwd()):raise SystemExit("runtime gate stale")
    entry=SnapshotBank.load(C_L);env=OrangeBikeDVGC(cfg,snapshot_bank=SnapshotBank(),cert_bank=entry)
    rollout=make_descent_landing_rollout(env,dparams,lparams,horizon=200,residual_ticks=8)
    prior=json.loads(Path(args.prior_report).read_text());rows=list(prior["rows"]);nodes=list(prior["nodes"])
    unresolved=[row for row in prior["rows"] if row["proposal"]["region"]=="early" and not row["P0"]["pass"]]
    selected=sorted(unresolved,key=lambda row:row["repeats"][0]["minimum_distance"])[:args.limit]
    searches=[]
    for position,row in enumerate(selected):
        proposal=row["proposal"];record=_load_record(proposal);seed=42_000_000+position
        factory=lambda count,record=record,seed=seed:_batched(env,record,count,seed)
        residual,best,history=bounded_cem(rollout,factory,seed=seed,samples=64,iterations=5,knot_count=2,bound=.2)
        state=_batched(env,record,1,seed);commands=jnp.asarray(residual[None])
        first=jax.device_get(rollout(state,commands,jax.random.PRNGKey(seed+500)))
        second=jax.device_get(rollout(state,commands,jax.random.PRNGKey(seed+500)))
        exact,mismatch=active_prefix_exact(first,second)
        repeats=[_outcome(first,0,exact,mismatch),_outcome(second,0,exact,mismatch)];p0=p0_decision(repeats)
        branches=[];p1={"pass":False,"reasons":["p0_not_passed"],"successes":0,"branches":0}
        if p0["pass"]:
            micro=_micro_states(env,record,seed+1000);branch_raw=jax.device_get(rollout(micro,jnp.repeat(commands,4,axis=0),jax.random.PRNGKey(seed+2000)))
            branches=[_outcome(branch_raw,i)|{"perturbation_vx_vz":PERTURBATIONS[i].tolist()} for i in range(4)]
            p1=p1_decision(p0,branches,repeats[0]["failure_type"])
            node=BackwardTubeNode(node_id=hashlib.sha256(f"descent:{proposal['physical_state_sha256']}:{residual.tobytes().hex()}".encode()).hexdigest()[:32],phase="descent",layer=int(proposal["shell_layer"]),region=proposal["region"],candidate_id=proposal["candidate_id"],source_state_hash=proposal["physical_state_sha256"],physical_state={"source_artifact":proposal["source_artifact"],"source_index":proposal["source_index"],"snapshot_sha256":proposal["snapshot_sha256"]},actor_observation=np.asarray(state.obs["state"])[0].tolist(),parent_node_id=proposal["nearest_downstream_node_id"],parent_tube="canonical_C_L",controller_type="bounded_residual_cem_64x5_h8",controller_artifact_sha256=canonical_hash({"residual":residual.tolist(),"pi_D":EXPECTED["pi_D"],"pi_L":EXPECTED["pi_L"]}),entry_tick=repeats[0]["downstream_entry_tick"],downstream_entry_state={"qpos":np.asarray(first["entry_qpos"])[0].tolist(),"qvel":np.asarray(first["entry_qvel"])[0].tolist(),"nearest_C_L_node_id":proposal["nearest_downstream_node_id"]},final_recovery=True,p0=True,p1=bool(p1["pass"]),branch_results=tuple(branches),nearest_neighbor_radius=0.0,provenance_hashes={"xml":EXPECTED["xml"],"C_L":EXPECTED["C_L"],"pi_D":EXPECTED["pi_D"],"pi_L":EXPECTED["pi_L"]});node.validate();nodes.append(node.to_dict())
        result={"proposal":proposal,"controller":"bounded_residual_cem_64x5_h8","residual_knots":residual.tolist(),"search_best":best,"search_history":history,"repeats":repeats,"P0":p0,"micro_branches":branches,"P1":p1}
        rows.append(result);searches.append(result)
        save_json(root/"descent_cem_pilot.partial.json",{"completed":position+1,"total":len(selected),"new_P0":sum(x["P0"]["pass"] for x in searches),"new_P1":sum(x["P1"]["pass"] for x in searches)})
    typed=[BackwardTubeNode(**node) for node in nodes];safe_ids={row["id"] for row in entry.records if row["final"]["label"]=="safe"}
    lineage=validate_parent_lineage(typed,safe_ids);gate_result=tube_gate(typed)
    report={"status":"PASS","artifact_role":"nominal_provisional_tube_construction_cem_pilot","tier":{"samples":64,"iterations":5,"horizon_ticks":8,"bound":.2},"searched":len(searches),"new_P0":sum(x["P0"]["pass"] for x in searches),"new_P1":sum(x["P1"]["pass"] for x in searches),"total_P0":sum(x["P0"]["pass"] for x in rows),"total_P1":sum(x["P1"]["pass"] for x in rows),**summarize_tube_nodes(typed),"lineage":lineage,"RSI_start_gate":gate_result,"searches":searches,"rows":rows,"nodes":nodes,"heldout_used":False,"delay":False,"new_CEM":True,"PPO":False,"provenance":{"prior_report":args.prior_report,"C_L":EXPECTED["C_L"],"pi_D":EXPECTED["pi_D"],"pi_L":EXPECTED["pi_L"],"xml":EXPECTED["xml"]}}
    save_json(root/"descent_cem_pilot_report.json",report);save_json(root/"descent_cem_pilot.completed.json",{"status":"PASS","new_P0":report["new_P0"],"new_P1":report["new_P1"],"RSI_start_gate":gate_result["status"]})
    print(json.dumps({key:report[key] for key in ("searched","new_P0","new_P1","total_P0","total_P1","candidate_coverage","layer_coverage","region_coverage")}|{"gate":gate_result["status"]},indent=2))


if __name__=="__main__":main()
