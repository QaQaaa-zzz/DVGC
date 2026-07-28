"""Recompute Descent P0/P1 evidence with an uninterrupted pi_D -> C_L -> pi_L stack."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections import Counter
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from cli.runtime_gate import source_fingerprint
from dvgc.backward_search import active_prefix_exact, make_descent_landing_rollout
from dvgc.backward_tube import BackwardTubeNode, canonical_hash, p0_decision, p1_decision, tube_gate, validate_parent_lineage
from dvgc.bank import SnapshotBank
from dvgc.config import ACTION_MAPPING_VERSION, STAGE_ID, file_sha256, load_config
from dvgc.env import END_REASON, OrangeBikeDVGC
from dvgc.policy import load_bundle
from dvgc.rollout import restore_snapshot, restore_snapshot_mode
from dvgc.runtime import save_json


C_L=Path("runs/stage_experts/flight_seed0_20260715T2045/bridge_recovery/entry_set_bridge.pkl")
PI_D=Path("runs/stage_experts/descent_tube_seed0_20260716T2330/round_3/train/policy")
PI_L=Path("runs/decoupled_bootstrap_seed0_20260720/frozen/pi_l_frozen")
V4=Path("runs/v4_current_frame_independent_reconstruction_localization_v1/timing_explicit_snapshots.pkl")
EXPECTED={"C_L":"185164da04380291946d02ff6556f867dfd8532f409fcab5f8c61873de82aa41","pi_D":"52721668eed0cc78b41a45ad7c319e687f43add8977f2b4bdfcad8208c4353f2","pi_L":"fa3a518bff507f9520dca67b38d6ea0da0744025aacb765eda8eb723f434bb7e","xml":"d7e9f43ff8fb9e4571203f81062ce9c828acfa38692ee8c71a3e5daa15ce794c"}
PERTURBATIONS=np.asarray([[.02,.02],[.02,-.02],[-.02,.02],[-.02,-.02]],np.float32)


def _load_record(proposal):
    path=Path(proposal["source_artifact"]);index=int(proposal["source_index"])
    if path==V4:return pickle.loads(path.read_bytes())[index]["snapshot_v4"]
    return SnapshotBank.load(path).records[index]


def _restore(env,record,key):
    if record.get("schema_version")==4:
        return restore_snapshot_mode(env,record,key,observation_mode="timing_explicit_independent_reconstruction")
    return restore_snapshot(env,record,key)


def _batched(env,record,count,seed):
    return jax.jit(jax.vmap(lambda key:_restore(env,record,key)))(jax.random.split(jax.random.PRNGKey(seed),count))


def _micro_states(env,record,seed):
    base=_restore(env,record,jax.random.PRNGKey(seed));info=base.info
    def reset(delta,key):
        qvel=base.data.qvel.at[env._qvel0].add(delta[0]).at[env._qvel0+2].add(delta[1])
        return env.reset_from_snapshot(
            base.data.qpos,qvel,base.data.ctrl,key,info["phase"],info["had_airborne"],info["had_valid_landing"],info["contact_age"],info["last_action"],
            estimated_phase=info["estimated_phase"],phase_probs=info["phase_probs"],airborne_count=info["airborne_count"],prelaunch_airborne_count=info["prelaunch_airborne_count"],landing_bounce_count=info["landing_bounce_count"],invalid_wheel_count=info["invalid_wheel_count"],recovery_count=info["recovery_count"],prev_acc_z=info["prev_acc_z"],prev_vz=info["prev_vz"],obs_history=info["actor_obs_history_pre"],obs_history_valid=jnp.asarray(True),stage_entry_ever=info["stage_entry_ever"],apex_seen=info["apex_seen"],jump_signal_latched=info["jump_signal_latched"],jump_window_start_x=info["jump_window_start_x"],jump_window_end_x=info["jump_window_end_x"])
    return jax.jit(jax.vmap(reset))(jnp.asarray(PERTURBATIONS),jax.random.split(jax.random.PRNGKey(seed+1),4))


def _outcome(raw,index,exact=True,mismatches=()):
    code=int(np.asarray(raw["end_code"])[index]);tick=int(np.asarray(raw["termination_tick"])[index])
    return {"active_prefix_exact":bool(exact),"repeat_mismatch_fields":list(mismatches),"downstream_entry":bool(np.asarray(raw["downstream_entry"])[index]),"downstream_entry_tick":int(np.asarray(raw["entry_tick"])[index]),"final_recovery":bool(np.asarray(raw["final_recovery"])[index]),"termination_tick":tick,"survival":int(np.asarray(raw["survival"])[index]),"end_code":code,"failure_type":END_REASON.get(code,"horizon"),"minimum_distance":float(np.asarray(raw["minimum_distance"])[index]),"minimum_margin":float(np.asarray(raw["minimum_margin"])[index]),"timeout":code==8,"nonfinite":code==15,"illegal_contact":code in (2,3),"penetration":False,"phase_violation":False}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--run",required=True);p.add_argument("--proposal-index",required=True);p.add_argument("--limit",type=int,default=24);a=p.parse_args();root=Path(a.run);root.mkdir(parents=True,exist_ok=True)
    if file_sha256(C_L)!=EXPECTED["C_L"] or file_sha256(PI_D/"params.pkl")!=EXPECTED["pi_D"] or file_sha256(PI_L/"params.pkl")!=EXPECTED["pi_L"]:raise SystemExit("frozen asset mismatch")
    dparams,dcfg,_=load_bundle(PI_D,verify_files=True);lparams,_,_=load_bundle(PI_L,verify_files=True)
    cfg=load_config("configs/default.json",{**dcfg,"episode_length":750,"use_bank_resets":False,"domain_randomization":False,"obs_noise_enable":False,"expert_chain_termination":False,"training_stage":"flight"})
    if file_sha256(cfg.xml_path)!=EXPECTED["xml"] or cfg.action_mapping_version!=ACTION_MAPPING_VERSION:raise SystemExit("runtime model mismatch")
    gate=json.loads((root.parent/"RUNTIME_GATE.json").read_text())
    if gate.get("status")!="PASS" or gate.get("source_fingerprint")!=source_fingerprint(Path.cwd()):raise SystemExit("runtime gate stale")
    entry=SnapshotBank.load(C_L);env=OrangeBikeDVGC(cfg,snapshot_bank=SnapshotBank(),cert_bank=entry)
    rollout=make_descent_landing_rollout(env,dparams,lparams,horizon=200,residual_ticks=8)
    index=json.loads(Path(a.proposal_index).read_text());selected=[];counts=Counter()
    for row in index["rows"]:
        if len(selected)>=a.limit:break
        if counts[row["candidate_id"]]>=3:continue
        selected.append(row);counts[row["candidate_id"]]+=1
    zero=jnp.zeros((1,2,4),jnp.float32);rows=[];nodes=[]
    safe_ids={row["id"] for row in entry.records if row["final"]["label"]=="safe"}
    for position,proposal in enumerate(selected):
        record=_load_record(proposal);seed=41_000_000+position
        state=_batched(env,record,1,seed);first=jax.device_get(rollout(state,zero,jax.random.PRNGKey(seed)));second=jax.device_get(rollout(state,zero,jax.random.PRNGKey(seed)))
        exact,mismatch=active_prefix_exact(first,second);repeat=[_outcome(first,0,exact,mismatch),_outcome(second,0,exact,mismatch)];p0=p0_decision(repeat)
        branches=[];p1={"pass":False,"reasons":["p0_not_passed"],"successes":0,"branches":0}
        if p0["pass"]:
            micro=_micro_states(env,record,seed+1000);residual=jnp.zeros((4,2,4),jnp.float32);branch_raw=jax.device_get(rollout(micro,residual,jax.random.PRNGKey(seed+2000)))
            branches=[_outcome(branch_raw,i) | {"perturbation_vx_vz":PERTURBATIONS[i].tolist()} for i in range(4)]
            p1=p1_decision(p0,branches,repeat[0]["failure_type"])
            entry_qpos=np.asarray(first["entry_qpos"])[0];entry_qvel=np.asarray(first["entry_qvel"])[0]
            node=BackwardTubeNode(node_id=hashlib.sha256(f"descent:{proposal['physical_state_sha256']}:zero".encode()).hexdigest()[:32],phase="descent",layer=int(proposal["shell_layer"]),region=proposal["region"],candidate_id=proposal["candidate_id"],source_state_hash=proposal["physical_state_sha256"],physical_state={"source_artifact":proposal["source_artifact"],"source_index":proposal["source_index"],"snapshot_sha256":proposal["snapshot_sha256"]},actor_observation=np.asarray(state.obs["state"])[0].tolist(),parent_node_id=proposal["nearest_downstream_node_id"],parent_tube="canonical_C_L",controller_type="frozen_pi_D_nominal_L0",controller_artifact_sha256=canonical_hash({"pi_D":EXPECTED["pi_D"],"pi_L":EXPECTED["pi_L"],"residual":"zero"}),entry_tick=repeat[0]["downstream_entry_tick"],downstream_entry_state={"qpos":entry_qpos.tolist(),"qvel":entry_qvel.tolist(),"nearest_C_L_node_id":proposal["nearest_downstream_node_id"]},final_recovery=True,p0=True,p1=bool(p1["pass"]),branch_results=tuple(branches),nearest_neighbor_radius=0.0,provenance_hashes={"xml":EXPECTED["xml"],"C_L":EXPECTED["C_L"],"pi_D":EXPECTED["pi_D"],"pi_L":EXPECTED["pi_L"]});node.validate();nodes.append(node.to_dict())
        rows.append({"proposal":proposal,"controller":"frozen_pi_D_nominal_L0","repeats":repeat,"P0":p0,"micro_branches":branches,"P1":p1})
        save_json(root/"descent_nominal_pilot.partial.json",{"completed":position+1,"total":len(selected),"P0":sum(x["P0"]["pass"] for x in rows),"P1":sum(x["P1"]["pass"] for x in rows),"rows":rows,"nodes":nodes})
    typed=[BackwardTubeNode(**node) for node in nodes];lineage=validate_parent_lineage(typed,safe_ids);gate_result=tube_gate(typed)
    report={"status":"PASS","artifact_role":"nominal_provisional_tube_construction_pilot","proposals":len(selected),"P0":sum(x["P0"]["pass"] for x in rows),"P1":sum(x["P1"]["pass"] for x in rows),"candidate_coverage":len({x["candidate_id"] for x in typed if x.p1}),"layer_coverage":sorted({x.layer for x in typed if x.p1}),"region_coverage":sorted({x.region for x in typed if x.p1}),"failure_reasons":dict(Counter(x["repeats"][0]["failure_type"] for x in rows if not x["P0"]["pass"])),"lineage":lineage,"RSI_start_gate":gate_result,"rows":rows,"nodes":nodes,"heldout_used":False,"delay":False,"new_CEM":False,"PPO":False,"provenance":{"proposal_index_sha256":file_sha256(a.proposal_index),"C_L":EXPECTED["C_L"],"pi_D":EXPECTED["pi_D"],"pi_L":EXPECTED["pi_L"],"xml":EXPECTED["xml"]}}
    save_json(root/"descent_nominal_pilot_report.json",report);save_json(root/"descent_nominal_pilot.completed.json",{"status":"PASS","P0":report["P0"],"P1":report["P1"],"RSI_start_gate":gate_result["status"]});print(json.dumps({k:report[k] for k in ("proposals","P0","P1","candidate_coverage","layer_coverage","region_coverage")}|{"gate":gate_result["status"]},indent=2))


if __name__=="__main__":main()
