"""Bounded natural-lineage handoff probe into the frozen compact Descent expert."""
from __future__ import annotations

import argparse
import json
import pickle
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from cli.audit_descent_bridge_sensitivity import _reference_action
from cli.relabel_apex_policies import state_sample, support_diagnostic
from cli.run_backward_descent_nominal_pilot import C_L, EXPECTED, PI_D, PI_L
from cli.runtime_gate import source_fingerprint
from dvgc.backward_search import compact_observation_command_adapter
from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256, load_config
from dvgc.descent_entry import DESCENT_ENTRY_FEATURE_NAMES, descent_entry_feature
from dvgc.entry import normalized_nearest, robust_normalization
from dvgc.env import END_REASON, OrangeBikeDVGC
from dvgc.policy import load_bundle
from dvgc.runtime import build_inference, save_json
from dvgc.stage_reachability import evaluate_entry


TUBE=Path("runs/descent_diverse_p1_predecessor_recovery_v5_p1_core_adapter/independent_audit_v1/descent_tube_v2.pkl")
EXPERT=Path("runs/descent_diverse_p1_predecessor_recovery_v5_p1_core_adapter")
DEFAULT_RUN=Path("runs/natural_compact_descent_bridge_v1/fixed_handoff_probe")
SEED=3_500_000_000


def handoff_ticks(closest_tick, offsets=(-8,-6,-4,-2,0,2,4)):
    return sorted({max(0,int(closest_tick)+int(offset)) for offset in offsets})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--run",default=str(DEFAULT_RUN));parser.add_argument("--tube",default=str(TUBE));args=parser.parse_args();root=Path(args.run);tube_path=Path(args.tube)
    if root.exists():raise SystemExit(f"refusing overwrite {root}")
    frozen={"C_L":file_sha256(C_L),"pi_D":file_sha256(PI_D/"params.pkl"),"pi_L":file_sha256(PI_L/"params.pkl")}
    expected={"C_L":EXPECTED["C_L"],"pi_D":EXPECTED["pi_D"],"pi_L":EXPECTED["pi_L"]}
    if frozen!=expected:raise SystemExit(f"frozen scientific asset mismatch: {frozen}")
    gate=json.loads(Path("docs/RUNTIME_GATE.json").read_text())
    if gate.get("status")!="PASS" or gate.get("source_fingerprint")!=source_fingerprint(Path.cwd()):raise SystemExit("runtime gate stale")
    artifact=pickle.loads((EXPERT/"adapter.pkl").read_bytes());tube=SnapshotBank.load(tube_path)
    if artifact["base_policy_sha256"]!=EXPECTED["pi_D"]:raise SystemExit("base policy mismatch")
    dparams,_,_=load_bundle(PI_D,verify_files=True);lparams,_,_=load_bundle(PI_L,verify_files=True)
    cfg=load_config("configs/backward_descent_rsi_pilot_v1.json",{"training_stage":"full","use_bank_resets":False,
        "stage_reachability_objective":"","expert_chain_termination":False,"domain_randomization":False,"obs_noise_enable":False})
    env=OrangeBikeDVGC(cfg,snapshot_bank=SnapshotBank(),cert_bank=SnapshotBank.load(C_L));step=jax.jit(env.step)
    d_infer=build_inference(env,dparams,deterministic=True);l_infer=build_inference(env,lparams,deterministic=True)
    adapter=compact_observation_command_adapter(jnp.asarray(artifact["prototypes"]),jnp.asarray(artifact["targets"]),
        jnp.asarray(artifact["normalizer_mean"]),jnp.asarray(artifact["normalizer_std"]),float(artifact["radius"]),float(artifact["core_radius"]))
    reference=pd.read_csv("data/reference_jump.csv");tube_features=np.asarray([r.get("entry_feature",descent_entry_feature(r["physical_feature"],cfg)) for r in tube.records],float)
    center,scale=robust_normalization(tube_features,cfg.descent_entry_scale_floors)
    support_metadata=dict(tube.metadata);support_metadata["support_features"]=[r["physical_feature"] for r in tube.records]
    local_matcher=bool(support_metadata.get("stage_entry_matcher",{}).get("radii"))

    def run(switch_tick):
        state=env.reset(jax.random.PRNGKey(SEED));key=jax.random.PRNGKey(SEED+1);approach=0
        while int(state.info["phase"])<1 and not float(state.done):
            state=step(state,np.asarray([0.,1.,0.,0.],np.float32));approach+=1
        positive=float(state.data.qvel[2])>0;previous_vz=float(state.data.qvel[2]);apex=None;landing=None;closest=None;stable=0;trace=[]
        for tick in range(220):
            if switch_tick is None or tick<switch_tick:
                action=_reference_action(env,cfg,state,reference,tick,3,105,10);controller="reference"
            elif not int(state.info["had_valid_landing"]):
                key,sub=jax.random.split(key);base,_=d_infer(state.obs,sub)
                action=np.asarray(adapter(state.obs["state"][None],base[None])[0]);controller="compact_descent";stable+=1
            else:
                key,sub=jax.random.split(key);action=np.asarray(l_infer(state.obs,sub)[0]);controller="landing";landing=landing or tick
            state=step(state,np.clip(action,-1,1));physical=np.asarray(env._physical_feature(state.data),float)
            feature=descent_entry_feature(physical,cfg)
            if local_matcher:
                diagnostic=support_diagnostic(physical,support_metadata);distance=diagnostic["distance"];index=diagnostic["anchor_index"];contribution=np.asarray(diagnostic["squared_scaled_contributions"])
            else:
                distance,index,contribution=normalized_nearest(feature,tube_features,center,scale)
            vz=float(physical[8]);positive|=vz>0
            if apex is None and positive and previous_vz>0 and vz<=0 and not float(state.done):apex=tick+1
            previous_vz=vz
            if apex is not None and (closest is None or distance<closest["distance"]):
                top=np.argsort(contribution)[-6:][::-1];closest={"tick":tick+1,"distance":distance,"nearest_tube_index":index,
                    "feature":feature.tolist(),"top_residuals":[{"name":DESCENT_ENTRY_FEATURE_NAMES[i],"raw":float(feature[i]-tube_features[index,i]),
                    "squared_normalized_contribution":float(contribution[i])} for i in top]}
            sample,_=state_sample(env,state,apex_seen=apex is not None,previous_vz=previous_vz)
            formal_entry=evaluate_entry("apex",sample,cfg,support_metadata)["valid"] if local_matcher else False
            trace.append({"tick":tick+1,"controller":controller,"distance":distance,"formal_entry":formal_entry,"phase":int(state.info["phase"]),
                "end_code":int(state.info["end_code"]),"action":np.asarray(action).tolist()})
            if float(state.done):break
        return {"switch_tick":switch_tick,"approach_ticks":approach,"apex_tick":apex,"closest":closest,"stable_descent_ticks":stable,
            "landing":landing is not None or bool(int(state.info["had_valid_landing"])),"final_recovery":bool(int(state.info["recovery_success"])),
            "formal_entries":sum(row["formal_entry"] for row in trace),"termination_reason":END_REASON.get(int(state.info["end_code"]),"unknown"),"steps":len(trace),"trace":trace}

    root.mkdir(parents=True);inputs={"tube_sha256":file_sha256(tube_path),"adapter_sha256":file_sha256(EXPERT/"adapter.pkl"),
        "policy_identity_hash":artifact["policy_identity_hash"],"C_L_sha256":file_sha256(C_L),"xml_sha256":EXPECTED["xml"],"seed":SEED}
    save_json(root/"manifest.json",{"status":"FROZEN_BEFORE_OUTCOMES","inputs":inputs,"reference_offset":105,"reference_stride":10,
        "handoff_offsets":[-8,-6,-4,-2,0,2,4],"selection":"nominal closest post-apex tick; fixed offsets","matcher_active":local_matcher})
    save_json(root/"cost_estimate.json",{"estimated_seconds":900,"natural_rollouts":10,"horizon":220,"PPO_steps":0,"new_action_search":False})
    nominal=run(None)
    if nominal["apex_tick"] is None or nominal["closest"] is None:raise SystemExit("nominal natural rollout did not cross apex")
    rows=[run(tick) for tick in handoff_ticks(nominal["closest"]["tick"])]
    order=sorted(range(len(rows)),key=lambda i:(not rows[i]["final_recovery"],not rows[i]["landing"],-rows[i]["stable_descent_ticks"],rows[i]["closest"]["distance"] if rows[i]["closest"] else float("inf")))
    top=rows[order[0]];replay=run(top["switch_tick"]);exact=top==replay
    save_json(root/"nominal.json",nominal);save_json(root/"handoff_rows.json",{"rows":rows,"top_replay":replay})
    final=sum(row["final_recovery"] for row in rows);landings=sum(row["landing"] for row in rows)
    report={"status":"PASS" if final else "FAIL","head":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        "nominal":{"apex_tick":nominal["apex_tick"],"closest":nominal["closest"],"termination_reason":nominal["termination_reason"]},
        "handoff_ticks":[row["switch_tick"] for row in rows],"landings":landings,"final_recoveries":final,
        "formal_entries":sum(row["formal_entries"] for row in rows),"maximum_stable_descent_ticks":max(row["stable_descent_ticks"] for row in rows),"termination_reasons":{reason:sum(r["termination_reason"]==reason for r in rows) for reason in sorted({r["termination_reason"] for r in rows})},
        "top":{"switch_tick":top["switch_tick"],"stable_descent_ticks":top["stable_descent_ticks"],"landing":top["landing"],
            "final_recovery":top["final_recovery"],"termination_reason":top["termination_reason"],"closest":top["closest"]},
        "top_exact_replay":exact,"PPO_authorization":False,
        "next":"bounded_pre_handoff_residual_probe" if not final else "natural_teacher_neighborhood_audit"}
    save_json(root/"NATURAL_COMPACT_DESCENT_BRIDGE_V1_REPORT.json",report);save_json(root/"completed.json",{"status":report["status"],"next":report["next"]})
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
