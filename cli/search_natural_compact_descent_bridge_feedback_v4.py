"""Bounded event-timing and pose-feedback search for the natural Descent bridge."""
from __future__ import annotations

import argparse,json,pickle,subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from cli.audit_descent_bridge_sensitivity import _reference_action
from cli.run_backward_descent_nominal_pilot import C_L,EXPECTED,PI_D,PI_L
from cli.run_descent_localized_consolidation_v1 import verified_assets_allowing_runtime_gate_refresh
from cli.runtime_gate import source_fingerprint
from cli.search_natural_compact_descent_bridge_local_v1 import latin_hypercube
from dvgc.backward_search import compact_observation_command_adapter
from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256,load_config
from dvgc.descent_entry import descent_entry_feature
from dvgc.entry import normalized_nearest,robust_normalization
from dvgc.env import END_REASON,OrangeBikeDVGC
from dvgc.policy import load_bundle
from dvgc.runtime import build_inference,save_json

TUBE=Path("runs/descent_diverse_p1_predecessor_recovery_v5_p1_core_adapter/independent_audit_v1/descent_tube_v2.pkl")
EXPERT=Path("runs/descent_diverse_p1_predecessor_recovery_v5_p1_core_adapter")
PRIOR=Path("runs/natural_compact_descent_bridge_v1/reference_timing_search_v3/NATURAL_COMPACT_DESCENT_TIMING_SEARCH_V3_REPORT.json")
DEFAULT_RUN=Path("runs/natural_compact_descent_bridge_v1/pose_feedback_search_v4")
SEED=3_540_000_000;SWITCHES=(18,20,22,24,26);GAIN_STEPS=np.asarray([.5,.1,.5,.1],np.float32)

def score(row):
    return (int(row["final_recovery"]),int(row["landing"]),int(not row["early_failure"]),int(row["minimum_pose_margin"]>=0),
            -float(row["minimum_distance"]),int(row["stable_descent_ticks"]),float(row["minimum_pose_margin"]))

def feedback_residual(feature,gains):
    roll,pitch,wx,wy=float(feature[3]),float(feature[4]),float(feature[9]),float(feature[10])
    residual=np.zeros(4,np.float32);residual[0]=gains[0]*roll+gains[1]*wx
    bend=gains[2]*pitch+gains[3]*wy;residual[2]=bend;residual[3]=-.5*bend
    return np.clip(residual,-.2,.2)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--run",default=str(DEFAULT_RUN));a=p.parse_args();root=Path(a.run)
    if root.exists():raise SystemExit(f"refusing overwrite {root}")
    prior=json.loads(PRIOR.read_text());valid,failed,raw=verified_assets_allowing_runtime_gate_refresh()
    if prior["status"]!="PROGRESS" or prior["next"]!="timing_aligned_local_residual_refinement":raise SystemExit("round-3 prerequisite mismatch")
    if not valid:raise SystemExit(f"frozen asset mismatch: {failed}; raw={raw}")
    gate=json.loads(Path("docs/RUNTIME_GATE.json").read_text())
    if gate.get("status")!="PASS" or gate.get("source_fingerprint")!=source_fingerprint(Path.cwd()):raise SystemExit("runtime gate stale")
    artifact=pickle.loads((EXPERT/"adapter.pkl").read_bytes());tube=SnapshotBank.load(TUBE);dparams,_,_=load_bundle(PI_D,verify_files=True);lparams,_,_=load_bundle(PI_L,verify_files=True)
    cfg=load_config("configs/backward_descent_rsi_pilot_v1.json",{"training_stage":"full","use_bank_resets":False,"stage_reachability_objective":"",
        "expert_chain_termination":False,"domain_randomization":False,"obs_noise_enable":False});env=OrangeBikeDVGC(cfg,snapshot_bank=SnapshotBank(),cert_bank=SnapshotBank.load(C_L));step=jax.jit(env.step)
    di=build_inference(env,dparams,deterministic=True);li=build_inference(env,lparams,deterministic=True)
    adapter=compact_observation_command_adapter(jnp.asarray(artifact["prototypes"]),jnp.asarray(artifact["targets"]),jnp.asarray(artifact["normalizer_mean"]),
        jnp.asarray(artifact["normalizer_std"]),float(artifact["radius"]),float(artifact["core_radius"]))
    reference=pd.read_csv("data/reference_jump.csv");features=np.asarray([r["entry_feature"] for r in tube.records],float);center,scale=robust_normalization(features,cfg.descent_entry_scale_floors)

    def run(switch,gains):
        gains=np.asarray(gains,np.float32);state=env.reset(jax.random.PRNGKey(SEED));key=jax.random.PRNGKey(SEED+1);approach=0
        while int(state.info["phase"])<1 and not float(state.done):state=step(state,np.asarray([0.,1.,0.,0.],np.float32));approach+=1
        positive=float(state.data.qvel[2])>0;previous=float(state.data.qvel[2]);apex=None;landing=False;stable=0;minimum=float("inf");pose_margin=float("inf");closest=None
        for tick in range(220):
            if tick<switch:
                action=_reference_action(env,cfg,state,reference,tick,2,105,10)
                if tick>=6:action=np.clip(action+feedback_residual(np.asarray(env._physical_feature(state.data),float),gains),-1,1)
            elif not int(state.info["had_valid_landing"]):
                key,sub=jax.random.split(key);base,_=di(state.obs,sub);action=np.asarray(adapter(state.obs["state"][None],base[None])[0]);stable+=1
            else:
                key,sub=jax.random.split(key);action=np.asarray(li(state.obs,sub)[0]);landing=True
            state=step(state,np.clip(action,-1,1));physical=np.asarray(env._physical_feature(state.data),float);feature=descent_entry_feature(physical,cfg);distance,index,_=normalized_nearest(feature,features,center,scale)
            minimum=min(minimum,distance);margin=min(np.deg2rad(float(cfg.max_pitch_deg))-abs(float(physical[4])),np.deg2rad(float(cfg.max_roll_deg))-abs(float(physical[3])));pose_margin=min(pose_margin,margin)
            if closest is None or distance<closest["distance"]:closest={"tick":tick+1,"distance":distance,"nearest_tube_index":index,"feature":feature.tolist()}
            vz=float(physical[8]);positive|=vz>0
            if apex is None and positive and previous>0 and vz<=0 and not float(state.done):apex=tick+1
            previous=vz
            if float(state.done):break
        return {"switch_tick":switch,"gains":gains.tolist(),"approach_ticks":approach,"apex_tick":apex,"early_failure":bool(float(state.done) and stable==0),
            "stable_descent_ticks":stable,"landing":landing or bool(int(state.info["had_valid_landing"])),"final_recovery":bool(int(state.info["recovery_success"])),
            "minimum_distance":minimum,"minimum_pose_margin":pose_margin,"closest":closest,"termination_reason":END_REASON.get(int(state.info["end_code"]),"unknown")}

    root.mkdir(parents=True);inputs={"prior_sha256":file_sha256(PRIOR),"tube_sha256":file_sha256(TUBE),"adapter_sha256":file_sha256(EXPERT/"adapter.pkl"),"xml":EXPECTED["xml"],"seed":SEED}
    save_json(root/"manifest.json",{"status":"FROZEN_BEFORE_OUTCOMES","inputs":inputs,"switch_ticks":SWITCHES,"reference_offset":105,"pulse_ticks":2,
        "feedback_start_tick":6,"feedback":"steer=k_roll*roll+k_wx*wx; hip=k_pitch*pitch+k_wy*wy; knee=-0.5*hip; residual clipped +/-0.2",
        "gain_steps":GAIN_STEPS.tolist(),"expansion":"64-point LHS over four gains normalized to +/- one finite-difference step","PPO_authorization":False})
    save_json(root/"cost_estimate.json",{"estimated_seconds":1800,"switch_screen":5,"finite_difference":9,"maximum_expansion":64,"horizon":220,"PPO_steps":0})
    zero=np.zeros(4,np.float32);screen=[run(switch,zero) for switch in SWITCHES];base=max(screen,key=score);pilot=[]
    for dim in range(4):
        for sign in (-1,1):
            gains=zero.copy();gains[dim]=sign*GAIN_STEPS[dim];row=run(base["switch_tick"],gains);row.update({"perturbed_dimension":dim,"sign":sign});pilot.append(row)
    improved=max([base]+pilot,key=score)!=base;save_json(root/"control_authority_pilot.json",{"switch_screen":screen,"baseline":base,"rows":pilot,"improved":improved})
    expansion=[]
    if improved:
        for normalized in latin_hypercube(SEED+100,64,4,1.):expansion.append(run(base["switch_tick"],normalized*GAIN_STEPS))
    top=max(screen+pilot+expansion,key=score);a1=run(top["switch_tick"],top["gains"]);a2=run(top["switch_tick"],top["gains"]);exact=top==a1==a2
    save_json(root/"search_results.json",{"screen":screen,"baseline":base,"pilot":pilot,"expansion":expansion,"top":top,"replay_a":a1,"replay_b":a2})
    progress=top["minimum_distance"]<base["minimum_distance"]-1e-3 or top["stable_descent_ticks"]>base["stable_descent_ticks"] or top["minimum_pose_margin"]>base["minimum_pose_margin"]+.005
    report={"status":"PASS" if top["final_recovery"] and exact else ("PROGRESS" if progress and exact else "FAIL"),"head":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        "screen_rollouts":len(screen),"pilot_rollouts":len(pilot)+1,"expansion_rollouts":len(expansion),"baseline":base,"top":top,
        "distance_improvement":base["minimum_distance"]-top["minimum_distance"],"stable_tick_gain":top["stable_descent_ticks"]-base["stable_descent_ticks"],
        "pose_margin_gain":top["minimum_pose_margin"]-base["minimum_pose_margin"],"exact_replay_twice":exact,"PPO_authorization":False,
        "next":"teacher_neighborhood_audit" if top["final_recovery"] else ("feedback_trust_region_round_2" if progress else "bounded_local_budget_exhausted")}
    save_json(root/"NATURAL_COMPACT_DESCENT_FEEDBACK_SEARCH_V4_REPORT.json",report);save_json(root/"completed.json",{"status":report["status"],"next":report["next"]});print(json.dumps(report,indent=2))

if __name__=="__main__":main()
