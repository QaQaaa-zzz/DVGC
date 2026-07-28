"""Bounded reference timing alignment for the natural compact Descent bridge."""
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
V1=Path("runs/natural_compact_descent_bridge_v1/local_residual_search_v1/NATURAL_COMPACT_DESCENT_LOCAL_SEARCH_V1_REPORT.json")
V2=Path("runs/natural_compact_descent_bridge_v1/local_residual_search_v2_early_window/NATURAL_COMPACT_DESCENT_LOCAL_SEARCH_V2_REPORT.json")
DEFAULT_RUN=Path("runs/natural_compact_descent_bridge_v1/reference_timing_search_v3")
SEED=3_530_000_000
OFFSETS=(95,105,115,125,135);PULSES=(1,2,3,4,5)

def score(row):
    return (int(row["final_recovery"]),int(row["landing"]),int(not row["early_failure"]),
            int(row["minimum_pose_margin"]>=0),-float(row["minimum_distance"]),
            int(row["stable_descent_ticks"]),float(row["minimum_pose_margin"]))

def timing_grid():return [(profile,offset,pulse) for profile in ("distance","margin") for offset in OFFSETS for pulse in PULSES]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--run",default=str(DEFAULT_RUN));a=p.parse_args();root=Path(a.run)
    if root.exists():raise SystemExit(f"refusing overwrite {root}")
    v1=json.loads(V1.read_text());v2=json.loads(V2.read_text());valid,failed,raw=verified_assets_allowing_runtime_gate_refresh()
    if v2["status"]!="PROGRESS" or v2["next"]!="bounded_early_window_round_3":raise SystemExit("round-2 prerequisite mismatch")
    if not valid:raise SystemExit(f"frozen asset mismatch: {failed}; raw={raw}")
    gate=json.loads(Path("docs/RUNTIME_GATE.json").read_text())
    if gate.get("status")!="PASS" or gate.get("source_fingerprint")!=source_fingerprint(Path.cwd()):raise SystemExit("runtime gate stale")
    late=np.asarray(v1["top"]["params"],np.float32);early=np.asarray(v2["top"]["params"],np.float32)
    artifact=pickle.loads((EXPERT/"adapter.pkl").read_bytes());tube=SnapshotBank.load(TUBE);dparams,_,_=load_bundle(PI_D,verify_files=True);lparams,_,_=load_bundle(PI_L,verify_files=True)
    cfg=load_config("configs/backward_descent_rsi_pilot_v1.json",{"training_stage":"full","use_bank_resets":False,
        "stage_reachability_objective":"","expert_chain_termination":False,"domain_randomization":False,"obs_noise_enable":False})
    env=OrangeBikeDVGC(cfg,snapshot_bank=SnapshotBank(),cert_bank=SnapshotBank.load(C_L));step=jax.jit(env.step)
    di=build_inference(env,dparams,deterministic=True);li=build_inference(env,lparams,deterministic=True)
    adapter=compact_observation_command_adapter(jnp.asarray(artifact["prototypes"]),jnp.asarray(artifact["targets"]),
        jnp.asarray(artifact["normalizer_mean"]),jnp.asarray(artifact["normalizer_std"]),float(artifact["radius"]),float(artifact["core_radius"]))
    reference=pd.read_csv("data/reference_jump.csv");features=np.asarray([r["entry_feature"] for r in tube.records],float);center,scale=robust_normalization(features,cfg.descent_entry_scale_floors)

    def run(profile,offset,pulse):
        state=env.reset(jax.random.PRNGKey(SEED));key=jax.random.PRNGKey(SEED+1);approach=0
        while int(state.info["phase"])<1 and not float(state.done):
            action=np.asarray([0.,1.,0.,0.],np.float32)
            if profile=="margin" and 10<=approach<16:action=np.clip(action+early[0],-1,1)
            state=step(state,action);approach+=1
        positive=float(state.data.qvel[2])>0;previous=float(state.data.qvel[2]);apex=None;landing=False;stable=0;minimum=float("inf");pose_margin=float("inf");closest=None;switch=16
        for tick in range(220):
            if tick<switch:
                action=_reference_action(env,cfg,state,reference,tick,pulse,offset,10)
                if profile=="margin" and tick<8:action=np.clip(action+early[1],-1,1)
                elif 8<=tick<16:action=np.clip(action+late[min((tick-8)//4,1)],-1,1)
            elif not int(state.info["had_valid_landing"]):
                key,sub=jax.random.split(key);base,_=di(state.obs,sub);action=np.asarray(adapter(state.obs["state"][None],base[None])[0]);stable+=1
            else:
                key,sub=jax.random.split(key);action=np.asarray(li(state.obs,sub)[0]);landing=True
            state=step(state,np.clip(action,-1,1));physical=np.asarray(env._physical_feature(state.data),float);feature=descent_entry_feature(physical,cfg)
            distance,index,_=normalized_nearest(feature,features,center,scale);minimum=min(minimum,distance)
            margin=min(np.deg2rad(float(cfg.max_pitch_deg))-abs(float(physical[4])),np.deg2rad(float(cfg.max_roll_deg))-abs(float(physical[3])));pose_margin=min(pose_margin,margin)
            if closest is None or distance<closest["distance"]:closest={"tick":tick+1,"distance":distance,"nearest_tube_index":index,"feature":feature.tolist()}
            vz=float(physical[8]);positive|=vz>0
            if apex is None and positive and previous>0 and vz<=0 and not float(state.done):apex=tick+1
            previous=vz
            if float(state.done):break
        return {"profile":profile,"reference_offset":offset,"pulse_ticks":pulse,"approach_ticks":approach,"apex_tick":apex,
            "early_failure":bool(float(state.done) and stable==0),"stable_descent_ticks":stable,"landing":landing or bool(int(state.info["had_valid_landing"])),
            "final_recovery":bool(int(state.info["recovery_success"])),"minimum_distance":minimum,"minimum_pose_margin":pose_margin,
            "closest":closest,"termination_reason":END_REASON.get(int(state.info["end_code"]),"unknown")}

    root.mkdir(parents=True);inputs={"v1_sha256":file_sha256(V1),"v2_sha256":file_sha256(V2),"tube_sha256":file_sha256(TUBE),
        "adapter_sha256":file_sha256(EXPERT/"adapter.pkl"),"xml":EXPECTED["xml"],"seed":SEED}
    save_json(root/"manifest.json",{"status":"FROZEN_BEFORE_OUTCOMES","inputs":inputs,"reference_offsets":OFFSETS,"pulse_ticks":PULSES,
        "profiles":{"distance":"v1 late residual only","margin":"v2 early plus v1 late residual"},"switch_tick":16,"grid_size":50,"PPO_authorization":False})
    save_json(root/"cost_estimate.json",{"estimated_seconds":1500,"natural_rollouts":52,"horizon":220,"PPO_steps":0})
    rows=[run(*spec) for spec in timing_grid()];baseline=next(r for r in rows if r["profile"]=="distance" and r["reference_offset"]==105 and r["pulse_ticks"]==3)
    top=max(rows,key=score);a1=run(top["profile"],top["reference_offset"],top["pulse_ticks"]);a2=run(top["profile"],top["reference_offset"],top["pulse_ticks"]);exact=top==a1==a2
    save_json(root/"search_results.json",{"baseline":baseline,"rows":rows,"top":top,"replay_a":a1,"replay_b":a2})
    progress=top["minimum_distance"]<baseline["minimum_distance"]-1e-3 or top["stable_descent_ticks"]>baseline["stable_descent_ticks"] or top["minimum_pose_margin"]>baseline["minimum_pose_margin"]+.005
    report={"status":"PASS" if top["final_recovery"] and exact else ("PROGRESS" if progress and exact else "FAIL"),"head":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        "grid_rollouts":50,"baseline":baseline,"top":top,"distance_improvement":baseline["minimum_distance"]-top["minimum_distance"],
        "stable_tick_gain":top["stable_descent_ticks"]-baseline["stable_descent_ticks"],"pose_margin_gain":top["minimum_pose_margin"]-baseline["minimum_pose_margin"],
        "exact_replay_twice":exact,"PPO_authorization":False,"next":"teacher_neighborhood_audit" if top["final_recovery"] else ("timing_aligned_local_residual_refinement" if progress else "bounded_local_budget_exhausted")}
    save_json(root/"NATURAL_COMPACT_DESCENT_TIMING_SEARCH_V3_REPORT.json",report);save_json(root/"completed.json",{"status":report["status"],"next":report["next"]})
    print(json.dumps(report,indent=2))

if __name__=="__main__":main()
