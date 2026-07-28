"""Bounded local residual search from natural reset into compact pi_D."""
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
from cli.run_backward_descent_nominal_pilot import C_L, EXPECTED, PI_D, PI_L
from cli.run_descent_localized_consolidation_v1 import verified_assets_allowing_runtime_gate_refresh
from cli.runtime_gate import source_fingerprint
from dvgc.backward_search import compact_observation_command_adapter
from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256, load_config
from dvgc.descent_entry import descent_entry_feature
from dvgc.entry import normalized_nearest, robust_normalization
from dvgc.env import END_REASON, OrangeBikeDVGC
from dvgc.policy import load_bundle
from dvgc.runtime import build_inference, save_json


TUBE=Path("runs/descent_diverse_p1_predecessor_recovery_v5_p1_core_adapter/independent_audit_v1/descent_tube_v2.pkl")
EXPERT=Path("runs/descent_diverse_p1_predecessor_recovery_v5_p1_core_adapter")
PROBE=Path("runs/natural_compact_descent_bridge_v1/fixed_handoff_probe/NATURAL_COMPACT_DESCENT_BRIDGE_V1_REPORT.json")
DEFAULT_RUN=Path("runs/natural_compact_descent_bridge_v1/local_residual_search_v1")
SEED=3_510_000_000
CHANNELS=("steer","drive","hip","knee")


def score(row):
    return (int(row["final_recovery"]),int(row["landing"]),int(not row["early_failure"]),
            -float(row["minimum_distance"]),int(row["stable_descent_ticks"]),float(row["minimum_pitch_margin"]))


def sensitive_dimensions(pilot, baseline, count=4):
    effects=[]
    for dimension in range(8):
        pair=[row for row in pilot if row.get("perturbed_dimension")==dimension]
        effects.append((max((score(row) for row in pair),default=score(baseline)),dimension))
    return [dimension for _,dimension in sorted(effects,reverse=True)[:count]]


def latin_hypercube(seed,samples,dimensions,bound):
    rng=np.random.default_rng(seed);values=np.empty((samples,dimensions),np.float32)
    for j in range(dimensions):
        values[:,j]=((rng.permutation(samples)+rng.random(samples))/samples*2-1)*bound
    return values


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--run",default=str(DEFAULT_RUN));args=parser.parse_args();root=Path(args.run)
    if root.exists():raise SystemExit(f"refusing overwrite {root}")
    prior=json.loads(PROBE.read_text());valid,failed,raw=verified_assets_allowing_runtime_gate_refresh()
    if prior["status"]!="FAIL" or prior["next"]!="bounded_pre_handoff_residual_probe":raise SystemExit("fixed handoff prerequisite mismatch")
    if not valid:raise SystemExit(f"frozen asset mismatch: {failed}; raw={raw}")
    gate=json.loads(Path("docs/RUNTIME_GATE.json").read_text())
    if gate.get("status")!="PASS" or gate.get("source_fingerprint")!=source_fingerprint(Path.cwd()):raise SystemExit("runtime gate stale")
    artifact=pickle.loads((EXPERT/"adapter.pkl").read_bytes());tube=SnapshotBank.load(TUBE)
    dparams,_,_=load_bundle(PI_D,verify_files=True);lparams,_,_=load_bundle(PI_L,verify_files=True)
    cfg=load_config("configs/backward_descent_rsi_pilot_v1.json",{"training_stage":"full","use_bank_resets":False,
        "stage_reachability_objective":"","expert_chain_termination":False,"domain_randomization":False,"obs_noise_enable":False})
    env=OrangeBikeDVGC(cfg,snapshot_bank=SnapshotBank(),cert_bank=SnapshotBank.load(C_L));step=jax.jit(env.step)
    d_infer=build_inference(env,dparams,deterministic=True);l_infer=build_inference(env,lparams,deterministic=True)
    adapter=compact_observation_command_adapter(jnp.asarray(artifact["prototypes"]),jnp.asarray(artifact["targets"]),
        jnp.asarray(artifact["normalizer_mean"]),jnp.asarray(artifact["normalizer_std"]),float(artifact["radius"]),float(artifact["core_radius"]))
    reference=pd.read_csv("data/reference_jump.csv");features=np.asarray([r["entry_feature"] for r in tube.records],float)
    center,scale=robust_normalization(features,cfg.descent_entry_scale_floors);window=(8,16);switch_tick=16

    def run(params):
        params=np.asarray(params,np.float32).reshape(2,4);state=env.reset(jax.random.PRNGKey(SEED));key=jax.random.PRNGKey(SEED+1);approach=0
        while int(state.info["phase"])<1 and not float(state.done):state=step(state,np.asarray([0.,1.,0.,0.],np.float32));approach+=1
        positive=float(state.data.qvel[2])>0;previous_vz=float(state.data.qvel[2]);apex=None;landing=False;stable=0;minimum=float("inf");pitch_margin=float("inf");closest=None
        for tick in range(220):
            if tick<switch_tick:
                action=_reference_action(env,cfg,state,reference,tick,3,105,10)
                if window[0]<=tick<window[1]:
                    segment=min((tick-window[0])*2//(window[1]-window[0]),1);action=np.clip(action+params[segment],-1,1)
            elif not int(state.info["had_valid_landing"]):
                key,sub=jax.random.split(key);base,_=d_infer(state.obs,sub);action=np.asarray(adapter(state.obs["state"][None],base[None])[0]);stable+=1
            else:
                key,sub=jax.random.split(key);action=np.asarray(l_infer(state.obs,sub)[0]);landing=True
            state=step(state,np.clip(action,-1,1));physical=np.asarray(env._physical_feature(state.data),float);feature=descent_entry_feature(physical,cfg)
            distance,index,_=normalized_nearest(feature,features,center,scale);minimum=min(minimum,distance)
            margin=np.deg2rad(float(cfg.max_pitch_deg))-abs(float(physical[4]));pitch_margin=min(pitch_margin,margin)
            if closest is None or distance<closest["distance"]:closest={"tick":tick+1,"distance":distance,"nearest_tube_index":index,"feature":feature.tolist()}
            vz=float(physical[8]);positive|=vz>0
            if apex is None and positive and previous_vz>0 and vz<=0 and not float(state.done):apex=tick+1
            previous_vz=vz
            if float(state.done):break
        return {"params":params.tolist(),"approach_ticks":approach,"apex_tick":apex,"early_failure":bool(float(state.done) and stable==0),
            "stable_descent_ticks":stable,"landing":landing or bool(int(state.info["had_valid_landing"])),
            "final_recovery":bool(int(state.info["recovery_success"])),"minimum_distance":minimum,"minimum_pitch_margin":pitch_margin,
            "closest":closest,"termination_reason":END_REASON.get(int(state.info["end_code"]),"unknown")}

    root.mkdir(parents=True);inputs={"prior_probe_sha256":file_sha256(PROBE),"tube_sha256":file_sha256(TUBE),
        "adapter_sha256":file_sha256(EXPERT/"adapter.pkl"),"policy_identity_hash":artifact["policy_identity_hash"],"xml":EXPECTED["xml"],"seed":SEED}
    save_json(root/"manifest.json",{"status":"FROZEN_BEFORE_OUTCOMES","inputs":inputs,"window_ticks":window,"switch_tick":switch_tick,
        "parameterization":"2 piecewise-constant segments x 4 bounded residual channels","pilot_epsilon":.08,"expansion_bound":.18,
        "pilot":"center + both signs for all eight dimensions","expansion":"64-point fixed LHS in four pilot-selected dimensions","PPO_authorization":False})
    save_json(root/"cost_estimate.json",{"estimated_seconds":1800,"pilot_rollouts":17,"maximum_expansion_rollouts":64,"horizon":220,"PPO_steps":0})
    zero=np.zeros((2,4),np.float32);baseline=run(zero);pilot=[]
    for dimension in range(8):
        for sign in (-1,1):
            params=zero.copy().reshape(-1);params[dimension]=sign*.08;row=run(params.reshape(2,4));row.update({"perturbed_dimension":dimension,"sign":sign});pilot.append(row)
    dims=sensitive_dimensions(pilot,baseline);best_pilot=max([baseline]+pilot,key=score);improved=score(best_pilot)>score(baseline)
    save_json(root/"finite_difference_pilot.json",{"baseline":baseline,"rows":pilot,"selected_dimensions":dims,"improved":improved})
    expansion=[]
    if improved:
        samples=latin_hypercube(SEED+100,64,len(dims),.18)
        for sample in samples:
            params=zero.copy().reshape(-1);params[dims]=sample;expansion.append(run(params.reshape(2,4)))
    candidates=[baseline]+pilot+expansion;top=max(candidates,key=score);replay_a=run(top["params"]);replay_b=run(top["params"]);exact=top==replay_a==replay_b
    save_json(root/"search_results.json",{"baseline":baseline,"pilot":pilot,"selected_dimensions":dims,"expansion":expansion,"top":top,"replay_a":replay_a,"replay_b":replay_b})
    progress=top["minimum_distance"]<baseline["minimum_distance"]-1e-3 or top["stable_descent_ticks"]>baseline["stable_descent_ticks"]
    report={"status":"PASS" if top["final_recovery"] and exact else ("PROGRESS" if progress and exact else "FAIL"),
        "head":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),"pilot_rollouts":17,"expansion_rollouts":len(expansion),
        "selected_dimensions":[{"segment":d//4,"channel":CHANNELS[d%4]} for d in dims],"baseline":baseline,"top":top,
        "distance_improvement":baseline["minimum_distance"]-top["minimum_distance"],"stable_tick_gain":top["stable_descent_ticks"]-baseline["stable_descent_ticks"],
        "exact_replay_twice":exact,"PPO_authorization":False,"next":"teacher_neighborhood_audit" if top["final_recovery"] else ("bounded_trust_region_round_2" if progress else "local_control_authority_blocker")}
    save_json(root/"NATURAL_COMPACT_DESCENT_LOCAL_SEARCH_V1_REPORT.json",report);save_json(root/"completed.json",{"status":report["status"],"next":report["next"]})
    print(json.dumps({k:report[k] for k in ("status","pilot_rollouts","expansion_rollouts","selected_dimensions","distance_improvement","stable_tick_gain","exact_replay_twice","top","next")},indent=2))


if __name__=="__main__":main()
