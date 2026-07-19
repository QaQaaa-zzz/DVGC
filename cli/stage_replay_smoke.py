"""Deterministic short-window snapshot/policy replay gate for label acquisition."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import jax
import numpy as np

from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256, load_config
from dvgc.env import OrangeBikeDVGC
from dvgc.rollout import restore_snapshot
from dvgc.runtime import build_inference, load_params, save_json


def trace_once(env, step_fn, inference, row, seed: int, horizon: int):
    key=jax.random.PRNGKey(seed);state=restore_snapshot(env,row,key);trace=[]
    for index in range(horizon):
        key,action_key,_=jax.random.split(key,3);action,_=inference(state.obs,action_key);state=step_fn(state,action)
        trace.append({"step":index+1,"action":np.asarray(jax.device_get(action),np.float32),
                      "qpos":np.asarray(jax.device_get(state.data.qpos),np.float32),
                      "qvel":np.asarray(jax.device_get(state.data.qvel),np.float32)})
        if float(np.asarray(jax.device_get(state.done)))>.5:break
    return trace


def max_error(a,b,key):
    if len(a)!=len(b):return float("inf")
    return max((float(np.max(np.abs(x[key]-y[key]))) for x,y in zip(a,b)),default=0.)


def main():
    p=argparse.ArgumentParser();p.add_argument("--policy",required=True);p.add_argument("--candidate-bank",required=True);p.add_argument("--output",required=True);p.add_argument("--config",default="configs/default.json");p.add_argument("--seed",type=int,default=9100000);p.add_argument("--states",type=int,default=3);p.add_argument("--horizon",type=int,default=8);a=p.parse_args()
    out=Path(a.output);bank=SnapshotBank.load(a.candidate_bank);rows=bank.records[:a.states]
    if not rows:raise SystemExit("Replay smoke requires at least one snapshot")
    cfg=load_config(a.config,{"training_stage":rows[0]["source_phase"],"use_bank_resets":False});env=OrangeBikeDVGC(cfg,snapshot_bank=SnapshotBank());step=jax.jit(env.step);inference=build_inference(env,load_params(Path(a.policy)/"params.pkl"),deterministic=True)
    results=[]
    for i,row in enumerate(rows):
        seed=a.seed+i;t0=trace_once(env,step,inference,row,seed,a.horizon);t1=trace_once(env,step,inference,row,seed,a.horizon)
        errors={key:max_error(t0,t1,key) for key in ("action","qpos","qvel")};passed=len(t0)==len(t1) and all(value==0. for value in errors.values())
        h=hashlib.sha256()
        for point in t0:
            for key in ("action","qpos","qvel"):h.update(np.ascontiguousarray(point[key]).tobytes())
        results.append({"candidate_id":row["id"],"seed":seed,"steps":len(t0),"first_step_exact":passed and len(t0)>=1,"short_window_exact":passed,"max_abs_error":errors,"trace_sha256":h.hexdigest()})
    passed=all(row["short_window_exact"] for row in results)
    save_json(out,{"status":"PASS" if passed else "FAIL","artifact_role":"stage_label_deterministic_replay_smoke","policy_hash":file_sha256(Path(a.policy)/"params.pkl"),"candidate_bank_sha256":file_sha256(a.candidate_bank),"seed":a.seed,"states":len(rows),"horizon":a.horizon,"strict_bitwise_equality":True,"rows":results,"label_acquisition_allowed":passed})
    if not passed:raise SystemExit(40)

if __name__=="__main__":main()
