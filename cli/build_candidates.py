"""Build a clean event-anchored candidate bank from the supplied trajectory envelopes."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import jax
import jax.numpy as jp
import numpy as np
import pandas as pd
from dvgc.bank import SnapshotBank
from dvgc.config import STAGE_ID, load_config
from dvgc.env import OrangeBikeDVGC

ANCHORS={"approach":(0,113),"takeoff":(113,129),"flight":(129,329),"landing":(300,350)}


def _seed(row, phase, cfg, training_only):
    euler=np.deg2rad([row.roll_angle,row.pitch_angle,row.yaw_angle]).astype(np.float32)
    common={"euler":euler,"steer":0.0,"hip":float(np.clip(row.hip_position,cfg.hip_min,cfg.hip_max)),"knee":float(np.clip(row.knee_position,cfg.knee_min,cfg.knee_max)),"linear_velocity":np.asarray([row.vel_x,row.vel_y,row.vel_z],np.float32),"angular_velocity":np.zeros(3,np.float32)}
    if phase in ("flight","landing"):
        common.update(seed_type="system_com",desired_com=np.asarray([row.pos_x,row.pos_y,row.pos_z],np.float32))
    else:
        vz=float(np.random.uniform(0.35,0.85)) if (phase=="takeoff" and training_only) else float(np.clip(row.vel_z,-.08,.08))
        common["linear_velocity"]=np.asarray([row.vel_x,row.vel_y,vz],np.float32)
        common.update(seed_type="ground",base_pos=np.asarray([row.pos_x,row.pos_y,cfg.nominal_base_z_ground+0.03],np.float32))
    return common


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase",required=True,choices=["landing","flight","takeoff","approach"])
    p.add_argument("--target",type=int,default=128)
    p.add_argument("--bank",required=True)
    p.add_argument("--reference",default="data/reference_jump.csv")
    p.add_argument("--config",default="configs/default.json")
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--aux-fraction",type=float,default=.20)
    a=p.parse_args(); rng=np.random.default_rng(a.seed)
    cfg=load_config(a.config,{"training_stage":a.phase,"use_bank_resets":False,"domain_randomization":False,"obs_noise_enable":False})
    env=OrangeBikeDVGC(cfg,snapshot_bank=SnapshotBank()); bank=SnapshotBank.load(a.bank)
    df=pd.read_csv(a.reference); lo,hi=ANCHORS[a.phase]
    attempts=0
    while len(bank.records_for_phase(a.phase))<a.target and attempts<a.target*30:
        attempts+=1; idx=int(rng.integers(lo,hi+1)); row=df.iloc[idx]
        training_only=(a.phase=="takeoff" and rng.random()<a.aux_fraction)
        seed=_seed(row,a.phase,cfg,training_only)
        state=env.reset_from_com_seed(seed,jax.random.PRNGKey(a.seed+attempts))
        # Convert the grounded proposal to the requested semantic phase.
        if a.phase=="approach":
            state=env.reset_from_snapshot(state.data.qpos,state.data.qvel,state.data.ctrl,jax.random.PRNGKey(a.seed+100000+attempts),jp.asarray(STAGE_ID["approach"],jp.int32),jp.asarray(0,jp.int32),jp.asarray(0,jp.int32),jp.asarray(0,jp.int32))
        if a.phase=="landing":
            step=jax.jit(env.step); zero=jp.zeros(env.action_size,jp.float32)
            for _ in range(80):
                state=step(state,zero)
                if int(np.asarray(jax.device_get(state.info["phase"])))==STAGE_ID["landing"]:
                    break
                if float(np.asarray(jax.device_get(state.done)))>.5: break
        rec=env.snapshot_record(state,a.phase)
        rec.update({"training_only":training_only,"bootstrap_eligible":True,"candidate_kind":"velocity_seed" if training_only else "reference_envelope","reference_index":idx})
        # Certification candidates must respect phase semantics.
        if a.phase=="takeoff":
            rec["had_airborne"]=0; rec["airborne_count"]=0; rec["policy_state"]["filter_phase"]=STAGE_ID["takeoff"]
        bank.add(rec,deduplicate=True,distance=.06)
    bank.metadata.update({"reference":"reference_jump.csv","reference_usage":"candidate envelopes only; never reward tracking"})
    bank.save(a.bank)
    report={"phase":a.phase,"target":a.target,"attempts":attempts,"summary":bank.summary()}
    Path(a.bank).with_suffix(".build.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))
if __name__=="__main__": main()
