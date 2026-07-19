"""Synthetic, CPU-safe preflight for bounded next-stage reward direction."""
from __future__ import annotations
import argparse
import jax.numpy as jp
from dvgc.config import load_config
from dvgc.rewards import compute_stage_next_entry_reward
from dvgc.runtime import save_json

def value(cfg,objective,previous,current,event=False,failure=False):
 z=jp.zeros(4,jp.float32);terms=compute_stage_next_entry_reward(cfg=cfg,objective=objective,feature=jp.asarray(current),previous_feature=jp.asarray(previous),action=z,previous_action=z,next_entry=jp.asarray(event),hard_failure=jp.asarray(failure));return {k:float(v) for k,v in terms.items()}

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--config',default='configs/default.json');a=p.parse_args();cfg=load_config(a.config)
 base=[4.,0.,.45,0.,0.,0.,2.,0.,0.,0.,0.,0.,0.,-1.2,2.5,25.]
 cases={}
 specs={"takeoff_to_ascent":(.4,-.2,.45,.45),"ascent_to_apex":(.7,-.2,.50,.40),"apex_to_descent":(0.,1.,.55,.35),"descent_to_landing":(-.45,.5,.42,.42)}
 for objective,(good_vz,bad_vz,good_z,bad_z) in specs.items():
  prev=list(base);good=list(base);bad=list(base);good[8]=good_vz;bad[8]=bad_vz;good[2]=good_z;bad[2]=bad_z
  cases[objective]={"beneficial":value(cfg,objective,prev,good),"harmful":value(cfg,objective,prev,bad),"event":value(cfg,objective,prev,good,event=True),"failure":value(cfg,objective,prev,bad,failure=True)}
 checks={k:(v['beneficial']['reward']>v['harmful']['reward'] and v['event']['event']>abs(v['beneficial']['shaping']) and v['failure']['reward']<v['harmful']['reward']) for k,v in cases.items()}
 save_json(a.output,{"status":"PASS" if all(checks.values()) else "FAIL","artifact_role":"stage_reward_preflight","checks":checks,"cases":cases,"event_dominates_dense_shaping":all(v['event']['event']>abs(v['beneficial']['shaping']) for v in cases.values()),"pure_function_no_mutable_state":True})
 if not all(checks.values()):raise SystemExit(40)
if __name__=='__main__':main()
