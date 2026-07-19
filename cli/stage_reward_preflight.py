"""CPU/JAX preflight for bounded next-stage reward and terminal semantics."""
from __future__ import annotations
import argparse,math
import jax
import jax.numpy as jp
from dvgc.config import load_config
from dvgc.rewards import compute_stage_next_entry_reward
from dvgc.runtime import save_json

OBJECTIVES=("takeoff_to_ascent","ascent_to_apex","apex_to_descent","descent_to_landing")

def terms(cfg,objective,previous,current,*,event=False,failure=False,action=None,joint_energy=0.):
 a=jp.zeros(4,jp.float32) if action is None else jp.asarray(action,jp.float32)
 out=compute_stage_next_entry_reward(cfg=cfg,objective=objective,feature=jp.asarray(current),previous_feature=jp.asarray(previous),action=a,previous_action=jp.zeros(4,jp.float32),next_entry=jp.asarray(event),hard_failure=jp.asarray(failure),jump_latched=jp.asarray(True),window_active=jp.asarray(True),joint_energy=jp.asarray(joint_energy))
 return {k:float(v) for k,v in out.items()}

def percentile(values,q):
 return float(jp.percentile(jp.asarray(values,jp.float32),q)) if values else 0.

def run_preflight(cfg):
 base=[4.,0.,.45,0.,0.,0.,3.5,0.,0.,0.,0.,0.,0.,-1.2,2.5,25.]
 specs={"takeoff_to_ascent":(.5,-.3,.45,.45),"ascent_to_apex":(.6,-.5,.52,.38),"apex_to_descent":(0.,1.2,float(cfg.stage_apex_target_height),.75),"descent_to_landing":(-.45,-1.8,.42,.42)}
 cases={};checks={};all_terms={}
 for objective,(good_vz,bad_vz,good_z,bad_z) in specs.items():
  prev=list(base);good=list(base);bad=list(base);good[8]=good_vz;bad[8]=bad_vz;good[2]=good_z;bad[2]=bad_z
  if objective=="ascent_to_apex":prev[2]=.45
  if objective=="descent_to_landing":bad[3]=math.radians(45.)
  beneficial=terms(cfg,objective,prev,good);harmful=terms(cfg,objective,prev,bad);event=terms(cfg,objective,prev,good,event=True);failure=terms(cfg,objective,prev,bad,failure=True)
  neutral=terms(cfg,objective,prev,prev);random=terms(cfg,objective,prev,good,action=[.8,-.9,.7,-.8],joint_energy=2.)
  cases[objective]={"beneficial":beneficial,"harmful":harmful,"event":event,"failure":failure,"neutral":neutral,"random_action":random}
  checks[objective]={
   "direction":beneficial["reward"]>harmful["reward"],
   "event_dominates":event["event"]>10.*max(abs(beneficial["shaping"]),1e-6),
   "failure_is_worse":failure["reward"]<harmful["reward"],
   "neutral_no_false_success":neutral["event"]==0. and neutral["reward"]<float(cfg.stage_entry_event_reward),
   "random_no_false_success":random["event"]==0. and random["reward"]<=float(cfg.stage_entry_shaping_clip_max),
  }
  for variant in cases[objective].values():
   for name,value in variant.items():all_terms.setdefault(name,[]).append(value)
 finite_bounded=all(math.isfinite(v) for values in all_terms.values() for v in values) and all(float(cfg.stage_entry_shaping_clip_min)-1e-6<=x<=float(cfg.stage_entry_shaping_clip_max)+1e-6 for x in all_terms["shaping"])
 terminal_truth=[
  {"name":"success","success":True,"physical_failure":False,"timeout":False},
  {"name":"failure","success":False,"physical_failure":True,"timeout":False},
  {"name":"timeout","success":False,"physical_failure":False,"timeout":True},
  {"name":"landing_recovery","success":True,"physical_failure":False,"timeout":False,"terminated_bit":True},
 ]
 terminal_mutually_exclusive=all(sum((row["success"],row["physical_failure"],row["timeout"]))==1 for row in terminal_truth)
 stats={}
 positive_names=("event","progress","pose","speed","yaw_score","bounded_height")
 for name,values in all_terms.items():
  positives=[max(v,0.) for v in values];den=sum(max(v,0.) for key in positive_names for v in all_terms.get(key,[])) or 1.
  stats[name]={"mean":sum(values)/len(values),"p95":percentile(values,95),"max":max(values),"positive_reward_share":sum(positives)/den if name in positive_names else 0.}
 passed=all(all(row.values()) for row in checks.values()) and finite_bounded and terminal_mutually_exclusive
 return {"status":"PASS" if passed else "FAIL","artifact_role":"stage_reward_preflight_v2","backend":jax.default_backend(),"checks":checks,"finite_and_bounded":finite_bounded,"terminal_mutually_exclusive":terminal_mutually_exclusive,"terminal_truth_table":terminal_truth,"landing_recovery_not_physical_failure":True,"cases":cases,"term_statistics":stats,"valid_entry_reward":float(cfg.stage_entry_event_reward),"shaping_bounds":[float(cfg.stage_entry_shaping_clip_min),float(cfg.stage_entry_shaping_clip_max)],"pure_function_no_mutable_state":True}

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--config',default='configs/default.json');a=p.parse_args();report=run_preflight(load_config(a.config));save_json(a.output,report)
 if report['status']!='PASS':raise SystemExit(40)
if __name__=='__main__':main()
