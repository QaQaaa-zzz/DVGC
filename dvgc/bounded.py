"""Fixed-bank evaluation helpers for bounded shared-Actor repair."""
from __future__ import annotations
from collections import Counter
import jax,numpy as np
from .bank import SnapshotBank
from .config import load_config
from .env import END_REASON,OrangeBikeDVGC
from .rollout import frozen_rollout,restore_snapshot
from .runtime import build_inference,build_policy_distribution

def evaluate_records(params,cfg_dict,stage,records,downstream,seed):
 cfg=load_config(overrides={**cfg_dict,'training_stage':stage,'domain_randomization':False,'obs_noise_enable':False,'use_bank_resets':False})
 env=OrangeBikeDVGC(cfg,snapshot_bank=SnapshotBank(),cert_bank=downstream); infer=build_inference(env,params,deterministic=True); step=jax.jit(env.step); rows=[]
 for i,row in enumerate(records):
  key=jax.random.PRNGKey(seed+i); _,out=frozen_rollout(env,infer,restore_snapshot(env,row,key),key,horizon=int(cfg.branch_horizon),step_fn=step)
  rows.append({**out,'candidate_id':row['id'],'candidate_kind':row.get('candidate_kind','unknown'),'flight_subinterval':row.get('flight_subinterval'),'termination_reason':END_REASON.get(out['end_code'],'unknown')})
 def rates(values):
  return {'episodes':len(values),'chain_rate':float(np.mean([x['chain'] for x in values])),'final_recovery_rate':float(np.mean([x['final'] for x in values])),'physical_failure_rate':float(np.mean([x['terminated'] and not x['final'] for x in values])),'timeout_rate':float(np.mean([x['truncated'] for x in values]))}
 report=rates(rows); report['termination_reason_counts']=dict(Counter(x['termination_reason'] for x in rows)); report['rows']=rows
 report['subintervals']={name:rates([x for x in rows if x['flight_subinterval']==name]) for name in ('ascent','apex','descent') if any(x['flight_subinterval']==name for x in rows)}
 return report

def drift_probe(params,reference_params,cfg_dict,groups,entry_bank,seed):
 report={}
 for gi,(name,(stage,records)) in enumerate(groups.items()):
  cfg=load_config(overrides={**cfg_dict,'training_stage':stage,'domain_randomization':False,'obs_noise_enable':False,'use_bank_resets':False})
  downstream=entry_bank if stage=='flight' else SnapshotBank(); env=OrangeBikeDVGC(cfg,snapshot_bank=SnapshotBank(),cert_bank=downstream)
  ref_dist=build_policy_distribution(env,reference_params); policy_dist=build_policy_distribution(env,params); ref_infer=build_inference(env,reference_params,deterministic=True); infer=build_inference(env,params,deterministic=True); step=jax.jit(env.step)
  rows=[]
  for i,row in enumerate(records):
   key=jax.random.PRNGKey(seed+gi*10000+i); state=restore_snapshot(env,row,key); m0,s0,a0=ref_dist(state.obs); m1,s1,a1=policy_dist(state.obs)
   m0,s0,a0,m1,s1,a1=map(np.asarray,(m0,s0,a0,m1,s1,a1)); kl=float(np.sum(np.log(s1/s0)+(s0*s0+(m0-m1)**2)/(2*s1*s1)-.5)); l2=float(np.linalg.norm(a0-a1))
   _,r0=frozen_rollout(env,ref_infer,restore_snapshot(env,row,key),key,horizon=int(cfg.branch_horizon),step_fn=step); _,r1=frozen_rollout(env,infer,restore_snapshot(env,row,key),key,horizon=int(cfg.branch_horizon),step_fn=step)
   rows.append({'id':row['id'],'kl':kl,'action_l2':l2,'reference_final':r0['final'],'policy_final':r1['final']})
  def stats(key):
   x=np.asarray([r[key] for r in rows]); return {'mean':float(x.mean()),'p95':float(np.quantile(x,.95)),'max':float(x.max())}
  report[name]={'states':len(rows),'kl':stats('kl'),'action_l2':stats('action_l2'),'reference_final_rate':float(np.mean([r['reference_final'] for r in rows])),'policy_final_rate':float(np.mean([r['policy_final'] for r in rows])),'rows':rows}
 return report
