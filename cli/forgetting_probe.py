"""Phase-conditioned policy drift and deterministic Recovery probes."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import jax,numpy as np
from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256,load_config
from dvgc.curriculum import select_flight_reset_records
from dvgc.env import OrangeBikeDVGC
from dvgc.policy import load_bundle
from dvgc.rollout import frozen_rollout,restore_snapshot
from dvgc.runtime import build_inference,build_policy_distribution

def summary(values):
 x=np.asarray(values,np.float64)
 return {"mean":float(x.mean()),"p95":float(np.quantile(x,.95)),"max":float(x.max())}

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--reference-policy',required=True); p.add_argument('--policy',required=True)
 p.add_argument('--entry-bank',required=True); p.add_argument('--landing-tube',required=True); p.add_argument('--flight-bank',required=True)
 p.add_argument('--output',required=True); p.add_argument('--seed',type=int,default=6100000); a=p.parse_args()
 out=Path(a.output)
 if out.exists(): raise SystemExit(f'Output exists: {out}')
 ref_params,ref_cfg,ref_manifest=load_bundle(a.reference_policy,verify_files=True); params,pcfg,manifest=load_bundle(a.policy,verify_files=True)
 entry=SnapshotBank.load(a.entry_bank); landing=SnapshotBank.load(a.landing_tube); flight=SnapshotBank.load(a.flight_bank)
 groups={
  'canonical_entry':('landing',[r for r in entry.records if r['final']['label']=='safe' and not r.get('training_only',False)]),
  'landing_full_safe':('landing',[r for r in landing.records if r['final']['label']=='safe' and not r.get('training_only',False)]),
  'landing_boundary':('landing',[r for r in landing.records if r['final']['label']=='boundary' and not r.get('training_only',False)]),
  'flight_late_descent':('flight',select_flight_reset_records(flight.records_for_phase('flight',include_training_only=False),'late_descent')),
 }
 reports={}
 for gi,(name,(stage,rows)) in enumerate(groups.items()):
  cfg=load_config(overrides={**pcfg,'training_stage':stage,'domain_randomization':False,'obs_noise_enable':False,'use_bank_resets':False})
  cert=entry if stage=='flight' else SnapshotBank(); env=OrangeBikeDVGC(cfg,snapshot_bank=SnapshotBank(),cert_bank=cert)
  ref_dist=build_policy_distribution(env,ref_params); dist=build_policy_distribution(env,params)
  ref_infer=build_inference(env,ref_params,deterministic=True); infer=build_inference(env,params,deterministic=True); step=jax.jit(env.step)
  evidence=[]
  for i,row in enumerate(rows):
   seed=a.seed+gi*10000+i; key=jax.random.PRNGKey(seed); state=restore_snapshot(env,row,key)
   m0,s0,a0=ref_dist(state.obs); m1,s1,a1=dist(state.obs)
   kl=np.sum(np.log(np.asarray(s1)/np.asarray(s0))+(np.asarray(s0)**2+(np.asarray(m0)-np.asarray(m1))**2)/(2*np.asarray(s1)**2)-.5)
   l2=np.linalg.norm(np.asarray(a0)-np.asarray(a1))
   _,ro=frozen_rollout(env,ref_infer,restore_snapshot(env,row,key),key,horizon=int(cfg.branch_horizon),step_fn=step)
   _,co=frozen_rollout(env,infer,restore_snapshot(env,row,key),key,horizon=int(cfg.branch_horizon),step_fn=step)
   evidence.append({'id':row['id'],'kl_reference_to_policy':float(kl),'action_l2':float(l2),'reference_final':ro['final'],'policy_final':co['final'],'reference_end_code':ro['end_code'],'policy_end_code':co['end_code']})
  reports[name]={'states':len(rows),'kl':summary([r['kl_reference_to_policy'] for r in evidence]),'action_l2':summary([r['action_l2'] for r in evidence]),'reference_final_rate':float(np.mean([r['reference_final'] for r in evidence])),'policy_final_rate':float(np.mean([r['policy_final'] for r in evidence])),'rows':evidence}
 report={'status':'PASS','reference_policy_version':ref_manifest['policy_version'],'policy_version':manifest['policy_version'],'seed':a.seed,'groups':reports,'inputs':{name:{'path':str(Path(path).resolve()),'sha256':file_sha256(path)} for name,path in (('entry_bank',a.entry_bank),('landing_tube',a.landing_tube),('flight_bank',a.flight_bank))}}
 out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2)); print(json.dumps({**report,'groups':{k:{x:y for x,y in v.items() if x!='rows'} for k,v in reports.items()}},indent=2))
if __name__=='__main__': main()
