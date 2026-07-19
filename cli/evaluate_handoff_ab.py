"""Paired zero-training evaluation of one policy under canonical versus extended C_L."""
from __future__ import annotations
import argparse,json
from collections import Counter
from pathlib import Path
import jax
from dvgc.bank import SnapshotBank
from dvgc.certification import DYNAMICS_VARIANTS
from dvgc.composite import CanonicalEntryMatcher,composite_rollout
from dvgc.config import file_sha256,load_config
from dvgc.env import END_REASON,OrangeBikeDVGC
from dvgc.policy import load_bundle
from dvgc.rollout import restore_snapshot
from dvgc.runtime import build_inference,save_json

def summarize(rows):
 n=len(rows);count=Counter()
 for r in rows:
  count['chain']+=r['chain'];count['final']+=r['final'];count['handoff_missed']+=r['final'] and not r['chain'];count['false_progress']+=r['chain'] and not r['final'];count['timeout']+=r['truncated'];count['physical_failure']+=not r['final'] and not r['truncated']
 return {'branches':n,**{k:int(v) for k,v in count.items()},**{f'{k}_rate':v/n if n else 0. for k,v in count.items()}}

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--policy',required=True);p.add_argument('--landing-policy',required=True);p.add_argument('--candidate-bank',required=True);p.add_argument('--entry-bank',action='append',required=True,help='label=path');p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=2600000000);p.add_argument('--branches-per-state',type=int,default=8);p.add_argument('--limit',type=int,default=0);p.add_argument('--config',default='configs/default.json');a=p.parse_args();out=Path(a.output)
 if out.exists():raise SystemExit('Handoff A/B output exists')
 entries=dict(value.split('=',1) for value in a.entry_bank)
 if set(entries)!= {'canonical','extended'}:raise SystemExit('Exactly canonical and extended entry banks are required')
 params,cfg_dict,manifest=load_bundle(a.policy,verify_files=True);lp,_,lm=load_bundle(a.landing_policy,verify_files=True);source=SnapshotBank.load(a.candidate_bank);rows=source.records_for_phase('flight',include_training_only=False);rows=rows[:a.limit] if a.limit else rows
 cfg=load_config(a.config,{**cfg_dict,'training_stage':'flight','expert_chain_termination':False,'domain_randomization':False,'obs_noise_enable':False,'use_bank_resets':False})
 for label,path in entries.items():
  bank=SnapshotBank.load(path)
  if bank.metadata.get('xml_sha256')!=source.metadata.get('xml_sha256'):raise SystemExit(f'{label} XML mismatch')
 variants=[]
 canonical_bank=SnapshotBank.load(entries['canonical'])
 for spec in DYNAMICS_VARIANTS:
  vc=load_config(a.config,{**cfg.to_dict(),**{k:v for k,v in spec.items() if k!='id'}});env=OrangeBikeDVGC(vc,snapshot_bank=SnapshotBank(),cert_bank=canonical_bank);inf={'flight':build_inference(env,params,deterministic=True),'landing':build_inference(env,lp,deterministic=True)};matchers={label:CanonicalEntryMatcher(env,'flight',path) for label,path in entries.items()};variants.append((spec['id'],env,jax.jit(env.step),inf,matchers))
 results={label:[] for label in entries}
 for label in entries:
  for i,row in enumerate(rows):
   for b in range(a.branches_per_state):
    variant,env,step,inf,matchers=variants[b%len(variants)];seed=a.seed+i*10000+b;key=jax.random.PRNGKey(seed);_,outcome=composite_rollout(env,('flight','landing'),inf,{'flight':matchers[label]},restore_snapshot(env,row,key),key,horizon=int(cfg.branch_horizon),step_fn=step,action_noise_std=float(cfg.action_noise_std))
    results[label].append({'candidate_id':row['id'],'candidate_index':i,'parent_id':row.get('parent'),'layer':row.get('layer'),'branch_index':b,'branch_seed':seed,'dynamics_variant':variant,'chain':bool(outcome['chain']),'final':bool(outcome['final']),'terminated':bool(outcome['terminated']),'truncated':bool(outcome['truncated']),'termination_reason':END_REASON.get(outcome['end_code'],'unknown')})
   print(f'[handoff A/B {label}] {i+1}/{len(rows)}',flush=True)
 keyed={label:{(r['candidate_id'],r['branch_index']):r for r in values} for label,values in results.items()};transitions=Counter()
 for key,old in keyed['canonical'].items():
  new=keyed['extended'][key];transitions[f"chain{int(old['chain'])}final{int(old['final'])}->chain{int(new['chain'])}final{int(new['final'])}"]+=1
 identical_control=file_sha256(entries['canonical'])==file_sha256(entries['extended']);identical_pass=not identical_control or all(keyed['canonical'][k]==keyed['extended'][k] for k in keyed['canonical'])
 payload={'status':'PASS' if identical_pass else 'FAIL','artifact_role':'paired_zero_training_handoff_ab','policy_hash':file_sha256(Path(a.policy)/'params.pkl'),'landing_policy_hash':file_sha256(Path(a.landing_policy)/'params.pkl'),'candidate_bank_sha256':file_sha256(a.candidate_bank),'seed':a.seed,'branches_per_state':a.branches_per_state,'shared_physics_env':True,'identical_bank_control':identical_control,'identical_bank_control_pass':identical_pass,'entry_banks':{k:{'path':str(Path(v).resolve()),'sha256':file_sha256(v),'radius':SnapshotBank.load(v).metadata['entry_matcher']['radius']} for k,v in entries.items()},'summaries':{k:summarize(v) for k,v in results.items()},'paired_transitions':dict(transitions),'rows':results}
 save_json(out,payload);print(json.dumps({k:v for k,v in payload.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
