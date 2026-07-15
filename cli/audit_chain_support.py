"""Audit first-contact support of successful Flight trajectories against C_L."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import jax,numpy as np
from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256,load_config
from dvgc.entry import ENTRY_FEATURE_NAMES
from dvgc.env import END_REASON,OrangeBikeDVGC
from dvgc.policy import load_bundle
from dvgc.rollout import frozen_rollout,restore_snapshot
from dvgc.runtime import build_inference

def evaluation_rows(report):
 rows=report.get('rows')
 if rows is None and isinstance(report.get('composite'),dict): rows=report['composite'].get('rows')
 if not isinstance(rows,list): raise ValueError('Evaluation report has no candidate rows')
 return rows

def main():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument('--policy',action='append',required=True,help='label=policy_dir')
 p.add_argument('--evaluation',action='append',required=True,help='label=fixed_evaluation.json')
 p.add_argument('--landing-policy',required=True); p.add_argument('--flight-bank',required=True); p.add_argument('--entry-bank',required=True)
 p.add_argument('--output',required=True); p.add_argument('--proposal-bank',required=True); p.add_argument('--seed',type=int,default=6200000); p.add_argument('--landing-seed',type=int,default=6300000); a=p.parse_args()
 out=Path(a.output); proposal_path=Path(a.proposal_bank)
 if out.exists() or proposal_path.exists(): raise SystemExit('Output already exists')
 specs=[]
 for value in a.policy:
  label,path=value.split('=',1); specs.append((label,path))
 evaluations={label:path for label,path in (value.split('=',1) for value in a.evaluation)}
 landing_params,landing_cfg,landing_manifest=load_bundle(a.landing_policy,verify_files=True)
 candidates=SnapshotBank.load(a.flight_bank); entry=SnapshotBank.load(a.entry_bank); rows=candidates.records_for_phase('flight',include_training_only=False)
 reports=[]; proposals=[]; global_index=0
 for pi,(label,path) in enumerate(specs):
  params,pcfg,manifest=load_bundle(path,verify_files=True); cfg=load_config(overrides={**pcfg,'training_stage':'flight','domain_randomization':False,'obs_noise_enable':False,'use_bank_resets':False})
  env=OrangeBikeDVGC(cfg,snapshot_bank=SnapshotBank(),cert_bank=entry); infer=build_inference(env,params,deterministic=True); step=jax.jit(env.step)
  safe=[r for r in entry.records if r['final']['label']=='safe' and not r.get('training_only',False)]
  safe_z=np.asarray(jax.device_get(env._safe_features)); center=np.asarray(jax.device_get(env._safe_center)); scale=np.asarray(jax.device_get(env._safe_scale))
  lcfg=load_config(overrides={**landing_cfg,'training_stage':'landing','domain_randomization':False,'obs_noise_enable':False,'use_bank_resets':False})
  lenv=OrangeBikeDVGC(lcfg,snapshot_bank=SnapshotBank()); linfer=build_inference(lenv,landing_params,deterministic=True); lstep=jax.jit(lenv.step)
  selected_ids={r['candidate_id'] for r in evaluation_rows(json.loads(Path(evaluations[label]).read_text())) if r.get('final',r.get('final_recovery',False))}
  selected=[row for row in rows if row['id'] in selected_ids]
  if len(selected)!=len(selected_ids): raise SystemExit(f'{label}: evaluation candidate IDs do not match bank')
  successes=[]
  for i,row in enumerate(selected):
   key=jax.random.PRNGKey(a.seed+pi*10000+i); state=restore_snapshot(env,row,key); captured=None; contact_step=None
   for t in range(int(cfg.branch_horizon)):
    key,ak=jax.random.split(key); action,_=infer(state.obs,ak); state=step(state,action)
    if captured is None and bool(np.asarray(jax.device_get(state.metrics['event/landing']))): captured=env.snapshot_record(state,'landing'); contact_step=t+1
    if bool(np.asarray(jax.device_get(state.done))): break
   final=bool(np.asarray(jax.device_get(state.info['recovery_success'])))
   if not final or captured is None:
    successes.append({'policy_label':label,'policy_version':manifest['policy_version'],'candidate_id':row['id'],'replay_final':final,'capture_missing':captured is None})
    global_index+=1; continue
   cstate=restore_snapshot(env,captured,jax.random.PRNGKey(a.seed+500000+global_index))
   feature=np.asarray(jax.device_get(env._landing_entry_feature(cstate.data,cstate.info['had_valid_landing']>0,cstate.info['contact_age']>0,cstate.info['landing_entry_age'])))
   z=(feature-center)/scale; delta=safe_z-z[None,:]; sq=delta*delta; distances=np.sqrt(sq.sum(axis=1)); nearest=int(np.argmin(distances)); distance=float(distances[nearest]); matched=distance<=float(env._safe_radius)
   lkey=jax.random.PRNGKey(a.landing_seed+global_index); _,landed=frozen_rollout(lenv,linfer,restore_snapshot(lenv,captured,lkey),lkey,horizon=int(lcfg.branch_horizon),step_fn=lstep)
   evidence={'policy_label':label,'policy_version':manifest['policy_version'],'candidate_id':row['id'],'candidate_kind':row.get('candidate_kind'),'flight_subinterval':row.get('flight_subinterval'),'first_valid_contact_step':contact_step,'matches_c_l':matched,'distance_to_c_l':distance,'nearest_entry_id':safe[nearest]['id'],'feature_names':ENTRY_FEATURE_NAMES,'squared_distance_contribution':sq[nearest].tolist(),'landing_policy_final':landed['final'],'landing_policy_termination':END_REASON.get(landed['end_code'],'unknown')}
   successes.append(evidence)
   if (not matched) and landed['final']:
    captured.update({'id':hashlib.sha256(f'entry-support:{a.seed}:{label}:{row["id"]}'.encode()).hexdigest()[:32],'candidate_kind':'landing_entry_support_proposal','entry_feature':feature.astype(np.float32),'entry_source_id':row['id'],'entry_source_policy':manifest['policy_version'],'entry_capture_mode':'successful_flight_first_valid_contact','entry_construction_seed':a.seed,'bootstrap_eligible':True})
    proposals.append(captured)
   global_index+=1
  valid=[x for x in successes if not x.get('capture_missing') and x.get('replay_final') is not False]
  reports.append({'policy_label':label,'policy_version':manifest['policy_version'],'expected_final_states':len(selected_ids),'final_contact_states':len(valid),'replay_failures':len(successes)-len(valid),'matched_c_l':sum(x['matches_c_l'] for x in valid),'landing_policy_recoverable':sum(x['landing_policy_final'] for x in valid),'recoverable_but_unmatched':sum(x['landing_policy_final'] and not x['matches_c_l'] for x in valid),'rows':successes})
 # Exact feature duplicates are rejected; physical certification remains mandatory.
 unique=[]
 for row in proposals:
  if any(np.linalg.norm((np.asarray(row['entry_feature'])-np.asarray(old['entry_feature']))/scale)<.15 for old in unique): continue
  unique.append(row)
 metadata={'entry_bank_role':'flight_success_contact_entry_proposals','construction_seed':a.seed,'landing_recovery_seed':a.landing_seed,'landing_policy_version':landing_manifest['policy_version'],'source_entry_bank':str(Path(a.entry_bank).resolve()),'source_entry_bank_sha256':file_sha256(a.entry_bank),'flight_bank_sha256':file_sha256(a.flight_bank),'xml_sha256':entry.metadata.get('xml_sha256'),'action_mapping_version':entry.metadata.get('action_mapping_version')}
 SnapshotBank(unique,metadata).save(proposal_path)
 report={'status':'PASS','policies':reports,'proposal_count_before_dedup':len(proposals),'proposal_count':len(unique),'proposal_bank':str(proposal_path.resolve()),'proposal_bank_sha256':file_sha256(proposal_path),'entry_bank_sha256':file_sha256(a.entry_bank),'matcher_radius':float(entry.metadata['entry_matcher']['radius']),'matcher_unchanged':True}
 out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2)); print(json.dumps({**report,'policies':[{k:v for k,v in x.items() if k!='rows'} for x in reports]},indent=2))
if __name__=='__main__': main()
