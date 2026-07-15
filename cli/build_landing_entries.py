"""Construct canonical Landing-entry proposals without Flight labels."""
from __future__ import annotations
import argparse, copy, hashlib, json
from pathlib import Path
import jax, numpy as np
from dvgc.bank import SnapshotBank
from dvgc.config import STAGE_ID, config_hash, file_sha256, load_config
from dvgc.entry import ENTRY_FEATURE_NAMES, entry_feature_from_physical, normalized_nearest, robust_normalization
from dvgc.env import OrangeBikeDVGC
from dvgc.policy import load_bundle
from dvgc.rollout import frozen_rollout, restore_snapshot
from dvgc.runtime import build_inference

def main():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument('--policy',required=True); p.add_argument('--source-bank',required=True); p.add_argument('--transition-source-bank',default=''); p.add_argument('--output-bank',required=True); p.add_argument('--config',default='configs/default.json'); p.add_argument('--seed',type=int,default=4100000); a=p.parse_args()
 out=Path(a.output_bank)
 if out.exists(): raise SystemExit(f'Output exists: {out}')
 params,pcfg,manifest=load_bundle(a.policy,verify_files=True); cfg=load_config(a.config,{**pcfg,'training_stage':'landing','domain_randomization':False,'obs_noise_enable':False,'use_bank_resets':False})
 if a.seed!=cfg.landing_entry_construction_seed: raise SystemExit('Construction seed mismatch')
 source=SnapshotBank.load(a.source_bank); rows=source.records_for_phase('landing',include_training_only=False); env=OrangeBikeDVGC(cfg,snapshot_bank=SnapshotBank()); infer=build_inference(env,params,deterministic=True); step=jax.jit(env.step)
 proposals=[]; rejected={}
 for i,row in enumerate(rows):
  if int(row.get('oracle_phase',-1))!=STAGE_ID['landing'] or not row.get('had_valid_landing') or not 1<=int(row.get('contact_age',0))<=cfg.landing_entry_window_steps:
   rejected['outside_entry_window']=rejected.get('outside_entry_window',0)+1; continue
  state=restore_snapshot(env,row,jax.random.PRNGKey(a.seed+i)); support=bool(np.asarray(jax.device_get(env._imu_support_estimate(state.data.qpos[:7],state.data.qvel[:6],state.info['had_valid_landing']))))
  feature=entry_feature_from_physical(row['physical_feature'],valid_landing=1,support=support,contact_age=row['contact_age'],cfg=cfg)
  _,result=frozen_rollout(env,infer,state,jax.random.PRNGKey(a.seed+i),horizon=cfg.branch_horizon,step_fn=step)
  if not result['final']: rejected['deterministic_no_final_recovery']=rejected.get('deterministic_no_final_recovery',0)+1; continue
  rec=copy.deepcopy(row); rec.update({'id':hashlib.sha256(f'landing-entry:{a.seed}:{row["id"]}'.encode()).hexdigest()[:32],'candidate_kind':'landing_entry','entry_feature':feature.astype(np.float32),'entry_source_id':row['id'],'entry_capture_mode':'existing_first_contact_window','entry_construction_seed':a.seed,'bootstrap_eligible':True}); proposals.append(rec)
 if a.transition_source_bank:
  transition=SnapshotBank.load(a.transition_source_bank)
  for i,row in enumerate(transition.records_for_phase('flight',include_training_only=False)):
   key=jax.random.PRNGKey(a.seed+100000+i); state=restore_snapshot(env,row,key); captured=None
   for _ in range(int(cfg.branch_horizon)):
    key,action_key=jax.random.split(key); action,_=infer(state.obs,action_key); state=step(state,action)
    if captured is None and bool(np.asarray(jax.device_get(state.metrics['event/landing']))):
     captured=env.snapshot_record(state,'landing')
    if bool(np.asarray(jax.device_get(state.done))): break
   if captured is None:
    rejected['transition_no_valid_landing']=rejected.get('transition_no_valid_landing',0)+1; continue
   if not bool(np.asarray(jax.device_get(state.info['recovery_success']))):
    rejected['transition_no_final_recovery']=rejected.get('transition_no_final_recovery',0)+1; continue
   feature=entry_feature_from_physical(captured['physical_feature'],valid_landing=1,support=True,contact_age=captured['landing_entry_age'],cfg=cfg)
   captured.update({'id':hashlib.sha256(f'landing-transition-entry:{a.seed}:{row["id"]}'.encode()).hexdigest()[:32],'candidate_kind':'landing_transition_entry','entry_feature':feature.astype(np.float32),'entry_source_id':row['id'],'entry_capture_mode':'first_valid_landing_event','entry_construction_seed':a.seed,'bootstrap_eligible':True})
   proposals.append(captured)
 center,scale=robust_normalization([r['entry_feature'] for r in proposals],cfg.landing_entry_scale_floors); accepted=[]; duplicates=0
 for row in proposals:
  if accepted and normalized_nearest(row['entry_feature'],[r['entry_feature'] for r in accepted],center,scale)[0]<cfg.landing_entry_dedup_distance_z: duplicates+=1; continue
  accepted.append(row)
 bank=SnapshotBank(accepted,{'entry_bank_role':'canonical_landing_entry_proposals','source_bank':str(Path(a.source_bank).resolve()),'source_bank_sha256':file_sha256(a.source_bank),'transition_source_bank':str(Path(a.transition_source_bank).resolve()) if a.transition_source_bank else None,'transition_source_bank_sha256':file_sha256(a.transition_source_bank) if a.transition_source_bank else None,'policy_version':manifest['policy_version'],'construction_seed':a.seed,'xml_sha256':file_sha256(cfg.xml_path),'action_mapping_version':cfg.action_mapping_version,'actor_history_steps':cfg.actor_history_steps,'candidate_config_hash':config_hash(cfg),'entry_feature_names':ENTRY_FEATURE_NAMES,'entry_center':center.tolist(),'entry_scale':scale.tolist(),'entry_scale_floors':list(cfg.landing_entry_scale_floors),'entry_window_steps':cfg.landing_entry_window_steps})
 bank.save(out); report={'status':'PASS' if accepted else 'FAIL','proposals':len(proposals),'accepted':len(accepted),'duplicates':duplicates,'rejections':rejected,'policy_version':manifest['policy_version'],'construction_seed':a.seed,'bank_sha256':file_sha256(out)}; out.with_suffix('.build.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
 if not accepted: raise SystemExit(2)
if __name__=='__main__': main()
