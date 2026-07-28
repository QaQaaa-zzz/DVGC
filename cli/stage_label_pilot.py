"""Small event-aligned next-stage label pilot (never a Tube certification)."""
from __future__ import annotations
import argparse,hashlib
from collections import Counter
from pathlib import Path
import jax
import jax.numpy as jp
import numpy as np
from dvgc.bank import SnapshotBank
from dvgc.config import ID_STAGE,file_sha256,load_config
from dvgc.env import END_REASON,OrangeBikeDVGC
from dvgc.rollout import restore_snapshot
from dvgc.runtime import build_inference,load_params,save_json
from dvgc.stage_reachability import CANONICAL_PHASE,evaluate_entry,protocol_payload,reachability_label

STAGE_SEED={"takeoff":9200000,"ascent":9300000,"apex":9400000,"descent":9500000,"landing":9600000}

NEXT_STAGE={"takeoff":"ascent","ascent":"apex","apex":"descent","descent":"landing","landing":"stable"}

def annotate_entry_snapshot(snapshot,row,stage,entry_id,controller_hash,seed,tick,quality):
 next_stage=NEXT_STAGE[stage]
 snapshot.update({"id":entry_id,"candidate_kind":"stage_entry_snapshot","entry_from_stage":stage,"entry_to_stage":next_stage,"trajectory_parent_id":row.get('trajectory_parent_id',row['id']),"upstream_candidate_id":row['id'],"controller_id":controller_hash,"rollout_seed":seed,"time_to_next_stage":tick,"entry_quality":quality})
 # Ascent, Apex and Descent share the Flight oracle phase.  Keep their local
 # stage semantic explicit so an event bank can be consumed directly by the
 # next stage without rewriting or guessing from physical state.
 if next_stage in ("ascent","apex","descent"):
  snapshot["flight_subinterval"]=next_stage
 return snapshot

def terminal_is_physical_failure(terminated,end_reason):
 return bool(terminated and end_reason not in ("recovery","chain_entry","next_stage_entry"))

def evenly(rows,count):
 if len(rows)<=count:return list(rows)
 return [rows[int(i)] for i in np.linspace(0,len(rows)-1,count,dtype=int)]

def sample_from_state(env,state,previous_vz):
 snap=env.snapshot_record(state,ID_STAGE[int(np.asarray(jax.device_get(state.info['phase'])))])
 end_code=int(np.asarray(jax.device_get(state.info['end_code'])));end_reason=END_REASON.get(end_code,f"unknown_{end_code}")
 snap.update({
  "canonical_phase":snap["source_phase"],"previous_vz":float(previous_vz),
  "apex_seen":bool(int(np.asarray(jax.device_get(state.info.get('apex_seen',0))))),
  # Use the environment's geometry-derived, velocity-synchronised event.
  # Root/CoM height alone can mislabel a grounded or wheelie reset as airborne.
  "dual_wheel_airborne":bool(float(np.asarray(jax.device_get(state.metrics["event/dual_wheel_liftoff"])))>.5),
  "first_valid_landing":bool(int(np.asarray(jax.device_get(state.info['had_valid_landing'])))),
  "support":bool(int(np.asarray(jax.device_get(state.info['contact_age'])))>0),
  "recovery_count":int(np.asarray(jax.device_get(state.info['recovery_count']))),
  "physical_failure":terminal_is_physical_failure(bool(int(np.asarray(jax.device_get(state.info['terminated'])))),end_reason),
  "prohibited_contact":end_reason=='prohibited_contact',"body_terrain_contact":end_reason=='prohibited_contact',
  "invalid_wheel_contact":end_reason=='invalid_wheel_step_contact',"deep_penetration":False,
  "nonfinite":not np.isfinite(snap['physical_feature']).all(),
 })
 return snap

def run_branch(env,step,inference,row,stage,seed,horizon,noise,support_metadata=None):
 key=jax.random.PRNGKey(seed);state=restore_snapshot(env,row,key);previous_vz=float(np.asarray(jax.device_get(state.data.qvel[2])));pending=None
 for tick in range(1,horizon+1):
  key,action_key,noise_key=jax.random.split(key,3);action,_=inference(state.obs,action_key)
  if noise:action=jp.clip(action+jax.random.normal(noise_key,action.shape)*noise,-1.,1.)
  state=step(state,action);sample=sample_from_state(env,state,previous_vz);entry=evaluate_entry(stage,sample,env._config,support_metadata)
  # Descent entry needs one fixed control tick without an immediate physical
  # failure; all other successor events are accepted on their aligned tick.
  if stage=="descent" and entry["valid"] and pending is None:pending=(tick,sample,entry)
  elif pending is not None:
   if not bool(np.asarray(jax.device_get(state.info['terminated']))):
    return True,pending[0],pending[1],pending[2],"next_stage_entry"
   pending=None
  elif entry["valid"]:
   return True,tick,sample,entry,"next_stage_entry"
  if float(np.asarray(jax.device_get(state.done)))>.5:
   code=int(np.asarray(jax.device_get(state.info['end_code'])));return False,None,None,entry,END_REASON.get(code,f"unknown_{code}")
  previous_vz=float(sample['physical_feature'][8])
 return False,None,None,{},"horizon_exhaustion"

def main():
 p=argparse.ArgumentParser();p.add_argument('--takeoff-bank',required=True);p.add_argument('--flight-bank',required=True);p.add_argument('--landing-bank',required=True);p.add_argument('--flight-policy',action='append',required=True);p.add_argument('--landing-policy',required=True);p.add_argument('--stage-support-bank',default='');p.add_argument('--output',required=True);p.add_argument('--entry-bank',required=True);p.add_argument('--states-per-stage',type=int,default=6);p.add_argument('--branches',type=int,default=4);p.add_argument('--horizon',type=int,default=200);p.add_argument('--action-noise',type=float,default=.03);p.add_argument('--config',default='configs/default.json');p.add_argument('--only-stage',choices=['takeoff','ascent','apex','descent','landing']);a=p.parse_args()
 banks={"takeoff":SnapshotBank.load(a.takeoff_bank),"flight":SnapshotBank.load(a.flight_bank),"landing":SnapshotBank.load(a.landing_bank)};selected={"takeoff":evenly(banks['takeoff'].records_for_phase('takeoff'),a.states_per_stage),"ascent":evenly([r for r in banks['flight'].records if r.get('flight_subinterval')=='ascent'],a.states_per_stage),"apex":evenly([r for r in banks['flight'].records if r.get('flight_subinterval')=='apex'],a.states_per_stage),"descent":evenly([r for r in banks['flight'].records if r.get('flight_subinterval')=='descent'],a.states_per_stage),"landing":evenly(banks['landing'].records_for_phase('landing'),a.states_per_stage)}
 if a.only_stage:selected={a.only_stage:selected[a.only_stage]}
 if any(len(v)!=a.states_per_stage for v in selected.values()):raise SystemExit({k:len(v) for k,v in selected.items()})
 envs={};entries=[];labels=[];terminal=Counter();protocol=None
 for stage,rows in selected.items():
  phase=CANONICAL_PHASE[stage];cfg=load_config(a.config,{"training_stage":phase,"use_bank_resets":False,"domain_randomization":False,"obs_noise_enable":False});support=SnapshotBank.load(a.stage_support_bank) if a.stage_support_bank and stage=='apex' else None;support_metadata=dict(support.metadata) if support else None
  if support_metadata is not None:support_metadata['support_features']=[r['physical_feature'] for r in support.records]
  protocol=protocol or protocol_payload(cfg,support_metadata);env=OrangeBikeDVGC(cfg,snapshot_bank=SnapshotBank(),stage_support_bank=support);step=jax.jit(env.step);policies=[a.landing_policy] if stage=='landing' else a.flight_policy;inferences=[build_inference(env,load_params(Path(policy)/'params.pkl'),deterministic=True) for policy in policies];envs[stage]=env
  for i,row in enumerate(rows):
   branch_records=[];successes=0
   for policy_index,(policy,inference) in enumerate(zip(policies,inferences)):
    for branch in range(a.branches):
     seed=STAGE_SEED[stage]+policy_index*1_000_000+i*100+branch;success,tick,snapshot,quality,reason=run_branch(env,step,inference,row,stage,seed,a.horizon,a.action_noise,support_metadata);successes+=int(success);terminal[f'{stage}:{reason}']+=1;entry_id=None;controller_hash=file_sha256(Path(policy)/'params.pkl')
     if snapshot is not None:
      entry_id=hashlib.sha256(f'{stage}:{row["id"]}:{seed}:{tick}'.encode()).hexdigest()[:32]
      annotate_entry_snapshot(snapshot,row,stage,entry_id,controller_hash,seed,tick,quality);entries.append(snapshot)
     branch_records.append({"branch_index":branch,"seed":seed,"controller_id":controller_hash,"success":success,"time_to_next_stage":tick,"entry_snapshot_id":entry_id,"entry_quality":quality,"failure_reason":None if success else reason})
   label=reachability_label(stage=stage,successes=successes,branches=len(branch_records),branch_records=branch_records,controller_bank_exhausted=True);label.update({"candidate_id":row['id'],"candidate_kind":row.get("candidate_kind"),"state_byte_hash":hashlib.sha256(b''.join(np.ascontiguousarray(np.asarray(row[k],np.float32)).tobytes() for k in ('qpos','qvel','ctrl','qacc_warmstart'))).hexdigest(),"controller_bank":[file_sha256(Path(policy)/'params.pkl') for policy in policies],"reference_index":row.get('reference_index'),"trajectory_parent":row.get('trajectory_parent_id',row.get('parent_candidate_id'))});labels.append(label)
 entry_bank=SnapshotBank(entries,{"artifact_role":"proposal_support_set_stage_entry_snapshots","protocol_sha256":protocol['protocol_sha256'],"not_certified_tube":True});entry_bank.save(a.entry_bank)
 summary={stage:dict(Counter(x['label'] for x in labels if x['stage']==stage)) for stage in selected};success_rates={stage:sum(x['s'] for x in labels if x['stage']==stage)/sum(x['n'] for x in labels if x['stage']==stage) for stage in selected}
 save_json(a.output,{"status":"PASS","artifact_role":"stage_candidate_label_pilot","labeler_version":3,"protocol_sha256":protocol['protocol_sha256'],"states_per_stage":a.states_per_stage,"branches_per_controller_state":a.branches,"horizon":a.horizon,"unique_states":len(labels),"total_rollouts":sum(x['n'] for x in labels),"labels":labels,"label_counts":summary,"success_rates":success_rates,"termination_reasons":dict(terminal),"entry_snapshots":len(entries),"entry_bank":str(a.entry_bank),"entry_bank_sha256":file_sha256(a.entry_bank),"proposal_support_only":True})
if __name__=='__main__':main()
