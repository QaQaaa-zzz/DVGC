"""Build a physically validated local Takeoff proposal pool from reference anchors."""
from __future__ import annotations
import argparse,copy,hashlib,json
from collections import Counter
from pathlib import Path
import jax,jax.numpy as jp,mujoco,numpy as np
from dvgc.bank import SnapshotBank
from dvgc.config import STAGE_ID,config_hash,file_sha256,load_config
from dvgc.env import END_REASON,OrangeBikeDVGC
from dvgc.flight_augmentation import interpolate_state,normalized_distance
from dvgc.runtime import save_json
from dvgc.reference_joints import stage_joint_state

FLOORS=np.asarray([.03,.01,.01,.01,.01,.01,.08,.03,.08,.08,.08,.08,.01,.02,.02,.5])
def finite(state):return all(np.isfinite(np.asarray(jax.device_get(x))).all() for x in (state.data.qpos,state.data.qvel,state.reward,state.obs['state']))
def body_contact(model,data,row):
 data.qpos[:]=row['qpos'];data.qvel[:]=row['qvel'];data.ctrl[:]=row['ctrl'];mujoco.mj_forward(model,data)
 terrain={g for g in range(model.ngeom) if int(model.geom_bodyid[g])==0};wheel_bodies={model.body(n).id for n in ('frontwheel','rearwheel')};bad=False;deep=False
 for i in range(data.ncon):
  c=data.contact[i];a,b=int(c.geom1),int(c.geom2);ar=int(model.geom_bodyid[a])!=0 and int(model.geom_bodyid[a]) not in wheel_bodies;br=int(model.geom_bodyid[b])!=0 and int(model.geom_bodyid[b]) not in wheel_bodies
  if (a in terrain and br) or (b in terrain and ar):bad=True;deep|=float(c.dist)<-.005
 return bad,deep
def main():
 p=argparse.ArgumentParser();p.add_argument('--anchors',required=True);p.add_argument('--output',required=True);p.add_argument('--report',required=True);p.add_argument('--target',type=int,default=120);p.add_argument('--proposals',type=int,default=3000);p.add_argument('--seed',type=int,default=9810000);p.add_argument('--config',default='configs/default.json');a=p.parse_args()
 src=SnapshotBank.load(a.anchors);anchors=sorted(src.records_for_phase('takeoff'),key=lambda r:int(r.get('reference_index',0)));cfg=load_config(a.config,{'training_stage':'takeoff','use_bank_resets':False,'domain_randomization':False,'obs_noise_enable':False});env=OrangeBikeDVGC(cfg,snapshot_bank=SnapshotBank());step=jax.jit(env.step);zero=jp.zeros(env.action_size,jp.float32);model=mujoco.MjModel.from_xml_path(cfg.xml_path);data=mujoco.MjData(model);rng=np.random.default_rng(a.seed);rows=copy.deepcopy(anchors);features=[np.asarray(r['physical_feature'],np.float64) for r in rows];scale=np.maximum(np.std(features,axis=0),FLOORS);reject=Counter();parents=Counter()
 key=stage_joint_state(model,None,'takeoff');hip_q=int(model.jnt_qposadr[model.joint('hip_joint').id]);knee_q=int(model.jnt_qposadr[model.joint('knee_joint').id]);hip_v=int(model.jnt_dofadr[model.joint('hip_joint').id]);knee_v=int(model.jnt_dofadr[model.joint('knee_joint').id])
 for row in anchors:
  if row.get('joint_state_source')!='xml_key:initial_state' or not np.allclose([row['qpos'][hip_q],row['qpos'][knee_q],row['qvel'][hip_v],row['qvel'][knee_v]],[key.hip,key.knee,key.hip_velocity,key.knee_velocity],atol=1e-6):raise SystemExit('Takeoff anchor violates XML key joint-state contract')
 for attempt in range(a.proposals):
  if len(rows)>=a.target:break
  i=int(rng.integers(0,len(anchors)-1));p0,p1=anchors[i],anchors[i+1];t=float(rng.uniform(.05,.95));q,v,c=interpolate_state(p0,p1,t)
  # Correlated segment perturbation only; no independent wide-dimensional mix.
  alpha=float(rng.normal(0,.04));q=q+alpha*(np.asarray(p1['qpos'])-np.asarray(p0['qpos']));q[3:7]/=np.linalg.norm(q[3:7]);v=v+alpha*(np.asarray(p1['qvel'])-np.asarray(p0['qvel']))
  state=env.reset_from_snapshot(jp.asarray(q,jp.float32),jp.asarray(v,jp.float32),jp.asarray(c,jp.float32),jax.random.PRNGKey(a.seed+attempt),jp.asarray(STAGE_ID['takeoff'],jp.int32),jp.asarray(0,jp.int32),jp.asarray(0,jp.int32),jp.asarray(0,jp.int32))
  if not finite(state):reject['nonfinite']+=1;continue
  snap=env.snapshot_record(state,'takeoff');bad,deep=body_contact(model,data,snap)
  if bad or deep:reject['body_or_deep_contact']+=1;continue
  feature=np.asarray(snap['physical_feature'],np.float64);distance=normalized_distance(feature,features,scale)
  if distance<.035:reject['normalized_duplicate']+=1;continue
  probe=state;failed=False
  for _ in range(5):
   probe=step(probe,zero)
   if not finite(probe):reject['short_nonfinite']+=1;failed=True;break
   if float(np.asarray(jax.device_get(probe.done)))>.5:
    reason=END_REASON.get(int(np.asarray(jax.device_get(probe.info['end_code']))),'unknown')
    if reason!='next_stage_entry':reject['short_'+reason]+=1;failed=True
    break
  if failed:continue
  parent=f"{p0['id']}|{p1['id']}";parents[parent]+=1;snap.update({'id':hashlib.sha256(f'{a.seed}:{attempt}:{parent}'.encode()).hexdigest()[:32],'candidate_kind':'takeoff_local_augmented','parent_anchor_pair':parent,'interpolation_fraction':t,'segment_perturbation':alpha,'generation_seed':a.seed,'proposal_index':attempt,'normalized_nearest_neighbor_distance':distance,'bootstrap_eligible':True,'training_only':False,'reference_index':int(round((1-t)*int(p0.get('reference_index',0))+t*int(p1.get('reference_index',0))))});rows.append(snap);features.append(feature)
 meta=dict(src.metadata);meta.update({'artifact_role':'takeoff_proposal_support_v1','certified_tube':False,'safe_claim_allowed':False,'source_anchor_bank':str(Path(a.anchors).resolve()),'source_anchor_bank_sha256':file_sha256(a.anchors),'generation_seed':a.seed,'xml_sha256':file_sha256(cfg.xml_path),'candidate_config_hash':config_hash(cfg),'normalized_dedup_distance':.035,'joint_state_contract':'takeoff XML key initial_state; root pose remains proposal-derived'})
 SnapshotBank(rows,meta).save(a.output);payload={'status':'PASS' if len(rows)>=a.target else 'FAIL','artifact_role':'takeoff_proposal_support_build','records':len(rows),'anchors':len(anchors),'augmented':len(rows)-len(anchors),'unique_parents':len(parents),'max_children_per_pair':max(parents.values(),default=0),'rejections':dict(reject),'bank':str(Path(a.output).resolve()),'bank_sha256':file_sha256(a.output),'proposal_support_only':True};save_json(a.report,payload);print(json.dumps(payload,indent=2));raise SystemExit(0 if payload['status']=='PASS' else 2)
if __name__=='__main__':main()
