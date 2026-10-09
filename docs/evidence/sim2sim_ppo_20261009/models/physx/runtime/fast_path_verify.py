"""Verify frozen-policy traces and write the bounded execution audit manifest."""
import json,hashlib
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0,'/home/qy/ISAAC——SIM')
from fast_path_analysis import load
from fast_path_probe import dump
p=Path('/home/qy/ISAAC——SIM/results/sim2sim_fast_path_20260928');root=p.parent.parent
sha=hashlib.sha256((root/'policy/actor.npz').read_bytes()).hexdigest();aud=[];cost=[];takeoff=[]
sources=json.loads((p/'sources.json').read_text())
for f in sorted(p.glob('*/*/summary.json')):
 d=json.loads(f.read_text())
 if 'actor_sha256' not in d:continue
 assert d['actor_sha256']==sha
 t=load(f.parent);assert np.isfinite(t['observation']).all() and np.isfinite(t['qpos']).all() and np.isfinite(t['qvel']).all()
 np.testing.assert_allclose(t['time_s'],t['tick']*.02,atol=1e-7)
 assert not t['action_applied'][-1] and t['action_applied'][:-1].all()
 assert len(t['tick'])==d['ticks']+1
 assert bool(t['event'][-1,5])==d['task_success']
 aud.append({'summary':str(f),'actor_sha256':d['actor_sha256'],'ticks':d['ticks'],'finite_core':True})
 cost.append({'case':str(f.parent.relative_to(p)),**d['counts']})
for case,src in sources.items():
 for engine,folder in [('MJX-Warp',Path(src)),('PhysX final',p/'final_validation'/case)]:
  t=load(folder);clear=t['wheel_kinematics'][:,:,3];zone=t['event'][:,1]>0
  rising=np.flatnonzero((clear>.002).all(axis=1)&(t['qvel'][:,2]>.05)&zone)
  near=np.flatnonzero((clear<.002).any(axis=1)&~zone)
  details=json.loads((folder/'details.json').read_text())
  if engine=='PhysX final':contact=np.flatnonzero(np.linalg.norm(t['contact_net_force'],axis=2).max(axis=1)>1.)
  else:contact=np.array([i for i,d in enumerate(details) if len(d['contacts']['indices'])])
  takeoff.append({'case':case,'engine':engine,'initial_wheel_clearance_m':clear[0].tolist(),'first_logged_contact_s':float(t['time_s'][contact[0]]) if len(contact) else None,'first_geometric_near_contact_s':float(t['time_s'][near[0]]) if len(near) else None,'ascending_both_wheels_clear_after_zone_s':float(t['time_s'][rising[0]]) if len(rising) else None,'contact_epoch':'MJX derived stage; PhysX last1ms substep at each20ms tick, not first true collision time','criterion':'descriptive geometry check: both clearances>2mm, root vz>.05, original zone seen; unchanged apex success oracle'})
dump(p/'takeoff_audit.json',takeoff);dump(p/'actor_all_runs_audit.json',aud)
for f in sorted(p.glob('physx_actuator*/status.json')):
 d=json.loads(f.read_text());assert d['status']=='complete';cost.append({'case':str(f.parent.relative_to(p)),'rollout_physics_steps':d['rollout_physics_steps'],'initialization_physics_steps':d['initialization_physics_steps']})
for f in sorted(p.glob('mjx_actuator*/status.json')):
 d=json.loads(f.read_text());assert d['completed'];cost.append({'case':str(f.parent.relative_to(p)),'rollout_physics_steps':d['completed_physics_steps'],'initialization_physics_steps':0})
failed=load(p/'source_neighbors/vx_plus')
cost.append({'case':'source_neighbors/vx_plus serialization failure retained','rollout_physics_steps':int(failed['tick'][-1])*4,'initialization_physics_steps':0})
dump(p/'execution_costs.json',{'cases':cost,'recorded_rollout_physics_steps':sum(x['rollout_physics_steps'] for x in cost),'recorded_initialization_physics_steps':sum(x['initialization_physics_steps'] for x in cost),'unknown_initialization':'target_baseline contact-view construction failed before rollout; initialization count unavailable','failed_zero_rollout':['source_natural','target_baseline'],'training_transitions':0})
files=['fast_path_probe.py','fast_path_target.py','target_observation_adapter.py','fast_path_analysis.py','fast_path_report.py','fast_path_verify.py','fast_path_batch.py','physx_task.py','policy_runtime.py','usd_model.py','response_protocol.py','probe_mjx_response.py','probe_physx_response.py','analyze_response.py','configs/isaac_source_ppo_fresh.json','policy/actor.npz','policy/resolved_config.json','model/source.xml','/home/qy/DVGC/JIT/src/jit_dvgc/env.py','/home/qy/DVGC/JIT/src/jit_dvgc/semantics.py']
manifest={'files':{x:hashlib.sha256((root/x).read_bytes()).hexdigest() for x in files},'final_configuration':str(p/'final_config.json'),'final_declaration':str(p/'final_declaration.json'),'actor_sha256':sha,'original_training_source_modified':False,'source_engine':'original TwoPhaseBikeEnv MJX-Warp 3.6.0','target_engine':'Isaac Sim 5.1 real PhysX .001s; .02s control','scope':'frozen Actor engineering apex replication, no learning'}
dump(p/'execution_manifest.json',manifest)
print('audited closed loop',len(aud),'recorded physics',sum(x['rollout_physics_steps'] for x in cost),'initialization',sum(x['initialization_physics_steps'] for x in cost));print(json.dumps(takeoff[:2],indent=2))
