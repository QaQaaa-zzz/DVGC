"""Reusable native-contact scoring for recorded PhysX probes."""
import json
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from landing_evaluation import evaluate_landing
def summarize_probe(folder,criteria=None,output_folder=None):
 d=Path(folder);result=[]
 dest=Path(output_folder) if output_folder is not None else d
 dest.mkdir(parents=True,exist_ok=True)
 model=mujoco.MjModel.from_xml_path(str(Path(__file__).resolve().parent/'model/source.xml'));kin=mujoco.MjData(model)
 data=np.load(d/'trace.npz');cs=json.loads((d/'contacts.json').read_text());steps=json.loads((d/'steps.json').read_text());budget=json.loads((d/'budget.json').read_text());decl=json.loads((d/'declaration.json').read_text());audit=json.loads((d/'audit.json').read_text())
 prefix=cs.get('prefix_physics_frames',0);fs=cs['frames'][prefix:];tr=data['trace'];n=tr.shape[1];T=len(tr)
 for lane in range(n):
  wheel_floor=np.zeros((T,2));wheel_step=np.zeros((T,2));wheel_front=np.zeros((T,2));nonwheel=np.zeros(T);valid=np.zeros(T,bool)
  # Delivery-valid begins only once native solver has emitted finite positive impulse.
  certified=bool(cs.get('precalibrated_before_scoring',False) or (fs and fs[0].get('sensor_valid',False)))
  for t in range(T):
   seen=set()
   for f in fs[t*4:(t+1)*4]:
    for c in f['contacts']:
     if c['normal_impulse']>0:certified=True
     tag=f'/env_{lane}/';paths=[c['actor0'],c['actor1']]
     ours=next((x for x in paths if tag in x and '/Bike/' in x),None)
     if ours is None:continue
     other=paths[1] if ours==paths[0] else paths[0]
     key=(f['physics_tick'],tuple(sorted([c['collider0'],c['collider1']])),tuple(c['position']),tuple(c['impulse']))
     if key in seen:continue
     seen.add(key)
     wheel=0 if ours.endswith('/frontwheel') else 1 if ours.endswith('/rearwheel') else None
     force=c['normal_impulse']/.02
     if wheel is None:nonwheel[t]+=force
     elif other.endswith('/step') and abs(c['normal'][2])<=.7:wheel_front[t,wheel]+=force
     elif abs(c['normal'][2])>.7:
      if '/floor_' in other:wheel_floor[t,wheel]+=force
      elif other.endswith('/step'):wheel_step[t,wheel]+=force
   valid[t]=certified
  rs=np.array([s[lane]['reward_state'] for s in steps]);mass=sum(audit['body_masses'])
  np.savez_compressed(dest/f'landing_lane_{lane}.npz',time_s=(np.arange(T)+1)*.02,position=rs[:,:3],rpy=rs[:,3:6],velocity=rs[:,6:9],omega_world=rs[:,9:12],wheel_floor_force=wheel_floor,wheel_step_force=wheel_step,wheel_step_front_force=wheel_front,nonwheel_force=nonwheel,contact_valid=valid,mass=mass)
  evalrows=[]
  for ti in range(T):
   state=tr[ti,lane];kin.qpos[:3]=state[1:4];kin.qpos[3:7]=state[4:8]
   for j,name in enumerate(audit['dof_names']):kin.qpos[model.joint(name).qposadr[0]]=state[8+j]
   mujoco.mj_kinematics(model,kin);clear=[]
   for g in range(model.ngeom):
    if 'wheel_collision' not in model.geom(g).name:continue
    R=kin.geom_xmat[g].reshape(3,3);support=np.linalg.norm(R[2]*model.geom_size[g]);xyz=kin.geom_xpos[g];surface=.16 if xyz[0]>=3.6 else 0.;clear.append(xyz[2]-support-surface)
   rot=Rotation.from_quat(state[[5,6,7,4]]);w=rot.inv().apply(rs[ti,9:12]);roll,pitch=rs[ti,3:5]
   rr=w[0];pr=w[1]
   evalrows.append(dict(time_s=(ti+1)*.02,contact_valid=bool(valid[ti]),support_floor_N=float(wheel_floor[ti].sum()),support_step_N=float(wheel_step[ti].sum()),nonwheel_N=float(nonwheel[ti]),illegal_wheel_N=float(wheel_front[ti].sum()),min_wheel_clearance_m=float(min(clear)),x=float(rs[ti,0]),vx=float(rs[ti,6]),vy=float(rs[ti,7]),vz=float(rs[ti,8]),roll=float(roll),pitch=float(pitch),roll_rate=float(rr),pitch_rate=float(pr)))
  landing=evaluate_landing(evalrows,mass*9.81,criteria)
  (dest/f'landing_inputs_lane_{lane}.json').write_text(json.dumps(evalrows,indent=2))
  (dest/f'landing_lane_{lane}.json').write_text(json.dumps(landing,indent=2))
  result.append(dict(case=d.name,lane=lane,action_mode=('fixed_action_replay' if decl.get('action_replay') else 'matched_noise_replay' if decl.get('noise_replay') else 'deterministic' if decl['action_seed']<0 else 'stochastic'),action_seed=decl['action_seed'],reset=decl['reset_mode'] if 'reset_mode' in decl else 'hard',num_envs=n,controls=T,end_code=steps[-1][lane]['end_code'],apex=steps[-1][lane]['apex'],max_z=float(rs[:,2].max()),return_=steps[-1][lane]['return'],contact_valid_from_s=float((np.argmax(valid)+1)*.02) if valid.any() else None,wheel_step_contact=bool(wheel_step.max()>0),stable_landing=landing['strict_landing_1p0s'],takeoff_s=landing['takeoff_time_s'],touchdown_s=landing['first_touchdown_time_s'],stable_hold_s=landing['max_stable_hold_s'],actor_sha256=budget['actor_sha256'],actor_path=decl['actor'],origin_x=float(data['origins'][lane,0]),origin_y=float(data['origins'][lane,1]),contact_validity=landing['validity']))
 return result
