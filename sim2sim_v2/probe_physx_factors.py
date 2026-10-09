"""Bounded PhysX contact and action factor diagnostics; never launches training."""
import argparse,os,json,hashlib,csv
from pathlib import Path
os.environ['OMNI_KIT_ACCEPT_EULA']='YES';os.environ['JAX_PLATFORMS']='cpu';os.environ['OPENBLAS_NUM_THREADS']='1'
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--microcase',action='store_true');p.add_argument('--n',type=int,default=1);p.add_argument('--max-steps',type=int,default=100);p.add_argument('--actor',required=True);p.add_argument('--config',required=True);p.add_argument('--calibrate-before',action='store_true');p.add_argument('--env-seed',type=int,default=820707);p.add_argument('--noise-replay');p.add_argument('--action-replay');p.add_argument('--origin',nargs=2,type=float,default=[0.,0.]);p.add_argument('--initial-x',type=float);p.add_argument('--initial-z',type=float);p.add_argument('--reset-mode',choices=['hard','soft','after_episode'],default='hard');p.add_argument('--action-seed',type=int,default=-1);a=p.parse_args()
out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
def save(name,value):(out/name).write_text(json.dumps(value,indent=2,allow_nan=False))
save('declaration.json',dict(vars(a),max_control_interactions=a.n*(a.max_steps*(2 if a.reset_mode=='after_episode' else 1)+(10 if a.calibrate_before else 0)),max_physics_interactions=a.n*(a.max_steps*(2 if a.reset_mode=='after_episode' else 1)+(10 if a.calibrate_before else 0))*4,learner_transitions=0))
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'disable_viewport_updates':True,'limit_cpu_threads':16,'extra_args':['--/app/settings/loadUserConfig=false','--/app/settings/persistent=false']})
import numpy as np
from scipy.special import expit
from physx_task import PhysxTask
from policy_runtime import Actor
from contact_diagnostics import ContactRecorder,aggregate
spec=json.loads(Path(a.config).read_text());task=PhysxTask(spec,a.n,a.env_seed)
rec=ContactRecorder(task)
if any(a.origin):
 import omni.usd
 from pxr import Gf
 shift=np.array([*a.origin,0.],np.float32);task.origins[:]+=shift
 for i in range(a.n):
  attr=omni.usd.get_context().get_stage().GetPrimAtPath(f'/World/envs/env_{i}/step').GetAttribute('xformOp:translate');v=attr.Get();attr.Set(Gf.Vec3d(*(np.asarray(v)+shift)))
# All protocols initialize the solver after attaching report APIs.
task.hard_reset()
task.reset(np.arange(a.n),ground_only=True)
if a.initial_x is not None or a.initial_z is not None:
 pos,quat=task.view.get_world_poses()
 if a.initial_x is not None:pos[:,0]=task.origins[:,0]+a.initial_x
 if a.initial_z is not None:pos[:,2]=task.origins[:,2]+a.initial_z
 task.view.set_world_poses(pos,quat);task.world.physics_sim_view.update_articulations_kinematic()
actor=Actor(a.actor)
prefix_steps=0;prefix_infos=[]
if a.calibrate_before:
 for _ in range(10):task.step(actor(task.obs));prefix_steps+=1
 rec.certify();task.hard_reset();task.reset(np.arange(a.n),ground_only=True)
if a.reset_mode=='after_episode':
 save('prefix_initial_state.json',dict(world_position=task.view.get_world_poses()[0].tolist(),quaternion_wxyz=task.view.get_world_poses()[1].tolist(),root_com_velocity=task.view.get_velocities().tolist(),joint_q=task.view.get_joint_positions().tolist(),joint_qd=task.view.get_joint_velocities().tolist(),observation=task.obs.tolist(),origins=task.origins.tolist()))
 for _ in range(a.max_steps):
  _,_,done,pinfo=task.step(actor(task.obs));prefix_infos.append(pinfo);prefix_steps+=1
  if done.all():break
 save('prefix_infos.json',prefix_infos)
 if prefix_steps<=1:
  save('failure.json',dict(error='first_episode_ended_within_one_control',prefix_infos=prefix_infos,actual_control_interactions=prefix_steps*a.n,actual_physics_interactions=len(rec.frames)*a.n));rec.close();app.close();raise RuntimeError('Stop: initial episode <=1tick')
 task.reset(np.arange(a.n),ground_only=True)
 prefix_contact_frames=len(rec.frames)
else:prefix_contact_frames=len(rec.frames) if a.calibrate_before else 0
save('prefix_infos.json',prefix_infos)
save('initial_state.json',dict(world_position=task.view.get_world_poses()[0].tolist(),quaternion_wxyz=task.view.get_world_poses()[1].tolist(),root_com_velocity=task.view.get_velocities().tolist(),joint_q=task.view.get_joint_positions().tolist(),joint_qd=task.view.get_joint_velocities().tolist(),observation=task.obs.tolist(),origins=task.origins.tolist()))
rng=np.random.default_rng(a.action_seed if a.action_seed>=0 else 0)
noise_replay=np.load(a.noise_replay) if a.noise_replay else None
action_replay=np.load(a.action_replay) if a.action_replay else None
traces=[];observations=[];actions=[];infos=[];noises=[];pre=[];params=[];ctrlcontacts=[]
try:
 for k in range(a.max_steps):
  obs=task.obs.copy();observations.append(obs)
  z=(obs-actor.p['mean'])/actor.p['std']
  for j in range(4):
   z=z@actor.p[f'hidden_{j}_kernel']+actor.p[f'hidden_{j}_bias']
   if j<3:z=z*expit(z)
  loc=z[:,:4];scale=np.logaddexp(0,z[:,4:])+0.001
  noise=rng.standard_normal(loc.shape).astype(np.float32) if a.action_seed>=0 else np.zeros_like(loc)
  if noise_replay is not None:noise=noise_replay[k].reshape(loc.shape)
  latent=loc+scale*noise;act=np.tanh(latent)
  if action_replay is not None:act=action_replay[k].reshape(loc.shape)
  params.append(z.copy());pre.append(latent.copy());noises.append(noise.copy())
  start=len(rec.frames);_,_,done,info=task.step(act)
  ctrlcontacts.append(aggregate(rec.frames[start:],task.dt));traces.append(task.trace.copy());actions.append(act.copy());infos.append(info)
  if not a.microcase and done.all():break
 rec.certify()
 save('contacts.json',{'sensor_valid_after_known_contacts':rec.validated,'api_applied':rec.applied,'precalibrated_before_scoring':a.calibrate_before,'prefix_physics_frames':prefix_contact_frames,'frames':rec.frames,'control_aggregates':ctrlcontacts,'warning':'No callback without prior certification is unknown, not airborne. Certification only establishes report delivery, not all event semantics.'})
 np.savez_compressed(out/'trace.npz',trace=np.array(traces),observations=np.array(observations),actions=np.array(actions),noise=np.array(noises),pre_tanh=np.array(pre),distribution_logits=np.array(params),origins=task.origins)
 save('steps.json',infos);task.audit['origins']=task.origins.tolist();save('audit.json',task.audit)
 if hasattr(task,'resolved_config'):task.resolved_config.write(out)
 save('budget.json',{'control_interactions':(len(traces)+prefix_steps)*a.n,'prefix_control_interactions':prefix_steps*a.n,'physics_interactions':len(rec.frames)*a.n,'learner_transitions':0,'actor_sha256':hashlib.sha256(Path(a.actor).read_bytes()).hexdigest(),'sensor_validated':rec.validated})
except Exception as e:
 save('failure.json',{'error':repr(e),'actual_control_interactions':len(traces)*a.n,'actual_physics_interactions':len(rec.frames)*a.n,'contact_count':sum(len(f['contacts']) for f in rec.frames)})
 raise
finally:
 rec.close();app.close()
