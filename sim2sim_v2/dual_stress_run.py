"""Paired hard1000 pulses: frozen Actor/explorer, native contact, no training."""
import argparse,os,json,sys,time,dataclasses
from pathlib import Path
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',XLA_PYTHON_CLIENT_PREALLOCATE='false',OMNI_KIT_ACCEPT_EULA='YES')
P=argparse.ArgumentParser();P.add_argument('--engine',choices=['mjx','physx'],required=True);P.add_argument('--output',required=True);P.add_argument('--batch',type=int,default=50);P.add_argument('--limit',type=int,default=1000);P.add_argument('--horizon',type=int,default=400);a=P.parse_args()
if a.limit%a.batch:raise ValueError('Fixed batch must divide case count')
ROOT=Path(__file__).resolve().parent;OUT=Path(a.output).resolve();OUT.mkdir(exist_ok=False,parents=True);RUN=ROOT/'runs/dual_stress1000_20261013';D=json.loads((RUN/'declaration.json').read_text())
if a.engine=='physx':
 os.environ['JAX_PLATFORMS']='cpu'
 from isaacsim import SimulationApp
 app=SimulationApp({'headless':True,'disable_viewport_updates':True,'limit_cpu_threads':16,'extra_args':['--/app/settings/loadUserConfig=false','--/app/settings/persistent=false']})
import logging
logging.getLogger('jax').setLevel(logging.WARNING)
import numpy as np,jax,jax.numpy as jp,mujoco
jax.config.update('jax_compilation_cache_dir','/home/qy/.cache/jax')
from scipy.spatial.transform import Rotation
from dual_stress_protocol import load_protocol,explorer_observation,compose,write_json,infer
from policy_runtime import Actor
from landing_evaluation import LandingCriteria,evaluate_landing
actor=Actor(D['actor']);spec,ex,index=load_protocol();infer_jit=jax.jit(lambda x:infer(ex,x)[:2])
criteria=LandingCriteria(schema='dual_stress_no_min_forward_20261013',min_forward_speed=None)
noise=np.load(RUN/'explorer_noise.npy')
start=time.monotonic();charged=0;scored=0;results=[]
def status(state,**kwargs):write_json(OUT/'status.json',dict(state=state,engine=a.engine,completed_cases=len(results),declared_cases=a.limit,charged_controls=charged,scored_controls=scored,physics_steps=charged*4,elapsed_s=time.monotonic()-start,training_updates=0,**kwargs))
if a.engine=='mjx':
 SRC=Path('/home/qy/DVGC/JIT/runs/code_snapshots/jump_ori_sim2sim_vxgt02_20260928/JIT/src');sys.path.insert(0,str(SRC))
 from jit_dvgc.config import load_config
 from jit_dvgc.env import TwoPhaseBikeEnv
 from mujoco import mjx
 from mujoco_playground._src import mjx_env
 from mujoco.mjx.warp.types import DATA_NON_VMAP
 cfg=load_config('/home/qy/DVGC/JIT/runs/phase_u/phase_u_jump_ori_sim2sim_vxgt02_10002432_seed820707_20260928/resolved_config.json',runtime_only=True);cfg=dataclasses.replace(cfg,sim2sim_randomization=None,sim2sim_low_speed_failure=None);env=TwoPhaseBikeEnv(cfg);m=env.mj_model
 def reset_one(q,v,key):
  orig=mjx_env.make_data
  def make(*args,**kwargs):kwargs.update(qpos=q,qvel=v);return orig(*args,**kwargs)
  mjx_env.make_data=make
  try:
   s=env.reset_natural(key)
   from jit_dvgc.semantics import initial_event_state
   from jit_dvgc.observation import actor_observation,privileged_observation
   ev=initial_event_state(q[0],cfg);ob=actor_observation(s.info['history'],ev.jump_signal);geom=env._observable_geometry(s.data,__import__('jit_dvgc.geometry',fromlist=['extract_geometry']).extract_geometry(s.data,env._geometry))
   return s.replace(obs={'state':ob,'privileged_state':privileged_observation(s.data,ob,geom)},info={**s.info,'events':ev})
  finally:mjx_env.make_data=orig
 reset=jax.jit(jax.vmap(reset_one))
 fields=['nacon','contact__geom','contact__worldid','contact__efc_address','contact__pos','contact__frame','efc__force']
 def one_step(s,act):
  captured={};orig=mjx_env.step
  def step4(model,data,ctrl,n_substeps):
   def one(d,_):
    d=mjx.step(model,d.replace(ctrl=ctrl));return d,{k:getattr(d._impl,k) for k in fields}
   d,c=jax.lax.scan(one,data,None,length=n_substeps);captured['c']=c;return d
  mjx_env.step=step4
  try:ns=env.step(s,act)
  finally:mjx_env.step=orig
  return ns,captured['c']
 axes={k:(0 if k=='efc__force' else None) for k in fields}
 step=jax.jit(jax.vmap(one_step,out_axes=(0,axes)))
else:
 from physx_task import PhysxTask
 from contact_diagnostics import ContactRecorder
 target=json.loads(Path('/home/qy/ISAAC——SIM/runs/experiments/physx_jump_ori_from9904128_kd5_5ms_10m_20260928/config.json').read_text())
 env=PhysxTask(target,a.batch,820707);m=env.m;rec=ContactRecorder(env);env.hard_reset();env.reset(np.arange(a.batch),ground_only=True)
 for _ in range(10):env.step(actor(env.obs));charged+=a.batch
 rec.certify();rec.frames.clear()
qa=[m.joint(n).qposadr[0] for n in (env.names if a.engine=='physx' else [])];va=[m.joint(n).dofadr[0] for n in (env.names if a.engine=='physx' else [])]
kin=mujoco.MjData(m);weight=float(m.body_mass.sum()*9.81)
wheel_ids=[g for g in range(m.ngeom) if 'wheel_collision' in m.geom(g).name]
write_json(OUT/'runtime.json',dict(engine=a.engine,mujoco=mujoco.__version__,jax=jax.__version__,batch=a.batch,dt=.005,control_dt=.02,criteria=dataclasses.asdict(criteria),weight_N=weight))
try:
 for begin in range(0,a.limit,a.batch):
  size=min(a.batch,a.limit-begin)
  if size!=a.batch:raise ValueError('Use divisor of episode count for constant shape')
  qbank=[];vbank=[]
  for case in range(begin,begin+size):
   group=case//250;lane=case%250;b=np.load(RUN/f'initial_onset_{D["onsets"][group]:02d}.npz');qbank.append(b['qpos'][lane]);vbank.append(b['qvel'][lane])
  qbank=np.array(qbank);vbank=np.array(vbank);obslog=[];qlog=[];vlog=[];actlog=[];reqlog=[];efflog=[];contactlog=[];endlog=[];alive=np.ones(size,bool);masks=[]
  if a.engine=='mjx':
   state=reset(jp.asarray(qbank),jp.asarray(vbank),jax.random.split(jax.random.PRNGKey(begin+10102026),size));jax.block_until_ready(state)
   scratch=mjx.make_data(m,impl='warp',naconmax=max(4096,size*64),naccdmax=max(320,size*64),njmax=256)
   state=state.replace(data=state.data.replace(_impl=state.data._impl.replace(**{n:getattr(scratch._impl,n) for n in DATA_NON_VMAP})))
   q=np.asarray(state.data.qpos);v=np.asarray(state.data.qvel)
  else:
   env.hard_reset();env.reset(np.arange(size),ground_only=True);ids=np.arange(size,dtype=np.int32);q=qbank.copy();v=vbank.copy();rot=Rotation.from_quat(q[:,[4,5,6,3]]);omega=rot.apply(v[:,3:6]);comvel=v[:,:3]+np.cross(omega,rot.apply(np.tile(m.body_ipos[1],(size,1))))
   env.view.set_world_poses((q[:,:3]+env.origins).astype(np.float32),q[:,3:7].astype(np.float32));env.view.set_velocities(np.c_[comvel,omega].astype(np.float32));env.view.set_joint_positions(q[:,qa].astype(np.float32));env.view.set_joint_velocities(v[:,va].astype(np.float32));env.world.physics_sim_view.update_articulations_kinematic()
   from jit_dvgc import semantics
   for lane in range(size):
    env.initial_x[lane]=q[lane,0];env.events[lane]=semantics.initial_event_state(np.asarray(q[lane,0]),env.cfg);ob=env.observers[lane];env.obs[lane]=ob.initial(q[lane,0]);ob.d.qpos[:]=q[lane];mujoco.mj_kinematics(m,ob.d);env.privileged_obs[lane]=ob.privileged(env.obs[lane],v[lane,:3],omega[lane],v[lane,va],env.names);env.prev_vel[lane]=v[lane,:3]
   env.previous_wheel_q=q[:,qa][:,env.wheels].copy();env.continuous_wheel_q=env.previous_wheel_q.copy();rec.frames.clear()
  np.testing.assert_allclose(q,qbank,atol=1e-6,rtol=0);np.testing.assert_allclose(v,vbank,atol=1e-6,rtol=0)
  for tick in range(a.horizon):
   obs=np.asarray(state.obs['state']) if a.engine=='mjx' else env.obs.copy();priv=np.asarray(state.obs['privileged_state']) if a.engine=='mjx' else env.privileged_obs.copy()
   base=actor(obs);action=base.copy();requested=np.zeros_like(base);effective=np.zeros_like(base)
   activepulse=np.array([D['onsets'][case//250]<=tick<D['onsets'][case//250]+3 for case in range(begin,begin+size)])&alive
   if activepulse.any():
    # Same frozen existing inference and original neighborhood context.
    ei=explorer_observation(priv[activepulse],q[activepulse],v[activepulse],index);mu,sd=map(np.asarray,infer_jit(jp.asarray(ei)));latent=mu+sd*noise[tick,begin:begin+size][activepulse];action[activepulse],requested[activepulse],effective[activepulse]=compose(base[activepulse],latent)
   mask=alive.copy();masks.append(mask);obslog.append(obs);actlog.append(action);reqlog.append(requested);efflog.append(effective);force=np.zeros((size,4),float)
   if a.engine=='mjx':
    state,cs=step(state,jp.asarray(action));jax.block_until_ready(state);cs=jax.device_get(cs);q=np.asarray(state.data.qpos);v=np.asarray(state.data.qvel);ends=np.asarray(state.info['end_code']);done=np.asarray(state.done,dtype=bool)
    for k in range(4):
     count=int(np.asarray(cs['nacon'][k]).ravel()[0]);assert count<max(4096,size*64),'contact scratch overflow'
     for j in range(count):
      world=int(cs['contact__worldid'][k,j]);adr=int(cs['contact__efc_address'][k,j,0])
      if adr<0:continue
      f=max(0.,float(cs['efc__force'][world,k,adr]))/4;g1,g2=map(int,cs['contact__geom'][k,j]);names=[m.geom(g1).name,m.geom(g2).name];terrain=next((n for n in names if n in ['floor','step']),None)
      if terrain is None:continue
      wheel=any('wheel' in n for n in names);normal=np.asarray(cs['contact__frame'][k,j]).reshape(3,3)[0];cp=cs['contact__pos'][k,j];top=abs(normal[2])>.7 and (terrain=='floor' or abs(cp[2]-.16)<.025)
      col=(0 if terrain=='floor' else 1) if wheel and top else (3 if wheel else 2);force[world,col]+=f
   else:
    _,_,done,info=env.step(action);charged+=size
    for frame in rec.frames:
     seen=set()
     assert frame['physics_heartbeat_delta']==1
     for c in frame['contacts']:
      key=(tuple(sorted([c['collider0'],c['collider1']])),tuple(c['position']),tuple(c['impulse']))
      if key in seen:continue
      seen.add(key)
      ours=next((s for s in [c['actor0'],c['actor1']] if '/Bike/' in s),None)
      if not ours:continue
      lane=int(ours.split('/env_')[1].split('/')[0]);other=c['actor1'] if ours==c['actor0'] else c['actor0']
      if '/floor_' not in other and not other.endswith('/step'):continue
      wheel=ours.endswith('/frontwheel') or ours.endswith('/rearwheel');top=abs(c['normal'][2])>.7 and ('/floor_' in other or abs(c['position'][2]-.16)<.025);col=(0 if '/floor_' in other else 1) if wheel and top else (3 if wheel else 2);force[lane,col]+=c['normal_impulse']/.02
    rec.frames.clear()
    raw=env.trace;q=np.tile(m.key_qpos[0],(size,1));q[:,:3]=raw[:,1:4];q[:,3:7]=raw[:,4:8];q[:,qa]=raw[:,8:13];v=np.zeros((size,m.nv));v[:,:3]=raw[:,18:21];v[:,3:6]=Rotation.from_quat(q[:,[4,5,6,3]]).inv().apply(raw[:,21:24]);v[:,va]=raw[:,13:18];ends=np.array([x['end_code'] for x in info])
   if a.engine=='mjx':charged+=size
   scored+=int(mask.sum());qlog.append(q.copy());vlog.append(v.copy());contactlog.append(force);endlog.append(ends if ends is not None else np.zeros(size,int))
   alive &= ~done
   # Physics continues for padded/inactive lanes; charged budget includes them.
   if tick%25==0:status('running',batch_begin=begin,control_tick=tick)
  qlog=np.array(qlog);vlog=np.array(vlog);forces=np.array(contactlog);mask=np.array(masks)
  rows_by_lane=[]
  for lane in range(size):
   rows=[]
   for tick in range(a.horizon):
    if not mask[tick,lane]:break
    kin.qpos[:]=qlog[tick,lane];mujoco.mj_kinematics(m,kin);clear=[]
    for g in wheel_ids:
     xyz=kin.geom_xpos[g];r=kin.geom_xmat[g].reshape(3,3);clear.append(xyz[2]-np.linalg.norm(r[2]*m.geom_size[g])-(.16 if 3.6<=xyz[0]<=7.6 else 0))
    angles=Rotation.from_quat(kin.qpos[[4,5,6,3]]).as_euler('xyz');vel=vlog[tick,lane];f=forces[tick,lane]
    rows.append(dict(time_s=(tick+1)*.02,contact_valid=True,support_floor_N=float(f[0]),support_step_N=float(f[1]),nonwheel_N=float(f[2]),illegal_wheel_N=float(f[3]),vz=float(vel[2]),min_wheel_clearance_m=float(min(clear)),roll=float(angles[0]),pitch=float(angles[1]),roll_rate=float(vel[3]),pitch_rate=float(vel[4]),vy=float(vel[1]),vx=float(vel[0]),x=float(kin.qpos[0])))
   landing=evaluate_landing(rows,weight,criteria);case=begin+lane
   results.append(dict(case=case,onset=D['onsets'][case//250],success=landing['strict_landing_1p0s'],takeoff_s=landing['takeoff_time_s'],touchdown_s=landing['first_touchdown_time_s'],hold_s=landing['max_stable_hold_s'],scored_ticks=len(rows),end_code=int(np.array(endlog)[len(rows)-1,lane]),requested_rms=float(np.sqrt(np.mean(np.array(reqlog)[:,lane]**2))),effective_rms=float(np.sqrt(np.mean(np.array(efflog)[:,lane]**2)))))
  np.savez_compressed(OUT/f'batch_{begin:04d}.npz',qpos=qlog,qvel=vlog,force=forces,observations=np.array(obslog),action=np.array(actlog),requested=np.array(reqlog),effective=np.array(efflog),mask=mask,initial_qpos=qbank,initial_qvel=vbank)
  write_json(OUT/'episodes.json',results);status('running')
 status('complete')
except Exception as e:status('error',error=repr(e));raise
finally:
 if a.engine=='physx':rec.close();app.close()
