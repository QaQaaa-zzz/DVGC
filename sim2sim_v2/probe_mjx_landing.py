"""Exact9904128 nominal MJX-Warp, record real solver contacts at all four substeps."""
import os,sys,json,time,hashlib,dataclasses,traceback
from pathlib import Path
os.environ.update(XLA_PYTHON_CLIENT_PREALLOCATE='false',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4')
ROOT=Path(__file__).resolve().parent
SRC=Path('/home/qy/DVGC/JIT/runs/code_snapshots/jump_ori_sim2sim_vxgt02_20260928/JIT/src')
sys.path.insert(0,str(SRC))
import numpy as np
import jax,jax.numpy as jp
import mujoco
from mujoco import mjx
from mujoco_playground._src import mjx_env
from scipy.spatial.transform import Rotation
from jit_dvgc.config import load_config
from jit_dvgc.env import TwoPhaseBikeEnv
from policy_runtime import Actor
from fast_path_probe import wheel_kinematics
from landing_evaluation import evaluate_landing,LandingCriteria
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--output',default='runs/p1_mjx_9904128_retry02');args=parser.parse_args()
OUT=ROOT/args.output;OUT.mkdir(parents=True,exist_ok=False)
cp=Path('/home/qy/DVGC/JIT/runs/phase_u/phase_u_jump_ori_sim2sim_vxgt02_10002432_seed820707_20260928')
actorpath=Path('/home/qy/ISAAC——SIM/runs/experiments/physx_jump_ori_from9904128_kd5_5ms_10m_20260928/initial_policy/actor.npz')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(n,x):(OUT/n).write_text(json.dumps(x,indent=2,allow_nan=False))
dump('declaration.json',dict(source_checkpoint=str(cp/'checkpoints/transition_9904128'),actor=str(actorpath),actor_sha256=sha(actorpath),source_root=str(SRC),source_config_sha256=sha(cp/'resolved_config.json'),source_env_sha256=sha(SRC/'jit_dvgc/env.py'),max_control_ticks=400,max_physics_steps=1600,training_transitions=0,protocol='Exact checkpoint config, nominal fixed natural reset; reset randomization explicitly disabled, no physics DR wrapper; low speed terminal preserved',landing_criteria=dataclasses.asdict(LandingCriteria())))
rows=[];contacts=[];evalrows=[];start=time.monotonic();steps=0
try:
 cfg=load_config(cp/'resolved_config.json',runtime_only=True);cfg=dataclasses.replace(cfg,sim2sim_randomization=None)
 env=TwoPhaseBikeEnv(cfg);m=env.mj_model;actor=Actor(actorpath)
 assert Path(sys.modules[TwoPhaseBikeEnv.__module__].__file__).is_relative_to(SRC)
 assert sha(env.xml_path)==cfg.model['xml_sha256']
 assert np.isclose(m.opt.timestep,.005) and env.dt==.02 and env.n_substeps==4
 identity=json.loads(actorpath.with_name('identity.json').read_text());assert identity['parent']['payload_sha256']==sha(cp/'checkpoints/transition_9904128/payload.pkl')
 dump('evaluated_config.json',dict(original_resolved_config=json.loads((cp/'resolved_config.json').read_text()),explicit_overrides={'sim2sim_randomization':None},physics_randomization_wrapper=False))
 reset=jax.jit(env.reset_natural);state=reset(jax.random.PRNGKey(980001));jax.block_until_ready(state)
 np.testing.assert_allclose(np.asarray(state.data.qpos)[:3],[1.5,0,.15],atol=1e-6)
 def instrumented_step(state,action):
  captured={};original=mjx_env.step
  def all_substeps(model,data,ctrl,n_substeps):
   def one(d,_):
    d=mjx.step(model,d.replace(ctrl=ctrl));return d,{name:getattr(d._impl,name) for name in ['nacon','contact__geom','contact__dim','contact__efc_address','contact__pos','contact__frame','efc__force']}
   last,stack=jax.lax.scan(one,data,None,length=n_substeps);captured['data']=stack;return last
  mjx_env.step=all_substeps
  try:result=env.step(state,action)
  finally:mjx_env.step=original
  return result,captured['data']
 step=jax.jit(instrumented_step)
 dump('runtime.json',dict(mujoco=mujoco.__version__,jax=jax.__version__,backend=jax.default_backend(),devices=[str(x) for x in jax.devices()],sim_dt=float(m.opt.timestep),ctrl_dt=env.dt,env_module=str(sys.modules[TwoPhaseBikeEnv.__module__].__file__),initial_qpos=np.asarray(state.data.qpos).tolist(),initial_qvel=np.asarray(state.data.qvel).tolist(),model_sha256=sha(env.xml_path),contact_epoch='Direct DataWarp contact__ and efc__force per RK4 substep; elliptic contact extraction matches installed support.contact_force_fn; no get_data or CPU contact reconstruction'))
 for tick in range(400):
  obs=np.asarray(state.obs['state']);action=actor(obs);state,sub=step(state,jp.asarray(action));jax.block_until_ready(state);steps+=4
  sub=jax.device_get(sub)
  dump('progress.json',dict(completed_control_steps=tick+1,actual_physics_steps=steps))
  hosts=range(4)
  floor_imp=0.;step_imp=0.;body_imp=0.;illegal_wheel_imp=0.;step_contacts=0;contactvalid=True
  for k in hosts:
   assert int(m.opt.cone)==int(mujoco.mjtCone.mjCONE_ELLIPTIC)
   nc=int(np.asarray(sub['nacon'][k]).ravel()[0]);efc=np.asarray(sub['efc__force'][k]).ravel()
   for ci in range(nc):
    adr=np.asarray(sub['contact__efc_address'][k,ci]).ravel();dim=int(sub['contact__dim'][k,ci])
    if adr[0]<0:continue
    force=np.zeros(6)
    for fi in range(dim):
     if 0<=adr[fi]<len(efc):force[fi]=efc[adr[fi]]
    assert np.isfinite(force).all()
    g1,g2=map(int,sub['contact__geom'][k,ci]);names=[m.geom(g1).name,m.geom(g2).name];body1=int(m.geom_bodyid[g1]);body2=int(m.geom_bodyid[g2]);terrain=next((n for n in names if n in ['floor','step']),None)
    wheel=any('wheel' in n for n in names);normal=np.asarray(sub['contact__frame'][k,ci]).reshape(3,3)[0];pos=np.asarray(sub['contact__pos'][k,ci])
    normalforce=max(0.,float(force[0]));imp=normalforce*.005
    top=abs(float(normal[2]))>.7 and (terrain=='floor' or (terrain=='step' and abs(float(pos[2])-.16)<.025))
    if wheel and terrain and top:
     if terrain=='floor':floor_imp+=imp
     else:step_imp+=imp;step_contacts+=1
    elif (body1==0)!=(body2==0):
     if wheel:illegal_wheel_imp+=imp
     else:body_imp+=imp
    contacts.append(dict(control_tick=tick+1,physics_tick=tick*4+k+1,time_s=(tick*4+k+1)*.005,geom=names,position=pos.tolist(),normal=normal.tolist(),force_contact=force.tolist(),normal_force_N=normalforce,normal_impulse_Ns=imp,wheel=wheel,terrain=terrain,top_support=top,efc_address=adr.tolist()))
  q=np.asarray(state.data.qpos);v=np.asarray(state.data.qvel);rpy=Rotation.from_quat(q[[4,5,6,3]]).as_euler('xyz');wk=wheel_kinematics(m,q,v)
  rows.append(dict(time_s=(tick+1)*.02,qpos=q,qvel=v,obs=obs,action=action,reward=float(state.reward),done=bool(state.done),end_code=int(state.info['end_code']),rpy=rpy,force_support_floor=floor_imp/.02,force_support_step=step_imp/.02,force_nonwheel=body_imp/.02,contact_valid=contactvalid,apex_seen=bool(state.info['events'].apex_seen)))
  evalrows.append(dict(time_s=(tick+1)*.02,contact_valid=contactvalid,support_floor_N=floor_imp/.02,support_step_N=step_imp/.02,nonwheel_N=body_imp/.02,illegal_wheel_N=illegal_wheel_imp/.02,vz=float(v[2]),min_wheel_clearance_m=float(min(wk[:,3])),roll=float(rpy[0]),pitch=float(rpy[1]),roll_rate=float(v[3]),pitch_rate=float(v[4]),vy=float(v[1]),vx=float(v[0]),x=float(q[0])))
  if tick%25==0:print('TICK',tick+1,flush=True)
  if bool(state.done):break
 arrays={k:np.asarray([r[k] for r in rows]) for k in rows[0]};np.savez_compressed(OUT/'trace.npz',**arrays);dump('contacts.json',contacts);dump('landing_inputs.json',evalrows)
 landing=evaluate_landing(evalrows,float(m.body_mass.sum()*9.81));dump('landing.json',landing)
 dump('status.json',dict(status='complete',control_steps=len(rows),physics_steps=steps,training_transitions=0,end_code=rows[-1]['end_code'],apex_seen=rows[-1]['apex_seen'],strict_landing_1p0s=landing['strict_landing_1p0s'],stable_landing_0p5s=landing['stable_landing_0p5s'],wall_s=time.monotonic()-start))
except Exception:
 dump('status.json',dict(status='error',error=traceback.format_exc(),actual_physics_steps=steps,training_transitions=0,completed_controls=len(rows)));raise
