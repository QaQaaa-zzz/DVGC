"""Fixed-action solver-history diagnostic; no policy or learner updates."""
import argparse,json,os,time
from pathlib import Path
os.environ.update(OMNI_KIT_ACCEPT_EULA='YES',JAX_PLATFORMS='cpu',OPENBLAS_NUM_THREADS='1')
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--n',type=int,default=1);p.add_argument('--steps',type=int,default=32);p.add_argument('--cycles',type=int,default=3);p.add_argument('--config',required=True);p.add_argument('--actions',required=True);a=p.parse_args()
out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
def save(name,data):(out/name).write_text(json.dumps(data,indent=2))
save('declaration.json',dict(vars(a),max_control_interactions=a.n*a.steps*a.cycles,max_physics_interactions=4*a.n*a.steps*a.cycles,accepted_updates=0,protocol='hard baseline / ordinary reset after fixed prefix / hard rebuild; all lanes same local natural state and fixed action'))
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'disable_viewport_updates':True,'limit_cpu_threads':16,'extra_args':['--/app/settings/loadUserConfig=false','--/app/settings/persistent=false']})
import numpy as np
import omni.usd
from pxr import PhysxSchema
from physx_task import PhysxTask
from contact_diagnostics import ContactRecorder
spec=json.loads(Path(a.config).read_text());task=PhysxTask(spec,a.n,820707);rec=ContactRecorder(task,lanes=[0]);actions=np.load(a.actions)[:a.steps].reshape(a.steps,4)
count=0
try:
 for cycle in range(a.cycles):
  mode='soft' if cycle==1 else 'hard'
  if mode=='hard':task.hard_reset()
  task.reset(np.arange(a.n),ground_only=True)
  scene=task.world.get_physics_context().get_current_physics_scene_prim()
  # Schema getters read stage attributes; do not mislabel them as SDK internals.
  scene_values={str(x.GetName()):str(x.Get()) for x in scene.GetAttributes()}
  audit=dict(scene_attributes=scene_values,solver_position_iterations=task.view.get_solver_position_iteration_counts().tolist(),solver_velocity_iterations=task.view.get_solver_velocity_iteration_counts().tolist(),sleep_thresholds=task.view.get_sleep_thresholds().tolist(),stabilization_thresholds=task.view.get_stabilization_thresholds().tolist(),gains=[x.tolist() for x in task.view.get_gains()],masses=task.pv.get_masses().tolist(),origins=task.origins.tolist(),prim_paths=task.view.prim_paths,readback_scope='iteration/sleep/stabilization getters read USD attributes; gains/masses use initialized tensor view',available_tensor_methods=[x for x in dir(task.pv) if any(k in x for k in ('solver','sleep','stabil','force'))])
  save(f'{cycle}_{mode}_audit.json',audit)
  frames=[];obs=[];initial=dict(pos=task.view.get_world_poses()[0].tolist(),quat=task.view.get_world_poses()[1].tolist(),vel=task.view.get_velocities().tolist(),q=task.view.get_joint_positions().tolist(),qd=task.view.get_joint_velocities().tolist(),obs=task.obs.tolist())
  save(f'{cycle}_{mode}_initial.json',initial)
  first=len(rec.frames)
  for act in actions:
   obs.append(task.obs[0].copy());task.step(np.broadcast_to(act,(a.n,4)).copy());count+=a.n;frames.append(task.trace.copy())
  np.savez_compressed(out/f'{cycle}_{mode}.npz',trace=np.array(frames),observations=np.array(obs),actions=actions)
  save(f'{cycle}_{mode}_contacts.json',rec.frames[first:])
  save('status.json',dict(state='running',completed_cycles=cycle+1,control_interactions=count,physics_interactions=count*4,accepted_updates=0))
 save('status.json',dict(state='complete',completed_cycles=a.cycles,control_interactions=count,physics_interactions=count*4,accepted_updates=0))
except Exception as e:
 save('status.json',dict(state='error',error=repr(e),control_interactions=count,physics_interactions=len(rec.frames)*a.n,accepted_updates=0));raise
finally:rec.close();app.close()
