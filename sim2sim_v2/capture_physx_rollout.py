"""One declared genuine training-mode rollout; zero accepted learner updates."""
import os,json,argparse,time
from pathlib import Path
os.environ.update(JAX_PLATFORMS='cpu',OPENBLAS_NUM_THREADS='1',OMNI_KIT_ACCEPT_EULA='YES')
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['warmfinal','fresh'],required=True);p.add_argument('--output',required=True);p.add_argument('--config',required=True);p.add_argument('--checkpoint');a=p.parse_args()
out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
def save(n,v):(out/n).write_text(json.dumps(v,indent=2))
save('declaration.json',dict(vars(a),control_interactions_limit=24576,physics_interactions_limit=98304,accepted_updates=0,role='training-mode stochastic diagnostic; natural and airborne resets separated',contact_lanes=[0,19,380,383]))
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'disable_viewport_updates':True,'limit_cpu_threads':16,'extra_args':['--/app/settings/loadUserConfig=false','--/app/settings/persistent=false']})
import logging
logging.getLogger('jax').setLevel(logging.WARNING)
import jax,numpy as np
from brax.training import types
from brax_physx_learner import Learner
from physx_task import PhysxTask
from contact_diagnostics import ContactRecorder
from learner_diagnostics import capture_rollout
spec=json.loads(Path(a.config).read_text());ppo=spec['ppo'];assert ppo['num_parallel_envs']==384 and ppo['unroll_length']==64
learner=Learner.restore(Path(a.checkpoint),ppo) if a.mode=='warmfinal' else Learner(None,ppo)
env=PhysxTask(spec,384,ppo['seed']);rec=ContactRecorder(env,lanes=[0,19,380,383]);env.hard_reset()
# Preserve source reset stream and actual stochastic initial states as in training.
initial_air=env.airborne.copy();initial_pos=np.array(env.view.get_world_poses()[0]);batch=[];info_batch=[];traces=[];resets=[];start=time.monotonic();ticks=0
try:
 learner.begin_rollout()
 for tick in range(64):
  obs={'state':env.obs.copy(),'privileged_state':env.privileged_obs.copy()};action,extra=learner.sample(obs)
  _,reward,done,info=env.step(action);ticks+=1;traces.append(env.trace.copy());info_batch.append(info);resets.append(env.airborne.copy())
  env.reset(np.flatnonzero(done));next_obs={'state':env.obs.copy(),'privileged_state':env.privileged_obs.copy()}
  batch.append(types.Transition(obs,action,reward,1-done.astype(np.float32),next_obs,dict(policy_extras=extra,state_extras=dict(truncation=np.zeros(384,np.float32),time_out=np.zeros(384,np.float32)))))
  save('status.json',dict(state='collecting',ticks=ticks,control_interactions=ticks*384,budget=24576,accepted_updates=0,wall_s=time.monotonic()-start))
 data=jax.tree.map(lambda *x:np.swapaxes(np.stack(x),0,1),*batch)
 capture_rollout(out/'capture',learner,data)
 rec.certify();save('contacts.json',{'contact_lanes':[0,19,380,383],'frames':rec.frames})
 save('steps.json',info_batch);save('audit.json',env.audit);env.resolved_config.write(out)
 np.savez_compressed(out/'trace.npz',trace=np.array(traces),airborne_reset=np.array(resets),initial_airborne=initial_air,initial_world_position=initial_pos,origins=env.origins)
 save('status.json',dict(state='complete',ticks=ticks,control_interactions=ticks*384,physics_interactions=len(rec.frames)*384,budget=24576,accepted_updates=0,wall_s=time.monotonic()-start))
except Exception as e:
 save('failure.json',dict(error=repr(e),ticks=ticks,control_interactions=ticks*384,physics_interactions=len(rec.frames)*384,accepted_updates=0));raise
finally:
 rec.close();app.close()
