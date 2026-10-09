"""Frozen-Actor closed-loop traces, actual original MJX-Warp or real PhysX.

Rows are states at tick k, with the observation/action for transition k -> k+1.
The final row has action_applied=False. No learner or reward-based selection.
"""
import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent
SOURCE = Path('/home/qy/DVGC/JIT/src')
JOINTS = ['rearwheel_joint', 'steering_joint', 'frontwheel_joint', 'hip_joint', 'knee_joint']


def dump(path, value):
    def safe(v):
        if isinstance(v,np.ndarray):return safe(v.tolist())
        if isinstance(v,np.generic):return safe(v.item())
        if isinstance(v,float) and not np.isfinite(v):return {'nonfinite':str(v)}
        if isinstance(v,dict):return {k:safe(x) for k,x in v.items()}
        if isinstance(v,(list,tuple)):return [safe(x) for x in v]
        return v
    # Preserve invalid diagnostic entries explicitly, never turn them into zero.
    Path(path).write_text(json.dumps(safe(value), indent=2, allow_nan=False) + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def events_dict(event):
    return {f.name: np.asarray(getattr(event, f.name)).item() for f in dataclasses.fields(event)}


def wheel_kinematics(m, qpos, qvel):
    """End-state geometric contact candidates/slip, not a force-contact oracle."""
    import mujoco
    d = mujoco.MjData(m); d.qpos[:] = qpos; d.qvel[:] = qvel
    mujoco.mj_fwdPosition(m, d); mujoco.mj_fwdVelocity(m, d)
    values = []
    for name in ['frontwheel_collision', 'rearwheel_collision']:
        g = m.geom(name).id; b = m.geom_bodyid[g]; r = d.geom_xmat[g].reshape(3, 3)
        n = r.T @ np.array([0., 0., -1.]); size = m.geom_size[g]
        local = size**2 * n / np.linalg.norm(size * n)
        point = d.geom_xpos[g] + r @ local
        velocity = np.zeros(6); mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, b, velocity, 0)
        point_velocity = velocity[3:] + np.cross(velocity[:3], point - d.xipos[b])
        step = m.geom('step'); over = abs(point[0] - step.pos[0]) <= step.size[0] and abs(point[1] - step.pos[1]) <= step.size[1]
        terrain = max(0., step.pos[2] + step.size[2]) if over else 0.
        values.append(np.r_[point, point[2] - terrain, point_velocity])
    return np.asarray(values)


def record(m, tick, obs, qpos, qvel, action, target, gyro, acc, effort, available, event, done, end_code):
    rotation = Rotation.from_quat(qpos[[4, 5, 6, 3]])
    qa = [int(m.joint(n).qposadr[0]) for n in JOINTS]; va = [int(m.joint(n).dofadr[0]) for n in JOINTS]
    return dict(tick=tick, time_s=tick*.02, observation=np.asarray(obs).copy(), frames=np.asarray(obs)[:75].reshape(3,25).copy(),
                action=np.asarray(action).copy(), target=np.asarray(target).copy(), qpos=np.asarray(qpos).copy(), qvel=np.asarray(qvel).copy(),
                rpy=rotation.as_euler('xyz'), omega_body=np.asarray(qvel)[3:6].copy(), omega_world=rotation.apply(qvel[3:6]),
                actor_gyro_body=np.asarray(gyro).copy(), actor_acc_body=np.asarray(acc).copy(),
                joint_q=np.asarray(qpos)[qa], joint_qd=np.asarray(qvel)[va], effort=np.asarray(effort).copy(), effort_available=np.asarray(available).copy(),
                wheel_kinematics=wheel_kinematics(m,qpos,qvel), event=np.asarray(list(event.values())), done=bool(done), end_code=int(end_code))


def source_run(case, spec, actor, out):
    import jax
    import jax.numpy as jp
    import mujoco
    from mujoco import mjx
    from jit_dvgc.config import load_config
    from jit_dvgc.env import TwoPhaseBikeEnv
    from policy_runtime import control
    cfg = load_config(ROOT/'policy/resolved_config.json',runtime_only=True)
    removed=bool(spec.get('remove_platform',False))
    env = TwoPhaseBikeEnv(cfg,convert_model=not removed)
    ablation=None
    if removed:
        from platform_ablation import move_platform_out_of_scene
        from jit_dvgc.model import put_warp_model
        ablation=move_platform_out_of_scene(env.mj_model)
        env._bundle=put_warp_model(env._bundle)
        # Flat floor remains under the old platform footprint; virtual x cue stays.
        env._geometry=env._geometry.replace(obstacle_top_z=0.)
    reset = jax.jit(env.reset_natural); step = jax.jit(env.step)
    state = reset(jax.random.PRNGKey(980001)); jax.block_until_ready(state)
    # Explicit declared ground perturbations only. Natural uses untouched reset.
    if case['reset_type'] != 'ground_natural':
        data = state.data.replace(qpos=jp.asarray(case['qpos'],jp.float32),qvel=jp.asarray(case['qvel'],jp.float32))
        data = mjx.forward(env.mjx_model,data); state = state.replace(data=data)
    np.testing.assert_allclose(np.asarray(state.data.qpos),case['qpos'],atol=2e-6,rtol=0)
    np.testing.assert_allclose(np.asarray(state.data.qvel),case['qvel'],atol=2e-6,rtol=0)
    m=env.mj_model; raw=json.loads((ROOT/'policy/resolved_config.json').read_text()); rows=[]; details=[]
    dump(out/'runtime.json',dict(engine='actual original TwoPhaseBikeEnv MJX-Warp',mujoco=mujoco.__version__,jax=jax.__version__,
         physics_dt=float(m.opt.timestep),control_dt=env.dt,substeps=env.n_substeps,model_sha256=sha(env.xml_path),terrain_ablation=ablation,
         config_identity=cfg.config_sha256,actor_sha256=sha(spec['actor']),reset_type=case['reset_type'],initial_override=case['reset_type']!='ground_natural'))
    for tick in range(spec['max_ticks']+1):
        data=state.data; obs=np.asarray(state.obs['state']); action=actor(obs); target=control(action,m,raw)
        qpos,qvel,sensors=np.asarray(data.qpos),np.asarray(data.qvel),np.asarray(data.sensordata)
        get_sensor=lambda n:sensors[int(m.sensor(n).adr[0]):int(m.sensor(n).adr[0])+int(m.sensor(n).dim[0])]
        event=events_dict(state.info['events']); done=bool(state.done)
        row=record(m,tick,obs,qpos,qvel,action,target,get_sensor('gyro_local'),get_sensor('acc_local'),np.asarray(data.actuator_force),np.ones(4,bool),event,done,state.info['end_code'])
        row['action_applied']=not done and tick<spec['max_ticks']
        row['acceleration_world']=Rotation.from_quat(qpos[[4,5,6,3]]).apply(get_sensor('acc_local'))+m.opt.gravity
        # Raw contacts belong to MJX's derived sampling stage; preserve their epoch label.
        host=mjx.get_data(m,data)
        contact=host.contact
        dist=np.asarray(contact.dist)[:host.ncon];valid=np.flatnonzero(dist<=0)
        detail=dict(event=event,contact_epoch='raw MJX derived stage after RK4',contacts=dict(
            indices=valid.tolist(),distance=dist[valid].tolist(),position=np.asarray(contact.pos)[valid].tolist(),
            geom=np.asarray(contact.geom)[valid].tolist()),metrics={k:float(v) for k,v in state.metrics.items() if k.startswith(('event/','terminal/'))})
        rows.append(row);details.append(detail)
        if not row['action_applied']: break
        state=step(state,jp.asarray(action));jax.block_until_ready(state)
        if tick%50==0: print('SOURCE_TICK',case['name'],tick,flush=True)
    return rows,details,dict(rollout_physics_steps=(len(rows)-1)*env.n_substeps,initialization_physics_steps=0)


def main():
    p=argparse.ArgumentParser();p.add_argument('--engine',choices=['mjx','physx'],required=True);p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--cases',nargs='+');p.add_argument('--target-config',type=Path)
    args=p.parse_args();spec=json.loads(args.protocol.read_text());assert sha(spec['actor'])==spec['actor_sha256']
    if spec['control_dt']!=.02 or not 1<=spec['max_ticks']<=400:raise ValueError('fixed 20ms, at most400ticks')
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    os.environ.update(XLA_PYTHON_CLIENT_PREALLOCATE='false',JAX_COMPILATION_CACHE_DIR=str(ROOT/'cache/fast_path_jax'),WARP_CACHE_PATH=str(ROOT/'cache/fast_path_warp'),OPENBLAS_NUM_THREADS='1')
    sys.dont_write_bytecode=True;sys.path.insert(0,str(SOURCE));app=None
    if args.engine=='physx':
        os.environ.update(JAX_PLATFORMS='cpu',OMNI_KIT_ACCEPT_EULA='YES')
        from isaacsim import SimulationApp
        app=SimulationApp({'headless':True,'disable_viewport_updates':True,'limit_cpu_threads':4,'extra_args':['--/app/settings/loadUserConfig=false','--/app/settings/persistent=false']})
    from policy_runtime import Actor
    actor=Actor(spec['actor']);started=time.monotonic();summary=[]
    dump(out/'declaration.json',dict(protocol=spec,engine=args.engine,code_sha256=sha(__file__),actor_sha256=sha(spec['actor']),training_transitions=0))
    dump(out/'status.json',dict(status='running'))
    try:
        for case in spec['cases']:
            if args.cases and case['name'] not in args.cases:continue
            case_out=out/case['name'];case_out.mkdir();dump(case_out/'case.json',case)
            if args.engine=='mjx':rows,details,counts=source_run(case,spec,actor,case_out)
            else:
                from fast_path_target import target_run
                rows,details,counts=target_run(case,spec,actor,case_out,json.loads(args.target_config.read_text()))
            arrays={k:np.asarray([r[k] for r in rows]) for k in rows[0]};np.savez_compressed(case_out/'trace.npz',**arrays)
            dump(case_out/'details.json',details)
            result=dict(name=case['name'],ticks=len(rows)-1,done=rows[-1]['done'],end_code=rows[-1]['end_code'],
                        task_success=any(d['event']['apex_seen'] for d in details),apex_height=float(max(r['qpos'][2] for r in rows)),
                        event_fields=list(details[0]['event']),counts=counts,actor_sha256=sha(spec['actor']))
            dump(case_out/'summary.json',result);summary.append(result);dump(out/'summary.json',summary);print('CASE_DONE',json.dumps(result),flush=True)
        assert sha(spec['actor'])==spec['actor_sha256'];dump(out/'status.json',dict(status='complete',cases=len(summary),wall_s=time.monotonic()-started,training_transitions=0))
    except Exception:
        dump(out/'status.json',dict(status='error',error=traceback.format_exc(),completed_cases=len(summary)));raise
    finally:
        if app:app.close()


if __name__=='__main__': main()
