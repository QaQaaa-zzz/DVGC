"""Real-PhysX half of the frozen-policy paired probe. Import after Kit starts."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from fast_path_probe import ROOT,record,dump,events_dict,sha


def target_run(case,spec,actor,out,target):
    import omni.usd
    from omni.physx import get_physx_interface
    from pxr import UsdPhysics,PhysxSchema,UsdShade
    from isaacsim.core.api import World
    from isaacsim.core.prims import RigidPrim
    from physx_task import PhysxTask,CFG,semantics
    from policy_runtime import Observation,control
    from fixed_rear_target import override_rear_target
    from target_observation_adapter import TargetObservationAdapter
    World.clear_instance();omni.usd.get_context().new_stage()
    config=json.loads((ROOT/'configs/isaac_source_ppo_fresh.json').read_text())
    config['source_ppo_port']=False;config['physics_dt']=float(target.get('physics_dt',.001))
    if 'target_drive_gains' in target:config['target_drive_gains']=target['target_drive_gains']
    counter=[];subscription=get_physx_interface().subscribe_physics_step_events(lambda dt:counter.append(float(dt)))
    env=PhysxTask(config,1,980001);m=env.m;stage=omni.usd.get_context().get_stage()
    ablation=None
    if bool(spec.get('remove_platform',False)) != bool(target.get('remove_platform',False)):
        raise ValueError('protocol/target platform ablation mismatch')
    if spec.get('remove_platform',False):
        from platform_ablation import move_platform_out_of_scene
        ablation=move_platform_out_of_scene(m)
        if not stage.RemovePrim('/World/envs/env_0/step'):raise RuntimeError('platform removal failed')
    mu=target.get('friction',.5)
    # Set wheel AND both terrain materials; changing only wheel under min is a false scan.
    wanted={m.geom(n).id for n in ['floor','step','frontwheel_collision','rearwheel_collision']}
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.MaterialAPI):
            name=prim.GetName();g=int(name.split('_')[-1]) if name.startswith('material_') else -1
            if g in wanted:
                api=UsdPhysics.MaterialAPI(prim);api.CreateStaticFrictionAttr(mu);api.CreateDynamicFrictionAttr(mu)
                PhysxSchema.PhysxMaterialAPI.Apply(prim).CreateFrictionCombineModeAttr('min')
    wheels=env.world.scene.add(RigidPrim('/World/envs/env_0/Bike/.*wheel',name='diagnostic_wheels',
        track_contact_forces=True,disable_stablization=False,reset_xform_properties=False))
    env.hard_reset();env.reset(np.array([0]),seeds=[980001],ground_only=True)
    qpos=np.asarray(case['qpos']);qvel=np.asarray(case['qvel']);rot=Rotation.from_quat(qpos[[4,5,6,3]])
    omega=rot.apply(qvel[3:6]);com_v=qvel[:3]+np.cross(omega,rot.apply(m.body_ipos[1]))
    env.view.set_world_poses(qpos[None,:3].astype(np.float32),qpos[None,3:7].astype(np.float32))
    env.view.set_velocities(np.r_[com_v,omega][None].astype(np.float32))
    env.view.set_joint_positions(qpos[None,env.qa].astype(np.float32));env.view.set_joint_velocities(qvel[None,env.va].astype(np.float32))
    env.initial_x[0]=qpos[0];env.prev_vel[0]=qvel[:3]
    env.observers[0]=Observation(m);env.obs[0]=env.observers[0].initial(qpos[0]);env.events[0]=semantics.initial_event_state(np.asarray(qpos[0]),CFG)
    env.world.physics_sim_view.update_articulations_kinematic()
    adapter_config=json.loads(Path(target['observation_adapter']).read_text()) if target.get('observation_adapter') else None
    adapter=TargetObservationAdapter(adapter_config)
    materials={}
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.MaterialAPI):
            name=prim.GetName();g=int(name.split('_')[-1]) if name.startswith('material_') else -1
            if g in wanted:
                api=UsdPhysics.MaterialAPI(prim);ph=PhysxSchema.PhysxMaterialAPI(prim)
                materials[m.geom(g).name]=dict(path=str(prim.GetPath()),static_friction=api.GetStaticFrictionAttr().Get(),dynamic_friction=api.GetDynamicFrictionAttr().Get(),combine=ph.GetFrictionCombineModeAttr().Get())
    if set(materials)!= {'floor','step','frontwheel_collision','rearwheel_collision'}:raise RuntimeError('missing actual contact material')
    for value in materials.values():
        if not np.isclose(value['static_friction'],mu) or not np.isclose(value['dynamic_friction'],mu) or value['combine']!='min':raise RuntimeError('effective friction scan not applied')
    bindings=[]
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            bound,_=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(materialPurpose='physics')
            bindings.append(dict(collider=str(prim.GetPath()),material=str(bound.GetPath())))
    audit=dict(engine='real PhysX',actor_sha256=sha(spec['actor']),target=target,physics_dt=env.world.get_physics_dt(),
               materials=materials,collider_bindings=bindings,actual_native_gains=[x.tolist() for x in env.view.get_gains()],
               actual_force_limits=env.view.get_max_efforts().tolist(),joint_names=env.names,
               effort_semantics='only explicit hip/knee have separately available motor effort; native rear/steer projected/actuation readbacks are logged separately, not asserted actual motor effort',
               contact_semantics='last physical substep wheel net contact force; bottom-point kinematic slip is separately labelled geometric proxy',
               initialization_physics_steps=len(counter),original_step_model_source_port=False)
    from runtime_contact_audit import read_contact_audit
    audit['contact_runtime']=read_contact_audit(stage,m)
    audit['terrain_ablation']=ablation
    audit['platform_collider_present']=bool(stage.GetPrimAtPath('/World/envs/env_0/step'))
    if ablation and audit['platform_collider_present']:raise RuntimeError('platform still present')
    audit['rear_override']=dict(fixed_rad_s=target.get('fixed_rear_speed_rad_s'),history_semantics='Actor requested normalized action retained in history/reward; physical rear target override logged separately',actor_output_modified=False)
    audit['observation_adapter']=adapter_config
    audit['observation_adapter_sha256']=sha(target['observation_adapter']) if target.get('observation_adapter') else None
    audit['contact_sensor_body_paths']=list(wheels.prim_paths)
    wheel_materials=np.asarray(wheels._physics_view.get_material_properties()).copy()
    audit['wheel_material_properties_physx_backend']=wheel_materials.tolist()
    if not np.allclose(wheel_materials[:,:,:2],mu,atol=1e-6,rtol=0):
        raise RuntimeError('PhysX wheel shape friction differs from rebuilt active materials')
    audit['effective_contact_pairs']={f'{wheel}/{terrain}':{
        'static_friction':min(materials[wheel]['static_friction'],materials[terrain]['static_friction']),
        'dynamic_friction':min(materials[wheel]['dynamic_friction'],materials[terrain]['dynamic_friction']),
        'combine':materials[wheel]['combine'],
        'evidence':'PhysX wheel shape readback + bound active USD terrain material, both min, after hard rebuild'}
        for wheel in ['frontwheel_collision','rearwheel_collision'] for terrain in ['floor','step']}
    audit['rear_speed_controller']=target.get('rear_speed_controller')
    dump(out/'runtime.json',audit);init_count=len(counter);rows=[];details=[];done=False;end_code=0
    from rear_speed_controller import RearSpeedController
    speed_controller=RearSpeedController(target['rear_speed_controller']) if target.get('rear_speed_controller') else None
    raw=json.loads((ROOT/'policy/resolved_config.json').read_text());acceleration=np.zeros(3)
    for tick in range(spec['max_ticks']+1):
        pos,quat=env.view.get_world_poses();q=env.view.get_joint_positions()[0];qd=env.view.get_joint_velocities()[0]
        w=env.view.get_angular_velocities()[0];r=Rotation.from_quat(quat[0,[1,2,3,0]])
        v=env.view.get_linear_velocities()[0]-np.cross(w,r.apply(m.body_ipos[1]))
        qp=np.r_[pos[0],quat[0],np.zeros(5)];qv=np.r_[v,r.inv().apply(w),np.zeros(5)];qp[env.qa]=q;qv[env.va]=qd
        raw_obs=env.obs[0].copy();obs=adapter.transform(raw_obs,qpos=qp,qvel=qv);action=actor(obs);policy_ctrl=control(action,m,raw);ctrl=override_rear_target(policy_ctrl,target.get('fixed_rear_speed_rad_s'),m.actuator_ctrlrange[1]);event=events_dict(env.events[0])
        rear_override=target.get('fixed_rear_speed_rad_s');hold_arm=False;speed_info=None
        if speed_controller is not None:
            from fast_path_probe import wheel_kinematics
            grounded=bool(np.min(wheel_kinematics(m,qp,qv)[:,3])<=.002)
            speed_info=speed_controller.update(v[0],grounded,.02,bool(event['jump_signal']))
            rear_override=speed_info['rear_speed_rad_s'];hold_arm=not speed_info['jump_released']
            ctrl=override_rear_target(policy_ctrl,rear_override,m.actuator_ctrlrange[1])
            if hold_arm:ctrl[2:]=env.baseq[env.arm]
        efforts=np.zeros(4);efforts[2:]=env.effort[0,env.arm];available=np.array([False,False,True,True])
        row=record(m,tick,obs,qp,qv,action,ctrl,obs[53:56],obs[56:59],efforts,available,event,done,end_code)
        row['action_applied']=not done and tick<spec['max_ticks'];row['acceleration_world']=acceleration.copy()
        row['raw_observation']=raw_obs
        row['policy_target']=policy_ctrl.copy()
        row['jump_released']=not hold_arm
        row['speed_hold_s']=speed_info['held_s'] if speed_info else 0.
        row['rear_target_readback']=float('nan')
        row['contact_net_force']=np.zeros((2,3)) if tick==0 else np.asarray(wheels.get_net_contact_forces(dt=env.dt)).copy()
        details.append(dict(event=event,speed_controller=speed_info,actuation_readback=env.pv.get_dof_actuation_forces()[0].tolist(),contact_force_available=tick>0))
        rows.append(row)
        if not row['action_applied']:break
        _,_,finished,infos=env.step(action[None],rear_speed_override=rear_override,hold_arm_targets=hold_arm);done=bool(finished[0]);end_code=infos[0]['end_code'];acceleration=env.trace[0,24:27].copy()
        applied=env.view.get_applied_actions()
        rows[-1]['rear_target_readback']=float(applied.joint_velocities[0,env.ai[1]])
        np.testing.assert_allclose(rows[-1]['rear_target_readback'],ctrl[1],rtol=1e-6,atol=1e-6)
        if tick%50==0:print('PHYSX_TICK',case['name'],tick,flush=True)
    counts=dict(rollout_physics_steps=len(counter)-init_count,initialization_physics_steps=init_count)
    assert counts['rollout_physics_steps']==(len(rows)-1)*env.substeps
    dump(out/'physics_steps.json',dict(dt_s=counter));subscription.unsubscribe();env.world.stop();env.world.clear();World.clear_instance()
    return rows,details,counts
