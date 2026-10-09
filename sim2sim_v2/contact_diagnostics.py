"""Real PhysX report recorder. No geometry-based contact inference."""
import numpy as np

class ContactRecorder:
    def __init__(self, task, lanes=None):
        import omni.usd
        from pxr import UsdPhysics, PhysxSchema
        from omni.physx import get_physx_simulation_interface, get_physx_interface
        self.lanes=lanes
        self.task=task; self.frames=[]; self.pending=[]; self.callbacks=0; self.tick=0
        self.validated=False; self.applied=[]
        for p in omni.usd.get_context().get_stage().Traverse():
            if p.HasAPI(UsdPhysics.RigidBodyAPI):
                PhysxSchema.PhysxContactReportAPI.Apply(p).CreateThresholdAttr().Set(0.)
                self.applied.append(str(p.GetPath()))
        self.heartbeat=0
        self.step_sub=get_physx_interface().subscribe_physics_step_events(self._heartbeat)
        self.sub=get_physx_simulation_interface().subscribe_contact_report_events(self._callback)
        self.original=task.world.step
        def step(*a,**kw):
            self.pending=[]; before=self.callbacks; heartbeat_before=self.heartbeat
            result=self.original(*a,**kw)
            self.frames.append({'physics_tick':self.tick,'callback_seen':self.callbacks>before,
                                'physics_heartbeat_delta':self.heartbeat-heartbeat_before,'sensor_valid':self.validated,'contacts':self.pending.copy(),
                                'joint_qd':task.view.get_joint_velocities().tolist(),
                                'effort':task.pv.get_dof_actuation_forces().tolist()})
            self.tick+=1
            return result
        task.world.step=step

    def _heartbeat(self,dt):
        self.heartbeat+=1

    def _callback(self, headers, data):
        from pxr import PhysicsSchemaTools
        self.callbacks+=1
        for h in headers:
            paths={k:str(PhysicsSchemaTools.intToSdfPath(getattr(h,k))) for k in ['actor0','actor1','collider0','collider1']}
            if self.lanes is not None and not any(f'/env_{lane}/' in (paths['actor0']+paths['actor1']) for lane in self.lanes):continue
            for j in range(h.contact_data_offset,h.contact_data_offset+h.num_contact_data):
                c=data[j]
                impulse=np.asarray(c.impulse,dtype=float)
                normal=np.asarray(c.normal,dtype=float)
                if np.isfinite(impulse).all() and np.linalg.norm(impulse)>0:self.validated=True
                self.pending.append(dict(paths,position=list(map(float,c.position)),normal=normal.tolist(),
                    impulse=impulse.tolist(),normal_impulse=float(abs(np.dot(impulse,normal))),
                    separation=float(c.separation),event_type=str(h.type)))

    def certify(self):
        """Explicitly certify after a known contact microcase emits finite reports."""
        points=[c for f in self.frames for c in f['contacts']]
        if not points or not all(np.isfinite(c['normal_impulse']) for c in points):
            raise RuntimeError('Contact reports not validated: no finite known-contact samples')
        self.validated=True

    def close(self):
        self.task.world.step=self.original; self.sub=None;self.step_sub=None


def aggregate(frames, dt):
    """Aggregate unique collider contact points, never double-count body APIs."""
    groups={}
    for f in frames:
        seen=set()
        for c in f['contacts']:
            key=(tuple(sorted([c['collider0'],c['collider1']])),tuple(c['position']),tuple(c['impulse']))
            if key in seen: continue
            seen.add(key)
            pair=tuple(sorted([c['actor0'],c['actor1']]))
            g=groups.setdefault(pair,{'normal_impulse':0.,'points':0})
            g['normal_impulse']+=c['normal_impulse'];g['points']+=1
    return [{'actors':list(k),**v,'mean_normal_force':v['normal_impulse']/(len(frames)*dt)} for k,v in groups.items()]
