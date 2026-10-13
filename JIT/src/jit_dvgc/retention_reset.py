"""Four equal TRAIN reset pools, preserving complete saved closed-loop context."""
from pathlib import Path
import numpy as np
from .retention_repair import read,sha

GROUPS=('nominal_complete','random_complete','pi0_success_snapshots','teacher_recoverable')
RECOVERY_GROUPS=('complete_start','current_teacher_verified','historical_success')

def inventory_groups(inventory):
    return RECOVERY_GROUPS if inventory.get('schema')=='jit_teacher_recovery_reset_v1' else GROUPS


def validate_inventory(inventory):
    if inventory.get('role')!='TRAIN':raise ValueError('four reset pools must be TRAIN')
    if inventory.get('schema')=='jit_teacher_recovery_reset_v1':
        current=inventory.get('round')
        if type(current) is not int or current<1:raise ValueError('explicit positive round required')
        for name,mass in zip(RECOVERY_GROUPS,(.4,.3,.3)):
            pool=inventory.get(name,{})
            if pool.get('probability')!=mass:raise ValueError('recovery reset probability drift')
            if name=='complete_start':
                if any(np.asarray(pool.get(k)).ndim!=1 or np.asarray(pool.get(k)).size==0
                    or not np.isfinite(np.asarray(pool.get(k))).all() for k in ('qpos','qvel')):
                    raise ValueError('finite complete start required')
                continue
            if not pool.get('entries'):raise ValueError('empty '+name+' reset pool')
            for row in pool['entries']:
                if row.get('role')!='TRAIN' or row.get('verified_success') is not True:
                    raise ValueError('verified TRAIN success required for reset snapshot')
                r=row.get('round')
                if type(r) is not int or (r!=current if name=='current_teacher_verified' else not 0<=r<current):
                    raise ValueError('snapshot round provenance mismatch')
                if (not row.get('snapshot') or not row.get('snapshot_context_sha256')
                    or not isinstance(row.get('snapshot_time'),(int,float)) or not np.isfinite(row['snapshot_time'])
                    or row['snapshot_time']<0 or not isinstance(row.get('snapshot_time_source'),dict)
                    or not row['snapshot_time_source'].get('path') or not row['snapshot_time_source'].get('sha256')):
                    raise ValueError('explicit original snapshot time and source required')
        return inventory
    for name in GROUPS:
        p=inventory[name]
        if p['probability']!=.25:raise ValueError('fixed equal reset probabilities required')
        if name.endswith('snapshots') or name=='teacher_recoverable':
            if not p['entries'] or any(x['role']!='TRAIN' for x in p['entries']):raise ValueError('empty/non-TRAIN snapshot pool')
        if name=='random_complete' and not p['lanes']:raise ValueError('empty random pool')
    return inventory


def original_snapshot_time(row):
    """Recover the declared absolute simulation time without resetting event clocks.

    Historical snapshot payloads omit MJX time. Their immutable sidecar control
    step is the explicit source for the new reset protocol, at unchanged20ms.
    """
    source=row['snapshot_time_source']
    if sha(source['path'])!=source['sha256']:raise ValueError('snapshot original time source changed')
    payload=read(source['path'])
    field=source.get('control_step_field')
    if not isinstance(field,str) or not field or source.get('control_dt')!=.02:
        raise ValueError('original snapshot control clock field and unchanged .02dt required')
    for key in field.split('.'):
        if not isinstance(payload,dict) or key not in payload:raise ValueError('original snapshot clock field missing')
        payload=payload[key]
    if type(payload) is not int or payload<0:raise ValueError('nonnegative original control step required')
    expected=payload*source['control_dt']
    if not np.isclose(row['snapshot_time'],expected,rtol=0,atol=1e-6):
        raise ValueError('declared snapshot time differs from original control clock')
    return expected


def group_indices(key,n,*,probabilities=None):
    import jax
    if probabilities is None:return jax.random.randint(key,(n,),0,4)
    import jax.numpy as jp
    return jax.random.categorical(key,jp.log(jp.asarray(probabilities)),shape=(n,))


class FourPoolReset:
    def __init__(self,env,inventory):
        import jax,jax.numpy as jp
        from .continuation.device_rollout import stack_worlds
        from .unified_envelope_snapshot import load_unified_envelope_snapshot,restore_unified_envelope_snapshot,snapshot_context_sha256
        from .handoff_snapshot import compatibility_identity
        inv=validate_inventory(inventory);self.env=env;self.original_step=env.step;self.starts=[];self.lengths=[];self.entries=[];states=[]
        self.group_names=inventory_groups(inv)
        self.fresh_groups=1 if self.group_names==RECOVERY_GROUPS else 2
        self.probabilities=jp.asarray([inv[name]['probability'] for name in self.group_names]) if self.group_names==RECOVERY_GROUPS else None
        for group,name in enumerate(self.group_names):
            self.starts.append(len(states));rows=[]
            if group==0:rows=[dict(qpos=inv[name]['qpos'],qvel=inv[name]['qvel'],ancestor='TRAIN_NOMINAL')]
            elif group==1 and self.group_names==GROUPS:
                if sha(inv[name]['path'])!=inv[name]['sha256']:raise ValueError('random pool changed')
                with np.load(inv[name]['path']) as f:rows=[dict(qpos=f['qpos'][i],qvel=f['qvel'][i],ancestor=f'TRAIN_RANDOM-{i}') for i in inv[name]['lanes']]
            else:rows=inv[name]['entries']
            self.lengths.append(len(rows))
            for ordinal,row in enumerate(rows):
                if group<self.fresh_groups:state=env._reset_jump_start_unified(jax.random.PRNGKey(101+ordinal),initial_qpos=jp.asarray(row['qpos']),initial_qvel=jp.asarray(row['qvel']))
                else:
                    if self.group_names==RECOVERY_GROUPS:
                        original_snapshot_time(row)
                    snap=load_unified_envelope_snapshot(Path(row['snapshot']))
                    if snapshot_context_sha256(snap)!=row['snapshot_context_sha256'] or snap.compatibility_identity!=compatibility_identity(env):raise ValueError('snapshot pool context mismatch')
                    state=restore_unified_envelope_snapshot(snap,env)
                    state=state.replace(data=state.data.replace(time=jp.asarray(row['snapshot_time'],jp.float32)))
                    np.testing.assert_array_equal(state.obs['state'],snap.observation)
                    np.testing.assert_array_equal(state.info['history'].frames,snap.observation_fifo)
                if self.group_names==RECOVERY_GROUPS:
                    state=state.replace(info={**state.info,'reset_from_jump_start':jp.asarray(group==0),
                        'reset_from_soft_tube':jp.asarray(group!=0)})
                    pulse=getattr(env,'_training_action_pulse',None)
                    if pulse is not None:
                        from .iterative_probe_training import initialize_training_action_pulse,apply_training_action_pulse
                        info=initialize_training_action_pulse(state.info,jax.random.PRNGKey(101+ordinal),pulse,full_task_start=group==0)
                        _,_,metrics=apply_training_action_pulse(info,jp.zeros(4),pulse)
                        state=state.replace(info=info,metrics={**state.metrics,**{k:jp.zeros_like(v) for k,v in metrics.items()}})
                self.entries.append(dict(group=group,name=name,row=ordinal,ancestor=row.get('root_episode_id',row.get('ancestor')),snapshot=row.get('snapshot')));states.append(state)
        # Only administrative reset flags/metrics may differ; no history/event defaults.
        keys=set().union(*(s.info.keys() for s in states));mkeys=set().union(*(s.metrics.keys() for s in states))
        def align(state,group,index):
            info=dict(state.info);metrics=dict(state.metrics)
            # Natural resets retain legacy uint32[2] keys; snapshot restoration
            # returns scalar typed keys. Normalize representation before stacking,
            # preserving the exact key bits and PRNG implementation of both.
            rng=info['rng']
            info['rng']=jax.random.wrap_key_data(jax.random.key_data(rng),impl=jax.random.key_impl(rng))
            for key in keys-set(info):
                if not key.startswith('reset_from_'):raise ValueError('nonadministrative reset info mismatch '+key)
                info[key]=jp.asarray(False)
            for key in mkeys-set(metrics):metrics[key]=jp.asarray(0.,jp.float32)
            info.update(retention_reset_group=jp.asarray(group,jp.int32),retention_reset_index=jp.asarray(index,jp.int32))
            metrics['retention/reset_group']=jp.asarray(group,jp.float32)
            return state.replace(info=info,metrics=metrics)
        normalized=[align(s,e['group'],i) for i,(s,e) in enumerate(zip(states,self.entries))]
        self.pool=stack_worlds(normalized);self.starts=jp.asarray(self.starts);self.lengths=jp.asarray(self.lengths)
    def reset_group(self,key,group):
        import jax,jax.numpy as jp
        from .continuation.device_rollout import take_world
        index=self.starts[group]+jax.random.randint(jax.random.fold_in(key,1),(),0,self.lengths[group]);state=take_world(self.pool,index)
        return state.replace(info={**state.info,'rng':jax.lax.cond(group<getattr(self,'fresh_groups',2),lambda _:jax.random.wrap_key_data(jax.random.key_data(key)),lambda _:state.info['rng'],None)})
    def reset(self,key):
        state=self.reset_group(key,group_indices(key,1,probabilities=getattr(self,'probabilities',None))[0])
        pulse=getattr(self.env,'_training_action_pulse',None)
        if pulse is not None and getattr(self,'group_names',GROUPS)==RECOVERY_GROUPS:
            from .iterative_probe_training import initialize_training_action_pulse
            state=state.replace(info=initialize_training_action_pulse(state.info,key,pulse,
                full_task_start=state.info['retention_reset_group']==0))
        return state
    def step(self,state,action):
        advanced=self.original_step(state,action)
        return advanced.replace(info={**advanced.info,'retention_reset_group':state.info['retention_reset_group'],'retention_reset_index':state.info['retention_reset_index']},metrics={**advanced.metrics,'retention/reset_group':state.info['retention_reset_group'].astype(__import__('jax').numpy.float32)})
    def install(self):self.env.reset=self.reset;self.env.step=self.step;return self.env


def install_declared_student_reset(env,raw):
    """Replace reset sampling only when the inventory has an immutable input lock."""
    path=raw.get('student_reset_inventory')
    if path is None:return env
    if raw.get('input_files',{}).get(path)!=sha(path):raise ValueError('student reset inventory input lock missing/drift')
    inventory=read(path)
    if inventory.get('schema')!='jit_teacher_recovery_reset_v1':raise ValueError('student recovery reset schema required')
    adapter=FourPoolReset(env,inventory)
    adapter.install()
    env._student_reset_adapter=adapter
    return env
