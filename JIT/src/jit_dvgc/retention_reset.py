"""Four equal TRAIN reset pools, preserving complete saved closed-loop context."""
from pathlib import Path
import numpy as np
from .retention_repair import read,sha

GROUPS=('nominal_complete','random_complete','pi0_success_snapshots','teacher_recoverable')


def validate_inventory(inventory):
    if inventory.get('role')!='TRAIN':raise ValueError('four reset pools must be TRAIN')
    for name in GROUPS:
        p=inventory[name]
        if p['probability']!=.25:raise ValueError('fixed equal reset probabilities required')
        if name.endswith('snapshots') or name=='teacher_recoverable':
            if not p['entries'] or any(x['role']!='TRAIN' for x in p['entries']):raise ValueError('empty/non-TRAIN snapshot pool')
        if name=='random_complete' and not p['lanes']:raise ValueError('empty random pool')
    return inventory


def group_indices(key,n):
    import jax
    return jax.random.randint(key,(n,),0,4)


class FourPoolReset:
    def __init__(self,env,inventory):
        import jax,jax.numpy as jp
        from .continuation.device_rollout import stack_worlds
        from .unified_envelope_snapshot import load_unified_envelope_snapshot,restore_unified_envelope_snapshot,snapshot_context_sha256
        from .handoff_snapshot import compatibility_identity
        inv=validate_inventory(inventory);self.env=env;self.original_step=env.step;self.starts=[];self.lengths=[];self.entries=[];states=[]
        for group,name in enumerate(GROUPS):
            self.starts.append(len(states));rows=[]
            if group==0:rows=[dict(qpos=inv[name]['qpos'],qvel=inv[name]['qvel'],ancestor='TRAIN_NOMINAL')]
            elif group==1:
                if sha(inv[name]['path'])!=inv[name]['sha256']:raise ValueError('random pool changed')
                with np.load(inv[name]['path']) as f:rows=[dict(qpos=f['qpos'][i],qvel=f['qvel'][i],ancestor=f'TRAIN_RANDOM-{i}') for i in inv[name]['lanes']]
            else:rows=inv[name]['entries']
            self.lengths.append(len(rows))
            for ordinal,row in enumerate(rows):
                if group<2:state=env._reset_jump_start_unified(jax.random.PRNGKey(101+ordinal),initial_qpos=jp.asarray(row['qpos']),initial_qvel=jp.asarray(row['qvel']))
                else:
                    snap=load_unified_envelope_snapshot(Path(row['snapshot']))
                    if snapshot_context_sha256(snap)!=row['snapshot_context_sha256'] or snap.compatibility_identity!=compatibility_identity(env):raise ValueError('snapshot pool context mismatch')
                    state=restore_unified_envelope_snapshot(snap,env)
                    state=state.replace(data=state.data.replace(time=jp.asarray(row['snapshot_time'],jp.float32)))
                    np.testing.assert_array_equal(state.obs['state'],snap.observation)
                    np.testing.assert_array_equal(state.info['history'].frames,snap.observation_fifo)
                self.entries.append(dict(group=group,name=name,row=ordinal,ancestor=row.get('root_episode_id',row.get('ancestor')),snapshot=row.get('snapshot')));states.append(state)
        # Only administrative reset flags/metrics may differ; no history/event defaults.
        keys=set().union(*(s.info.keys() for s in states));mkeys=set().union(*(s.metrics.keys() for s in states))
        def align(state,group,index):
            info=dict(state.info);metrics=dict(state.metrics)
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
        return state.replace(info={**state.info,'rng':jax.lax.cond(group<2,lambda _:jax.random.wrap_key_data(jax.random.key_data(key)),lambda _:state.info['rng'],None)})
    def reset(self,key):
        return self.reset_group(key,group_indices(key,1)[0])
    def step(self,state,action):
        advanced=self.original_step(state,action)
        return advanced.replace(info={**advanced.info,'retention_reset_group':state.info['retention_reset_group'],'retention_reset_index':state.info['retention_reset_index']},metrics={**advanced.metrics,'retention/reset_group':state.info['retention_reset_group'].astype(__import__('jax').numpy.float32)})
    def install(self):self.env.reset=self.reset;self.env.step=self.step;return self.env
