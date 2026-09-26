"""TRAIN-only, phase-balanced states and same-state two-teacher full-task labels.

No teacher switching occurs inside a continuation. Imported development traces
become training data; they must never subsequently be reported as held-out tests.
"""
from pathlib import Path
from collections import Counter
import argparse
import json
import numpy as np

STAGES = ('preparation', 'ascent', 'descent', 'recovery')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False)+'\n')


def choose_teachers(success, returns):
    """Success first, identical reward return second; stable index tie-breaking."""
    success, returns = np.asarray(success, bool), np.asarray(returns, float)
    if success.shape != returns.shape or success.ndim != 2 or not np.isfinite(returns).all():
        raise ValueError('finite teacher by state arrays required')
    winners = np.argmax(np.where(success, returns, -np.inf), axis=0)
    return np.where(success.any(axis=0), winners, -1)


def stage_masks(tape, require_source_success=True):
    eligible = np.asarray(tape['prefix_mask'] if 'prefix_mask' in tape else tape['mask'], bool) & ~np.asarray(tape['snap/info/done'], bool)
    if require_source_success:
        eligible &= np.asarray(tape['success'], bool).any(axis=0)[None, :]
    ascending = np.asarray(tape['snap/up/ascending_seen'], bool)
    contact = np.asarray(tape['snap/down/valid_contact_seen'], bool)
    vz = np.asarray(tape['qvel'])[..., 2]
    return dict(preparation=eligible & ~ascending & ~contact,
                ascent=eligible & ascending & ~contact & (vz >= 0),
                descent=eligible & ascending & ~contact & (vz < 0),
                recovery=eligible & contact)


def runtime(template, reward_mode):
    from .pulse_exploration_runtime import networks
    from .ppo import make_checkpoint_policy
    from .probe_bank import load_probe_bank
    t = dict(template, reward_mode=reward_mode)
    env, payload, member, *_ = networks(t)
    # Historical phase payload record lacks formal environment provenance.
    base = next(m for m in load_probe_bank(Path(t['bank']))['members'] if m['name']==t['proposer'])
    record = {**base['policy'], **member['policy']}
    return env, make_checkpoint_policy(env, payload, deterministic=True), record


def prepare(spec, output):
    """spec.source_traces=[{source,path,template?}], teachers=[{name,template}]."""
    from .exploration_continuation import snapshot_from_arrays
    from .unified_envelope_snapshot import save_unified_envelope_snapshot, snapshot_context_sha256, physical_state_sha256
    from .evidence_integrity import canonical_sha256
    from .probe_bank import _file_sha
    output=Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    rng=np.random.default_rng(spec['seed']); quota=int(spec['states_per_teacher_stage'])
    if quota<=0: raise ValueError('positive state quota required')
    teachers={t['name']:t['template'] for t in spec['teachers']}
    sources=spec['source_traces']; pool={}; source_templates={}
    # Reservoir keys yield uniform state sampling in each source/stage stratum.
    # Limit one state per episode/stage to prevent long recoveries dominating.
    for s in sources:
        source=s.get('source',s.get('teacher')); template=s.get('template',teachers.get(source))
        if template is None: raise ValueError('source requires a runtime template')
        source_templates[source]=template
        with np.load(s['path']) as tape:
            for stage, mask in stage_masks(tape,spec.get('require_source_success',True)).items():
                group=pool.setdefault((source,stage),[])
                for lane in np.flatnonzero(mask.any(axis=0)):
                    tick=int(rng.choice(np.flatnonzero(mask[:,lane])))
                    group.append((float(rng.random()),str(Path(s['path']).resolve()),tick,int(lane)))
                group.sort(); del group[quota:]
    missing={str(k):len(v) for k,v in pool.items() if len(v)<quota}
    if missing or len(pool)!=len(source_templates)*4: raise ValueError('insufficient phase-balanced states: '+str(missing))
    selected=[]
    for (source,stage), rows in sorted(pool.items()):
        selected.extend(dict(source=source,stage=stage,trace=p,tick=t,lane=l) for _,p,t,l in rows)
    trace_hashes={p:_file_sha(Path(p)) for p in sorted({r['trace'] for r in selected})}
    entries=[]
    for source, template in source_templates.items():
        env,_,record=runtime(template,spec.get('reward_mode','phase_recovery'))
        source_rows=[r for r in selected if r['source']==source]
        for path in sorted({r['trace'] for r in source_rows}):
            with np.load(path) as tape:
                cached={k[5:]:tape[k] for k in tape.files if k.startswith('snap/')}
                for row in [r for r in source_rows if r['trace']==path]:
                    arrays={k:v[row['tick'],row['lane']] for k,v in cached.items()}
                    snap=snapshot_from_arrays(arrays,env=env,record=record,parent_trajectory=path+'::'+str(row['lane']),parent_state_sha256=trace_hashes[path])
                    destination=output/'snapshots'/f'{len(entries):05d}'
                    save_unified_envelope_snapshot(destination,snap)
                    entries.append(dict(row,index=len(entries),snapshot=str(destination),snapshot_context_sha256=snapshot_context_sha256(snap),state_sha256=physical_state_sha256(snap),phase='upstream' if snap.active_phase==0 else 'downstream'))
    write(output/'candidates.json',entries)
    write(output/'manifest.json',dict(schema='jit_teacher_fusion_states_v1',role='train',final_test_used=False,count=len(entries),source_locks=trace_hashes,spec=spec,continuation_semantics='fresh_continuation_v1 preserving physics, actor FIFO, and event context',counts=dict(Counter(r['source']+'/'+r['stage'] for r in entries))))
    return entries


def evaluate(spec, output):
    """Evaluate every teacher on every candidate; no source-teacher shortcut."""
    import jax
    import jax.numpy as jp
    from .unified_envelope_snapshot import load_unified_envelope_snapshot
    from .unified_continuation_labels import fresh_unified_continuation_start
    from .continuation.device_rollout import stack_worlds,prepare_parallel_worlds,_shared_warp
    from .iterative_probe_training import SUPPORT_SCHEMA
    from .evidence_integrity import canonical_sha256
    if jax.default_backend()!='gpu': raise RuntimeError('GPU required for teacher continuation evaluation')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    rows=read(spec['candidates']); count=len(rows); horizon=int(spec.get('horizon',400))
    if not rows or horizon<=0: raise ValueError('nonempty candidates and positive horizon required')
    teachers=spec['teachers']
    if not teachers or len({t['name'] for t in teachers})!=len(teachers): raise ValueError('unique nonempty teacher names required')
    if len({r['snapshot_context_sha256'] for r in rows})!=count: raise ValueError('duplicate candidate contexts')
    maximum=count*horizon*len(teachers)
    if maximum>spec['budget']: raise ValueError('teacher labeling exceeds declared budget')
    scores=[]; tapes=[]; charged=0; runtime_identity=None
    for teacher in teachers:
        env,policy,record=runtime(teacher['template'],spec.get('reward_mode','phase_recovery'))
        identity=(env._resolved_config.reward, env._down_config.reward, env._resolved_config.physical_limits, env._down_config.descent)
        if runtime_identity is not None and identity != runtime_identity: raise ValueError('teachers must use identical reward and endpoint contracts')
        runtime_identity=identity
        states=[fresh_unified_continuation_start(load_unified_envelope_snapshot(Path(r['snapshot'])),env) for r in rows]
        states=[s.replace(info={**s.info,'down_events':s.info['down_events'].replace(post_contact_ticks=jp.asarray(0,jp.int32),recovery_success=jp.asarray(False))}) for s in states]
        initial=prepare_parallel_worlds(stack_worlds(states),env,count);del states
        step=jax.vmap(env.step)
        def rollout(initial):
            def advance(carry,t):
                state,alive=carry
                keys=jax.random.split(jax.random.fold_in(jax.random.PRNGKey(spec['seed']),t),count)
                action=jax.vmap(policy)(state.obs,keys)[0]
                nxt=step(state,jp.where(alive[:,None],action,0))
                finite=jp.all(jp.isfinite(nxt.data.qpos),-1)&jp.all(jp.isfinite(nxt.data.qvel),-1)&jp.isfinite(nxt.reward)&jp.all(jp.isfinite(action),-1)&jp.all(jp.isfinite(state.obs['state']),-1)
                frame=dict(obs=state.obs['state'],privileged_obs=state.obs['privileged_state'],action=action,mask=alive,reward=nxt.reward,success=nxt.info['success']&~nxt.info['physical_failure']&finite,physical_failure=nxt.info['physical_failure'],end_code=nxt.info['end_code'],finite=finite)
                def choose(path,n,o):return n if _shared_warp(path) else jp.where(alive.reshape((count,)+(1,)*(n.ndim-1)),n,o)
                nxt=jax.tree_util.tree_map_with_path(choose,nxt,state)
                return (nxt,alive&~nxt.done.astype(bool)&~nxt.info['success']&finite),frame
            return jax.lax.scan(advance,(initial,jp.ones(count,bool)),jp.arange(horizon))[1]
        write(output/'status.json',dict(phase='running',teacher=teacher['name'],charged_interactions=charged,reserved_interactions=maximum))
        tape=jax.device_get(jax.jit(rollout)(initial)); charged+=count*horizon
        mask=tape['mask']; success=np.any(tape['success']&mask,axis=0)
        returns=np.where(mask,tape['reward'],0).sum(axis=0)
        finite=np.all(tape['finite']|~mask,axis=0)
        if not finite.all(): raise ValueError('nonfinite teacher continuation; not a negative label')
        path=output/(teacher['name']+'_continuations.npz');np.savez_compressed(path,**tape)
        scores.append(dict(name=teacher['name'],success=success.tolist(),returns=returns.tolist(),steps=mask.sum(axis=0).tolist(),record=record,trace=str(path)))
        tapes.append(tape)
        del initial,env,policy,step
        jax.clear_caches()
    winners=choose_teachers([s['success'] for s in scores],[s['returns'] for s in scores])
    selected=[]; dataset={k:[] for k in ('obs','privileged_obs','action','weight','state_index','teacher_index')}
    group_counts=Counter((r['source'],r['stage']) for i,r in enumerate(rows) if winners[i]>=0)
    for i,winner in enumerate(winners):
        if winner<0:continue
        row=rows[i];teacher=scores[winner];tape=tapes[winner];valid=tape['mask'][:,i];n=int(valid.sum())
        for k in ('obs','privileged_obs','action'):dataset[k].append(tape[k][:,i][valid])
        dataset['weight'].append(np.full(n,1./(n*group_counts[row['source'],row['stage']]),np.float32))
        dataset['state_index'].append(np.full(n,i,np.int32));dataset['teacher_index'].append(np.full(n,winner,np.int32))
        selected.append(dict(key=row['snapshot_context_sha256'],phase=row['phase'],snapshot=row['snapshot'],state_sha256=row['state_sha256'],snapshot_context_sha256=row['snapshot_context_sha256'],trajectory_id=row['trace']+'::'+str(row['lane']),root_cell=[row['source'],row['stage']],witnessed=True,evidence_status='witnessed',role='train',sampling_weight=1./group_counts[row['source'],row['stage']],witness=teacher['name'],witness_trace=teacher['trace'],witness_trace_lane=i,stage=row['stage']))
    if not selected:raise ValueError('neither teacher succeeds from any candidate')
    arrays={k:np.concatenate(v) for k,v in dataset.items()}
    arrays.update(observations=arrays.pop('obs'),actions=arrays.pop('action'),weights=arrays.pop('weight'))
    np.savez_compressed(output/'imitation.npz',**arrays)
    from .probe_bank import _file_sha
    inputs={str(Path(r['snapshot'])/name):_file_sha(Path(r['snapshot'])/name) for r in selected for name in ('identity.json','snapshot.pkl')}
    support=dict(schema=SUPPORT_SCHEMA,role='train',final_test_used=False,entries=selected,inputs=inputs,selection='same-state complete teacher success, common reward tie-break',phase_counts=dict(Counter(r['phase'] for r in selected)),coordinate_translation_applied=False)
    support['support_sha256']=canonical_sha256(support);write(output/'support.json',support)
    write(output/'teacher_scores.json',dict(teachers=scores,winners=winners.tolist(),candidates=rows,reward_mode=spec.get('reward_mode','phase_recovery')))
    write(output/'status.json',dict(phase='completed',charged_interactions=charged,active_interactions=sum(int(t['mask'].sum()) for t in tapes),candidates=count,witnessed=len(selected),imitation_rows=len(arrays['observations']),winners=dict(Counter(scores[w]['name'] for w in winners if w>=0)),excluded_both_fail=int(np.sum(winners<0)),dataset=str(output/'imitation.npz'),support=str(output/'support.json')))
    return read(output/'status.json')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','evaluate']);parser.add_argument('--spec',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();globals()[args.command](read(args.spec),args.output)


if __name__=='__main__':main()
