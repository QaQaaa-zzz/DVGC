"""Bounded engineering comparison of eager and fused full-state restoration."""
from pathlib import Path
import json
import time
import hashlib
import jax
import jax.numpy as jp
import numpy as np
from .batch_snapshot_restore import restore_snapshot_batch
from .continuation.device_rollout import prepare_parallel_worlds, _shared_warp
from .unified_envelope_snapshot import load_unified_envelope_snapshot, snapshot_context_sha256


def compare_trees(left, right, *, rtol=1e-6, atol=1e-6):
    if jax.tree.structure(left) != jax.tree.structure(right):
        return {'passed':False, 'structure_mismatch':True}
    reports=[]
    for (path,a),b in zip(jax.tree_util.tree_flatten_with_path(left)[0],jax.tree.leaves(right)):
        if str(a.dtype).startswith('key<'):
            a,b=jax.random.key_data(a),jax.random.key_data(b)
        a,b=np.asarray(a),np.asarray(b)
        exact=np.array_equal(a,b)
        okay=exact or (np.issubdtype(a.dtype,np.floating) and np.allclose(a,b,rtol=rtol,atol=atol,equal_nan=False))
        if not exact:
            reports.append(dict(path=jax.tree_util.keystr(path),passed=bool(okay),
                max_abs=float(np.max(np.abs(a.astype(float)-b.astype(float)))) if a.size else 0.))
    return dict(passed=all(row['passed'] for row in reports),exact=not reports,
                leaf_count=len(jax.tree.leaves(left)),rtol=rtol,atol=atol,nonexact=reports)


def audit_restore_tapes(output, observation_size, qpos_size, qvel_size):
    """Locate first differences only over each pair's common active prefix."""
    out=Path(output);offset=0;slices={}
    fields=dict(observation_before=observation_size,action=4,qpos_after=qpos_size,qvel_after=qvel_size)
    for name,width in fields.items():
        slices[name]=slice(offset,offset+width);offset+=width
    comparisons={}
    for left,right in [('serial','serial_repeat'),('serial','fused'),('fused','fused_repeat')]:
        a,b=np.load(out/(left+'_tape.npz')),np.load(out/(right+'_tape.npz'));rows=[]
        for lane in range(a['tape'].shape[1]):
            n=min(int(a['counts'][lane]),int(b['counts'][lane]));row=dict(lane=lane,common_active_steps=n)
            for name,sl in slices.items():
                diff=np.abs(a['tape'][:n,lane,sl]-b['tape'][:n,lane,sl])
                ticks=np.flatnonzero(np.any(diff>0,axis=-1));first=int(ticks[0]) if len(ticks) else None
                row[name]=dict(first_nonexact_control_tick=first,
                    max_abs_at_first=float(diff[first].max()) if first is not None else 0.,
                    max_abs_common_active=float(diff.max()) if diff.size else 0.)
            rows.append(row)
        comparisons[left+'_vs_'+right]=rows
    result=dict(schema='jit_restore_first_divergence_audit_v1',zero_based_control_ticks=True,
        fields=fields,comparisons=comparisons,
        tape_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*_tape.npz')})
    (out/'first_divergence.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    return result


def validate_restore(spec_path, candidates_path, output, count=8, timing_counts=()):
    from .exploration_continuation import FrozenSuffixEvaluator
    from .probe_bank import load_probe_bank
    from .pulse_exploration_runtime import endpoint_state, endpoint_success, recovery_mode, suffix_label
    if not 1 <= count <= 8:
        raise ValueError('initial suffix parity panel must contain 1 to 8 candidates')
    if any(n not in (256,1024) for n in timing_counts):
        raise ValueError('expanded restore timing is bounded to 256/1024 candidates')
    if jax.default_backend()!='gpu':
        raise RuntimeError('actual GPU required')
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    def write(name,value):
        (out/name).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
    spec=json.loads(Path(spec_path).read_text());rows=json.loads(Path(candidates_path).read_text())
    rows=[r for r in rows if not r.get('prefix_terminal',False)]
    indices=np.linspace(0,len(rows)-1,count,dtype=int).tolist()
    selected=[rows[i] for i in indices]
    horizon=int(spec['horizon']);budget=4*count*horizon
    write('declaration.json',dict(role='engineering_TRAIN_snapshot_restore_parity',max_interactions=budget,
        snapshots=[r['snapshot'] for r in selected],indices=indices,horizon=horizon,
        phases=[r['phase'] for r in selected],
        source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__),Path(__file__).with_name('batch_snapshot_restore.py'),Path(__file__).with_name('unified_envelope_snapshot.py')]},
        restore_timing_counts=list(timing_counts),restore_integrations=0,training_transitions=0,
        spec_sha256=hashlib.sha256(Path(spec_path).read_bytes()).hexdigest(),
        candidates_sha256=hashlib.sha256(Path(candidates_path).read_bytes()).hexdigest()))
    bank=load_probe_bank(Path(spec['bank']));names=[m['name'] for m in bank['members'] if 'evaluator' in m['roles']]
    evaluator=FrozenSuffixEvaluator(spec['bank'],names,horizon,out/'runtime',budget)
    env,policy,_=evaluator._runtime(spec['order'][0])
    env._reward_mode=spec.get('reward_mode',getattr(env,'_reward_mode','phase_recovery'))
    snapshots=[]
    for row in selected:
        snap=load_unified_envelope_snapshot(Path(row['snapshot']))
        if snapshot_context_sha256(snap)!=row['snapshot_context_sha256']:raise ValueError('context drift')
        snapshots.append(snap)
    timings={};states={}
    for backend in ('serial','fused'):
        start=time.monotonic()
        states[backend]=restore_snapshot_batch(snapshots,env,backend=backend)
        jax.block_until_ready(states[backend]);timings[backend]=time.monotonic()-start
        print(backend,'restore seconds',timings[backend],flush=True)
    parity=compare_trees(jax.device_get(states['serial']),jax.device_get(states['fused']))
    write('state_parity.json',parity);write('restore_timings.json',timings)
    if not parity['passed']:raise ValueError('restored state parity failed; suffix not dispatched')
    def prepare(s):
        if recovery_mode(spec):
            s=s.replace(info={**s.info,'down_events':s.info['down_events'].replace(
                post_contact_ticks=jp.zeros(count,jp.int32),recovery_success=jp.zeros(count,bool))})
        return prepare_parallel_worlds(s,env,count)
    def rollout(initial):
        def condition(c):return (c[0]<horizon)&jp.any(c[2])
        def advance(c):
            tick,s,alive,counts,tape=c
            keys=jax.random.split(jax.random.fold_in(jax.random.PRNGKey(0),tick),count)
            actions=jax.vmap(policy)(s.obs,keys)[0]
            nxt=jax.vmap(lambda st,a:endpoint_state(env.step(st,a),spec))(s,jp.where(alive[:,None],actions,0))
            def choose(path,n,o):
                return n if _shared_warp(path) else jp.where(alive.reshape((count,)+(1,)*(n.ndim-1)),n,o)
            nxt=jax.tree_util.tree_map_with_path(choose,nxt,s)
            finite=jp.all(jp.isfinite(nxt.data.qpos),axis=-1)&jp.all(jp.isfinite(nxt.data.qvel),axis=-1)
            frame=jp.concatenate((s.obs['state'],actions,nxt.data.qpos,nxt.data.qvel),axis=-1)
            tape=tape.at[tick].set(frame)
            return tick+1,nxt,alive&~nxt.done.astype(bool)&~endpoint_success(nxt,spec)&finite,counts+alive.astype(jp.int32),tape
        width=initial.obs['state'].shape[-1]+4+initial.data.qpos.shape[-1]+initial.data.qvel.shape[-1]
        tick,final,alive,counts,tape=jax.lax.while_loop(condition,advance,
            (jp.array(0),initial,jp.ones(count,bool),jp.zeros(count,jp.int32),jp.zeros((horizon,count,width))))
        return tick,final,counts,tape
    run=jax.jit(rollout);results={};charged=0;endpoints={}
    for backend in ('serial','serial_repeat','fused','fused_repeat'):
        write('status.json',dict(phase='running_suffix',backend=backend,charged_interactions=charged,
            reserved_attempt_interactions=count*horizon))
        # Reconstruct and allocate immediately before each call: no reuse of
        # scratch that a prior Warp rollout could have mutated.
        initial=prepare(restore_snapshot_batch(snapshots,env,backend=backend.removesuffix('_repeat')))
        jax.block_until_ready(initial)
        start=time.monotonic();tick,final,counts,tape=jax.device_get(run(initial));charged+=int(tick)*count
        np.savez_compressed(out/(backend+'_tape.npz'),tape=tape[:int(tick)],counts=counts)

        timings[backend+'_compile_rollout']=time.monotonic()-start
        results[backend]=(final,counts)
        endpoints[backend]=[dict(steps=int(counts[i]),end_code=int(final.info['end_code'][i]),
            outcome=suffix_label(bool(endpoint_success(final,spec)[i]),bool(final.info['physical_failure'][i]),
                bool(final.info['timeout'][i]),bool(final.done[i]),int(counts[i])>=horizon)) for i in range(count)]
        if recovery_mode(spec):
            for endpoint in endpoints[backend]:
                if endpoint['outcome'][0]==1:endpoint['outcome']=(1,'stable_forward_recovery')
        write('status.json',dict(phase='suffix_complete',backend=backend,charged_interactions=charged))
        print(backend,'suffix endpoints',endpoints[backend],flush=True)
    suffix_parity=dict(serial_repeat_numerics=compare_trees(results['serial'],results['serial_repeat']),
        fused_repeat_numerics=compare_trees(results['fused'],results['fused_repeat']),serial_repeat_equal=endpoints['serial']==endpoints['serial_repeat'],
        fused_repeat_equal=endpoints['fused']==endpoints['fused_repeat'],endpoints_equal=endpoints['serial']==endpoints['fused'],endpoints=endpoints,
        final_state=compare_trees(results['serial'],results['fused']))
    write('suffix_parity.json',suffix_parity)
    write('restore_timings.json',timings)
    audit_restore_tapes(out, snapshots[0].observation.size, snapshots[0].qpos.size, snapshots[0].qvel.size)
    if not all((suffix_parity['endpoints_equal'], suffix_parity['final_state']['passed'],
                suffix_parity['serial_repeat_equal'], suffix_parity['fused_repeat_equal'],
                suffix_parity['serial_repeat_numerics']['passed'], suffix_parity['fused_repeat_numerics']['passed'])):
        write('status.json',dict(phase='error',charged_interactions=charged,error='suffix numerical or endpoint parity failed',
            serial_repeat_equal=suffix_parity['serial_repeat_equal'],fused_repeat_equal=suffix_parity['fused_repeat_equal']))
        raise ValueError('suffix numerical or endpoint parity failed')
    expanded=[]
    for n in timing_counts:
        if n>len(rows):raise ValueError('not enough distinct candidate rows for requested timing')
        selected_large=[rows[i] for i in np.linspace(0,len(rows)-1,n,dtype=int)]
        start=time.monotonic();snaps=[]
        for row in selected_large:
            snap=load_unified_envelope_snapshot(Path(row['snapshot']))
            if snapshot_context_sha256(snap)!=row['snapshot_context_sha256']:raise ValueError('context drift')
            snaps.append(snap)
        entry=dict(count=n,load_seconds=time.monotonic()-start);large={}
        for backend in ('serial','fused'):
            start=time.monotonic();large[backend]=restore_snapshot_batch(snaps,env,backend=backend)
            jax.block_until_ready(large[backend]);entry[backend+'_seconds']=time.monotonic()-start
            print(n,backend,entry[backend+'_seconds'],flush=True)
        entry['parity']=compare_trees(jax.device_get(large['serial']),jax.device_get(large['fused']))
        expanded.append(entry);write('expanded_timings.json',expanded)
        if not entry['parity']['passed']:raise ValueError('expanded restore parity failed')
        del large
    write('restore_timings.json',timings)
    result=dict(phase='completed',charged_interactions=charged,
        active_interactions=sum(int(np.sum(x[1])) for x in results.values()),
        state_parity=parity['passed'],suffix_parity=True,expanded_counts=list(timing_counts),
        limitation='single policy and bounded candidate panel; timing shares GPU with unrelated job')
    write('status.json',result)
    return result
