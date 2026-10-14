"""Bounded real student visitation and same-context dual-teacher consolidation."""
import json,pickle,time
from pathlib import Path
import numpy as np
from .generative_bridge.contracts import digest,file_sha
from .generative_bridge.protocol import atomic_json

def run_student_block(config,output,recovery_inference,*,updates=2000,visit_steps=4,seed=0):
    if type(visit_steps) is not int or not 0<=visit_steps<400:raise ValueError('bounded predeclared visitation steps required')
    import jax,jax.numpy as jp
    from tensorboardX import SummaryWriter
    from brax.training.acme import running_statistics as rs
    from .unified_formal import build_unified_formal_environment
    from .generative_bridge.worker import source_payload
    from .ppo import make_network_factory
    from .retention_reset import FourPoolReset
    from .recovery_curriculum import resolve_teacher_sources
    from .continuation.device_rollout import stack_worlds,take_world,prepare_parallel_worlds,_shared_warp
    from .unified_envelope_snapshot import capture_unified_envelope_snapshot,save_unified_envelope_snapshot
    from .handoff_bank import pytree_sha256
    from .skill_transfer_dual_teacher import label_verified_query,consolidation_weights,observation_conflicts,SCHEMA
    from .recovery_acquisition_bc import warmup_recovery_actor
    c=json.loads(Path(config).read_text());out=Path(output);out.mkdir(parents=True,exist_ok=False)
    from .skill_transfer import audit
    audit(c)
    if jax.default_backend()!='gpu':raise RuntimeError('production S queries require GPU')
    cfg,artifact,env=build_unified_formal_environment(c['runtime_config']['path']);env._training_action_pulse=None
    policy,payload=source_payload(c['pi0_descriptor']['path'])
    pi0=(payload.observation_normalizer,payload.actor_params,payload.critic_params)
    student=pi0
    if c.get('student_initializer'):
        with Path(c['student_initializer']).open('rb') as f:student=pickle.load(f)
        if pytree_sha256(student[0])!=pytree_sha256(pi0[0]):raise ValueError('student normalizer differs')
    with Path(recovery_inference).open('rb') as f:recovery=pickle.load(f)
    if pytree_sha256(recovery[0])!=pytree_sha256(pi0[0]):raise ValueError('teacher normalizer mismatch')
    teachers={name:dict(actor_sha256=pytree_sha256(params[1]),normalizer_sha256=pytree_sha256(params[0])) for name,params in [('pi0',pi0),('R',recovery)]}
    def source_path(key):
        ref=c[key]
        if file_sha(ref['path'])!=ref['sha256']:raise ValueError('student source drift: '+key)
        return Path(ref['path'])
    summary=json.loads(source_path('recovery_summary').read_text())
    if summary['params_sha256']!=file_sha(recovery_inference):raise ValueError('R proof does not bind supplied inference')
    sources=resolve_teacher_sources(c['teacher_manifest']['path'],c['source_rows']['path'])
    roots=[row for _,row in sources if summary['per_root'][row['root_id']]['stable_label']==1]
    inv_path=source_path('student_reset_inventory')
    inv=json.loads(Path(inv_path).read_text());inv['teacher_recoverable']['entries']=roots
    adapter=FourPoolReset(env,inv)
    # Initial visitation is exactly half new / half old. Category balancing is
    # explicit in labels; finite seven-old draw uses 3/2/2 counts.
    choices=[(3,i) for i in range(len(roots))]+[(i%3,i//3) for i in range(len(roots))]
    indices=[int(adapter.starts[g])+i%int(adapter.lengths[g]) for g,i in choices]
    states=prepare_parallel_worlds(stack_worlds([take_world(adapter.pool,i) for i in indices]),env,len(indices))
    # The pool's administrative metric is not emitted by the native env step.
    # Remove it before a zero-prefix branch so JAX carry structure is stable.
    states=states.replace(metrics={k:v for k,v in states.metrics.items() if k!='retention/reset_group'})
    network=make_network_factory()({'state':76,'privileged_state':106},4,preprocess_observations_fn=rs.normalize)
    def action(actor,obs):return network.parametric_action_distribution.mode(network.policy_network.apply(pi0[0],actor,{'state':obs}))
    student_action=jax.jit(lambda s:action(student[1],s.obs['state']))
    step=jax.jit(jax.vmap(env.step));visit_trace=[]
    atomic_json(out/'declaration.json',dict(schema='frontierbridge_S_block_v1',config=c,recovery_inference=str(recovery_inference),recovery_sha256=file_sha(recovery_inference),teachers=teachers,full_train_root_denominator=8,acquired_root_ids=[r['root_id'] for r in roots],visit_steps=visit_steps,initial_new_old_mass=[.5,.5],old_category_draw_counts=[3,2,2],teacher_repeats=3,student_initializer=c.get('student_initializer','pi0'),student_actor_sha256=pytree_sha256(student[1]),fresh_BC_optimizer=True,updates=updates,final_TEST_opened=False,automatic_adoption=False))
    for tick in range(visit_steps):
        a=student_action(states);visit_trace.append(dict(observation=np.asarray(states.obs['state']),action=np.asarray(a),qpos=np.asarray(states.data.qpos)))
        states=step(states,a);jax.block_until_ready(states)
        atomic_json(out/'status.json',dict(stage='S_visitation',state='running',completed_control_steps=tick+1,declared_control_steps=visit_steps))
    np.savez(out/'student_visitation.npz',observations=np.stack([r['observation'] for r in visit_trace]) if visit_trace else np.asarray(states.obs['state'])[None],actions=np.stack([r['action'] for r in visit_trace]) if visit_trace else np.empty((0,len(indices),4)),qpos=np.stack([r['qpos'] for r in visit_trace]) if visit_trace else np.asarray(states.data.qpos)[None])
    queries=[];query_states=[]
    groups=['old_nominal','old_random','old_recovery','new_recovery']
    for lane,((group,index),poolindex) in enumerate(zip(choices,indices)):
        s=take_world(states,lane)
        if bool(s.done) or bool(s.info['success']):continue
        if not np.isfinite(np.asarray(s.obs['state'])).all():raise FloatingPointError('nonfinite student visit')
        entry=adapter.entries[poolindex];root=roots[index]['root_id'] if group==3 else str(entry['ancestor'])
        sample=f'query_{lane:03d}_step{visit_steps:03d}'
        snap=capture_unified_envelope_snapshot(s,env=env,parent_trajectory=str(out/'student_visitation.npz'),parent_state_sha256=file_sha(out/'student_visitation.npz'),config_sha256=c['runtime_config']['sha256'],policy_actor_sha256=pytree_sha256(student[1]),policy_payload_sha256=file_sha(c['student_initializer']) if c.get('student_initializer') else policy['payload_sha256'],policy_iteration=policy['iteration'],capture_recovery_context=True)
        dest=out/'contexts'/sample;save_unified_envelope_snapshot(dest,snap)
        # Receipt binds every serialized full-context file, not merely qpos.
        receipt=dest/'query_context.json';atomic_json(receipt,dict(files=[dict(path=str(p),sha256=file_sha(p)) for p in sorted(dest.iterdir()) if p.is_file()],episode_step=snap.episode_step,time=snap.snapshot_time))
        q=dict(sample_id=sample,root_id=root,ancestor_root_id=str(entry['ancestor']),student_actor_sha256=pytree_sha256(student[1]),context_sha256=file_sha(receipt),context_receipt=dict(path=str(receipt),sha256=file_sha(receipt)),split='TRAIN',source_type='student_visited_query',student_closed_loop=True,teacher_used_as_driver=False,task_group=groups[group],time_before=snap.snapshot_time,remaining_horizon=400-snap.episode_step,control_step=snap.episode_step,model_sha256=policy['xml_sha256'],protocol_sha256=sources[0][0]['protocol_sha256'],timestamp=time.strftime('%Y-%m-%dT%H:%M:%S'),observation_before=np.asarray(s.obs['state']).tolist())
        queries.append(q);query_states.append(s)
    if not queries:raise RuntimeError('no nonterminal S visited queries')
    count=len(queries);initial=prepare_parallel_worlds(stack_worlds(query_states+query_states),env,2*count)
    maxh=max(q['remaining_horizon'] for q in queries);limits=jp.asarray([q['remaining_horizon'] for q in queries]*2)
    def branch_action(state):
        obs=state.obs['state'];a0=action(pi0[1],obs);ar=action(recovery[1],obs)
        return jp.where((jp.arange(2*count)>=count)[:,None],ar,a0)
    first=np.asarray(jax.jit(branch_action)(initial))
    def rollout(initial):
        def condition(carry):return (carry[0]<maxh)&jp.any(carry[2])
        def body(carry):
            t,s,alive,bad,success,counts=carry;a=branch_action(s)
            nxt=jax.vmap(env.step)(s,jp.where(alive[:,None],a,0))
            finite=jp.all(jp.isfinite(nxt.data.qpos),axis=-1)&jp.all(jp.isfinite(nxt.data.qvel),axis=-1)&jp.all(jp.isfinite(nxt.obs['state']),axis=-1)
            success=success|(alive&nxt.info['success']);bad=bad|(alive&~finite)
            nxt=jax.tree_util.tree_map_with_path(lambda path,n,o:n if _shared_warp(path) else jp.where(alive.reshape((2*count,)+(1,)*(n.ndim-1)),n,o),nxt,s)
            counts=counts+alive.astype(jp.int32);alive=alive&~nxt.done.astype(bool)&~nxt.info['success']&finite&(t+1<limits)
            return t+1,nxt,alive,bad,success,counts
        return jax.lax.while_loop(condition,body,(jp.array(0),initial,jp.ones(2*count,bool),jp.zeros(2*count,bool),jp.zeros(2*count,bool),jp.zeros(2*count,jp.int32)))
    run=jax.jit(rollout);repeat_labels=[];cost=0
    for repeat in range(3):
        atomic_json(out/'status.json',dict(stage='S_teacher_queries',state='running',repeat=repeat,queries=count,charged_interactions=cost))
        tick,final,alive,bad,success,counts=run(initial);jax.block_until_ready(final)
        labels=np.where(np.asarray(bad),-1,np.asarray(success).astype(int));repeat_labels.append(labels.tolist());cost+=int(tick)*2*count
        np.savez(out/f'branches_repeat{repeat}.npz',labels=labels,counts=np.asarray(counts),final_qpos=np.asarray(final.data.qpos),final_qvel=np.asarray(final.data.qvel),final_observation=np.asarray(final.obs['state']),initial_action=first)
    labels=[];outcomes=[]
    for i,q in enumerate(queries):
        refs={}
        for name,offset in [('pi0',0),('R',count)]:
            record={k:q[k] for k in ('context_sha256','root_id','ancestor_root_id','split','time_before','remaining_horizon','model_sha256','protocol_sha256','control_step')};raw=[r[i+offset] for r in repeat_labels]
            record.update(teacher=name,execution_mode='independent_frozen_teacher',success_criterion='stable_forward_recovery',observation_sha256=digest(q['observation_before']),repeat_labels=['execution_error' if v<0 else v for v in raw],normalized_action=first[i+offset].tolist(),**teachers[name])
            p=out/'branches'/f'{q["sample_id"]}_{name}.json';p.parent.mkdir(exist_ok=True);atomic_json(p,record);refs[name]=dict(path=str(p),sha256=file_sha(p))
        label=label_verified_query(q,refs,teachers)
        outcomes.append(dict(query=q,branch_receipts=refs,labeled=label is not None))
        if label is not None:labels.append(label)
    conflicts=observation_conflicts(labels);excluded={s for row in conflicts for s in row['sample_ids']};labels=[r for r in labels if r['sample_id'] not in excluded]
    atomic_json(out/'query_receipts.json',dict(queries=outcomes,conflicts=conflicts,total_queries=count,labeled_queries=len(labels),charged_interactions=cost,student_visitation_interactions=len(indices)*visit_steps))
    atomic_json(out/'labels.json',dict(schema=SCHEMA,teachers=teachers,labels=labels))
    if not labels:raise RuntimeError('all teacher query outcomes abstained; preserved raw evidence')
    weights=consolidation_weights(labels);path=out/'online_labels.npz';np.savez(path,observations=np.asarray([r['observation_before'] for r in labels],np.float32),actions=np.asarray([r['target_action'] for r in labels],np.float32),weights=weights)
    manifest=dict(schema=SCHEMA,path=str(path),sha256=file_sha(path),count=len(labels),sample_roots=[r['root_id'] for r in labels],sample_origins=['student_visited_query']*len(labels),sampling_weights=weights.tolist(),label_receipt=dict(path=str(out/'labels.json'),sha256=file_sha(out/'labels.json')));atomic_json(out/'online_manifest.json',manifest)
    if c.get('previous_online_banks'):
        from .skill_transfer_dual_teacher import merge_verified_labels
        previous=[]
        for ref in c['previous_online_banks']:
            if file_sha(ref['path'])!=ref['sha256']:raise ValueError('past queried bank changed')
            previous.append(ref['path'])
        manifest=merge_verified_labels(previous+[str(out/'labels.json')],out/'aggregate')
    writer=SummaryWriter(str(out/'tensorboard'))
    def metrics(row):
        writer.add_scalar('S_BC/loss',row['loss'],row['update']);writer.add_scalar('S_BC/supervised_reward',-row['loss'],row['update']);writer.flush()
        atomic_json(out/'status.json',dict(stage='S_BC',state='running',completed=row['update'],declared=updates,labeled_queries=len(labels)))
    try:warmup_recovery_actor(network,student,manifest,out/'learner',updates=updates,seed=seed,metrics_callback=metrics)
    finally:writer.close()
    atomic_json(out/'status.json',dict(stage='S_BC',state='completed',completed=updates,declared=updates,labeled_queries=len(labels),full_train_root_denominator=8,automatic_adoption=False))
    return str(out/'learner'/f'update_{updates:04d}.pkl')
