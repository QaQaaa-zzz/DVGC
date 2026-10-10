"""One scoped PPO + optional demo + frozen-source retention loss adapter."""
from pathlib import Path
import numpy as np
from .contracts import file_sha
from ..policy_distillation import load_dataset


def load_optional_demo(manifest):
    if manifest is None:return None
    if manifest.get('schema') not in ('jit_bridge_demo_v1_1','jit_bridge_demo_v1_2'):raise ValueError('demo schema required')
    count=manifest.get('count')
    if type(count) is not int or count<0:raise ValueError('explicit demo count required')
    if count==0:
        if manifest.get('entries',[]) or manifest.get('path') or manifest.get('sha256'):
            raise ValueError('empty demo cannot hide data or a damaged file')
        return None
    if file_sha(manifest['path'])!=manifest['sha256']:raise ValueError('demo hash mismatch')
    data=load_dataset(manifest['path'])
    if len(data[0])!=count or data[0].shape[1]!=76:raise ValueError('demo dimensions/count mismatch')
    return data


def make_joint_student_trainer(trainer, demo_manifest, *, retention,
        transitions=128000, demo_coefficient_start=.2, demo_coefficient_end=.05,
        keep_coefficient=.2, demo_batch_size=256, retention_batch_size=256,
        demo_sampler=None, usage_sink=None, retention_reference=None, audit_enabled=False, max_first_behavior_kl=None,
        first_update_audit_path=None, freeze_actor_normalizer=False, demo_clock="normalizer_count", actual_gradient_audit=False):
    """Must be invoked inside the existing guard_ppo_updates scope.

    Empty data creates no device target, sampler or demo RNG operation. Both
    coefficients zero preserves the original function object, not base+0*NaN.
    """
    if demo_clock not in ('normalizer_count','completed_transitions'):
        raise ValueError('unknown demo clock')
    if freeze_actor_normalizer and demo_clock!='completed_transitions':
        raise ValueError('frozen normalizer requires explicit completed transitions')
    if demo_clock=='completed_transitions' and not freeze_actor_normalizer:
        raise ValueError('explicit clock requires instrumented fixed-normalizer pilot')
    coefficients=(demo_coefficient_start,demo_coefficient_end,keep_coefficient)
    if any(not np.isfinite(v) or v<0 for v in coefficients):raise ValueError('invalid auxiliary coefficient')
    if any(type(v) is not int or v<=0 for v in (transitions,demo_batch_size,retention_batch_size)):
        raise ValueError('positive transition and batch budgets required')
    if max_first_behavior_kl is not None and (not np.isfinite(max_first_behavior_kl) or max_first_behavior_kl<0):
        raise ValueError('finite nonnegative first behavior KL limit required')
    demo=load_optional_demo(demo_manifest)
    usage=usage_sink if usage_sink is not None else {}
    usage.update(demo_samples_by_root={},demo_loss_calls=0)
    if demo is not None:
        sample_roots=demo_manifest.get('sample_roots')
        if not isinstance(sample_roots,list) or len(sample_roots)!=len(demo[0]):
            raise ValueError('per-example root provenance required for demo usage')
    use_demo=demo is not None and (demo_coefficient_start>0 or demo_coefficient_end>0)
    if keep_coefficient:
        if retention is None:raise ValueError('declared keep term requires locked successful anchors')
        obs,weights=map(np.asarray,retention)
        if (obs.ndim!=2 or obs.shape[1]!=76 or len(obs)==0 or not np.isfinite(obs).all()
            or weights.shape!=(len(obs),) or not np.isfinite(weights).all()
            or np.any(weights<0) or weights.sum()<=0):raise ValueError('invalid keep anchors')
        retention=(obs,weights/weights.sum())

    def train(**kwargs):
        if kwargs.get('restore_params') is None:raise ValueError('copy source Actor and normalizer required')
        effective_trainer=trainer
        if freeze_actor_normalizer:
            from .student_clock import instrument_frozen_normalizer
            effective_trainer=instrument_frozen_normalizer(trainer)
        if not use_demo and not keep_coefficient:return effective_trainer(**kwargs)
        import jax
        import jax.numpy as jp
        from brax.training.agents.ppo import losses
        from ..policy_distillation import _count_float
        from ..policy_retention import action_mse
        initializer_normalizer=kwargs['restore_params'][0]
        source_normalizer,source_actor=(kwargs['restore_params'][:2] if retention_reference is None else retention_reference)
        source_normalizer,source_actor=jax.tree.map(jax.lax.stop_gradient,(source_normalizer,source_actor)) if retention_reference is not None else (source_normalizer,source_actor)
        start_count=_count_float(initializer_normalizer.count)
        original=losses.compute_ppo_loss
        if use_demo: demo_obs,demo_targets,demo_probs=map(jp.asarray,demo)
        if keep_coefficient: keep_obs,keep_probs=map(jp.asarray,retention)
        if audit_enabled:
            from .learning_audit import fixed_weighted_probe_indices
            usage['fixed_probe_sampling']='256 or fewer deterministic equal-mass strata of declared weights; telemetry only'
            if use_demo:
                probe_ids=fixed_weighted_probe_indices(demo[2])
                probe_obs,probe_targets=demo_obs[probe_ids],demo_targets[probe_ids]
                probe_probs=jp.ones(len(probe_ids))/len(probe_ids)
                probe_origins=np.asarray(demo_manifest['sample_origins'])[probe_ids]
                usage['fixed_demo_probe_indices']=probe_ids.tolist()
            if keep_coefficient:
                probe_keep_ids=fixed_weighted_probe_indices(retention[1])
                probe_keep_obs=keep_obs[probe_keep_ids]
                probe_keep_probs=jp.ones(len(probe_keep_ids))/len(probe_keep_ids)
                usage['fixed_retention_probe_indices']=probe_keep_ids.tolist()
        sampler=demo_sampler or jax.random.choice
        def record_usage(indices):
            for i in np.asarray(indices).reshape(-1):
                rid=sample_roots[int(i)]
                usage['demo_samples_by_root'][rid]=usage['demo_samples_by_root'].get(rid,0)+1
            usage['demo_loss_calls']+=1

        def record_audit(values):
            row={key:float(value) for key,value in values.items()}
            index=usage.get('learning_audit_loss_calls',0)
            row['executed_loss_index']=index
            usage['learning_audit_loss_calls']=index+1
            if index<32:usage.setdefault('learning_audit_first_losses',[]).append(row)
            usage['learning_audit_latest']=row
            if index==0:
                excess=row['audit/installed_behavior_kl']-row['audit/installed_self_kl']
                finite=all(np.isfinite(v) for v in row.values())
                exceeded=max_first_behavior_kl is not None and (not finite or excess>max_first_behavior_kl)
                evidence=dict(schema='jit_bridge_first_loss_gate_v1_2',metrics={k:v if np.isfinite(v) else None for k,v in row.items()},
                    excess_behavior_kl=excess if np.isfinite(excess) else None,maximum_excess_behavior_kl=max_first_behavior_kl,
                    status='diagnostic_stop' if exceeded else 'passed',
                    installed_normalizer_timing='Brax non-adaptive learning rate updates running normalizer before SGD',
                    accepted_optimizer_updates='unknown; asynchronous callback stop is not a zero-update guarantee')
                usage['first_loss_gate']=evidence
                if first_update_audit_path is not None:
                    from .protocol import atomic_json
                    atomic_json(first_update_audit_path,evidence)
                if exceeded:raise FloatingPointError('first-loss behavior KL exceeds declared limit; diagnostic stop')

        def loss(params,normalizer_params,data,rng,ppo_network,**options):
            base,metrics=original(params,normalizer_params,data,rng,ppo_network,**options)
            apply=ppo_network.policy_network.apply
            mode=ppo_network.parametric_action_distribution.mode
            total=base;demo_mse=jp.asarray(0.);keep_mse=jp.asarray(0.);scale=jp.asarray(0.)
            if use_demo:
                ids=sampler(jax.random.fold_in(rng,173),len(demo_obs),(demo_batch_size,),p=demo_probs)
                demo_ids=ids
                jax.debug.callback(record_usage,ids)
                prediction=mode(apply(normalizer_params,params.policy,{'state':demo_obs[ids]}))
                demo_mse=action_mse(prediction,demo_targets[ids])
                from .student_clock import demo_weight
                clock=(_count_float(normalizer_params.count)-start_count if demo_clock=='normalizer_count'
                    else jp.max(data.extras['policy_extras']['completed_training_transitions']))
                scale=demo_weight(clock,transitions,demo_coefficient_start,demo_coefficient_end)
                total=total+scale*demo_mse
            if keep_coefficient:
                ids=jax.random.choice(jax.random.fold_in(rng,719),len(keep_obs),(retention_batch_size,),p=keep_probs)
                keep_ids=ids
                anchor={'state':keep_obs[ids]}
                teacher=mode(apply(source_normalizer,source_actor,anchor))
                prediction=mode(apply(normalizer_params,params.policy,anchor))
                keep_mse=action_mse(prediction,teacher)
                total=total+keep_coefficient*keep_mse
            if actual_gradient_audit:
                from .actual_update import weighted_actor_metrics
                def ppo_objective(actor):
                    return original(params.replace(policy=actor),normalizer_params,data,rng,ppo_network,**options)[1]['policy_loss']
                def entropy_objective(actor):
                    return original(params.replace(policy=actor),normalizer_params,data,rng,ppo_network,**options)[1]['entropy_loss']
                def demo_objective_actual(actor):
                    if not use_demo:return jp.asarray(0.)
                    pred=mode(apply(normalizer_params,actor,{'state':demo_obs[demo_ids]}))
                    return scale*action_mse(pred,demo_targets[demo_ids])
                def keep_objective_actual(actor):
                    if not keep_coefficient:return jp.asarray(0.)
                    pred=mode(apply(normalizer_params,actor,{'state':keep_obs[keep_ids]}))
                    target=jax.lax.stop_gradient(mode(apply(source_normalizer,source_actor,{'state':keep_obs[keep_ids]})))
                    return keep_coefficient*action_mse(pred,target)
                metrics={**metrics,**weighted_actor_metrics(params.policy,dict(ppo=ppo_objective,entropy=entropy_objective,demo=demo_objective_actual,keep=keep_objective_actual)),
                    'actual/reward_per_transition':jp.mean(data.reward)}
            if audit_enabled:
                from .learning_audit import joint_loss_metrics,critic_loss_metrics
                def policy_objective(actor):
                    return original(params.replace(policy=actor),normalizer_params,data,rng,ppo_network,**options)[1]['policy_loss']
                def value_objective(critic):
                    return original(params.replace(value=critic),normalizer_params,data,rng,ppo_network,**options)[1]['v_loss']
                # Fixed bounded stratified offline probes; they never enter PPO replay.
                def demo_objective(actor):
                    if not use_demo:return jp.asarray(0.)
                    pred=mode(apply(normalizer_params,actor,{'state':probe_obs}))
                    return jp.sum(jp.mean(jp.square(pred-probe_targets),axis=-1)*probe_probs)
                def keep_objective(actor):
                    if not keep_coefficient:return jp.asarray(0.)
                    pred=mode(apply(normalizer_params,actor,{'state':probe_keep_obs}))
                    target=jax.lax.stop_gradient(mode(apply(source_normalizer,source_actor,{'state':probe_keep_obs})))
                    return jp.sum(jp.mean(jp.square(pred-target),axis=-1)*probe_keep_probs)
                metrics={**metrics,**joint_loss_metrics(ppo_network,params.policy,normalizer_params,data,
                    policy_objective,demo_objective,keep_objective,clipping_epsilon=options.get('clipping_epsilon',.3)),
                    **critic_loss_metrics(ppo_network,params.value,normalizer_params,data,value_objective,options=options),
                    'fixed_probe/demo_mse':demo_objective(params.policy),
                    'fixed_probe/keep_mse':keep_objective(params.policy),
                    'audit/normalizer_count_delta_from_initializer':_count_float(normalizer_params.count)-start_count}
                if use_demo:
                    fixed_error=jp.square(mode(apply(normalizer_params,params.policy,{'state':probe_obs}))-probe_targets)
                    for segment in ('bridge_prefix','source_tail'):
                        mask=jp.asarray(probe_origins==segment)
                        count=jp.maximum(mask.sum(),1)
                        for channel in range(4):
                            metrics[f'fixed_probe/{segment}/mse_channel_{channel}']=jp.sum(fixed_error[:,channel]*mask)/count
                jax.debug.callback(record_audit,{key:value for key,value in metrics.items()
                    if key.startswith(('audit/','fixed_probe/'))})
            return total,{**metrics,'ppo_loss':base,'demo_action_mse':demo_mse,
                'effective_lambda_demo':scale,'retention_action_mse':keep_mse,
                'demo_samples_per_loss':jp.asarray(demo_batch_size if use_demo else 0),
                'retention_coefficient':jp.asarray(keep_coefficient),'total_loss':total}
        losses.compute_ppo_loss=loss
        try:
            result=effective_trainer(**kwargs)
            jax.effects_barrier()
            return result
        finally:losses.compute_ppo_loss=original
    return train


def trainer_from_config(trainer, raw, run_dir):
    """Canonical formal entry point, still under the existing full-gradient guard."""
    import json
    if raw.get('continuous_learner') is not None:
        from .learner_continuation import continuing_trainer
        trainer=continuing_trainer(trainer,raw,run_dir)
    from ..policy_retention import load_anchor
    if raw.get('action_retention') is not None:
        raise ValueError('bridge joint loss cannot nest action_retention wrappers')
    if raw.get('initialization',{}).get('actor')!='warm_start_frozen_unified':
        raise ValueError('bridge student must copy source Actor and normalizer')
    contract=raw['generative_bridge_student']
    if contract.get('schema') not in ('jit_bridge_student_v1_1','jit_bridge_student_v1_2'):raise ValueError('student contract schema required')
    if contract.get('schema')=='jit_bridge_student_v1_2' and 'max_first_behavior_kl' not in contract:
        raise ValueError('v1.2 requires predeclared max_first_behavior_kl')
    frozen=Path(raw['initialization']['source_frozen_policy'])
    source=json.loads(frozen.read_text())['policy']
    if contract.get('source_actor_sha256')!=source['actor_sha256'] or contract.get('source_normalizer_sha256')!=source['normalizer_sha256']:
        raise ValueError('student auxiliary source identity drift')
    demo=contract.get('demo_manifest')
    if demo is not None:
        path=Path(demo['path'])
        if file_sha(path)!=demo['sha256']:raise ValueError('demo manifest hash drift')
        demo=json.loads(path.read_text())
    keep=contract.get('retention')
    reference=None
    reference_source=source
    if contract.get('schema')=='jit_bridge_student_v1_2':
        reference,reference_source=load_retention_reference(contract['retention_reference_actor'])
    anchors=load_retention_traces(contract['retention_trace_observations']) if contract.get('retention_trace_observations') is not None else (load_anchor(keep,reference_source['name']) if keep is not None else None)
    usage={}
    wrapped=make_joint_student_trainer(trainer,demo,retention=anchors,usage_sink=usage,
        transitions=raw['ppo']['requested_transitions'],
        demo_coefficient_start=contract['demo_coefficient_start'],
        demo_coefficient_end=contract['demo_coefficient_end'],
        keep_coefficient=contract['retention_coefficient'],
        demo_batch_size=contract['demo_batch_size'],retention_batch_size=contract['retention_batch_size'],retention_reference=reference,
        audit_enabled=contract.get('schema')=='jit_bridge_student_v1_2',
        max_first_behavior_kl=contract.get('max_first_behavior_kl'),
        first_update_audit_path=Path(run_dir)/'first_loss_gate.json',
        freeze_actor_normalizer=contract.get('freeze_actor_normalizer',False),
        demo_clock=contract.get('demo_clock','normalizer_count'),actual_gradient_audit=contract.get('actual_gradient_audit',False))
    def train(**kwargs):
        from .protocol import atomic_json
        atomic_json(Path(run_dir)/'bridge_student_contract.json',{**contract,
            'demo_count':0 if demo is None else demo['count'],
            'empty_demo_behavior':'disable_sampler_continue_ppo_keep',
            'critic_and_optimizer':raw.get('continuous_learner','fresh_inherited_probe_protocol'),
            'offline_demo_ppo_replay':False,'source_actor_sha256':source['actor_sha256']})
        warm=contract.get('warmup_initializer')
        if raw.get('continuous_learner') is not None:
            if warm is not None:raise ValueError('continuous learner cannot apply an offline warmup initializer')
            kwargs['restore_value_fn']=True
        if warm is not None:
            import pickle
            from ..handoff_bank import pytree_sha256
            if contract.get('schema')!='jit_bridge_student_v1_2':raise ValueError('warmup restore requires v1_2')
            if file_sha(warm['path'])!=warm['sha256']:raise ValueError('warmup checkpoint hash drift')
            if warm['source_actor_sha256']!=reference_source['actor_sha256']:raise ValueError('warmup source Actor drift')
            with Path(warm['path']).open('rb') as stream:warm_params=pickle.load(stream)
            if not isinstance(warm_params,tuple) or len(warm_params)!=3:raise ValueError('warmup inference tuple required')
            if (pytree_sha256(warm_params[0])!=warm['normalizer_sha256'] or
                warm['normalizer_sha256']!=reference_source['normalizer_sha256']):
                raise ValueError('warmup changed source normalizer')
            if not all(np.isfinite(x).all() for x in __import__('jax').tree.leaves(warm_params[1])):
                raise ValueError('nonfinite warmup Actor')
            kwargs['restore_params']=(warm_params[0],warm_params[1],kwargs['restore_params'][2])
        completed=False
        try:
            result=wrapped(**kwargs);completed=True;return result
        finally:
            if contract.get('schema')=='jit_bridge_student_v1_2':
                atomic_json(Path(run_dir)/'learning_probe.json',{
                    'schema':'jit_bridge_learning_probe_v1_2','training_completed':completed,
                    'first_executed_losses':usage.get('learning_audit_first_losses',[]),
                    'latest_executed_loss':usage.get('learning_audit_latest'),
                    'executed_loss_calls':usage.get('learning_audit_loss_calls',0),
                    'first_loss_gate':usage.get('first_loss_gate'),
                    'probe_scope':'fixed deterministic weighted-strata demo/retention probes <=256 each; PPO gradients use actual on-policy batch',
                    'fixed_demo_probe_indices':usage.get('fixed_demo_probe_indices',[]),
                    'fixed_retention_probe_indices':usage.get('fixed_retention_probe_indices',[]),
                    'kl_definition':'installed Brax KL(behavior || current), numerical +1e-5 inside log; self KL is recorded',
                    'four_combinations':{'status':'not_executed_by_training_loss'},
                    'optimizer_update_acceptance':'loss execution does not itself prove optimizer acceptance'})
            atomic_json(Path(run_dir)/'bridge_demo_usage.json',{**usage,'training_completed':completed,
                'count_semantics':'actual executed joint loss batches; excludes compilation; failures are not accepted training',
                'round_used_teacher_demo':completed and sum(usage['demo_samples_by_root'].values())>0})
    return train


def load_retention_reference(reference):
    """Load the explicit frozen reference independently of PPO restore_params."""
    from ..unified_policy_freeze import load_frozen_unified_manifest, _checkpoint_identity, _load_policy_formal_config
    from ..checkpoint import load_checkpoint
    from ..handoff_bank import pytree_sha256
    if file_sha(reference['path'])!=reference['sha256']:raise ValueError('retention manifest hash drift')
    policy=load_frozen_unified_manifest(Path(reference['path']))['policy']
    config=_load_policy_formal_config(Path(policy['formal_config']))
    payload=load_checkpoint(Path(policy['checkpoint']),expected=_checkpoint_identity(config))
    values=(payload.observation_normalizer,payload.actor_params)
    for key,value in zip(('normalizer_sha256','actor_sha256'),values):
        if pytree_sha256(value)!=policy[key]:raise ValueError('retention reference identity drift')
    return values,policy


def load_retention_traces(reference):
    """Hash-bound full successful TRAIN pre-action observations, no DEV input."""
    if reference.get('role')!='train' or reference.get('full_success') is not True:
        raise ValueError('successful TRAIN retention trajectory declaration required')
    if file_sha(reference['path'])!=reference['sha256']:raise ValueError('retention trace hash drift')
    with np.load(reference['path'],allow_pickle=False) as data:
        obs=np.asarray(data['actor_observation_before'],np.float32)
        weights=np.asarray(data['weights'],np.float64) if 'weights' in data else np.ones(len(obs))
    if obs.ndim!=2 or obs.shape[1]!=76 or not len(obs) or not np.isfinite(obs).all():
        raise ValueError('invalid full trajectory retention observations')
    if weights.shape!=(len(obs),) or not np.isfinite(weights).all() or np.any(weights<0) or weights.sum()<=0:
        raise ValueError('invalid full trajectory retention weights')
    return obs,weights/weights.sum()
