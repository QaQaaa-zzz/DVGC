"""One scoped PPO + optional demo + frozen-source retention loss adapter."""
from pathlib import Path
import numpy as np
from .contracts import file_sha
from ..policy_distillation import load_dataset


def load_optional_demo(manifest):
    if manifest is None:return None
    if manifest.get('schema')!='jit_bridge_demo_v1_1':raise ValueError('demo schema required')
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
        demo_sampler=None, usage_sink=None):
    """Must be invoked inside the existing guard_ppo_updates scope.

    Empty data creates no device target, sampler or demo RNG operation. Both
    coefficients zero preserves the original function object, not base+0*NaN.
    """
    coefficients=(demo_coefficient_start,demo_coefficient_end,keep_coefficient)
    if any(not np.isfinite(v) or v<0 for v in coefficients):raise ValueError('invalid auxiliary coefficient')
    if any(type(v) is not int or v<=0 for v in (transitions,demo_batch_size,retention_batch_size)):
        raise ValueError('positive transition and batch budgets required')
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
        if not use_demo and not keep_coefficient:return trainer(**kwargs)
        import jax
        import jax.numpy as jp
        from brax.training.agents.ppo import losses
        from ..policy_distillation import _count_float
        from ..policy_retention import action_mse
        source_normalizer,source_actor,_=kwargs['restore_params']
        start_count=_count_float(source_normalizer.count)
        original=losses.compute_ppo_loss
        if use_demo: demo_obs,demo_targets,demo_probs=map(jp.asarray,demo)
        if keep_coefficient: keep_obs,keep_probs=map(jp.asarray,retention)
        sampler=demo_sampler or jax.random.choice
        def record_usage(indices):
            for i in np.asarray(indices).reshape(-1):
                rid=sample_roots[int(i)]
                usage['demo_samples_by_root'][rid]=usage['demo_samples_by_root'].get(rid,0)+1
            usage['demo_loss_calls']+=1

        def loss(params,normalizer_params,data,rng,ppo_network,**options):
            base,metrics=original(params,normalizer_params,data,rng,ppo_network,**options)
            apply=ppo_network.policy_network.apply
            mode=ppo_network.parametric_action_distribution.mode
            total=base;demo_mse=jp.asarray(0.);keep_mse=jp.asarray(0.);scale=jp.asarray(0.)
            if use_demo:
                ids=sampler(jax.random.fold_in(rng,173),len(demo_obs),(demo_batch_size,),p=demo_probs)
                jax.debug.callback(record_usage,ids)
                prediction=mode(apply(normalizer_params,params.policy,{'state':demo_obs[ids]}))
                demo_mse=action_mse(prediction,demo_targets[ids])
                fraction=jp.clip((_count_float(normalizer_params.count)-start_count)/transitions,0,1)
                scale=demo_coefficient_start+(demo_coefficient_end-demo_coefficient_start)*fraction
                total=total+scale*demo_mse
            if keep_coefficient:
                ids=jax.random.choice(jax.random.fold_in(rng,719),len(keep_obs),(retention_batch_size,),p=keep_probs)
                anchor={'state':keep_obs[ids]}
                teacher=mode(apply(source_normalizer,source_actor,anchor))
                prediction=mode(apply(normalizer_params,params.policy,anchor))
                keep_mse=action_mse(prediction,teacher)
                total=total+keep_coefficient*keep_mse
            return total,{**metrics,'ppo_loss':base,'demo_action_mse':demo_mse,
                'effective_lambda_demo':scale,'retention_action_mse':keep_mse,
                'demo_samples_per_loss':jp.asarray(demo_batch_size if use_demo else 0),
                'retention_coefficient':jp.asarray(keep_coefficient),'total_loss':total}
        losses.compute_ppo_loss=loss
        try:
            result=trainer(**kwargs)
            jax.effects_barrier()
            return result
        finally:losses.compute_ppo_loss=original
    return train


def trainer_from_config(trainer, raw, run_dir):
    """Canonical formal entry point, still under the existing full-gradient guard."""
    import json
    from ..policy_retention import load_anchor
    if raw.get('action_retention') is not None:
        raise ValueError('bridge joint loss cannot nest action_retention wrappers')
    if raw.get('initialization',{}).get('actor')!='warm_start_frozen_unified':
        raise ValueError('bridge student must copy source Actor and normalizer')
    contract=raw['generative_bridge_student']
    if contract.get('schema')!='jit_bridge_student_v1_1':raise ValueError('student contract schema required')
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
    anchors=load_anchor(keep,source['name']) if keep is not None else None
    usage={}
    wrapped=make_joint_student_trainer(trainer,demo,retention=anchors,usage_sink=usage,
        transitions=raw['ppo']['requested_transitions'],
        demo_coefficient_start=contract['demo_coefficient_start'],
        demo_coefficient_end=contract['demo_coefficient_end'],
        keep_coefficient=contract['retention_coefficient'],
        demo_batch_size=contract['demo_batch_size'],retention_batch_size=contract['retention_batch_size'])
    def train(**kwargs):
        from .protocol import atomic_json
        atomic_json(Path(run_dir)/'bridge_student_contract.json',{**contract,
            'demo_count':0 if demo is None else demo['count'],
            'empty_demo_behavior':'disable_sampler_continue_ppo_keep',
            'critic_and_optimizer':'fresh_inherited_probe_protocol',
            'offline_demo_ppo_replay':False,'source_actor_sha256':source['actor_sha256']})
        completed=False
        try:
            result=wrapped(**kwargs);completed=True;return result
        finally:
            atomic_json(Path(run_dir)/'bridge_demo_usage.json',{**usage,'training_completed':completed,
                'count_semantics':'actual executed joint loss batches; excludes compilation; failures are not accepted training',
                'round_used_teacher_demo':completed and sum(usage['demo_samples_by_root'].values())>0})
    return train
