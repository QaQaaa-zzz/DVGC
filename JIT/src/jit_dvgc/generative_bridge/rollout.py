"""Optional fields and prefix action selection for the canonical GPU evaluator."""
import numpy as np
from .contracts import file_sha


def observation_fields(previous,next_state,action,*,enabled):
    if not enabled:return {}
    return {'actor_observation_before':previous.obs['state'],
            'actor_observation_after':next_state.obs['state'],
            'normalized_action_executed':action,
            'actuator_targets_after_mapping':next_state.data.ctrl}


def prefix_action(tick,source_action,prefixes,source_only):
    import jax.numpy as jp
    return jp.where(((tick<16)&~source_only)[:,None],
                    prefixes[:,jp.minimum(tick,15),:],source_action)


def load_prefix_plan(spec,rows):
    plan=spec.get('bridge_action_plan')
    if plan is None:return None
    if len(spec['order'])!=1 or spec.get('success_criterion')!='stable_forward_recovery':
        raise ValueError('bridge requires one frozen tail Actor and full recovery endpoint')
    if spec.get('reuse_results'):raise ValueError('bridge cannot reuse ordinary actor evidence')
    if file_sha(plan['path'])!=plan['sha256']:raise ValueError('prefix plan identity drift')
    if spec['horizon']!=400:raise ValueError('prefix and tail must share 400 ticks')
    with np.load(plan['path'],allow_pickle=False) as raw:
        prefixes=raw['actions'];source_only=raw['source_only'];contexts=raw['root_contexts']
    if prefixes.shape!=(len(rows),16,4) or not np.isfinite(prefixes).all() or np.any(np.abs(prefixes)>1):
        raise ValueError('invalid full normalized bridge actions')
    if source_only.shape!=(len(rows),) or source_only.dtype!=bool:
        raise ValueError('source-only mask required')
    if contexts.tolist()!=[r['snapshot_context_sha256'] for r in rows]:
        raise ValueError('prefix/root ordering mismatch')
    return prefixes,source_only


def closed_loop_action(tick, observations, keys, tail_policy, prefix_policy=None, source_only=None):
    """Select a closed-loop Actor; state, clocks and counters never reset here."""
    import jax
    if prefix_policy is None:
        return jax.vmap(tail_policy)(observations, keys)[0]
    def first(_):
        import jax.numpy as jp
        actions=jax.vmap(prefix_policy)(observations,keys)[0]
        if source_only is not None:
            actions=jp.where(source_only[:,None],jax.vmap(tail_policy)(observations,keys)[0],actions)
        return actions
    return jax.lax.cond(tick < 16,
        first,
        lambda _: jax.vmap(tail_policy)(observations, keys)[0], operand=None)


def validate_evaluation_options(spec, members):
    prefix = spec.get('closed_loop_prefix_policy')
    if prefix is not None:
        if (not isinstance(prefix, str) or prefix not in members
                or len(spec['order']) != 1 or spec['order'][0] not in members
                or spec.get('horizon') != 400
                or spec.get('success_criterion') != 'stable_forward_recovery'
                or 'bridge_action_plan' in spec or spec.get('reuse_results')):
            raise ValueError('invalid closed-loop prefix/tail evaluation contract')
    if spec.get('closed_loop_prefix_initializer') is not None and prefix is None:
        raise ValueError('prefix initializer requires prefix policy')
    if spec.get('closed_loop_prefix_source_only') is not None:
        mask=np.asarray(spec['closed_loop_prefix_source_only'])
        if prefix is None or mask.dtype!=bool or mask.ndim!=1:raise ValueError('invalid closed-loop source mask')
    if spec.get('warmup_initializer') is not None:
        if len(spec['order']) != 1 or spec.get('reuse_results'):
            raise ValueError('warmup candidate requires one tail and fresh evaluation')
    return prefix


def evaluation_controller_provenance(spec, actual_policy, members):
    prefix = spec.get('closed_loop_prefix_policy')
    result = dict(actor_sha256=actual_policy['actor_sha256'], actor_witness_eligible=True)
    if spec.get('warmup_initializer') is not None:
        result.update(controller_kind='warmup_candidate', actor_witness_eligible=False,
                      inference_override=actual_policy.get('inference_override'), adopted=False)
    if prefix is not None:
        result.update(controller_kind='composite_closed_loop', actor_witness_eligible=False,
            prefix_policy=prefix, prefix_actor_sha256=members[prefix]['policy']['actor_sha256'],
            tail_actor_sha256=actual_policy['actor_sha256'], handoff_tick=16, reset_at_handoff=False)
    if spec.get('bridge_action_plan') is not None:
        result.update(controller_kind='composite_teacher', actor_witness_eligible=False)
    return result


def warmup_evaluation_policy(env, source_policy, reference):
    """Build an in-memory candidate on the same locked environment/runtime."""
    import jax
    from pathlib import Path
    from ..checkpoint import load_checkpoint
    from ..unified_formal import load_unified_policy_formal_config
    from ..unified_training import checkpoint_identity
    from ..ppo import make_checkpoint_policy
    from .learning_audit import warmup_inference_override
    config = load_unified_policy_formal_config(Path(source_policy['formal_config']))
    source = load_checkpoint(Path(source_policy['checkpoint']), expected=checkpoint_identity(config, env))
    candidate, identity = warmup_inference_override(source, source_policy, reference)
    return jax.jit(make_checkpoint_policy(env, candidate, deterministic=True)), identity
