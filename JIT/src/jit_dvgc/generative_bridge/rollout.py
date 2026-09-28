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
