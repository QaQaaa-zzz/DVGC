"""Fresh-round explorer mixtures and fail-closed on-policy sample admission."""
import json
from pathlib import Path
import numpy as np
from .pulse_protocol import EpisodeKeyRegistry, requested_draws
from .contracts import file_sha

EXPLORER_NAMESPACE = 0x4558504C


def validate_collection_options(spec):
    cfg=spec.get('explorer_admission_v1_2')
    if cfg is None:return None
    if any(spec.get(k) for k in ('reuse_prefix_collection','reuse_collection','reuse_results','historical_roots')):
        raise ValueError('explorer updates require a fresh current-round collection')
    if (spec.get('controller_mode')!='learned_residual' or spec.get('explorer_backend')!='rsl_rl'
        or spec.get('pulse_steps')!=3 or spec.get('pulse_start_schedule')!=[0]
        or spec.get('delta_limit')!=[.25]*4 or spec.get('pulse_event_schedule')
        or spec.get('pulse_protocol_v1_2') or spec.get('initial_velocity_randomization')
        or spec.get('nominal_source_rollout')):
        raise ValueError('Stage B requires the declared RSL complete-start three-step pulse contract')
    if spec.get('neighborhood') is not None:
        from ..neighborhood import _config
        _config(spec['neighborhood'])
        path=spec.get('neighborhood_map');sha=spec.get('neighborhood_map_sha256')
        if not path or not sha or file_sha(path)!=sha:raise ValueError('frozen neighborhood map missing or changed')
        payload=json.loads(Path(path).read_text())
        if payload.get('config')!=spec['neighborhood'] or payload.get('frozen_before_collection') is not True:
            raise ValueError('neighborhood must be frozen before current collection')
    if (not isinstance(cfg,dict) or not isinstance(cfg.get('run_id'),str) or not cfg['run_id']
        or not isinstance(cfg.get('collection_id'),str) or not cfg['collection_id']
        or cfg.get('round')!=spec.get('round_index') or cfg.get('uniform_episode_fraction')!=.2
        or len(cfg.get('episode_ids',[]))!=spec.get('num_envs') or not cfg['episode_ids']):
        raise ValueError('invalid current-round explorer lineage/mixture')
    if not spec.get('explorer_checkpoint') and spec.get('explorer_initialization')!={'mode':'symmetric','latent_std':.6}:
        raise ValueError('fresh Stage B explorer requires symmetric mean0 latent std0.6')
    return cfg


def episode_policy_keys(spec):
    cfg=validate_collection_options(spec)
    if cfg is None:return None
    registry=EpisodeKeyRegistry()
    return np.stack([np.asarray(registry.claim(cfg['master_seed'],EXPLORER_NAMESPACE,cfg['round'],i,'policy'))
                     for i in cfg['episode_ids']])


def collection_mixture(spec):
    import jax
    cfg=validate_collection_options(spec)
    if cfg is None:return None,None
    registry=EpisodeKeyRegistry();learned=[];draws=[]
    for i in cfg['episode_ids']:
        key=registry.claim(cfg['master_seed'],EXPLORER_NAMESPACE,cfg['round'],i,'condition')
        learned.append(not bool(jax.random.bernoulli(key,.2)))
        key=registry.claim(cfg['master_seed'],EXPLORER_NAMESPACE,cfg['round'],i,'pulse')
        draws.append(np.asarray(requested_draws(key)))
    return np.asarray(learned,bool),np.stack(draws)


def mix_sample(raw,delta,log_prob,value,learned,uniform,pulse_mask):
    import jax.numpy as jp
    valid=learned & pulse_mask
    return (jp.where(learned[:,None],raw,jp.nan),
            jp.where(learned[:,None],delta,uniform),
            jp.where(valid,log_prob,jp.nan),jp.where(learned,value,0.),valid)


def behavior_identity(state, normalizer=None):
    from ..handoff_bank import pytree_sha256
    params=state['params']
    actor={k:v for k,v in params.items() if k in ('actor','actor_encoder','policy')}
    if not actor:raise ValueError('explorer Actor parameters missing')
    if normalizer is None:
        normalizer={'mean':state['normalizer_mean'],'std':state['normalizer_std']}
    return dict(behavior_actor_sha256=pytree_sha256(actor),
                behavior_normalizer_sha256=pytree_sha256(normalizer))


def collection_receipt(spec,state,learned,normalizer=None):
    cfg=validate_collection_options(spec)
    identity=behavior_identity(state,normalizer)
    for k,v in identity.items():
        if k in cfg and cfg[k]!=v:raise ValueError('declared explorer behavior identity drift: '+k)
    return dict(cfg,**identity,neighborhood=spec.get('neighborhood'),
                neighborhood_map_sha256=spec.get('neighborhood_map_sha256'),fresh_collection=True,
                episode_modes=['learned' if flag else 'uniform' for flag in learned],
                mode_codes={'learned':1,'uniform':0},uniform_branch_on_policy=False,
                learned_episode_count=int(np.asarray(learned).sum()),
                uniform_episode_count=int((~np.asarray(learned,bool)).sum()),
                policy_probability_space='raw latent Gaussian; no uniform logprob substituted')


def validate_admission(spec,receipt,tape,actual_identity):
    cfg=validate_collection_options(spec)
    if cfg is None:raise ValueError('mixture tape requires explicit current-round admission')
    for key in ('run_id','round','collection_id','master_seed','episode_ids',
                'behavior_actor_sha256','behavior_normalizer_sha256'):
        if key not in cfg or cfg[key]!=receipt.get(key):raise ValueError('explorer lineage mismatch: '+key)
    if receipt.get('fresh_collection') is not True:raise ValueError('historical root replay is forbidden')
    for key in ('neighborhood','neighborhood_map_sha256'):
        if spec.get(key)!=receipt.get(key):raise ValueError('explorer neighborhood behavior mismatch')
    for key,value in actual_identity.items():
        if receipt.get(key)!=value:raise ValueError('explorer behavior mismatch: '+key)
    modes=receipt['episode_modes']
    if len(modes)!=spec['num_envs'] or any(m not in ('learned','uniform') for m in modes):
        raise ValueError('invalid per-episode explorer modes')
    mask=np.asarray(tape['mask'],bool);prefix=np.asarray(tape['prefix_mask'],bool)
    expected=mask & prefix & np.asarray([m=='learned' for m in modes])[None,:]
    learned=np.broadcast_to(np.asarray([m=='learned' for m in modes]),mask.shape)
    for key,truth in [('explorer_learned',learned),('on_policy_mask',expected),('log_prob_valid',expected)]:
        if np.shape(tape[key])!=mask.shape or not np.array_equal(tape[key],truth):
            raise ValueError('invalid explorer admission mask: '+key)
    if mask.shape!=prefix.shape or not np.isfinite(np.asarray(tape['log_prob'])[expected]).all():
        raise ValueError('missing actual learned raw-action log probability')
    if not np.isfinite(np.asarray(tape['raw_action'])[expected]).all():
        raise ValueError('missing actual learned raw action')
    uniform=~learned
    if (not np.isnan(np.asarray(tape['log_prob'])[uniform]).all()
        or not np.isnan(np.asarray(tape['raw_action'])[uniform]).all()):
        raise ValueError('uniform branch must not fabricate learned raw action/log probability')
    return expected


def training_mask(spec,tape,feedback):
    mask=np.asarray(tape['mask'],bool)
    if 'on_policy_mask' in tape:
        receipt=spec.get('_validated_explorer_admission')
        if receipt is None:raise ValueError('mixture samples require validated admission before PPO')
        mask=validate_admission(spec,receipt,tape,spec['_validated_behavior_identity'])
    elif spec.get('explorer_admission_v1_2') is not None:
        raise ValueError('new explorer protocol requires recorded on-policy provenance')
    return mask & np.asarray(feedback['eligible'],bool)[None,:]


def claim_update(collection,receipt):
    path=Path(collection)/'explorer_update_claim.json'
    try:
        with path.open('x') as stream:json.dump({k:receipt[k] for k in ('run_id','round','collection_id')},stream)
    except FileExistsError as exc:
        raise ValueError('collection already claimed for explorer update; historical replay forbidden') from exc


def admit_update(spec,collection,state,tape,normalizer=None):
    path=Path(collection)
    if spec.get('explorer_admission_v1_2') is None:
        if 'on_policy_mask' in tape or (path/'explorer_admission.json').exists():
            raise ValueError('cannot bypass current-round explorer admission')
        return spec
    receipt=json.loads((path/'explorer_admission.json').read_text())
    for name in ('prefixes.npz','behavior.msgpack','update_state.msgpack'):
        if file_sha(path/name)!=receipt['files'][name]:raise ValueError('explorer collection artifact drift: '+name)
    identity=behavior_identity(state,normalizer)
    validate_admission(spec,receipt,tape,identity)
    claim_update(path,receipt)
    return {**spec,'_validated_explorer_admission':receipt,'_validated_behavior_identity':identity}
