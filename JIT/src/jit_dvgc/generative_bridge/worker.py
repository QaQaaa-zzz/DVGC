"""Offline G worker; the production supervisor owns budgets and subprocess timeout."""
from pathlib import Path
import json
import numpy as np
from .contracts import file_sha
from .protocol import atomic_json


def source_payload(frozen):
    from ..checkpoint import CheckpointIdentity,load_checkpoint
    policy=json.loads(Path(frozen).read_text())['policy'];path=Path(policy['checkpoint'])
    raw=json.loads((path/'identity.json').read_text())
    identity=CheckpointIdentity(**{k:tuple(raw[k]) if isinstance(raw[k],list) else raw[k]
        for k in ('config_sha256','xml_sha256','actor_frame_fields','actor_task_fields','action_order')})
    payload=load_checkpoint(path,expected=identity)
    from ..handoff_bank import pytree_sha256
    for k,v in [('actor_sha256',payload.actor_params),('normalizer_sha256',payload.observation_normalizer)]:
        if pytree_sha256(v)!=policy[k]:raise ValueError('source checkpoint hash drift')
    return policy,payload


def generator_template(frozen,seed):
    import jax
    import jax.numpy as jp
    from .network import ConditionalUNet
    from .diffusion import create_train_state
    policy,payload=source_payload(frozen)
    net=ConditionalUNet();key,init=jax.random.split(jax.random.PRNGKey(seed))
    params=net.init(init,jp.zeros((1,16,4)),jp.zeros((1,76)),jp.zeros((1,),jp.int32))
    state=create_train_state(params,key,{'mean':payload.observation_normalizer.mean['state'],
        'std':payload.observation_normalizer.std['state']})
    identity={k:policy[k] for k in ('actor_sha256','normalizer_sha256','xml_sha256')}
    return net,state,identity


def run_generator(config):
    from .artifacts import load_corpus,GeneratorUpdateStage
    from .diffusion import train_pretrain,restore_state
    import jax
    if jax.default_backend()!='gpu':raise RuntimeError('production G worker requires GPU')
    net,state,identity=generator_template(config['source_frozen_policy'],config['seed'])
    corpus=load_corpus(config['corpus'])
    if file_sha(config['dev_fixture'])!=config['dev_fixture_sha256']:raise ValueError('fixed dev fixture drift')
    with np.load(config['dev_fixture'],allow_pickle=False) as a:fixture=tuple(a[k] for k in ('observations','actions','timesteps','noise'))
    predict=lambda p,x,o,k:net.apply(p,x,o,k)
    root=Path(config['output'])
    if config['mode']=='pretrain':
        _,report=train_pretrain(state,corpus,predict=predict,dev_fixture=fixture,output=root,
            identity=identity,updates=config['updates'],charged_updates_before=0,max_wall_seconds=config['max_wall_seconds'])
        manifest=root/report['selected']/'manifest.json'
        result={'checkpoint_manifest':str(manifest),'checkpoint_manifest_sha256':file_sha(manifest),
            'status':'completed','updates':report['updates'],'mode':'pretrain'}
    elif config['mode']=='incremental':
        manifest=Path(config['incumbent']['checkpoint_manifest'])
        if file_sha(manifest)!=config['incumbent']['checkpoint_manifest_sha256']:raise ValueError('incumbent identity changed')
        state=restore_state(manifest.parent,state,identity)
        result=GeneratorUpdateStage(root,state=state,predict=predict,dev_fixture=fixture,identity=identity,
            updates=config['updates'],max_wall_seconds=config['max_wall_seconds'])(config['corpus'])
    else:raise ValueError('unknown G worker mode')
    atomic_json(Path(config['result']),result)
