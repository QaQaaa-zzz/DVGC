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


def generator_reference(config):
    """Keep G conditioning fixed when an accepted Actor changes the teacher tail."""
    reference=config.get('generator_reference_frozen_policy')
    if reference is None:
        return config['source_frozen_policy']
    if not isinstance(reference,dict) or set(reference)!={'path','sha256'}:
        raise ValueError('generator reference requires path and sha256')
    if file_sha(reference['path'])!=reference['sha256']:
        raise ValueError('generator reference manifest hash drift')
    baseline,_=source_payload(reference['path'])
    tail,_=source_payload(config['source_frozen_policy'])
    if baseline['xml_sha256']!=tail['xml_sha256']:
        raise ValueError('generator reference physics differs from teacher tail')
    return reference['path']


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
    net,state,identity=generator_template(generator_reference(config),config['seed'])
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
            'status':'completed','updates':report['updates'],'mode':'pretrain',
            'total_charged_updates':report['total_charged_updates'],'charged_updates_scope':'lifetime',
            'state_updates':json.loads(manifest.read_text())['updates'],'inference_parameters':'ema'}
    elif config['mode']=='incremental':
        manifest=Path(config['incumbent']['checkpoint_manifest'])
        if file_sha(manifest)!=config['incumbent']['checkpoint_manifest_sha256']:raise ValueError('incumbent identity changed')
        state=restore_state(manifest.parent,state,identity)
        from ..handoff_bank import pytree_sha256
        declared_state=config['incumbent'].get('selected_state_sha256')
        if declared_state is not None and declared_state!=pytree_sha256(state):
            raise ValueError('incumbent full-state identity changed')
        charged_before=config.get('charged_updates_before')
        if charged_before is None and config['incumbent'].get('charged_updates_scope')=='lifetime':
            charged_before=config['incumbent']['total_charged_updates']
        if charged_before is None and config.get('generator_update_policy','fixed_dev_best')=='last_valid':
            raise ValueError('last_valid requires reconciled lifetime charged_updates_before for legacy incumbent')
        result=GeneratorUpdateStage(root,state=state,predict=predict,dev_fixture=fixture,identity=identity,
            updates=config['updates'],max_wall_seconds=config['max_wall_seconds'],
            generator_update_policy=config.get('generator_update_policy','fixed_dev_best'),
            charged_updates_before=charged_before)(config['corpus'])
        if result['status']=='skipped_no_new_data':
            result={**config['incumbent'],**result}
    else:raise ValueError('unknown G worker mode')
    atomic_json(Path(config['result']),result)


def run_warmup(config):
    """Offline supervised worker; supervisor owns GPU authorization and DEV choice.

    Required config: source_frozen_policy, retention_reference_actor {path,sha256},
    demo_manifest {path,sha256}, retention_trace_observations, output, result,
    seed, updates. Outputs all inference candidates, never silently adopts one.
    """
    import jax
    from brax.training.acme import running_statistics as rs
    from ..ppo import make_network_factory
    from .student import load_retention_reference,load_retention_traces
    from .warmup import warmup_actor
    from .learning_audit import action_probe
    if jax.default_backend()!='gpu':raise RuntimeError('production warmup worker requires GPU')
    policy,payload=source_payload(config['source_frozen_policy'])
    reference,reference_policy=load_retention_reference(config['retention_reference_actor'])
    if reference_policy['actor_sha256']!=policy['actor_sha256']:
        raise ValueError('first v1.2 warmup reference must be initializer P0')
    ref=config['demo_manifest']
    if file_sha(ref['path'])!=ref['sha256']:raise ValueError('warmup demo manifest hash drift')
    demo=json.loads(Path(ref['path']).read_text())
    anchors=load_retention_traces(config['retention_trace_observations'])
    sizes={key:int(value.shape[-1]) for key,value in payload.observation_normalizer.mean.items()}
    networks=make_network_factory()(sizes,4,preprocess_observations_fn=rs.normalize)
    initializer=(payload.observation_normalizer,payload.actor_params,payload.critic_params)
    def probe(update,params):
        value=action_probe(networks,params[0],params[1],demo,reference=reference)
        obs,weights=anchors
        prediction=np.asarray(networks.parametric_action_distribution.mode(networks.policy_network.apply(params[0],params[1],{'state':obs})))
        target=np.asarray(networks.parametric_action_distribution.mode(networks.policy_network.apply(reference[0],reference[1],{'state':obs})))
        value['retention_mse_channels']=np.sum((prediction-target)**2*weights[:,None],axis=0).tolist()
        return value
    from torch.utils.tensorboard import SummaryWriter
    writer=None
    def record_metrics(row):
        nonlocal writer
        # warmup_actor owns exclusive output creation; create TB only afterwards.
        if writer is None:writer=SummaryWriter(str(Path(config['output'])/'tensorboard'))
        for key,value in row.items():
            if key!='update':writer.add_scalar('warmup/'+key,value,row['update'])
        writer.flush()
    try:
        warmup_actor(networks,initializer,reference,demo,anchors,config['output'],seed=config['seed'],
            updates=config['updates'],probe_callback=probe,metrics_callback=record_metrics)
    finally:
        if writer is not None:writer.close()
    status=json.loads((Path(config['output'])/'warmup_status.json').read_text())
    candidates=[{**row,'source_actor_sha256':policy['actor_sha256'],'normalizer_sha256':policy['normalizer_sha256']} for row in status['checkpoints']]
    result=dict(status=status['status'],completed_updates=status['completed_updates'],environment_interactions=0,
        candidates=candidates,selected=None,selection_status='requires_predeclared_physical_dev_scores',
        reference_actor_sha256=reference_policy['actor_sha256'],optimizer_restored=False)
    atomic_json(config['result'],result)
    return result
