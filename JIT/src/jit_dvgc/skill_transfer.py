"""Explicit recovery-expert stages; old retention experiments remain immutable."""
import json
import pickle
from pathlib import Path
from .generative_bridge.protocol import atomic_json
from .generative_bridge.contracts import file_sha


def audit(config):
    from .recovery_curriculum import resolve_teacher_sources
    c=json.loads(Path(config).read_text()) if isinstance(config,(str,Path)) else config
    for key in ('pi0_descriptor','teacher_manifest','source_rows','runtime_config'):
        d=c[key]
        if file_sha(d['path'])!=d['sha256']:raise ValueError('source drift: '+key)
    sources=resolve_teacher_sources(c['teacher_manifest']['path'],c['source_rows']['path'])
    return dict(schema='frontierbridge_skill_transfer_audit_v1',roots=[r['root_id'] for _,r in sources],
        train_roots=len(sources),final_test_opened=False,automatic_adoption=False,
        keep_coefficient=0,retention_veto=False)


def prepare(config,output):
    c=json.loads(Path(config).read_text());checked=audit(c)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    atomic_json(output/'config.json',c);atomic_json(output/'source_audit.json',checked)
    return dict(config=str(output/'config.json'),audit=checked,environment_transitions=0)


def report(output):
    root=Path(output)
    return dict(run=str(root.resolve()),stages={str(p.relative_to(root)):json.loads(p.read_text())
        for p in root.glob('*/status.json')},final_test_opened=False,automatic_adoption=False)


def acquire_bc(config,output,*,roots=None,updates=2000,seed=0,resume=None):
    import jax
    from brax.training.acme import running_statistics as rs
    from torch.utils.tensorboard import SummaryWriter
    from .ppo import make_network_factory
    from .generative_bridge.worker import source_payload
    from .recovery_acquisition_bc import warmup_recovery_actor
    c=json.loads(Path(config).read_text());checked=audit(c)
    if jax.default_backend()!='gpu':raise RuntimeError('production acquisition requires GPU')
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'source_audit.json',checked);atomic_json(out/'config.json',c)
    _,p=source_payload(c['pi0_descriptor']['path'])
    initializer=(p.observation_normalizer,p.actor_params,p.critic_params)
    network=make_network_factory()({'state':76,'privileged_state':106},4,preprocess_observations_fn=rs.normalize)
    writer=SummaryWriter(str(out/'tensorboard'))
    def metrics(row):
        for key in ('loss','actor_grad_norm','actor_delta_norm'):
            writer.add_scalar('BC/'+key,row[key],row['update'])
        writer.add_scalar('BC/supervised_reward',-row['loss'],row['update']);writer.flush()
        atomic_json(out/'status.json',dict(state='running',stage='R_BC',completed=row['update'],declared=updates,
            environment_transitions=0,keep=0,retention_veto=False,adopted=False))
    try:
        warmup_recovery_actor(network,initializer,json.loads(Path(c['teacher_manifest']['path']).read_text()),
            out/'learner',seed=seed,updates=updates,root_ids=roots,resume_checkpoint=resume,metrics_callback=metrics)
        atomic_json(out/'status.json',dict(state='completed',stage='R_BC',completed=updates,declared=updates,
            environment_transitions=0,keep=0,retention_veto=False,adopted=False))
    except BaseException as exc:
        atomic_json(out/'status.json',dict(state='error',stage='R_BC',error=repr(exc)));raise
    finally:writer.close()
    return str(out/'learner'/'last_completed.json')
