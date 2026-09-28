"""Read-only reproducible audit of explicitly supplied historical artifacts."""
import argparse
from collections import Counter
from dataclasses import asdict
from importlib.metadata import version, PackageNotFoundError
import json
from pathlib import Path
import subprocess
import numpy as np
from jit_dvgc.checkpoint import load_checkpoint, CheckpointIdentity
from jit_dvgc.handoff_bank import pytree_sha256
from jit_dvgc.generative_bridge.contracts import file_sha
from jit_dvgc.generative_bridge.protocol import atomic_json


def main():
    import jax
    p=argparse.ArgumentParser()
    p.add_argument('--pointer',type=Path,required=True)
    p.add_argument('--round',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();pointer=json.loads(a.pointer.read_text())
    frozen_path=Path(pointer['frozen_policy']);frozen=json.loads(frozen_path.read_text())['policy']
    checkpoint=Path(frozen['checkpoint']);raw=json.loads((checkpoint/'identity.json').read_text())
    identity=CheckpointIdentity(**{k:tuple(raw[k]) if isinstance(raw[k],list) else raw[k]
        for k in ('config_sha256','xml_sha256','actor_frame_fields','actor_task_fields','action_order')})
    payload=load_checkpoint(checkpoint,expected=identity)
    checks={k:pytree_sha256(v)==frozen[k] for k,v in [('actor_sha256',payload.actor_params),
        ('critic_sha256',payload.critic_params),('normalizer_sha256',payload.observation_normalizer)]}
    if not all(checks.values()):raise ValueError('frozen policy tree identity drift')
    cfg=json.loads(Path(frozen['formal_config']).read_text())
    if file_sha(frozen['formal_config'])!=frozen['formal_config_file_sha256']:raise ValueError('source config drift')
    before=json.loads((a.round/'bank_evaluation/results.json').read_text())
    after=json.loads((a.round/'after_learning/results.json').read_text())
    support=json.loads((a.round/'training_support.json').read_text())
    traces={x['trace']:x['trace_sha256'] for r in after for x in r['attempts']}
    trace_audit=[]
    for path,sha in traces.items():
        if file_sha(path)!=sha:raise ValueError('historical trace hash drift')
        with np.load(path,allow_pickle=False) as t:
            trace_audit.append({'path':path,'sha256':sha,'fields':t.files,
                'raw_preobs_present':'actor_observation_before' in t,
                'mask_shape':list(t['mask'].shape),'real_control_transitions':int(t['mask'].sum())})
    packages={}
    for package in ('jax','jaxlib','flax','optax','brax','mujoco','mujoco-mjx','warp-lang'):
        try:packages[package]=version(package)
        except PackageNotFoundError:packages[package]=None
    report={'audit_only_not_selected_for_new_run':True,'pointer':str(a.pointer),
        'pointer_sha256':file_sha(a.pointer),'pointer_contents':pointer,'source_frozen_policy':str(frozen_path),
        'source_identity':frozen,'identity_checks':checks,'packages':packages,
        'actor_parameters':sum(x.size for x in jax.tree.leaves(payload.actor_params)),
        'critic_parameters':sum(x.size for x in jax.tree.leaves(payload.critic_params)),
        'actor_leaf_shapes':[list(x.shape) for x in jax.tree.leaves(payload.actor_params)],
        'normalizer_shapes':{k:list(v.shape) for k,v in payload.observation_normalizer.mean.items()},
        'control_dt':.02,'action_order':list(identity.action_order),
        'success_criterion':cfg['success_criterion'],'reward_mode':cfg.get('reward_mode'),
        'pending_fraction':cfg.get('pending_fraction'),'jump_start_probability':cfg['jump_start_probability'],
        'audited_round':str(a.round),'source_label_counts':dict(Counter(str(r['label']) for r in before)),
        'valid_pending':sum(r['label']==0 and not r['prefix_terminal'] for r in before),
        'support_status_counts':dict(Counter(r['evidence_status'] for r in support['entries'])),
        'support_phase_counts':dict(Counter(r['phase'] for r in support['entries'])),
        'support_unique_ancestors':len({r['trajectory_id'] for r in support['entries']}),
        'student_label_counts':dict(Counter(str(r['label']) for r in after)),
        'promotion':json.loads((a.round/'promotion.json').read_text()),'trace_audit':trace_audit,
        'new_generator_admissible_windows':0,'generator_checkpoint':None,
        'missing':['selected_new_experiment_source','raw_pre_action_observations_in_audited_legacy_suffixes',
            'locked_generator_train_dev_split','locked_core_protected_solver_dev_panels','execution_budgets'],
        'physics_interactions':0,'gradient_updates':0,'GPU_physics_validation':'not_run',
        'research_performance':'not_run'}
    atomic_json(a.output,report)
    print(json.dumps({k:report[k] for k in ['actor_parameters','critic_parameters','pending_fraction',
        'valid_pending','student_label_counts','new_generator_admissible_windows','packages']},indent=2))

if __name__=='__main__':main()
