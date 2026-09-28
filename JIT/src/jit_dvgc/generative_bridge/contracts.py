"""Immutable input contracts. Preparation is deliberately separate from execution."""
import hashlib
import json
from pathlib import Path

SCHEMA = 'jit_generative_bridge_v1_1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(',', ':')).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def support_fingerprint(support):
    # Lock the entire original view, including quotas, phase/group selection and inputs.
    return digest(support)


def validate_pending_support_unchanged(before, after):
    if before != support_fingerprint(after):
        raise ValueError('original pending membership/weights or support contract changed')


def validate_spec(spec, *, execute=False):
    if spec.get('schema') != SCHEMA:
        raise ValueError('merged v1.1 schema required; historical specs are read-only')
    checks = [('task.actor_obs_dim',76),('task.critic_obs_dim',106),
        ('task.control_dt',.02),('task.total_horizon',400),
        ('task.required_success_criterion','stable_forward_recovery'),
        ('task.reward_mode','inherit_locked_source'),('task.reset_mixture','inherit_locked_source'),
        ('generator.horizon',16),('generator.action_dim',4),('generator.observation_dim',76),
        ('generator.widths',[64,128,256]),('generator.expected_parameter_count',3131716),
        ('search.total_candidates',32),('search.bridge_horizon',16),
        ('search.diffusion_candidates',16),('search.colored_noise_candidates',15),
        ('search.source_only_candidates',1),('feedback.teacher_success_bonus',0.),
        ('learning_contract.teacher_success_required_for_ppo',False),
        ('learning_contract.preserve_all_valid_pending_eligibility',True),
        ('generator.train_noise_steps',100),('generator.inference_steps',20),
        ('generator.inference_timesteps',list(range(99,-1,-5))),('generator.ddim_eta',0.),
        ('generator.clip_diffusion_latent',False),('generator.clip_predicted_clean_action',True),
        ('generator.use_clipped_model_output',True),('generator.groups',8),
        ('generator.norm_epsilon',1e-5),('generator.dropout',0.),('generator.attention',False),
        ('generator.resblocks_per_level',2),('generator.condition_dim',512),
        ('generator.obs_encoder',[128,256]),('generator.time_encoder',[256,256]),
        ('generator.time_sinusoidal_dim',128),('generator.prediction_type','epsilon'),
        ('generator.incremental_training.learning_rate',1e-5),
        ('generator.incremental_training.warmup_updates',0),
        ('generator.incremental_training.include_incumbent_in_selection',True),
        ('data.bidirectional_corpus.actor_requires_adoption',True),
        ('data.bidirectional_corpus.allow_unadopted_actor_success',False),
        ('data.bidirectional_corpus.require_train_role',True),
        ('data.bidirectional_corpus.require_raw_pre_action_observation',True),
        ('data.bidirectional_corpus.mix_weights',{'history':.5,'teacher_new':.25,'actor_new':.25}),
        ('data.bidirectional_corpus.promote_actor_prefix_to_bridge_without_revalidation',False),
        ('student.retention_coefficient',.2),('student.demo_coefficient_start',.2),
        ('student.demo_coefficient_end',.05),('student.offline_demo_used_as_ppo_replay',False),
        ('student.sample_demo_when_empty',False),('actor_success_export.automatic_recapture',False)]
    for dotted, expected in checks:
        value=spec
        try:
            for key in dotted.split('.'): value=value[key]
        except (KeyError,TypeError) as error:
            raise ValueError('missing v1.1 contract field: '+dotted) from error
        if value != expected: raise ValueError('v1.1 contract mismatch: '+dotted)
    if spec['data']['final_test_open'] is not False:
        raise ValueError('final TEST must remain closed')
    if spec['generator']['incremental_training']['max_updates'] > 2000:
        raise ValueError('incremental update ceiling exceeded')
    missing=['paths.'+k for k,v in spec['paths'].items() if v is None]
    for key in ('max_new_interactions','max_new_supervised_updates','max_compute_wall_seconds'):
        v=spec['execution'].get(key)
        if v is None: missing.append('execution.'+key)
        elif type(v) not in (int,float) or not 0 < v < float('inf'):
            raise ValueError('finite positive execution budget required: '+key)
    if execute:
        if spec.get('execute') is not True or missing:
            raise ValueError('execution disabled or unresolved sources/budgets: '+', '.join(missing))
        locks=spec.get('input_sha256',{})
        for key,path in spec['paths'].items():
            if key=='output_dir': continue
            if path not in locks or file_sha(path)!=locks[path]:
                raise ValueError('missing or mismatched input hash: '+key)
    return {'schema':SCHEMA,'status':'needs_input_resolution' if missing else 'prepared',
            'missing':missing,'execute':False,'spec_sha256':digest(spec)}
