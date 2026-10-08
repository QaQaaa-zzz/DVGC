"""Strict opt-in protocol and zero-simulation budget inspection."""
from copy import deepcopy
from pathlib import Path
from .contracts import digest,file_sha
from .pulse_protocol import normalize_pulse_contract
from ..pulse_schedule import collection_budget

PROFILE_SCHEMA='jit_bridge_continuation_profile_v1'
FIXED=dict(schema=PROFILE_SCHEMA,pulse_contract_version='scheduled_single_pulse_v1',
    pulse_start_schedule=[0,5,10,15],pulse_batch_mode='mixed',delta_limit=[.25]*4,
    single_pulse_per_episode=True,uniform_episode_fraction=0.,teacher_colored_noise_candidates=0,
    continuous_learning=True,nominal_evaluation=False,generator_update_policy='last_valid',
    generator_old_dev_metric='monitor_only',generator_inference_parameters='ema',
    final_test_open=False,automatic_extension=False,automatic_retry=False)


def validate_profile(value):
    if not isinstance(value,dict) or set(value)!=set(FIXED)|{'pulse_steps'}:
        raise ValueError('unknown or missing profile fields')
    for key,expected in FIXED.items():
        if value[key]!=expected or (isinstance(expected,bool) and value[key] is not expected):
            raise ValueError('profile contract changed: '+key)
    normalize_pulse_contract({**value,'horizon':400})
    return deepcopy(value)


def load_profile(path):
    import yaml
    return validate_profile(yaml.safe_load(Path(path).read_text()))


def profile_budget(rounds):
    if type(rounds) is not int or not 1<=rounds<=200:raise ValueError('rounds must be in [1,200]')
    return dict(max_physics=1500000*rounds,per_round_physics=1500000,
        max_wall_seconds=86400 if rounds<=2 else 604800,
        student_transitions=128000*rounds,generator_updates=2000*rounds,explorer_epochs_per_round=4)


def budget_dry_run(plan):
    profile=validate_profile(plan['profile'])
    if plan['budgets']!=profile_budget(plan['rounds']):raise ValueError('frozen profile budget changed')
    collection=collection_budget({**profile,'num_envs':128,'horizon':400},plan.get('round_offset',0)+1)
    stages=dict(collection=collection['charged_maximum'],source_suffix=128*400,
        teacher_source=32*400,teacher_search=32*17*400,teacher_replay=32*17*400,
        student_with_internal_diagnostics=136000,student_train_feedback=128*400,
        student_stress=9*32*400,generator_offline=0,explorer_update=0)
    total=sum(stages.values())
    if total>plan['budgets']['per_round_physics']:raise ValueError('stage maxima exceed round budget')
    return dict(schema='jit_bridge_budget_dry_run_v1',simulation_executed=False,optimization_executed=False,
        profile_sha256=digest(profile),collection=collection,stage_maxima=stages,
        per_round_maximum=total,total_maximum=total*plan['rounds'],limits=plan['budgets'],
        generator_physical_monitor=dict(enabled=False,reason='separate bounded physical monitor required for capability claims'),
        student_stress_frequency='every completed round; original fixed development panel',
        acceptance='complete bundles and actual state/timing identities; no performance veto')


def inspect_plan(plan):
    from .production import implementation_identity
    if implementation_identity(plan['repository'])!=plan['implementation_commit']:
        raise ValueError('implementation identity changed')
    for path,sha in plan['locks'].items():
        if file_sha(path)!=sha:raise ValueError('locked input changed: '+path)
    for path,sha in plan['implementation_files'].items():
        if file_sha(path)!=sha:raise ValueError('implementation file changed: '+path)
    result=budget_dry_run(plan)
    item=plan['continuation_bundle']
    if file_sha(item['path'])!=item['sha256']:raise ValueError('continuation bundle changed')
    return result
