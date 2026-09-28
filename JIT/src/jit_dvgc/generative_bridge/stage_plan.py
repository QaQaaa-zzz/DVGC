"""Finite v1.2 stage declarations; permission never bypasses evidence gates."""
from copy import deepcopy


def make_stage_plan(source_checkpoint, actor_sha256, normalizer_sha256):
    return {'schema':'jit_bridge_stage_plan_v1_2','execute':True,
        'source_checkpoint':source_checkpoint,'actor_sha256':actor_sha256,
        'normalizer_sha256':normalizer_sha256,'master_seed':9281201,
        'planned_independent_seeds':[9281201,9281202,9281203],
        'executed_seed_count':0,'final_test_open':False,'automatic_extension':False,
        'source_roles':{'baseline_actor':'P0','teacher_tail_actor':'P0',
            'student_initializer':'P0_or_declared_warmup','retention_reference_actor':'P0'},
        'stages':{
            'A0':{'execute':True,'physics_cap':1600,'supervised_updates':0,
                  'description':'source binding, action parity, four nominal repeats'},
            'A1':{'execute':True,'physics_cap':1820000,'generator_updates':20000,
                  'generator_dev_episodes':64,'train_episodes':1024,'id_dev_episodes':512,'temporal_dev_episodes':384,
                  'teacher_roots_cap':32,'teacher_worlds':32,'teacher_replay_worlds':32,
                  'numerical_recheck_cap':3200,'requires':['source_identity_valid','action_parity','nominal_success']},
            'A2':{'execute':True,'physics_cap':2600000,'ppo_transitions':384000,
                  'ppo_per_arm':128000,'bc_updates_cap':2000,'generator_updates_cap':2000,
                  'arms':['PPO_keep','cumulative_demo_PPO_keep','cumulative_demo_warmup_PPO_keep'],
                  'four_combination_physics_cap':25600,
                  'requires':['source_identity_valid','action_parity','nominal_success','data_ready','teacher_semantics_valid']},
            'B':{'execute':False,'round_cap':2,'requires':['source_identity_valid','action_parity',
                 'nominal_success','student_absorption','adopted_candidate','demo_control_benefit']}},
        'budgets':{'physics_cap':4421600,'wall_seconds':86400,'retry_physics_cap':0,
                   'generator_updates_cap':22000,'bc_updates_cap':2000},
        'comparison_claim':'mechanism comparison; equal PPO is not equal total budget',
        'initialization':'inference Actor+normalizer; fresh critic and optimizer, not exact training resume'}


def require_stage(plan, name, evidence):
    stage=plan['stages'][name]
    missing=[k for k in stage.get('requires',[]) if evidence.get(k) is not True]
    if missing:raise ValueError('stage evidence missing: '+', '.join(missing))
    if stage['execute'] is not True:raise ValueError('stage not enabled')
    return stage


def next_incumbent(current, candidate, acceptance, cumulative_demo_bank):
    """Promotion-aware transition; an ordinary rejection cannot erase valid demos."""
    if type(acceptance.get('adopted')) is not bool:raise ValueError('explicit adoption decision required')
    result=deepcopy(candidate if acceptance['adopted'] else current)
    result['demo_bank']=deepcopy(cumulative_demo_bank)
    return result
