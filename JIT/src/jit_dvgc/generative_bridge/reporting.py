"""Preparation reports have zero execution; example budgets are not authorization."""
from .contracts import validate_spec
from .rewards import WEIGHTS


def prepare(spec):
    report=validate_spec(spec)
    # Same total physical budget, with saved search spending preallocated to PPO.
    # PPO block alignment leaves 3200 for explicitly separate original acquisition.
    common=2*161*400+1600
    arms=[]
    for name in spec['ablation_plan']['arms']:
        ppo_only=name=='ppo_only_matched_total_budget'
        arms.append({'name':name,'physical_interaction_ceiling':680800,
            'student_transitions':547200 if ppo_only else 128000,
            'search_interactions':0 if ppo_only else 422400,
            'common_diagnostic_and_acceptance':common,
            'additional_original_acquisition':3200 if ppo_only else 0,
            'supervised_update_budget':0 if ppo_only or name=='frozen_generator' else 'requires_separate_declaration',
            'candidate_mix':None if ppo_only else ({'source_only':1,'colored_noise':31} if name=='colored_or_icem_teacher'
                else {'source_only':1,'colored_noise':15,'diffusion':16}),
            'actor_to_generator':name=='full_bidirectional_updates',
            'retention_coefficient':.2,'actor_task_reward_and_resets':'same_locked_source',
            'wall_time_budget':'requires_separate_declaration','execute':False})
    # Current probe training uses 3200-transition blocks; preserve its exact rule.
    block=128*25
    for arm in arms:
        if arm['student_transitions']%block:
            remainder=arm['student_transitions']%block
            arm['student_transitions']-=remainder
            arm['additional_original_acquisition']+=remainder
    report.update(execute=False,comparison_arms=arms,
        generator_comparison=spec['generator_comparison'],reward_weights=WEIGHTS,
        cost_ledger={'scope':'new_experiment_only_excludes_CPU_fixture_optimizer_tests','actual_physics_interactions':0,'actual_supervised_updates':0,
            'cpu_logic_tests':'reported_separately','gpu_physics':'not_run','research_performance':'not_run'},
        not_run=['source binding','GPU semantic smoke','G pretraining','teacher solve',
                 'student physical PPO','fixed Actor G comparison','formal experiments'])
    return report
