from copy import deepcopy
import pytest


def profile():
    return dict(schema='jit_bridge_continuation_profile_v1',pulse_contract_version='scheduled_single_pulse_v1',
        pulse_start_schedule=[0,5,10,15],pulse_batch_mode='mixed',pulse_steps=3,
        delta_limit=[.25]*4,single_pulse_per_episode=True,uniform_episode_fraction=0.,
        teacher_colored_noise_candidates=0,continuous_learning=True,nominal_evaluation=False,
        generator_update_policy='last_valid',generator_old_dev_metric='monitor_only',
        generator_inference_parameters='ema',final_test_open=False,automatic_extension=False,automatic_retry=False)


def test_profile_is_strict_and_supports_variable_duration():
    from jit_dvgc.generative_bridge.continuation_profile import validate_profile
    for duration in (1,3,5,8):
        p=profile();p['pulse_steps']=duration
        assert validate_profile(p)['pulse_steps']==duration
    for change in ({'unknown':1},{'continuous_learning':False},{'generator_update_policy':'fixed_dev_best'},
                   {'pulse_steps':True},{'pulse_steps':0},{'pulse_steps':401},{'delta_limit':[1]*4}):
        with pytest.raises(ValueError):validate_profile({**profile(),**change})


def test_budget_separates_padding_and_keeps_every_stage_within_cap():
    from jit_dvgc.generative_bridge.continuation_profile import budget_dry_run,profile_budget
    plan=dict(profile=profile(),rounds=2,budgets=profile_budget(2))
    report=budget_dry_run(plan)
    assert report['collection']==dict(charged_maximum=2304,active_prefix_maximum=1344,padding_maximum=960,collection_steps=18)
    assert report['per_round_maximum']<=1500000
    assert plan['budgets']['max_physics']==3000000 and plan['budgets']['max_wall_seconds']==86400
    bad=deepcopy(plan);bad['budgets']['max_physics']=1
    with pytest.raises(ValueError):budget_dry_run(bad)


def test_round_passes_scheduled_profile_into_actual_collection(tmp_path):
    from jit_dvgc.generative_bridge.closed_loop import ClosedLoopRound
    from jit_dvgc.generative_bridge.contracts import file_sha
    seed=tmp_path/'seed.json';seed.write_text('{}')
    runner=ClosedLoopRound.__new__(ClosedLoopRound);runner.root=tmp_path
    runner.spec={**profile(),'continuation':{'seed_support':{'path':str(seed),'sha256':file_sha(seed)}},
                 'round_index':3,'series_id':'pilot','seed':1}
    runner.runtime={'pulse_steps':3,'horizon':400}
    class Collected(Exception):pass
    def measured(name,mode,cfg,maximum):
        assert name=='collection' and mode=='collect' and maximum==2304
        assert cfg['pulse_start_schedule']==[0,5,10,15] and cfg['pulse_batch_mode']=='mixed'
        assert cfg['explorer_admission_v1_2']['uniform_episode_fraction']==0.
        from jit_dvgc.generative_bridge.explorer_admission import validate_collection_options
        validate_collection_options(cfg)
        raise Collected
    runner.measured=measured
    with pytest.raises(Collected):runner._run()
