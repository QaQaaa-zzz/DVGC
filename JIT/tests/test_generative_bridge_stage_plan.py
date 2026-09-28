import pytest
from jit_dvgc.generative_bridge.stage_plan import make_stage_plan, require_stage, next_incumbent


def test_finite_stages_and_no_automatic_explorer():
    plan=make_stage_plan('source','actor','normalizer')
    assert plan['stages']['A1']['physics_cap']==1820000
    assert plan['stages']['A2']['ppo_transitions']==384000
    assert plan['stages']['B']['execute'] is False
    with pytest.raises(ValueError,match='nominal'):
        require_stage(plan,'A1',{'source_identity_valid':True,'action_parity':True,'nominal_success':False})
    require_stage(plan,'A1',{'source_identity_valid':True,'action_parity':True,'nominal_success':True})
    with pytest.raises(ValueError,match='adopted'):
        require_stage(plan,'B',{'source_identity_valid':True,'action_parity':True,'nominal_success':True})


def test_rejected_candidate_preserves_actor_and_demo_bank():
    before={'actor':'old','demo_bank':'old_bank'}
    rejected=next_incumbent(before,{'actor':'new'},{'adopted':False},'expanded_bank')
    assert rejected=={'actor':'old','demo_bank':'expanded_bank'}
    assert next_incumbent(before,{'actor':'new'},{'adopted':True},'expanded_bank')['actor']=='new'
