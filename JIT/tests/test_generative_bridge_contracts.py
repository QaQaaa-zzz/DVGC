"""CPU fixtures only: no physical success or learned policy is asserted."""
from copy import deepcopy
import importlib
import numpy as np
import pytest


def module(name):
    return importlib.import_module('jit_dvgc.generative_bridge.' + name)


def test_teacher_nullable_matrix_and_no_causal_claim():
    out = module('outcomes')
    for status, found in [('not_scheduled', None), ('incomplete', None), ('invalid', None),
                          ('searched_no_solution', False), ('verified_solution', True)]:
        row = out.root_outcome({'root_id':'r', 'teacher_status':status, 'student_label':1,
            'student_adopted':True, 'n_direct_demo_samples_used':0}, round_demo_samples_used=25)
        assert row['teacher_found'] is found
        assert row['round_used_teacher_demo'] is True
        assert row['causal_attribution'] == 'not_identified_in_single_run'
    assert out.root_outcome({'root_id':'r', 'teacher_status':'searched_no_solution',
        'student_label':1, 'student_adopted':True})['outcome_class'] == 'teacher_unsolved_student_succeeded'


def test_pending_membership_and_weights_locked_not_teacher_subset():
    m = module('contracts')
    support = {'entries':[{'key':'a','sampling_weight':.25}, {'key':'b','sampling_weight':.75}],
               'pending_fraction':.25}
    lock = m.support_fingerprint(support)
    m.validate_pending_support_unchanged(lock, deepcopy(support))
    for bad in [{'entries':support['entries'][:1]},
                {**support, 'entries':[{'key':'a','sampling_weight':.5}, support['entries'][1]]}]:
        with pytest.raises(ValueError): m.validate_pending_support_unchanged(lock,bad)


def trace(origin='adopted_actor', role='train', split='generator_train', n=18):
    obs=np.arange((n+1)*76,dtype=np.float32).reshape(n+1,76)/1000
    return {'metadata':dict(root_id='r', root_episode_id='parent', root_context_sha256='context',
        actor_sha256='actor', normalizer_sha256='norm', model_sha256='model',
        protocol_sha256='protocol', origin_type=origin, role=role, inherited_split=split,
        full_success=True, complete=True, prior_teacher_status='searched_no_solution',
        source_actor_sha256='old', success_criterion='stable_forward_recovery'),
        'arrays':dict(actor_observation_before=obs[:-1].copy(),actor_observation_after=obs[1:].copy(),
            normalized_action_executed=np.zeros((n,4),np.float32),
            valid_mask=np.ones(n,bool), done=np.arange(n)==n-1,
            success=np.arange(n)==n-1, failure=np.zeros(n,bool),timeout=np.zeros(n,bool),
            phase_before=np.zeros(n,int), action_origin=np.array(['actor_only']*n))}


def test_adopted_actor_unsolved_train_success_enters_without_bridge_fields():
    m=module('feedback_data'); t=trace(); original=deepcopy(t)
    row=m.admit_trace(t, adoption={'adopted':True,'actor_sha256':'actor','normalizer_sha256':'norm'},
        expected={'model_sha256':'model','protocol_sha256':'protocol'}, splits={'parent':'generator_train'})
    assert row['eligible'] and row['window_count']==3
    assert row['bridge_verified_against_actor_sha256'] is None
    assert t['metadata']==original['metadata']
    assert row['prior_teacher_status']=='searched_no_solution'


@pytest.mark.parametrize('change,reason', [({'role':'solver_dev'},'role'),
    ({'inherited_split':'generator_dev'},'split'), ({'full_success':False},'success')])
def test_excluded_data(change,reason):
    m=module('feedback_data');t=trace();t['metadata'].update(change)
    row=m.admit_trace(t,adoption={'adopted':True,'actor_sha256':'actor','normalizer_sha256':'norm'},
        expected={'model_sha256':'model','protocol_sha256':'protocol'},splits={'parent':t['metadata']['inherited_split']})
    assert not row['eligible'] and reason in row['reason']


def test_unadopted_missing_preobs_and_identity_drift():
    m=module('feedback_data');t=trace()
    assert 'adopt' in m.admit_trace(t,adoption={'adopted':False},expected={},splits={'parent':'generator_train'})['reason']
    del t['arrays']['actor_observation_before']
    assert 'preobs' in m.admit_trace(t,adoption={'adopted':True,'actor_sha256':'actor','normalizer_sha256':'norm'},expected={},splits={'parent':'generator_train'})['reason']
    t=trace()
    with pytest.raises(ValueError): m.admit_trace(t,adoption={'adopted':True,'actor_sha256':'other','normalizer_sha256':'norm'},expected={},splits={'parent':'generator_train'})


def test_windows_reject_discontinuity_and_never_pad():
    m=module('data');t=trace(n=15)
    assert len(m.build_action_windows(t)['actions'])==0
    t=trace();t['arrays']['actor_observation_after'][0,0]+=1
    with pytest.raises(ValueError,match='continu'):m.build_action_windows(t)


def test_empty_mix_and_only_actor_new():
    m=module('feedback_data')
    assert m.realized_mix({'history':4,'teacher_new':0,'actor_new':2})=={'history':2/3,'teacher_new':0.,'actor_new':1/3}
    assert m.realized_mix({'history':0,'teacher_new':0,'actor_new':2})['actor_new']==1
    assert not any(m.realized_mix({'history':0,'teacher_new':0,'actor_new':0}).values())


def test_optional_demo_never_reads_empty_but_corruption_errors(tmp_path):
    m=module('student')
    assert m.load_optional_demo(None) is None
    assert m.load_optional_demo({'schema':'jit_bridge_demo_v1_1','count':0,'entries':[]}) is None
    with pytest.raises(ValueError):m.load_optional_demo({'schema':'jit_bridge_demo_v1_1','count':0,'entries':[{}]})
    with pytest.raises((ValueError,FileNotFoundError)):m.load_optional_demo({'schema':'jit_bridge_demo_v1_1','count':1,'path':str(tmp_path/'missing.npz'),'sha256':'x'})


def test_reward_spec_strict_teacher_independence_and_unknown():
    m=module('rewards')
    base=dict(pulse_cells=['c'],stage_reached=True,initial_label=0,label=1,
        learning_attempted=True,successor_adopted=True)
    for status in ['not_scheduled','searched_no_solution','verified_solution']:
        reward,mask,_,_=m.feedback([{**base,'teacher_status':status}],[])
        assert mask[0] and reward[0]==2.02
    assert m.feedback([{**base,'successor_adopted':False}],[])[0][0]==.02
    assert m.feedback([{**base,'label':0}],[])[0][0]==-.08
    assert not m.feedback([{**base,'successor_adopted':None}],[])[1][0]
    assert m.feedback([{**base,'prefix_terminal':True,'prefix_physical_failure':True,'pulse_applied_steps':1}],[])[0][0]==-2


def test_physical_pulse_failure_without_cell_still_penalized_once():
    m=module('rewards')
    r=dict(pulse_cells=[],stage_reached=True,prefix_terminal=True,prefix_physical_failure=True,
           pulse_applied_steps=1,initial_label=None,label=None)
    reward,mask,_,parts=m.feedback([r],[])
    assert mask[0] and reward[0]==-2 and parts['novelty']==[0.]


def test_five_arm_budgets_sum_exactly_and_no_silent_template_drift():
    import yaml
    from pathlib import Path
    spec=yaml.safe_load((Path(__file__).parents[1]/'configs/generative_bridge_v1_1.yaml').read_text())
    for arm in module('reporting').prepare(spec)['comparison_arms']:
        assert sum(arm[k] for k in ('student_transitions','search_interactions',
            'common_diagnostic_and_acceptance','additional_original_acquisition'))==arm['physical_interaction_ceiling']
    for path,value in [('generator.train_noise_steps',99),('generator.ddim_eta',1.),
                       ('data.bidirectional_corpus.actor_requires_adoption',False),
                       ('student.retention_coefficient',-1.)]:
        bad=deepcopy(spec);keys=path.split('.');node=bad
        for key in keys[:-1]:node=node[key]
        node[keys[-1]]=value
        with pytest.raises(ValueError):module('contracts').validate_spec(bad)


def test_teacher_demo_export_empty_and_actual_root_weights(tmp_path):
    m=module('data')
    empty=m.export_student_demonstrations([],tmp_path/'empty')
    assert module('student').load_optional_demo(empty) is None
    assert not list((tmp_path/'empty').glob('*.npz'))
    t=trace(origin='verified_teacher')
    t['metadata'].update(teacher_status='verified_solution',verification_receipt_sha256='verified')
    t['arrays']['action_origin'][:16]='bridge_prefix'
    t['arrays']['action_origin'][16:]='source_tail'
    # Allocate a nontruncating unicode array for the longer teacher origins.
    t['arrays']['action_origin']=np.array(['bridge_prefix']*16+['source_tail']*2)
    manifest=m.export_student_demonstrations([t],tmp_path/'demo')
    obs,action,weights=module('student').load_optional_demo(manifest)
    assert len(obs)==18 and weights[:16].sum()==pytest.approx(.5)
    assert weights[16:].sum()==pytest.approx(.5)
    assert manifest['sample_roots']==['r']*18


def test_new_source_group_cannot_launder_bootstrap_or_unadopted_actor():
    m=module('feedback_data')
    with pytest.raises(ValueError):
        m.build_corpus([],[],[trace(origin='bootstrap_actor')],adoption={'adopted':False},
                       expected={},splits={'parent':'generator_train'})


@pytest.mark.parametrize('field,value',[('success',np.nan),('done',2.),('phase_before',.5)])
def test_invalid_trace_flags_never_admitted(field,value):
    t=trace();t['arrays'][field]=t['arrays'][field].astype(float);t['arrays'][field][-1]=value
    with pytest.raises(ValueError):module('feedback_data').admit_trace(t,
        adoption={'adopted':True,'actor_sha256':'actor','normalizer_sha256':'norm'},expected={},
        splits={'parent':'generator_train'})


def test_opt_in_reward_runtime_routing_preserves_legacy():
    from jit_dvgc.pulse_exploration import pulse_feedback
    from jit_dvgc.generative_bridge.rewards import WEIGHTS
    r=dict(pulse_cells=['c'],stage_reached=True,initial_label=0,label=1,
           learning_attempted=True,successor_adopted=None)
    reward,mask,_,_=pulse_feedback([r],[],WEIGHTS,quality_mode='discovery_conversion',
                                  reward_contract='jit_bidirectional_reward_v1_1')
    assert not mask[0] and reward[0]==0
    with pytest.raises(ValueError):pulse_feedback([r],[],{**WEIGHTS,'conversion':2.},
        quality_mode='discovery_conversion',reward_contract='jit_bidirectional_reward_v1_1')


def test_adopted_history_retains_original_acceptance_receipt(tmp_path):
    m=module('feedback_data');art=module('artifacts')
    first=m.build_corpus([],[],[trace()],adoption={'adopted':True,'actor_sha256':'actor','normalizer_sha256':'norm'},
        expected={},splits={'parent':'generator_train'})
    saved=art.load_corpus(art.save_corpus(first,tmp_path/'corpus'))
    second=m.build_corpus(saved['groups']['actor_new'],[],[],adoption={'adopted':False},
        expected={},splits={'parent':'generator_train'})
    assert len(second['groups']['history'])==1
    assert second['groups']['history'][0]['adoption']['actor_sha256']=='actor'


def test_full_32_candidate_failure_is_distinct_from_incomplete():
    m=module('teacher')
    candidates=[{'candidate_id':str(i),'full_success':False,'action_delta_cost':1.,'task_return':0.} for i in range(32)]
    assert m.search_status(candidates,expected_count=32)=='searched_no_solution'
    assert m.search_status(candidates[:-1],expected_count=32)=='incomplete'
    candidates[-1]['invalid']=True
    assert m.search_status(candidates,expected_count=32)=='invalid'


def test_source_only_recheck_success_is_not_a_teacher_rescue():
    m=module('teacher')
    with pytest.raises(ValueError,match='source'):
        m.select_teacher([{'candidate_id':'0','kind':'source_only','full_success':True,
                           'action_delta_cost':0.,'task_return':100.}])
