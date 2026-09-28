import importlib
import json
from copy import deepcopy
import pytest


def test_teacher_all_unsolved_subset_and_resume_after_generator_failure(tmp_path):
    m=importlib.import_module('jit_dvgc.generative_bridge.protocol')
    support={'entries':[{'key':'a','sampling_weight':.5},{'key':'b','sampling_weight':.5}]}
    events=[]
    def solve():return {'a':{'teacher_status':'searched_no_solution','candidate_count':32}}
    def train(s,d):
        assert s==support and d is None
        events.append('ppo_keep');return {'checkpoint':'student','actor_sha256':'student','normalizer_sha256':'norm'}
    def evaluate(student):
        return {'a':{'student_label':1},'b':{'student_label':1}}
    def accept(student,evaluation):return {'adopted':True,'actor_sha256':'student','normalizer_sha256':'norm'}
    def corpus(*args):return {'new_data':True,'fixture':True}
    fail=[True]
    def update(data):
        if fail[0]:raise RuntimeError('G failed')
        return {'selected':'generator_new'}
    kwargs=dict(support=support,roots=['a','b'],teacher_roots=['a'],solve=solve,
        train=train,evaluate=evaluate,accept=accept,export=corpus,update=update,
        source={'actor':'old','generator':'old_g'},budgets={'teacher':12800,'student':128000,'evaluation':800},
        maximum_interactions=141600)
    with pytest.raises(RuntimeError):m.run_round(tmp_path,**kwargs)
    assert events==['ppo_keep'] and not (tmp_path/'current_source.json').exists()
    fail[0]=False
    result=m.run_round(tmp_path,**kwargs)
    assert events==['ppo_keep']
    assert result['outcomes'][0]['outcome_class']=='teacher_unsolved_student_succeeded'
    assert result['outcomes'][1]['teacher_found'] is None
    assert (tmp_path/'current_source.json').exists()
    assert list(tmp_path.glob('errors/*.json'))
    changed=deepcopy(kwargs);changed['support']['entries'][0]['sampling_weight']=.9
    with pytest.raises(ValueError):m.run_round(tmp_path,**changed)


def test_teacher_engineering_error_not_fallback_and_budget_reserves(tmp_path):
    m=importlib.import_module('jit_dvgc.generative_bridge.protocol')
    with pytest.raises(ValueError,match='budget'):
        m.reserve_round({'teacher':12800,'student':128000,'evaluation':800},12800)
    with pytest.raises(ValueError):m.validate_teacher_results(['a','b'],['a'],{'a':{'teacher_status':'invalid'}})
    assert m.validate_teacher_results(['a','b'],['a'],{'a':{'teacher_status':'searched_no_solution'}})['b']['teacher_status']=='not_scheduled'


def test_preparation_keeps_execution_off_and_matches_total_budget():
    import yaml
    from pathlib import Path
    m=importlib.import_module('jit_dvgc.generative_bridge.reporting')
    spec=yaml.safe_load((Path(__file__).parents[1]/'configs/generative_bridge_v1_1.yaml').read_text())
    plan=m.prepare(spec)
    assert plan['execute'] is False and plan['status']=='needs_input_resolution'
    totals={v['physical_interaction_ceiling'] for v in plan['comparison_arms']}
    assert totals=={680800}
    assert plan['generator_comparison']['fixed_same_tail_actor_for_both']
    assert plan['cost_ledger']['actual_physics_interactions']==0


def test_corpus_receipts_and_real_generator_retry_preserve_failed_attempt(tmp_path):
    import numpy as np
    import jax
    import jax.numpy as jp
    from .test_generative_bridge_contracts import trace
    from jit_dvgc.generative_bridge.feedback_data import build_corpus
    m=importlib.import_module('jit_dvgc.generative_bridge.artifacts')
    d=importlib.import_module('jit_dvgc.generative_bridge.diffusion')
    corpus=build_corpus([],[],[trace()],adoption={'adopted':True,'actor_sha256':'actor','normalizer_sha256':'norm'},
                       expected={},splits={'parent':'generator_train'})
    receipt=m.save_corpus(corpus,tmp_path/'corpus')
    json.dumps(receipt,allow_nan=False)
    loaded=m.load_corpus(receipt)
    np.testing.assert_array_equal(loaded['groups']['actor_new'][0]['arrays']['actor_observation_before'],
                                 corpus['groups']['actor_new'][0]['arrays']['actor_observation_before'])
    state=d.create_train_state({'w':jp.array(.1)},jax.random.PRNGKey(0),{'mean':jp.zeros(76),'std':jp.ones(76)})
    dev=(jp.zeros((1,76)),jp.zeros((1,16,4)),jp.zeros((1,),int),jp.ones((1,16,4)))
    failing=[True]
    def predict(p,x,obs,k):return jp.ones_like(x)*(jp.nan if failing[0] else p['w'])
    stage=m.GeneratorUpdateStage(tmp_path/'update',state=state,predict=predict,dev_fixture=dev,
        identity={'fixture':True},updates=1,max_wall_seconds=30,batch_size=1)
    with pytest.raises(FloatingPointError):stage(receipt)
    assert (tmp_path/'update/attempt_0000').exists()
    failing[0]=False
    result=stage(receipt)
    assert result['status']=='completed' and result['attempt']==1
    assert (tmp_path/'update/attempt_0001/generator_selection.json').exists()
    assert (tmp_path/'update/attempt_0000/cost_receipt.json').exists()
    assert stage(receipt)==result


def test_round_uses_actual_demo_counts_and_never_overwrites_teacher(tmp_path):
    m=importlib.import_module('jit_dvgc.generative_bridge.protocol')
    kwargs=dict(support={'entries':[{'key':'a','sampling_weight':1.}]},roots=['a'],teacher_roots=['a'],
        solve=lambda:{'a':{'teacher_status':'verified_solution','demo':{'count':18}}},
        train=lambda *a:{'checkpoint':'p','actor_sha256':'p','normalizer_sha256':'n','demo_samples_by_root':{'a':4}},
        evaluate=lambda *a:{'a':{'student_label':1}},
        accept=lambda *a:{'adopted':True,'actor_sha256':'p','normalizer_sha256':'n'},
        export=lambda *a:{'new_data':False},update=lambda c:{'selected':'old_g'},
        source={'actor':'old','generator':'old_g'},budgets={'teacher':12800,'student':128000,'evaluation':400},
        maximum_interactions=141200)
    result=m.run_round(tmp_path/'valid',**kwargs)
    row=result['outcomes'][0]
    assert row['round_used_teacher_demo'] and row['n_direct_demo_samples_used']==4
    kwargs['evaluate']=lambda *a:{'a':{'student_label':1,'teacher_status':'searched_no_solution'}}
    with pytest.raises(ValueError):m.run_round(tmp_path/'bad',**kwargs)


def test_generator_receipt_replay_rejects_corrupted_payload(tmp_path):
    import jax
    import jax.numpy as jp
    from jit_dvgc.generative_bridge.diffusion import create_train_state,save_state
    from jit_dvgc.generative_bridge.contracts import file_sha
    m=importlib.import_module('jit_dvgc.generative_bridge.artifacts')
    state=create_train_state({'w':jp.array(1.)},jax.random.PRNGKey(0),{'mean':jp.zeros(76),'std':jp.ones(76)})
    save_state(tmp_path/'g',state,{'fixture':True})
    manifest=tmp_path/'g/manifest.json';receipt={'checkpoint_manifest':str(manifest),'checkpoint_manifest_sha256':file_sha(manifest)}
    m.validate_generator_receipt(receipt)
    (tmp_path/'g/state.msgpack').write_bytes(b'corrupt')
    with pytest.raises(ValueError):m.validate_generator_receipt(receipt)
