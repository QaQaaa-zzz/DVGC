from types import SimpleNamespace
import pytest

from jit_dvgc.generative_bridge import source_phase as phase
from jit_dvgc.generative_bridge.protocol import atomic_json


def test_fresh_panel_selection_retains_denominator_and_separates_generator_dev():
    rows=[dict(root_id=str(i),root_episode_id=str(i),data_role='train',label=0) for i in range(50)]
    rows += [dict(root_id='dev',root_episode_id='dev',data_role='generator_dev',label=1)]
    rows += [dict(root_id='terminal',root_episode_id='terminal',data_role='train',label=0,prefix_terminal=True)]
    panel=phase.fresh_panels(rows,seed=99)
    assert len(panel['original_pending'])==50
    assert len(panel['new_roots'])==32
    assert panel==phase.fresh_panels(rows,seed=99)
    assert panel['bootstrap'][0]['root_id']=='dev'
    assert panel['splits']['dev']=='generator_dev'


@pytest.mark.parametrize('invalid',[False,True])
def test_a1_orchestration_preserves_teacher_unknown_stop(tmp_path,monkeypatch,invalid):
    calls=[]
    runner=SimpleNamespace(root=tmp_path,runtime={'evaluation_batch_size':32},source=dict(name='P0',actor_sha256='actor',
        normalizer_sha256='normalizer',xml_sha256='model'),spec=dict(seed=7,
        source_frozen_policy='/frozen',teacher_layout='source_control_in_32_world_batch'),
        protocol='protocol',costs=[],status=lambda *a,**k:None)
    runner.generator=lambda *a,**k:(calls.append('generator') or {'updates':20000})
    runner.smoke=lambda:(calls.append('smoke') or {'status':'passed'})
    runner.teacher_search=lambda x:(calls.append('teachers') or {'root':{'teacher_status':'invalid' if invalid else 'searched_no_solution'}})
    def child(*args,**kwargs):
        calls.append(args[2])
        if args[2]=='seed_support':
            spec=phase.read(args[3])
            assert spec['seed_stride']==10
            assert 'evaluation_batch_size' not in spec
            atomic_json(tmp_path/'seed_support/support.json',{'entries':[{'phase':'upstream'},{'phase':'downstream'}]})
    monkeypatch.setattr(phase,'_measured_child',child)
    monkeypatch.setattr(phase,'prepare_collection_plan',lambda *a,**k:{'batches':[dict(
        collection_spec='collect',collection_output='collection',full_episode=False,
        suffix_spec='suffix',suffix_output='evaluation')]})
    monkeypatch.setattr(phase,'aggregate_collection',lambda *a,**k:{'rows':[]})
    monkeypatch.setattr(phase,'build_training_support',lambda *a,**k:None)
    monkeypatch.setattr(phase,'retention_from_successes',lambda *a,**k:{'role':'train'})
    monkeypatch.setattr(phase,'bootstrap_from_results',lambda *a,**k:{'corpus':{},'dev_fixture':'fixture'})
    monkeypatch.setattr(phase,'read_teacher_traces',lambda *a:[])
    from jit_dvgc.generative_bridge import student_demo_bank
    def bank(*args,**kwargs):
        calls.append('demo');atomic_json(tmp_path/'student_demo_bank/manifest.json',{'count':0});return {'count':0}
    monkeypatch.setattr(student_demo_bank,'build_student_demo_bank',bank)
    if invalid:
        with pytest.raises(ValueError,match='unknown or incomplete'):phase.run_source_phase(runner)
        assert 'demo' not in calls
        assert not (tmp_path/'source_phase_result.json').exists()
    else:
        result=phase.run_source_phase(runner)
        assert result['new_training_transitions']==0
        assert result['collection_plan']==str(tmp_path/'source_collections/plan.json')
        assert calls==['seed_support','collect','evaluate','generator','smoke','teachers','demo']
