from copy import deepcopy
import pytest
from jit_dvgc.generative_bridge.protocol import atomic_json
from jit_dvgc.generative_bridge.contracts import file_sha


def test_budget_is_finite_and_scales_exactly():
    from jit_dvgc.generative_bridge.closed_loop import continuation_budget
    b=continuation_budget(200)
    assert b['student_transitions']==25600000
    assert b['generator_updates']==400000
    assert b['max_physics']==300000000
    assert b['max_wall_seconds']==604800
    for invalid in (0,201,True,1.5):
        with pytest.raises(ValueError):continuation_budget(invalid)


@pytest.mark.parametrize('learned_only',[False,True])
def test_continuation_runs_next_rounds_without_replaying_baseline(tmp_path,monkeypatch,learned_only):
    import jit_dvgc.generative_bridge.closed_loop as loop
    from jit_dvgc import probe_bank
    from jit_dvgc.generative_bridge import worker
    atomic_json(tmp_path/'parent/production.json',{'source_runtime':{'bank':str(tmp_path/'bank.json'),'reward_mode':'same'}})
    atomic_json(tmp_path/'parent/stages/fresh_source_phase.json',{'result':{'retention_ref':'old'}})
    atomic_json(tmp_path/'bank.json',dict(task='jump',max_ticks=400,label_interaction_budget=100,max_candidates_per_process=32))
    actor=dict(frozen_policy=str(tmp_path/'actor.json'),policy=dict(name='accepted',actor_sha256='a'))
    atomic_json(tmp_path/'actor.json',{})
    bundle=dict(round=7,actor=actor,explorer='saved_E',explorer_sha256='Ehash',generator={'saved':'G'},
        dev_fixture='dev',dev_fixture_sha256='d',neighborhood_history=[{'old':1}],tail_lineage=['old_tail'])
    atomic_json(tmp_path/'bundle.json',bundle)
    panel={'a':dict(condition={'same':1},labels=[1,0])}
    atomic_json(tmp_path/'baseline.json',panel);atomic_json(tmp_path/'initial.json',panel)
    plan=dict(output=str(tmp_path/'run'),schema='jit_bridge_experimental_continuation_v1',rounds=2,
        round_offset=7,experimental_adoption=True,formal_adoption=False,budgets=loop.continuation_budget(2),
        authorization='user_requested_additional_rounds',repository=str(tmp_path),implementation_commit='commit',
        parent_campaign=str(tmp_path/'parent'),baseline_frozen_policy=str(tmp_path/'actor.json'),seed=10,locks={},implementation_files={},
        prior_physics_charged=55,minimum_free_disk_bytes=20*1024**3,
        continuation_bundle={'path':str(tmp_path/'bundle.json'),'sha256':file_sha(tmp_path/'bundle.json')},
        reference_panels={'baseline':str(tmp_path/'baseline.json'),'initial':str(tmp_path/'initial.json')},
        execution_gate={'kind':'gpu_shared'},neighborhood={'enabled':True},
        uniform_episode_fraction=0.,teacher_colored_noise_candidates=0)
    if not learned_only:
        plan.pop('uniform_episode_fraction');plan.pop('teacher_colored_noise_candidates')
    (tmp_path/'run').mkdir()
    monkeypatch.setattr(loop,'start_notifications',lambda p:None)
    monkeypatch.setattr(loop,'implementation_identity',lambda p:'commit')
    monkeypatch.setattr(worker,'source_payload',lambda p:({'xml_sha256':'xml','actor_sha256':'a','normalizer_sha256':'n'},None))
    monkeypatch.setattr(probe_bank,'lock_probe_bank',lambda b,p:atomic_json(p,b))
    seen=[]
    class FakeRound:
        def __init__(self,s):self.spec=s
        def stress(self,*a):raise AssertionError('baseline must not replay')
        def run(self):
            s=self.spec;seen.append(deepcopy(s));v=deepcopy(s['continuation'])
            v.update(round=s['round_index'],stress=panel,explorer='updated_E',explorer_sha256='updated_hash')
            atomic_json(__import__('pathlib').Path(s['output'])/'costs.json',[{'charged_interactions':100,'charged_updates':0}])
            return v
    monkeypatch.setattr(loop,'ClosedLoopRound',FakeRound)
    result=loop.run_closed_loop(plan)
    assert [x['round_index'] for x in seen]==[8,9]
    assert seen[0]['continuation']['explorer']=='saved_E'
    assert seen[1]['continuation']['explorer']=='updated_E'
    assert all(x['continuation']['generator']=={'saved':'G'} for x in seen)
    assert seen[0]['continuation']['tail_lineage']==['old_tail']
    assert all(x['budgets']['max_physics']==1500000 for x in seen)
    assert all(x['uniform_episode_fraction']==(0. if learned_only else .2)
               and x['teacher_colored_noise_candidates']==(0 if learned_only else 15) for x in seen)
    assert all(x['teacher_layout']==('source_control_in_candidate_batch' if learned_only
                                    else 'source_control_in_32_world_batch') for x in seen)
    assert result['round']==9
    with pytest.raises(ValueError,match='explicit recovery'):loop.run_closed_loop(plan)
