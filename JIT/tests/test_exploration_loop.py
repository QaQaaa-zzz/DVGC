"""Outer orchestration fixtures: no simulators, trainers or real subprocesses."""
import json
import sys
import types
from pathlib import Path

import pytest

from jit_dvgc import exploration_loop as loop
from jit_dvgc import gated_execution, exploration_pool


def test_budget_includes_matched_arms_panels_and_growing_bank():
    spec = {"rounds": 2, "policy_steps": 3200, "pending_limit": 8, "stage_timeout_seconds": 10,
            "explorer": {"num_envs": 2, "batches": 3, "max_candidates_per_episode": 4,
                         "horizon": 400, "evaluation_episodes": 2}}
    result = loop.budget_contract(spec, 2)
    assert result['candidate_limit_per_arm_per_round'] == 32
    assert result['rounds'][0]['forward'] == 8000
    assert result['rounds'][0]['baseline_suffix'] == 2 * 32 * 2 * 400
    assert result['rounds'][1]['baseline_suffix'] == 2 * 64 * 3 * 400
    assert result['maximum_interactions'] == sum(row['maximum_interactions'] for row in result['rounds'])
    with pytest.raises(ValueError, match='evaluator'):
        loop.budget_contract(spec, 0)
    spec['rounds'] = 3
    with pytest.raises(ValueError):
        loop.budget_contract(spec, 2)


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    def member(name, iteration):
        return dict(name=name, roles=['proposer','evaluator'], frozen_policy='/fixture/'+name,
                    policy=dict(actor_sha256=name,iteration=iteration))
    bank = dict(members=[member('pi_start',0)], task='fixture', max_ticks=400,
                label_interaction_budget=999999, max_candidates_per_process=1)
    monkeypatch.setattr(loop, 'load_probe_bank', lambda _: bank)
    monkeypatch.setattr(loop, '_file_sha', lambda _: 'fixture-sha')
    monkeypatch.setattr(loop, 'export_evidence', lambda *args: None)
    monkeypatch.setattr(loop, 'import_candidates', lambda *args, **kwargs: {'pending': True})
    monkeypatch.setattr(loop, 'pending_rows', lambda pool, limit: [{'pending':True}] if pool['pending'] else [])
    monkeypatch.setattr(exploration_pool, 'select_pending', lambda pool, limit: [{}] if pool['pending'] else [])
    monkeypatch.setattr(loop, 'transition_counts', lambda *args: {'verified_root_cells':1})
    trained = []
    def make_config(support, frozen, bootstrap, path, run_id, iteration, steps, seed, **kwargs):
        trained.append(frozen)
        loop.write(path, {'steps':steps})
    monkeypatch.setitem(sys.modules, 'jit_dvgc.iterative_probe_training', types.SimpleNamespace(
        candidate_support_view=lambda *args, **kwargs: {}, make_config=make_config))
    monkeypatch.setitem(sys.modules, 'jit_dvgc.exploration_loop_support', types.SimpleNamespace(
        append_witnessed_support=lambda support, pool, path: support))
    frozen_names = []
    def freeze(directory, **kwargs):
        frozen_names.append(kwargs['name'])
        loop.write(directory/'frozen_unified_policy.json', {'name':kwargs['name']})
    monkeypatch.setitem(sys.modules, 'jit_dvgc.unified_policy_freeze', types.SimpleNamespace(
        freeze_development_checkpoint=freeze))
    def lock(spec, path):
        result = {**bank, 'members':[member('pi_start',0)] + [member(name,i+1) for i,name in enumerate(frozen_names)]}
        loop.write(path,result)
        return result
    monkeypatch.setattr(loop,'lock_probe_bank',lock)
    witness = tmp_path/'witness.json'
    witness.write_text('{}')
    spec = dict(schema=loop.SCHEMA, role='TRAIN',final_test_used=False, bank='/fixture/bank.json',
        proposer='pi_start',witnessed_support=str(witness),bootstrap_config='/fixture/bootstrap.json',
        rounds=2,policy_steps=3200,pending_limit=8,pending_fraction=.25,stage_timeout_seconds=10,
        policy_seed=1,python='/fake/python',repo=str(tmp_path),input_files={'/fixture/input':'fixture-sha'},
        source_locks={'/fixture/source':'fixture-sha'},gate={'status_path':'/external/pipeline_status.json'},
        explorer=dict(num_envs=1,batches=1,max_candidates_per_episode=1,horizon=400,evaluation_episodes=1,seed=7))
    spec['maximum_interactions']=loop.budget_contract(spec,1)['maximum_interactions']
    path=tmp_path/'spec.json';loop.write(path,spec)
    launched=[]
    controls={'no_pending':False,'fail_stage':None,'blocked':False}
    def child(plan_path, output, wait):
        plan=loop.read(plan_path);stage=plan['stages'][0];launched.append(plan)
        if controls['fail_stage']==len(launched):
            raise RuntimeError('late fixture failure')
        if controls['blocked']:
            return {'phase':'blocked'}
        args=stage['argv']
        if stage['name']=='explore':
            out=Path(args[args.index('--output')+1])
            loop.write(out/'status.json',{'phase':'completed','charged_interactions':10})
        elif stage['name']=='train_policy':
            run_id=args[args.index('--run-id')+1]
            out=Path(stage['env']['JIT_RUN_ROOT'])/run_id
            loop.write(out/'formal_report.json',{'completed_training_transitions':3200,'train_panel_interactions':1600})
        else:
            out=Path(args[args.index('--output')+1])
            loop.write(out/'status.json',{'status':'selection_complete','charged_interactions':2})
            loop.write(out/'pool.json',{'pending':not controls['no_pending']})
        return {'phase':'completed'}
    monkeypatch.setattr(gated_execution,'run_gated_plan',child)
    return path,tmp_path/'output',spec,launched,controls,trained


def test_exact_two_rounds_change_current_pi_and_propagate_gate(fixture):
    path,out,spec,launched,controls,trained=fixture
    result=loop.run(path,out)
    assert result['phase']=='completed' and result['completed_rounds']==2
    assert len(launched)==14
    assert all(plan['gate']==spec['gate'] for plan in launched)
    assert all(plan['stages'][0]['env']['JAX_PLATFORMS']=='cuda,cpu' for plan in launched)
    explorers=[loop.read(Path(p['stages'][0]['argv'][3])) for p in launched if p['stages'][0]['name']=='explore']
    assert [e['proposer'] for e in explorers]==['pi_start','pi_start','output_checkpoint_000','output_checkpoint_000']
    assert trained==['/fixture/pi_start','/fixture/output_checkpoint_000']
    assert all(cost['accounting']=='actual_completed' for cost in result['costs'])
    assert result['charged_interactions']<=result['maximum_interactions']


def test_no_pending_stops_before_training(fixture):
    path,out,spec,launched,controls,trained=fixture
    controls['no_pending']=True
    result=loop.run(path,out)
    assert result['phase']=='stopped_no_pending'
    assert len(launched)==4 and not trained
    assert not (out/'round_001').exists()


def test_late_failure_keeps_completed_actuals_and_failed_reservation(fixture):
    path,out,spec,launched,controls,trained=fixture
    controls['fail_stage']=3
    with pytest.raises(RuntimeError,match='late fixture failure'):
        loop.run(path,out)
    result=loop.read(out/'pipeline_status.json')
    assert result['phase']=='error' and result['no_automatic_retry']
    assert len(launched)==3
    assert [c['accounting'] for c in result['costs']]==['actual_completed','actual_completed','reserved_until_completion']
    assert all('wall_seconds' in c for c in result['costs'])
    assert result['charged_interactions']==12+result['costs'][-1]['maximum_interactions']


def test_blocked_child_does_not_read_missing_artifacts(fixture):
    path,out,spec,launched,controls,trained=fixture
    controls['blocked']=True
    with pytest.raises(RuntimeError,match='phase=blocked'):
        loop.run(path,out)
    assert len(launched)==1
    assert loop.read(out/'pipeline_status.json')['phase']=='error'


def test_generated_argument_files_are_locked(fixture):
    path,out,spec,launched,controls,trained=fixture
    loop.run(path,out)
    for plan in launched:
        argv=plan['stages'][0]['argv']
        for position,arg in enumerate(argv[:-1]):
            if arg in ('--spec','--config','--pool','--bank'):
                assert str(Path(argv[position+1]).resolve()) in plan['input_files']


def test_reused_arrival_contract_rejects_different_sampling(tmp_path):
    source=tmp_path/'arrivals';source.mkdir()
    expected={'seed':12,'num_envs':8,'baseline_cells':'new'}
    loop.write(source/'declaration.json',{'spec':{**expected,'baseline_cells':'old'}})
    loop.write(source/'status.json',{'phase':'completed','charged_interactions':19200})
    loop.write(tmp_path/'old',{})
    assert loop.validate_reused_arrivals(source,expected)==19200
    with pytest.raises(ValueError,match='contract'):
        loop.validate_reused_arrivals(source,{**expected,'seed':13})
