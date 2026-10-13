"""Explicit engineering recovery preserves failed artifacts and scientific state."""
import json
from pathlib import Path
import pytest
from jit_dvgc.generative_bridge.protocol import atomic_json
from jit_dvgc.generative_bridge.contracts import file_sha,digest


def test_reconcile_replaces_whole_worker_reservation_with_completed_and_partial_upper():
    from jit_dvgc.generative_bridge.recovery_campaign import reconcile_teacher_costs
    parent=[dict(stage='collection',charged_interactions=2304,charged_updates=0,phase='completed'),
        dict(stage='teacher_worker',charged_interactions=2662400,charged_updates=0,phase='failed')]
    worker=[dict(stage='teacher_0000',charged_interactions=1207,charged_updates=0,phase='completed'),
        dict(stage='review_batch_0000_1',charged_interactions=6800,charged_updates=0,phase='running')]
    original=json.dumps(parent,sort_keys=True)
    stages={'teacher_0000':{'charged_interactions':1207}}
    result=reconcile_teacher_costs(parent,worker,stages)
    assert sum(x['charged_interactions'] for x in result)==2304+1207+6800
    assert result[-1]['original_worker_reservation']==2662400
    assert result[-1]['partial_upper_interactions']==6800
    assert json.dumps(parent,sort_keys=True)==original
    with pytest.raises(ValueError,match='completed'):
        reconcile_teacher_costs(parent,worker,{})
    worker[0]['charged_interactions']=1208
    with pytest.raises(ValueError,match='completed'):
        reconcile_teacher_costs(parent,worker,stages)


def test_resume_collection_uses_saved_outputs_without_new_charge_and_rejects_changed_rng(tmp_path):
    from jit_dvgc.generative_bridge.recovery_campaign import reuse_completed_collection
    old=tmp_path/'old';new=tmp_path/'new';collection=old/'collection';collection.mkdir(parents=True)
    cfg=dict(seed=8113002,round_index=1,delta_limit=[.25]*4,neighborhood_map=str(old/'neighborhood_map.json'))
    atomic_json(collection/'hyperparameters.json',cfg)
    atomic_json(collection/'status.json',dict(phase='completed',charged_interactions=2304))
    atomic_json(collection/'candidates.json',[])
    ref=dict(output=str(collection),previous_round=str(old),
        files={str(p):file_sha(p) for p in collection.glob('*.json')},charged_interactions=2304)
    actual={**cfg,'neighborhood_map':str(new/'neighborhood_map.json')}
    assert reuse_completed_collection(ref,actual,new)==collection
    actual['seed']+=1
    with pytest.raises(ValueError,match='collection protocol'):reuse_completed_collection(ref,actual,new)
    actual['seed']-=1;(collection/'candidates.json').write_text('[1]')
    with pytest.raises(ValueError,match='changed'):reuse_completed_collection(ref,actual,new)


def test_resume_round_spec_preserves_seed_and_source_bank_and_carries_old_cost(tmp_path,monkeypatch):
    from jit_dvgc.generative_bridge.recovery_campaign import resume_round_spec
    import jit_dvgc.generative_bridge.production as production
    root=tmp_path/'new';root.mkdir();old=tmp_path/'old';old.mkdir()
    original=dict(output=str(old),seed=8113002,round_index=1,series_id='original_scientific_campaign',
        source_runtime=dict(bank='/immutable/original_bank',seed=8113002),
        locks={},implementation_commit='old',implementation_files={},budgets=dict(max_physics=3500000),
        teacher_recovery_authorization={'path':'old','sha256':'h'},campaign_output='/old')
    atomic_json(old/'production.json',original)
    costs=[dict(stage='collection',charged_interactions=2304)]
    info=dict(previous_round=str(old),previous_spec_sha256=file_sha(old/'production.json'),
        carried_round_costs=costs,previous_contract_identity='oldcontract',stages={},
        collection={'output':str(old/'collection')},teacher_worker={'previous':'oldworker','stages':{}},completed_teacher_roots={})
    d=dict(campaign_output=str(root),repository='/repo',implementation_commit='new',locks={},engineering_recovery=info)
    atomic_json(root/'declaration.json',d)
    monkeypatch.setattr(production,'implementation_files',lambda repo:{})
    result=resume_round_spec(d)
    assert result['seed']==8113002 and result['source_runtime']==original['source_runtime']
    assert result['series_id']==original['series_id'] and result['round_index']==1
    assert result['implementation_commit']=='new'
    assert read_json(root/'round_0001/costs.json')==costs
    assert read_json(old/'production.json')==original
    assert result['recovery']['previous_contract_identity']=='oldcontract'


def read_json(path):return json.loads(Path(path).read_text())


def test_collection_reuse_compares_original_raw_spec_not_added_runtime_defaults(tmp_path):
    from jit_dvgc.generative_bridge.recovery_campaign import reuse_completed_collection
    old=tmp_path/'old';new=tmp_path/'new';out=old/'collection';out.mkdir(parents=True)
    cfg={'seed':8113002,'round_index':1};spec=old/'collection_spec.json';atomic_json(spec,cfg)
    atomic_json(out/'hyperparameters.json',{**cfg,'inference_precision':'highest','pulse_descent_clearance':0.})
    atomic_json(out/'status.json',{'phase':'completed','charged_interactions':2304})
    ref=dict(output=str(out),previous_round=str(old),spec_path=str(spec),
        files={str(p):file_sha(p) for p in [spec,*out.glob('*.json')]},charged_interactions=2304)
    assert reuse_completed_collection(ref,cfg,new)==out
