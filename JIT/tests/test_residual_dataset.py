import copy
import numpy as np
import pytest
from jit_dvgc.residual_dataset import link_action_pairs, split_groups


def fixture():
    rows=[]
    for i in range(1,5):
        rows.append(dict(candidate_id=str(i),trajectory_id='t',trajectory_step=i,
            state_sha256=str(i)*64,snapshot_context_sha256=str(i+4)*64,
            protocol_sha256='a'*64,phase='upstream',
            jump_start_reachability=dict(perturbation_anchor_x_m=2.9,lookback_m=.15),
            perturbation=dict(nominal_actions=[[0.]*4]*i,perturbed_actions=[[.1,0.,0.,0.]]*i,
                effective_deltas=[[.1,0.,0.,0.]]*i,basis_vector=[1.,0.,0.,0.],strength=.1)))
    witnesses=[dict(label=1,witness_status='observed_landing') for _ in rows]
    points=[dict(root_cell=str(i)) for i in range(4)]
    observations={i:[float(i),2.] for i in range(4)}
    return rows,witnesses,points,observations


def test_action_is_next_tick_and_witness_is_later():
    r,w,p,o=fixture();r[-1]['perturbation']['nominal_actions'][1]=[.2]*4
    r[-1]['perturbation']['perturbed_actions'][1]=[.3,.2,.2,.2]
    # Extend the exact prefix on every snapshot after this action.
    for row in r[1:-1]:
        for key in ('nominal_actions','perturbed_actions'):
            row['perturbation'][key]=copy.deepcopy(r[-1]['perturbation'][key][:row['trajectory_step']])
    pairs,stats=link_action_pairs(r,w,p,o,{'0'},'base','endpoint')
    assert len(pairs)==3 and stats['no_next_saved_action']==1
    assert pairs[0]['observation']==o[0]
    assert pairs[0]['base_action']==[.2]*4
    assert pairs[0]['action_origin']['tick']==1
    assert pairs[0]['witness_origin']['tick']==2
    assert pairs[0]['state_sha256']==r[1]['state_sha256']


def test_unknown_and_old_cells_do_not_receive_credit():
    r,w,p,o=fixture();w[1]['label']=None;w[1]['witness_status']='unknown';w[2]['label']=0
    pairs,_=link_action_pairs(r,w,p,o,{'3'},'base','endpoint')
    assert pairs==[]


def test_prefix_tampering_rejected():
    r,w,p,o=fixture();r[0]['perturbation']['nominal_actions'][0]=[.9]*4
    with pytest.raises(ValueError,match='prefix'):
        link_action_pairs(r,w,p,o,set(),'base','endpoint')


def test_duplicate_tick_rejected():
    r,w,p,o=fixture();r[1]['trajectory_step']=1
    with pytest.raises(ValueError,match='tick|prefix'):
        link_action_pairs(r,w,p,o,set(),'base','endpoint')


def test_split_is_grouped_and_order_independent():
    a=split_groups(['a','b','c','d','a'],seed=12,development_fraction=.25)
    b=split_groups(['d','a','c','b'],seed=12,development_fraction=.25)
    assert a==b and len(a['development'])==1
    assert not set(a['fit']) & set(a['development'])
    with pytest.raises(ValueError):split_groups(['a'],seed=1,development_fraction=.25)


def test_empty_extra_history_has_declared_feature_width():
    from jit_dvgc.residual_exploration import _features
    contract=dict(observation_names=['x','v'],history_steps=0,goal_names=['g'],feature_mean=[0.]*7,feature_scale=[1.]*7)
    assert _features([1.,2.],[],[0.]*4,[.1],contract).shape==(7,)


def test_causal_export_rejects_forged_origin(monkeypatch):
    from jit_dvgc import residual_dataset as rd
    built=dict(partitions={'fit':[{'action_origin':{'tick':2}}]},split={'fit':['a'],'development':['b']},contract={},bank=[],baseline=[],inputs=[])
    monkeypatch.setattr(rd,'build_export',lambda spec:built)
    monkeypatch.setattr(rd,'_verify_files',lambda inputs:None)
    doc=dict(schema=rd.SOURCE_SCHEMA,role='TRAIN',partition='fit',inputs=[],export_spec={},records=[{'action_origin':{'tick':1}}],split=built['split'])
    data=dict(partition='fit',contract={},frozen_base_bank=[],cumulative_baseline={'root_cells':[]})
    with pytest.raises(ValueError,match='origin'):
        rd.validate_causal_export(doc,data)


def test_development_partition_cannot_fit():
    from jit_dvgc.residual_exploration import fit
    with pytest.raises(ValueError,match='partition'):
        fit(dict(schema='residual_causal_dataset_v2',partition='development'),steps=1,max_steps=1)


def test_export_contract_is_complete_and_normalizes_fit_only(monkeypatch):
    from jit_dvgc import residual_dataset as rd
    from jit_dvgc.residual_exploration import validate_contract
    pairs=[dict(trajectory_group=g,observation=[float(i),1.],base_action=[0.]*4,goal=[0.]*7) for i,g in enumerate('abcd')]
    monkeypatch.setattr(rd,'_checked_comparison',lambda root:(pairs,{},set(),{'model_sha256':'a'*64,'physical_cell_schema_sha256':'b'*64},[],set()))
    built=rd.build_export(dict(comparison_root='unused',split_seed=12,development_fraction=.25,delta_limit=[.15]*4,slew_limit=[.3]*4))
    assert validate_contract(built['contract'])==13
    expected=np.mean([r['observation'][0] for r in built['partitions']['fit']])
    assert built['contract']['feature_mean'][0]==pytest.approx(expected)


def test_changed_raw_receipt_rejected_before_reconstruction(tmp_path,monkeypatch):
    from jit_dvgc import residual_dataset as rd
    p=tmp_path/'receipt.json';p.write_text('{}');sha=rd.file_sha(p);p.write_text('{"changed":true}')
    doc=dict(schema=rd.SOURCE_SCHEMA,role='TRAIN',partition='fit',inputs=[dict(path=str(p),sha256=sha)])
    with pytest.raises(ValueError,match='changed'):
        rd.validate_causal_export(doc,dict(partition='fit'))


def test_partition_cannot_be_relabelled():
    from jit_dvgc import residual_dataset as rd
    with pytest.raises(ValueError,match='partition'):
        rd.validate_causal_export(dict(schema=rd.SOURCE_SCHEMA,role='TRAIN',partition='development'),dict(partition='fit'))


def test_original_observation_semantics_must_match():
    from jit_dvgc import residual_dataset as rd
    from pathlib import Path
    files={str(Path(rd.__file__).parent/name):rd.file_sha(Path(rd.__file__).parent/name) for name in ('observation.py','constants.py')}
    rd.verify_observation_sources(files)
    files[next(iter(files))]='0'*64
    with pytest.raises(ValueError,match='observation'):
        rd.verify_observation_sources(files)


def test_declared_unit_scaling_does_not_amplify_unseen_directions(monkeypatch):
    from jit_dvgc import residual_dataset as rd
    pairs=[dict(trajectory_group=g,observation=[float(i)],base_action=[0.]*4,goal=[2.9,2.75,0.,1.,0.,0.,.15]) for i,g in enumerate('abcd')]
    monkeypatch.setattr(rd,'_checked_comparison',lambda root:(pairs,{},set(),{'model_sha256':'a'*64,'physical_cell_schema_sha256':'b'*64},[],set()))
    built=rd.build_export(dict(comparison_root='unused',split_seed=12,development_fraction=.25,delta_limit=[.15]*4,slew_limit=[.3]*4,normalization_mode='actor_fit_std_goal_units_v1'))
    c=built['contract'];assert c['feature_mean'][-11:]==[0.]*11
    assert c['feature_scale'][-11:]==[1.]*11
    assert c['normalization_mode']=='actor_fit_std_goal_units_v1'
