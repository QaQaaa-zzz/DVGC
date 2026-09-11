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


def tape_fixture():
    rows,witnesses,points,_=fixture()
    records=[]
    for t in range(4):
        records.append(dict(tick=t,observation=[float(t),2.],state_sha256=str(t)*64,
            context_sha256=str(t+4)*64,next_state_sha256=str(t+1)*64,
            next_context_sha256=str(t+5)*64,base_action=[0.]*4,
            applied_action=[.1,0.,0.,0.],requested_delta=[.1,0.,0.,0.],
            effective_delta=[.1,0.,0.,0.],previous_delta=[0.]*4 if t==0 else [.1,0.,0.,0.],controller_history=[]))
    tape=dict(protocol_sha256='a'*64,trajectory_id='t',goal=[2.9,2.75,1.,0.,0.,0.,.1],records=records,tape_sha256='f'*64)
    return rows,witnesses,points,tape


def test_complete_tape_credits_initial_action_and_all_raw_actions():
    from jit_dvgc import residual_dataset as rd
    r,w,p,t=tape_fixture()
    pairs,stats=rd.link_tape_action_pairs(r,w,p,{'t':t},{'0'},'base','endpoint',condition_group='condition')
    assert len(pairs)==4 and stats['raw_actions']==4
    assert pairs[0]['action_origin']['tick']==0
    assert 'candidate_index' not in pairs[0]['action_origin']
    assert pairs[0]['witness_origin']['tick']==2
    assert pairs[0]['previous_delta']==[0.]*4
    assert pairs[-1]['witness_origin']['tick']==4
    assert all(x['condition_group']=='condition' for x in pairs)


@pytest.mark.parametrize('change',['state','prefix','goal','unknown'])
def test_tape_join_rejects_drift_and_never_credits_unknown(change):
    from jit_dvgc import residual_dataset as rd
    r,w,p,t=tape_fixture()
    if change=='state':r[0]['state_sha256']='e'*64
    if change=='prefix':r[0]['perturbation']['nominal_actions'][0]=[.5]*4
    if change=='goal':t['goal'][0]=3.0
    if change=='unknown':
        for item in w:item.update(label=None,witness_status='unknown')
        pairs,stats=rd.link_tape_action_pairs(r,w,p,{'t':t},set(),'base','endpoint',condition_group='condition')
        assert pairs==[] and stats['no_later_novel_witness']==4
    else:
        with pytest.raises(ValueError):
            rd.link_tape_action_pairs(r,w,p,{'t':t},set(),'base','endpoint',condition_group='condition')


def comparison_fixture(root):
    from jit_dvgc import residual_dataset as rd
    rows=[]
    for axis in range(4):
        for sign in (-1,1):
            basis=[0.]*4;basis[axis]=float(sign)
            rows.append(dict(trajectory_group=f't{axis}{sign}',condition_group=str(root),
                observation=[1. if str(root).endswith('fit') else 100.],base_action=[0.]*4,
                goal=[2.9 if str(root).endswith('fit') else 3.0,2.75,*basis,.1],
                target_delta=[x*.1 for x in basis],previous_delta=[0.]*4))
    return rows,{},set(),{'model_sha256':'a'*64,'physical_cell_schema_sha256':'b'*64},[],set()


def test_explicit_conditions_keep_all_signed_channels_and_fit_only_scaling(tmp_path,monkeypatch):
    from jit_dvgc import residual_dataset as rd
    monkeypatch.setattr(rd,'_checked_comparison',lambda root,**kw:comparison_fixture(root))
    spec=dict(comparison_partitions={'fit':[str(tmp_path/'fit')],'development':[str(tmp_path/'dev')]},
        require_action_tapes=True,delta_limit=[.15]*4,slew_limit=[.3]*4,normalization_mode='actor_fit_std_goal_units_v1')
    built=rd.build_export(spec)
    assert len(built['partitions']['fit'])==8
    assert not set(built['split']['fit']) & set(built['split']['development'])
    assert built['contract']['feature_mean'][0]==1.
    assert built['channel_diagnostics']['development']['knee']['negative']==1
    spec['comparison_partitions']['development']=spec['comparison_partitions']['fit']
    with pytest.raises(ValueError,match='disjoint'):
        rd.build_export(spec)


def test_partition_rejects_missing_signed_channel(tmp_path,monkeypatch):
    from jit_dvgc import residual_dataset as rd
    def checked(root,**kw):
        values=list(comparison_fixture(root));values[0]=values[0][:-1];return values
    monkeypatch.setattr(rd,'_checked_comparison',checked)
    with pytest.raises(ValueError,match='signed.*channel|channel.*coverage'):
        rd.build_export(dict(comparison_partitions={'fit':[str(tmp_path/'fit')],'development':[str(tmp_path/'dev')]},
            require_action_tapes=True,delta_limit=[.15]*4,slew_limit=[.3]*4))


def test_tape_export_preserves_file_binding_and_counts_excluded_channels():
    from jit_dvgc import residual_dataset as rd
    r,w,p,t=tape_fixture();t['source_binding']={'path':'/tape.json','sha256':'e'*64}
    pairs,stats=rd.link_tape_action_pairs(r,w,p,{'t':t},{'0','1','2','3'},'base','endpoint',condition_group='condition')
    assert not pairs
    assert stats['channels']['steer']['positive']==dict(raw_actions=4,exported_actions=0,no_later_novel_witness=4)
    pairs,_=rd.link_tape_action_pairs(r,w,p,{'t':t},set(),'base','endpoint',condition_group='condition')
    assert pairs[0]['action_origin']['action_tape']==t['source_binding']


def test_same_physical_condition_cannot_cross_comparison_roots(tmp_path,monkeypatch):
    from jit_dvgc import residual_dataset as rd
    def checked(root,**kw):
        values=comparison_fixture(root)
        for row in values[0]:row['goal'][0]=2.9
        return values
    monkeypatch.setattr(rd,'_checked_comparison',checked)
    with pytest.raises(ValueError,match='conditions.*disjoint'):
        rd.build_export(dict(comparison_partitions={'fit':[str(tmp_path/'fit')],'development':[str(tmp_path/'dev')]},
            require_action_tapes=True,delta_limit=[.15]*4,slew_limit=[.3]*4))
