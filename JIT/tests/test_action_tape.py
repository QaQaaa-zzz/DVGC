import copy
import hashlib
import importlib.util
import json

import pytest


def module():
    assert importlib.util.find_spec('jit_dvgc.action_tape') is not None, 'action tape validation is missing'
    from jit_dvgc import action_tape
    return action_tape


def tape():
    rows=[]
    for i in range(2):
        rows.append(dict(tick=i,observation=[1.,2.],state_sha256=str(i)*64,context_sha256=str(i+2)*64,
            next_state_sha256=str(i+1)*64,next_context_sha256=str(i+3)*64,
            base_action=[.9,0,0,0],applied_action=[1.,0,0,0],requested_delta=[.2,0,0,0],
            effective_delta=[.1,0,0,0],previous_delta=([0.]*4 if i==0 else [.2,0,0,0]),controller_history=[]))
    return dict(schema='jit_causal_action_tape_v1',protocol_sha256='a'*64,trajectory_id='t/variant_0',
        policy_actor_sha256='b'*64,policy_payload_sha256='c'*64,goal=[2.9,2.75,1,0,0,0,.2],
        initial_state_sha256='0'*64,initial_context_sha256='2'*64,final_state_sha256='2'*64,
        final_context_sha256='4'*64,record_count=2,environment_interactions=2,records=rows)


def write(path,doc):
    doc=copy.deepcopy(doc)
    doc['tape_sha256']=hashlib.sha256(json.dumps(doc,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    path.write_text(json.dumps(doc))
    return path


def test_load_preserves_saturated_requested_and_effective_residual(tmp_path):
    path=write(tmp_path/'t.json',tape())
    got=module().load_action_tape(path,expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    assert got['records'][1]['previous_delta']==[.2,0,0,0]
    assert got['records'][1]['effective_delta']==[.1,0,0,0]


@pytest.mark.parametrize('mutation', ['tick','chain','context','effective','previous','bounds','count','final'])
def test_rejects_broken_causal_tape_even_if_self_hash_recomputed(tmp_path,mutation):
    doc=tape();row=doc['records'][1]
    if mutation=='tick':row['tick']=3
    if mutation=='chain':row['state_sha256']='f'*64
    if mutation=='context':row['context_sha256']='f'*64
    if mutation=='effective':row['effective_delta'][0]=.2
    if mutation=='previous':row['previous_delta'][0]=0.
    if mutation=='bounds':row['applied_action'][0]=1.2
    if mutation=='count':doc['environment_interactions']=3
    if mutation=='final':doc['final_context_sha256']='f'*64
    with pytest.raises(ValueError):module().load_action_tape(write(tmp_path/'t.json',doc))


def test_rejects_file_and_self_hash_tampering(tmp_path):
    m=module();p=write(tmp_path/'t.json',tape())
    with pytest.raises(ValueError):m.load_action_tape(p,expected_sha256='f'*64)
    d=json.loads(p.read_text());d['records'][0]['observation'][0]=8;p.write_text(json.dumps(d))
    with pytest.raises(ValueError):m.load_action_tape(p)


@pytest.mark.parametrize('options', [dict(record_action_tape=1),dict(record_action_tape=True),dict(residual_explorer='x')])
def test_bad_tape_options_rejected_before_environment_access(tmp_path,options):
    from jit_dvgc.acquisition.causal_jump import collect_jump_start_connected_candidates
    with pytest.raises(ValueError,match='tape|residual'):
        collect_jump_start_connected_candidates([],tmp_path/'out',env=None,policy=None,policy_record={},
            frozen_manifest_sha256='a'*64,nominal_centerline=tmp_path/'none',protocol_seed=1,
            strengths=[.1],action_names=['steer'],signs=[1],**options)


def test_window_controller_zero_residual_keeps_exact_base_and_updates_history():
    from jit_dvgc.acquisition import causal_jump as c
    assert hasattr(c,'_residual_action'), 'explicit residual action integration missing'
    from jit_dvgc.residual_exploration import initialize, reset_state
    contract=dict(observation_names=['a','b'],goal_names=['g']*7,action_names=['a','b','c','d'],
        history_steps=1,model_sha256='a'*64,observation_contract_sha256='b'*64,
        endpoint_protocol_sha256='c'*64,physical_cell_schema_sha256='d'*64,
        feature_mean=[0.]*15,feature_scale=[1.]*15,delta_limit=[.15]*4,slew_limit=[.3]*4,
        action_low=[-1.]*4,action_high=[1.]*4,control_dt=.02)
    contract['goal_names']=[str(i) for i in range(7)]
    artifact=dict(contract=contract,variables=initialize(contract))
    import numpy as np
    base=np.array([.9,-.5,.2,0],np.float32);s=reset_state(contract)
    for active in (False,True,True,False):
        action,s=c._residual_action(artifact,[1.,2.],base,[0.]*7,s,active)
        np.testing.assert_array_equal(action,base)
        np.testing.assert_array_equal(s.history,[[1.,2.]])
        np.testing.assert_array_equal(s.previous_delta,[0.]*4)


def test_real_acquisition_loop_records_every_preaction_and_preserves_legacy_defaults(tmp_path,monkeypatch):
    # Deterministic state-machine fixture exercises the acquisition loop; no physics run.
    from types import SimpleNamespace as NS
    import numpy as np
    from jit_dvgc.acquisition import causal_jump as c
    from jit_dvgc import frontier_label_shard_runner as runner
    m=module()
    class State(NS):
        def replace(self,**kwargs):return State(**{**vars(self),**kwargs})
    def state(t):
        return State(done=t==4,data=NS(qpos=np.array([2.5+t*.1,0,.2]),qvel=np.zeros(3)),
            obs={'state':np.array([t,t+1.],np.float32)},metrics={},info={
                'reset_from_soft_tube':False,'reset_from_jump_start':True,'expert_switching_used':False,
                'active_phase':0,'up_events':NS(apex_seen=False),'down_events':NS(valid_contact_seen=t==4),
                'episode_step':t})
    env=NS(_bundle=NS(xml_sha256='e'*64),tube_pool=NS(artifact=NS(entries=[],manifest={'manifest_sha256':'f'*64})),
        _reset_jump_start_unified=lambda key:state(0))
    monkeypatch.setattr(c.jax,'jit',lambda fn:fn)
    monkeypatch.setattr(c.jax,'block_until_ready',lambda s:s)
    monkeypatch.setattr(runner,'_build_memory_stable_step',lambda env:lambda s,a:state(s.info['episode_step']+1))
    jump=c.physical_state_sha256_from_state(state(0))
    monkeypatch.setattr(c,'load_nominal_jump_centerline',lambda p:dict(effective_centerline_max_x_m=3.1,x_min_m=2.5,
        jump_start_state_sha256=jump,centerline_sha256='d'*64))
    def capture(s,**kwargs):
        return NS(qpos=s.data.qpos,qvel=s.data.qvel,episode_step=s.info['episode_step'],phase_episode_step=s.info['episode_step'])
    monkeypatch.setattr(c,'capture_unified_envelope_snapshot',capture)
    monkeypatch.setattr(c,'snapshot_context_sha256',lambda s:hashlib.sha256(str(s.episode_step).encode()).hexdigest())
    monkeypatch.setattr(c,'save_unified_envelope_snapshot',lambda *a:None)
    kwargs=dict(env=env,policy=lambda obs,key:np.array([.9,0,0,0],np.float32),policy_record=dict(
        iteration=1,policy_role='envelope_expansion_authority',xml_sha256='e'*64,name='pi1',actor_sha256='a'*64,
        payload_sha256='b'*64,formal_config_sha256='c'*64),frozen_manifest_sha256='f'*64,
        nominal_centerline=tmp_path/'centerline',protocol_seed=1,strengths=[.15],action_names=['steer'],signs=[1],
        lookbacks_m=[.15],max_forward_ticks=4,evidence_mode='probe_bank_arrivals_v1',probe_bank_sha256='c'*64,
        logical_role='train',start_contract_sha256='b'*64,sampling_mode='trajectory_slices_v2')
    anchors=[dict(phase='upstream',x_target_m=2.8,proposal_family_index=0,parent_group_id='../unsafe',state_sha256='a'*64)]
    default=c.collect_jump_start_connected_candidates(anchors,tmp_path/'default',**kwargs)
    explicit=c.collect_jump_start_connected_candidates(anchors,tmp_path/'explicit',record_action_tape=False,residual_explorer=None,**kwargs)
    assert (tmp_path/'default/protocol.json').read_bytes()==(tmp_path/'explicit/protocol.json').read_bytes()
    assert default==explicit
    recorded=c.collect_jump_start_connected_candidates(anchors,tmp_path/'recorded',record_action_tape=True,**kwargs)
    receipt=recorded['trajectory_receipts'][0]
    doc=m.load_action_tape(tmp_path/'recorded'/receipt['action_tape']['path'],expected_sha256=receipt['action_tape']['sha256'])
    assert len(doc['records'])==receipt['environment_interactions']==4
    assert [r['observation'][0] for r in doc['records']]==[0,1,2,3]
    assert doc['records'][2]['requested_delta'][0]==pytest.approx(.15)
    assert doc['records'][2]['effective_delta'][0]==pytest.approx(.1)
    assert doc['records'][3]['requested_delta']==[0]*4
    for row in recorded['entries']:
        assert row['state_sha256']==doc['records'][row['trajectory_step']-1]['next_state_sha256']
        assert row['snapshot_context_sha256']==doc['records'][row['trajectory_step']-1]['next_context_sha256']


def test_catalog_tape_validation_checks_candidate_prefix_and_context(tmp_path):
    m=module()
    assert hasattr(m,'validate_catalog_action_tapes'), 'catalog tape binding missing'
    d=tape();p=write(tmp_path/'tape.json',d)
    protocol=dict(record_action_tape=True,protocol_sha256='a'*64,controller_mode='fixed_perturbation_v1',
        strengths=[.2],causal_lookbacks_m=[.15],selected_action_names=['steer'],selected_signs=[1],
        action_tape_anchors=[dict(parent_group_id='t',x_target_m=2.9,proposal_family_index=0)])
    row=dict(trajectory_id='t/variant_0',trajectory_step=1,episode_step=1,protocol_sha256='a'*64,
        policy_actor_sha256='b'*64,policy_payload_sha256='c'*64,state_sha256='1'*64,snapshot_context_sha256='3'*64,
        perturbation=dict(nominal_actions=[[.9,0,0,0]],perturbed_actions=[[1.,0,0,0]],effective_deltas=[[.1,0,0,0]]))
    catalog=dict(record_action_tape=True,controller_mode='fixed_perturbation_v1',protocol_sha256='a'*64,
        policy_actor_sha256='b'*64,policy_payload_sha256='c'*64,environment_interactions=2,
        trajectory_receipts=[dict(trajectory_id='t/variant_0',environment_interactions=2,anchor_x_m=2.9,lookback_m=.15,strength=.2,direction={'basis_vector':[1,0,0,0]},action_tape=dict(path='tape.json',sha256=hashlib.sha256(p.read_bytes()).hexdigest()))],entries=[row])
    m.validate_catalog_action_tapes(tmp_path,catalog,protocol)
    row['perturbation']['nominal_actions'][0][0]=.8
    with pytest.raises(ValueError,match='prefix'):m.validate_catalog_action_tapes(tmp_path,catalog,protocol)
    row['perturbation']['nominal_actions'][0][0]=.9
    row['snapshot_context_sha256']='e'*64
    with pytest.raises(ValueError,match='context'):m.validate_catalog_action_tapes(tmp_path,catalog,protocol)


def test_fixed_tape_rejects_an_undeclared_channel_even_with_valid_arithmetic(tmp_path):
    d=tape()
    for r in d['records']:
        r['requested_delta']=[.2,.1,0,0]
        r['applied_action']=[1.,.1,0,0]
        r['effective_delta']=[.1,.1,0,0]
    d['records'][1]['previous_delta']=[.2,.1,0,0]
    with pytest.raises(ValueError,match='fixed perturbation'):
        module().load_action_tape(write(tmp_path/'t.json',d))


def catalog_fixture(tmp_path):
    d=tape()
    protocol=dict(record_action_tape=True,protocol_sha256='a'*64,controller_mode='fixed_perturbation_v1',
        strengths=[.2],causal_lookbacks_m=[.15],selected_action_names=['steer'],selected_signs=[1],
        action_tape_anchors=[dict(parent_group_id='t',x_target_m=2.9,proposal_family_index=0)])
    receipt=dict(trajectory_id='t/variant_0',environment_interactions=2,anchor_x_m=2.9,lookback_m=.15,strength=.2,
        direction={'basis_vector':[1,0,0,0]},action_tape={'path':'tape.json'})
    catalog=dict(record_action_tape=True,controller_mode='fixed_perturbation_v1',protocol_sha256='a'*64,
        policy_actor_sha256='b'*64,policy_payload_sha256='c'*64,environment_interactions=2,
        trajectory_receipts=[receipt],entries=[])
    def publish():
        p=write(tmp_path/'tape.json',d)
        receipt['action_tape']['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
    publish()
    return d,catalog,protocol,publish


@pytest.mark.parametrize('field',['anchor','lookback','strength','direction'])
def test_zero_candidate_tape_goal_is_bound_to_declared_schedule(tmp_path,field):
    d,c,p,publish=catalog_fixture(tmp_path)
    if field=='anchor':d['goal'][0]+= .05
    elif field=='lookback':d['goal'][1]-= .05
    elif field=='strength':p['strengths']=[.1]
    else:p['selected_action_names']=['hip']
    publish()
    with pytest.raises(ValueError,match='goal|schedule'):module().validate_catalog_action_tapes(tmp_path,c,p)


def residual_catalog_fixture(tmp_path):
    from jit_dvgc import residual_exploration as re
    from flax import serialization
    import base64
    d,c,p,publish=catalog_fixture(tmp_path)
    base=tmp_path/'base';base.write_bytes(b'base')
    contract=dict(observation_names=['a','b'],goal_names=module().GOAL_NAMES,action_names=['steer','rear_wheel_drive','hip','knee'],
        history_steps=0,feature_mean=[0.]*13,feature_scale=[1.]*13,delta_limit=[.2]*4,slew_limit=[.4]*4,
        action_low=[-1.]*4,action_high=[1.]*4,control_dt=.02,model_sha256='a'*64,
        observation_contract_sha256='b'*64,endpoint_protocol_sha256='c'*64,physical_cell_schema_sha256='d'*64,
        goal_contract='exogenous fixed perturbation schedule; no future outcome features')
    dataset=dict(role='TRAIN',contract=contract,frozen_base_bank=[{'path':str(base),'sha256':re.file_sha(base)}])
    payload=dict(schema='frozen_residual_explorer_v1',architecture='conditional_mlp_64_32_v1',dataset=dataset,
        dataset_sha256=re.digest(dataset),parameters=base64.b64encode(serialization.to_bytes(re.initialize(contract))).decode())
    artifact=tmp_path/'explorer.json';artifact.write_text(json.dumps(dict(payload=payload,explorer_sha256=re.digest(payload))))
    ref=dict(path=str(artifact),sha256=re.file_sha(artifact),explorer_sha256=re.digest(payload))
    p['frozen_unified_manifest_sha256']=re.file_sha(base)
    p['residual_gate']='shared fixed spatial window; history advances each tick; zero delta outside; slew >= twice amplitude'
    c['residual_explorer']=p['residual_explorer']=ref
    c['controller_mode']=p['controller_mode']='frozen_residual_spatial_window_v1'
    d.update(explorer_sha256=ref['explorer_sha256'],delta_limit=[.2]*4,slew_limit=[.4]*4)
    for r in d['records']:
        r['explorer_state_before']=dict(history=[],previous_delta=r['previous_delta'])
        r['explorer_state_after']=dict(history=[],previous_delta=r['requested_delta'])
    publish()
    return d,c,p,publish


@pytest.mark.parametrize('mutation',['delta','slew','observation_width'])
def test_tape_controller_fields_match_actual_frozen_artifact(tmp_path,mutation):
    d,c,p,publish=residual_catalog_fixture(tmp_path)
    module().validate_catalog_action_tapes(tmp_path,c,p)
    if mutation=='delta':d['delta_limit']=[.3]*4;d['slew_limit']=[.6]*4
    elif mutation=='slew':d['slew_limit']=[.6]*4
    else:
        for r in d['records']:r['observation'].append(0.)
    publish()
    with pytest.raises(ValueError,match='contract'):module().validate_catalog_action_tapes(tmp_path,c,p)


def test_candidate_observation_must_equal_actual_snapshot(tmp_path):
    from types import SimpleNamespace
    import numpy as np
    m=module()
    assert hasattr(m,'validate_tape_candidate_observation'), 'snapshot observation binding missing'
    d=tape();row=dict(trajectory_step=1)
    m.validate_tape_candidate_observation(d,row,SimpleNamespace(observation=np.asarray([1.,2.],np.float32)))
    d['records'][1]['observation']=[9.,2.]
    with pytest.raises(ValueError,match='observation'):
        m.validate_tape_candidate_observation(d,row,SimpleNamespace(observation=np.asarray([1.,2.],np.float32)))


def test_rehashed_receipt_and_goal_cannot_move_declared_anchor(tmp_path):
    d,c,p,publish=catalog_fixture(tmp_path)
    d['goal'][0]+=.1;d['goal'][1]+=.1
    c['trajectory_receipts'][0]['anchor_x_m']+=.1
    publish()
    with pytest.raises(ValueError,match='schedule'):module().validate_catalog_action_tapes(tmp_path,c,p)


def test_rehashed_controller_history_cannot_add_undeclared_memory(tmp_path):
    d,c,p,publish=residual_catalog_fixture(tmp_path)
    history=[[0.,0.]]
    for r in d['records']:
        r['controller_history']=history
        r['explorer_state_before']['history']=history
        history=[r['observation']]
        r['explorer_state_after']['history']=history
    publish()
    with pytest.raises(ValueError,match='contract'):module().validate_catalog_action_tapes(tmp_path,c,p)


def test_residual_artifact_bytes_must_match_catalog_file_hash(tmp_path):
    d,c,p,publish=residual_catalog_fixture(tmp_path)
    from pathlib import Path
    path=Path(p['residual_explorer']['path']);path.write_text(path.read_text()+' ')
    with pytest.raises(ValueError,match='file hash'):module().validate_catalog_action_tapes(tmp_path,c,p)
