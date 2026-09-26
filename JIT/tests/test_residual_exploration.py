"""Synthetic CPU contracts; no simulator or discovery performance evidence."""
import json
import numpy as np
import pytest
import jax
from jit_dvgc import residual_exploration as re


def fixture(tmp_path):
    base = tmp_path / 'base.bin'
    base.write_bytes(b'frozen base fixture')
    contract = dict(observation_names=['roll', 'speed'], history_steps=2,
                    goal_names=['target_x'], action_names=['steer', 'rear', 'hip', 'knee'],
                    feature_mean=[0.] * 11, feature_scale=[1.] * 11,
                    delta_limit=[.2]*4, slew_limit=[.03]*4,
                    action_low=[-1.]*4, action_high=[1.]*4, control_dt=.02,
                    model_sha256='a'*64, observation_contract_sha256='b'*64,
                    endpoint_protocol_sha256='f'*64, physical_cell_schema_sha256='0'*64)
    bank = [{'path': str(base), 'sha256': re.file_sha(base)}]
    row = dict(role='TRAIN', observation=[.2, 1.], history=[[0., 0.], [.1, 1.]],
               base_action=[0.]*4, goal=[1.], previous_delta=[0.]*4,
               target_delta=[.02]*4, root_cell='novel', base_sha256=bank[0]['sha256'],
               state_sha256='c'*64, context_sha256='d'*64,
               acquisition_protocol_sha256='e'*64,
               forward_prefix_complete=True, continuation_status='success',
               continuation_state_sha256='c'*64, continuation_context_sha256='d'*64,
               endpoint='first_valid_landing', physical_failure=False,
               continuation_endpoint_protocol_sha256=contract['endpoint_protocol_sha256'])
    source = tmp_path/'source.json'
    source.write_text(json.dumps(dict(schema='residual_action_witnesses_v1', role='TRAIN',
                                      contract_sha256=re.digest(contract), records=[row])))
    data = dict(schema='residual_warmstart_dataset_v1', role='TRAIN', contract=contract,
                frozen_base_bank=bank, cumulative_baseline={'name':'fixture', 'root_cells':['old']},
                sources=[{'path': str(source), 'sha256':re.file_sha(source)}])
    return data, source


def test_zero_and_limits_reset(tmp_path):
    data, _ = fixture(tmp_path)
    model = re.initialize(data['contract'], seed=0)
    obs=np.array([.2, 1.]); goal=np.array([1.]); base=np.array([.99, -.99, 0., 0.])
    state=re.reset_state(data['contract'])
    action, state2 = re.apply_residual(model, data['contract'], obs, base, goal, state)
    np.testing.assert_array_equal(action, base.astype(np.float32))
    for _ in range(30):
        action, delta = re.bound_delta(np.array([5.,-5.,5.,-5.]), base, state.previous_delta,
                                      data['contract'])
        assert np.max(np.abs(delta)) <= .200001
        assert np.max(np.abs(delta - state.previous_delta)) <= .030001
        assert np.max(np.abs(action)) <= 1.
        state = state.replace(previous_delta=delta)
    reset = re.reset_state(data['contract'])
    np.testing.assert_array_equal(reset.previous_delta, np.zeros(4))
    np.testing.assert_array_equal(reset.history, np.zeros((2, 2)))
    assert not np.array_equal(state2.history, reset.history)


def test_fit_artifact_and_frozen_wrapper(tmp_path):
    data, _=fixture(tmp_path)
    batch=re.load_training_data(data)
    model, report=re.fit(data, steps=40, max_steps=40, seed=3, learning_rate=.01)
    assert report['final_loss'] < report['initial_loss']
    assert report['training_kind']=='supervised_warm_start'
    assert report['environment_interactions']==0
    artifact=tmp_path/'explorer.json'
    identity=re.save_artifact(artifact, model, data, report)
    loaded=re.load_artifact(artifact, expected_fingerprint=identity)
    wrapper=re.FrozenResidualPolicy(loaded, lambda obs,key:(np.zeros(4), {}),
                                   base_sha256=data['frozen_base_bank'][0]['sha256'])
    action,state=wrapper.step({'state':np.array([.2,1.])}, jax.random.PRNGKey(0),
                              np.array([1.]), wrapper.reset())
    assert np.all(np.asarray(action)>0)
    assert batch['weights'].tolist()==[1.]
    with pytest.raises(ValueError,match='budget'):
        re.fit(data, steps=41, max_steps=40)
    (tmp_path/'base.bin').write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):
        re.load_artifact(artifact)


@pytest.mark.parametrize('change', ['role','unknown','context','failure','baseline','source','bounds'])
def test_reject_unsafe_data(tmp_path,change):
    data, source=fixture(tmp_path)
    content=json.loads(source.read_text())
    row=content['records'][0]
    if change=='role': data['role']='TEST'
    if change=='unknown': row['continuation_status']='untested'
    if change=='context': row['continuation_context_sha256']='f'*64
    if change=='failure': row['physical_failure']=True
    if change=='baseline': data['cumulative_baseline']['root_cells']=['novel']
    if change=='bounds': row['target_delta']=[.5]*4
    if change=='source': content['changed']=True
    source.write_text(json.dumps(content))
    if change!='source': data['sources'][0]['sha256']=re.file_sha(source)
    with pytest.raises(ValueError): re.load_training_data(data)


def test_duplicate_cells_share_one_novelty_weight(tmp_path):
    data,source=fixture(tmp_path)
    content=json.loads(source.read_text())
    content['records']*=3
    source.write_text(json.dumps(content))
    data['sources'][0]['sha256']=re.file_sha(source)
    batch=re.load_training_data(data)
    assert np.sum(batch['weights'])==pytest.approx(1.)


def test_jitted_feedback_restores_full_explorer_state(tmp_path):
    data,_=fixture(tmp_path)
    model,report=re.fit(data,steps=15,max_steps=15)
    contract=data['contract']
    state=re.reset_state(contract)
    step=jax.jit(lambda obs,base,goal,state: re.apply_residual(model,contract,obs,base,goal,state))
    obs=np.array([.2,1.]); base=np.zeros(4); goal=np.ones(1)
    _,captured=step(obs,base,goal,state)
    first,next_state=step(obs,base,goal,captured)
    replay,replay_state=step(obs,base,goal,captured)
    np.testing.assert_array_equal(first,replay)
    np.testing.assert_array_equal(next_state.history,replay_state.history)
    np.testing.assert_array_equal(next_state.previous_delta,replay_state.previous_delta)
    assert np.max(np.abs(first)) <= .060001


def test_changed_artifact_and_unknown_base_rejected(tmp_path):
    data,_=fixture(tmp_path)
    model,report=re.fit(data,steps=1,max_steps=1)
    path=tmp_path/'explorer.json'
    identity=re.save_artifact(path,model,data,report)
    loaded=re.load_artifact(path)
    with pytest.raises(ValueError,match='unknown'):
        re.FrozenResidualPolicy(loaded,lambda obs,key:np.zeros(4),base_sha256='f'*64)
    with pytest.raises(FileExistsError): re.save_artifact(path,model,data,report)
    doc=json.loads(path.read_text())
    doc['payload']['dataset']['contract']['delta_limit'][0]=10.
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match='fingerprint'): re.load_artifact(path)


def test_cli_fits_explicit_fixture(tmp_path):
    import importlib.util
    from pathlib import Path
    spec=importlib.util.spec_from_file_location('fit_residual_cli',Path(__file__).parents[1]/'cli/fit_residual_explorer.py')
    cli=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    data,_=fixture(tmp_path)
    source=tmp_path/'dataset.json'; source.write_text(json.dumps(data))
    output=tmp_path/'trained.json'
    assert cli.main(['--dataset',str(source),'--output',str(output),'--steps','2','--max-steps','2'])==0
    artifact=re.load_artifact(output)
    assert artifact['explorer_sha256']


def test_wrong_endpoint_protocol_rejected(tmp_path):
    data,source=fixture(tmp_path)
    doc=json.loads(source.read_text())
    doc['records'][0]['continuation_endpoint_protocol_sha256']='1'*64
    source.write_text(json.dumps(doc)); data['sources'][0]['sha256']=re.file_sha(source)
    with pytest.raises(ValueError,match='witness'): re.load_training_data(data)


def test_fit_learns_valid_examples_beyond_initial_slew_interval(tmp_path):
    """Historical nonzero residual state must not create a zero-gradient dead zone."""
    data,source=fixture(tmp_path)
    doc=json.loads(source.read_text())
    doc['records'][0]['previous_delta']=[.15]*4
    doc['records'][0]['target_delta']=[.16]*4
    source.write_text(json.dumps(doc)); data['sources'][0]['sha256']=re.file_sha(source)
    model,report=re.fit(data,steps=40,max_steps=40,seed=3,learning_rate=.01)
    assert report['final_loss'] < .5*report['initial_loss']
    state=re.reset_state(data['contract']).replace(previous_delta=np.full(4,.15))
    action,next_state=re.apply_residual(model,data['contract'],np.array([.2,1.]),np.zeros(4),np.ones(1),state)
    assert np.max(np.abs(np.asarray(next_state.previous_delta)-.15))<=.030001
    assert np.max(np.abs(np.asarray(action)))<=.200001


def test_explorer_rejects_broadcast_previous_delta_state(tmp_path):
    data,_=fixture(tmp_path)
    model=re.initialize(data['contract'])
    state=re.reset_state(data['contract']).replace(previous_delta=np.zeros(1))
    with pytest.raises(ValueError,match='previous residual'):
        re.apply_residual(model,data['contract'],np.array([.2,1.]),np.zeros(4),np.ones(1),state)
