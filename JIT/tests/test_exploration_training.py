import numpy as np
from jit_dvgc.exploration_training import episode_advantages

def test_terminal_credit_propagates_but_padding_does_not():
    reward=np.array([[0.],[2.],[900.]])
    value=np.zeros_like(reward);mask=np.array([[1.],[1.],[0.]])
    adv,ret=episode_advantages(reward,value,mask,gamma=.5,lam=1.)
    np.testing.assert_allclose(ret[:,0],[1.,2.,0.])
    np.testing.assert_allclose(adv,ret)

def test_complete_episode_does_not_bootstrap_value_beyond_terminal():
    adv,ret=episode_advantages(np.array([[3.],[0.]]),np.array([[1.],[999.]]),np.array([[1.],[0.]]),gamma=.99,lam=.95)
    assert adv[0,0]==2 and ret[0,0]==3


def _resume_fixture(tmp_path):
    import json,hashlib
    root=tmp_path/'run';checkpoint=root/'checkpoints'/'batch_0032'
    checkpoint.mkdir(parents=True)
    state=checkpoint/'state.msgpack';state.write_bytes(b'immutable payload')
    spec=dict(reward_mode='trajectory_quality_v1',controller_mode='learned_residual',batches=32,learning_rate=3e-5)
    identity=dict(schema='jit_frozen_policy_residual_ppo_checkpoint_v1',state_sha256=hashlib.sha256(state.read_bytes()).hexdigest(),spec=spec,ledger={'cells':['already visited']})
    (checkpoint/'identity.json').write_text(json.dumps(identity))
    (root/'training_metrics.json').write_text(json.dumps([dict(batch=32,cumulative_optimizer_updates=64)]))
    return state,spec,identity


def test_legacy_resume_preserves_ledger_and_shuffle_update_counters(tmp_path):
    from jit_dvgc.exploration_training import resume_metadata
    state,spec,_=_resume_fixture(tmp_path)
    identity,batches,updates,locks=resume_metadata(state,{**spec,'batches':1024,'resume_checkpoint':str(state),'process_plot_interval':32})
    assert (batches,updates)==(32,64)
    assert identity['ledger']['cells']==['already visited']
    assert len(locks)==3


def test_resume_rejects_changed_reward_or_optimizer_and_corrupt_payload(tmp_path):
    import pytest
    from jit_dvgc.exploration_training import resume_metadata
    state,spec,_=_resume_fixture(tmp_path)
    for change in ({'learning_rate':1e-3},{'reward_weights':{'novelty':2}}):
        with pytest.raises(ValueError,match='contract changed'):resume_metadata(state,{**spec,**change})
    state.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='hash mismatch'):resume_metadata(state,spec)


def test_new_resume_uses_global_counters_without_legacy_log(tmp_path):
    import json
    from jit_dvgc.exploration_training import resume_metadata
    state,spec,identity=_resume_fixture(tmp_path)
    identity.update(training_batches=1056,optimizer_updates=2112)
    (state.parent/'identity.json').write_text(json.dumps(identity))
    (state.parent.parent.parent/'training_metrics.json').unlink()
    _,batches,updates,_=resume_metadata(state,spec)
    assert (batches,updates)==(1056,2112)
