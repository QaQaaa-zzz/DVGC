import numpy as np
import pytest
from jit_dvgc.exploration_reward import trajectory_reward_batch


def test_all_arrivals_credited_independent_of_candidate_cap_and_failure_paid_once():
    data=dict(mask=np.array([[1],[1],[1],[0]],bool),finite=np.ones((4,1),bool),
              physical_failure=np.array([[0],[0],[1],[1]],bool),terminal=np.array([[0],[0],[1],[1]],bool),
              normalized_delta=np.ones((4,1,4)))
    reward,ledger,evidence,parts=trajectory_reward_batch({'policy_sha256':'a'*64,'cells':[]},data,
        [{'cells':['new1','new2','failed']}],expected_policy_sha256='a'*64,
        weights={'novelty':1.,'physical_failure':50.,'residual_energy':.01})
    assert ledger['cells']==['new1','new2']
    assert parts['physical_failure'].sum()==-50
    assert parts['novelty'].sum()==2
    assert parts['residual_energy'].sum()==pytest.approx(-.03)
    np.testing.assert_allclose(reward,sum(parts.values()))
    assert reward[-1,0]==0
    assert evidence['envelope_admission'] is False


def test_repeated_cell_not_repeated_reward_and_policy_mismatch_rejected():
    data=dict(mask=np.ones((3,2),bool),finite=np.ones((3,2),bool),physical_failure=np.zeros((3,2),bool),terminal=np.zeros((3,2),bool),normalized_delta=np.zeros((3,2,4)))
    ledger={'policy_sha256':'a'*64,'cells':['old']}
    episodes=[{'cells':['new','new','old']},{'cells':['new','new','old']}]
    reward,_,_,_=trajectory_reward_batch(ledger,data,episodes,expected_policy_sha256='a'*64,weights={'novelty':1,'physical_failure':0,'residual_energy':0})
    assert reward.sum()==1
    np.testing.assert_allclose(reward[0],[.5,.5])
    assert reward[1:].sum()==0
    with pytest.raises(ValueError):trajectory_reward_batch(ledger,data,episodes,expected_policy_sha256='b'*64,weights={'novelty':1,'physical_failure':0,'residual_energy':0})


def test_process_files_preserve_named_actor_critic_losses_and_reward_parts(tmp_path):
    import json,csv
    from jit_dvgc.exploration_process import append_update,export_process
    update={'update':1,'actor_loss':-.2,'critic_loss':3.,'total_loss':1.3}
    append_update(tmp_path,update)
    row=dict(batch=1,new_cells=7,total_reward=-43.,reward_components={'novelty':7.,'physical_failure':-50.,'residual_energy':0.},loss_components=update)
    export_process(tmp_path,[row])
    assert json.loads((tmp_path/'optimizer_updates.jsonl').read_text())==update
    with (tmp_path/'training_process.csv').open() as f:record=next(csv.DictReader(f))
    assert float(record['critic_loss'])==3
    assert float(record['reward_physical_failure'])==-50
