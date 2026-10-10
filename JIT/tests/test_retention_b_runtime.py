import numpy as np
import pytest


def test_four_groups_fail_closed_and_draw_equal_probability():
    from jit_dvgc.retention_reset import group_indices,validate_inventory
    bad={'role':'DEV'}
    with pytest.raises(ValueError,match='TRAIN'):validate_inventory(bad)
    import jax
    groups=np.asarray(group_indices(jax.random.PRNGKey(101),4096))
    assert set(groups)=={0,1,2,3}
    assert np.all(np.abs(np.bincount(groups)/4096-.25)<.03)


def test_budget_counts_baseline_and_B_warning_repeat():
    from jit_dvgc.retention_b import budget
    b=budget(7)
    assert b['snapshot_nodes']==5*(7+8)*17*400
    assert b['B_warning_repeat']==4*2*64*400
    assert b['D2_cumulative_maximum']==b['total']+12900
    assert b['total']<1987100 and b['reset_GPU']==0
    assert b['B_updates']==2000 and b['PPO_updates']==0


def test_keep_failure_warning_and_selection_are_per_cell():
    from jit_dvgc.retention_b import warning_cells,select_B
    assert warning_cells({'B':[1]*16,'D':[1]*16},{'B':[1]*15+[0],'D':[1]*16})==['B']
    assert select_B({0:0,100:2,500:2},{0:True,100:True,500:True})==100
    assert select_B({0:0,100:2,500:1},{0:True,100:False,500:True})==500


def test_warmup_stops_only_on_declared_checkpoint_guard(tmp_path):
    import jax.numpy as jp
    from types import SimpleNamespace as NS
    from brax.training.acme import running_statistics as rs
    from jit_dvgc.generative_bridge.warmup import warmup_actor
    from jit_dvgc.generative_bridge.data import export_student_demonstrations
    from .test_generative_bridge_student_v12 import teacher
    norm=rs.init_state(jp.zeros(76));demo=export_student_demonstrations([teacher()],tmp_path/'demo')
    network=NS(policy_network=NS(apply=lambda norm,p,o:jp.ones((len(o['state']),4))*p),parametric_action_distribution=NS(mode=lambda x:x))
    warmup_actor(network,(norm,jp.array(.5),jp.array(7.)),(norm,jp.array(.5)),demo,(np.ones((2,76)),np.ones(2)),tmp_path/'B',updates=2000,batch_size=2,full_learner=True,checkpoint_callback=lambda step,p:step==100)
    import json
    status=json.loads((tmp_path/'B/warmup_status.json').read_text())
    assert status['status']=='stopped_retention_guard' and status['completed_updates']==100
    assert (tmp_path/'B/learner_update_0100.pkl').exists() and not (tmp_path/'B/update_0500.pkl').exists()


def test_student_prefix_preserves_batch_source_and_handover():
    import jax,jax.numpy as jp
    from jit_dvgc.generative_bridge.rollout import closed_loop_action
    obs={'state':jp.ones((2,76))};keys=jax.random.split(jax.random.PRNGKey(1),2)
    tail=lambda o,k:(jp.zeros(4),{})
    prefix=lambda o,k:(jp.ones(4),{})
    first=closed_loop_action(jp.array(15),obs,keys,tail,prefix,source_only=jp.array([True,False]))
    after=closed_loop_action(jp.array(16),obs,keys,tail,prefix,source_only=jp.array([True,False]))
    np.testing.assert_array_equal(first,[[0]*4,[1]*4]);np.testing.assert_array_equal(after,np.zeros((2,4)))


def test_fixed_train_probe_reads_manifest_provenance(tmp_path):
    from types import SimpleNamespace as NS
    import jax.numpy as jp
    from .test_generative_bridge_student_v12 import teacher
    from jit_dvgc.generative_bridge.data import export_student_demonstrations
    from jit_dvgc.retention_b import fixed_train_probe
    manifest=export_student_demonstrations([teacher()],tmp_path/'demo')
    network=NS(policy_network=NS(apply=lambda norm,p,o:jp.ones((len(o['state']),4))*p),parametric_action_distribution=NS(mode=lambda x:x))
    result=fixed_train_probe(network,(None,jp.array(.5),None),(None,jp.array(.5)),manifest,(np.ones((4,76)),np.ones(4)/4),{'episodes':[{'group':'nominal','steps':2},{'group':'random','steps':2}]})
    assert len(result['demo'])==1 and set(result['keep'])=={'nominal','random'}
    assert result['keep']['nominal']['mse_by_action']==[0]*4


def test_repeat_classification_never_turns_mixed_labels_into_absorption():
    from jit_dvgc.retention_b_report import classify_repeats
    assert classify_repeats([1,1,1])=='stable_success'
    assert classify_repeats([0,0,0])=='stable_failure'
    assert classify_repeats([0,1,0])=='ambiguous'
    assert classify_repeats([1,None,1])=='unknown'
