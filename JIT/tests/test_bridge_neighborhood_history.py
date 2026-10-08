import numpy as np
from jit_dvgc.neighborhood import FrozenNeighborhood,FIELDS


def row(actor,label):
    return dict(coordinates=dict.fromkeys(FIELDS,0.),phase='upstream',snapshot_context_sha256='x',
                evaluated_actor_sha256=actor,source_actor_sha256=actor,initial_label=label,
                label=label,data_role='train',learning_attempted=False)


def test_history_visible_without_relabeling_old_actor():
    cfg={'evidence_scope':'train_history_v1'}
    index=FrozenNeighborhood([row('old',1)],'new',cfg)
    out=index.query(np.zeros((1,12)),[0])[0]
    assert index.record_count==1 and out[16]==1
    np.testing.assert_array_equal(out[12:16],[0,0,1,0])


def test_current_conflict_unknown_but_other_policy_success_retained():
    index=FrozenNeighborhood([row('new',1),row('new',0),row('old',1)],'new',{'evidence_scope':'train_history_v1'})
    out=index.query(np.zeros((1,12)),[0])[0]
    np.testing.assert_array_equal(out[12:16],[0,0,1,0])


def test_non_train_evidence_rejected():
    import pytest
    r=row('old',1);r['data_role']='student_dev'
    with pytest.raises(ValueError,match='TRAIN'):
        FrozenNeighborhood([r],'new',{'evidence_scope':'train_history_v1'})


def test_teacher_evidence_is_distinct_from_actor_success():
    r=row('teacher:id',1);r['evidence_kind']='verified_teacher'
    out=FrozenNeighborhood([r],'new',{'evidence_scope':'train_history_v1'}).query(np.zeros((1,12)),[0])[0]
    np.testing.assert_array_equal(out[12:16],[0,0,0,1])


def test_real_neighborhood_mixture_ppo_and_identity(tmp_path):
    import pytest
    import jax.numpy as jp
    from jit_dvgc.rsl_pulse import initialize,infer,update_batch,validate_neighborhood_identity
    from jit_dvgc.generative_bridge import explorer_admission as a
    from jit_dvgc.generative_bridge.neighborhood_history import neighborhood_config,freeze_history
    from jit_dvgc.generative_bridge.contracts import file_sha
    cfg=neighborhood_config();path=tmp_path/'map.json';freeze_history(path,[row('old',1)],'new',cfg,1)
    s=dict(controller_mode='learned_residual',explorer_backend='rsl_rl',pulse_steps=3,
        pulse_start_schedule=[0],delta_limit=[.25]*4,num_envs=8,round_index=1,
        explorer_initialization={'mode':'symmetric','latent_std':.6},
        explorer_admission_v1_2=dict(run_id='test',collection_id='one',round=1,master_seed=1,
                                    episode_ids=list(range(8)),uniform_episode_fraction=.2),
        seed=23,learning_rate=.001,epochs=1,minibatch_size=4,clip=.2,target_kl=.01,
        entropy_coefficient=.001,value_coefficient=.5,max_grad_norm=1.,
        neighborhood=cfg,neighborhood_map=str(path),neighborhood_map_sha256=file_sha(path))
    state=initialize(s,np.zeros(106),np.ones(106))
    context=FrozenNeighborhood([row('old',1)],'new',cfg).query(np.zeros((8,12)),[0]*8)
    obs=np.concatenate((np.random.default_rng(7).normal(size=(3,8,106)),np.tile(context[None],(3,1,1))),axis=-1).astype('float32')
    assert obs.shape[-1]==386
    mu,sd,val=map(np.asarray,infer(state,jp.asarray(obs)));raw=mu+sd*.2
    lp=(-.5*((raw-mu)/sd)**2-np.log(sd)-.5*np.log(2*np.pi)).sum(-1)
    learned=np.tile([True,False]*4,(3,1));raw[~learned]=np.nan;lp[~learned]=np.nan
    tape=dict(observation=obs,raw_action=raw,log_prob=lp,value=np.where(learned,val,0.),
              mask=np.ones((3,8),bool),prefix_mask=np.ones((3,8),bool),
              explorer_learned=learned,on_policy_mask=learned,log_prob_valid=learned)
    receipt=a.collection_receipt(s,state,learned[0]);s['explorer_admission_v1_2'].update(a.behavior_identity(state))
    s.update(_validated_explorer_admission=receipt,_validated_behavior_identity=a.behavior_identity(state))
    updated,metrics,_,_=update_batch(s,state,tape,dict(eligible=[True]*8,rewards=[1,-9,-1,-9,1,-9,-1,-9],component_sums={}))
    assert metrics['optimizer_updates']>0 and metrics['effective_training_samples']==12
    assert updated['neighborhood']['evidence_scope']=='train_history_v1'
    with pytest.raises(ValueError,match='neighborhood'):
        validate_neighborhood_identity({'neighborhood':None},updated)
    # A post-collection map change cannot be used for learning this rollout.
    path.write_text(path.read_text()+' ')
    with pytest.raises(ValueError,match='map'):
        a.validate_admission(s,receipt,tape,a.behavior_identity(state))


def test_teacher_source_conflict_quarantines_current_ability():
    from jit_dvgc.generative_bridge.neighborhood_history import source_recheck_evidence
    original={**row('new',0),'root_id':'x','root_episode_id':'ancestor'}
    teachers={'x':dict(teacher_status='invalid',source_actor_sha256='new',source_control_labels=[0,1],source_recheck_label=None)}
    repeats=source_recheck_evidence(teachers,[original],'receipt')
    out=FrozenNeighborhood([original]+repeats,'new',{'evidence_scope':'train_history_v1'}).query(np.zeros((1,12)),[0])[0]
    np.testing.assert_array_equal(out[12:16],[0,0,0,0])
