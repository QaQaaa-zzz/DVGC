"""Admission, remaining-window targets, and ancestor sampling use real files."""
from pathlib import Path
import json
import numpy as np
import pytest

from jit_dvgc.retention_repair import read,write,sha


def fixture(tmp_path):
    parent=tmp_path/'coverage';diag=tmp_path/'diagnostic'
    parent.mkdir();diag.mkdir();(parent/'B').mkdir();(diag/'report').mkdir()
    policy=dict(actor_sha256='a'*64,normalizer_sha256='b'*64,critic_sha256='c'*64,xml_sha256='d'*64)
    student='e'*64;ids=[f'TRAIN-ancestor-{i}' for i in range(8)]
    roots=[dict(root=dict(root_id=r,root_episode_id=r,role='TRAIN',data_role='train'),lane=2) for r in ids]
    write(parent/'roots.json',roots);write(diag/'train_roots.json',roots)
    obs=np.arange(8*18*76,dtype=np.float32).reshape(8*18,76)
    actions=np.full((len(obs),4),.2,np.float32)
    origins=(['bridge_prefix']*16+['source_tail']*2)*8
    weights=np.tile([1/256]*16+[1/32]*2,8)
    np.savez_compressed(parent/'demo.npz',observations=obs,actions=actions,weights=weights)
    dm=dict(schema='jit_bridge_demo_v1_2',path=str(parent/'demo.npz'),sha256=sha(parent/'demo.npz'),count=len(obs),
            sample_roots=np.repeat(ids,18).tolist(),sample_origins=origins,role='TRAIN')
    write(parent/'demo_manifest.json',dm)
    write(parent/'plan.json',dict(train_roots=str(parent/'roots.json'),demo_manifest=str(parent/'demo_manifest.json'),models={'pi0':{'policy':policy}}))
    (parent/'B/update_1000.pkl').write_bytes(b'frozen actor fixture')
    sr=dict(actor_sha256=student,normalizer_sha256=policy['normalizer_sha256'],critic_sha256=policy['critic_sha256'])
    write(parent/'B/learner_update_1000.json',sr)
    write(parent/'B_completed.json',dict(phase='completed',selected_update=1000))
    write(parent/'status.json',dict(phase='completed'))
    labels=[[0,0,0],[1,1,1],[0,1,0]]+[[1,1,1]]*5
    trials=[]
    for repeat in range(3):
        trials.append([dict(root_id=r,role='TRAIN',combinations={'independent':dict(label=labels[i][repeat],
            attempt=dict(label=labels[i][repeat],actor_sha256=student,normalizer_sha256=policy['normalizer_sha256']))}) for i,r in enumerate(ids)])
    write(parent/'B_checkpoint_results.json',{'1000':dict(train=trials[0],actor_sha256=student)})
    write(parent/'selected_diagnostics.json',dict(selected_update=1000,TRAIN_confirmation_update=1000,TRAIN_confirmations=trials[1:]))
    plan=dict(schema='jit_retention_self_visit_v1',parent=str(parent),train_roots=str(diag/'train_roots.json'),
              model={'policy':policy},student_actor_sha256=student,student=dict(path=str(parent/'B/update_1000.pkl'),
              sha256=sha(parent/'B/update_1000.pkl'),normalizer_sha256=policy['normalizer_sha256'],source_actor_sha256=policy['actor_sha256']),
              query_steps=[4,8,12],training_updates=0,data_role='TRAIN')
    write(diag/'plan.json',plan);write(diag/'status.json',dict(phase='completed',completed_roots=8,evaluation_status='complete'))
    queries=[]
    for i,rid in enumerate(ids):
        folder=diag/f'root_{i:02d}';folder.mkdir()
        rows=[]
        for step in (4,8,12):
            q=dict(root_id=rid,role='TRAIN',step=step,remaining_horizon=16-step,student_full_label=labels[i][0],
                   classification='source_recoverable',source_labels=[1]*3,winner_labels=[1]*3)
            if (i==0 and step in (4,12)) or (i in (1,2) and step==4):
                _teacher_query(q,folder,policy,student)
            rows.append(q)
        write(folder/'queries.json',rows);queries.extend(rows)
    sync(diag,queries)
    return parent,diag,queries,ids


def sync(diag,queries):
    for i in range(8):write(diag/f'root_{i:02d}/queries.json',queries[3*i:3*i+3])
    write(diag/'report/query_matrix.json',queries)
    write(diag/'report/summary.json',dict(query_count=len(queries),coverage={'complete':len(queries)==24},evaluation_status='complete'))


def _teacher_query(q,folder,policy,student):
    step=q['step'];k=16-step;n=k+2;cid=1;context=str(step%10)*64
    capture_path=folder/f'capture_{step}.npz'
    co=np.arange(12*17*76,dtype=np.float32).reshape(12,17,76)
    np.savez_compressed(capture_path,actor_observation_after=co,mask=np.ones((12,17),bool))
    start=co[step-1,2];capture=dict(capture_step=step,capture_lane=2,capture_status='captured',role='TRAIN',data_role='train',
        parent_root_id=q['root_id'],root_episode_id=q['root_id'],actor_sha256=student,snapshot_context_sha256=context,
        snapshot_time=step*.02,trace=str(capture_path),trace_sha256=sha(capture_path),trace_lane=2)
    actions=np.full((17,16,4),.25,np.float32);actions[0]=0
    proposal=folder/f'j{step:02d}_proposals.npz'
    np.savez_compressed(proposal,actions=actions,source_only=np.arange(17)==0,root_contexts=np.array([context]*17))
    lock=dict(selected_candidate_id=cid,proposals_sha256=sha(proposal),layout=17,fixed_before_repeats=True,
              remaining_horizon=k,snapshot_context_sha256=context)
    write(folder/f'j{step:02d}_winner_lock.json',lock)
    winners=[];sources=[];all_labels=[]
    for repeat in range(3):
        suffix='search' if repeat==0 else f'repeat{repeat}'
        d=folder/f'j{step:02d}_{suffix}';d.mkdir()
        pre=np.repeat((start[None]+np.arange(n+2,dtype=np.float32)[:,None])[:,None,:],17,axis=1)
        trace=dict(actor_observation_before=pre,actor_observation_after=pre+1,
          normalized_action_executed=np.full((n+2,17,4),.25,np.float32),mask=np.zeros((n+2,17),bool),
          done=np.zeros((n+2,17),bool),success=np.zeros((n+2,17),bool),physical_failure=np.zeros((n+2,17),bool),
          timeout=np.zeros((n+2,17),bool),phase_before=np.zeros((n+2,17),np.int32),
          action_origin_code=np.tile(np.array([1]*k+[2]*4)[:,None],(1,17)),
          time_before=np.repeat((step*.02+np.arange(n+2)*.02)[:,None],17,axis=1))
        trace['mask'][:n]=True;trace['success'][n-1]=True;trace['success'][:,0]=False
        trace['done'][n-1]=True;trace['physical_failure'][n-1,0]=True;trace['action_origin_code'][:,0]=0
        trace['normalized_action_executed'][k:]=.5
        f=d/'trace.npz';np.savez_compressed(f,**trace)
        a=dict(trace=str(f),trace_sha256=sha(f),trace_lane=cid,label=1,steps=n,outcome='stable_forward_recovery',
            actor_sha256=policy['actor_sha256'],normalizer_sha256=policy['normalizer_sha256'],model_sha256=policy['xml_sha256'],
            snapshot_context_sha256=context,recording_schema='jit_bridge_trace_v1',controller_kind='composite_teacher',
            source_only=False,reset_at_handoff=False,handoff_tick=k,action_origin='bridge_prefix_then_source_tail')
        source={**a,'trace_lane':0,'label':0,'source_only':True,'outcome':'airborne_physical_failure'}
        winners.append(a);sources.append(source);labels=[0]+[1]*16;all_labels.append(labels)
        write(d/'results.json',[dict(candidate_id=l,label=labels[l],attempts=[source if l==0 else {**a,'trace_lane':l}]) for l in range(17)])
    q.update(classification='G_incremental_verified',source_labels=[0]*3,winner_labels=[1]*3,
             winner_lock=lock,winner_attempts=winners,source_attempts=sources,all_candidate_labels=all_labels,
             proposals=str(proposal),capture=capture)


def prepare(parent,diag,out):
    from jit_dvgc.retention_self_visit_learning import prepare_demo
    return prepare_demo(parent,diag,out)


def test_prepares_only_failed_ancestor_fixed_second_repeat_and_balances_real_segments(tmp_path):
    parent,diag,queries,ids=fixture(tmp_path)
    manifest=read(prepare(parent,diag,tmp_path/'prepared'))
    from jit_dvgc.generative_bridge.student import load_optional_demo
    o,a,w=load_optional_demo(manifest)
    with np.load(parent/'demo.npz') as old:
        np.testing.assert_array_equal(o[:144],old['observations'])
        np.testing.assert_array_equal(a[:144],old['actions'])
    assert len(o)==164 and manifest['old_count']==144 and manifest['new_count']==20
    assert manifest['eligible_query_count']==2
    assert {e['ancestor'] for e in manifest['new_entries']}=={ids[0]}
    assert all('repeat2' in e['trace'] for e in manifest['new_entries'])
    roots=np.array(manifest['sample_roots']);segments=np.array(manifest['sample_origins'])
    for rid in ids:assert w[roots==rid].sum()==pytest.approx(1/8)
    p=read(tmp_path/'prepared/frame_provenance.json')
    trajectory=np.array([r['trajectory_id'] for r in p])
    for tid in np.unique(trajectory[roots==ids[0]]):
        assert w[trajectory==tid].sum()==pytest.approx(1/24)
        assert w[(trajectory==tid)&(segments=='bridge_prefix')].sum()==pytest.approx(1/48)
    assert p[144]['local_transition']==0 and p[144]['capture_step']==4
    assert p[156]['segment']=='source_tail' and p[158]['capture_step']==12
    assert read(tmp_path/'prepared/source_lock.json')['training_updates']==0


@pytest.mark.parametrize('change',['running','partial','duplicate','dev','replay_fail','normalizer','handoff','winner_lane','old_weights','no_new'])
def test_preparation_fails_closed_before_output(tmp_path,change):
    parent,diag,queries,ids=fixture(tmp_path)
    q=queries[0]
    if change=='running':write(diag/'status.json',dict(phase='running'))
    elif change=='partial':queries.pop()
    elif change=='duplicate':queries[1]=queries[0]
    elif change=='dev':q['role']='SOLVER_DEV'
    elif change=='replay_fail':q['winner_labels']=[1,1,0]
    elif change=='normalizer':q['winner_attempts'][-1]['normalizer_sha256']='wrong'
    elif change=='handoff':q['winner_attempts'][-1]['handoff_tick']=16
    elif change=='winner_lane':q['winner_attempts'][-1]['trace_lane']=0
    elif change=='no_new':
        for x in queries:x['classification']='source_recoverable'
    elif change=='old_weights':
        with np.load(parent/'demo.npz') as z:arrays={k:z[k] for k in z.files}
        arrays['weights']=np.ones(144)/144;np.savez_compressed(parent/'demo.npz',**arrays)
        m=read(parent/'demo_manifest.json');m['sha256']=sha(parent/'demo.npz');write(parent/'demo_manifest.json',m)
    sync(diag,queries)
    with pytest.raises(ValueError):prepare(parent,diag,tmp_path/'prepared')
    assert not (tmp_path/'prepared').exists()


@pytest.mark.parametrize('change',['mask_gap','wrong_origin','nonfinite','failure','terminal_before_end','discontinuity','prefix_action','time'])
def test_true_trace_contract_is_checked_after_hash_validation(tmp_path,change):
    parent,diag,queries,_=fixture(tmp_path);q=queries[0];a=q['winner_attempts'][-1];path=Path(a['trace']);lane=a['trace_lane']
    with np.load(path) as z:arrays={k:z[k] for k in z.files}
    if change=='mask_gap':arrays['mask'][2,lane]=False
    elif change=='wrong_origin':arrays['action_origin_code'][12,lane]=1
    elif change=='nonfinite':arrays['actor_observation_before'][2,lane,0]=np.nan
    elif change=='failure':arrays['physical_failure'][3,lane]=True
    elif change=='terminal_before_end':arrays['done'][3,lane]=True
    elif change=='discontinuity':arrays['actor_observation_after'][2,lane,0]+=1
    elif change=='prefix_action':arrays['normalized_action_executed'][2,lane,0]=-.9
    elif change=='time':arrays['time_before'][0,lane]=10
    np.savez_compressed(path,**arrays);a['trace_sha256']=sha(path);q['source_attempts'][-1]['trace_sha256']=sha(path)
    rows=read(path.parent/'results.json')
    for r in rows:r['attempts'][0]['trace_sha256']=sha(path)
    rows[lane]['attempts'][0]=a;rows[0]['attempts'][0]=q['source_attempts'][-1]
    write(path.parent/'results.json',rows);sync(diag,queries)
    with pytest.raises(ValueError):prepare(parent,diag,tmp_path/'prepared')


def test_changed_raw_trace_is_rejected_even_with_successful_labels(tmp_path):
    parent,diag,queries,_=fixture(tmp_path)
    Path(queries[0]['winner_attempts'][-1]['trace']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='hash'):prepare(parent,diag,tmp_path/'prepared')
