import json
from pathlib import Path

import pytest

from jit_dvgc.generative_bridge.source_data import prepare_collection_plan,aggregate_collection


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def test_plan_counts_namespaces_and_charged_extra_suffix(tmp_path):
    template=dict(success_criterion='stable_forward_recovery',reward_mode='original_all_phases',
                  proposer='source',bank='/source/bank.json',explorer_checkpoint='/old/explorer',
                  reuse_results='/old/labels',bridge_action_plan='/old/teacher')
    plan=prepare_collection_plan(template,tmp_path/'plan',master_seed=7)
    assert len(plan['batches'])==62
    assert plan['role_counts']==dict(train=1024,generator_dev=64,student_dev_id=512,student_dev_temporal=384)
    assert plan['maximum_interactions']==(1024+64)*403+(512+384)*400
    keys=set()
    for batch in plan['batches']:
        spec=json.loads(Path(batch['collection_spec']).read_text())
        assert spec['explorer_checkpoint'] is None
        assert 'reuse_results' not in spec and 'bridge_action_plan' not in spec
        assert spec['delta_limit']==[batch['amplitude']]*4
        if batch['panel']=='student_dev_temporal':
            assert batch['amplitude']==.25
        assert spec['pulse_start_schedule']==[batch['onset']]
        for episode in batch['episode_ids']:
            assert (batch['role'],episode) not in keys
            keys.add((batch['role'],episode))
    assert len(keys)==1984


def terminal_plan(tmp_path,label):
    root=tmp_path/'collection'
    row=dict(snapshot_context_sha256='terminal-context',prefix_terminal=True,prefix_label=label,
             prefix_file=str(root/'prefixes.npz'),index=0,label=None,
             logical_episode=dict(episode_id=0,role='student_dev',master_seed=7,round=0))
    write(root/'candidates.json',[row]);write(root/'status.json',dict(phase='completed',charged_interactions=400))
    plan=dict(batches=[dict(collection_output=str(root),full_episode=True,count=1,episode_ids=[0],
        role='student_dev',panel='student_dev_id',amplitude=.25,onset=0)],master_seed=7,round=0,
        source_policy='source',maximum_interactions=400)
    path=tmp_path/'plan.json';write(path,plan)
    return path


@pytest.mark.parametrize('label',[0,None,1])
def test_terminal_prefix_never_dropped_from_denominator(tmp_path,label):
    plan=terminal_plan(tmp_path,label)
    result=aggregate_collection(plan,tmp_path/'rows.json',source_actor_sha256='actor')
    assert result['denominator']==1
    assert result['rows'][0]['label']==label
    assert result['rows'][0]['root_episode_id'].endswith('prefixes.npz::0')


def test_rejects_logical_identity_drift(tmp_path):
    plan=terminal_plan(tmp_path,0)
    raw=json.loads(plan.read_text());raw['master_seed']=8;write(plan,raw)
    with pytest.raises(ValueError,match='logical episode'):
        aggregate_collection(plan,tmp_path/'rows.json',source_actor_sha256='actor')


def test_support_keeps_all_pending_and_excludes_development(tmp_path):
    from jit_dvgc.generative_bridge.source_data import build_training_support
    from jit_dvgc.evidence_integrity import canonical_sha256
    from jit_dvgc.generative_bridge.contracts import file_sha
    entries=[];locks={}
    def snapshot(key):
        folder=tmp_path/'snapshots'/key
        write(folder/'identity.json',{'parent_trajectory':key})
        (folder/'snapshot.pkl').write_bytes(b'fixture')
        for p in folder.iterdir():locks[str(p)]=file_sha(p)
        return str(folder)
    for phase in ('upstream','downstream'):
        entries.append(dict(key=phase,phase=phase,snapshot=snapshot(phase),sampling_weight=1.,
            witnessed=True,evidence_status='witnessed',role='train',trajectory_id=phase,labels={'source':1}))
    support=dict(schema='jit_iterative_witnessed_support_v1',role='train',final_test_used=False,
        entries=entries,inputs=locks.copy())
    support['support_sha256']=canonical_sha256(support)
    write(tmp_path/'seed/support.json',support)
    write(tmp_path/'seed/status.json',dict(source_policy='source',historical_support_imported=False,phase='completed'))
    rows=[]
    for i in range(130):
        key=f'pending{i}'
        rows.append(dict(root_id=key,root_episode_id=key,snapshot_context_sha256=key,phase='upstream',
            snapshot=snapshot(key),state_sha256=key,prefix_file='/trace',index=i,cell=key,
            data_role='train',label=0,coordinates={}))
    rows.append(dict(data_role='generator_dev',label=0))
    aggregated=dict(source_policy='source',prior_labels_imported=False,rows=rows,inputs=locks)
    write(tmp_path/'rows.json',aggregated)
    result=build_training_support(tmp_path/'seed/support.json',tmp_path/'rows.json',tmp_path/'support.json',source_policy='source')
    pending=[r for r in result['entries'] if r['evidence_status']=='pending']
    assert len(pending)==130
    assert sum(r['sampling_weight'] for r in pending)==pytest.approx(.5)
    assert result['pending_fraction']==.5
