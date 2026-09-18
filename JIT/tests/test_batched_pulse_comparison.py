import numpy as np
import pytest


def test_batches_cover_exact_budget_and_rotate_global_onsets():
    from jit_dvgc.batched_pulse_comparison import batch_plan
    from jit_dvgc.pulse_schedule import lane_onsets
    plan=batch_plan(10000,256,9182702)
    assert len(plan)==40 and sum(b['count'] for b in plan)==10000
    assert plan[-1]['count']==16 and len({b['seed'] for b in plan})==40
    seen=[]
    for b in plan:
        seen.extend(range(b['offset'],b['offset']+b['count']))
        spec=dict(num_envs=b['count'],pulse_batch_mode='mixed',pulse_start_schedule=[5,10,15,20,25,0],pulse_steps=3,horizon=400)
        onsets=lane_onsets(spec,b['offset'])
        np.testing.assert_array_equal(onsets,np.array(spec['pulse_start_schedule'])[(np.arange(b['count'])+b['offset'])%6])
    assert seen==list(range(10000))


@pytest.mark.parametrize('count,size',[(0,256),(100,-1),(1.5,2)])
def test_invalid_batch_budget(count,size):
    from jit_dvgc.batched_pulse_comparison import batch_plan
    with pytest.raises(ValueError):batch_plan(count,size,1)


def _fixture(tmp_path, keys=('baseline','fresh_rsi','phase_u')):
    from jit_dvgc.batched_pulse_comparison import batch_plan,verify_batch
    from jit_dvgc.rsi_comparison import write,file_sha
    methods=[dict(key=k,label=k,nominal_trace=str(tmp_path/f'{k}_nominal.npz')) for k in keys]
    spec=dict(methods=methods,root_qpos_address=0,episodes=3,batches=batch_plan(3,2,99),contract={'horizon':2},maximum_interactions=len(keys)*6,role='fixture')
    receipts=[]
    for batch in spec['batches']:
        directory=tmp_path/'batches'/f'{batch["index"]:04d}'
        n=batch['count']
        for m in methods:
            p=directory/m['key'];p.mkdir(parents=True)
            tape={k:np.zeros((2,n),bool) for k in ('success','physical_failure','mask')}
            tape.update(prefix_mask=np.ones((2,n),bool),terminal=np.ones((2,n),bool),end_code=np.zeros((2,n),int),
                        qpos=np.zeros((2,n,3)),time=np.full((2,n),.02),
                        front_wheel_clearance=np.zeros((2,n)),rear_wheel_clearance=np.zeros((2,n)),
                        valid_contact_seen=np.ones((2,n),bool),recovery_ticks=np.zeros((2,n),int))
            for k in ('delta','requested_delta','effective_delta'):tape[k]=np.zeros((2,n,4))
            tape['delta'][:]=batch['seed']
            tape['success'][-1]=m['key']=='phase_u'
            np.savez_compressed(p/'prefixes.npz',**tape)
            np.savez_compressed(m['nominal_trace'],**tape)
            write(p/'status.json',dict(phase='completed',charged_interactions=2*n))
        verified=verify_batch(spec,batch,directory);write(directory/'verification.json',verified)
        receipts.append(dict(batch=batch,verification=str(directory/'verification.json'),verification_sha256=file_sha(directory/'verification.json')))
    write(tmp_path/'spec.json',spec);write(tmp_path/'receipts.json',receipts)
    return spec


def test_batched_report_keeps_every_episode_and_cost(tmp_path):
    import csv
    from jit_dvgc.analysis.batched_pulse_report import report
    _fixture(tmp_path)
    result=report(tmp_path)
    assert result['methods']['phase_u']['successes']==3
    assert result['methods']['baseline']['episodes']==3
    assert result['charged_interactions']==18
    assert result['aligned_exclusions']==dict(baseline=3,fresh_rsi=3,phase_u=3)
    assert result['display_contract']['world_x_max_m']==5.5
    assert result['display_contract']['aligned_x_max_m']==5.5
    assert (tmp_path/'comparison/aligned_density.png').exists()
    assert (tmp_path/'comparison/plot_data.npz').exists()
    rows=list(csv.DictReader((tmp_path/'comparison/episodes.csv').open()))
    assert len(rows)==9
    for m in ('baseline','fresh_rsi','phase_u'):
        assert sorted(int(r['episode']) for r in rows if r['method']==m)==[0,1,2]
    derived=report(tmp_path,tmp_path/'new_report')
    assert derived['methods']==result['methods']
    assert 'new_report/comparison.png' in (tmp_path/'INDEX.md').read_text()


def test_verify_batch_rejects_unpaired_randomness(tmp_path):
    from jit_dvgc.batched_pulse_comparison import verify_batch
    spec=_fixture(tmp_path);directory=tmp_path/'batches/0000'
    p=directory/'phase_u/prefixes.npz';tape=dict(np.load(p));tape['delta'][0,0,0]+=1
    np.savez_compressed(p,**tape)
    with pytest.raises(AssertionError):verify_batch(spec,spec['batches'][0],directory)


def test_initial_pulse_tape_rejects_any_requested_disturbance_after_tick_two():
    from jit_dvgc.batched_pulse_comparison import verify_initial_pulse_tape
    tape=dict(prefix_mask=np.ones((5,2),bool),mask=np.zeros((5,2),bool),
              requested_delta=np.zeros((5,2,4)))
    tape['mask'][:3]=True;tape['requested_delta'][:3]=.1
    verify_initial_pulse_tape(tape,3)
    tape['requested_delta'][3,0,0]=.1
    with pytest.raises(ValueError,match='after declared initial pulse'):
        verify_initial_pulse_tape(tape,3)


@pytest.mark.parametrize('value,expected', [([0],[0]), ((0,5),[0,5])])
def test_normalize_pulse_start_schedule(value,expected):
    from jit_dvgc.batched_pulse_comparison import normalize_pulse_start_schedule
    assert normalize_pulse_start_schedule(value)==expected


@pytest.mark.parametrize('value', [[],[-1],[0,1.5],'0'])
def test_reject_invalid_pulse_start_schedule(value):
    from jit_dvgc.batched_pulse_comparison import normalize_pulse_start_schedule
    with pytest.raises(ValueError):normalize_pulse_start_schedule(value)


def test_batched_report_supports_four_methods(tmp_path):
    from jit_dvgc.analysis.batched_pulse_report import report
    spec=_fixture(tmp_path,keys=('baseline','fresh_rsi','phase_u','lineage_repair_0070'))
    result=report(tmp_path)
    assert set(result['methods'])=={m['key'] for m in spec['methods']}
    assert (tmp_path/'comparison/comparison.png').exists()


def test_batched_report_requires_declared_generated_nominal_receipt(tmp_path):
    from jit_dvgc.analysis.batched_pulse_report import report
    from jit_dvgc.rsi_comparison import write
    spec=_fixture(tmp_path,keys=('baseline','repair_0070'))
    spec['methods'][-1]['generate_nominal']=True;write(tmp_path/'spec.json',spec)
    with pytest.raises(ValueError,match='generated nominal receipts'):
        report(tmp_path)


def test_load_additional_method_locks_policy_and_declares_nominal(tmp_path):
    from jit_dvgc.batched_pulse_comparison import load_additional_methods
    from jit_dvgc.rsi_comparison import write
    checkpoint=tmp_path/'checkpoint';checkpoint.mkdir()
    write(checkpoint/'identity.json',{'name':'identity'});(checkpoint/'payload.pkl').write_bytes(b'payload')
    frozen=tmp_path/'frozen.json';write(frozen,{'name':'lineage_repair_0070'})
    formal=tmp_path/'formal.json';write(formal,{'name':'formal'})
    policy=dict(xml_sha256='xml',action_order=['a'],actor_frame_fields=['frame'],actor_task_fields=['task'],
                formal_config=str(formal),checkpoint=str(checkpoint))
    bank=tmp_path/'bank.json';write(bank,{'members':[dict(name='lineage_repair_0070',frozen_policy=str(frozen),policy=policy)]})
    manifest=tmp_path/'methods.json';write(manifest,{'schema':'jit_additional_comparison_methods_v1','methods':[
        dict(key='repair_0070',label='lineage_repair_0070',short_label='repair_0070',bank=str(bank),proposer='lineage_repair_0070')]})
    inputs={};template=dict(bank='old',proposer='old',pulse_start_schedule=[0],phase_policy={'must':'be removed'})
    methods=load_additional_methods(manifest,template,tmp_path/'run',inputs,policy)
    assert len(methods)==1 and methods[0]['key']=='repair_0070'
    assert methods[0]['template']['pulse_start_schedule']==[0]
    assert methods[0]['short_label']=='repair_0070'
    assert 'phase_policy' not in methods[0]['template']
    assert methods[0]['generate_nominal'] is True
    assert methods[0]['nominal_trace'].endswith('/run/nominal/repair_0070/prefixes.npz')
    assert {str(manifest),str(bank),str(frozen),str(formal),str(checkpoint/'identity.json'),str(checkpoint/'payload.pkl')} <= set(inputs)
