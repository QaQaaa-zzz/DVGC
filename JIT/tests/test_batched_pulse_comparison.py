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


def _fixture(tmp_path):
    from jit_dvgc.batched_pulse_comparison import batch_plan,verify_batch
    from jit_dvgc.rsi_comparison import write,file_sha
    methods=[dict(key=k,label=k,nominal_trace=str(tmp_path/f'{k}_nominal.npz')) for k in ('baseline','fresh_rsi','phase_u')]
    spec=dict(methods=methods,root_qpos_address=0,episodes=3,batches=batch_plan(3,2,99),contract={'horizon':2},maximum_interactions=18,role='fixture')
    receipts=[]
    for batch in spec['batches']:
        directory=tmp_path/'batches'/f'{batch["index"]:04d}'
        n=batch['count']
        for m in methods:
            p=directory/m['key'];p.mkdir(parents=True)
            tape={k:np.zeros((2,n),bool) for k in ('success','physical_failure','mask')}
            tape.update(prefix_mask=np.ones((2,n),bool),terminal=np.ones((2,n),bool),end_code=np.zeros((2,n),int),
                        qpos=np.zeros((2,n,3)),time=np.full((2,n),.02),
                        front_wheel_clearance=np.zeros((2,n)),rear_wheel_clearance=np.zeros((2,n)))
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
