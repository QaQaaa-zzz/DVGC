import numpy as np
from jit_dvgc.rsi_comparison import episode_results


def test_early_failure_and_timeout_remain_in_comparison_denominator():
    shape = (4, 3)
    tape = {k: np.zeros(shape, bool) for k in ('prefix_mask', 'success', 'physical_failure', 'terminal', 'mask')}
    tape['prefix_mask'][:, 0] = True
    tape['prefix_mask'][:2, 1] = True
    tape['prefix_mask'][:, 2] = True
    tape['success'][3, 0] = True; tape['terminal'][3, 0] = True
    tape['physical_failure'][1, 1] = True; tape['terminal'][1, 1] = True
    tape['time'] = np.broadcast_to(np.arange(1, 5)[:, None] * .02, shape)
    tape['qpos'] = np.zeros((4, 3, 3))
    rows = episode_results(tape, 0)
    assert len(rows) == 3 and sum(r['success'] for r in rows) == 1
    assert rows[1]['control_steps'] == 2 and rows[1]['pulse_applied_steps'] == 0
    assert rows[2]['horizon_exhausted'] is True


def test_conflicting_success_and_failure_is_not_success():
    tape = {k: np.ones((1, 1), bool) for k in ('prefix_mask', 'success', 'physical_failure', 'terminal', 'mask')}
    tape.update(time=np.array([[.02]]), qpos=np.zeros((1, 1, 3)))
    row = episode_results(tape, 0)[0]
    assert row['conflict'] and not row['success']


def test_environment_timeout_is_counted_separately_from_rollout_horizon():
    from jit_dvgc.constants import END_TIMEOUT
    tape = {k: np.zeros((1, 1), bool) for k in ('success', 'physical_failure', 'mask')}
    tape.update(prefix_mask=np.ones((1, 1), bool), terminal=np.ones((1, 1), bool),
                time=np.array([[8.]]), qpos=np.zeros((1, 1, 3)), end_code=np.array([[END_TIMEOUT]]))
    row = episode_results(tape, 0)[0]
    assert row['environment_timeout'] and not row['horizon_exhausted'] and not row['success']


def test_report_retains_baseline_counts_and_all_failed_episodes(tmp_path):
    from jit_dvgc.constants import END_TIMEOUT
    from jit_dvgc.rsi_comparison import report
    from jit_dvgc.jump_evidence_validation import write, read
    write(tmp_path/'spec.json',dict(root_qpos_address=0,episodes=2,baseline='initial',training_steps=3200,role='development'))
    for method in ('baseline','fresh_rsi'):
        for condition,n in [('nominal',1),('random',2)]:
            p=tmp_path/'evaluation'/f'{method}_{condition}';p.mkdir(parents=True)
            tape={k:np.zeros((2,n),bool) for k in ('success','physical_failure','terminal','mask')}
            tape.update(prefix_mask=np.ones((2,n),bool),end_code=np.full((2,n),END_TIMEOUT),
                        qpos=np.zeros((2,n,3)),time=np.full((2,n),.02),
                        front_wheel_clearance=np.zeros((2,n)),rear_wheel_clearance=np.zeros((2,n)))
            for k in ('delta','requested_delta','effective_delta'):tape[k]=np.zeros((2,n,4))
            tape['terminal'][-1]=True
            np.savez_compressed(p/'prefixes.npz',**tape)
            write(p/'status.json',dict(charged_interactions=2*n))
    summary=report(tmp_path)
    assert summary['baseline']['successes']==0 and summary['baseline']['episodes']==2
    assert summary['baseline']['environment_timeouts']==2
    assert summary['baseline_policy']=='initial'
    assert summary['aligned_exclusions']=={'baseline':2,'fresh_rsi':2}
    assert read(tmp_path/'comparison/summary.json')['baseline']['episodes']==2
