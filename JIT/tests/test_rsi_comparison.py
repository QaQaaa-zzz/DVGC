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
