from dataclasses import replace
from pathlib import Path
import jax.numpy as jp
from jit_dvgc.config import load_config
from jit_dvgc.descent_semantics import initial_descent_events, advance_descent_events, DescentSignals


def test_full_two_seconds_after_contact_and_reset_on_instability():
    cfg = replace(load_config(Path(__file__).parents[1]/'configs/descent_recovery_smoke.json').descent,
                  continuous_stability=True, recovery_ticks=100, min_post_contact_forward_progress=0.)
    state = initial_descent_events(jp.array(0.)).replace(airborne_seen=jp.array(True))
    def step(s, **kw):
        signals = dict(x=1.,front_clearance=0.,rear_clearance=0.,maximum_wheel_penetration=0.,
                       body_contact=False,finite=True,roll=0.,pitch=0.,backward_exit=False,forward_velocity=1.)
        signals.update(kw)
        return advance_descent_events(s,DescentSignals(**{k:jp.asarray(v) for k,v in signals.items()}),cfg)
    state=step(state)
    assert state.post_contact_ticks == 0 and not state.recovery_success
    for _ in range(99):state=step(state)
    assert not state.recovery_success
    for kw in ({'forward_velocity':0.}, {'roll':.3}, {'pitch':.4}, {'front_clearance':.1}, {'body_contact':True}):
        interrupted=step(state,**kw)
        assert interrupted.post_contact_ticks == 0 and not interrupted.recovery_success
    assert step(state).recovery_success
