from dataclasses import replace
from pathlib import Path
import pytest
import jax.numpy as jp
from jit_dvgc.config import load_config
from jit_dvgc.descent_semantics import initial_descent_events, advance_descent_events, DescentSignals


@pytest.mark.parametrize("ticks", [25, 100])
def test_full_recovery_window_after_contact_and_reset_on_instability(ticks):
    cfg = replace(load_config(Path(__file__).parents[1]/'configs/descent_recovery_smoke.json').descent,
                  continuous_stability=True, recovery_ticks=ticks, min_post_contact_forward_progress=0.)
    state = initial_descent_events(jp.array(0.)).replace(airborne_seen=jp.array(True))
    def step(s, **kw):
        signals = dict(x=1.,front_clearance=0.,rear_clearance=0.,maximum_wheel_penetration=0.,
                       body_contact=False,finite=True,roll=0.,pitch=0.,backward_exit=False,forward_velocity=1.)
        signals.update(kw)
        return advance_descent_events(s,DescentSignals(**{k:jp.asarray(v) for k,v in signals.items()}),cfg)
    state=step(state)
    assert state.post_contact_ticks == 0 and not state.recovery_success
    for _ in range(ticks-1):state=step(state)
    assert not state.recovery_success
    for kw in ({'forward_velocity':0.}, {'roll':.3}, {'pitch':.4}, {'front_clearance':.1}, {'body_contact':True}):
        interrupted=step(state,**kw)
        assert interrupted.post_contact_ticks == 0 and not interrupted.recovery_success
    assert step(state).recovery_success


def test_forward_survival_accepts_positive_speed_but_not_zero():
    cfg=replace(load_config(Path(__file__).parents[1]/'configs/descent_recovery_smoke.json').descent,
        continuous_stability=True,forward_survival_only=True,stable_min_forward_velocity=0.,
        recovery_ticks=100,min_post_contact_forward_progress=0.)
    state=initial_descent_events(jp.array(0.)).replace(airborne_seen=jp.array(True),valid_contact_seen=jp.array(True))
    sig=DescentSignals(*[jp.asarray(x) for x in (1.,.03,.03,0.,False,True,.2,.3,False)],forward_velocity=jp.asarray(.001))
    for _ in range(99):state=advance_descent_events(state,sig,cfg)
    assert not state.recovery_success
    assert advance_descent_events(state,sig,cfg).recovery_success
    stopped=advance_descent_events(state,sig.replace(forward_velocity=jp.asarray(0.)),cfg)
    assert stopped.post_contact_ticks==0 and not stopped.recovery_success


def test_half_second_contract_terminates_without_one_metre_gate():
    from jit_dvgc.descent_semantics import classify_descent_terminal
    cfg=load_config(Path(__file__).parents[1]/'configs/descent_stable_forward_0p5s.json')
    assert cfg.descent.recovery_ticks == 25
    assert cfg.descent.continuous_stability and not cfg.descent.forward_survival_only
    state=initial_descent_events(jp.array(0.)).replace(airborne_seen=jp.array(True))
    def advance(x):
        sig=DescentSignals(*[jp.asarray(v) for v in (x,0.,0.,0.,False,True,0.,0.,False)],forward_velocity=jp.asarray(.5))
        return sig,advance_descent_events(state,sig,cfg.descent)
    sig,state=advance(0.)
    for i in range(1,25):sig,state=advance(i*.01)
    assert not state.recovery_success
    sig,state=advance(.25)
    terminal=classify_descent_terminal(sig,state,cfg.descent,cfg.physical_limits,jp.asarray(26),400)
    assert terminal.success and terminal.terminated and not terminal.physical_failure
