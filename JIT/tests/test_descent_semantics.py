from __future__ import annotations
import math
import jax
import jax.numpy as jp
import pytest
from jit_dvgc.config import DescentConfig, PhysicalLimits
from jit_dvgc.constants import END_BACKWARD_EXIT, END_NONFINITE, END_PITCH_LIMIT, END_PROHIBITED_CONTACT, END_RECOVERY_SUCCESS, END_ROLL_LIMIT, END_TIMEOUT
from jit_dvgc.descent_semantics import DescentEventState, DescentSignals, advance_descent_events, classify_descent_terminal, initial_descent_events

CFG=DescentConfig(25,0.05,0.01,0.02,0.05,True,True,1.,1.,1.,1.,1.,1.,1.)
LIM=PhysicalLimits(0.6,0.8,1.0)
def sig(**kw):
 d=dict(x=1.,front_clearance=.1,rear_clearance=.1,maximum_wheel_penetration=0.,body_contact=False,finite=True,roll=0.,pitch=0.,backward_exit=False); d.update(kw); return DescentSignals(**{k:jp.asarray(v) for k,v in d.items()})
def advance(s, x=1., **kw): return advance_descent_events(s,sig(x=x,**kw),CFG)
def airborne(): return advance(initial_descent_events(jp.asarray(0.)), x=1.)

def test_contact_requires_airborne_and_both_wheels():
 s=initial_descent_events(jp.asarray(0.)); s=advance(s,front_clearance=.01,rear_clearance=.1); assert not bool(s.airborne_seen)
 s=advance(s,front_clearance=.1,rear_clearance=.1); assert bool(s.airborne_seen) and not bool(s.valid_contact_seen)
 s=advance(s,front_clearance=.0,rear_clearance=.1); assert bool(s.valid_contact_seen) and float(s.contact_x)==1.

@pytest.mark.parametrize("kw",[dict(maximum_wheel_penetration=.1),dict(body_contact=True)])
def test_bad_contact_rejected(kw):
 s=airborne(); s=advance(s,front_clearance=0.,**kw); assert not bool(s.valid_contact_seen)

def test_bounce_does_not_reset_and_success_requires_25_ticks_and_progress():
 s=airborne(); s=advance(s,front_clearance=0.,x=2.); assert bool(s.valid_contact_seen)
 for _ in range(23): s=advance(s,x=2.)
 assert int(s.post_contact_ticks)==24 and not bool(s.recovery_success)
 s=advance(s,x=2.05,front_clearance=.2,rear_clearance=.2); assert bool(s.recovery_success)

def test_failure_causes_and_priority_and_timeout():
 s=airborne(); s=advance(s,front_clearance=0.,x=2.)
 for kwargs,code in [(dict(finite=False),END_NONFINITE),(dict(roll=1.),END_ROLL_LIMIT),(dict(pitch=1.),END_PITCH_LIMIT),(dict(body_contact=True),END_PROHIBITED_CONTACT),(dict(backward_exit=True),END_BACKWARD_EXIT)]:
  t=classify_descent_terminal(sig(**kwargs),s,CFG,LIM,jp.asarray(1),100); assert bool(t.physical_failure) and int(t.end_code)==code
 s=s.replace(recovery_success=jp.asarray(True)); t=classify_descent_terminal(sig(body_contact=True),s,CFG,LIM,jp.asarray(1),100); assert not bool(t.success)
 t=classify_descent_terminal(sig(),s,CFG,LIM,jp.asarray(99),100); assert bool(t.truncated) and not bool(t.terminated) and int(t.end_code)==END_TIMEOUT

def test_success_reason_and_jit_vmap():
 s=airborne();
 s=advance(s,x=2.,front_clearance=0.)
 for _ in range(24): s=advance(s,x=2.05,front_clearance=0.)
 t=classify_descent_terminal(sig(x=2.05,front_clearance=0.),s,CFG,LIM,jp.asarray(25),100); assert bool(t.success) and int(t.end_code)==END_RECOVERY_SUCCESS
 fn=jax.jit(lambda x: advance_descent_events(initial_descent_events(jp.asarray(0.)),x,CFG)); batch=DescentSignals(*[jp.asarray([v,v]) for v in (1.,.1,.1,0.,False,True,0.,0.,False)]); out=jax.vmap(fn)(batch)
 assert out.airborne_seen.shape==(2,)

def test_phase_u_semantics_regression(jit_root):
 from jit_dvgc.config import load_config
 from jit_dvgc.semantics import initial_event_state
 c=load_config(jit_root/"configs/phase_u_smoke.json"); e=initial_event_state(jp.asarray(2.5),c); assert bool(e.jump_signal)
