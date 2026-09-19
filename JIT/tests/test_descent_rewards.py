from __future__ import annotations
import jax
import jax.numpy as jp
import pytest
from jit_dvgc.config import DescentConfig
from jit_dvgc.descent_rewards import DescentRewardInputs, descent_recovery_reward

CFG=DescentConfig(
    recovery_ticks=25, min_airborne_clearance=.05, contact_clearance_threshold=.01,
    max_wheel_penetration=.02, min_post_contact_forward_progress=.05,
    terminate_on_body_contact=True, reward_contact=5., reward_recovery_tick=1.,
    penalty_bad_contact=20., penalty_failure=40., penalty_timeout=10.,
    reward_forward_progress=2., reward_success=30., total_min=-100., total_max=50.,
    roll_posture_coeff=3., pitch_posture_coeff=.5, roll_rate_penalty_coeff=.5,
    pitch_rate_penalty_coeff=.15, action_smoothness_penalty_coeff=.01,
)
def inp(**kw):
 d=dict(x_delta=0.,valid_contact=False,previous_valid_contact=False,post_contact=True,
        recovery_success=False,previous_recovery_success=False,bad_contact=False,
        physical_failure=False,timeout=False,roll=0.,pitch=0.,roll_rate=0.,pitch_rate=0.,
        action=jp.zeros(4),last_action=jp.zeros(4)); d.update(kw)
 return DescentRewardInputs(**{k:jp.asarray(v) for k,v in d.items()})
def test_forward_only_positive_and_contact_one_shot():
 r=descent_recovery_reward(inp(x_delta=.2,valid_contact=True),CFG); assert float(r.components.forward_progress)==pytest.approx(.4); assert float(r.components.contact)==5.
 r=descent_recovery_reward(inp(x_delta=-.2,valid_contact=True,previous_valid_contact=True),CFG); assert float(r.components.forward_progress)==0.; assert float(r.components.contact)==0.
def test_recovery_tick_success_bad_failure_timeout_are_separate():
 assert float(descent_recovery_reward(inp(),CFG).components.recovery_tick)==1.
 assert float(descent_recovery_reward(inp(recovery_success=True),CFG).components.success)==30.
 assert float(descent_recovery_reward(inp(bad_contact=True),CFG).components.bad_contact)==-20.
 assert float(descent_recovery_reward(inp(physical_failure=True),CFG).components.failure)==-40.
 assert float(descent_recovery_reward(inp(timeout=True),CFG).components.timeout)==-10.

def test_phase_u_posture_rates_and_action_smoothness_are_components():
 r=descent_recovery_reward(inp(roll=0.1, pitch=0.1, roll_rate=2., pitch_rate=2., action=jp.ones(4), last_action=jp.zeros(4)),CFG)
 assert float(r.components.roll_posture) == pytest.approx(3.*(-.1*((0.1*180/jp.pi)-5.)), rel=1e-5)
 assert float(r.components.roll_rate) == pytest.approx(-.5*.125*4.)
 assert float(r.components.pitch_rate) == pytest.approx(-.15*.125*4.)
 assert float(r.components.action_smoothness) == pytest.approx(-.01*4.)

def test_reward_clips_to_configured_bounds():
 r=descent_recovery_reward(inp(x_delta=100.,valid_contact=True,recovery_success=True),CFG)
 assert float(r.total)==50.
 r=descent_recovery_reward(inp(bad_contact=True,physical_failure=True,timeout=True,roll_rate=100.,pitch_rate=100.,action=jp.ones(4)*100.),CFG)
 assert float(r.total)==-100.
def test_one_shot_and_combined_sum_jit_vmap():
 r=descent_recovery_reward(inp(x_delta=.1,valid_contact=True,recovery_success=True),CFG); assert float(r.total)==pytest.approx(sum(float(v) for v in r.components.__dict__.values()))
 fn=jax.jit(lambda x: descent_recovery_reward(x,CFG).total)
 batch=DescentRewardInputs(*[jp.asarray([v,v]) for v in (0.,False,False,True,False,False,False,False,False,0.,0.,0.,0.,jp.zeros(4),jp.zeros(4))])
 assert fn(batch).shape==(2,)
