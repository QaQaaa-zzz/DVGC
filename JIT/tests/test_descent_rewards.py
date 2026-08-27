from __future__ import annotations
import jax
import jax.numpy as jp
import pytest
from jit_dvgc.config import DescentConfig
from jit_dvgc.descent_rewards import DescentRewardInputs, descent_recovery_reward

CFG=DescentConfig(25,.05,.01,.02,.05,True,True,1.,1.,1.,1.,1.,2.,10.)
def inp(**kw):
 d=dict(x_delta=0.,valid_contact=False,previous_valid_contact=False,post_contact=True,recovery_success=False,previous_recovery_success=False,bad_contact=False,physical_failure=False,timeout=False); d.update(kw); return DescentRewardInputs(**{k:jp.asarray(v) for k,v in d.items()})
def test_forward_only_positive_and_contact_one_shot():
 r=descent_recovery_reward(inp(x_delta=.2,valid_contact=True),CFG); assert float(r.components.forward_progress)==pytest.approx(.4); assert float(r.components.contact)==1.
 r=descent_recovery_reward(inp(x_delta=-.2,valid_contact=True,previous_valid_contact=True),CFG); assert float(r.components.forward_progress)==0.; assert float(r.components.contact)==0.
def test_recovery_tick_success_bad_failure_timeout_are_separate():
 assert float(descent_recovery_reward(inp(),CFG).components.recovery_tick)==1.
 assert float(descent_recovery_reward(inp(recovery_success=True),CFG).components.success)==10.
 assert float(descent_recovery_reward(inp(bad_contact=True),CFG).components.bad_contact)==-1.
 assert float(descent_recovery_reward(inp(physical_failure=True),CFG).components.failure)==-1.
 assert float(descent_recovery_reward(inp(timeout=True),CFG).components.timeout)==-1.
def test_one_shot_and_combined_sum_jit_vmap():
 r=descent_recovery_reward(inp(x_delta=.1,valid_contact=True,recovery_success=True),CFG); assert float(r.total)==pytest.approx(sum(float(v) for v in r.components.__dict__.values()))
 fn=jax.jit(lambda x: descent_recovery_reward(x,CFG).total); batch=DescentRewardInputs(*[jp.asarray([v,v]) for v in (0.,False,False,True,False,False,False,False,False)])
 assert fn(batch).shape==(2,)
