from types import SimpleNamespace
import jax
import jax.numpy as jp

from dvgc.composite import CompositeSession,composite_rollout


class Match:
    def __init__(self,value): self.value=value
    def match(self,state): return self.value,0.25


class State:
    def __init__(self,step=0,done=0,final=0):
        self.obs={"state":jp.zeros(2)}; self.done=jp.asarray(done); self.info={"episode_step":jp.asarray(step),"recovery_success":jp.asarray(final),"terminated":jp.asarray(done),"truncated":jp.asarray(0),"end_code":jp.asarray(1 if final else 0)}


def infer(value): return lambda obs,key:(jp.full((1,),value),{})


def test_handoff_is_irreversible_and_chain_is_separate_from_final():
    def step(state,action): return State(int(state.info["episode_step"])+1,done=0,final=0)
    session=CompositeSession(SimpleNamespace(),("flight","landing"),{"flight":infer(0),"landing":infer(1)},{"flight":Match(True)},State(),jax.random.PRNGKey(0))
    session.step(step_fn=step); assert session.active_stage=="landing" and len(session.handoffs)==1
    session.step(step_fn=step); assert session.active_stage=="landing" and len(session.handoffs)==1


def test_final_without_handoff_is_chain_missed_final():
    def step(state,action): return State(1,done=1,final=1)
    _,result=composite_rollout(SimpleNamespace(),("flight","landing"),{"flight":infer(0),"landing":infer(1)},{"flight":Match(False)},State(),jax.random.PRNGKey(0),horizon=2,step_fn=step)
    assert result["final"] and not result["chain"] and result["chain_missed_final"]
