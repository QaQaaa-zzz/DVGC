import numpy as np
import jax.numpy as jnp

from cli.run_backward_descent_nominal_pilot import select_incremental_proposals
from cli.run_backward_descent_cem_pilot import select_unresolved_for_cem
from dvgc.backward_search import (active_prefix_exact, backward_lexicographic_order,
                                  bounded_cem, compact_observation_residual_adapter,
                                  mask_pre_handoff_residual)


def test_backward_order_prioritizes_final_then_entry_not_survival():
    rows={"final_recovery":np.array([0,1,0],bool),"downstream_entry":np.array([1,1,1],bool),"minimum_distance":np.array([.1,9.,.01]),"minimum_margin":np.zeros(3),"survival":np.array([24,1,24])}
    assert backward_lexicographic_order(rows).tolist()==[1,2,0]


def test_active_prefix_ignores_terminal_tail_but_checks_events():
    base={"downstream_entry":np.array([True]),"final_recovery":np.array([False]),"survival":np.array([3]),"entry_tick":np.array([2]),"termination_tick":np.array([4]),"end_code":np.array([5]),"minimum_distance":np.array([.2]),"minimum_margin":np.array([-.1]),"entry_qpos":np.zeros((1,12)),"entry_qvel":np.zeros((1,11)),"actions":np.zeros((6,1,4)),"active_mask":np.ones((6,1),bool),"phase_trace":np.zeros((6,1),int)}
    other={k:np.copy(v) for k,v in base.items()};other["actions"][5]=9
    assert active_prefix_exact(base,other)[0]
    other["actions"][2]=1
    assert not active_prefix_exact(base,other)[0]


def test_incremental_selection_skips_prior_and_honors_region_and_cap():
    rows=[{"proposal_id":str(i),"candidate_id":candidate,"region":region} for i,(candidate,region) in enumerate([
        ("a","early"),("a","early"),("a","early"),("a","early"),("b","middle"),("c","early")])]
    selected=select_incremental_proposals(rows,{"1"},4,"early",per_candidate_cap=2)
    assert [row["proposal_id"] for row in selected]==["0","2","5"]


def test_bounded_cem_uses_fixed_budget_and_respects_bounds():
    def rollout(state,knots,key):
        count=knots.shape[0];score=jnp.sum(knots**2,axis=(1,2))
        return {"final_recovery":jnp.zeros(count,bool),"downstream_entry":jnp.zeros(count,bool),"minimum_distance":score,"minimum_margin":-score,"survival":jnp.ones(count,jnp.int32),"end_code":jnp.zeros(count,jnp.int32)}
    residual,summary,history=bounded_cem(rollout,lambda count:jnp.zeros(count),seed=3,samples=16,iterations=2,knot_count=2,bound=.2)
    assert residual.shape==(2,4) and np.max(np.abs(residual))<=.2
    assert len(history)==16 and summary["minimum_distance"]>=0


def test_residual_mask_broadcasts_per_batch_row():
    knot=jnp.arange(12,dtype=jnp.float32).reshape(3,4)
    masked=mask_pre_handoff_residual(knot,jnp.asarray([False,True,False]),jnp.asarray([True,True,False]))
    np.testing.assert_array_equal(masked,np.vstack([np.arange(4),np.zeros(4),np.zeros(4)]))


def test_compact_adapter_is_exact_at_prototype_and_zero_outside():
    apply=compact_observation_residual_adapter(jnp.array([[0.,0.]]),jnp.array([[.2,-.1]]),
                                               jnp.zeros(2),jnp.ones(2),.5)
    base=jnp.zeros((2,2));obs=jnp.array([[0.,0.],[2.,2.]])
    np.testing.assert_allclose(np.asarray(apply(obs,base)),np.array([[.2,-.1],[0.,0.]]),atol=1e-7)


def test_cem_selection_never_retries_attempted_or_p0_proposals():
    def row(identifier,controller,p0,distance,region="middle"):
        return {"proposal":{"proposal_id":identifier,"region":region},"controller":controller,"P0":{"pass":p0},"repeats":[{"minimum_distance":distance}]}
    rows=[row("a","frozen_pi_D_nominal_L0",False,2),row("a","bounded_residual_cem_64x5_h8",False,1),row("b","frozen_pi_D_nominal_L0",True,.1),row("c","frozen_pi_D_nominal_L0",False,3),row("d","frozen_pi_D_nominal_L0",False,1,"early")]
    assert [x["proposal"]["proposal_id"] for x in select_unresolved_for_cem(rows,3)]==["d","c"]
    assert [x["proposal"]["proposal_id"] for x in select_unresolved_for_cem(rows,3,"middle")]==["c"]


def test_second_tier_selects_only_failed_first_tier_once():
    def row(identifier,controller,p0,distance):return {"proposal":{"proposal_id":identifier,"region":"middle"},"controller":controller,"P0":{"pass":p0},"repeats":[{"minimum_distance":distance}]}
    rows=[row("a","frozen_pi_D_nominal_L0",False,3),row("a","bounded_residual_cem_64x5_h8",False,1),row("b","bounded_residual_cem_64x5_h8",True,.1),row("c","bounded_residual_cem_64x5_h8",False,2),row("c","bounded_residual_cem_128x6_h12",False,1)]
    assert [x["proposal"]["proposal_id"] for x in select_unresolved_for_cem(rows,4,tier="second")]==["a"]
