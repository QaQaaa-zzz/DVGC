"""MJX L0 full-downstream rollouts and bounded local CEM for backward Tubes."""
from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from dvgc.descent_probe import formal_dynamic_margin
from dvgc.runtime import build_inference


def make_descent_landing_rollout(env: Any, descent_params: Any, landing_params: Any,
                                 *, horizon: int = 200, residual_ticks: int = 8,
                                 ticks_per_knot: int = 4):
    """Roll one uninterrupted MJX lineage, switching only policy after C_L."""
    descent = build_inference(env, descent_params, deterministic=True)
    landing = build_inference(env, landing_params, deterministic=True)
    step = jax.vmap(env.step); feature_fn = jax.vmap(env._physical_feature)

    def rollout(state, residual_knots, key):
        count = residual_knots.shape[0]
        active=jnp.ones((count,),bool); handed=jnp.zeros((count,),bool)
        survival=jnp.zeros((count,),jnp.int32); entry_tick=jnp.full((count,),-1,jnp.int32)
        termination_tick=jnp.full((count,),horizon,jnp.int32); end_code=jnp.zeros((count,),jnp.int32)
        final=jnp.zeros((count,),bool); minimum_distance=jnp.full((count,),jnp.inf,jnp.float32)
        minimum_margin=jnp.full((count,),jnp.inf,jnp.float32)
        entry_qpos=jnp.zeros((count,env.mj_model.nq),jnp.float32);entry_qvel=jnp.zeros((count,env.mj_model.nv),jnp.float32)
        actions=jnp.zeros((horizon,count,env.action_size),jnp.float32)
        active_mask=jnp.zeros((horizon,count),bool); phase_trace=jnp.zeros((horizon,count),jnp.int32)

        def body(carry,tick):
            (state,active,handed,survival,entry_tick,termination_tick,end_code,final,
             minimum_distance,minimum_margin,entry_qpos,entry_qvel,actions,active_mask,phase_trace)=carry
            da,_=descent(state.obs,jax.random.fold_in(key,tick));la,_=landing(state.obs,jax.random.fold_in(key,tick+100000))
            base=jnp.where(handed[:,None],la,da)
            knot=residual_knots[:,jnp.minimum(tick//ticks_per_knot,residual_knots.shape[1]-1)]
            residual=jnp.where((tick<residual_ticks)&(~handed),knot,jnp.zeros_like(knot))
            command=jnp.clip(base+residual,-1.,1.);next_state=step(state,command)
            chain_now=next_state.info["chain_ever"]>0;new_handoff=(~handed)&chain_now
            entry_tick=jnp.where(new_handoff,tick+1,entry_tick);handed=handed|chain_now
            entry_qpos=jnp.where(new_handoff[:,None],next_state.data.qpos,entry_qpos)
            entry_qvel=jnp.where(new_handoff[:,None],next_state.data.qvel,entry_qvel)
            done=next_state.done.astype(bool);alive=active&(~done)
            survival=survival+alive.astype(jnp.int32);termination_tick=jnp.where(active&done,tick+1,termination_tick)
            end_code=jnp.where(active&done,next_state.info["end_code"],end_code)
            final=final|(active&(next_state.info["recovery_success"]>0))
            distance=next_state.metrics["state/tube_distance_z"]
            minimum_distance=jnp.where(active,jnp.minimum(minimum_distance,distance),minimum_distance)
            margin=formal_dynamic_margin(feature_fn(next_state.data),env._config)
            minimum_margin=jnp.where(active,jnp.minimum(minimum_margin,margin),minimum_margin)
            actions=actions.at[tick].set(command);active_mask=active_mask.at[tick].set(active)
            phase_trace=phase_trace.at[tick].set(next_state.info["phase"])
            return ((next_state,alive,handed,survival,entry_tick,termination_tick,end_code,final,
                     minimum_distance,minimum_margin,entry_qpos,entry_qvel,actions,active_mask,phase_trace),None)

        initial=(state,active,handed,survival,entry_tick,termination_tick,end_code,final,
                 minimum_distance,minimum_margin,entry_qpos,entry_qvel,actions,active_mask,phase_trace)
        final_carry,_=jax.lax.scan(body,initial,jnp.arange(horizon))
        (_,_,handed,survival,entry_tick,termination_tick,end_code,final,minimum_distance,
         minimum_margin,entry_qpos,entry_qvel,actions,active_mask,phase_trace)=final_carry
        return {"downstream_entry":handed,"final_recovery":final,"survival":survival,
                "entry_tick":entry_tick,"termination_tick":termination_tick,"end_code":end_code,
                "minimum_distance":minimum_distance,"minimum_margin":minimum_margin,
                "entry_qpos":entry_qpos,"entry_qvel":entry_qvel,
                "actions":actions,"active_mask":active_mask,"phase_trace":phase_trace}
    return jax.jit(rollout)


def backward_lexicographic_order(result: dict[str,np.ndarray]) -> np.ndarray:
    return np.lexsort((-result["survival"],-result["minimum_margin"],result["minimum_distance"],
                       -result["downstream_entry"].astype(np.int32),-result["final_recovery"].astype(np.int32)))


def bounded_cem(rollout: Callable, state_factory: Callable[[int],Any], *, seed: int,
                samples: int, iterations: int, knot_count: int, bound: float = .2,
                warm_start: np.ndarray | None = None) -> tuple[np.ndarray,dict,list[dict]]:
    """Fixed-budget CEM whose ordering is Final -> downstream entry -> distance."""
    rng=np.random.default_rng(seed)
    mean=np.zeros((knot_count,4),np.float32) if warm_start is None else np.asarray(warm_start,np.float32).reshape(knot_count,4).copy()
    std=np.full_like(mean,bound*.5);elite_count=max(4,samples//8);history=[];best=None
    for generation in range(iterations):
        knots=np.clip(rng.normal(mean,std,size=(samples,knot_count,4)),-bound,bound).astype(np.float32)
        if generation==0 and warm_start is not None: knots[0]=mean
        raw=jax.device_get(rollout(state_factory(samples),jnp.asarray(knots),jax.random.PRNGKey(seed+generation)))
        order=backward_lexicographic_order(raw);elite=knots[order[:elite_count]];mean=elite.mean(0);std=np.maximum(elite.std(0),bound*.02)
        for index in order[:min(8,len(order))]:
            history.append({"generation":generation,"sample":int(index),"final_recovery":bool(raw["final_recovery"][index]),"downstream_entry":bool(raw["downstream_entry"][index]),"minimum_distance":float(raw["minimum_distance"][index]),"minimum_margin":float(raw["minimum_margin"][index]),"survival":int(raw["survival"][index]),"end_code":int(raw["end_code"][index])})
        index=int(order[0]);score=(not bool(raw["final_recovery"][index]),not bool(raw["downstream_entry"][index]),float(raw["minimum_distance"][index]),-float(raw["minimum_margin"][index]),-int(raw["survival"][index]))
        if best is None or score<best[0]:
            best=(score,knots[index].copy(),{k:np.asarray(v[index] if k not in {"actions","active_mask","phase_trace"} else v[:,index]) for k,v in raw.items()})
    assert best is not None
    summary={k:(v.tolist() if np.asarray(v).ndim else np.asarray(v).item()) for k,v in best[2].items()}
    return best[1],summary,history


def active_prefix_exact(first: dict,second: dict,index: int=0) -> tuple[bool,list[str]]:
    tick=min(int(np.asarray(first["termination_tick"])[index]),int(np.asarray(second["termination_tick"])[index]));failed=[]
    for key in ("downstream_entry","final_recovery","survival","entry_tick","termination_tick","end_code","minimum_distance","minimum_margin","entry_qpos","entry_qvel"):
        if not np.array_equal(np.asarray(first[key])[index],np.asarray(second[key])[index]):failed.append(key)
    for key in ("actions","active_mask","phase_trace"):
        if not np.array_equal(np.asarray(first[key])[:tick,index],np.asarray(second[key])[:tick,index]):failed.append(key)
    return not failed,failed
