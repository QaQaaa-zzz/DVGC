"""Irreversible stage-expert composition without physical-state resets."""
from __future__ import annotations

from dataclasses import dataclass,field
from typing import Any,Callable,Mapping,Sequence
import jax,numpy as np

from .bank import SnapshotBank
from .config import STAGE_ID,file_sha256
from .entry import ENTRY_FEATURE_NAMES


class CanonicalEntryMatcher:
    def __init__(self,env: Any,source_stage: str,bank_path: str):
        self.env=env; self.source_stage=source_stage; self.bank_path=str(bank_path); self.bank_sha256=file_sha256(bank_path); bank=SnapshotBank.load(bank_path)
        next_stage={"approach":"takeoff","takeoff":"flight","flight":"landing"}[source_stage]
        safe=bank.records_for_phase(next_stage,final_labels=["safe"],include_training_only=False)
        if not safe: raise ValueError(f"Entry set has no Final-safe {next_stage} states")
        matcher=bank.metadata.get("entry_matcher") if source_stage=="flight" else None
        features=np.asarray([r["entry_feature"] if matcher else r["physical_feature"] for r in safe],np.float32)
        if matcher:
            if tuple(matcher["feature_names"])!=ENTRY_FEATURE_NAMES: raise ValueError("Landing entry feature schema mismatch")
            self.center=np.asarray(matcher["center"],np.float32); self.scale=np.asarray(matcher["scale"],np.float32); self.radius=float(matcher["radius"])
        else:
            self.center=np.median(features,axis=0); mad=1.4826*np.median(np.abs(features-self.center),axis=0); std=features.std(axis=0); self.scale=np.maximum(np.maximum(mad,.25*std),1e-4); self.radius=float(env._config.tube_match_radius_z)
        self.features=(features-self.center)/self.scale

    def match(self,state: Any) -> tuple[bool,float]:
        if self.source_stage=="flight":
            feature=np.asarray(jax.device_get(self.env._landing_entry_feature(state.data,state.info["had_valid_landing"]>0,state.info["contact_age"]>0,state.info["landing_entry_age"])),np.float32)
            age=int(np.asarray(jax.device_get(state.info["landing_entry_age"]))); eligible=1<=age<=int(self.env._config.landing_entry_window_steps)
        else:
            feature=np.asarray(jax.device_get(self.env._physical_feature(state.data)),np.float32)
            phase=int(np.asarray(jax.device_get(state.info["phase"])))
            eligible=(phase==STAGE_ID["flight"] and bool(np.asarray(jax.device_get(state.info["had_airborne"])))) if self.source_stage=="takeoff" else bool(np.asarray(jax.device_get(state.metrics["event/takeoff"])))
        distance=float(np.min(np.linalg.norm(self.features-(feature-self.center)[None,:]/self.scale[None,:],axis=1)))
        return bool(eligible and distance<=self.radius),distance


@dataclass
class CompositeSession:
    env: Any
    stages: Sequence[str]
    inference: Mapping[str,Callable]
    matchers: Mapping[str,Any]
    state: Any
    rng: Any
    active_index: int=0
    handoffs: list[dict[str,Any]]=field(default_factory=list)

    @property
    def active_stage(self) -> str: return self.stages[self.active_index]

    def step(self,*,step_fn: Callable|None=None,action_noise_std: float=0.0) -> Any:
        stage=self.active_stage; self.rng,ak,nk=jax.random.split(self.rng,3); action,_=self.inference[stage](self.state.obs,ak)
        if action_noise_std: action=np.clip(np.asarray(action)+np.asarray(jax.random.normal(nk,action.shape))*action_noise_std,-1,1)
        self.state=(jax.jit(self.env.step) if step_fn is None else step_fn)(self.state,action)
        if self.active_index<len(self.stages)-1:
            matched,distance=self.matchers[stage].match(self.state)
            if matched:
                old=self.active_index; self.active_index+=1
                self.handoffs.append({"from":self.stages[old],"to":self.active_stage,"step":int(np.asarray(jax.device_get(self.state.info["episode_step"]))),"distance":distance})
        return self.state


def composite_rollout(env: Any,stages: Sequence[str],inference: Mapping[str,Callable],matchers: Mapping[str,Any],state: Any,rng: Any,*,horizon: int,step_fn: Callable|None=None,action_noise_std: float=0.0):
    session=CompositeSession(env,tuple(stages),inference,matchers,state,rng); trace=[]
    for _ in range(int(horizon)):
        trace.append(session.step(step_fn=step_fn,action_noise_std=action_noise_std))
        if bool(np.asarray(jax.device_get(session.state.done))): break
    final=bool(np.asarray(jax.device_get(session.state.info.get("recovery_success",0)))); chain=bool(session.handoffs)
    return session,{"chain":chain,"final":final,"chain_missed_final":bool(final and not chain),"terminated":bool(np.asarray(jax.device_get(session.state.info.get("terminated",0)))),"truncated":bool(np.asarray(jax.device_get(session.state.info.get("truncated",0)))),"end_code":int(np.asarray(jax.device_get(session.state.info.get("end_code",0)))),"steps":len(trace),"handoffs":session.handoffs,"active_stage":session.active_stage}
