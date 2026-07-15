"""Immutable stage-expert ownership and controller-stack provenance."""
from __future__ import annotations

from dataclasses import asdict,dataclass
import hashlib,json
from pathlib import Path
from typing import Any,Mapping

from .config import POLICY_NETWORK_VERSION,SNAPSHOT_SCHEMA,file_sha256
from .policy import load_bundle

ACTION_ORDER=("steer","rear_wheel_drive","hip","knee")
REGISTRY_VERSION=1


def _digest(value: Any) -> str:
    raw=json.dumps(value,sort_keys=True,separators=(",",":"),default=str).encode()
    return hashlib.sha256(raw).hexdigest()


def policy_bundle_hash(path: str|Path) -> str:
    root=Path(path)
    return _digest({name:file_sha256(root/name) for name in ("params.pkl","config.json","manifest.json")})


def observation_schema_hash(config: Mapping[str,Any]) -> str:
    return _digest({"policy_network_version":POLICY_NETWORK_VERSION,"actor_key":"state","actor_frame_dim":35,"actor_history_steps":int(config["actor_history_steps"]),"actor_shape":[35*int(config["actor_history_steps"])],"critic_key":"privileged_state","critic_shape":[29],"oracle_actor_fields":[]})


def action_schema_hash(config: Mapping[str,Any]) -> str:
    return _digest({"action_mapping_version":config["action_mapping_version"],"order":ACTION_ORDER,"bounds":[-1.0,1.0]})


def policy_state_schema_hash(config: Mapping[str,Any]) -> str:
    return _digest({"snapshot_schema":SNAPSHOT_SCHEMA,"recurrent":"stateless_mlp","obs_history_shape":[max(0,int(config["actor_history_steps"])-1),35],"fields":["last_action","obs_history","actor_observation","filter_phase","phase_probs","prev_acc_z","prev_vz"]})


@dataclass(frozen=True)
class ExpertSpec:
    policy_id: str
    policy_hash: str
    bundle_hash: str
    stage: str
    checkpoint_path: str
    observation_schema_hash: str
    action_schema_hash: str
    policy_state_schema_hash: str
    recurrent_schema: str
    xml_sha256: str
    candidate_bank_sha256: str|None
    downstream_entry_set_sha256: str|None
    downstream_controller_stack_hash: str
    controller_stack_hash: str


class StageExpertRegistry:
    def __init__(self,specs: Mapping[str,ExpertSpec],*,runtime_source_fingerprint: str):
        self.specs=dict(specs); self.runtime_source_fingerprint=str(runtime_source_fingerprint)
        self.registry_hash=_digest({"version":REGISTRY_VERSION,"runtime":self.runtime_source_fingerprint,"specs":{k:asdict(v) for k,v in sorted(self.specs.items())}})

    @classmethod
    def build(cls,policies: Mapping[str,str|Path],entry_sets: Mapping[str,str|Path],*,runtime_source_fingerprint: str) -> "StageExpertRegistry":
        specs={}; downstream_hash=_digest({"stack":"terminal"})
        for stage in ("landing","flight","takeoff","approach"):
            if stage not in policies: continue
            params,cfg,manifest=load_bundle(policies[stage],verify_files=True); del params
            declared_stage=manifest.get("stage") or manifest.get("training_stage") or cfg.get("training_stage")
            if declared_stage!=stage: raise ValueError(f"Expert {stage} bundle declares stage {declared_stage}")
            entry_path=entry_sets.get(stage); entry_hash=file_sha256(entry_path) if entry_path else None
            if stage!="landing" and not entry_hash: raise ValueError(f"Expert {stage} requires a downstream entry set")
            policy_hash=file_sha256(Path(policies[stage])/"params.pkl")
            stack_hash=_digest({"stage":stage,"policy_hash":policy_hash,"entry_set":entry_hash,"downstream_stack":downstream_hash})
            spec=ExpertSpec(policy_id=str(manifest["policy_version"]),policy_hash=policy_hash,bundle_hash=policy_bundle_hash(policies[stage]),stage=stage,checkpoint_path=str(Path(policies[stage]).resolve()),observation_schema_hash=observation_schema_hash(cfg),action_schema_hash=action_schema_hash(cfg),policy_state_schema_hash=policy_state_schema_hash(cfg),recurrent_schema="stateless_mlp",xml_sha256=str(manifest["xml_sha256"]),candidate_bank_sha256=manifest.get("candidate_bank_sha256"),downstream_entry_set_sha256=entry_hash,downstream_controller_stack_hash=downstream_hash,controller_stack_hash=stack_hash)
            specs[stage]=spec; downstream_hash=stack_hash
        if not specs: raise ValueError("Expert registry is empty")
        schemas={(s.observation_schema_hash,s.action_schema_hash,s.policy_state_schema_hash,s.xml_sha256) for s in specs.values()}
        if len(schemas)!=1: raise ValueError("Stage experts have incompatible XML or observation/action/PolicyState schemas")
        return cls(specs,runtime_source_fingerprint=runtime_source_fingerprint)

    def validate_files(self) -> None:
        for spec in self.specs.values():
            if file_sha256(Path(spec.checkpoint_path)/"params.pkl")!=spec.policy_hash: raise ValueError(f"Expert policy changed: {spec.stage}")
            if policy_bundle_hash(spec.checkpoint_path)!=spec.bundle_hash: raise ValueError(f"Expert bundle changed: {spec.stage}")

    def to_dict(self) -> dict[str,Any]:
        return {"registry_version":REGISTRY_VERSION,"registry_hash":self.registry_hash,"runtime_source_fingerprint":self.runtime_source_fingerprint,"experts":{k:asdict(v) for k,v in sorted(self.specs.items())}}

    def save(self,path: str|Path) -> None:
        target=Path(path)
        if target.exists(): raise FileExistsError(f"Registry output exists: {target}")
        target.parent.mkdir(parents=True,exist_ok=True); target.write_text(json.dumps(self.to_dict(),indent=2,sort_keys=True))

    @classmethod
    def load(cls,path: str|Path) -> "StageExpertRegistry":
        payload=json.loads(Path(path).read_text()); specs={k:ExpertSpec(**v) for k,v in payload["experts"].items()}; obj=cls(specs,runtime_source_fingerprint=payload["runtime_source_fingerprint"])
        if obj.registry_hash!=payload["registry_hash"]: raise ValueError("Expert registry hash mismatch")
        obj.validate_files(); return obj
