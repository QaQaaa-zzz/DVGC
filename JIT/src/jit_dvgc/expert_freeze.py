"""Freeze identity-bound phase experts without copying or retraining them."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .checkpoint import CheckpointIdentity, load_checkpoint
from .config import load_config
from .constants import ACTION_ORDER, ACTOR_FRAME_FIELDS, ACTOR_TASK_FIELDS
from .handoff_bank import pytree_sha256


FROZEN_EXPERT_SCHEMA = "jit_frozen_phase_experts_v1"


@dataclass(frozen=True)
class FrozenExpertRecord:
    name: str
    phase: str
    checkpoint: str
    config: str
    config_sha256: str
    xml_sha256: str
    training_transitions: int
    payload_sha256: str
    normalizer_sha256: str
    actor_sha256: str
    critic_sha256: str
    actor_frame_fields: tuple[str, ...]
    actor_task_fields: tuple[str, ...]
    action_order: tuple[str, ...]


def checkpoint_identity_from_config(config: Any) -> CheckpointIdentity:
    return CheckpointIdentity(
        config_sha256=config.config_sha256,
        xml_sha256=str(config.model["xml_sha256"]),
        actor_frame_fields=ACTOR_FRAME_FIELDS,
        actor_task_fields=ACTOR_TASK_FIELDS,
        action_order=ACTION_ORDER,
    )


def _read_payload_sha256(checkpoint: Path) -> str:
    sidecar = json.loads((Path(checkpoint) / "identity.json").read_text(encoding="utf-8"))
    digest = str(sidecar.get("payload_sha256", ""))
    if len(digest) != 64:
        raise ValueError("checkpoint identity.json is missing payload_sha256")
    return digest


def inspect_expert(
    *, name: str, expected_phase: str, config_path: Path, checkpoint: Path
) -> FrozenExpertRecord:
    config_path = Path(config_path)
    checkpoint = Path(checkpoint)
    config = load_config(config_path)
    if config.phase != expected_phase:
        raise ValueError(
            f"{name} config phase mismatch: expected {expected_phase}, got {config.phase}"
        )
    identity = checkpoint_identity_from_config(config)
    payload = load_checkpoint(checkpoint, expected=identity)
    return FrozenExpertRecord(
        name=name,
        phase=config.phase,
        checkpoint=str(checkpoint),
        config=str(config_path),
        config_sha256=config.config_sha256,
        xml_sha256=identity.xml_sha256,
        training_transitions=int(payload.training_transitions),
        payload_sha256=_read_payload_sha256(checkpoint),
        normalizer_sha256=pytree_sha256(payload.observation_normalizer),
        actor_sha256=pytree_sha256(payload.actor_params),
        critic_sha256=pytree_sha256(payload.critic_params),
        actor_frame_fields=ACTOR_FRAME_FIELDS,
        actor_task_fields=ACTOR_TASK_FIELDS,
        action_order=ACTION_ORDER,
    )


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_frozen_record(record: Mapping[str, Any]) -> tuple[Any, Any]:
    """Strictly reload a frozen record and prove its checkpoint/hash binding."""
    config = load_config(Path(record["config"]))
    if config.phase != record["phase"]:
        raise ValueError("frozen expert phase drift")
    if config.config_sha256 != record["config_sha256"]:
        raise ValueError("frozen expert config hash drift")
    expected = checkpoint_identity_from_config(config)
    if expected.xml_sha256 != record["xml_sha256"]:
        raise ValueError("frozen expert XML hash drift")
    payload = load_checkpoint(Path(record["checkpoint"]), expected=expected)
    if int(payload.training_transitions) != int(record["training_transitions"]):
        raise ValueError("frozen expert transition drift")
    if _read_payload_sha256(Path(record["checkpoint"])) != record["payload_sha256"]:
        raise ValueError("frozen expert payload drift")
    if pytree_sha256(payload.observation_normalizer) != record["normalizer_sha256"]:
        raise ValueError("frozen expert normalizer drift")
    if pytree_sha256(payload.actor_params) != record["actor_sha256"]:
        raise ValueError("frozen expert actor drift")
    if pytree_sha256(payload.critic_params) != record["critic_sha256"]:
        raise ValueError("frozen expert critic drift")
    if tuple(record["action_order"]) != ACTION_ORDER:
        raise ValueError("frozen expert action-order drift")
    return config, payload


def freeze_phase_experts(
    output_dir: Path,
    *,
    pi_up_config: Path,
    pi_up_checkpoint: Path,
    pi_down_config: Path,
    pi_down_checkpoint: Path,
) -> dict[str, Any]:
    """Create a provenance manifest that freezes the selected expert parameters."""
    up = inspect_expert(
        name="pi_up_star",
        expected_phase="propulsion_ascent",
        config_path=pi_up_config,
        checkpoint=pi_up_checkpoint,
    )
    down = inspect_expert(
        name="pi_down_star",
        expected_phase="descent_recovery",
        config_path=pi_down_config,
        checkpoint=pi_down_checkpoint,
    )
    if up.xml_sha256 != down.xml_sha256:
        raise ValueError("phase experts must use the same XML")
    if up.action_order != down.action_order:
        raise ValueError("phase experts must use the same action order")

    experts = {
        "pi_up_star": asdict(up),
        "pi_down_star": asdict(down),
    }
    for value in experts.values():
        value["actor_frame_fields"] = list(value["actor_frame_fields"])
        value["actor_task_fields"] = list(value["actor_task_fields"])
        value["action_order"] = list(value["action_order"])
    protocol = {
        "schema": FROZEN_EXPERT_SCHEMA,
        "status": "frozen",
        "immutable_parameters": True,
        "copied_checkpoints": False,
        "experts": experts,
    }
    manifest = {
        **protocol,
        "freeze_protocol_sha256": _canonical_sha256(protocol),
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "frozen_experts.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def load_frozen_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != FROZEN_EXPERT_SCHEMA or payload.get("status") != "frozen":
        raise ValueError("not a frozen phase-expert manifest")
    protocol = {key: value for key, value in payload.items() if key != "freeze_protocol_sha256"}
    if _canonical_sha256(protocol) != payload.get("freeze_protocol_sha256"):
        raise ValueError("frozen expert manifest hash mismatch")
    if set(payload.get("experts", {})) != {"pi_up_star", "pi_down_star"}:
        raise ValueError("frozen expert manifest must contain pi_up_star and pi_down_star")
    for record in payload["experts"].values():
        verify_frozen_record(record)
    return payload
