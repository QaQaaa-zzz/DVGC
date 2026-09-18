"""Identity-checked Actor-only initialization for Phase U retraining."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .checkpoint import CheckpointIdentity, load_checkpoint
from .config import file_sha256
from .handoff_bank import pytree_sha256
from .phase_expert_init import ActorOnlyInitialization


_SCHEMA = "jit_frozen_development_checkpoint_v1"
_CLAIM_BOUNDARY = {
    "envelope_expansion_authority": False,
    "pi_unified_star_claim": False,
    "jce_jel_claim": False,
    "certified_safe_tube_claim": False,
}


def _read_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("frozen policy manifest must be a JSON object")
    return manifest


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _manifest_checkpoint(manifest_path: Path, declared: Any) -> Path:
    if not isinstance(declared, str) or not declared:
        raise ValueError("frozen policy checkpoint is missing")
    checkpoint = Path(declared)
    if not checkpoint.is_absolute():
        checkpoint = manifest_path.parent / checkpoint
    return checkpoint.resolve()


def load_phase_u_actor_initialization(
    frozen_policy: Path,
) -> ActorOnlyInitialization:
    """Restore only a frozen development checkpoint's normalizer and Actor."""

    manifest_path = Path(frozen_policy).resolve()
    manifest = _read_manifest(manifest_path)
    if manifest.get("schema") != _SCHEMA:
        raise ValueError("Phase U warm start requires a frozen development checkpoint")
    if manifest.get("status") != "frozen":
        raise ValueError("Phase U warm-start manifest is not frozen")
    if manifest.get("immutable_parameters") is not True:
        raise ValueError("Phase U warm-start manifest is not immutable")
    if manifest.get("copied_checkpoint") is not False:
        raise ValueError("development checkpoint must remain referenced, not copied")
    if manifest.get("training_transitions") != 0:
        raise ValueError("freezing must not add training transitions")
    if manifest.get("environment_interactions") != 0:
        raise ValueError("freezing must not add environment interactions")
    if manifest.get("expert_switching_used") is not False:
        raise ValueError("development checkpoint cannot use expert switching")
    if manifest.get("claim_boundary") != _CLAIM_BOUNDARY:
        raise ValueError("frozen development checkpoint claim boundary drift")

    protocol = {
        key: value
        for key, value in manifest.items()
        if key != "freeze_protocol_sha256"
    }
    if _canonical_sha256(protocol) != manifest.get("freeze_protocol_sha256"):
        raise ValueError("frozen development checkpoint manifest hash mismatch")

    policy = manifest.get("policy")
    if not isinstance(policy, Mapping):
        raise ValueError("frozen development checkpoint policy is missing")
    if policy.get("policy_role") != "development_checkpoint":
        raise ValueError("frozen development checkpoint policy role drift")
    if policy.get("data_role") != "train":
        raise ValueError("frozen development checkpoint must have TRAIN data role")

    expected = CheckpointIdentity(
        config_sha256=str(policy.get("formal_config_sha256", "")),
        xml_sha256=str(policy.get("xml_sha256", "")),
        actor_frame_fields=tuple(policy.get("actor_frame_fields", ())),
        actor_task_fields=tuple(policy.get("actor_task_fields", ())),
        action_order=tuple(policy.get("action_order", ())),
    )
    checkpoint = _manifest_checkpoint(manifest_path, policy.get("checkpoint"))
    payload = load_checkpoint(checkpoint, expected=expected)

    payload_sha256 = file_sha256(checkpoint / "payload.pkl")
    if payload_sha256 != policy.get("payload_sha256"):
        raise ValueError("frozen development checkpoint payload hash drift")
    if file_sha256(checkpoint / "identity.json") != policy.get(
        "checkpoint_identity_sha256"
    ):
        raise ValueError("frozen development checkpoint identity hash drift")
    if payload.training_transitions != policy.get("source_training_transitions"):
        raise ValueError("frozen development checkpoint transition drift")

    normalizer_sha256 = pytree_sha256(payload.observation_normalizer)
    if normalizer_sha256 != policy.get("normalizer_sha256"):
        raise ValueError("frozen development checkpoint normalizer hash drift")
    actor_sha256 = pytree_sha256(payload.actor_params)
    if actor_sha256 != policy.get("actor_sha256"):
        raise ValueError("frozen development checkpoint actor hash drift")
    critic_sha256 = pytree_sha256(payload.critic_params)
    if critic_sha256 != policy.get("critic_sha256"):
        raise ValueError("frozen development checkpoint critic hash drift")

    return ActorOnlyInitialization(
        observation_normalizer=payload.observation_normalizer,
        actor_params=payload.actor_params,
        parent_transition=int(payload.training_transitions),
        payload_sha256=payload_sha256,
        actor_sha256=actor_sha256,
        provenance={
            "actor_initialized": True,
            "normalizer_initialized": True,
            "critic_fresh": True,
            "optimizer_fresh": True,
            "source_frozen_policy": str(manifest_path),
            "source_frozen_policy_sha256": file_sha256(manifest_path),
            "source_checkpoint": str(checkpoint),
            "source_checkpoint_identity_sha256": str(
                policy["checkpoint_identity_sha256"]
            ),
            "source_payload_sha256": payload_sha256,
            "source_normalizer_sha256": normalizer_sha256,
            "source_actor_sha256": actor_sha256,
            "source_critic_sha256": critic_sha256,
            "source_xml_sha256": str(policy["xml_sha256"]),
            "source_actor_frame_fields": list(policy["actor_frame_fields"]),
            "source_actor_task_fields": list(policy["actor_task_fields"]),
            "source_action_order": list(policy["action_order"]),
            "source_formal_config_sha256": str(policy["formal_config_sha256"]),
            "source_formal_config_file_sha256": str(
                policy["formal_config_file_sha256"]
            ),
            "source_formal_report_sha256": str(
                policy["source_formal_report_sha256"]
            ),
        },
    )
