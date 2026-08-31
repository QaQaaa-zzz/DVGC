"""Freeze one completed unified checkpoint as an envelope-iteration authority.

This module creates an identity/provenance manifest only.  It does not copy,
retrain, evaluate, or promote the policy to ``pi_unified_star``.  The frozen
record is the policy authority under which boundary candidates and continuation
labels for one envelope iteration must be generated.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .checkpoint import CheckpointIdentity, load_checkpoint
from .config import file_sha256, load_config
from .constants import ACTION_ORDER, ACTOR_FRAME_FIELDS, ACTOR_TASK_FIELDS
from .handoff_bank import pytree_sha256
from .unified_formal import load_unified_formal_config


FROZEN_UNIFIED_POLICY_SCHEMA = "jit_frozen_unified_policy_v1"


@dataclass(frozen=True)
class FrozenUnifiedPolicyRecord:
    name: str
    iteration: int
    policy_role: str
    checkpoint: str
    formal_config: str
    formal_config_sha256: str
    xml_sha256: str
    source_training_run_id: str
    source_training_transitions: int
    source_reset_mixture: Mapping[str, Any]
    payload_sha256: str
    normalizer_sha256: str
    actor_sha256: str
    critic_sha256: str
    actor_frame_fields: tuple[str, ...]
    actor_task_fields: tuple[str, ...]
    action_order: tuple[str, ...]


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_run_id(config: Any) -> str:
    declaration = config.raw.get("run_declaration")
    if not isinstance(declaration, Mapping):
        raise ValueError("unified config is missing run_declaration")
    run_id = declaration.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("unified config run_declaration is missing run_id")
    return run_id


def _checkpoint_identity(config: Any) -> CheckpointIdentity:
    up_config = load_config(Path(config.up_config_path))
    down_config = load_config(Path(config.down_config_path))
    if up_config.config_sha256 != config.up_config_sha256:
        raise ValueError("unified freeze upstream config hash drift")
    if down_config.config_sha256 != config.down_config_sha256:
        raise ValueError("unified freeze downstream config hash drift")
    up_xml = str(up_config.model["xml_sha256"])
    down_xml = str(down_config.model["xml_sha256"])
    if up_xml != down_xml:
        raise ValueError("unified freeze phase XML mismatch")
    return CheckpointIdentity(
        config_sha256=config.config_sha256,
        xml_sha256=up_xml,
        actor_frame_fields=ACTOR_FRAME_FIELDS,
        actor_task_fields=ACTOR_TASK_FIELDS,
        action_order=ACTION_ORDER,
    )


def _validate_formal_report(report: Mapping[str, Any], config: Any) -> None:
    if report.get("schema") != "jit_pi_unified_formal_report_v1":
        raise ValueError("unified freeze requires a formal unified report")
    if report.get("status") != "completed":
        raise ValueError("unified source run is not completed")
    target = int(config.ppo.requested_transitions)
    if int(report.get("requested_training_transitions", -1)) != target:
        raise ValueError("unified formal requested-transition drift")
    if int(report.get("completed_training_transitions", -1)) != target:
        raise ValueError("unified formal run did not complete the requested target")
    if report.get("checkpoint_restored") is not True:
        raise ValueError("unified formal final checkpoint was not restore-verified")
    if report.get("reset_mixture") != config.reset_mixture.as_dict():
        raise ValueError("unified formal reset-mixture drift")
    if report.get("expert_switching_used") is not False:
        raise ValueError("unified formal source unexpectedly used expert switching")
    if report.get("validation_data_used") is not False:
        raise ValueError("unified formal source used validation data")
    if report.get("test_data_used") is not False:
        raise ValueError("unified formal source used TEST data")
    checkpoints = tuple(int(x) for x in report.get("checkpoint_transitions", ()))
    if target not in checkpoints:
        raise ValueError("unified formal report does not contain the final checkpoint")


def inspect_unified_policy(
    *,
    config_path: Path,
    checkpoint: Path,
    iteration: int,
    formal_report: Path | None = None,
) -> FrozenUnifiedPolicyRecord:
    """Verify a completed unified checkpoint and return its immutable record."""
    if int(iteration) < 0:
        raise ValueError("unified envelope iteration must be nonnegative")
    config_path = Path(config_path)
    checkpoint = Path(checkpoint)
    config = load_unified_formal_config(config_path)
    source_run_id = _source_run_id(config)

    if checkpoint.name != f"transition_{config.ppo.requested_transitions}":
        raise ValueError("unified freeze requires the exact completed final checkpoint")
    run_dir = checkpoint.parent.parent
    if run_dir.name != source_run_id:
        raise ValueError("unified checkpoint run directory does not match config run_id")

    report_path = (
        Path(formal_report) if formal_report is not None else run_dir / "formal_report.json"
    )
    report = _read_json(report_path)
    _validate_formal_report(report, config)

    identity = _checkpoint_identity(config)
    payload = load_checkpoint(checkpoint, expected=identity)
    target = int(config.ppo.requested_transitions)
    if int(payload.training_transitions) != target:
        raise ValueError("unified checkpoint training-transition drift")

    sidecar = _read_json(checkpoint / "identity.json")
    declared_payload_sha = str(sidecar.get("payload_sha256", ""))
    actual_payload_sha = file_sha256(checkpoint / "payload.pkl")
    if declared_payload_sha != actual_payload_sha:
        raise ValueError("unified checkpoint payload identity drift")

    return FrozenUnifiedPolicyRecord(
        name=f"pi_{int(iteration)}",
        iteration=int(iteration),
        policy_role="envelope_expansion_authority",
        checkpoint=str(checkpoint),
        formal_config=str(config_path),
        formal_config_sha256=config.config_sha256,
        xml_sha256=identity.xml_sha256,
        source_training_run_id=source_run_id,
        source_training_transitions=target,
        source_reset_mixture=config.reset_mixture.as_dict(),
        payload_sha256=actual_payload_sha,
        normalizer_sha256=pytree_sha256(payload.observation_normalizer),
        actor_sha256=pytree_sha256(payload.actor_params),
        critic_sha256=pytree_sha256(payload.critic_params),
        actor_frame_fields=ACTOR_FRAME_FIELDS,
        actor_task_fields=ACTOR_TASK_FIELDS,
        action_order=ACTION_ORDER,
    )


def _record_payload(record: FrozenUnifiedPolicyRecord) -> dict[str, Any]:
    payload = asdict(record)
    payload["source_reset_mixture"] = dict(record.source_reset_mixture)
    payload["actor_frame_fields"] = list(record.actor_frame_fields)
    payload["actor_task_fields"] = list(record.actor_task_fields)
    payload["action_order"] = list(record.action_order)
    return payload


def verify_frozen_unified_record(record: Mapping[str, Any]) -> FrozenUnifiedPolicyRecord:
    """Strictly reload a frozen record and prove checkpoint/config/hash binding."""
    inspected = inspect_unified_policy(
        config_path=Path(record["formal_config"]),
        checkpoint=Path(record["checkpoint"]),
        iteration=int(record["iteration"]),
    )
    expected = _record_payload(inspected)
    if dict(record) != expected:
        differing = sorted(
            key for key in set(record) | set(expected) if record.get(key) != expected.get(key)
        )
        raise ValueError(f"frozen unified policy record drift: {differing}")
    return inspected


def freeze_unified_policy(
    output_dir: Path,
    *,
    config_path: Path,
    checkpoint: Path,
    iteration: int,
    formal_report: Path | None = None,
) -> dict[str, Any]:
    """Freeze one completed unified checkpoint without copying or evaluating it."""
    record = inspect_unified_policy(
        config_path=config_path,
        checkpoint=checkpoint,
        iteration=iteration,
        formal_report=formal_report,
    )
    protocol = {
        "schema": FROZEN_UNIFIED_POLICY_SCHEMA,
        "status": "frozen",
        "immutable_parameters": True,
        "copied_checkpoint": False,
        "training_transitions": 0,
        "environment_interactions": 0,
        "expert_switching_used": False,
        "policy": _record_payload(record),
        "claim_boundary": {
            "envelope_expansion_authority": True,
            "pi_unified_star_claim": False,
            "jce_jel_claim": False,
            "certified_safe_tube_claim": False,
        },
    }
    manifest = {**protocol, "freeze_protocol_sha256": _canonical_sha256(protocol)}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "frozen_unified_policy.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_frozen_unified_manifest(path: Path) -> dict[str, Any]:
    payload = _read_json(Path(path))
    if (
        payload.get("schema") != FROZEN_UNIFIED_POLICY_SCHEMA
        or payload.get("status") != "frozen"
    ):
        raise ValueError("not a frozen unified-policy manifest")
    protocol = {
        key: value for key, value in payload.items() if key != "freeze_protocol_sha256"
    }
    if _canonical_sha256(protocol) != payload.get("freeze_protocol_sha256"):
        raise ValueError("frozen unified-policy manifest hash mismatch")
    if payload.get("immutable_parameters") is not True:
        raise ValueError("frozen unified policy is not immutable")
    if payload.get("copied_checkpoint") is not False:
        raise ValueError("frozen unified policy unexpectedly copied its checkpoint")
    if payload.get("training_transitions") != 0 or payload.get("environment_interactions") != 0:
        raise ValueError("freezing a unified policy must use zero interactions")
    claims = payload.get("claim_boundary", {})
    if claims != {
        "envelope_expansion_authority": True,
        "pi_unified_star_claim": False,
        "jce_jel_claim": False,
        "certified_safe_tube_claim": False,
    }:
        raise ValueError("frozen unified-policy claim boundary drift")
    verify_frozen_unified_record(payload["policy"])
    return payload
