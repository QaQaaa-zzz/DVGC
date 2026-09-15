"""Freeze one completed unified checkpoint as an envelope-iteration authority.

Ordinary freezing creates an identity/provenance manifest only. The explicit
initialization-only path copies unchanged parameters into a new runtime identity.
Neither path retrains, evaluates, or promotes the policy to ``pi_unified_star``. The frozen
record is the policy authority under which boundary candidates and continuation
labels for one envelope iteration must be generated.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

from .checkpoint import CheckpointIdentity, load_checkpoint, save_checkpoint
from .config import file_sha256, load_config
from .constants import ACTION_ORDER, ACTOR_FRAME_FIELDS, ACTOR_TASK_FIELDS
from .handoff_bank import pytree_sha256
from .unified_formal import (
    load_unified_actor_warm_start_config,
    load_unified_formal_config,
)


FROZEN_UNIFIED_POLICY_SCHEMA = "jit_frozen_unified_policy_v1"
FROZEN_DEVELOPMENT_CHECKPOINT_SCHEMA = "jit_frozen_development_checkpoint_v1"
FROZEN_INITIALIZATION_POLICY_SCHEMA = "jit_frozen_initialization_policy_v1"


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


@dataclass(frozen=True)
class FrozenDevelopmentCheckpointRecord(FrozenUnifiedPolicyRecord):
    """TRAIN diagnostic identity; iteration is source metadata, not authority."""

    data_role: str
    formal_config_file_sha256: str
    source_formal_report: str
    source_formal_report_sha256: str
    checkpoint_identity_sha256: str
    source_requested_training_transitions: int


@dataclass(frozen=True)
class FrozenInitializationPolicyRecord(FrozenUnifiedPolicyRecord):
    """Unchanged source parameters bound to a separately declared endpoint."""

    data_role: str
    formal_config_file_sha256: str
    checkpoint_identity_sha256: str
    source_frozen_policy: str
    source_frozen_policy_sha256: str
    initialization_receipt: str
    initialization_receipt_sha256: str


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _load_policy_formal_config(path: Path):
    raw = _read_json(Path(path))
    if raw.get("initialization", {}).get("actor") == "warm_start_frozen_unified":
        return load_unified_actor_warm_start_config(Path(path))
    return load_unified_formal_config(Path(path))


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
    up_config = load_config(Path(config.up_config_path), runtime_only=True) if config.raw.get("historical_up_runtime") else load_config(Path(config.up_config_path))
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
    config = _load_policy_formal_config(config_path)
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


def inspect_development_checkpoint(
    *, config_path: Path, checkpoint: Path, name: str, allow_initialization: bool = False,
) -> FrozenDevelopmentCheckpointRecord:
    """Verify a declared positive milestone from a completed run for TRAIN use.

    The original training budget/config remains intact. This is deliberately
    separate from final-checkpoint envelope authority inspection.
    """
    if (not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", name)
            or re.fullmatch(r"pi_\d+", name)):
        raise ValueError("development checkpoint needs a safe non-formal policy name")
    config_path = Path(config_path).resolve()
    checkpoint = Path(checkpoint).resolve()
    config = _load_policy_formal_config(config_path)
    run_id = _source_run_id(config)
    run_dir = checkpoint.parent.parent
    if checkpoint.parent.name != "checkpoints" or run_dir.name != run_id:
        raise ValueError("development checkpoint run directory does not match config run_id")
    match = re.fullmatch(r"transition_(0|[1-9][0-9]*)", checkpoint.name)
    if match is None or (int(match[1]) not in config.formal.checkpoint_transitions and not (allow_initialization and int(match[1]) == 0)):
        raise ValueError("development checkpoint is not a declared positive milestone")
    transition = int(match[1])
    if transition > config.ppo.requested_transitions:
        raise ValueError("development checkpoint exceeds the declared training budget")
    report_path = run_dir / "formal_report.json"
    report = _read_json(report_path)
    _validate_formal_report(report, config)
    if (report.get("checkpoint_transitions") != list(config.formal.checkpoint_transitions)
            or report.get("train_panel_transitions") != list(config.formal.train_panel_transitions)):
        raise ValueError("development source report checkpoint schedule drift")
    if report.get("checkpoint_evaluation") != config.raw.get("checkpoint_evaluation"):
        raise ValueError("development source checkpoint evaluation declaration drift")
    identity = _checkpoint_identity(config)
    payload = load_checkpoint(checkpoint, expected=identity)
    if payload.training_transitions != transition:
        raise ValueError("development checkpoint training-transition drift")
    iteration = config.raw.get("claim_boundary", {}).get("iteration", 0)
    if type(iteration) is not int or iteration < 0:
        raise ValueError("development source iteration metadata is invalid")
    return FrozenDevelopmentCheckpointRecord(
        name=name, iteration=iteration, policy_role="development_checkpoint",
        checkpoint=str(checkpoint), formal_config=str(config_path),
        formal_config_sha256=config.config_sha256, xml_sha256=identity.xml_sha256,
        source_training_run_id=run_id, source_training_transitions=transition,
        source_reset_mixture=config.reset_mixture.as_dict(),
        payload_sha256=file_sha256(checkpoint / "payload.pkl"),
        normalizer_sha256=pytree_sha256(payload.observation_normalizer),
        actor_sha256=pytree_sha256(payload.actor_params),
        critic_sha256=pytree_sha256(payload.critic_params),
        actor_frame_fields=ACTOR_FRAME_FIELDS, actor_task_fields=ACTOR_TASK_FIELDS,
        action_order=ACTION_ORDER, data_role="train",
        formal_config_file_sha256=file_sha256(config_path),
        source_formal_report=str(report_path),
        source_formal_report_sha256=file_sha256(report_path),
        checkpoint_identity_sha256=file_sha256(checkpoint / "identity.json"),
        source_requested_training_transitions=int(config.ppo.requested_transitions),
    )


def _initialization_inputs(config_path: Path, source_frozen_policy: Path, name: str):
    if (not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", name)
            or re.fullmatch(r"pi_\d+", name)):
        raise ValueError("initialization policy needs a safe non-formal policy name")
    source = load_frozen_unified_manifest(source_frozen_policy)["policy"]
    source_config = _load_policy_formal_config(Path(source["formal_config"]))
    config = _load_policy_formal_config(config_path)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", _source_run_id(config)):
        raise ValueError("initialization config requires a safe run_id")
    identity = _checkpoint_identity(config)
    source_identity = _checkpoint_identity(source_config)
    if replace(identity, config_sha256=source_identity.config_sha256) != source_identity:
        raise ValueError("initialization policy observation/action/XML contract drift")
    if config.up_config_sha256 != source_config.up_config_sha256:
        raise ValueError("initialization policy upstream runtime contract drift")
    # This rebind changes only the declared recovery endpoint. Physics, rewards,
    # controller interfaces and reset/training parameters remain identical.
    source_down = _read_json(Path(source_config.down_config_path))
    target_down = _read_json(Path(config.down_config_path))
    for phase in (source_down, target_down):
        phase["descent"] = {key: value for key, value in phase.get("descent", {}).items()
                            if key not in {"recovery_ticks", "min_post_contact_forward_progress"}}
    if source_down != target_down:
        raise ValueError("initialization policy downstream runtime contract drift")
    payload = load_checkpoint(Path(source["checkpoint"]), expected=source_identity)
    return config, identity, source, payload


def _initialization_receipt(config_path, config, checkpoint, source_path, source):
    return {
        "schema": "jit_zero_training_initialization_v1", "status": "completed",
        "operation": "endpoint_identity_rebind_with_unchanged_parameters",
        "new_training_transitions": 0, "environment_interactions": 0,
        "source_frozen_policy": str(source_path),
        "source_frozen_policy_sha256": file_sha256(source_path),
        "source_policy": source,
        "formal_config": str(config_path), "formal_config_sha256": config.config_sha256,
        "formal_config_file_sha256": file_sha256(config_path),
        "checkpoint": str(checkpoint),
        "checkpoint_identity_sha256": file_sha256(checkpoint / "identity.json"),
        "payload_sha256": file_sha256(checkpoint / "payload.pkl"),
        "preserved_parameters": {field: source[field] for field in
                                 ("actor_sha256", "normalizer_sha256", "critic_sha256")},
        "new_runtime_training_completed": False,
    }


def inspect_initialization_policy(
    *, config_path: Path, checkpoint: Path, source_frozen_policy: Path,
    initialization_receipt: Path, name: str,
) -> FrozenInitializationPolicyRecord:
    """Verify original provenance and exact parameter equality with zero new PPO."""
    config_path, checkpoint, source_frozen_policy, initialization_receipt = (
        Path(path).resolve() for path in
        (config_path, checkpoint, source_frozen_policy, initialization_receipt)
    )
    config, identity, source, _ = _initialization_inputs(config_path, source_frozen_policy, name)
    payload = load_checkpoint(checkpoint, expected=identity)
    if payload.training_transitions != 0 or checkpoint.name != "transition_0":
        raise ValueError("initialization checkpoint must have zero new training transitions")
    hashes = dict(actor_sha256=pytree_sha256(payload.actor_params),
                  normalizer_sha256=pytree_sha256(payload.observation_normalizer),
                  critic_sha256=pytree_sha256(payload.critic_params))
    if any(hashes[field] != source[field] for field in hashes):
        raise ValueError("initialization checkpoint changed source parameters")
    expected = _initialization_receipt(config_path, config, checkpoint, source_frozen_policy, source)
    if _read_json(initialization_receipt) != expected:
        raise ValueError("initialization receipt provenance or zero-training contract drift")
    iteration = config.raw.get("claim_boundary", {}).get("iteration", 0)
    if type(iteration) is not int or iteration < 0:
        raise ValueError("initialization source iteration metadata is invalid")
    return FrozenInitializationPolicyRecord(
        name=name, iteration=iteration, policy_role="initialization_only",
        checkpoint=str(checkpoint), formal_config=str(config_path),
        formal_config_sha256=config.config_sha256, xml_sha256=identity.xml_sha256,
        source_training_run_id=_source_run_id(config), source_training_transitions=0,
        source_reset_mixture=config.reset_mixture.as_dict(),
        payload_sha256=file_sha256(checkpoint / "payload.pkl"), **hashes,
        actor_frame_fields=ACTOR_FRAME_FIELDS, actor_task_fields=ACTOR_TASK_FIELDS,
        action_order=ACTION_ORDER, data_role="train",
        formal_config_file_sha256=file_sha256(config_path),
        checkpoint_identity_sha256=file_sha256(checkpoint / "identity.json"),
        source_frozen_policy=str(source_frozen_policy),
        source_frozen_policy_sha256=file_sha256(source_frozen_policy),
        initialization_receipt=str(initialization_receipt),
        initialization_receipt_sha256=file_sha256(initialization_receipt),
    )


def verify_frozen_unified_record(record: Mapping[str, Any]) -> FrozenUnifiedPolicyRecord:
    """Strictly reload a frozen record and prove checkpoint/config/hash binding."""
    if record.get("policy_role") == "initialization_only":
        inspected = inspect_initialization_policy(
            config_path=Path(record["formal_config"]), checkpoint=Path(record["checkpoint"]),
            source_frozen_policy=Path(record["source_frozen_policy"]),
            initialization_receipt=Path(record["initialization_receipt"]), name=record["name"],
        )
    elif record.get("policy_role") == "development_checkpoint":
        inspected = inspect_development_checkpoint(
            config_path=Path(record["formal_config"]),
            checkpoint=Path(record["checkpoint"]), name=record["name"],
            allow_initialization=record.get("source_training_transitions") == 0,
        )
    else:
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


def freeze_development_checkpoint(
    output_dir: Path, *, config_path: Path, checkpoint: Path, name: str, allow_initialization: bool = False,
) -> dict[str, Any]:
    """Freeze a completed-run milestone for diagnostic TRAIN comparisons only."""
    record = inspect_development_checkpoint(
        config_path=config_path, checkpoint=checkpoint, name=name, allow_initialization=allow_initialization,
    )
    protocol = {
        "schema": FROZEN_DEVELOPMENT_CHECKPOINT_SCHEMA,
        "status": "frozen", "immutable_parameters": True,
        "copied_checkpoint": False, "training_transitions": 0,
        "environment_interactions": 0, "expert_switching_used": False,
        "policy": _record_payload(record),
        "claim_boundary": {
            "envelope_expansion_authority": False, "pi_unified_star_claim": False,
            "jce_jel_claim": False, "certified_safe_tube_claim": False,
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


def freeze_initialization_policy(
    output_dir: Path, *, config_path: Path, source_frozen_policy: Path, name: str,
) -> dict[str, Any]:
    """Rebind unchanged source parameters to a recovery endpoint, with zero PPO.

    The new checkpoint has local transition zero. The receipt retains the full
    verified source policy record, including its original training provenance.
    No new-runtime formal training report is created or required.
    """
    output_dir, config_path, source_frozen_policy = (
        Path(path).resolve() for path in (output_dir, config_path, source_frozen_policy)
    )
    config, identity, source, payload = _initialization_inputs(config_path, source_frozen_policy, name)
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = output_dir / _source_run_id(config) / "checkpoints/transition_0"
    save_checkpoint(checkpoint, replace(payload, identity=identity, training_transitions=0))
    receipt_path = output_dir / "initialization_receipt.json"
    receipt_path.write_text(json.dumps(
        _initialization_receipt(config_path, config, checkpoint, source_frozen_policy, source),
        indent=2, sort_keys=True, allow_nan=False,
    ) + "\n", encoding="utf-8")
    record = inspect_initialization_policy(
        config_path=config_path, checkpoint=checkpoint, source_frozen_policy=source_frozen_policy,
        initialization_receipt=receipt_path, name=name,
    )
    protocol = {
        "schema": FROZEN_INITIALIZATION_POLICY_SCHEMA, "status": "frozen",
        "immutable_parameters": True, "copied_checkpoint": True,
        "training_transitions": 0, "new_training_transitions": 0,
        "environment_interactions": 0, "expert_switching_used": False,
        "policy": _record_payload(record),
        "claim_boundary": {
            "envelope_expansion_authority": False, "pi_unified_star_claim": False,
            "jce_jel_claim": False, "certified_safe_tube_claim": False,
        },
    }
    manifest = {**protocol, "freeze_protocol_sha256": _canonical_sha256(protocol)}
    (output_dir / "frozen_unified_policy.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8",
    )
    return manifest


def load_frozen_unified_manifest(path: Path) -> dict[str, Any]:
    payload = _read_json(Path(path))
    if (
        payload.get("schema") not in {
            FROZEN_UNIFIED_POLICY_SCHEMA, FROZEN_DEVELOPMENT_CHECKPOINT_SCHEMA,
            FROZEN_INITIALIZATION_POLICY_SCHEMA,
        }
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
    initialization = payload["schema"] == FROZEN_INITIALIZATION_POLICY_SCHEMA
    if payload.get("copied_checkpoint") is not initialization:
        raise ValueError("frozen unified policy unexpectedly copied its checkpoint")
    if payload.get("training_transitions") != 0 or payload.get("environment_interactions") != 0:
        raise ValueError("freezing a unified policy must use zero interactions")
    claims = payload.get("claim_boundary", {})
    if initialization and payload.get("new_training_transitions") != 0:
        raise ValueError("initialization policy requires zero new training transitions")
    diagnostic = initialization or payload["schema"] == FROZEN_DEVELOPMENT_CHECKPOINT_SCHEMA
    if diagnostic and payload.get("expert_switching_used") is not False:
        raise ValueError("development checkpoint cannot use expert switching")
    expected_role = ("initialization_only" if initialization else
                     "development_checkpoint" if diagnostic else "envelope_expansion_authority")
    if payload.get("policy", {}).get("policy_role") != expected_role:
        raise ValueError("frozen unified-policy schema/role mismatch")
    if claims != {
        "envelope_expansion_authority": not diagnostic,
        "pi_unified_star_claim": False,
        "jce_jel_claim": False,
        "certified_safe_tube_claim": False,
    }:
        raise ValueError("frozen unified-policy claim boundary drift")
    verify_frozen_unified_record(payload["policy"])
    return payload
