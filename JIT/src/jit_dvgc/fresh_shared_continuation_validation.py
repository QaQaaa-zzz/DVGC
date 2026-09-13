"""Fresh independent validation and calibration for shared C_up/C_down fields.

This stage is predeclared after the shared 76->8 tanh->1 architecture has been
frozen and both phase-specific fields have been fit on TRAIN only.  It creates
new parent trajectories from fresh seeds, generates fixed perturbation panels
through authoritative unified dynamics, labels continuations under frozen pi_0,
and calibrates phase-specific thresholds without refitting any model parameter.

The already-consumed Iteration-0 validation outcomes are never read.  Its
candidate catalog may be read for identity-only overlap rejection so the fresh
bank cannot silently duplicate an old validation state.
"""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import jax
from jax import numpy as jp
import numpy as np

from .checkpoint import CheckpointIdentity, load_checkpoint
from .config import file_sha256
from .constants import ACTION_ORDER, ACTOR_FRAME_FIELDS, ACTOR_TASK_FIELDS
from .env import TwoPhaseBikeEnv
from .expert_freeze import load_frozen_manifest, verify_frozen_record
from .expansion_validation_runtime import (
    _collect_candidates,
    _label_candidates,
    _validate_completed_acquisition,
    enumerate_validation_attempts,
    restore_validation_anchor_as_unified,
)
from .handoff_bank import pytree_sha256
from .handoff_snapshot import HandoffSnapshot, load_snapshot, save_snapshot
from .iteration_train_evidence import (
    canonical_sha256,
    load_frozen_iteration_train_evidence,
)
from .policy_conditioned_continuation_field import _metrics
from .ppo import make_checkpoint_policy
from .unified_formal import build_unified_formal_environment, load_unified_formal_config
from .unified_policy_freeze import load_frozen_unified_manifest
from .unified_training import checkpoint_identity
from .upstream_boundary import physical_state_sha256 as legacy_state_sha256
from .upstream_checkpoint_train_evidence import (
    load_frozen_upstream_checkpoint_train_evidence,
)
from .upstream_matched_checkpoint_domain_cv import _sigmoid


CONFIG_SCHEMA = "jit_fresh_shared_continuation_validation_config_v1"
PROTOCOL_SCHEMA = "jit_fresh_shared_continuation_validation_protocol_v1"
RUNTIME_SCHEMA = "jit_fresh_shared_continuation_validation_runtime_v1"
SOURCE_SCHEMA = "jit_fresh_validation_source_parent_catalog_v1"
CALIBRATION_SCHEMA = "jit_shared_continuation_phase_calibration_v1"
SUMMARY_SCHEMA = "jit_fresh_shared_continuation_validation_summary_v1"
STATUS = "predeclared_after_shared_full_train_refit_before_fresh_validation_outcomes"


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha(value: Any, *, field: str) -> str:
    text = str(value).lower()
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise ValueError(f"{field} must be SHA-256")
    return text


def _repository_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
    ).strip()


def _truth(value: Any) -> bool:
    return bool(np.asarray(jax.device_get(value)))


def _near_any(observation: np.ndarray, references: np.ndarray, *, atol: float) -> bool:
    if not len(references):
        return False
    return bool(np.any(np.all(np.abs(references - observation) <= float(atol), axis=1)))


def _rows_from_catalog(path: Path) -> tuple[dict[str, Any], ...]:
    payload = _read_object(path)
    rows = payload.get("entries")
    if not isinstance(rows, list):
        raise ValueError("identity catalog requires entries")
    return tuple(dict(row) for row in rows)


def load_fresh_shared_validation_config(path: Path) -> dict[str, Any]:
    config = _read_object(Path(path))
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("unsupported fresh shared validation config")
    protocol = config.get("protocol")
    if not isinstance(protocol, Mapping) or protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("fresh shared validation protocol drift")
    if protocol.get("status") != STATUS:
        raise ValueError("fresh shared validation status drift")
    if int(protocol.get("iteration", -1)) != 0 or protocol.get("policy_name") != "pi_0":
        raise ValueError("fresh shared validation policy drift")
    if protocol.get("fresh_seed_families") != [1000007, 1000008]:
        raise ValueError("fresh validation seed-family contract drift")
    if 1000006 in protocol.get("fresh_seed_families", []):
        raise ValueError("consumed validation seed cannot enter fresh validation")
    for field in (
        "frozen_policy_file_sha256",
        "policy_actor_sha256",
        "policy_payload_sha256",
        "xml_sha256",
        "shared_refit_summary_sha256",
        "architecture_manifest_sha256",
        "upstream_field_manifest_sha256",
        "upstream_field_file_sha256",
        "downstream_field_manifest_sha256",
        "downstream_field_file_sha256",
        "upstream_train_manifest_sha256",
        "downstream_train_manifest_sha256",
        "consumed_validation_identity_catalog_sha256",
    ):
        _sha(protocol.get(field), field=field)
    source = protocol.get("source_parent_generation")
    if not isinstance(source, Mapping):
        raise ValueError("fresh validation source-parent generation missing")
    if source.get("upstream_source_transitions") != [4988928, 7987200, 9977856]:
        raise ValueError("fresh upstream support-domain drift")
    if source.get("downstream_source_transitions") != [4988928]:
        raise ValueError("fresh downstream support-domain drift")
    if source.get("upstream_role") != "ascending_entry" or source.get("downstream_role") != "post_apex":
        raise ValueError("fresh validation source role drift")
    if int(source.get("max_ticks", -1)) != 128 or source.get("deterministic") is not True:
        raise ValueError("fresh validation source rollout contract drift")
    checkpoints = source.get("source_checkpoints")
    if not isinstance(checkpoints, Mapping) or sorted(int(k) for k in checkpoints) != [4988928, 7987200, 9977856]:
        raise ValueError("fresh validation source checkpoint set drift")
    panels = protocol.get("panels")
    expected_upstream = {
        "action_names": ["steer", "rear_wheel_drive", "hip", "knee"],
        "signs": [-1, 1],
        "strengths": [0.025, 0.1],
        "durations": [4, 8, 16],
        "max_label_ticks": 400,
        "terminal_clipping": True,
    }
    expected_downstream = {
        "action_names": ["hip"],
        "signs": [1],
        "strengths": [0.15, 0.2, 0.3, 0.32, 0.35, 0.4, 0.45, 0.5],
        "durations": [30],
        "max_label_ticks": 400,
        "terminal_clipping": True,
    }
    if not isinstance(panels, Mapping) or panels.get("upstream") != expected_upstream or panels.get("downstream") != expected_downstream:
        raise ValueError("fresh validation panel drift")
    near = protocol.get("near_duplicate_audit")
    if near != {
        "actor_observation_atol": 0.01,
        "reject_train_exact_state": True,
        "reject_train_near_duplicate_observation": True,
        "reject_consumed_validation_exact_state": True,
        "reject_consumed_validation_near_duplicate_observation": True,
        "reject_duplicate_fresh_state": True,
        "no_replacement_after_candidate_exclusion": True,
    }:
        raise ValueError("fresh validation independence audit drift")
    budget = protocol.get("interaction_budget")
    if budget != {
        "source_parent_rollout_maximum_environment_interactions": 768,
        "attempt_count": 304,
        "maximum_acquisition_environment_interactions": 3168,
        "maximum_labeling_environment_interactions": 121600,
        "training_transitions": 0,
    }:
        raise ValueError("fresh validation interaction budget drift")
    calibration = protocol.get("calibration")
    if calibration != {
        "decision_rule": "accept_if_score_strictly_greater_than_max_fresh_validation_negative_score",
        "minimum_validation_roc_auc": 0.70,
        "minimum_validation_positive_recall": 0.20,
        "require_positive_support_in_every_validation_parent": True,
        "require_accepted_positive_in_every_validation_parent": True,
        "accepted_validation_negative_count_must_be_zero": True,
        "validation_hyperparameter_search": False,
        "model_parameters_refit_on_validation": False,
        "threshold_is_safety_certificate": False,
    }:
        raise ValueError("fresh validation calibration drift")
    if protocol.get("data_policy") != {
        "fresh_validation_outcomes_may_calibrate_phase_thresholds": True,
        "fresh_validation_rows_may_enter_train_or_tube": False,
        "consumed_validation_outcomes_read": False,
        "consumed_validation_identity_only_overlap_audit": True,
        "test_data_used": False,
        "final_evaluation_data_used": False,
    }:
        raise ValueError("fresh validation data policy drift")
    if protocol.get("claim_boundary") != {
        "fresh_independent_validation_only": True,
        "shared_architecture_frozen": True,
        "field_parameters_frozen": True,
        "phase_specific_threshold_calibration_allowed": True,
        "certified_probability_claim": False,
        "certified_safe_set_claim": False,
        "tube_1_constructed": False,
        "pi_1_trained": False,
        "jce_jel_claim": False,
    }:
        raise ValueError("fresh validation claim boundary drift")
    if not str(config.get("output_dir", "")):
        raise ValueError("fresh validation output_dir missing")
    if canonical_sha256(protocol) != config.get("expected_protocol_sha256"):
        raise ValueError("fresh validation protocol SHA drift")
    return config


def _verify_shared_refit(protocol: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(str(protocol["shared_refit_root"]))
    summary = _read_object(root / "summary.json")
    if summary.get("summary_sha256") != protocol["shared_refit_summary_sha256"]:
        raise ValueError("fresh validation shared refit summary drift")
    if summary.get("status") != "completed" or summary.get("architecture_frozen") is not True:
        raise ValueError("fresh validation shared refit is not production-ready")
    if summary.get("fields_calibrated") is not False or summary.get("validation_rows_used") != 0:
        raise ValueError("fresh validation fields were already calibrated or validation-touched")
    architecture = _read_object(root / "architecture_manifest.json")
    if architecture.get("architecture_manifest_sha256") != protocol["architecture_manifest_sha256"]:
        raise ValueError("fresh validation architecture manifest drift")
    if architecture.get("shared_up_down_architecture") is not True:
        raise ValueError("fresh validation architecture is not shared")
    result = {"summary": summary, "architecture": architecture, "phases": {}}
    for phase in ("upstream", "downstream"):
        manifest = _read_object(root / phase / "manifest.json")
        expected_manifest = protocol[f"{phase}_field_manifest_sha256"]
        expected_field = protocol[f"{phase}_field_file_sha256"]
        if manifest.get("manifest_sha256") != expected_manifest:
            raise ValueError(f"fresh validation {phase} field manifest drift")
        field_path = root / phase / "field.npz"
        if file_sha256(field_path) != expected_field or manifest.get("field_file_sha256") != expected_field:
            raise ValueError(f"fresh validation {phase} field file drift")
        if manifest.get("status") != "completed_uncalibrated" or manifest.get("calibrated") is not False:
            raise ValueError(f"fresh validation {phase} field must remain uncalibrated")
        if manifest.get("architecture") != "76->8_tanh->1" or manifest.get("parameter_count") != 625:
            raise ValueError(f"fresh validation {phase} architecture drift")
        result["phases"][phase] = {"manifest": manifest, "field_path": field_path}
    return result


def audit_fresh_shared_validation_preflight(config_path: Path) -> dict[str, Any]:
    config = load_fresh_shared_validation_config(config_path)
    protocol = config["protocol"]
    shared = _verify_shared_refit(protocol)
    frozen_path = Path(str(protocol["frozen_policy"]))
    frozen = load_frozen_unified_manifest(frozen_path)
    policy = frozen["policy"]
    if file_sha256(frozen_path) != protocol["frozen_policy_file_sha256"]:
        raise ValueError("fresh validation frozen policy file drift")
    for field, expected in (
        ("actor_sha256", protocol["policy_actor_sha256"]),
        ("payload_sha256", protocol["policy_payload_sha256"]),
        ("xml_sha256", protocol["xml_sha256"]),
    ):
        if policy.get(field) != expected:
            raise ValueError(f"fresh validation policy {field} drift")
    upstream_manifest, _ = load_frozen_upstream_checkpoint_train_evidence(
        Path(str(protocol["upstream_train_evidence"]))
    )
    if upstream_manifest.get("manifest_sha256") != protocol["upstream_train_manifest_sha256"]:
        raise ValueError("fresh validation upstream TRAIN drift")
    downstream_manifest, _ = load_frozen_iteration_train_evidence(
        Path(str(protocol["downstream_train_evidence"]))
    )
    if downstream_manifest.get("manifest_sha256") != protocol["downstream_train_manifest_sha256"]:
        raise ValueError("fresh validation downstream TRAIN drift")
    identity_catalog = Path(str(protocol["consumed_validation_identity_catalog"]))
    if file_sha256(identity_catalog) != protocol["consumed_validation_identity_catalog_sha256"]:
        raise ValueError("fresh validation consumed-bank identity catalog drift")
    _rows_from_catalog(identity_catalog)
    source_cfg = protocol["source_parent_generation"]
    experts = load_frozen_manifest(Path(str(source_cfg["frozen_experts_manifest"])))
    phase_u_config, _ = verify_frozen_record(experts["experts"]["pi_up_star"])
    source_env = TwoPhaseBikeEnv(phase_u_config)
    identity = CheckpointIdentity(
        config_sha256=phase_u_config.config_sha256,
        xml_sha256=source_env._bundle.xml_sha256,
        actor_frame_fields=ACTOR_FRAME_FIELDS,
        actor_task_fields=ACTOR_TASK_FIELDS,
        action_order=ACTION_ORDER,
    )
    source_checkpoints = {}
    for key, value in source_cfg["source_checkpoints"].items():
        payload = load_checkpoint(Path(str(value)), expected=identity)
        if int(payload.training_transitions) != int(key):
            raise ValueError("fresh validation source checkpoint transition drift")
        source_checkpoints[str(key)] = pytree_sha256(payload.actor_params)
    return {
        "schema": "jit_fresh_shared_continuation_validation_preflight_v1",
        "status": "fresh_validation_preflight_ready",
        "protocol_sha256": str(config["expected_protocol_sha256"]),
        "shared_refit_summary_sha256": str(shared["summary"]["summary_sha256"]),
        "architecture_manifest_sha256": str(shared["architecture"]["architecture_manifest_sha256"]),
        "fresh_seed_families": list(protocol["fresh_seed_families"]),
        "upstream_parent_count": 6,
        "downstream_parent_count": 2,
        "attempt_count": 304,
        "source_checkpoint_actor_sha256": source_checkpoints,
        "environment_interactions": 0,
        "training_transitions": 0,
        "consumed_validation_outcomes_read": False,
        "test_data_used": False,
        "final_evaluation_data_used": False,
    }


def _collect_source_parents(
    protocol: Mapping[str, Any],
    *,
    output: Path,
) -> tuple[dict[str, Any], dict[tuple[str, int], HandoffSnapshot]]:
    source_cfg = protocol["source_parent_generation"]
    experts = load_frozen_manifest(Path(str(source_cfg["frozen_experts_manifest"])))
    phase_u_config, _ = verify_frozen_record(experts["experts"]["pi_up_star"])
    env = TwoPhaseBikeEnv(phase_u_config)
    identity = CheckpointIdentity(
        config_sha256=phase_u_config.config_sha256,
        xml_sha256=env._bundle.xml_sha256,
        actor_frame_fields=ACTOR_FRAME_FIELDS,
        actor_task_fields=ACTOR_TASK_FIELDS,
        action_order=ACTION_ORDER,
    )
    reset_fn = jax.jit(env.reset_natural)
    max_ticks = int(source_cfg["max_ticks"])
    seeds = [int(value) for value in protocol["fresh_seed_families"]]
    upstream_transitions = {int(value) for value in source_cfg["upstream_source_transitions"]}
    downstream_transitions = {int(value) for value in source_cfg["downstream_source_transitions"]}
    required_transitions = sorted(upstream_transitions | downstream_transitions)
    entries: list[dict[str, Any]] = []
    interactions = 0
    maximum = int(protocol["interaction_budget"]["source_parent_rollout_maximum_environment_interactions"])

    for transitions in required_transitions:
        checkpoint_path = Path(str(source_cfg["source_checkpoints"][str(transitions)]))
        payload = load_checkpoint(checkpoint_path, expected=identity)
        actor_sha = pytree_sha256(payload.actor_params)
        policy = jax.jit(make_checkpoint_policy(env, payload, deterministic=True))
        step_fn = jax.jit(env.step)
        bank_rel = Path("source_parents") / f"source_{transitions}"
        bank = output / bank_rel
        (bank / "snapshots").mkdir(parents=True, exist_ok=True)

        for seed in seeds:
            state = reset_fn(jax.random.PRNGKey(seed))
            previous_ascending = False
            apex_tick: int | None = None
            captured: dict[str, bool] = {
                "ascending_entry": False,
                "post_apex": transitions not in downstream_transitions,
            }
            parent = f"transition_{transitions}__{seed}"
            base_key = jax.random.PRNGKey(seed)
            for tick in range(max_ticks + 1):
                events = state.info["events"]
                ascending = _truth(events.ascending_seen)
                apex = _truth(events.apex_seen)
                role: str | None = None
                if transitions in upstream_transitions and ascending and not previous_ascending:
                    role = "ascending_entry"
                if apex and apex_tick is None:
                    apex_tick = tick
                post_apex_now = (
                    transitions in downstream_transitions
                    and apex_tick is not None
                    and tick == apex_tick + 1
                    and not captured["post_apex"]
                )
                roles = []
                if role is not None and not captured[role]:
                    roles.append(role)
                if post_apex_now:
                    roles.append("post_apex")
                for selected_role in roles:
                    snapshot = env.capture_handoff_snapshot(
                        state,
                        policy_sha256=actor_sha,
                        parent_trajectory=parent,
                        parent_tick=tick,
                    )
                    relative = Path("snapshots") / f"seed_{seed}_{selected_role}"
                    save_snapshot(bank / relative, snapshot)
                    entries.append(
                        {
                            "phase": "upstream" if selected_role == "ascending_entry" else "downstream",
                            "source_training_transitions": transitions,
                            "source_checkpoint": str(checkpoint_path),
                            "source_actor_sha256": actor_sha,
                            "source_bank": str(bank_rel),
                            "snapshot": str(relative),
                            "parent_group_id": parent,
                            "snapshot_parent_trajectory": parent,
                            "role": selected_role,
                            "tick": int(tick),
                            "seed": int(seed),
                            "state_sha256": legacy_state_sha256(snapshot),
                            "actor_observation": np.asarray(snapshot.observation, dtype=np.float32).tolist(),
                        }
                    )
                    captured[selected_role] = True
                if all(captured.values()):
                    break
                if _truth(state.info.get("terminated", False)) or _truth(state.info.get("truncated", False)):
                    break
                if tick == max_ticks:
                    break
                action_result = policy(state.obs, jax.random.fold_in(base_key, tick))
                action = action_result[0] if isinstance(action_result, tuple) else action_result
                state = step_fn(state, action)
                jax.block_until_ready(state)
                interactions += 1
                previous_ascending = ascending
            if not all(captured.values()):
                missing = sorted(name for name, done in captured.items() if not done)
                raise ValueError(f"fresh validation source parent missing roles {missing} for {parent}")

    if interactions > maximum:
        raise ValueError("fresh validation source-parent collection exceeded budget")
    expected_up = len(upstream_transitions) * len(seeds)
    expected_down = len(downstream_transitions) * len(seeds)
    up_entries = [row for row in entries if row["phase"] == "upstream"]
    down_entries = [row for row in entries if row["phase"] == "downstream"]
    if len(up_entries) != expected_up or len(down_entries) != expected_down:
        raise ValueError("fresh validation source-parent count drift")
    report = {
        "schema": SOURCE_SCHEMA,
        "status": "completed",
        "fresh_seed_families": seeds,
        "upstream_parent_count": len(up_entries),
        "downstream_parent_count": len(down_entries),
        "environment_interactions": interactions,
        "maximum_environment_interactions": maximum,
        "training_transitions": 0,
        "entries": entries,
    }
    _write_json(output / "source_parent_catalog.json", report)
    return report, _load_source_anchor_map(output, report)


def _load_source_anchor_map(
    output: Path,
    report: Mapping[str, Any],
) -> dict[tuple[str, int], HandoffSnapshot]:
    result: dict[tuple[str, int], HandoffSnapshot] = {}
    by_phase = {
        "upstream": [dict(row) for row in report["entries"] if row["phase"] == "upstream"],
        "downstream": [dict(row) for row in report["entries"] if row["phase"] == "downstream"],
    }
    for phase in ("upstream", "downstream"):
        for index, row in enumerate(by_phase[phase]):
            snapshot = load_snapshot(output / str(row["source_bank"]) / str(row["snapshot"]))
            if legacy_state_sha256(snapshot) != row["state_sha256"]:
                raise ValueError("fresh validation source snapshot identity drift")
            result[(phase, index)] = snapshot
    return result


def _audit_fresh_anchor_independence(
    protocol: Mapping[str, Any],
    *,
    source_report: Mapping[str, Any],
) -> dict[str, Any]:
    _up_manifest, up_raw = load_frozen_upstream_checkpoint_train_evidence(
        Path(str(protocol["upstream_train_evidence"]))
    )
    _down_manifest, down_raw = load_frozen_iteration_train_evidence(
        Path(str(protocol["downstream_train_evidence"]))
    )
    train_by_phase = {
        "upstream": [dict(row) for row in up_raw if row.get("phase") == "upstream"],
        "downstream": [dict(row) for row in down_raw if row.get("phase") == "downstream"],
    }
    old_rows = _rows_from_catalog(Path(str(protocol["consumed_validation_identity_catalog"])))
    old_states = {str(row.get("state_sha256", "")) for row in old_rows}
    old_obs = np.asarray([row["actor_observation"] for row in old_rows], dtype=np.float32)
    atol = float(protocol["near_duplicate_audit"]["actor_observation_atol"])
    phase_stats = {}
    for phase in ("upstream", "downstream"):
        anchors = [dict(row) for row in source_report["entries"] if row["phase"] == phase]
        train = train_by_phase[phase]
        train_groups = {str(row["parent_group_id"]) for row in train}
        train_states = {str(row["state_sha256"]) for row in train}
        train_obs = np.asarray([row["actor_observation"] for row in train], dtype=np.float32)
        seen_obs: list[np.ndarray] = []
        seen_states: set[str] = set()
        for row in anchors:
            group = str(row["parent_group_id"])
            state = str(row["state_sha256"])
            obs = np.asarray(row["actor_observation"], dtype=np.float32)
            if group in train_groups:
                raise ValueError("fresh validation source parent overlaps TRAIN group")
            if state in train_states or _near_any(obs, train_obs, atol=atol):
                raise ValueError("fresh validation source anchor overlaps TRAIN state/observation")
            if state in old_states or _near_any(obs, old_obs, atol=atol):
                raise ValueError("fresh validation source anchor overlaps consumed validation identity")
            if state in seen_states:
                raise ValueError("fresh validation repeats source anchor state")
            if seen_obs and _near_any(obs, np.asarray(seen_obs), atol=atol):
                raise ValueError("fresh validation source anchors are near-duplicates")
            seen_states.add(state)
            seen_obs.append(obs)
        phase_stats[phase] = {
            "parent_count": len(anchors),
            "train_parent_overlap_count": 0,
            "train_exact_or_near_overlap_count": 0,
            "consumed_validation_exact_or_near_overlap_count": 0,
            "fresh_anchor_duplicate_or_near_duplicate_count": 0,
        }
    return {
        "status": "independent",
        "actor_observation_atol": atol,
        "consumed_validation_outcomes_read": False,
        "phase_stats": phase_stats,
    }


def _scientific_protocol(
    protocol: Mapping[str, Any],
    *,
    source_report: Mapping[str, Any],
) -> dict[str, Any]:
    by_phase = {
        "upstream": [dict(row) for row in source_report["entries"] if row["phase"] == "upstream"],
        "downstream": [dict(row) for row in source_report["entries"] if row["phase"] == "downstream"],
    }
    return {
        "iteration": 0,
        "policy_name": "pi_0",
        "validation_seed": int(protocol["fresh_seed_families"][0]),
        "sources": {
            phase: {
                "anchors": [
                    {
                        "source_bank": row["source_bank"],
                        "snapshot": row["snapshot"],
                        "parent_group_id": row["parent_group_id"],
                        "state_sha256": row["state_sha256"],
                        "role": row["role"],
                        "tick": row["tick"],
                    }
                    for row in by_phase[phase]
                ]
            }
            for phase in ("upstream", "downstream")
        },
        "panels": dict(protocol["panels"]),
        "near_duplicate_audit": {
            "actor_observation_atol": float(protocol["near_duplicate_audit"]["actor_observation_atol"])
        },
        "interaction_budget": {
            "attempt_count": int(protocol["interaction_budget"]["attempt_count"]),
            "maximum_acquisition_environment_interactions": int(
                protocol["interaction_budget"]["maximum_acquisition_environment_interactions"]
            ),
            "maximum_labeling_environment_interactions": int(
                protocol["interaction_budget"]["maximum_labeling_environment_interactions"]
            ),
            "training_transitions": 0,
        },
        "claim_boundary": dict(protocol["claim_boundary"]),
        "data_policy": dict(protocol["data_policy"]),
    }


def _runtime_protocol(
    config_path: Path,
    config: Mapping[str, Any],
    scientific: Mapping[str, Any],
) -> dict[str, Any]:
    base = {
        "schema": RUNTIME_SCHEMA,
        "status": "predeclared",
        "repository_head": _repository_head(),
        "config_path": str(config_path),
        "config_file_sha256": file_sha256(config_path),
        "scientific_protocol_sha256": canonical_sha256(scientific),
        "declared_protocol_sha256": str(config["expected_protocol_sha256"]),
        "shared_refit_summary_sha256": str(config["protocol"]["shared_refit_summary_sha256"]),
        "policy_actor_sha256": str(config["protocol"]["policy_actor_sha256"]),
        "policy_payload_sha256": str(config["protocol"]["policy_payload_sha256"]),
        "fresh_seed_families": list(config["protocol"]["fresh_seed_families"]),
        "attempt_count": int(config["protocol"]["interaction_budget"]["attempt_count"]),
        "attempt_schedule_sha256": canonical_sha256({"attempts": list(enumerate_validation_attempts(scientific))}),
        "model_parameters_frozen": True,
        "phase_thresholds_uncalibrated_before_run": True,
        "consumed_validation_outcomes_read": False,
        "training_transitions": 0,
        "test_data_used": False,
        "final_evaluation_data_used": False,
    }
    return {**base, "protocol_sha256": canonical_sha256(base)}


def _load_phase_field(root: Path, phase: str) -> dict[str, np.ndarray]:
    with np.load(root / phase / "field.npz") as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def _field_scores(field: Mapping[str, np.ndarray], observations: np.ndarray) -> np.ndarray:
    mean = np.asarray(field["mean"], dtype=np.float32)
    std = np.asarray(field["std"], dtype=np.float32)
    x = np.clip((observations - mean) / std, -10.0, 10.0).astype(np.float32)
    hidden = np.tanh(x @ np.asarray(field["w1"], dtype=np.float32) + np.asarray(field["b1"], dtype=np.float32))
    logits = hidden @ np.asarray(field["w2"], dtype=np.float32) + float(np.asarray(field["b2"]))
    return _sigmoid(np.asarray(logits, dtype=np.float64))


def _calibrate_phase(
    phase: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    field: Mapping[str, np.ndarray],
    calibration_cfg: Mapping[str, Any],
    field_manifest_sha256: str,
    field_file_sha256: str,
    validation_runtime_protocol_sha256: str,
) -> dict[str, Any]:
    selected = [dict(row) for row in rows if row["phase"] == phase]
    labels = np.asarray([int(row["label"]) for row in selected], dtype=np.float64)
    observations = np.asarray([row["actor_observation"] for row in selected], dtype=np.float32)
    groups = [str(row["parent_group_id"]) for row in selected]
    if set(labels.tolist()) != {0.0, 1.0}:
        return {
            "schema": CALIBRATION_SCHEMA,
            "status": "failed_class_support",
            "phase": phase,
            "calibration_passed": False,
            "candidate_count": len(selected),
            "positive_count": int(np.sum(labels == 1.0)),
            "negative_count": int(np.sum(labels == 0.0)),
            "acceptance_threshold_exclusive": None,
            "gate": {"both_classes_present": False},
        }
    scores = _field_scores(field, observations)
    metrics = _metrics(labels, scores)
    threshold = float(np.max(scores[labels == 0.0]))
    accepted = scores > threshold
    accepted_negative = int(np.sum(accepted & (labels == 0.0)))
    positive_recall = float(np.mean(accepted[labels == 1.0]))
    all_groups = sorted(set(groups))
    positive_groups = sorted({groups[i] for i in range(len(groups)) if labels[i] == 1.0})
    accepted_positive_groups = sorted(
        {groups[i] for i in range(len(groups)) if labels[i] == 1.0 and bool(accepted[i])}
    )
    gate = {
        "both_classes_present": True,
        "validation_roc_auc_at_least_minimum": bool(
            metrics["roc_auc"] >= float(calibration_cfg["minimum_validation_roc_auc"])
        ),
        "validation_positive_recall_at_least_minimum": bool(
            positive_recall >= float(calibration_cfg["minimum_validation_positive_recall"])
        ),
        "positive_support_in_every_validation_parent": set(positive_groups) == set(all_groups),
        "accepted_positive_in_every_validation_parent": set(accepted_positive_groups) == set(all_groups),
        "accepted_validation_negative_count_zero": accepted_negative == 0,
    }
    passed = bool(all(gate.values()))
    predictions = [
        {
            "candidate_id": str(row["candidate_id"]),
            "state_sha256": str(row["state_sha256"]),
            "parent_group_id": str(row["parent_group_id"]),
            "label": int(labels[index]),
            "score": float(scores[index]),
            "accepted": bool(accepted[index]),
        }
        for index, row in enumerate(selected)
    ]
    return {
        "schema": CALIBRATION_SCHEMA,
        "status": "completed",
        "phase": phase,
        "field_manifest_sha256": field_manifest_sha256,
        "field_file_sha256": field_file_sha256,
        "validation_runtime_protocol_sha256": validation_runtime_protocol_sha256,
        "candidate_count": len(selected),
        "positive_count": int(np.sum(labels == 1.0)),
        "negative_count": int(np.sum(labels == 0.0)),
        "parent_group_count": len(all_groups),
        "metrics": metrics,
        "acceptance_threshold_exclusive": threshold,
        "validation_positive_recall_at_threshold": positive_recall,
        "accepted_validation_negative_count": accepted_negative,
        "validation_parent_groups": all_groups,
        "positive_validation_parent_groups": positive_groups,
        "accepted_positive_parent_groups": accepted_positive_groups,
        "gate": gate,
        "calibration_passed": passed,
        "predictions": predictions,
        "model_parameters_refit_on_validation": False,
        "threshold_is_safety_certificate": False,
    }


def execute_fresh_shared_validation(
    config_path: Path,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    config_path = Path(config_path)
    preflight = audit_fresh_shared_validation_preflight(config_path)
    config = load_fresh_shared_validation_config(config_path)
    protocol = config["protocol"]
    output = Path(str(config["output_dir"]))

    if output.exists():
        if not resume:
            raise FileExistsError(f"fresh validation output already exists: {output}")
        if (output / "summary.json").exists():
            existing = _read_object(output / "summary.json")
            if existing.get("status") == "completed":
                return existing
        if not (output / "source_parent_catalog.json").exists():
            raise ValueError("fresh validation cannot resume incomplete source-parent generation")
        source_report = _read_object(output / "source_parent_catalog.json")
        anchors = _load_source_anchor_map(output, source_report)
    else:
        output.mkdir(parents=True, exist_ok=False)
        _write_json(output / "preflight.json", preflight)
        source_report, anchors = _collect_source_parents(protocol, output=output)
        independence = _audit_fresh_anchor_independence(protocol, source_report=source_report)
        _write_json(output / "source_independence_audit.json", independence)

    if not (output / "source_independence_audit.json").exists():
        independence = _audit_fresh_anchor_independence(protocol, source_report=source_report)
        _write_json(output / "source_independence_audit.json", independence)
    else:
        independence = _read_object(output / "source_independence_audit.json")
    if independence.get("status") != "independent":
        raise ValueError("fresh validation source-parent independence did not close")

    scientific = _scientific_protocol(protocol, source_report=source_report)
    if len(enumerate_validation_attempts(scientific)) != 304:
        raise ValueError("fresh validation attempt count drift")
    runtime = _runtime_protocol(config_path, config, scientific)
    if (output / "runtime_protocol.json").exists():
        existing_runtime = _read_object(output / "runtime_protocol.json")
        if existing_runtime != runtime:
            raise ValueError("cannot resume fresh validation under changed runtime protocol")
    else:
        _write_json(output / "runtime_protocol.json", runtime)
        _write_json(output / "scientific_protocol.json", scientific)

    if jax.default_backend() != "gpu":
        raise RuntimeError("fresh shared validation requires the visible JAX GPU")
    frozen = load_frozen_unified_manifest(Path(str(protocol["frozen_policy"])))
    policy_record = frozen["policy"]
    formal = load_unified_formal_config(Path(policy_record["formal_config"]))
    runtime_config, runtime_artifact, env = build_unified_formal_environment(
        Path(policy_record["formal_config"])
    )
    if runtime_config.config_sha256 != formal.config_sha256:
        raise ValueError("fresh validation formal config drift")
    upstream_manifest, upstream_raw = load_frozen_upstream_checkpoint_train_evidence(
        Path(str(protocol["upstream_train_evidence"]))
    )
    downstream_manifest, downstream_raw = load_frozen_iteration_train_evidence(
        Path(str(protocol["downstream_train_evidence"]))
    )
    if runtime_artifact.manifest["manifest_sha256"] != downstream_manifest["source_tube_manifest_sha256"]:
        raise ValueError("fresh validation unified source Tube drift")
    payload = load_checkpoint(
        Path(policy_record["checkpoint"]), expected=checkpoint_identity(runtime_config, env)
    )
    policy = jax.jit(make_checkpoint_policy(env, payload, deterministic=True))
    step_fn = jax.jit(env.step)

    for phase in ("upstream", "downstream"):
        for index in range(len(scientific["sources"][phase]["anchors"])):
            state = restore_validation_anchor_as_unified(
                anchors[(phase, index)], phase=phase, env=env, parent_group_index=index
            )
            jax.block_until_ready(state)

    combined_train = [dict(row) for row in upstream_raw if row.get("phase") == "upstream"] + [
        dict(row) for row in downstream_raw if row.get("phase") == "downstream"
    ]
    if (output / "candidate_catalog.json").exists():
        if not resume:
            raise FileExistsError("fresh validation acquisition exists; use --resume")
        catalog = _validate_completed_acquisition(
            output=output, runtime_protocol=runtime, policy_record=policy_record
        )
    else:
        catalog = _collect_candidates(
            protocol=scientific,
            runtime_protocol=runtime,
            env=env,
            policy=policy,
            policy_record=policy_record,
            anchors=anchors,
            train_rows=combined_train,
            output=output,
            step_fn=step_fn,
        )

    labels_report = _label_candidates(
        protocol=scientific,
        runtime_protocol=runtime,
        catalog=catalog,
        env=env,
        policy=policy,
        policy_record=policy_record,
        output=output,
        step_fn=step_fn,
        resume=resume,
    )
    labels = json.loads((output / "labels.json").read_text(encoding="utf-8"))
    if not isinstance(labels, list):
        raise ValueError("fresh validation labels artifact drift")

    shared_root = Path(str(protocol["shared_refit_root"]))
    calibrations = {}
    for phase in ("upstream", "downstream"):
        calibration = _calibrate_phase(
            phase,
            labels,
            field=_load_phase_field(shared_root, phase),
            calibration_cfg=protocol["calibration"],
            field_manifest_sha256=str(protocol[f"{phase}_field_manifest_sha256"]),
            field_file_sha256=str(protocol[f"{phase}_field_file_sha256"]),
            validation_runtime_protocol_sha256=str(runtime["protocol_sha256"]),
        )
        calibrations[phase] = calibration
        _write_json(output / phase / "calibration.json", calibration)

    passed = all(bool(calibrations[p].get("calibration_passed")) for p in ("upstream", "downstream"))
    source_interactions = int(source_report["environment_interactions"])
    acquisition_interactions = int(catalog["environment_interactions"])
    labeling_interactions = int(labels_report["environment_interactions"])
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "completed",
        "artifact_role": "fresh_independent_shared_continuation_validation_and_calibration",
        "iteration": 0,
        "policy_name": "pi_0",
        "policy_actor_sha256": str(protocol["policy_actor_sha256"]),
        "policy_payload_sha256": str(protocol["policy_payload_sha256"]),
        "declared_protocol_sha256": str(config["expected_protocol_sha256"]),
        "runtime_protocol_sha256": str(runtime["protocol_sha256"]),
        "shared_refit_summary_sha256": str(protocol["shared_refit_summary_sha256"]),
        "fresh_seed_families": list(protocol["fresh_seed_families"]),
        "source_parent_counts": {
            "upstream": int(source_report["upstream_parent_count"]),
            "downstream": int(source_report["downstream_parent_count"]),
        },
        "attempt_count": int(catalog["attempt_count"]),
        "candidate_count": int(catalog["candidate_count"]),
        "label_count": int(labels_report["label_count"]),
        "positive_count": int(labels_report["positive_count"]),
        "negative_count": int(labels_report["negative_count"]),
        "phase_candidate_counts": dict(labels_report["phase_candidate_counts"]),
        "phase_positive_counts": dict(labels_report["phase_positive_counts"]),
        "parent_group_stats": dict(labels_report["parent_group_stats"]),
        "candidate_exclusion_counts": dict(catalog["exclusion_counts"]),
        "calibration": {
            phase: {
                "calibration_passed": bool(calibrations[phase].get("calibration_passed")),
                "validation_roc_auc": (
                    float(calibrations[phase]["metrics"]["roc_auc"])
                    if calibrations[phase].get("metrics") else None
                ),
                "acceptance_threshold_exclusive": calibrations[phase].get("acceptance_threshold_exclusive"),
                "validation_positive_recall_at_threshold": calibrations[phase].get("validation_positive_recall_at_threshold"),
                "gate": dict(calibrations[phase].get("gate", {})),
            }
            for phase in ("upstream", "downstream")
        },
        "fresh_validation_passed": passed,
        "source_parent_environment_interactions": source_interactions,
        "acquisition_environment_interactions": acquisition_interactions,
        "labeling_environment_interactions": labeling_interactions,
        "total_environment_interactions": source_interactions + acquisition_interactions + labeling_interactions,
        "training_transitions": 0,
        "model_parameter_updates": 0,
        "consumed_validation_outcomes_read": False,
        "fresh_validation_rows_may_enter_train_or_tube": False,
        "test_data_used": False,
        "final_evaluation_data_used": False,
        "tube_1_construction_authorized": passed,
        "tube_1_constructed": False,
        "pi_1_trained": False,
        "claim_boundary": dict(protocol["claim_boundary"]),
        "next_scientific_gate": (
            "construct core-retaining Tube_1 from TRAIN states using the frozen shared fields and fresh-validation-calibrated phase thresholds; do not place validation rows in Tube_1"
            if passed
            else "do not construct Tube_1; inspect the fixed fresh-validation phase/group failure without refitting or reusing this bank for model selection"
        ),
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    _write_json(output / "summary.json", summary)
    return summary
