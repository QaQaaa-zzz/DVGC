"""Frozen unified-policy continuation labels for envelope expansion candidates.

Candidates are TRAIN-only real-dynamics states produced by
``unified_boundary.py``.  Each candidate is evaluated as a fresh continuation
start under one immutable unified policy.  Physics, actor FIFO, phase event
context, and last action are preserved; only administrative episode counters
and accumulated return are reset, matching the existing Tube reset semantics.

This module never trains, switches experts, or upgrades a candidate into a Tube
entry.  It only produces policy-bound binary continuation evidence.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import jax
from jax import numpy as jp
import numpy as np

from .config import file_sha256
from .constants import END_REASONS
from .tube_rsi import PHASE_DOWNSTREAM, PHASE_UPSTREAM
from .unified_envelope_snapshot import (
    UnifiedEnvelopeSnapshot,
    load_unified_envelope_snapshot,
    physical_state_sha256,
    snapshot_context_sha256,
    restore_unified_envelope_snapshot,
)


UNIFIED_CONTINUATION_LABEL_SCHEMA = "jit_unified_continuation_labels_v1"
DEFAULT_UNIFIED_CONTINUATION_MAX_TICKS = 400
DEFAULT_UNIFIED_CONTINUATION_PROTOCOL_SEED = 9_511_001


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _truth(value: Any) -> bool:
    return bool(np.asarray(jax.device_get(value)))


def _integer(value: Any) -> int:
    return int(np.asarray(jax.device_get(value)))


def _finite_state(state: Any, action: Any | None = None) -> bool:
    arrays = (
        np.asarray(jax.device_get(state.data.qpos)),
        np.asarray(jax.device_get(state.data.qvel)),
        np.asarray(jax.device_get(state.obs["state"])),
    )
    if action is not None:
        arrays = (*arrays, np.asarray(jax.device_get(action)))
    return all(np.isfinite(value).all() for value in arrays)


def _candidate_snapshot_path(catalog_path: Path, row: Mapping[str, Any]) -> Path:
    return Path(catalog_path).parent / str(row["source_bank"]) / str(row["snapshot"])


def validate_unified_boundary_catalog(
    catalog: Mapping[str, Any],
    *,
    policy_record: Mapping[str, Any],
    frozen_manifest_sha256: str,
) -> tuple[Mapping[str, Any], ...]:
    """Validate one completed TRAIN-only candidate bank before policy rollout."""
    if catalog.get("schema") != "jit_unified_boundary_catalog_v1":
        raise ValueError("unified continuation labels require a unified boundary catalog")
    if catalog.get("status") != "completed":
        raise ValueError("unified boundary catalog is not completed")
    if catalog.get("artifact_role") != "unlabeled_policy_conditioned_frontier_candidates":
        raise ValueError("unified boundary catalog artifact role drift")
    if catalog.get("split") != "train":
        raise ValueError("unified continuation labeling currently accepts expansion TRAIN only")
    if catalog.get("training_transitions") != 0:
        raise ValueError("unified boundary acquisition unexpectedly trained a policy")
    if catalog.get("expert_switching_used") is not False:
        raise ValueError("unified boundary acquisition used expert switching")
    if catalog.get("test_data_used") is not False:
        raise ValueError("unified boundary acquisition used TEST data")
    if catalog.get("validation_data_used") is not False:
        raise ValueError("unified boundary acquisition used validation data")
    if catalog.get("final_evaluation_data_used") is not False:
        raise ValueError("unified boundary acquisition used final evaluation data")
    if catalog.get("frozen_unified_manifest_sha256") != frozen_manifest_sha256:
        raise ValueError("unified boundary catalog frozen-policy manifest mismatch")
    if int(catalog.get("iteration", -1)) != int(policy_record["iteration"]):
        raise ValueError("unified boundary catalog policy iteration mismatch")
    if catalog.get("policy_name") != policy_record["name"]:
        raise ValueError("unified boundary catalog policy name mismatch")
    if catalog.get("policy_actor_sha256") != policy_record["actor_sha256"]:
        raise ValueError("unified boundary catalog actor SHA-256 mismatch")
    if catalog.get("policy_payload_sha256") != policy_record["payload_sha256"]:
        raise ValueError("unified boundary catalog checkpoint payload mismatch")
    claims = catalog.get("claim_boundary", {})
    if claims != {
        "unlabeled_acquisition_only": True,
        "tube_expansion_claim": False,
        "jce_jel_claim": False,
        "certified_safe_set_claim": False,
    }:
        raise ValueError("unified boundary catalog claim boundary drift")

    rows = tuple(catalog.get("entries", ()))
    if not rows:
        raise ValueError("unified boundary catalog has no candidates")
    if int(catalog.get("candidate_count", -1)) != len(rows):
        raise ValueError("unified boundary catalog candidate count mismatch")

    required = {
        "candidate_id",
        "candidate_kind",
        "split",
        "phase",
        "phase_index",
        "snapshot",
        "source_bank",
        "state_sha256",
        "parent_group_id",
        "parent_state_sha256",
        "policy_iteration",
        "policy_actor_sha256",
        "policy_payload_sha256",
        "protocol_sha256",
    }
    seen_ids: set[str] = set()
    seen_states: set[str] = set()
    for row in rows:
        missing = required.difference(row)
        if missing:
            raise ValueError(f"unified boundary candidate missing fields: {sorted(missing)}")
        if row["candidate_kind"] != "reachable_unified_frontier_probe":
            raise ValueError("unsupported unified boundary candidate kind")
        if row["split"] != "train":
            raise ValueError("unified boundary catalog contains non-TRAIN candidate")
        phase = str(row["phase"])
        phase_index = int(row["phase_index"])
        if (phase, phase_index) not in (("upstream", 0), ("downstream", 1)):
            raise ValueError("unified boundary candidate phase identity mismatch")
        if int(row["policy_iteration"]) != int(policy_record["iteration"]):
            raise ValueError("candidate policy iteration mismatch")
        if row["policy_actor_sha256"] != policy_record["actor_sha256"]:
            raise ValueError("candidate actor SHA-256 mismatch")
        if row["policy_payload_sha256"] != policy_record["payload_sha256"]:
            raise ValueError("candidate checkpoint payload mismatch")
        if row["protocol_sha256"] != catalog.get("protocol_sha256"):
            raise ValueError("candidate acquisition protocol mismatch")
        candidate_id = str(row["candidate_id"])
        state_sha = str(row["state_sha256"])
        if candidate_id in seen_ids:
            raise ValueError("duplicate unified boundary candidate_id")
        if state_sha in seen_states:
            raise ValueError("duplicate unified boundary physical state")
        seen_ids.add(candidate_id)
        seen_states.add(state_sha)
    return rows


def validate_candidate_snapshot(
    snapshot: UnifiedEnvelopeSnapshot,
    row: Mapping[str, Any],
    *,
    policy_record: Mapping[str, Any],
) -> None:
    if "snapshot_context_sha256" in row and snapshot_context_sha256(snapshot) != row["snapshot_context_sha256"]:
        raise ValueError("candidate snapshot controller/event/time context drift")
    if physical_state_sha256(snapshot) != row["state_sha256"]:
        raise ValueError("candidate snapshot physical-state SHA-256 mismatch")
    if snapshot.parent_trajectory != row["parent_group_id"]:
        raise ValueError("candidate snapshot parent trajectory mismatch")
    if snapshot.parent_state_sha256 != row["parent_state_sha256"]:
        raise ValueError("candidate snapshot parent-state SHA-256 mismatch")
    if snapshot.policy_iteration != int(policy_record["iteration"]):
        raise ValueError("candidate snapshot policy iteration mismatch")
    if snapshot.policy_actor_sha256 != policy_record["actor_sha256"]:
        raise ValueError("candidate snapshot actor SHA-256 mismatch")
    if snapshot.policy_payload_sha256 != policy_record["payload_sha256"]:
        raise ValueError("candidate snapshot checkpoint payload mismatch")
    if snapshot.config_sha256 != policy_record["formal_config_sha256"]:
        raise ValueError("candidate snapshot formal-config SHA-256 mismatch")
    if snapshot.xml_sha256 != policy_record["xml_sha256"]:
        raise ValueError("candidate snapshot XML SHA-256 mismatch")
    if snapshot.active_phase != int(row["phase_index"]):
        raise ValueError("candidate snapshot active phase mismatch")
    if snapshot.phase_transitioned and "snapshot_context_sha256" not in row:
        raise ValueError("boundary acquisition candidate unexpectedly crossed phase")


def fresh_unified_continuation_start(snapshot: UnifiedEnvelopeSnapshot, env: Any) -> Any:
    """Restore candidate physics/history while starting a fresh continuation budget."""
    state = restore_unified_envelope_snapshot(snapshot, env)
    if _truth(state.done):
        raise ValueError("unified continuation candidate restored terminal")
    active_phase = jp.asarray(state.info["active_phase"], jp.int32)
    up_events = state.info["up_events"].replace(episode_step=jp.asarray(0, jp.int32))
    info = {
        **state.info,
        "up_events": up_events,
        "start_phase": active_phase,
        "phase_transitioned": jp.asarray(False),
        "episode_step": jp.asarray(0, jp.int32),
        "phase_episode_step": jp.asarray(0, jp.int32),
        "episode_return": jp.asarray(0.0, jp.float32),
    }
    metrics = {
        **state.metrics,
        "event/tube_phase_transition": jp.asarray(0.0, jp.float32),
        "state/active_phase": active_phase.astype(jp.float32),
    }
    return state.replace(
        info=info,
        metrics=metrics,
        reward=jp.asarray(0.0, jp.float32),
        done=jp.asarray(0.0, jp.float32),
    )


def classify_unified_continuation_outcome(
    *,
    start_phase: int,
    terminal_success: bool,
    physical_failure: bool,
    timeout: bool,
    done: bool,
    apex_seen: bool,
    phase_transitioned: bool,
    recovery_success: bool,
    reached_rollout_horizon: bool,
) -> tuple[bool, str]:
    """Return strict continuation label and mutually exclusive outcome class."""
    if start_phase not in (PHASE_UPSTREAM, PHASE_DOWNSTREAM):
        raise ValueError("invalid unified continuation start phase")
    event_chain = recovery_success
    if start_phase == PHASE_UPSTREAM:
        event_chain = event_chain and apex_seen and phase_transitioned
    positive = bool(terminal_success and event_chain)
    if positive:
        return True, "success"
    if physical_failure:
        return False, "physical_failure"
    if timeout:
        return False, "timeout"
    if done:
        return False, "task_failure"
    if reached_rollout_horizon:
        return False, "horizon_exhausted"
    return False, "open_nonterminal"


def classify_first_valid_landing_outcome(
    *,
    valid_contact_seen: bool,
    physical_failure_before_landing: bool,
    timeout: bool,
    done: bool,
    reached_rollout_horizon: bool,
) -> tuple[bool, str]:
    """Classify jump continuation only through the first valid landing.

    Recovery after contact is intentionally outside this criterion.  The
    rollout must stop as soon as valid contact or an earlier terminal event is
    observed so a post-landing failure cannot rewrite a successful landing.
    """
    if valid_contact_seen:
        return True, "first_valid_landing"
    if physical_failure_before_landing:
        return False, "airborne_physical_failure"
    if timeout:
        return False, "timeout_before_landing"
    if done:
        return False, "task_failure_before_landing"
    if reached_rollout_horizon:
        return False, "horizon_exhausted_before_landing"
    return False, "open_nonterminal_before_landing"


def label_unified_continuations(
    catalog_path: Path,
    output_dir: Path,
    *,
    env: Any,
    policy: Callable[[Any, Any], Any],
    policy_record: Mapping[str, Any],
    frozen_manifest_sha256: str,
    max_ticks: int = DEFAULT_UNIFIED_CONTINUATION_MAX_TICKS,
    protocol_seed: int = DEFAULT_UNIFIED_CONTINUATION_PROTOCOL_SEED,
    compiled_step_fn: Callable[[Any, Any], Any] | None = None,
    acquisition_policy_record: Mapping[str, Any] | None = None,
    acquisition_frozen_manifest_sha256: str | None = None,
    success_criterion: str = "stable_recovery",
) -> dict[str, Any]:
    """Label each exact frontier candidate under one deterministic frozen pi_k."""
    max_ticks = int(max_ticks)
    if max_ticks <= 0:
        raise ValueError("unified continuation max_ticks must be positive")
    if int(policy_record.get("iteration", -1)) < 0:
        raise ValueError("frozen unified policy iteration is invalid")
    if policy_record.get("policy_role") != "envelope_expansion_authority":
        raise ValueError("frozen unified policy is not an expansion authority")
    if policy_record.get("xml_sha256") != env._bundle.xml_sha256:
        raise ValueError("unified continuation policy/runtime XML mismatch")
    if success_criterion not in {"stable_recovery", "first_valid_landing"}:
        raise ValueError("unsupported unified continuation success criterion")

    acquisition_record = (
        policy_record
        if acquisition_policy_record is None
        else acquisition_policy_record
    )
    acquisition_frozen_sha = (
        frozen_manifest_sha256
        if acquisition_frozen_manifest_sha256 is None
        else str(acquisition_frozen_manifest_sha256)
    )
    if acquisition_record.get("xml_sha256") != env._bundle.xml_sha256:
        raise ValueError("acquisition/evaluation policy XML mismatch")

    catalog_path = Path(catalog_path)
    catalog = _read_json(catalog_path)
    rows = validate_unified_boundary_catalog(
        catalog,
        policy_record=acquisition_record,
        frozen_manifest_sha256=acquisition_frozen_sha,
    )
    acquisition_protocol_path = catalog_path.parent / "protocol.json"
    acquisition_protocol = _read_json(acquisition_protocol_path)
    if acquisition_protocol.get("protocol_sha256") != catalog.get("protocol_sha256"):
        raise ValueError("unified acquisition protocol/catalog SHA mismatch")

    expected_horizon = int(env.resolved_config.ppo.episode_horizon)
    if max_ticks != expected_horizon:
        raise ValueError(
            "unified continuation labeling must use the source episode horizon "
            f"({expected_horizon})"
        )
    maximum_interactions = len(rows) * max_ticks
    protocol = {
        "schema": UNIFIED_CONTINUATION_LABEL_SCHEMA,
        "status": "predeclared",
        "purpose": "frozen_unified_policy_continuation_labels_for_envelope_expansion",
        "split": "train",
        "iteration": int(policy_record["iteration"]),
        "policy_name": str(policy_record["name"]),
        "policy_actor_sha256": str(policy_record["actor_sha256"]),
        "policy_payload_sha256": str(policy_record["payload_sha256"]),
        "policy_formal_config_sha256": str(policy_record["formal_config_sha256"]),
        "frozen_unified_manifest_sha256": str(frozen_manifest_sha256),
        "candidate_catalog": str(catalog_path),
        "candidate_catalog_file_sha256": file_sha256(catalog_path),
        "candidate_catalog_protocol_sha256": str(catalog["protocol_sha256"]),
        "candidate_count": len(rows),
        "branches_per_candidate": 1,
        "policy_mode": "deterministic",
        "success_criterion": success_criterion,
        "acquisition_policy_name": str(acquisition_record["name"]),
        "acquisition_policy_actor_sha256": str(acquisition_record["actor_sha256"]),
        "acquisition_policy_payload_sha256": str(acquisition_record["payload_sha256"]),
        "protocol_seed": int(protocol_seed),
        "max_ticks_per_candidate": max_ticks,
        "execution_mode": "single_gpu_serial_jitted_step_early_stop_v1",
        "maximum_environment_interactions": maximum_interactions,
        "candidate_start_semantics": (
            "preserve exact qpos/qvel/control, actor FIFO, last action, and phase event context; "
            "reset episode_step, phase_episode_step, up-event episode_step, phase-transition flag, "
            "and accumulated return; evaluate as a fresh continuation start"
        ),
        "positive_label_semantics": (
            {
                "all_phases": (
                    "the evaluator reaches the first valid landing before physical failure; "
                    "post-landing recovery and post-landing failure are outside the label"
                ),
                "alive_only_is_positive": False,
                "post_landing_recovery_required": False,
            }
            if success_criterion == "first_valid_landing"
            else {
                "upstream": (
                    "frozen pi_k reaches Apex phase transition and then downstream stable-recovery "
                    "terminal success within the declared horizon"
                ),
                "downstream": (
                    "frozen pi_k reaches downstream stable-recovery terminal success within the "
                    "declared horizon"
                ),
                "alive_only_is_positive": False,
            }
        ),
        "training_transitions": 0,
        "expert_switching_used": False,
        "test_data_used": False,
        "validation_data_used": False,
        "final_evaluation_data_used": False,
        "claim_boundary": {
            "expansion_train_labels_only": True,
            "continuation_field_trained": False,
            "tube_expansion_claim": False,
            "jce_jel_claim": False,
            "certified_safe_set_claim": False,
        },
    }
    protocol_sha = _canonical_sha256(protocol)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / "protocol.json").write_text(
        json.dumps(
            {**protocol, "protocol_sha256": protocol_sha},
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    step_fn = compiled_step_fn if compiled_step_fn is not None else jax.jit(env.step)
    base_key = jax.random.PRNGKey(int(protocol_seed))
    labeled: list[dict[str, Any]] = []
    outcome_counts: Counter[str] = Counter()
    phase_candidate_counts: Counter[str] = Counter()
    phase_positive_counts: Counter[str] = Counter()
    interactions = 0

    try:
        for candidate_index, row in enumerate(rows):
            snapshot_path = _candidate_snapshot_path(catalog_path, row)
            snapshot = load_unified_envelope_snapshot(snapshot_path)
            validate_candidate_snapshot(snapshot, row, policy_record=acquisition_record)
            state = fresh_unified_continuation_start(snapshot, env)
            if not _finite_state(state):
                raise ValueError("unified continuation candidate start is nonfinite")
            start_phase = _integer(state.info["active_phase"])
            if start_phase != int(row["phase_index"]):
                raise ValueError("unified continuation fresh-start phase mismatch")
            if _truth(state.info["expert_switching_used"]):
                raise ValueError("unified continuation start used expert switching")

            apex_seen = _truth(state.info["up_events"].apex_seen)
            phase_transitioned = _truth(state.info["phase_transitioned"])
            recovery_success = _truth(state.info["down_events"].recovery_success)
            valid_contact_seen = _truth(state.info["down_events"].valid_contact_seen)
            rollout_interactions = 0
            candidate_key = jax.random.fold_in(base_key, int(candidate_index))
            for tick in range(max_ticks):
                action_key = jax.random.fold_in(candidate_key, int(tick))
                result = policy(state.obs, action_key)
                action = result[0] if isinstance(result, tuple) else result
                action_array = np.asarray(jax.device_get(action), dtype=np.float32).reshape(-1)
                if action_array.shape != (4,) or not np.isfinite(action_array).all():
                    raise ValueError("frozen unified policy returned an invalid action")
                state = step_fn(state, action)
                jax.block_until_ready(state)
                interactions += 1
                rollout_interactions += 1
                if not _finite_state(state, action):
                    raise ValueError("unified continuation rollout became nonfinite")
                if _truth(state.info["expert_switching_used"]):
                    raise ValueError("unified continuation rollout used expert switching")
                apex_seen |= _truth(state.info["up_events"].apex_seen)
                phase_transitioned |= _truth(state.info["phase_transitioned"])
                recovery_success |= _truth(state.info["down_events"].recovery_success)
                valid_contact_seen |= _truth(state.info["down_events"].valid_contact_seen)
                if _truth(state.done) or (
                    success_criterion == "first_valid_landing" and valid_contact_seen
                ):
                    break

            done = _truth(state.done)
            terminal_success = _truth(state.info["success"])
            physical_failure = _truth(state.info["physical_failure"])
            timeout = _truth(state.info["timeout"])
            reached_horizon = rollout_interactions >= max_ticks and not done
            if success_criterion == "first_valid_landing":
                positive, outcome_class = classify_first_valid_landing_outcome(
                    valid_contact_seen=valid_contact_seen,
                    physical_failure_before_landing=(
                        physical_failure and not valid_contact_seen
                    ),
                    timeout=timeout,
                    done=done,
                    reached_rollout_horizon=reached_horizon,
                )
            else:
                positive, outcome_class = classify_unified_continuation_outcome(
                    start_phase=start_phase,
                    terminal_success=terminal_success,
                    physical_failure=physical_failure,
                    timeout=timeout,
                    done=done,
                    apex_seen=apex_seen,
                    phase_transitioned=phase_transitioned,
                    recovery_success=recovery_success,
                    reached_rollout_horizon=reached_horizon,
                )
            end_code = _integer(state.info["end_code"])
            end_reason = END_REASONS.get(end_code, f"unknown_{end_code}")
            phase_name = str(row["phase"])
            phase_candidate_counts[phase_name] += 1
            phase_positive_counts[phase_name] += int(positive)
            outcome_counts[outcome_class] += 1
            labeled.append(
                {
                    "candidate_id": str(row["candidate_id"]),
                    "candidate_kind": str(row["candidate_kind"]),
                    "split": "train",
                    "phase": phase_name,
                    "phase_index": start_phase,
                    "snapshot": str(row["snapshot"]),
                    "source_bank": str(row["source_bank"]),
                    "state_sha256": str(row["state_sha256"]),
                    **({"snapshot_context_sha256": row["snapshot_context_sha256"]}
                       if "snapshot_context_sha256" in row else {}),
                    "parent_group_id": str(row["parent_group_id"]),
                    "parent_state_sha256": str(row["parent_state_sha256"]),
                    "actor_observation": np.asarray(
                        snapshot.observation, dtype=np.float32
                    ).tolist(),
                    "label": int(positive),
                    "continuation_success": bool(positive),
                    "outcome_class": outcome_class,
                    "environment_interactions": rollout_interactions,
                    "terminal_done": done,
                    "terminal_success": terminal_success,
                    "physical_failure": physical_failure,
                    "timeout": timeout,
                    "end_code": end_code,
                    "end_reason": end_reason,
                    "apex_seen": bool(apex_seen),
                    "phase_transitioned": bool(phase_transitioned),
                    "valid_contact_seen": bool(valid_contact_seen),
                    "recovery_success": bool(recovery_success),
                    "final_active_phase": _integer(state.info["active_phase"]),
                    "policy_iteration": int(policy_record["iteration"]),
                    "policy_actor_sha256": str(policy_record["actor_sha256"]),
                    "policy_payload_sha256": str(policy_record["payload_sha256"]),
                    "evaluator_policy_name": str(policy_record["name"]),
                    "evaluator_actor_sha256": str(policy_record["actor_sha256"]),
                    "evaluator_payload_sha256": str(policy_record["payload_sha256"]),
                    "success_criterion": success_criterion,
                    "acquisition_protocol_sha256": str(catalog["protocol_sha256"]),
                    "label_protocol_sha256": protocol_sha,
                }
            )

        if interactions > maximum_interactions:
            raise ValueError("unified continuation labeling exceeded interaction ceiling")
        if len(labeled) != len(rows):
            raise ValueError("unified continuation label count did not close")
        if sum(outcome_counts.values()) != len(rows):
            raise ValueError("unified continuation outcome accounting did not close")
        report = {
            "schema": UNIFIED_CONTINUATION_LABEL_SCHEMA,
            "status": "completed",
            "artifact_role": "pi_k_conditioned_expansion_train_labels",
            "split": "train",
            "iteration": int(policy_record["iteration"]),
            "policy_name": str(policy_record["name"]),
            "policy_actor_sha256": str(policy_record["actor_sha256"]),
            "policy_payload_sha256": str(policy_record["payload_sha256"]),
            "evaluator_policy_name": str(policy_record["name"]),
            "success_criterion": success_criterion,
            "acquisition_policy_name": str(acquisition_record["name"]),
            "acquisition_policy_actor_sha256": str(acquisition_record["actor_sha256"]),
            "acquisition_policy_payload_sha256": str(acquisition_record["payload_sha256"]),
            "frozen_unified_manifest_sha256": str(frozen_manifest_sha256),
            "candidate_catalog_file_sha256": file_sha256(catalog_path),
            "candidate_catalog_protocol_sha256": str(catalog["protocol_sha256"]),
            "protocol_sha256": protocol_sha,
            "candidate_count": len(rows),
            "label_count": len(labeled),
            "positive_count": sum(int(row["label"]) for row in labeled),
            "negative_count": sum(1 - int(row["label"]) for row in labeled),
            "phase_candidate_counts": dict(sorted(phase_candidate_counts.items())),
            "phase_positive_counts": dict(sorted(phase_positive_counts.items())),
            "outcome_counts": dict(sorted(outcome_counts.items())),
            "environment_interactions": interactions,
            "maximum_environment_interactions": maximum_interactions,
            "training_transitions": 0,
            "expert_switching_used": False,
            "test_data_used": False,
            "validation_data_used": False,
            "final_evaluation_data_used": False,
            "claim_boundary": {
                "expansion_train_labels_only": True,
                "continuation_field_trained": False,
                "tube_expansion_claim": False,
                "jce_jel_claim": False,
                "certified_safe_set_claim": False,
            },
        }
        (output / "labels.json").write_text(
            json.dumps(labeled, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        report["labels_file_sha256"] = file_sha256(output / "labels.json")
        (output / "summary.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return report
    except BaseException as exc:
        failure = {
            "schema": UNIFIED_CONTINUATION_LABEL_SCHEMA,
            "status": "engineering_error",
            "iteration": int(policy_record["iteration"]),
            "policy_actor_sha256": str(policy_record["actor_sha256"]),
            "policy_payload_sha256": str(policy_record["payload_sha256"]),
            "protocol_sha256": protocol_sha,
            "completed_candidate_count": len(labeled),
            "environment_interactions": interactions,
            "maximum_environment_interactions": maximum_interactions,
            "training_transitions": 0,
            "expert_switching_used": False,
            "test_data_used": False,
            "validation_data_used": False,
            "final_evaluation_data_used": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
        (output / "summary.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        raise
