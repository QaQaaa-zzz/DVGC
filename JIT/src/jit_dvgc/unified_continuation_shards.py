"""Execution-only sharding for frozen unified continuation labeling.

Sharding changes only process lifetime. The logical candidate catalog, frozen
policy, continuation protocol, global candidate ordering, PRNG indexing, physics,
horizon, and labels remain identical to one serial run.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import jax
import numpy as np

from .evidence_integrity import validate_label_row
from .config import file_sha256
from .constants import END_REASONS
from .unified_continuation_labels import (
    UNIFIED_CONTINUATION_LABEL_SCHEMA,
    classify_first_valid_landing_outcome,
    classify_unified_continuation_outcome,
    fresh_unified_continuation_start,
    validate_candidate_snapshot,
    validate_unified_boundary_catalog,
)
from .unified_envelope_snapshot import load_unified_envelope_snapshot


SHARD_SCHEMA = "jit_unified_continuation_label_shard_v1"
MERGE_SCHEMA = "jit_unified_continuation_label_merge_v1"
POLICY_KEY_SCHEME = (
    "candidate_key=fold_in(PRNGKey(protocol_seed),global_candidate_index);"
    "action_key=fold_in(candidate_key,tick)"
)


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _read_json_list(path: Path) -> list[dict[str, Any]]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"JSON array required: {path}")
    if not all(isinstance(row, dict) for row in value):
        raise ValueError(f"JSON object rows required: {path}")
    return [dict(row) for row in value]


def _write_json(path: Path, value: Any) -> None:
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


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


def contiguous_shard_bounds(
    total_count: int, shard_index: int, shard_count: int
) -> tuple[int, int]:
    total_count = int(total_count)
    shard_index = int(shard_index)
    shard_count = int(shard_count)
    if total_count <= 0:
        raise ValueError("sharded labeling requires a positive candidate count")
    if shard_count <= 0 or shard_count > total_count:
        raise ValueError("shard_count must lie in [1, candidate_count]")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("shard_index must lie in [0, shard_count)")
    base, remainder = divmod(total_count, shard_count)
    start = shard_index * base + min(shard_index, remainder)
    stop = start + base + int(shard_index < remainder)
    return start, stop


def build_logical_label_protocol(
    *,
    catalog_path: Path,
    catalog: Mapping[str, Any],
    policy_record: Mapping[str, Any],
    frozen_manifest_sha256: str,
    max_ticks: int,
    protocol_seed: int,
    acquisition_policy_record: Mapping[str, Any] | None = None,
    acquisition_frozen_manifest_sha256: str | None = None,
    success_criterion: str = "stable_recovery",
) -> dict[str, Any]:
    if success_criterion not in {"stable_recovery", "first_valid_landing"}:
        raise ValueError("unsupported sharded continuation success criterion")
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
    count = int(catalog["candidate_count"])
    maximum_interactions = count * int(max_ticks)
    return {
        "schema": UNIFIED_CONTINUATION_LABEL_SCHEMA,
        "status": "predeclared",
        "purpose": "frozen_unified_policy_continuation_labels_for_envelope_expansion",
        "split": "train",
        "iteration": int(policy_record["iteration"]),
        "policy_name": str(policy_record["name"]),
        "policy_actor_sha256": str(policy_record["actor_sha256"]),
        "policy_payload_sha256": str(policy_record["payload_sha256"]),
        "evaluator_policy_name": str(policy_record["name"]),
        "policy_formal_config_sha256": str(policy_record["formal_config_sha256"]),
        "frozen_unified_manifest_sha256": str(frozen_manifest_sha256),
        "acquisition_policy_name": str(acquisition_record["name"]),
        "acquisition_policy_actor_sha256": str(acquisition_record["actor_sha256"]),
        "acquisition_policy_payload_sha256": str(acquisition_record["payload_sha256"]),
        "acquisition_frozen_unified_manifest_sha256": acquisition_frozen_sha,
        "candidate_catalog": str(Path(catalog_path)),
        "candidate_catalog_file_sha256": file_sha256(Path(catalog_path)),
        "candidate_catalog_protocol_sha256": str(catalog["protocol_sha256"]),
        "candidate_count": count,
        "branches_per_candidate": 1,
        "policy_mode": "deterministic",
        "success_criterion": success_criterion,
        "post_landing_recovery_required": success_criterion != "first_valid_landing",
        "protocol_seed": int(protocol_seed),
        "max_ticks_per_candidate": int(max_ticks),
        "maximum_environment_interactions": maximum_interactions,
        "candidate_start_semantics": (
            "preserve exact qpos/qvel/control, actor FIFO, last action, and phase event context; "
            "reset episode_step, phase_episode_step, up-event episode_step, phase-transition flag, "
            "and accumulated return; evaluate as a fresh continuation start"
        ),
        "positive_label_semantics": (
            {
                "all_phases": "first valid landing before physical failure",
                "post_landing_recovery_required": False,
                "alive_only_is_positive": False,
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


def _candidate_snapshot_path(catalog_path: Path, row: Mapping[str, Any]) -> Path:
    return Path(catalog_path).parent / str(row["source_bank"]) / str(row["snapshot"])


def label_unified_continuation_shard(
    catalog_path: Path,
    output_dir: Path,
    *,
    env: Any,
    policy: Callable[[Any, Any], Any],
    policy_record: Mapping[str, Any],
    frozen_manifest_sha256: str,
    shard_index: int,
    shard_count: int,
    max_ticks: int,
    protocol_seed: int,
    compiled_step_fn: Callable[[Any, Any], Any] | None = None,
    acquisition_policy_record: Mapping[str, Any] | None = None,
    acquisition_frozen_manifest_sha256: str | None = None,
    success_criterion: str = "stable_recovery",
) -> dict[str, Any]:
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
    catalog_path = Path(catalog_path)
    catalog = _read_json_object(catalog_path)
    rows = validate_unified_boundary_catalog(
        catalog,
        policy_record=acquisition_record,
        frozen_manifest_sha256=acquisition_frozen_sha,
    )
    acquisition_protocol = _read_json_object(catalog_path.parent / "protocol.json")
    if acquisition_protocol.get("protocol_sha256") != catalog.get("protocol_sha256"):
        raise ValueError("unified acquisition protocol/catalog SHA mismatch")

    max_ticks = int(max_ticks)
    expected_horizon = int(env.resolved_config.ppo.episode_horizon)
    if max_ticks != expected_horizon:
        raise ValueError(
            "sharded continuation labeling must use the frozen source episode horizon"
        )
    if policy_record.get("xml_sha256") != env._bundle.xml_sha256:
        raise ValueError("sharded continuation policy/runtime XML mismatch")

    start, stop = contiguous_shard_bounds(len(rows), shard_index, shard_count)
    protocol = build_logical_label_protocol(
        catalog_path=catalog_path,
        catalog=catalog,
        policy_record=policy_record,
        frozen_manifest_sha256=frozen_manifest_sha256,
        max_ticks=max_ticks,
        protocol_seed=protocol_seed,
        acquisition_policy_record=acquisition_record,
        acquisition_frozen_manifest_sha256=acquisition_frozen_sha,
        success_criterion=success_criterion,
    )
    protocol_sha = _canonical_sha256(protocol)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "protocol.json", {**protocol, "protocol_sha256": protocol_sha})
    execution = {
        "schema": SHARD_SCHEMA,
        "status": "predeclared",
        "logical_protocol_sha256": protocol_sha,
        "shard_index": int(shard_index),
        "shard_count": int(shard_count),
        "candidate_start_index": start,
        "candidate_stop_index_exclusive": stop,
        "selected_candidate_count": stop - start,
        "logical_candidate_count": len(rows),
        "policy_key_scheme": POLICY_KEY_SCHEME,
        "maximum_environment_interactions": (stop - start) * max_ticks,
    }
    _write_json(output / "execution.json", execution)

    step_fn = compiled_step_fn if compiled_step_fn is not None else jax.jit(env.step)
    base_key = jax.random.PRNGKey(int(protocol_seed))
    labeled: list[dict[str, Any]] = []
    outcome_counts: Counter[str] = Counter()
    phase_candidate_counts: Counter[str] = Counter()
    phase_positive_counts: Counter[str] = Counter()
    interactions = 0
    maximum_interactions = (stop - start) * max_ticks

    try:
        for candidate_index in range(start, stop):
            row = rows[candidate_index]
            snapshot_path = _candidate_snapshot_path(catalog_path, row)
            snapshot = load_unified_envelope_snapshot(snapshot_path)
            validate_candidate_snapshot(snapshot, row, policy_record=acquisition_record)
            state = fresh_unified_continuation_start(snapshot, env)
            if not _finite_state(state):
                raise ValueError("sharded continuation candidate start is nonfinite")
            start_phase = _integer(state.info["active_phase"])
            if start_phase != int(row["phase_index"]):
                raise ValueError("sharded continuation fresh-start phase mismatch")
            if _truth(state.info["expert_switching_used"]):
                raise ValueError("sharded continuation start used expert switching")

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
                action_array = np.asarray(
                    jax.device_get(action), dtype=np.float32
                ).reshape(-1)
                if action_array.shape != (4,) or not np.isfinite(action_array).all():
                    raise ValueError("frozen unified policy returned an invalid action")
                state = step_fn(state, action)
                jax.block_until_ready(state)
                interactions += 1
                rollout_interactions += 1
                if not _finite_state(state, action):
                    raise ValueError("sharded continuation rollout became nonfinite")
                if _truth(state.info["expert_switching_used"]):
                    raise ValueError("sharded continuation rollout used expert switching")
                apex_seen |= _truth(state.info["up_events"].apex_seen)
                phase_transitioned |= _truth(state.info["phase_transitioned"])
                recovery_success |= _truth(state.info["down_events"].recovery_success)
                valid_contact_seen |= _truth(state.info["down_events"].valid_contact_seen)
                if success_criterion == "first_valid_landing" and valid_contact_seen:
                    break
                if _truth(state.done):
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
                    "candidate_index": int(candidate_index),
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
                    "label_protocol_seed": int(protocol_seed),
                    "policy_key_candidate_index": int(candidate_index),
                    "policy_key_scheme": POLICY_KEY_SCHEME,
                }
            )

        if interactions > maximum_interactions:
            raise ValueError("sharded continuation labeling exceeded shard ceiling")
        if len(labeled) != stop - start:
            raise ValueError("sharded continuation label count did not close")
        report = {
            "schema": SHARD_SCHEMA,
            "status": "completed_shard",
            "logical_protocol_sha256": protocol_sha,
            "shard_index": int(shard_index),
            "shard_count": int(shard_count),
            "candidate_start_index": start,
            "candidate_stop_index_exclusive": stop,
            "candidate_count": len(labeled),
            "logical_candidate_count": len(rows),
            "positive_count": sum(int(row["label"]) for row in labeled),
            "negative_count": sum(1 - int(row["label"]) for row in labeled),
            "phase_candidate_counts": dict(sorted(phase_candidate_counts.items())),
            "phase_positive_counts": dict(sorted(phase_positive_counts.items())),
            "outcome_counts": dict(sorted(outcome_counts.items())),
            "environment_interactions": interactions,
            "maximum_environment_interactions": maximum_interactions,
            "policy_key_scheme": POLICY_KEY_SCHEME,
            "training_transitions": 0,
            "expert_switching_used": False,
            "test_data_used": False,
            "validation_data_used": False,
            "final_evaluation_data_used": False,
        }
        _write_json(output / "labels.json", labeled)
        report["labels_file_sha256"] = file_sha256(output / "labels.json")
        _write_json(output / "summary.json", report)
        _write_json(output / "execution.json", {**execution, "status": "completed"})
        return report
    except BaseException as exc:
        failure = {
            "schema": SHARD_SCHEMA,
            "status": "engineering_error",
            "logical_protocol_sha256": protocol_sha,
            "shard_index": int(shard_index),
            "shard_count": int(shard_count),
            "candidate_start_index": start,
            "candidate_stop_index_exclusive": stop,
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
        _write_json(output / "summary.json", failure)
        raise


def merge_unified_continuation_shards(
    catalog_path: Path,
    shard_dirs: Sequence[Path],
    output_dir: Path,
) -> dict[str, Any]:
    catalog_path = Path(catalog_path)
    catalog = _read_json_object(catalog_path)
    catalog_rows = catalog.get("entries")
    if not isinstance(catalog_rows, list) or not catalog_rows:
        raise ValueError("merge catalog entries missing")
    logical_count = len(catalog_rows)
    if int(catalog.get("candidate_count", -1)) != logical_count:
        raise ValueError("merge catalog candidate count drift")
    shard_dirs = tuple(Path(path) for path in shard_dirs)
    if not shard_dirs:
        raise ValueError("merge requires at least one shard")

    protocols: list[dict[str, Any]] = []
    executions: list[dict[str, Any]] = []
    labels_by_index: dict[int, dict[str, Any]] = {}
    shard_audit: list[dict[str, Any]] = []

    for shard_dir in shard_dirs:
        protocol = _read_json_object(shard_dir / "protocol.json")
        execution = _read_json_object(shard_dir / "execution.json")
        summary = _read_json_object(shard_dir / "summary.json")
        labels = _read_json_list(shard_dir / "labels.json")
        if summary.get("labels_file_sha256") is not None and summary["labels_file_sha256"] != file_sha256(shard_dir / "labels.json"):
            raise ValueError("shard label file hash drift")
        protocol_sha = str(protocol.get("protocol_sha256", ""))
        protocol_base = {k: v for k, v in protocol.items() if k != "protocol_sha256"}
        if _canonical_sha256(protocol_base) != protocol_sha:
            raise ValueError(f"shard logical protocol self-hash drift: {shard_dir}")
        if execution.get("status") != "completed":
            raise ValueError(f"shard execution is not completed: {shard_dir}")
        if summary.get("status") != "completed_shard":
            raise ValueError(f"shard summary is not completed: {shard_dir}")
        if execution.get("logical_protocol_sha256") != protocol_sha:
            raise ValueError("shard execution/protocol identity drift")
        if summary.get("logical_protocol_sha256") != protocol_sha:
            raise ValueError("shard summary/protocol identity drift")
        if str(protocol.get("candidate_catalog_file_sha256")) != file_sha256(catalog_path):
            raise ValueError("shard catalog file identity drift")
        if int(protocol.get("candidate_count", -1)) != logical_count:
            raise ValueError("shard logical candidate count drift")

        start = int(execution["candidate_start_index"])
        stop = int(execution["candidate_stop_index_exclusive"])
        index = int(execution["shard_index"])
        count = int(execution["shard_count"])
        expected_start, expected_stop = contiguous_shard_bounds(
            logical_count, index, count
        )
        if (start, stop) != (expected_start, expected_stop):
            raise ValueError("shard execution range drift")
        if len(labels) != stop - start or int(summary["candidate_count"]) != len(labels):
            raise ValueError("shard label count drift")
        for row in labels:
            validate_label_row(row, name=protocol["policy_name"], actor=protocol["policy_actor_sha256"],
                               payload=protocol["policy_payload_sha256"], criterion=protocol["success_criterion"])
            if row["environment_interactions"] > protocol["max_ticks_per_candidate"]:
                raise ValueError("shard per-candidate horizon exceeded")
            candidate_index = int(row.get("candidate_index", -1))
            if candidate_index < start or candidate_index >= stop:
                raise ValueError("shard label lies outside declared range")
            if candidate_index in labels_by_index:
                raise ValueError("duplicate candidate index across shards")
            catalog_row = catalog_rows[candidate_index]
            for field in (
                "candidate_id",
                "state_sha256",
                "phase",
                "phase_index",
                "parent_group_id",
                "parent_state_sha256",
                "source_bank",
                "snapshot",
                "snapshot_context_sha256",
            ):
                if row.get(field) != catalog_row.get(field):
                    raise ValueError(f"shard label/catalog {field} drift")
            if int(row.get("policy_key_candidate_index", -1)) != candidate_index:
                raise ValueError("shard policy-key global candidate index drift")
            if row.get("policy_key_scheme") != POLICY_KEY_SCHEME:
                raise ValueError("shard policy-key scheme drift")
            if row.get("label_protocol_sha256") != protocol_sha:
                raise ValueError("shard label protocol identity drift")
            labels_by_index[candidate_index] = row

        protocols.append(protocol)
        executions.append(execution)
        shard_audit.append(
            {
                "shard_dir": str(shard_dir),
                "shard_index": index,
                "shard_count": count,
                "candidate_start_index": start,
                "candidate_stop_index_exclusive": stop,
                "candidate_count": len(labels),
                "labels_file_sha256": file_sha256(shard_dir / "labels.json"),
                "summary_file_sha256": file_sha256(shard_dir / "summary.json"),
            }
        )

    protocol_shas = {str(row["protocol_sha256"]) for row in protocols}
    if len(protocol_shas) != 1:
        raise ValueError("shards do not share one logical protocol")
    shard_counts = {int(row["shard_count"]) for row in executions}
    if len(shard_counts) != 1:
        raise ValueError("shard_count differs across shards")
    shard_count = next(iter(shard_counts))
    shard_indices = {int(row["shard_index"]) for row in executions}
    if shard_indices != set(range(shard_count)):
        raise ValueError("merge requires exactly one completed shard for every index")
    if len(labels_by_index) != logical_count:
        raise ValueError("merged shard coverage is incomplete")

    protocol = protocols[0]
    protocol_sha = str(protocol["protocol_sha256"])
    merged = [labels_by_index[index] for index in range(logical_count)]
    outcome_counts = Counter(str(row["outcome_class"]) for row in merged)
    phase_candidate_counts = Counter(str(row["phase"]) for row in merged)
    phase_positive_counts = Counter(
        str(row["phase"]) for row in merged if int(row["label"]) == 1
    )
    interactions = sum(int(row["environment_interactions"]) for row in merged)
    maximum_interactions = int(protocol["maximum_environment_interactions"])
    if interactions > maximum_interactions:
        raise ValueError("merged labeling exceeded logical interaction ceiling")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "protocol.json", protocol)
    _write_json(output / "labels.json", merged)
    report = {
        "schema": UNIFIED_CONTINUATION_LABEL_SCHEMA,
        "status": "completed",
        "artifact_role": "pi_k_conditioned_expansion_train_labels",
        "split": "train",
        "iteration": int(protocol["iteration"]),
        "policy_name": str(protocol["policy_name"]),
        "policy_actor_sha256": str(protocol["policy_actor_sha256"]),
        "policy_payload_sha256": str(protocol["policy_payload_sha256"]),
        "evaluator_policy_name": str(
            protocol.get("evaluator_policy_name", protocol["policy_name"])
        ),
        "success_criterion": str(
            protocol.get("success_criterion", "stable_recovery")
        ),
        "post_landing_recovery_required": bool(
            protocol.get("post_landing_recovery_required", True)
        ),
        "acquisition_policy_name": str(
            protocol.get("acquisition_policy_name", protocol["policy_name"])
        ),
        "acquisition_policy_actor_sha256": str(
            protocol.get(
                "acquisition_policy_actor_sha256",
                protocol["policy_actor_sha256"],
            )
        ),
        "acquisition_policy_payload_sha256": str(
            protocol.get(
                "acquisition_policy_payload_sha256",
                protocol["policy_payload_sha256"],
            )
        ),
        "frozen_unified_manifest_sha256": str(protocol["frozen_unified_manifest_sha256"]),
        "candidate_catalog_file_sha256": str(protocol["candidate_catalog_file_sha256"]),
        "candidate_catalog_protocol_sha256": str(
            protocol["candidate_catalog_protocol_sha256"]
        ),
        "protocol_sha256": protocol_sha,
        "candidate_count": logical_count,
        "label_count": logical_count,
        "positive_count": sum(int(row["label"]) for row in merged),
        "negative_count": sum(1 - int(row["label"]) for row in merged),
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
    report["labels_file_sha256"] = file_sha256(output / "labels.json")
    _write_json(output / "summary.json", report)
    merge_audit = {
        "schema": MERGE_SCHEMA,
        "status": "completed",
        "logical_protocol_sha256": protocol_sha,
        "policy_key_scheme": POLICY_KEY_SCHEME,
        "candidate_count": logical_count,
        "shard_count": shard_count,
        "coverage": [0, logical_count],
        "shards": sorted(shard_audit, key=lambda row: row["shard_index"]),
        "merged_labels_file_sha256": file_sha256(output / "labels.json"),
        "training_transitions": 0,
        "validation_data_used": False,
        "test_data_used": False,
        "final_evaluation_data_used": False,
    }
    _write_json(output / "merge_audit.json", merge_audit)
    return report
