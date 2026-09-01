"""Paired non-final policy gate for iterative empirical envelope expansion.

This capability compares two exact frozen unified policies on one locked bank.
It is an iteration-selection diagnostic, not final JCE/JEL evidence.  The bank
is fixed before either policy is evaluated by this runner:

* core states are the complete declared source-Tube core;
* boundary states are frozen TRAIN continuation-negative frontier states from
  the baseline policy and must not already be present in the target Tube.

The gate never trains, switches experts, consumes validation, or touches TEST.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import jax
import numpy as np

from ..checkpoint import CheckpointIdentity, load_checkpoint
from ..config import file_sha256, load_config
from ..iteration_train_evidence import canonical_sha256, load_frozen_iteration_train_evidence
from ..ppo import make_checkpoint_policy
from ..soft_tube import load_soft_tube
from ..unified_continuation_labels import (
    classify_unified_continuation_outcome,
    fresh_unified_continuation_start,
)
from ..unified_envelope_snapshot import (
    load_unified_envelope_snapshot,
    physical_state_sha256 as unified_state_sha256,
)
from ..unified_env import UnifiedTubeRSIEnv
from ..unified_formal import load_unified_formal_config
from ..unified_policy_freeze import load_frozen_unified_manifest


CONFIG_SCHEMA = "jit_paired_policy_gate_config_v1"
PROTOCOL_SCHEMA = "jit_paired_policy_gate_protocol_v1"
REPORT_SCHEMA = "jit_paired_policy_gate_report_v1"


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256_state(state: Any) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(jax.device_get(state.data.qpos)).tobytes())
    digest.update(np.asarray(jax.device_get(state.data.qvel)).tobytes())
    return digest.hexdigest()


def load_paired_policy_gate_config(path: Path) -> dict[str, Any]:
    config = _read_object(Path(path))
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("unsupported paired policy gate config schema")
    protocol = config.get("protocol")
    if not isinstance(protocol, Mapping) or protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("paired policy gate protocol schema drift")
    if protocol.get("status") != "predeclared_before_gate_execution":
        raise ValueError("paired policy gate protocol must be predeclared")
    source_iteration = int(protocol.get("source_iteration", -1))
    candidate_iteration = int(protocol.get("candidate_iteration", -1))
    if source_iteration < 0 or candidate_iteration != source_iteration + 1:
        raise ValueError("paired policy gate iteration order drift")
    policies = protocol.get("policies")
    if not isinstance(policies, Mapping) or set(policies) != {"baseline", "candidate"}:
        raise ValueError("paired policy gate requires baseline and candidate policies")
    for role, expected_iteration in (("baseline", source_iteration), ("candidate", candidate_iteration)):
        value = policies[role]
        if not isinstance(value, Mapping):
            raise ValueError(f"paired policy gate {role} policy declaration missing")
        if int(value.get("iteration", -1)) != expected_iteration:
            raise ValueError(f"paired policy gate {role} iteration drift")
        if value.get("name") != f"pi_{expected_iteration}":
            raise ValueError(f"paired policy gate {role} policy name drift")
        for field in ("actor_sha256", "payload_sha256"):
            text = str(value.get(field, ""))
            if len(text) != 64:
                raise ValueError(f"paired policy gate {role} {field} invalid")
        if not str(value.get("frozen_manifest", "")):
            raise ValueError(f"paired policy gate {role} frozen manifest missing")
    core = protocol.get("core")
    if not isinstance(core, Mapping):
        raise ValueError("paired policy gate core declaration missing")
    if core.get("selection") != "all_source_tube_entries":
        raise ValueError("paired policy gate core selection drift")
    if core.get("preservation_rule") != "zero_baseline_success_to_candidate_failure":
        raise ValueError("paired policy gate core preservation rule drift")
    if core.get("require_baseline_success_each_phase") is not True:
        raise ValueError("paired policy gate must reject vacuous core preservation")
    boundary = protocol.get("boundary")
    if not isinstance(boundary, Mapping):
        raise ValueError("paired policy gate boundary declaration missing")
    if boundary.get("selection") != "baseline_train_continuation_negative_only":
        raise ValueError("paired policy gate boundary selection drift")
    if boundary.get("require_baseline_negative_reproduction") is not True:
        raise ValueError("paired policy gate must reproduce baseline boundary failures")
    if int(boundary.get("minimum_candidate_success_parent_groups", 0)) <= 0:
        raise ValueError("paired policy gate boundary parent-group minimum invalid")
    roots = boundary.get("snapshot_search_roots")
    if not isinstance(roots, list) or not roots:
        raise ValueError("paired policy gate snapshot roots missing")
    runtime = protocol.get("runtime")
    if runtime != {
        "policy_mode": "deterministic",
        "max_ticks": 400,
        "protocol_seed": int(runtime.get("protocol_seed", -1)) if isinstance(runtime, Mapping) else -1,
    }:
        raise ValueError("paired policy gate runtime contract drift")
    if int(runtime["protocol_seed"]) < 0:
        raise ValueError("paired policy gate protocol seed invalid")
    data_policy = protocol.get("data_policy")
    if data_policy != {
        "split": "train_audit_only",
        "validation_data_used": False,
        "test_data_used": False,
        "final_evaluation_data_used": False,
        "training_transitions": 0,
        "expert_switching_used": False,
    }:
        raise ValueError("paired policy gate data policy drift")
    claims = protocol.get("claim_boundary")
    if claims != {
        "iteration_selection_gate_only": True,
        "empirical_envelope_expansion_claim_requires_both_gates": True,
        "jce_jel_claim": False,
        "certified_safe_set_claim": False,
    }:
        raise ValueError("paired policy gate claim boundary drift")
    if not str(config.get("output_dir", "")):
        raise ValueError("paired policy gate output_dir missing")
    if canonical_sha256(protocol) != config.get("expected_protocol_sha256"):
        raise ValueError("paired policy gate protocol SHA-256 drift")
    return config


def summarize_paired_gate_records(
    records: Sequence[Mapping[str, Any]],
    *,
    minimum_candidate_success_parent_groups: int,
    require_baseline_success_each_phase: bool = True,
) -> dict[str, Any]:
    rows = [dict(row) for row in records]
    core = [row for row in rows if row.get("bank_role") == "core"]
    boundary = [row for row in rows if row.get("bank_role") == "boundary"]
    if not core or not boundary:
        raise ValueError("paired policy gate requires non-empty core and boundary banks")
    phases = ("upstream", "downstream")
    core_phase_baseline_success = {
        phase: sum(bool(row["baseline_success"]) for row in core if row["phase"] == phase)
        for phase in phases
    }
    core_regressions = [
        row for row in core if bool(row["baseline_success"]) and not bool(row["candidate_success"])
    ]
    core_improvements = [
        row for row in core if not bool(row["baseline_success"]) and bool(row["candidate_success"])
    ]
    core_nonvacuous = all(core_phase_baseline_success[phase] > 0 for phase in phases)
    core_pass = len(core_regressions) == 0 and (
        core_nonvacuous if require_baseline_success_each_phase else True
    )

    boundary_reproduction_failures = [row for row in boundary if bool(row["baseline_success"])]
    candidate_boundary_successes = [row for row in boundary if bool(row["candidate_success"])]
    successful_parent_groups = {
        str(row["parent_group_id"]) for row in candidate_boundary_successes
    }
    boundary_pass = (
        not boundary_reproduction_failures
        and len(candidate_boundary_successes) > 0
        and len(successful_parent_groups) >= int(minimum_candidate_success_parent_groups)
    )

    def _phase_counts(source: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
        result: dict[str, dict[str, int]] = {}
        for phase in phases:
            phase_rows = [row for row in source if row["phase"] == phase]
            result[phase] = {
                "state_count": len(phase_rows),
                "baseline_success_count": sum(bool(row["baseline_success"]) for row in phase_rows),
                "candidate_success_count": sum(bool(row["candidate_success"]) for row in phase_rows),
                "improvement_count": sum(
                    (not bool(row["baseline_success"])) and bool(row["candidate_success"])
                    for row in phase_rows
                ),
                "regression_count": sum(
                    bool(row["baseline_success"]) and (not bool(row["candidate_success"]))
                    for row in phase_rows
                ),
            }
        return result

    return {
        "core": {
            "state_count": len(core),
            "baseline_success_count": sum(bool(row["baseline_success"]) for row in core),
            "candidate_success_count": sum(bool(row["candidate_success"]) for row in core),
            "regression_count": len(core_regressions),
            "improvement_count": len(core_improvements),
            "baseline_success_each_phase": core_phase_baseline_success,
            "nonvacuous": core_nonvacuous,
            "passed": core_pass,
            "phase_counts": _phase_counts(core),
        },
        "boundary": {
            "state_count": len(boundary),
            "baseline_reproduction_failure_count": len(boundary_reproduction_failures),
            "candidate_success_count": len(candidate_boundary_successes),
            "candidate_success_parent_group_count": len(successful_parent_groups),
            "minimum_candidate_success_parent_groups": int(minimum_candidate_success_parent_groups),
            "passed": boundary_pass,
            "phase_counts": _phase_counts(boundary),
        },
        "accepted": bool(core_pass and boundary_pass),
    }


def _load_policy(protocol: Mapping[str, Any], role: str) -> tuple[dict[str, Any], dict[str, Any]]:
    declaration = protocol["policies"][role]
    manifest_path = Path(str(declaration["frozen_manifest"]))
    manifest = load_frozen_unified_manifest(manifest_path)
    record = dict(manifest["policy"])
    for field in ("iteration", "name", "actor_sha256", "payload_sha256"):
        if record.get(field) != declaration[field]:
            raise ValueError(f"paired policy gate {role} frozen {field} drift")
    return manifest, record


def _build_runtime(
    protocol: Mapping[str, Any], baseline_record: Mapping[str, Any], candidate_record: Mapping[str, Any]
) -> tuple[Any, Any, Any]:
    baseline_config = load_unified_formal_config(Path(str(baseline_record["formal_config"])))
    candidate_config = load_unified_formal_config(Path(str(candidate_record["formal_config"])))
    for field in ("up_config_sha256", "down_config_sha256", "runtime_naccdmax"):
        if getattr(baseline_config, field) != getattr(candidate_config, field):
            raise ValueError(f"paired policy gate policy runtime {field} mismatch")
    if baseline_config.ppo.episode_horizon != candidate_config.ppo.episode_horizon:
        raise ValueError("paired policy gate policy horizon mismatch")
    if int(protocol["runtime"]["max_ticks"]) != baseline_config.ppo.episode_horizon:
        raise ValueError("paired policy gate runtime horizon drift")
    up_config = load_config(Path(baseline_config.up_config_path))
    down_config = load_config(Path(baseline_config.down_config_path))
    core_tube = load_soft_tube(Path(str(protocol["core"]["source_tube"])))
    if core_tube.manifest.get("manifest_sha256") != protocol["core"]["source_tube_manifest_sha256"]:
        raise ValueError("paired policy gate source core Tube identity drift")
    target_tube = load_soft_tube(Path(str(protocol["boundary"]["target_tube"])))
    if target_tube.manifest.get("manifest_sha256") != protocol["boundary"]["target_tube_manifest_sha256"]:
        raise ValueError("paired policy gate target Tube identity drift")
    env = UnifiedTubeRSIEnv(
        up_config,
        down_config,
        core_tube,
        runtime_naccdmax=baseline_config.runtime_naccdmax,
        natural_reset_probability=0.0,
    )
    if env._bundle.xml_sha256 != baseline_record["xml_sha256"] or env._bundle.xml_sha256 != candidate_record["xml_sha256"]:
        raise ValueError("paired policy gate XML identity mismatch")
    return env, core_tube, target_tube


def _checkpoint_policy(env: Any, record: Mapping[str, Any]):
    identity = CheckpointIdentity(
        config_sha256=str(record["formal_config_sha256"]),
        xml_sha256=str(record["xml_sha256"]),
        actor_frame_fields=tuple(record["actor_frame_fields"]),
        actor_task_fields=tuple(record["actor_task_fields"]),
        action_order=tuple(record["action_order"]),
    )
    payload = load_checkpoint(Path(str(record["checkpoint"])), expected=identity)
    if int(payload.training_transitions) != int(record["source_training_transitions"]):
        raise ValueError("paired policy gate checkpoint transition drift")
    return jax.jit(make_checkpoint_policy(env, payload, deterministic=True))


def _resolve_unified_snapshots(
    search_roots: Sequence[Path], target_states: set[str]
) -> dict[str, Path]:
    matches: dict[str, list[Path]] = {state: [] for state in target_states}
    for root in search_roots:
        if not root.is_dir():
            raise FileNotFoundError(f"paired policy gate snapshot root missing: {root}")
        for identity_path in root.rglob("identity.json"):
            try:
                identity = _read_object(identity_path)
            except (OSError, json.JSONDecodeError, ValueError):
                continue
            if identity.get("schema") != "jit_unified_envelope_snapshot_v1":
                continue
            state_sha = str(identity.get("physical_state_sha256", ""))
            if state_sha in matches:
                matches[state_sha].append(identity_path.parent.resolve())
    unresolved = sorted(state for state, paths in matches.items() if not paths)
    if unresolved:
        raise FileNotFoundError(
            f"paired policy gate cannot resolve {len(unresolved)} boundary snapshots; first={unresolved[0]}"
        )
    return {state: sorted(paths)[0] for state, paths in matches.items()}


def _lock_bank(
    protocol: Mapping[str, Any],
    core_tube: Any,
    target_tube: Any,
    baseline_record: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    core_rows: list[dict[str, Any]] = []
    phase_local = Counter()
    for global_index, row in enumerate(core_tube.entries):
        phase = str(row["phase"])
        if phase not in ("upstream", "downstream"):
            raise ValueError("paired policy gate core phase drift")
        core_rows.append(
            {
                "bank_role": "core",
                "phase": phase,
                "phase_index": 0 if phase == "upstream" else 1,
                "entry_index": int(phase_local[phase]),
                "global_index": global_index,
                "state_sha256": str(row["state_sha256"]),
                "parent_group_id": str(row.get("parent_group_id", f"core:{phase}:{global_index}")),
            }
        )
        phase_local[phase] += 1
    if len(core_rows) != len(core_tube.entries):
        raise ValueError("paired policy gate core bank count drift")

    evidence_root = Path(str(protocol["boundary"]["frozen_train_evidence"]))
    manifest, rows = load_frozen_iteration_train_evidence(evidence_root)
    if manifest.get("manifest_sha256") != protocol["boundary"]["frozen_train_manifest_sha256"]:
        raise ValueError("paired policy gate TRAIN evidence manifest drift")
    if int(manifest.get("iteration", -1)) != int(baseline_record["iteration"]):
        raise ValueError("paired policy gate TRAIN evidence iteration drift")
    if manifest.get("policy_actor_sha256") != baseline_record["actor_sha256"]:
        raise ValueError("paired policy gate TRAIN evidence actor drift")
    if manifest.get("policy_payload_sha256") != baseline_record["payload_sha256"]:
        raise ValueError("paired policy gate TRAIN evidence payload drift")
    selected = [dict(row) for row in rows if int(row.get("label", -1)) == 0]
    if not selected:
        raise ValueError("paired policy gate has no baseline-negative boundary states")
    selected_states = {str(row["state_sha256"]) for row in selected}
    target_states = {str(row["state_sha256"]) for row in target_tube.entries}
    overlap = sorted(selected_states.intersection(target_states))
    if overlap:
        raise ValueError("paired policy gate boundary bank contains a state already admitted to target Tube")
    resolved = _resolve_unified_snapshots(
        [Path(str(root)) for root in protocol["boundary"]["snapshot_search_roots"]],
        selected_states,
    )
    boundary_rows: list[dict[str, Any]] = []
    for row in sorted(selected, key=lambda value: (str(value["phase"]), str(value["parent_group_id"]), str(value["state_sha256"]))):
        state_sha = str(row["state_sha256"])
        snapshot_path = resolved[state_sha]
        snapshot = load_unified_envelope_snapshot(snapshot_path)
        if unified_state_sha256(snapshot) != state_sha:
            raise ValueError("paired policy gate boundary snapshot physical-state drift")
        if snapshot.policy_iteration != int(baseline_record["iteration"]):
            raise ValueError("paired policy gate boundary snapshot iteration drift")
        if snapshot.policy_actor_sha256 != baseline_record["actor_sha256"]:
            raise ValueError("paired policy gate boundary snapshot actor drift")
        if snapshot.policy_payload_sha256 != baseline_record["payload_sha256"]:
            raise ValueError("paired policy gate boundary snapshot payload drift")
        boundary_rows.append(
            {
                "bank_role": "boundary",
                "phase": str(row["phase"]),
                "phase_index": int(row["phase_index"]),
                "state_sha256": state_sha,
                "parent_group_id": str(row["parent_group_id"]),
                "source_label": 0,
                "snapshot": str(snapshot_path),
            }
        )
    if {row["phase"] for row in boundary_rows} != {"upstream", "downstream"}:
        raise ValueError("paired policy gate boundary bank must cover both phases")
    return core_rows, boundary_rows


def _rollout(
    env: Any,
    policy: Any,
    state: Any,
    *,
    step_fn: Any,
    start_phase: int,
    max_ticks: int,
    seed: int,
) -> dict[str, Any]:
    if bool(np.asarray(jax.device_get(state.done))):
        raise ValueError("paired policy gate start state is terminal")
    if bool(np.asarray(jax.device_get(state.info["expert_switching_used"]))):
        raise ValueError("paired policy gate start used expert switching")
    apex_seen = bool(np.asarray(jax.device_get(state.info["up_events"].apex_seen)))
    phase_transitioned = bool(np.asarray(jax.device_get(state.info["phase_transitioned"])))
    recovery_success = bool(np.asarray(jax.device_get(state.info["down_events"].recovery_success)))
    interactions = 0
    for tick in range(int(max_ticks)):
        key = jax.random.fold_in(jax.random.PRNGKey(int(seed)), int(tick))
        result = policy(state.obs, key)
        action = result[0] if isinstance(result, tuple) else result
        action_array = np.asarray(jax.device_get(action), dtype=np.float32).reshape(-1)
        if action_array.shape != (4,) or not np.isfinite(action_array).all():
            raise ValueError("paired policy gate policy returned invalid action")
        state = step_fn(state, action)
        jax.block_until_ready(state)
        interactions += 1
        if bool(np.asarray(jax.device_get(state.info["expert_switching_used"]))):
            raise ValueError("paired policy gate rollout used expert switching")
        apex_seen |= bool(np.asarray(jax.device_get(state.info["up_events"].apex_seen)))
        phase_transitioned |= bool(np.asarray(jax.device_get(state.info["phase_transitioned"])))
        recovery_success |= bool(np.asarray(jax.device_get(state.info["down_events"].recovery_success)))
        if bool(np.asarray(jax.device_get(state.done))):
            break
    done = bool(np.asarray(jax.device_get(state.done)))
    terminal_success = bool(np.asarray(jax.device_get(state.info["success"])))
    physical_failure = bool(np.asarray(jax.device_get(state.info["physical_failure"])))
    timeout = bool(np.asarray(jax.device_get(state.info["timeout"])))
    positive, outcome = classify_unified_continuation_outcome(
        start_phase=int(start_phase),
        terminal_success=terminal_success,
        physical_failure=physical_failure,
        timeout=timeout,
        done=done,
        apex_seen=apex_seen,
        phase_transitioned=phase_transitioned,
        recovery_success=recovery_success,
        reached_rollout_horizon=(interactions >= int(max_ticks) and not done),
    )
    return {
        "success": bool(positive),
        "outcome_class": str(outcome),
        "environment_interactions": interactions,
        "terminal_success": terminal_success,
        "physical_failure": physical_failure,
        "timeout": timeout,
        "apex_seen": apex_seen,
        "phase_transitioned": phase_transitioned,
        "recovery_success": recovery_success,
    }


def run_paired_policy_gate(config_path: Path) -> dict[str, Any]:
    """Execute a predeclared paired core-preservation/boundary-gain audit."""
    config_path = Path(config_path)
    config = load_paired_policy_gate_config(config_path)
    protocol = dict(config["protocol"])
    output = Path(str(config["output_dir"]))
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "protocol.json", {**protocol, "protocol_sha256": canonical_sha256(protocol)})
    interactions = 0
    try:
        if jax.default_backend() != "gpu":
            raise RuntimeError("paired policy gate requires the visible JAX GPU backend")
        _, baseline_record = _load_policy(protocol, "baseline")
        _, candidate_record = _load_policy(protocol, "candidate")
        if baseline_record["xml_sha256"] != candidate_record["xml_sha256"]:
            raise ValueError("paired policy gate frozen policy XML mismatch")
        env, core_tube, target_tube = _build_runtime(protocol, baseline_record, candidate_record)
        core_bank, boundary_bank = _lock_bank(protocol, core_tube, target_tube, baseline_record)
        bank = {
            "schema": "jit_paired_policy_gate_bank_v1",
            "status": "locked_before_policy_rollout",
            "source_iteration": int(protocol["source_iteration"]),
            "candidate_iteration": int(protocol["candidate_iteration"]),
            "core_count": len(core_bank),
            "boundary_count": len(boundary_bank),
            "core": core_bank,
            "boundary": boundary_bank,
            "validation_data_used": False,
            "test_data_used": False,
            "final_evaluation_data_used": False,
        }
        bank["bank_sha256"] = canonical_sha256(bank)
        _write_json(output / "bank.json", bank)

        baseline_policy = _checkpoint_policy(env, baseline_record)
        candidate_policy = _checkpoint_policy(env, candidate_record)
        max_ticks = int(protocol["runtime"]["max_ticks"])
        base_seed = int(protocol["runtime"]["protocol_seed"])
        reset_tube = jax.jit(env.reset_tube_index)
        step_fn = jax.jit(env.step)
        records: list[dict[str, Any]] = []

        for index, row in enumerate([*core_bank, *boundary_bank]):
            if row["bank_role"] == "core":
                baseline_state = reset_tube(np.int32(row["phase_index"]), np.int32(row["entry_index"]))
                candidate_state = reset_tube(np.int32(row["phase_index"]), np.int32(row["entry_index"]))
                if _sha256_state(baseline_state) != row["state_sha256"] or _sha256_state(candidate_state) != row["state_sha256"]:
                    raise ValueError("paired policy gate core reset physical-state drift")
            else:
                snapshot = load_unified_envelope_snapshot(Path(str(row["snapshot"])))
                baseline_state = fresh_unified_continuation_start(snapshot, env)
                candidate_state = fresh_unified_continuation_start(snapshot, env)
            state_seed = base_seed + index * 10_000
            baseline = _rollout(
                env,
                baseline_policy,
                baseline_state,
                step_fn=step_fn,
                start_phase=int(row["phase_index"]),
                max_ticks=max_ticks,
                seed=state_seed,
            )
            interactions += int(baseline["environment_interactions"])
            candidate = _rollout(
                env,
                candidate_policy,
                candidate_state,
                step_fn=step_fn,
                start_phase=int(row["phase_index"]),
                max_ticks=max_ticks,
                seed=state_seed,
            )
            interactions += int(candidate["environment_interactions"])
            records.append(
                {
                    **row,
                    "baseline_success": bool(baseline["success"]),
                    "candidate_success": bool(candidate["success"]),
                    "baseline_outcome_class": baseline["outcome_class"],
                    "candidate_outcome_class": candidate["outcome_class"],
                    "baseline_environment_interactions": baseline["environment_interactions"],
                    "candidate_environment_interactions": candidate["environment_interactions"],
                }
            )

        gates = summarize_paired_gate_records(
            records,
            minimum_candidate_success_parent_groups=int(
                protocol["boundary"]["minimum_candidate_success_parent_groups"]
            ),
            require_baseline_success_each_phase=bool(
                protocol["core"]["require_baseline_success_each_phase"]
            ),
        )
        report = {
            "schema": REPORT_SCHEMA,
            "status": "completed",
            "source_iteration": int(protocol["source_iteration"]),
            "candidate_iteration": int(protocol["candidate_iteration"]),
            "baseline_policy_name": str(baseline_record["name"]),
            "baseline_actor_sha256": str(baseline_record["actor_sha256"]),
            "baseline_payload_sha256": str(baseline_record["payload_sha256"]),
            "candidate_policy_name": str(candidate_record["name"]),
            "candidate_actor_sha256": str(candidate_record["actor_sha256"]),
            "candidate_payload_sha256": str(candidate_record["payload_sha256"]),
            "protocol_sha256": canonical_sha256(protocol),
            "bank_sha256": bank["bank_sha256"],
            "core_gate": gates["core"],
            "boundary_gate": gates["boundary"],
            "iteration_accepted": bool(gates["accepted"]),
            "empirical_envelope_expansion_accepted": bool(gates["accepted"]),
            "environment_interactions": interactions,
            "training_transitions": 0,
            "expert_switching_used": False,
            "validation_data_used": False,
            "test_data_used": False,
            "final_evaluation_data_used": False,
            "jce_jel_claim": False,
            "certified_safe_set_claim": False,
        }
        _write_json(output / "records.json", {"records": records})
        _write_json(output / "summary.json", report)
        return report
    except BaseException as exc:
        failure = {
            "schema": REPORT_SCHEMA,
            "status": "engineering_error",
            "protocol_sha256": canonical_sha256(protocol),
            "environment_interactions": interactions,
            "training_transitions": 0,
            "expert_switching_used": False,
            "validation_data_used": False,
            "test_data_used": False,
            "final_evaluation_data_used": False,
            "jce_jel_claim": False,
            "certified_safe_set_claim": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
        _write_json(output / "summary.json", failure)
        raise
