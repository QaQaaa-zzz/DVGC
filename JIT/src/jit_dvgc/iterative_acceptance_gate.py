"""Locked-baseline pi_k -> pi_(k+1) gate for automatic envelope iterations.

The baseline is fully committed before the candidate policy is trained:

* every source-Tube state is rolled out under selected pi_k with a fixed seed;
* every predeclared acceptance-role state already carries its pi_k continuation
  label, and only baseline-negative rows are locked for boundary gain;
* the exact acceptance policy-key seed/candidate index is retained.

After pi_(k+1) is frozen, the candidate is evaluated against this immutable
artifact.  The baseline is never rerun, eliminating the historical
"locked-negative but baseline reproduced positive" PRNG mismatch.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np

import jit_dvgc.analysis.paired_policy_gate as paired
from .config import file_sha256, load_config
from .iterative_frontier_protocol import ROLE_SCHEMA, canonical_sha256
from .soft_tube import load_soft_tube
from .unified_continuation_labels import (
    classify_first_valid_landing_outcome,
    classify_unified_continuation_outcome,
    fresh_unified_continuation_start,
)
from .unified_envelope_snapshot import load_unified_envelope_snapshot
from .unified_formal import load_unified_policy_formal_config
from .unified_policy_freeze import load_frozen_unified_manifest
from .unified_env import UnifiedTubeRSIEnv


LOCK_SCHEMA = "jit_iterative_acceptance_baseline_lock_v1"
REPORT_SCHEMA = "jit_paired_policy_gate_report_v1"
CORE_SEED_BASE = 9_524_001


def _read(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _verify_hash(payload: Mapping[str, Any], field: str) -> None:
    declared = str(payload.get(field, ""))
    base = {key: value for key, value in payload.items() if key != field}
    if len(declared) != 64 or canonical_sha256(base) != declared:
        raise ValueError(f"{field} self-hash drift")


def _acceptance_policy_success_criterion(
    manifest: Mapping[str, Any],
    selected: Mapping[str, Any],
) -> str:
    """Validate whether acceptance labels are single-policy or family member labels."""
    if manifest.get("policy_identity_kind") != "frozen_policy_family":
        if manifest.get("policy_actor_sha256") != selected.get("actor_sha256"):
            raise ValueError("acceptance role actor drift")
        if manifest.get("policy_payload_sha256") != selected.get("payload_sha256"):
            raise ValueError("acceptance role payload drift")
        return "stable_recovery"
    if manifest.get("continuation_success_criterion") != (
        "first_valid_landing_before_physical_failure"
    ):
        raise ValueError("acceptance policy-family criterion drift")
    family = manifest.get("continuation_policy_family")
    members = family.get("members") if isinstance(family, Mapping) else None
    if not isinstance(members, list):
        raise ValueError("acceptance policy family members missing")
    matches = [row for row in members if row.get("name") == selected.get("policy_name")]
    if len(matches) != 1:
        raise ValueError("selected policy is not exactly one acceptance family member")
    member = matches[0]
    if member.get("actor_sha256") != selected.get("actor_sha256"):
        raise ValueError("acceptance family selected actor drift")
    if member.get("payload_sha256") != selected.get("payload_sha256"):
        raise ValueError("acceptance family selected payload drift")
    return "first_valid_landing"


def _selected_family_outcome(
    row: Mapping[str, Any], policy_name: str
) -> dict[str, Any]:
    outcomes = row.get("per_policy_outcomes")
    outcome = outcomes.get(policy_name) if isinstance(outcomes, Mapping) else None
    if not isinstance(outcome, Mapping):
        raise ValueError("acceptance row lacks selected family policy outcome")
    return {
        "label": int(outcome["label"]),
        "outcome_class": str(outcome["outcome_class"]),
        "environment_interactions": int(outcome["environment_interactions"]),
    }


def _candidate_core_success_criterion(lock: Mapping[str, Any]) -> str:
    """Keep the candidate preservation metric aligned with the active method."""
    criterion = str(lock.get("acceptance_success_criterion", "stable_recovery"))
    if criterion not in {"stable_recovery", "first_valid_landing"}:
        raise ValueError("candidate core success criterion drift")
    return criterion


def _reuse_core_baseline(
    lock: Mapping[str, Any],
    *,
    selected: Mapping[str, Any],
    policy_record: Mapping[str, Any],
    source_tube_manifest_sha256: str,
    source_state_sha256: list[str],
) -> list[dict[str, Any]]:
    """Validate and reuse a previously locked baseline over the same source Tube."""
    if lock.get("schema") != LOCK_SCHEMA or lock.get("status") != (
        "locked_before_candidate_training"
    ):
        raise ValueError("reusable core baseline lock schema/status drift")
    if int(lock.get("source_iteration", -1)) != int(selected["iteration"]):
        raise ValueError("reusable core baseline iteration drift")
    if lock.get("baseline_actor_sha256") != policy_record.get("actor_sha256"):
        raise ValueError("reusable core baseline actor drift")
    if lock.get("baseline_payload_sha256") != policy_record.get("payload_sha256"):
        raise ValueError("reusable core baseline payload drift")
    if lock.get("source_tube_manifest_sha256") != source_tube_manifest_sha256:
        raise ValueError("reusable core baseline Tube drift")
    rows = lock.get("core")
    if not isinstance(rows, list) or len(rows) != int(lock.get("core_state_count", -1)):
        raise ValueError("reusable core baseline rows drift")
    if [str(row.get("state_sha256", "")) for row in rows] != list(
        source_state_sha256
    ):
        raise ValueError("reusable core baseline physical-state order drift")
    return [dict(row) for row in rows]


def _load_selected(path: Path):
    selected = _read(path)
    if not isinstance(selected, dict) or selected.get("schema") != "jit_selected_iteration_policy_v1":
        raise ValueError("acceptance gate requires selected iteration policy")
    if selected.get("status") != "selected" or selected.get("engineering_selection") is not True:
        raise ValueError("acceptance baseline policy is not selected")
    _verify_hash(selected, "selection_sha256")
    frozen_path = Path(str(selected["frozen_policy"]))
    if file_sha256(frozen_path) != selected["frozen_policy_file_sha256"]:
        raise ValueError("selected frozen policy file drift")
    frozen = load_frozen_unified_manifest(frozen_path)
    record = dict(frozen["policy"])
    for field in ("actor_sha256", "payload_sha256", "xml_sha256", "formal_config_sha256"):
        if record.get(field) != selected.get(field):
            raise ValueError(f"selected baseline {field} drift")
    return selected, record


def _load_acceptance_role(root: Path, selected: Mapping[str, Any], source_tube_sha: str):
    root = Path(root)
    manifest = _read(root / "role_manifest.json")
    if not isinstance(manifest, dict) or manifest.get("schema") != ROLE_SCHEMA:
        raise ValueError("acceptance role manifest schema drift")
    if manifest.get("status") != "completed" or manifest.get("role") != "acceptance":
        raise ValueError("acceptance role not completed")
    _verify_hash(manifest, "role_manifest_sha256")
    if int(manifest["iteration"]) != int(selected["iteration"]):
        raise ValueError("acceptance role iteration drift")
    success_criterion = _acceptance_policy_success_criterion(manifest, selected)
    if manifest["source_tube_manifest_sha256"] != source_tube_sha:
        raise ValueError("acceptance role source Tube drift")
    labels_path = root / "logical_labels.json"
    if file_sha256(labels_path) != manifest["logical_labels_file_sha256"]:
        raise ValueError("acceptance logical labels file drift")
    payload = _read(labels_path)
    _verify_hash(payload, "labels_sha256")
    rows = payload.get("entries")
    if not isinstance(rows, list) or not rows:
        raise ValueError("acceptance logical labels are empty")
    if any(row.get("split") != "acceptance" for row in rows):
        raise ValueError("acceptance logical split drift")
    plan = _read(Path(str(manifest["plan"])))
    _verify_hash(plan, "plan_sha256")
    if plan["plan_sha256"] != manifest["plan_sha256"]:
        raise ValueError("acceptance plan identity drift")
    labeling_seed = int(plan["seeds"]["acceptance"]["labeling"])
    return manifest, tuple(dict(row) for row in rows), labeling_seed, success_criterion


def _runtime(record: Mapping[str, Any], source_tube: Path):
    formal = load_unified_policy_formal_config(Path(str(record["formal_config"])))
    if formal.config_sha256 != record["formal_config_sha256"]:
        raise ValueError("acceptance baseline formal config drift")
    artifact = load_soft_tube(Path(source_tube))
    if artifact.manifest["manifest_sha256"] != formal.soft_tube_manifest_sha256:
        raise ValueError("selected baseline/source Tube identity drift")
    up = load_config(Path(formal.up_config_path))
    down = load_config(Path(formal.down_config_path))
    env = UnifiedTubeRSIEnv(
        up,
        down,
        artifact,
        runtime_naccdmax=formal.runtime_naccdmax,
        natural_reset_probability=0.0,
    )
    if env._bundle.xml_sha256 != record["xml_sha256"]:
        raise ValueError("acceptance runtime XML drift")
    return formal, artifact, env


def _make_compiled_rollout(
    env, policy, max_ticks: int, *, success_criterion: str = "stable_recovery"
):
    max_ticks = int(max_ticks)
    if success_criterion not in {"stable_recovery", "first_valid_landing"}:
        raise ValueError("unsupported acceptance rollout success criterion")
    stop_at_landing = success_criterion == "first_valid_landing"

    @jax.jit
    def rollout(initial_state, base_key):
        apex_seen = jnp.asarray(initial_state.info["up_events"].apex_seen, dtype=bool)
        phase_transitioned = jnp.asarray(initial_state.info["phase_transitioned"], dtype=bool)
        recovery_success = jnp.asarray(initial_state.info["down_events"].recovery_success, dtype=bool)
        valid_contact_seen = jnp.asarray(
            initial_state.info["down_events"].valid_contact_seen, dtype=bool
        )
        expert_switching = jnp.asarray(initial_state.info["expert_switching_used"], dtype=bool)
        carry = (
            jnp.asarray(0, jnp.int32),
            initial_state,
            apex_seen,
            phase_transitioned,
            recovery_success,
            valid_contact_seen,
            expert_switching,
        )

        def cond(value):
            tick, state, *_ = value
            active = jnp.logical_and(
                tick < max_ticks, jnp.logical_not(jnp.asarray(state.done, dtype=bool))
            )
            if stop_at_landing:
                active = jnp.logical_and(active, jnp.logical_not(value[5]))
            return active

        def body(value):
            tick, state, apex, transitioned, recovered, landed, switched = value
            key = jax.random.fold_in(base_key, tick)
            result = policy(state.obs, key)
            action = result[0] if isinstance(result, tuple) else result
            state = env.step(state, action)
            return (
                tick + 1,
                state,
                jnp.logical_or(apex, jnp.asarray(state.info["up_events"].apex_seen, dtype=bool)),
                jnp.logical_or(transitioned, jnp.asarray(state.info["phase_transitioned"], dtype=bool)),
                jnp.logical_or(recovered, jnp.asarray(state.info["down_events"].recovery_success, dtype=bool)),
                jnp.logical_or(
                    landed,
                    jnp.asarray(state.info["down_events"].valid_contact_seen, dtype=bool),
                ),
                jnp.logical_or(switched, jnp.asarray(state.info["expert_switching_used"], dtype=bool)),
            )

        tick, state, apex, transitioned, recovered, landed, switched = jax.lax.while_loop(
            cond, body, carry
        )
        return (
            tick,
            state.done,
            state.info["success"],
            state.info["physical_failure"],
            state.info["timeout"],
            apex,
            transitioned,
            recovered,
            switched,
            landed,
        )

    return rollout


def _outcome(
    output,
    *,
    start_phase: int,
    max_ticks: int,
    success_criterion: str = "stable_recovery",
) -> dict[str, Any]:
    values = jax.device_get(output)
    tick = int(np.asarray(values[0]))
    done = bool(np.asarray(values[1]))
    terminal_success = bool(np.asarray(values[2]))
    physical_failure = bool(np.asarray(values[3]))
    timeout = bool(np.asarray(values[4]))
    apex = bool(np.asarray(values[5]))
    transitioned = bool(np.asarray(values[6]))
    recovered = bool(np.asarray(values[7]))
    switched = bool(np.asarray(values[8]))
    valid_contact_seen = bool(np.asarray(values[9]))
    if switched:
        raise ValueError("acceptance rollout used expert switching")
    if success_criterion == "first_valid_landing":
        positive, outcome = classify_first_valid_landing_outcome(
            valid_contact_seen=valid_contact_seen,
            physical_failure_before_landing=(
                physical_failure and not valid_contact_seen
            ),
            timeout=timeout,
            done=done,
            reached_rollout_horizon=(tick >= int(max_ticks) and not done),
        )
    elif success_criterion == "stable_recovery":
        positive, outcome = classify_unified_continuation_outcome(
            start_phase=int(start_phase),
            terminal_success=terminal_success,
            physical_failure=physical_failure,
            timeout=timeout,
            done=done,
            apex_seen=apex,
            phase_transitioned=transitioned,
            recovery_success=recovered,
            reached_rollout_horizon=(tick >= int(max_ticks) and not done),
        )
    else:
        raise ValueError("unsupported acceptance outcome success criterion")
    return {
        "success": bool(positive),
        "outcome_class": str(outcome),
        "environment_interactions": tick,
    }


def lock_baseline(
    *,
    selected_policy: Path,
    source_tube: Path,
    acceptance_root: Path,
    output_dir: Path,
    reusable_core_baseline: Path | None = None,
):
    output = Path(output_dir)
    if output.exists():
        lock_path = output / "baseline_lock.json"
        if lock_path.is_file():
            existing = _read(lock_path)
            _verify_hash(existing, "lock_sha256")
            return existing
        raise FileExistsError(f"incomplete acceptance lock output exists: {output}")

    selected, record = _load_selected(Path(selected_policy))
    formal, artifact, env = _runtime(record, Path(source_tube))
    acceptance_manifest, acceptance_rows, labeling_seed, acceptance_criterion = _load_acceptance_role(
        Path(acceptance_root), selected, artifact.manifest["manifest_sha256"]
    )
    if acceptance_criterion == "first_valid_landing":
        selected_outcomes = {
            str(row["candidate_id"]): _selected_family_outcome(
                row, str(selected["policy_name"])
            )
            for row in acceptance_rows
        }
        boundary_negative = [
            row
            for row in acceptance_rows
            if selected_outcomes[str(row["candidate_id"])]["label"] == 0
        ]
    else:
        selected_outcomes = {}
        boundary_negative = [row for row in acceptance_rows if int(row["label"]) == 0]
    if not boundary_negative:
        raise ValueError("acceptance role has no baseline-negative states")
    negative_groups = {str(row["parent_group_id"]) for row in boundary_negative}
    phase_negative = Counter(str(row["phase"]) for row in boundary_negative)
    if acceptance_criterion == "stable_recovery" and any(
        phase_negative[p] <= 0 for p in ("upstream", "downstream")
    ):
        raise ValueError("acceptance role must contain a negative in both phases")
    if len(negative_groups) < 2:
        raise ValueError("acceptance baseline needs negatives from at least two parent groups")

    interactions = 0
    reused_core_lock = None
    if reusable_core_baseline is not None:
        reusable_path = Path(reusable_core_baseline)
        reused_core_lock = _read(reusable_path)
        _verify_hash(reused_core_lock, "lock_sha256")
        core_rows = _reuse_core_baseline(
            reused_core_lock,
            selected=selected,
            policy_record=record,
            source_tube_manifest_sha256=str(artifact.manifest["manifest_sha256"]),
            source_state_sha256=[str(row["state_sha256"]) for row in artifact.entries],
        )
    else:
        baseline_policy = paired._checkpoint_policy(env, record)
        rollout = _make_compiled_rollout(env, baseline_policy, formal.ppo.episode_horizon)
        reset = jax.jit(env.reset_tube_index)
        phase_local = Counter()
        core_rows = []
        for global_index, row in enumerate(artifact.entries):
            phase = str(row["phase"])
            phase_index = 0 if phase == "upstream" else 1
            local = int(phase_local[phase])
            phase_local[phase] += 1
            state = reset(np.int32(phase_index), np.int32(local))
            if paired._sha256_state(state) != row["state_sha256"]:
                raise ValueError("acceptance core reset physical-state drift")
            seed = CORE_SEED_BASE + global_index * 10_000
            result = _outcome(
                rollout(state, jax.random.PRNGKey(seed)),
                start_phase=phase_index,
                max_ticks=formal.ppo.episode_horizon,
            )
            interactions += int(result["environment_interactions"])
            core_rows.append(
                {
                    "bank_role": "core",
                    "phase": phase,
                    "phase_index": phase_index,
                    "entry_index": local,
                    "global_index": global_index,
                    "state_sha256": str(row["state_sha256"]),
                    "parent_group_id": str(
                        row.get("parent_group_id", f"core:{global_index}")
                    ),
                    "baseline_seed": seed,
                    "baseline_success": bool(result["success"]),
                    "baseline_outcome_class": result["outcome_class"],
                    "baseline_environment_interactions": int(
                        result["environment_interactions"]
                    ),
                }
            )

    boundary_rows = []
    catalog_path = Path(str(acceptance_manifest["source_acquisition_catalog"]))
    snapshot_root = catalog_path.parent
    for candidate_index, row in enumerate(acceptance_rows):
        baseline_outcome = (
            selected_outcomes[str(row["candidate_id"])]
            if acceptance_criterion == "first_valid_landing"
            else {
                "label": int(row["label"]),
                "outcome_class": str(row["outcome_class"]),
                "environment_interactions": int(row["environment_interactions"]),
            }
        )
        if int(baseline_outcome["label"]) != 0:
            continue
        snapshot_path = snapshot_root / str(row["source_bank"]) / str(row["snapshot"])
        if not snapshot_path.is_dir():
            raise FileNotFoundError(f"acceptance snapshot missing: {snapshot_path}")
        boundary_rows.append(
            {
                "bank_role": "boundary",
                "phase": str(row["phase"]),
                "phase_index": int(row["phase_index"]),
                "candidate_id": str(row["candidate_id"]),
                "state_sha256": str(row["state_sha256"]),
                "parent_group_id": str(row["parent_group_id"]),
                "snapshot": str(snapshot_path.resolve()),
                "baseline_label": 0,
                "baseline_success": False,
                "baseline_outcome_class": str(baseline_outcome["outcome_class"]),
                "baseline_environment_interactions": int(
                    baseline_outcome["environment_interactions"]
                ),
                "success_criterion": acceptance_criterion,
                "baseline_policy_key_scheme": "candidate_key=fold_in(PRNGKey(labeling_seed),candidate_index); action_key=fold_in(candidate_key,tick)",
                "baseline_labeling_seed": labeling_seed,
                "baseline_candidate_index": candidate_index,
            }
        )

    baseline_success = sum(bool(row["baseline_success"]) for row in core_rows)
    lock = {
        "schema": LOCK_SCHEMA,
        "status": "locked_before_candidate_training",
        "source_iteration": int(selected["iteration"]),
        "baseline_policy_name": str(selected["policy_name"]),
        "selected_policy": str(selected_policy),
        "selected_policy_sha256": str(selected["selection_sha256"]),
        "baseline_actor_sha256": str(record["actor_sha256"]),
        "baseline_payload_sha256": str(record["payload_sha256"]),
        "baseline_xml_sha256": str(record["xml_sha256"]),
        "source_tube": str(source_tube),
        "source_tube_manifest_sha256": str(artifact.manifest["manifest_sha256"]),
        "source_tube_entry_count": len(artifact.entries),
        "acceptance_role_manifest_sha256": str(acceptance_manifest["role_manifest_sha256"]),
        "acceptance_success_criterion": acceptance_criterion,
        "core_state_count": len(core_rows),
        "core_baseline_success_count": baseline_success,
        "core_baseline_reused": reused_core_lock is not None,
        "reusable_core_baseline": (
            None if reusable_core_baseline is None else str(reusable_core_baseline)
        ),
        "boundary_negative_state_count": len(boundary_rows),
        "boundary_negative_parent_group_count": len(negative_groups),
        "boundary_negative_phase_counts": dict(sorted(phase_negative.items())),
        "minimum_candidate_success_parent_groups": 2,
        "core": core_rows,
        "boundary": boundary_rows,
        "environment_interactions": interactions,
        "training_transitions": 0,
        "expert_switching_used": False,
        "validation_data_used": False,
        "test_data_used": False,
        "final_evaluation_data_used": False,
        "claim_boundary": {
            "baseline_locked_before_candidate_training": True,
            "baseline_boundary_negatives_not_rerolled_during_gate": True,
            "candidate_policy_outcomes_inspected": False,
            "certified_safe_set_claim": False,
            "jce_jel_claim": False,
        },
    }
    lock["lock_sha256"] = canonical_sha256(lock)
    output.mkdir(parents=True, exist_ok=False)
    _write(output / "baseline_lock.json", lock)
    return lock


def run_candidate_gate(*, baseline_lock: Path, candidate_frozen_policy: Path, output_dir: Path):
    lock = _read(Path(baseline_lock))
    if not isinstance(lock, dict) or lock.get("schema") != LOCK_SCHEMA:
        raise ValueError("candidate gate baseline lock schema drift")
    if lock.get("status") != "locked_before_candidate_training":
        raise ValueError("candidate gate baseline is not locked")
    _verify_hash(lock, "lock_sha256")
    selected, baseline_record = _load_selected(Path(str(lock["selected_policy"])))
    if selected["selection_sha256"] != lock["selected_policy_sha256"]:
        raise ValueError("candidate gate selected baseline drift")

    candidate_path = Path(candidate_frozen_policy)
    candidate_manifest = load_frozen_unified_manifest(candidate_path)
    candidate_record = dict(candidate_manifest["policy"])
    source_iteration = int(lock["source_iteration"])
    if int(candidate_record["iteration"]) != source_iteration + 1:
        raise ValueError("candidate gate requires pi_k -> pi_(k+1)")
    if candidate_record["name"] != f"pi_{source_iteration + 1}":
        raise ValueError("candidate gate policy name drift")
    if candidate_record["xml_sha256"] != baseline_record["xml_sha256"]:
        raise ValueError("candidate gate XML drift")

    formal, artifact, env = _runtime(baseline_record, Path(str(lock["source_tube"])))
    if artifact.manifest["manifest_sha256"] != lock["source_tube_manifest_sha256"]:
        raise ValueError("candidate gate source Tube drift")
    candidate_policy = paired._checkpoint_policy(env, candidate_record)
    core_criterion = _candidate_core_success_criterion(lock)
    core_rollout = _make_compiled_rollout(
        env,
        candidate_policy,
        formal.ppo.episode_horizon,
        success_criterion=core_criterion,
    )
    boundary_criterion = str(lock.get("acceptance_success_criterion", "stable_recovery"))
    boundary_rollout = _make_compiled_rollout(
        env,
        candidate_policy,
        formal.ppo.episode_horizon,
        success_criterion=boundary_criterion,
    )
    reset = jax.jit(env.reset_tube_index)
    records = []
    interactions = 0

    for row in lock["core"]:
        state = reset(np.int32(row["phase_index"]), np.int32(row["entry_index"]))
        if paired._sha256_state(state) != row["state_sha256"]:
            raise ValueError("candidate gate core reset physical-state drift")
        result = _outcome(
            core_rollout(state, jax.random.PRNGKey(int(row["baseline_seed"]))),
            start_phase=int(row["phase_index"]),
            max_ticks=formal.ppo.episode_horizon,
            success_criterion=core_criterion,
        )
        interactions += int(result["environment_interactions"])
        records.append(
            {
                **dict(row),
                "candidate_success": bool(result["success"]),
                "candidate_outcome_class": result["outcome_class"],
                "candidate_environment_interactions": int(result["environment_interactions"]),
            }
        )

    for row in lock["boundary"]:
        snapshot = load_unified_envelope_snapshot(Path(str(row["snapshot"])))
        state = fresh_unified_continuation_start(snapshot, env)
        base = jax.random.PRNGKey(int(row["baseline_labeling_seed"]))
        key = jax.random.fold_in(base, int(row["baseline_candidate_index"]))
        result = _outcome(
            boundary_rollout(state, key),
            start_phase=int(row["phase_index"]),
            max_ticks=formal.ppo.episode_horizon,
            success_criterion=boundary_criterion,
        )
        interactions += int(result["environment_interactions"])
        records.append(
            {
                **dict(row),
                "candidate_success": bool(result["success"]),
                "candidate_outcome_class": result["outcome_class"],
                "candidate_environment_interactions": int(result["environment_interactions"]),
            }
        )

    gates = paired.summarize_paired_gate_records(
        records,
        minimum_candidate_success_parent_groups=int(lock["minimum_candidate_success_parent_groups"]),
        require_baseline_success_each_phase=True,
    )
    # The boundary baseline is a locked negative label, not a rerun.  Therefore
    # reproduction failures are definitionally zero and cannot drift with PRNG.
    if int(gates["boundary"]["baseline_reproduction_failure_count"]) != 0:
        raise ValueError("locked-negative boundary unexpectedly reproduced positive")
    report = {
        "schema": REPORT_SCHEMA,
        "status": "completed",
        "source_iteration": source_iteration,
        "candidate_iteration": source_iteration + 1,
        "baseline_policy_name": str(baseline_record["name"]),
        "baseline_actor_sha256": str(baseline_record["actor_sha256"]),
        "baseline_payload_sha256": str(baseline_record["payload_sha256"]),
        "candidate_policy_name": str(candidate_record["name"]),
        "candidate_actor_sha256": str(candidate_record["actor_sha256"]),
        "candidate_payload_sha256": str(candidate_record["payload_sha256"]),
        "protocol_sha256": str(lock["lock_sha256"]),
        "bank_sha256": str(lock["lock_sha256"]),
        "boundary_source": {
            "selection": "pretrained_candidate_blind_locked_baseline_negative_acceptance_role",
            "acceptance_role_manifest_sha256": str(lock["acceptance_role_manifest_sha256"]),
            "baseline_rerolled_during_gate": False,
            "success_criterion": boundary_criterion,
        },
        "core_gate": gates["core"],
        "core_source": {
            "baseline_success_criterion": "stable_recovery",
            "candidate_success_criterion": core_criterion,
            "preservation_population": "locked_baseline_successes_only",
        },
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
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"candidate gate output already exists: {output}")
    output.mkdir(parents=True, exist_ok=False)
    _write(output / "records.json", {"records": records})
    _write(output / "summary.json", report)
    return report
