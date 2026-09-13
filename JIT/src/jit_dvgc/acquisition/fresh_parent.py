"""Fresh real-dynamics parent trajectories for acceptance-boundary acquisition.

Physical diversity comes from a small, predeclared action excitation applied at
one canonical natural task start, followed by the exact frozen unified policy.
Only trajectories that complete the full unified recovery task may contribute
handoff anchors. This is non-final audit evidence, never training data.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import jax
import numpy as np

from ..config import file_sha256
from ..constants import ACTION_ORDER
from ..repair_acceptance import (
    REPAIR_ACCEPTANCE_SCHEMA,
    canonical_sha256,
    consumed_gate_exclusions,
    prepare_repair_acceptance_predeclaration,
)
from ..soft_tube import load_soft_tube
from ..unified_boundary import action_sparse_directions
from ..unified_continuation_labels import fresh_unified_continuation_start
from ..unified_envelope_snapshot import (
    UnifiedEnvelopeSnapshot,
    capture_unified_envelope_snapshot,
    load_unified_envelope_snapshot,
    physical_state_sha256,
    save_unified_envelope_snapshot,
)
from ..upstream_boundary import are_near_duplicates


FRESH_PARENT_SOURCE_SCHEMA = "jit_fresh_parent_handoff_source_v1"
FRESH_PARENT_TYPE = "natural_action_excitation_handoff_v1"
DEFAULT_PARENT_EXCITATION_STRENGTH = 0.10
DEFAULT_PARENT_EXCITATION_DURATION = 2
DEFAULT_UPSTREAM_APEX_OFFSET = -10
DEFAULT_DOWNSTREAM_APEX_OFFSET = 10
DEFAULT_PARENT_NEAR_DUPLICATE_ATOL = 1.0e-5
DEFAULT_MINIMUM_PARENT_GROUPS = 4
DEFAULT_BOUNDARY_STRENGTHS = (0.15, 0.30, 0.50)
DEFAULT_BOUNDARY_DURATIONS = (2, 4, 8)
DEFAULT_BOUNDARY_ACTIVE_DIMENSIONS = 2
POLICY_KEY_SCHEME = (
    "trajectory_key=fold_in(PRNGKey(protocol_seed),trajectory_index);"
    "action_key=fold_in(trajectory_key,tick)"
)


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


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


def _physical_state_sha(state: Any) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(np.asarray(jax.device_get(state.data.qpos))).tobytes())
    digest.update(np.ascontiguousarray(np.asarray(jax.device_get(state.data.qvel))).tobytes())
    return digest.hexdigest()


def baseline_probe_declaration(
    root: Path,
    *,
    baseline_actor_sha256: str,
    baseline_payload_sha256: str,
) -> tuple[dict[str, Any], set[str]]:
    """Bind one completed baseline-only readiness probe and its exact states."""
    root = Path(root)
    labels_path, summary_path = root / "labels.json", root / "summary.json"
    if not labels_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(f"baseline-only probe labels/summary missing: {root}")
    labels, summary = _read_json(labels_path), _read_json(summary_path)
    if not isinstance(labels, list) or not labels:
        raise ValueError(f"baseline-only probe labels must be nonempty: {root}")
    if not isinstance(summary, dict) or summary.get("status") != "completed":
        raise ValueError(f"baseline-only probe summary is not completed: {root}")
    if summary.get("policy_actor_sha256") != baseline_actor_sha256:
        raise ValueError("baseline-only probe actor identity drift")
    if summary.get("policy_payload_sha256") != baseline_payload_sha256:
        raise ValueError("baseline-only probe payload identity drift")
    if int(summary.get("training_transitions", 0)) != 0:
        raise ValueError("baseline-only readiness probe unexpectedly trained")
    for key in ("validation_data_used", "test_data_used", "final_evaluation_data_used"):
        if summary.get(key, False) is not False:
            raise ValueError(f"baseline-only readiness probe used forbidden data: {key}")

    states: set[str] = set()
    phase_counts = Counter()
    label_counts = Counter()
    groups: dict[str, set[str]] = {"upstream": set(), "downstream": set()}
    for row in labels:
        if row.get("split") != "train":
            raise ValueError("baseline-only readiness probe is not TRAIN-only")
        if row.get("policy_actor_sha256") != baseline_actor_sha256:
            raise ValueError("baseline-only probe row actor drift")
        if row.get("policy_payload_sha256") != baseline_payload_sha256:
            raise ValueError("baseline-only probe row payload drift")
        phase, label = str(row.get("phase", "")), int(row.get("label", -1))
        if phase not in groups or label not in (0, 1):
            raise ValueError("baseline-only probe phase/label invalid")
        state_sha = str(row.get("state_sha256", ""))
        if len(state_sha) != 64 or state_sha in states:
            raise ValueError("baseline-only probe state identity invalid or duplicate")
        states.add(state_sha)
        phase_counts[phase] += 1
        label_counts[label] += 1
        groups[phase].add(str(row.get("parent_group_id", "")))
    declared_count = int(summary.get("label_count", summary.get("candidate_count", -1)))
    if declared_count != len(labels):
        raise ValueError("baseline-only probe label count drift")
    declaration = {
        "root": str(root),
        "labels_file_sha256": file_sha256(labels_path),
        "summary_file_sha256": file_sha256(summary_path),
        "label_protocol_sha256": str(summary.get("protocol_sha256", "")),
        "state_count": len(states),
        "positive_count": int(label_counts[1]),
        "negative_count": int(label_counts[0]),
        "phase_state_counts": {p: int(phase_counts[p]) for p in groups},
        "phase_parent_group_counts": {p: len(groups[p]) for p in groups},
        "baseline_actor_sha256": baseline_actor_sha256,
        "baseline_payload_sha256": baseline_payload_sha256,
        "candidate_policy_outcomes_inspected": False,
        "training_transitions": 0,
        "validation_data_used": False,
        "test_data_used": False,
        "final_evaluation_data_used": False,
    }
    return declaration, states


def consumed_baseline_probe_exclusions(
    protocol: Mapping[str, Any],
) -> tuple[set[str], dict[str, Any]]:
    raw = protocol.get("consumed_baseline_probes", [])
    if not isinstance(raw, list):
        raise ValueError("consumed_baseline_probes must be a list")
    actor = str(protocol.get("baseline_actor_sha256", ""))
    payload = str(protocol.get("baseline_payload_sha256", ""))
    states: set[str] = set()
    verified, roots = [], set()
    for declared in raw:
        if not isinstance(declared, Mapping):
            raise ValueError("consumed baseline probe declaration must be an object")
        root = str(declared.get("root", ""))
        if not root or root in roots:
            raise ValueError("consumed baseline probe roots must be unique")
        roots.add(root)
        actual, probe_states = baseline_probe_declaration(
            Path(root), baseline_actor_sha256=actor, baseline_payload_sha256=payload
        )
        for key in (
            "labels_file_sha256",
            "summary_file_sha256",
            "state_count",
            "positive_count",
            "negative_count",
        ):
            if str(declared.get(key)) != str(actual[key]):
                raise ValueError(f"consumed baseline probe {key} drift: {root}")
        states.update(probe_states)
        verified.append(actual)
    return states, {
        "probe_count": len(verified),
        "probes": verified,
        "union_state_count": len(states),
    }


def prepare_fresh_parent_acceptance_predeclaration(
    *,
    baseline_frozen_policy: Path,
    target_tube: Path,
    consumed_gate_roots: Sequence[Path],
    consumed_baseline_probe_roots: Sequence[Path],
    acquisition_seed: int,
    labeling_seed: int,
    parent_excitation_strength: float = DEFAULT_PARENT_EXCITATION_STRENGTH,
    parent_excitation_duration: int = DEFAULT_PARENT_EXCITATION_DURATION,
    upstream_apex_offset: int = DEFAULT_UPSTREAM_APEX_OFFSET,
    downstream_apex_offset: int = DEFAULT_DOWNSTREAM_APEX_OFFSET,
    parent_near_duplicate_atol: float = DEFAULT_PARENT_NEAR_DUPLICATE_ATOL,
    minimum_parent_groups: int = DEFAULT_MINIMUM_PARENT_GROUPS,
    boundary_strengths: Sequence[float] = DEFAULT_BOUNDARY_STRENGTHS,
    boundary_durations: Sequence[int] = DEFAULT_BOUNDARY_DURATIONS,
    boundary_active_action_dimensions: int = DEFAULT_BOUNDARY_ACTIVE_DIMENSIONS,
    minimum_negative_states_per_phase: int = 10,
    minimum_negative_parent_groups_per_phase: int = 3,
) -> dict[str, Any]:
    """Predeclare a candidate-blind acceptance search from new physical parents."""
    if not 0.0 < float(parent_excitation_strength) <= 1.0:
        raise ValueError("parent excitation strength must lie in (0, 1]")
    if int(parent_excitation_duration) <= 0:
        raise ValueError("parent excitation duration must be positive")
    if int(upstream_apex_offset) >= 0 or int(downstream_apex_offset) <= 0:
        raise ValueError("handoff offsets must straddle Apex")
    if float(parent_near_duplicate_atol) <= 0.0:
        raise ValueError("parent near-duplicate tolerance must be positive")
    if int(minimum_parent_groups) < int(minimum_negative_parent_groups_per_phase):
        raise ValueError("parent-source minimum cannot be below gate parent minimum")

    base = prepare_repair_acceptance_predeclaration(
        baseline_frozen_policy=Path(baseline_frozen_policy),
        target_tube=Path(target_tube),
        consumed_gate_roots=tuple(Path(x) for x in consumed_gate_roots),
        acquisition_seed=int(acquisition_seed),
        labeling_seed=int(labeling_seed),
        anchors_per_phase=1,
        minimum_anchors_per_phase=1,
        frontier_score_ceiling=0.5,
        strengths=tuple(float(x) for x in boundary_strengths),
        durations=tuple(int(x) for x in boundary_durations),
        action_names=ACTION_ORDER,
        signs=(-1, 1),
        active_action_dimensions=int(boundary_active_action_dimensions),
        minimum_negative_states_per_phase=int(minimum_negative_states_per_phase),
        minimum_negative_parent_groups_per_phase=int(minimum_negative_parent_groups_per_phase),
    )
    protocol = dict(base["protocol"])
    probes = []
    for root in consumed_baseline_probe_roots:
        declaration, _ = baseline_probe_declaration(
            Path(root),
            baseline_actor_sha256=str(protocol["baseline_actor_sha256"]),
            baseline_payload_sha256=str(protocol["baseline_payload_sha256"]),
        )
        probes.append(declaration)
    if len({row["root"] for row in probes}) != len(probes):
        raise ValueError("consumed baseline probe roots must be unique")

    protocol["purpose"] = (
        "fresh_nonfinal_acceptance_bank_from_candidate_blind_real_dynamics_"
        "natural_excitation_handoff_parents"
    )
    protocol["consumed_baseline_probes"] = probes
    protocol["design_diagnosis"] = {
        **dict(protocol["design_diagnosis"]),
        "prior_baseline_only_readiness_evidence_inspected": bool(probes),
        "parent_group_exhaustion_in_tube_frontier_source": True,
        "new_parent_source_selected_without_candidate_policy_information": True,
    }
    protocol["acquisition"] = {
        "protocol_seed": int(acquisition_seed),
        "anchor_source": {
            "type": FRESH_PARENT_TYPE,
            "reset": "canonical_deterministic_natural_task_start",
            "physical_diversity_source": "bounded_action_excitation_then_frozen_pi_k",
            "action_names": list(ACTION_ORDER),
            "signs": [-1, 1],
            "active_action_dimensions": 1,
            "strength": float(parent_excitation_strength),
            "duration": int(parent_excitation_duration),
            "excitation_start_tick": 0,
            "upstream_apex_offset": int(upstream_apex_offset),
            "downstream_apex_offset": int(downstream_apex_offset),
            "require_terminal_full_recovery_success": True,
            "near_duplicate_atol": float(parent_near_duplicate_atol),
            "minimum_parent_groups": int(minimum_parent_groups),
            "parent_group_semantics": "one canonical action-axis/sign excitation direction",
            "policy_key_scheme": POLICY_KEY_SCHEME,
        },
        "boundary_probe": {
            "action_names": list(ACTION_ORDER),
            "signs": [-1, 1],
            "active_action_dimensions": int(boundary_active_action_dimensions),
            "strengths": [float(x) for x in boundary_strengths],
            "durations": [int(x) for x in boundary_durations],
            "reject_phase_crossing": True,
        },
    }
    protocol["isolation"] = {
        **dict(protocol["isolation"]),
        "exclude_consumed_baseline_probe_exact_states": True,
        "exclude_target_tube_states_during_acquisition": True,
        "parent_groups_derive_from_physical_excitation_not_rng_seed": True,
    }
    result = {
        "schema": REPAIR_ACCEPTANCE_SCHEMA,
        "expected_protocol_sha256": canonical_sha256(protocol),
        "protocol": protocol,
    }
    consumed_gate_exclusions(protocol)
    consumed_baseline_probe_exclusions(protocol)
    return result


def planned_parent_groups(predeclaration: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    protocol = predeclaration["protocol"]
    source = protocol["acquisition"]["anchor_source"]
    if source.get("type") != FRESH_PARENT_TYPE:
        raise ValueError("unsupported fresh parent source type")
    if int(source["active_action_dimensions"]) != 1:
        raise ValueError("fresh parent source requires one-axis excitation")
    directions = action_sparse_directions(
        action_names=tuple(source["action_names"]),
        signs=tuple(source["signs"]),
        active_action_dimensions=1,
    )
    result = []
    for index, row in enumerate(directions):
        sign = int(row["sign"])
        result.append(
            {
                "trajectory_index": index,
                "parent_group_id": (
                    f"pi{int(protocol['source_iteration'])}_natural_excitation_"
                    f"{row['action_name']}_{'neg' if sign < 0 else 'pos'}"
                ),
                "action_dimension": int(row["action_dimension"]),
                "action_name": str(row["action_name"]),
                "sign": sign,
                "basis_vector": list(row["basis_vector"]),
                "strength": float(source["strength"]),
                "duration": int(source["duration"]),
            }
        )
    return tuple(result)


def audit_fresh_parent_predeclaration(predeclaration: Mapping[str, Any]) -> dict[str, Any]:
    if predeclaration.get("schema") != REPAIR_ACCEPTANCE_SCHEMA:
        raise ValueError("fresh parent predeclaration schema mismatch")
    protocol = predeclaration.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("fresh parent predeclaration protocol missing")
    if canonical_sha256(protocol) != predeclaration.get("expected_protocol_sha256"):
        raise ValueError("fresh parent predeclaration SHA-256 drift")
    gate_states, _, gate_audit = consumed_gate_exclusions(protocol)
    probe_states, probe_audit = consumed_baseline_probe_exclusions(protocol)
    target = load_soft_tube(Path(protocol["bank_lock"]["target_tube"]))
    target_states = {str(row["state_sha256"]) for row in target.entries}
    groups = planned_parent_groups(predeclaration)
    boundary = protocol["acquisition"]["boundary_probe"]
    directions = action_sparse_directions(
        action_names=tuple(boundary["action_names"]),
        signs=tuple(boundary["signs"]),
        active_action_dimensions=int(boundary["active_action_dimensions"]),
    )
    parent_max = len(groups) * int(protocol["labeling"]["max_ticks"])
    per_anchor_max = sum(
        int(duration)
        for duration in boundary["durations"]
        for _strength in boundary["strengths"]
        for _direction in directions
    )
    maximum_anchor_count = 2 * len(groups)
    return {
        "schema": "jit_fresh_parent_acceptance_design_audit_v1",
        "status": "predeclared_no_interactions",
        "source_iteration": int(protocol["source_iteration"]),
        "candidate_iteration": int(protocol["candidate_iteration"]),
        "parent_source_type": FRESH_PARENT_TYPE,
        "planned_parent_group_count": len(groups),
        "planned_parent_groups": list(groups),
        "minimum_parent_groups": int(protocol["acquisition"]["anchor_source"]["minimum_parent_groups"]),
        "parent_maximum_environment_interactions": parent_max,
        "maximum_anchor_count": maximum_anchor_count,
        "boundary_direction_count": len(directions),
        "boundary_maximum_environment_interactions": maximum_anchor_count * per_anchor_max,
        "maximum_environment_interactions": parent_max + maximum_anchor_count * per_anchor_max,
        "excluded_consumed_gate_state_count": len(gate_states),
        "excluded_consumed_baseline_probe_state_count": len(probe_states),
        "excluded_target_tube_state_count": len(target_states),
        "excluded_exact_state_union_count": len(gate_states | probe_states | target_states),
        "consumed_gates": gate_audit,
        "consumed_baseline_probes": probe_audit,
        "training_transitions": 0,
        "environment_interactions": 0,
        "validation_data_used": False,
        "test_data_used": False,
        "final_evaluation_data_used": False,
    }


@dataclass(frozen=True)
class FreshParentAnchor:
    phase: str
    phase_index: int
    parent_group_id: str
    parent_trajectory_index: int
    snapshot_path: Path
    state_sha256: str
    apex_offset: int


def _near_duplicate_pair(
    upstream: UnifiedEnvelopeSnapshot,
    downstream: UnifiedEnvelopeSnapshot,
    accepted: Sequence[tuple[UnifiedEnvelopeSnapshot, UnifiedEnvelopeSnapshot]],
    *,
    atol: float,
) -> bool:
    return any(
        are_near_duplicates(upstream, old_up, qpos_atol=atol, qvel_atol=atol, observation_atol=atol)
        and are_near_duplicates(downstream, old_down, qpos_atol=atol, qvel_atol=atol, observation_atol=atol)
        for old_up, old_down in accepted
    )


def _excluded_states(protocol: Mapping[str, Any]) -> set[str]:
    gate_states, _, _ = consumed_gate_exclusions(protocol)
    probe_states, _ = consumed_baseline_probe_exclusions(protocol)
    target = load_soft_tube(Path(protocol["bank_lock"]["target_tube"]))
    return gate_states | probe_states | {str(row["state_sha256"]) for row in target.entries}


def collect_fresh_parent_anchors(
    predeclaration: Mapping[str, Any],
    output_dir: Path,
    *,
    env: Any,
    policy: Callable[[Any, Any], Any],
    policy_record: Mapping[str, Any],
    compiled_step_fn: Callable[[Any, Any], Any] | None = None,
) -> tuple[tuple[FreshParentAnchor, ...], dict[str, Any]]:
    """Collect paired Apex-10/Apex+10 anchors from successful physical trajectories."""
    protocol = predeclaration["protocol"]
    source = protocol["acquisition"]["anchor_source"]
    if policy_record["actor_sha256"] != protocol["baseline_actor_sha256"]:
        raise ValueError("fresh parent baseline actor drift")
    if policy_record["payload_sha256"] != protocol["baseline_payload_sha256"]:
        raise ValueError("fresh parent baseline payload drift")
    if policy_record["xml_sha256"] != env._bundle.xml_sha256:
        raise ValueError("fresh parent runtime XML drift")
    if not hasattr(env, "_reset_natural_unified"):
        raise ValueError("fresh parent runtime lacks authoritative unified natural reset")

    output = Path(output_dir)
    parent_bank = output / "parent_bank"
    (parent_bank / "snapshots").mkdir(parents=True, exist_ok=False)
    step_fn = compiled_step_fn or jax.jit(env.step)
    reset_fn = jax.jit(env._reset_natural_unified)
    seed = int(protocol["acquisition"]["protocol_seed"])
    base_key = jax.random.PRNGKey(seed)
    reset_key = jax.random.fold_in(base_key, 0x4E415455)
    reset_state = reset_fn(reset_key)
    jax.block_until_ready(reset_state)
    if _truth(reset_state.info["reset_from_soft_tube"]) or _integer(reset_state.info["active_phase"]) != 0:
        raise ValueError("fresh parent source did not use canonical upstream natural reset")
    reset_sha = _physical_state_sha(reset_state)
    groups = planned_parent_groups(predeclaration)
    excluded = _excluded_states(protocol)
    upstream_offset = int(source["upstream_apex_offset"])
    downstream_offset = int(source["downstream_apex_offset"])
    max_ticks = int(protocol["labeling"]["max_ticks"])

    anchors: list[FreshParentAnchor] = []
    accepted_pairs: list[tuple[UnifiedEnvelopeSnapshot, UnifiedEnvelopeSnapshot]] = []
    records: list[dict[str, Any]] = []
    exclusions = Counter()
    interactions = 0
    for group in groups:
        state = reset_fn(reset_key)
        jax.block_until_ready(state)
        if _physical_state_sha(state) != reset_sha:
            raise ValueError("canonical natural reset physical state drift")
        trajectory_key = jax.random.fold_in(base_key, int(group["trajectory_index"]))
        history: deque[tuple[int, Any]] = deque(maxlen=abs(upstream_offset) + 2)
        history.append((0, state))
        transition_tick = None
        upstream_state = downstream_state = None
        effective_delta_l1 = 0.0
        terminal_success = False
        for tick in range(max_ticks):
            action_key = jax.random.fold_in(trajectory_key, tick)
            result = policy(state.obs, action_key)
            nominal = result[0] if isinstance(result, tuple) else result
            nominal_np = np.asarray(jax.device_get(nominal), np.float32).reshape(-1)
            if nominal_np.shape != (len(ACTION_ORDER),) or not np.isfinite(nominal_np).all():
                raise ValueError("fresh parent frozen policy returned invalid action")
            if tick < int(group["duration"]):
                action_np = np.clip(
                    nominal_np + np.asarray(group["basis_vector"], np.float32) * np.float32(group["strength"]),
                    -1.0,
                    1.0,
                ).astype(np.float32)
            else:
                action_np = nominal_np
            effective_delta_l1 += float(np.abs(action_np - nominal_np).sum())
            previous = state
            state = step_fn(state, jax.device_put(action_np))
            jax.block_until_ready(state)
            interactions += 1
            if not _finite_state(state, action_np):
                exclusions["nonfinite"] += 1
                break
            if _truth(state.info["expert_switching_used"]):
                raise ValueError("fresh parent source used expert switching")
            current_tick = tick + 1
            history.append((current_tick, state))
            transitioned_now = _integer(previous.info["active_phase"]) == 0 and _integer(state.info["active_phase"]) == 1
            if transitioned_now and transition_tick is None:
                transition_tick = current_tick
                target_tick = transition_tick + upstream_offset
                upstream_state = next((saved for saved_tick, saved in history if saved_tick == target_tick), None)
            if transition_tick is not None and current_tick == transition_tick + downstream_offset:
                downstream_state = state
            if _truth(state.done):
                terminal_success = _truth(state.info["success"])
                break

        if effective_delta_l1 <= 1.0e-8:
            exclusions["ineffective_excitation"] += 1
            continue
        if not terminal_success:
            exclusions["not_full_recovery_success"] += 1
            continue
        if upstream_state is None or downstream_state is None:
            exclusions["missing_handoff_offsets"] += 1
            continue
        if _integer(upstream_state.info["active_phase"]) != 0 or _integer(downstream_state.info["active_phase"]) != 1:
            exclusions["anchor_phase_mismatch"] += 1
            continue
        group_id = str(group["parent_group_id"])
        up = capture_unified_envelope_snapshot(
            upstream_state,
            env=env,
            parent_trajectory=group_id,
            parent_state_sha256=reset_sha,
            config_sha256=str(policy_record["formal_config_sha256"]),
            policy_actor_sha256=str(policy_record["actor_sha256"]),
            policy_payload_sha256=str(policy_record["payload_sha256"]),
            policy_iteration=int(policy_record["iteration"]),
        )
        down = capture_unified_envelope_snapshot(
            downstream_state,
            env=env,
            parent_trajectory=group_id,
            parent_state_sha256=reset_sha,
            config_sha256=str(policy_record["formal_config_sha256"]),
            policy_actor_sha256=str(policy_record["actor_sha256"]),
            policy_payload_sha256=str(policy_record["payload_sha256"]),
            policy_iteration=int(policy_record["iteration"]),
        )
        up_sha, down_sha = physical_state_sha256(up), physical_state_sha256(down)
        if up_sha in excluded or down_sha in excluded:
            exclusions["excluded_exact_anchor_state"] += 1
            continue
        if _near_duplicate_pair(up, down, accepted_pairs, atol=float(source["near_duplicate_atol"])):
            exclusions["near_duplicate_parent_pair"] += 1
            continue
        pair_index = len(accepted_pairs)
        up_rel = Path("snapshots") / f"parent_{pair_index:03d}_upstream"
        down_rel = Path("snapshots") / f"parent_{pair_index:03d}_downstream"
        save_unified_envelope_snapshot(parent_bank / up_rel, up)
        save_unified_envelope_snapshot(parent_bank / down_rel, down)
        accepted_pairs.append((up, down))
        anchors.extend(
            (
                FreshParentAnchor("upstream", 0, group_id, int(group["trajectory_index"]), parent_bank / up_rel, up_sha, upstream_offset),
                FreshParentAnchor("downstream", 1, group_id, int(group["trajectory_index"]), parent_bank / down_rel, down_sha, downstream_offset),
            )
        )
        records.append(
            {
                **dict(group),
                "status": "accepted_successful_parent",
                "reset_physical_state_sha256": reset_sha,
                "transition_tick": transition_tick,
                "upstream_anchor_state_sha256": up_sha,
                "downstream_anchor_state_sha256": down_sha,
                "effective_excitation_l1": effective_delta_l1,
                "terminal_full_recovery_success": True,
            }
        )

    accepted_groups = {a.parent_group_id for a in anchors}
    status = "completed" if len(accepted_groups) >= int(source["minimum_parent_groups"]) else "not_ready"
    report = {
        "schema": FRESH_PARENT_SOURCE_SCHEMA,
        "status": status,
        "source_type": FRESH_PARENT_TYPE,
        "policy_name": str(policy_record["name"]),
        "policy_actor_sha256": str(policy_record["actor_sha256"]),
        "policy_payload_sha256": str(policy_record["payload_sha256"]),
        "reset_physical_state_sha256": reset_sha,
        "planned_parent_group_count": len(groups),
        "accepted_parent_group_count": len(accepted_groups),
        "accepted_anchor_count": len(anchors),
        "minimum_parent_groups": int(source["minimum_parent_groups"]),
        "exclusion_counts": dict(sorted(exclusions.items())),
        "records": records,
        "environment_interactions": interactions,
        "maximum_environment_interactions": len(groups) * max_ticks,
        "training_transitions": 0,
        "expert_switching_used": False,
        "validation_data_used": False,
        "test_data_used": False,
        "final_evaluation_data_used": False,
    }
    _write_json(parent_bank / "summary.json", report)
    if status != "completed":
        raise ValueError("fresh physical parent source did not meet predeclared group minimum")
    return tuple(anchors), report


def collect_snapshot_anchor_boundary_candidates(
    predeclaration: Mapping[str, Any],
    anchors: Sequence[FreshParentAnchor],
    output_dir: Path,
    *,
    env: Any,
    policy: Callable[[Any, Any], Any],
    policy_record: Mapping[str, Any],
    frozen_manifest_sha256: str,
    compiled_step_fn: Callable[[Any, Any], Any] | None = None,
) -> dict[str, Any]:
    """Generate phase-local challenge candidates from admitted parent anchors."""
    contract = predeclaration["protocol"]
    acquisition = contract["acquisition"]
    boundary = acquisition["boundary_probe"]
    directions = action_sparse_directions(
        action_names=tuple(boundary["action_names"]),
        signs=tuple(boundary["signs"]),
        active_action_dimensions=int(boundary["active_action_dimensions"]),
    )
    strengths = tuple(float(x) for x in boundary["strengths"])
    durations = tuple(int(x) for x in boundary["durations"])
    if any(not 0.0 < x <= 1.0 for x in strengths) or any(x <= 0 for x in durations):
        raise ValueError("fresh parent boundary strength/duration invalid")
    source_min = int(acquisition["anchor_source"]["minimum_parent_groups"])
    phase_groups = {p: {a.parent_group_id for a in anchors if a.phase == p} for p in ("upstream", "downstream")}
    if any(len(phase_groups[p]) < source_min for p in phase_groups):
        raise ValueError("fresh parent anchors lost phasewise group readiness")

    excluded = _excluded_states(contract)
    step_fn = compiled_step_fn or jax.jit(env.step)
    base_key = jax.random.PRNGKey(int(acquisition["protocol_seed"]))
    maximum_interactions = sum(
        int(duration)
        for _anchor in anchors
        for duration in durations
        for _strength in strengths
        for _direction in directions
    )
    logical_protocol = {
        "schema": "jit_unified_boundary_protocol_v1",
        "status": "predeclared",
        "purpose": "fresh_acceptance_boundary_from_successful_natural_excitation_handoff_parents",
        "split": "train",
        "iteration": int(policy_record["iteration"]),
        "policy_name": str(policy_record["name"]),
        "policy_actor_sha256": str(policy_record["actor_sha256"]),
        "policy_payload_sha256": str(policy_record["payload_sha256"]),
        "policy_formal_config_sha256": str(policy_record["formal_config_sha256"]),
        "frozen_unified_manifest_sha256": str(frozen_manifest_sha256),
        "source_tube_manifest_sha256": str(env.tube_pool.artifact.manifest["manifest_sha256"]),
        "parent_source_type": FRESH_PARENT_TYPE,
        "parent_group_count": len({a.parent_group_id for a in anchors}),
        "anchor_count": len(anchors),
        "protocol_seed": int(acquisition["protocol_seed"]),
        "direction_family": f"action_sparse_{int(boundary['active_action_dimensions'])}",
        "selected_action_names": list(boundary["action_names"]),
        "selected_signs": list(boundary["signs"]),
        "active_action_dimensions": int(boundary["active_action_dimensions"]),
        "strengths": list(strengths),
        "durations": list(durations),
        "maximum_environment_interactions": maximum_interactions,
        "excluded_exact_state_count": len(excluded),
        "training_transitions": 0,
        "expert_switching_used": False,
        "validation_data_used": False,
        "test_data_used": False,
        "final_evaluation_data_used": False,
        "claim_boundary": {
            "unlabeled_acquisition_only": True,
            "tube_expansion_claim": False,
            "jce_jel_claim": False,
            "certified_safe_set_claim": False,
        },
    }
    protocol_sha = canonical_sha256(logical_protocol)
    output = Path(output_dir)
    boundary_bank = output / "boundary_bank"
    (boundary_bank / "snapshots").mkdir(parents=True, exist_ok=False)
    _write_json(output / "protocol.json", {**logical_protocol, "protocol_sha256": protocol_sha})

    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    exclusions = Counter()
    interactions = variant_index = 0
    for anchor in sorted(anchors, key=lambda a: (a.phase_index, a.parent_trajectory_index)):
        snapshot = load_unified_envelope_snapshot(anchor.snapshot_path)
        if physical_state_sha256(snapshot) != anchor.state_sha256:
            raise ValueError("fresh parent anchor snapshot identity drift")
        for duration in durations:
            for strength in strengths:
                for direction in directions:
                    state = fresh_unified_continuation_start(snapshot, env)
                    if _integer(state.info["active_phase"]) != anchor.phase_index:
                        raise ValueError("fresh parent boundary anchor phase drift")
                    current_variant, variant_index = variant_index, variant_index + 1
                    nominal_actions: list[list[float]] = []
                    perturbed_actions: list[list[float]] = []
                    effective_deltas: list[list[float]] = []
                    rejected = None
                    for perturb_step in range(duration):
                        variant_key = jax.random.fold_in(base_key, current_variant)
                        action_key = jax.random.fold_in(variant_key, perturb_step)
                        result = policy(state.obs, action_key)
                        nominal = result[0] if isinstance(result, tuple) else result
                        nominal_np = np.asarray(jax.device_get(nominal), np.float32).reshape(-1)
                        if nominal_np.shape != (len(ACTION_ORDER),) or not np.isfinite(nominal_np).all():
                            raise ValueError("fresh parent boundary policy returned invalid action")
                        perturbed_np = np.clip(
                            nominal_np + np.asarray(direction["basis_vector"], np.float32) * np.float32(strength),
                            -1.0,
                            1.0,
                        ).astype(np.float32)
                        state = step_fn(state, jax.device_put(perturbed_np))
                        jax.block_until_ready(state)
                        interactions += 1
                        nominal_actions.append(nominal_np.tolist())
                        perturbed_actions.append(perturbed_np.tolist())
                        effective_deltas.append((perturbed_np - nominal_np).tolist())
                        if not _finite_state(state, perturbed_np):
                            rejected = "nonfinite"
                            break
                        if _truth(state.info["expert_switching_used"]):
                            raise ValueError("fresh parent boundary used expert switching")
                        if _truth(state.done):
                            rejected = "terminal"
                            break
                        if _integer(state.info["active_phase"]) != anchor.phase_index:
                            rejected = "phase_transition"
                            break
                    if rejected is not None:
                        exclusions[rejected] += 1
                        continue
                    candidate = capture_unified_envelope_snapshot(
                        state,
                        env=env,
                        parent_trajectory=anchor.parent_group_id,
                        parent_state_sha256=anchor.state_sha256,
                        config_sha256=str(policy_record["formal_config_sha256"]),
                        policy_actor_sha256=str(policy_record["actor_sha256"]),
                        policy_payload_sha256=str(policy_record["payload_sha256"]),
                        policy_iteration=int(policy_record["iteration"]),
                    )
                    state_sha = physical_state_sha256(candidate)
                    if state_sha in excluded:
                        exclusions["excluded_exact_state"] += 1
                        continue
                    if state_sha in seen:
                        exclusions["duplicate"] += 1
                        continue
                    seen.add(state_sha)
                    relative = Path("snapshots") / f"candidate_{len(entries):06d}"
                    save_unified_envelope_snapshot(boundary_bank / relative, candidate)
                    entries.append(
                        {
                            "candidate_id": f"pi{int(policy_record['iteration'])}_fresh_parent_{anchor.phase}_{len(entries):06d}",
                            "candidate_kind": "reachable_unified_frontier_probe",
                            "split": "train",
                            "phase": anchor.phase,
                            "phase_index": anchor.phase_index,
                            "snapshot": str(relative),
                            "source_bank": "boundary_bank",
                            "state_sha256": state_sha,
                            "parent_group_id": anchor.parent_group_id,
                            "parent_state_sha256": anchor.state_sha256,
                            "parent_trajectory_index": anchor.parent_trajectory_index,
                            "parent_apex_offset": anchor.apex_offset,
                            "parent_source_type": FRESH_PARENT_TYPE,
                            "policy_iteration": int(policy_record["iteration"]),
                            "policy_actor_sha256": str(policy_record["actor_sha256"]),
                            "policy_payload_sha256": str(policy_record["payload_sha256"]),
                            "protocol_sha256": protocol_sha,
                            "perturbation": {
                                **dict(direction),
                                "strength": float(strength),
                                "duration": int(duration),
                                "variant_index": current_variant,
                                "nominal_actions": nominal_actions,
                                "perturbed_actions": perturbed_actions,
                                "effective_deltas": effective_deltas,
                            },
                        }
                    )

    report = {
        "schema": "jit_unified_boundary_catalog_v1",
        "status": "completed",
        "artifact_role": "unlabeled_policy_conditioned_frontier_candidates",
        "split": "train",
        "iteration": int(policy_record["iteration"]),
        "policy_name": str(policy_record["name"]),
        "policy_actor_sha256": str(policy_record["actor_sha256"]),
        "policy_payload_sha256": str(policy_record["payload_sha256"]),
        "frozen_unified_manifest_sha256": str(frozen_manifest_sha256),
        "source_tube_manifest_sha256": str(env.tube_pool.artifact.manifest["manifest_sha256"]),
        "protocol_sha256": protocol_sha,
        "parent_source_type": FRESH_PARENT_TYPE,
        "parent_group_count": len({a.parent_group_id for a in anchors}),
        "anchor_count": len(anchors),
        "attempted_candidate_count": variant_index,
        "candidate_count": len(entries),
        "environment_interactions": interactions,
        "maximum_environment_interactions": maximum_interactions,
        "exclusion_counts": dict(sorted(exclusions.items())),
        "training_transitions": 0,
        "expert_switching_used": False,
        "validation_data_used": False,
        "test_data_used": False,
        "final_evaluation_data_used": False,
        "claim_boundary": {
            "unlabeled_acquisition_only": True,
            "tube_expansion_claim": False,
            "jce_jel_claim": False,
            "certified_safe_set_claim": False,
        },
        "entries": entries,
    }
    _write_json(output / "catalog.json", report)
    _write_json(output / "summary.json", {k: v for k, v in report.items() if k != "entries"})
    return report
