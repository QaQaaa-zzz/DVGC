"""TRAIN-only real-dynamics frontier acquisition for frozen unified policies.

The source Tube is training guidance, not a certified set. Low-score Tube
entries are used only as auditable frontier-probe anchors. New candidate states
must be produced by the authoritative unified dynamics under bounded action
perturbations; direct qpos/qvel dilation is intentionally unsupported.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import jax
import numpy as np

from .constants import ACTION_ORDER
from .soft_tube import SoftTubeArtifact
from .unified_envelope_snapshot import (
    capture_unified_envelope_snapshot,
    physical_state_sha256,
    save_unified_envelope_snapshot,
)
from .upstream_boundary import (
    action_basis_directions,
    canonical_sha256,
    validate_durations,
    validate_strengths,
)


UNIFIED_BOUNDARY_CATALOG_SCHEMA = "jit_unified_boundary_catalog_v1"
UNIFIED_BOUNDARY_PROTOCOL_SCHEMA = "jit_unified_boundary_protocol_v1"
DEFAULT_UNIFIED_BOUNDARY_STRENGTHS = (0.025, 0.05, 0.10)
DEFAULT_UNIFIED_BOUNDARY_DURATIONS = (1, 2)
DEFAULT_ANCHORS_PER_PHASE = 8
DEFAULT_FRONTIER_SCORE_CEILING = 0.5


def _truth(value: Any) -> bool:
    return bool(np.asarray(jax.device_get(value)))


def _finite_state(state: Any, action: Any | None = None) -> bool:
    arrays = (
        np.asarray(jax.device_get(state.data.qpos)),
        np.asarray(jax.device_get(state.data.qvel)),
        np.asarray(jax.device_get(state.obs["state"])),
    )
    if action is not None:
        arrays = (*arrays, np.asarray(jax.device_get(action)))
    return all(np.isfinite(value).all() for value in arrays)


def action_sparse_directions(
    action_order: Sequence[str] = ACTION_ORDER,
    *,
    action_names: Sequence[str] | None = None,
    signs: Sequence[int] | None = None,
    active_action_dimensions: int = 1,
) -> tuple[dict[str, Any], ...]:
    """Return deterministic sparse perturbation directions in normalized action space.

    ``active_action_dimensions=1`` delegates to the historical one-hot direction
    generator and therefore preserves the exact legacy direction payload.  Wider
    sparse directions enumerate every selected action combination and every
    selected sign combination deterministically.  The generated vector is still
    added to the frozen-policy action and clipped per action component to [-1, 1].
    """
    width = int(active_action_dimensions)
    order = tuple(str(name) for name in action_order)
    names = order if action_names is None else tuple(str(name) for name in action_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("action_names must be non-empty and unique")
    unknown = sorted(set(names).difference(order))
    if unknown:
        raise ValueError(f"unknown action_names: {unknown}")
    if width <= 0 or width > len(names):
        raise ValueError("active_action_dimensions must lie in [1, len(action_names)]")

    selected_signs = (-1, 1) if signs is None else tuple(int(sign) for sign in signs)
    if not selected_signs or len(set(selected_signs)) != len(selected_signs):
        raise ValueError("signs must be non-empty and unique")
    if any(sign not in (-1, 1) for sign in selected_signs):
        raise ValueError("signs must contain only -1 or +1")

    if width == 1:
        return action_basis_directions(
            action_order=order,
            action_names=names,
            signs=selected_signs,
        )

    name_set = set(names)
    ordered_actions = tuple(
        (dimension, name)
        for dimension, name in enumerate(order)
        if name in name_set
    )
    sign_set = set(selected_signs)
    ordered_signs = tuple(sign for sign in (-1, 1) if sign in sign_set)

    result: list[dict[str, Any]] = []
    for active in combinations(ordered_actions, width):
        for active_signs in product(ordered_signs, repeat=width):
            vector = [0.0] * len(order)
            dimensions: list[int] = []
            active_names: list[str] = []
            for (dimension, name), sign in zip(active, active_signs):
                vector[dimension] = float(sign)
                dimensions.append(int(dimension))
                active_names.append(str(name))
            result.append(
                {
                    "action_dimensions": dimensions,
                    "action_names": active_names,
                    "signs": [int(sign) for sign in active_signs],
                    "basis_vector": vector,
                }
            )
    return tuple(result)


@dataclass(frozen=True)
class TubeBoundaryAnchor:
    phase: str
    phase_index: int
    entry_index: int
    global_index: int
    row: Mapping[str, Any]

    @property
    def state_sha256(self) -> str:
        return str(self.row["state_sha256"])

    @property
    def parent_group_id(self) -> str:
        return str(self.row["parent_group_id"])

    @property
    def value_score(self) -> float:
        return float(self.row["value_score"])


def _validate_source_tube(artifact: SoftTubeArtifact) -> None:
    manifest = artifact.manifest
    if manifest.get("schema") != "jit_soft_tube_v1" or manifest.get("status") != "completed":
        raise ValueError("unified boundary acquisition requires a completed Soft Tube")
    if manifest.get("training_guidance_only") is not True or manifest.get("certified_safe") is not False:
        raise ValueError("unified boundary source Tube claim boundary drift")
    if manifest.get("test_data_used") is not False or manifest.get("validation_data_used") is not False:
        raise ValueError("unified boundary source Tube is not TRAIN-only")
    if any(str(row.get("split", "")) != "train" for row in artifact.entries):
        raise ValueError("unified boundary source Tube contains a non-TRAIN row")


def _phase_rows(artifact: SoftTubeArtifact, phase: str) -> list[tuple[int, Mapping[str, Any]]]:
    return [
        (index, row)
        for index, row in enumerate(row for row in artifact.entries if row["phase"] == phase)
    ]


def select_tube_boundary_anchors(
    artifact: SoftTubeArtifact,
    *,
    max_per_phase: int = DEFAULT_ANCHORS_PER_PHASE,
    frontier_score_ceiling: float = DEFAULT_FRONTIER_SCORE_CEILING,
) -> tuple[tuple[TubeBoundaryAnchor, ...], dict[str, Any]]:
    """Select weak-score, parent-group-unique TRAIN frontier probes per phase.

    The score ceiling is a fixed bootstrap continuation decision boundary, not a
    fitted geometric Tube boundary. Anchors above the ceiling are core support
    and are never pulled into frontier acquisition merely to fill a quota.
    These anchors are not asserted to lie on a mathematical boundary.
    """
    _validate_source_tube(artifact)
    if int(max_per_phase) <= 0:
        raise ValueError("max_per_phase must be positive")
    ceiling = float(frontier_score_ceiling)
    if not np.isfinite(ceiling) or not 0.0 <= ceiling <= 1.0:
        raise ValueError("frontier_score_ceiling must be finite and lie in [0, 1]")

    upstream_count = sum(row["phase"] == "upstream" for row in artifact.entries)
    selected: list[TubeBoundaryAnchor] = []
    phase_audit: dict[str, Any] = {}
    for phase, phase_index in (("upstream", 0), ("downstream", 1)):
        phase_rows = _phase_rows(artifact, phase)
        if not phase_rows:
            raise ValueError(f"source Tube has no {phase} support")
        eligible = [
            item for item in phase_rows if float(item[1]["value_score"]) <= ceiling
        ]
        ordered = sorted(
            eligible,
            key=lambda item: (
                float(item[1]["value_score"]),
                str(item[1]["parent_group_id"]),
                str(item[1]["state_sha256"]),
            ),
        )
        chosen: list[tuple[int, Mapping[str, Any]]] = []
        seen_states: set[str] = set()
        seen_groups: set[str] = set()

        for local_index, row in ordered:
            state_hash = str(row["state_sha256"])
            group = str(row["parent_group_id"])
            if state_hash in seen_states or group in seen_groups:
                continue
            chosen.append((local_index, row))
            seen_states.add(state_hash)
            seen_groups.add(group)
            if len(chosen) >= int(max_per_phase):
                break

        for local_index, row in chosen:
            global_index = local_index if phase_index == 0 else upstream_count + local_index
            selected.append(
                TubeBoundaryAnchor(
                    phase=phase,
                    phase_index=phase_index,
                    entry_index=int(local_index),
                    global_index=int(global_index),
                    row=row,
                )
            )
        phase_audit[phase] = {
            "support_count": len(phase_rows),
            "eligible_support_count": len(eligible),
            "eligible_parent_group_count": len(
                {str(row["parent_group_id"]) for _index, row in eligible}
            ),
            "excluded_above_score_ceiling_count": len(phase_rows) - len(eligible),
            "selected_count": len(chosen),
            "selected": [
                {
                    "entry_index": int(local_index),
                    "global_index": int(
                        local_index if phase_index == 0 else upstream_count + local_index
                    ),
                    "state_sha256": str(row["state_sha256"]),
                    "parent_group_id": str(row["parent_group_id"]),
                    "value_score": float(row["value_score"]),
                    "sampling_weight": float(row["sampling_weight"]),
                    "role": str(row.get("role", "")),
                    "source_bank": str(row.get("source_bank", "")),
                }
                for local_index, row in chosen
            ],
        }

    audit = {
        "schema": "jit_unified_boundary_anchor_audit_v1",
        "status": "completed",
        "split": "train",
        "selection": "bootstrap_score_at_or_below_ceiling_parent_group_unique_state_unique",
        "anchor_semantics": "weak_bootstrap_frontier_probe_not_certified_boundary",
        "frontier_score_ceiling": ceiling,
        "max_per_phase": int(max_per_phase),
        "selected_anchor_count": len(selected),
        "source_tube_manifest_sha256": artifact.manifest["manifest_sha256"],
        "test_data_used": False,
        "validation_data_used": False,
        "by_phase": phase_audit,
    }
    return tuple(selected), audit


def collect_unified_boundary_candidates(
    anchors: Sequence[TubeBoundaryAnchor],
    output_dir: Path,
    *,
    env: Any,
    policy: Callable[[Any, Any], Any],
    policy_record: Mapping[str, Any],
    frozen_manifest_sha256: str,
    protocol_seed: int,
    frontier_score_ceiling: float = DEFAULT_FRONTIER_SCORE_CEILING,
    strengths: Sequence[float] = DEFAULT_UNIFIED_BOUNDARY_STRENGTHS,
    durations: Sequence[int] = DEFAULT_UNIFIED_BOUNDARY_DURATIONS,
    action_names: Sequence[str] | None = None,
    signs: Sequence[int] | None = None,
    active_action_dimensions: int = 1,
) -> dict[str, Any]:
    """Generate unlabeled phase-local reachable candidates under frozen pi_k."""
    strengths = validate_strengths(strengths)
    durations = validate_durations(durations)
    directions = action_sparse_directions(
        action_names=action_names,
        signs=signs,
        active_action_dimensions=active_action_dimensions,
    )
    width = int(active_action_dimensions)
    ceiling = float(frontier_score_ceiling)
    if not np.isfinite(ceiling) or not 0.0 <= ceiling <= 1.0:
        raise ValueError("frontier_score_ceiling must be finite and lie in [0, 1]")
    if not anchors:
        raise ValueError("unified boundary acquisition requires anchors")
    if any(anchor.phase not in ("upstream", "downstream") for anchor in anchors):
        raise ValueError("unsupported unified boundary anchor phase")
    if any(anchor.value_score > ceiling for anchor in anchors):
        raise ValueError("unified boundary anchor exceeds frontier score ceiling")
    parent_keys = [(anchor.phase, anchor.parent_group_id) for anchor in anchors]
    if len(parent_keys) != len(set(parent_keys)):
        raise ValueError("unified boundary anchors must be parent-group unique per phase")
    state_hashes = [anchor.state_sha256 for anchor in anchors]
    if len(state_hashes) != len(set(state_hashes)):
        raise ValueError("unified boundary anchors must be physical-state unique")
    if int(policy_record.get("iteration", -1)) < 0:
        raise ValueError("frozen unified policy iteration is invalid")
    if policy_record.get("policy_role") != "envelope_expansion_authority":
        raise ValueError("frozen unified policy is not an expansion authority")
    if policy_record.get("xml_sha256") != env._bundle.xml_sha256:
        raise ValueError("unified boundary policy/runtime XML mismatch")

    artifact = env.tube_pool.artifact
    _validate_source_tube(artifact)
    support_hashes = {str(row["state_sha256"]) for row in artifact.entries}
    if width == 1:
        selected_action_names = list(dict.fromkeys(row["action_name"] for row in directions))
        selected_signs = sorted({int(row["sign"]) for row in directions})
        direction_family = (
            "action_basis"
            if len(directions) == 2 * len(ACTION_ORDER)
            else "action_basis_subset"
        )
        direction_protocol: dict[str, Any] = {}
        state_generation = (
            "reset exact TRAIN Tube entry in authoritative unified runtime; compute deterministic "
            "frozen unified-policy action; apply bounded selected action-basis perturbation; "
            "advance only through env.step; reject terminal/nonfinite/phase-crossing states"
        )
    else:
        selected_action_names = list(
            dict.fromkeys(
                name
                for row in directions
                for name in row["action_names"]
            )
        )
        selected_signs = sorted(
            {
                int(sign)
                for row in directions
                for sign in row["signs"]
            }
        )
        direction_family = f"action_sparse_{width}"
        direction_protocol = {"active_action_dimensions": width}
        state_generation = (
            "reset exact TRAIN Tube entry in authoritative unified runtime; compute deterministic "
            "frozen unified-policy action; apply bounded selected sparse multi-axis action "
            "perturbation; advance only through env.step; reject terminal/nonfinite/phase-crossing "
            "states"
        )

    maximum_interactions = sum(
        int(duration)
        for _anchor in anchors
        for duration in durations
        for _strength in strengths
        for _direction in directions
    )
    anchor_identities = [
        {
            "phase": anchor.phase,
            "phase_index": int(anchor.phase_index),
            "entry_index": int(anchor.entry_index),
            "global_index": int(anchor.global_index),
            "parent_group_id": anchor.parent_group_id,
            "state_sha256": anchor.state_sha256,
            "bootstrap_value_score": anchor.value_score,
        }
        for anchor in sorted(
            anchors,
            key=lambda item: (
                item.phase_index,
                item.value_score,
                item.parent_group_id,
                item.state_sha256,
            ),
        )
    ]
    protocol = {
        "schema": UNIFIED_BOUNDARY_PROTOCOL_SCHEMA,
        "status": "predeclared",
        "purpose": "policy_conditioned_real_dynamics_frontier_acquisition",
        "split": "train",
        "iteration": int(policy_record["iteration"]),
        "policy_name": str(policy_record["name"]),
        "policy_actor_sha256": str(policy_record["actor_sha256"]),
        "policy_payload_sha256": str(policy_record["payload_sha256"]),
        "policy_formal_config_sha256": str(policy_record["formal_config_sha256"]),
        "frozen_unified_manifest_sha256": str(frozen_manifest_sha256),
        "source_tube_manifest_sha256": str(artifact.manifest["manifest_sha256"]),
        "source_tube_entry_count": int(artifact.manifest["entry_count"]),
        "anchor_count": len(anchors),
        "anchor_selection": {
            "rule": "bootstrap_score_at_or_below_ceiling_parent_group_unique_state_unique",
            "frontier_score_ceiling": ceiling,
            "parent_group_unique_per_phase": True,
            "physical_state_unique": True,
            "anchor_identities": anchor_identities,
        },
        "anchor_semantics": "weak_bootstrap_frontier_probe_not_certified_boundary",
        "protocol_seed": int(protocol_seed),
        "action_order": list(ACTION_ORDER),
        "direction_family": direction_family,
        **direction_protocol,
        "selected_action_names": selected_action_names,
        "selected_signs": selected_signs,
        "strengths": list(strengths),
        "durations": list(durations),
        "maximum_environment_interactions": int(maximum_interactions),
        "training_transitions": 0,
        "expert_switching_used": False,
        "test_data_used": False,
        "validation_data_used": False,
        "final_evaluation_data_used": False,
        "state_generation": state_generation,
        "claim_boundary": {
            "unlabeled_acquisition_only": True,
            "tube_expansion_claim": False,
            "jce_jel_claim": False,
            "certified_safe_set_claim": False,
        },
    }
    protocol_sha = canonical_sha256(protocol)
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
    bank = output / "boundary_bank"
    (bank / "snapshots").mkdir(parents=True, exist_ok=False)

    reset = jax.jit(env.reset_tube_index)
    step = jax.jit(env.step)
    base_key = jax.random.PRNGKey(int(protocol_seed))
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    exclusions = {
        "terminal": 0,
        "nonfinite": 0,
        "phase_transition": 0,
        "existing_support": 0,
        "duplicate": 0,
    }
    attempted = 0
    interactions = 0
    variant_index = 0

    for anchor in sorted(
        anchors,
        key=lambda item: (item.phase_index, item.value_score, item.parent_group_id, item.state_sha256),
    ):
        for duration in durations:
            for strength in strengths:
                for direction in directions:
                    attempted += 1
                    state = reset(np.int32(anchor.phase_index), np.int32(anchor.entry_index))
                    jax.block_until_ready(state)
                    if int(np.asarray(state.info["active_phase"])) != anchor.phase_index:
                        raise ValueError("unified boundary reset active-phase mismatch")
                    if _truth(state.info["expert_switching_used"]):
                        raise ValueError("unified boundary reset used expert switching")
                    if not _truth(state.info["reset_from_soft_tube"]):
                        raise ValueError("unified boundary anchor did not originate from Tube reset")
                    if int(np.asarray(state.info["tube_entry_index"])) != anchor.entry_index:
                        raise ValueError("unified boundary reset Tube entry identity mismatch")

                    nominal_actions: list[list[float]] = []
                    perturbed_actions: list[list[float]] = []
                    effective_deltas: list[list[float]] = []
                    rejected: str | None = None
                    current_variant = variant_index
                    variant_index += 1
                    for perturb_step in range(int(duration)):
                        variant_key = jax.random.fold_in(base_key, int(current_variant))
                        action_key = jax.random.fold_in(variant_key, int(perturb_step))
                        result = policy(state.obs, action_key)
                        nominal = result[0] if isinstance(result, tuple) else result
                        nominal = np.asarray(jax.device_get(nominal), dtype=np.float32).reshape(-1)
                        if nominal.shape != (len(ACTION_ORDER),) or not np.isfinite(nominal).all():
                            raise ValueError("frozen unified policy returned an invalid action")
                        requested = nominal + np.asarray(
                            direction["basis_vector"], dtype=np.float32
                        ) * np.float32(strength)
                        perturbed = np.clip(requested, -1.0, 1.0).astype(np.float32)
                        state = step(state, perturbed)
                        jax.block_until_ready(state)
                        interactions += 1
                        nominal_actions.append(nominal.tolist())
                        perturbed_actions.append(perturbed.tolist())
                        effective_deltas.append((perturbed - nominal).tolist())
                        if _truth(state.info["expert_switching_used"]):
                            raise ValueError("unified boundary acquisition used expert switching")
                        if not _finite_state(state, perturbed):
                            rejected = "nonfinite"
                            break
                        if _truth(state.done):
                            rejected = "terminal"
                            break
                        if int(np.asarray(state.info["active_phase"])) != anchor.phase_index:
                            rejected = "phase_transition"
                            break
                    if rejected is not None:
                        exclusions[rejected] += 1
                        continue

                    snapshot = capture_unified_envelope_snapshot(
                        state,
                        env=env,
                        parent_trajectory=anchor.parent_group_id,
                        parent_state_sha256=anchor.state_sha256,
                        config_sha256=str(policy_record["formal_config_sha256"]),
                        policy_actor_sha256=str(policy_record["actor_sha256"]),
                        policy_payload_sha256=str(policy_record["payload_sha256"]),
                        policy_iteration=int(policy_record["iteration"]),
                    )
                    state_hash = physical_state_sha256(snapshot)
                    if state_hash in support_hashes:
                        exclusions["existing_support"] += 1
                        continue
                    if state_hash in seen:
                        exclusions["duplicate"] += 1
                        continue
                    seen.add(state_hash)
                    relative = Path("snapshots") / f"candidate_{len(entries):06d}"
                    save_unified_envelope_snapshot(bank / relative, snapshot)
                    qpos = np.asarray(snapshot.qpos)
                    qvel = np.asarray(snapshot.qvel)
                    metrics = state.metrics
                    entries.append(
                        {
                            "candidate_id": f"pi{policy_record['iteration']}_{anchor.phase}_{len(entries):06d}",
                            "candidate_kind": "reachable_unified_frontier_probe",
                            "split": "train",
                            "phase": anchor.phase,
                            "phase_index": anchor.phase_index,
                            "snapshot": str(relative),
                            "source_bank": "boundary_bank",
                            "state_sha256": state_hash,
                            "parent_group_id": anchor.parent_group_id,
                            "parent_state_sha256": anchor.state_sha256,
                            "anchor_entry_index": anchor.entry_index,
                            "anchor_global_index": anchor.global_index,
                            "anchor_value_score": anchor.value_score,
                            "anchor_sampling_weight": float(anchor.row["sampling_weight"]),
                            "anchor_role": str(anchor.row.get("role", "")),
                            "anchor_source_bank": str(anchor.row.get("source_bank", "")),
                            "policy_iteration": int(policy_record["iteration"]),
                            "policy_actor_sha256": str(policy_record["actor_sha256"]),
                            "policy_payload_sha256": str(policy_record["payload_sha256"]),
                            "protocol_sha256": protocol_sha,
                            "perturbation": {
                                **direction,
                                "strength": float(strength),
                                "duration": int(duration),
                                "variant_index": int(current_variant),
                                "nominal_actions": nominal_actions,
                                "perturbed_actions": perturbed_actions,
                                "effective_deltas": effective_deltas,
                            },
                            "episode_step": snapshot.episode_step,
                            "phase_episode_step": snapshot.phase_episode_step,
                            "x": float(qpos[0]),
                            "z": float(qpos[2]),
                            "vx": float(qvel[0]),
                            "vz": float(qvel[2]),
                            "roll": float(np.asarray(jax.device_get(metrics["signal/roll"]))),
                            "pitch": float(np.asarray(jax.device_get(metrics["signal/pitch"]))),
                            "yaw": float(np.asarray(jax.device_get(metrics["signal/yaw"]))),
                        }
                    )

    if interactions > maximum_interactions:
        raise ValueError("unified boundary acquisition exceeded its predeclared interaction ceiling")
    report = {
        "schema": UNIFIED_BOUNDARY_CATALOG_SCHEMA,
        "status": "completed",
        "artifact_role": "unlabeled_policy_conditioned_frontier_candidates",
        "split": "train",
        "iteration": int(policy_record["iteration"]),
        "policy_name": str(policy_record["name"]),
        "policy_actor_sha256": str(policy_record["actor_sha256"]),
        "policy_payload_sha256": str(policy_record["payload_sha256"]),
        "frozen_unified_manifest_sha256": str(frozen_manifest_sha256),
        "source_tube_manifest_sha256": str(artifact.manifest["manifest_sha256"]),
        "protocol_sha256": protocol_sha,
        "frontier_score_ceiling": ceiling,
        "anchor_count": len(anchors),
        "attempted_candidate_count": attempted,
        "candidate_count": len(entries),
        "environment_interactions": interactions,
        "maximum_environment_interactions": maximum_interactions,
        "training_transitions": 0,
        "expert_switching_used": False,
        "test_data_used": False,
        "validation_data_used": False,
        "final_evaluation_data_used": False,
        "exclusion_counts": exclusions,
        "claim_boundary": {
            "unlabeled_acquisition_only": True,
            "tube_expansion_claim": False,
            "jce_jel_claim": False,
            "certified_safe_set_claim": False,
        },
        "entries": entries,
    }
    (output / "catalog.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    summary = {key: value for key, value in report.items() if key != "entries"}
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report