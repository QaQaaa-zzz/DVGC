"""TRAIN-only downstream transition-band refinement after duration bracketing.

This stage is entered only after the coarse duration-only search has established
that upstream is ready while downstream remains label-degenerate and long
perturbation bursts are being clipped by episode termination.  It refines the
known downstream bracket on the contiguous integer grid 17..32 ticks.

A terminal-causing perturbation is never used directly as a continuation label.
Instead, the last finite nonterminal state before that terminal transition is
saved as a real-dynamics candidate, and the frozen unified policy is restarted
from that exact state for the authoritative continuation label.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Callable, Mapping, Sequence

import jax
import numpy as np

from .checkpoint import load_checkpoint
from .config import file_sha256
from .constants import ACTION_ORDER, END_REASONS
from .ppo import make_checkpoint_policy
from .soft_tube import load_soft_tube
from .unified_boundary import TubeBoundaryAnchor, select_tube_boundary_anchors
from .unified_continuation_labels import label_unified_continuations
from .unified_envelope_snapshot import (
    capture_unified_envelope_snapshot,
    physical_state_sha256,
    save_unified_envelope_snapshot,
)
from .unified_formal import build_unified_formal_environment, load_unified_formal_config
from .unified_policy_freeze import load_frozen_unified_manifest
from .unified_training import checkpoint_identity
from .unified_transition_band_search import (
    phase_transition_band_readiness,
    unique_label_rows,
)
from .upstream_boundary import action_basis_directions, validate_strengths


DOWNSTREAM_REFINEMENT_CONFIG_SCHEMA = "jit_downstream_transition_refinement_config_v1"
DOWNSTREAM_REFINEMENT_PROTOCOL_SCHEMA = "jit_downstream_transition_refinement_protocol_v1"
DOWNSTREAM_REFINEMENT_SUMMARY_SCHEMA = "jit_downstream_transition_refinement_summary_v1"
BOUNDARY_PROTOCOL_SCHEMA = "jit_unified_boundary_protocol_v1"
BOUNDARY_CATALOG_SCHEMA = "jit_unified_boundary_catalog_v1"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()


def _repository_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("downstream refinement requires a Git checkout") from exc


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


def load_downstream_refinement_config(path: Path) -> dict[str, Any]:
    raw = _read_json(path)
    if raw.get("schema") != DOWNSTREAM_REFINEMENT_CONFIG_SCHEMA:
        raise ValueError("unsupported downstream refinement config schema")
    if int(raw.get("iteration", -1)) != 0:
        raise ValueError("current downstream refinement is locked to iteration 0")
    if not str(raw.get("frozen_policy", "")) or not str(raw.get("output_dir", "")):
        raise ValueError("downstream refinement requires frozen_policy and output_dir")

    prior = raw.get("prior_search")
    if not isinstance(prior, Mapping):
        raise ValueError("downstream refinement requires prior_search")
    for key in (
        "root",
        "expected_status",
        "expected_protocol_sha256",
        "expected_accumulated_unique_label_count",
        "expected_upstream",
        "expected_downstream",
    ):
        if key not in prior:
            raise ValueError(f"prior_search missing {key}")
    if prior["expected_status"] != "search_exhausted":
        raise ValueError("downstream refinement must inherit an exhausted coarse search")
    if prior["expected_upstream"].get("ready") is not True:
        raise ValueError("downstream refinement requires upstream already ready")
    if prior["expected_downstream"].get("ready") is not False:
        raise ValueError("downstream refinement requires downstream unresolved")

    acquisition = raw.get("fixed_acquisition")
    if not isinstance(acquisition, Mapping):
        raise ValueError("downstream refinement requires fixed_acquisition")
    if acquisition.get("phase") != "downstream":
        raise ValueError("downstream refinement may probe downstream only")
    if int(acquisition.get("anchors_per_phase", 0)) <= 0:
        raise ValueError("anchors_per_phase must be positive")
    ceiling = float(acquisition.get("frontier_score_ceiling", np.nan))
    if not np.isfinite(ceiling) or not 0.0 <= ceiling <= 1.0:
        raise ValueError("frontier_score_ceiling must lie in [0, 1]")
    if tuple(acquisition.get("action_names", ())) != ACTION_ORDER:
        raise ValueError("downstream refinement must preserve canonical action order")
    if tuple(int(x) for x in acquisition.get("signs", ())) != (-1, 1):
        raise ValueError("downstream refinement signs must remain [-1, +1]")
    if tuple(float(x) for x in acquisition.get("strengths", ())) != (0.025, 0.05, 0.10):
        raise ValueError("downstream refinement strengths must remain fixed")
    grid = tuple(int(x) for x in acquisition.get("duration_grid", ()))
    if grid != tuple(range(17, 33)):
        raise ValueError("downstream refinement duration_grid must be contiguous 17..32")
    clipping = acquisition.get("terminal_clipping", {})
    if clipping != {
        "enabled": True,
        "capture": "last_finite_nonterminal_state_before_terminal",
        "terminal_outcome_is_not_the_continuation_label": True,
    }:
        raise ValueError("downstream refinement terminal-clipping contract drift")
    for key in ("acquisition_protocol_seed_base", "label_protocol_seed_base"):
        if int(acquisition.get(key, -1)) < 0:
            raise ValueError(f"{key} must be nonnegative")

    continuation = raw.get("continuation_labeling")
    if not isinstance(continuation, Mapping):
        raise ValueError("downstream refinement requires continuation_labeling")
    if int(continuation.get("max_ticks", 0)) != 400:
        raise ValueError("downstream refinement must preserve 400-tick continuation horizon")
    if int(continuation.get("branches_per_candidate", 0)) != 1:
        raise ValueError("deterministic downstream refinement requires one branch")

    readiness = raw.get("readiness")
    if not isinstance(readiness, Mapping):
        raise ValueError("downstream refinement requires readiness")
    for key in (
        "minimum_positive_candidates",
        "minimum_negative_candidates",
        "minimum_parent_groups_with_positive",
        "minimum_parent_groups_with_negative",
    ):
        if int(readiness.get(key, 0)) <= 0:
            raise ValueError(f"readiness {key} must be positive")

    expected_claims = {
        "training_only_search": True,
        "upstream_transition_band_frozen": True,
        "continuation_field_trained": False,
        "tube_1_constructed": False,
        "pi_1_trained": False,
        "jce_jel_claim": False,
        "certified_safe_set_claim": False,
    }
    if raw.get("claim_boundary") != expected_claims:
        raise ValueError("downstream refinement claim boundary drift")
    return raw


def _validate_prior_search(
    config: Mapping[str, Any], policy_record: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prior = config["prior_search"]
    root = Path(prior["root"])
    summary = _read_json(root / "summary.json")
    accumulated = _read_json(root / "accumulated_train_labels.json")
    rows = accumulated.get("entries")
    if not isinstance(rows, list):
        raise ValueError("prior accumulated TRAIN labels must contain entries")
    if summary.get("status") != prior["expected_status"]:
        raise ValueError("prior search status drift")
    if summary.get("protocol_sha256") != prior["expected_protocol_sha256"]:
        raise ValueError("prior search protocol SHA-256 drift")
    if int(summary.get("accumulated_unique_label_count", -1)) != int(
        prior["expected_accumulated_unique_label_count"]
    ):
        raise ValueError("prior accumulated label count drift")
    if len(rows) != int(prior["expected_accumulated_unique_label_count"]):
        raise ValueError("prior accumulated label file count drift")
    if summary.get("policy_actor_sha256") != policy_record["actor_sha256"]:
        raise ValueError("prior search actor identity drift")
    if summary.get("policy_payload_sha256") != policy_record["payload_sha256"]:
        raise ValueError("prior search payload identity drift")
    if summary.get("source_tube_manifest_sha256") != config["source_tube_manifest_sha256"]:
        raise ValueError("prior search source Tube drift")
    for phase, key in (("upstream", "expected_upstream"), ("downstream", "expected_downstream")):
        actual = summary.get("readiness", {}).get(phase)
        expected = prior[key]
        if actual != expected:
            raise ValueError(f"prior search {phase} readiness drift")
    unique = unique_label_rows([rows])
    if len(unique) != len(rows):
        raise ValueError("prior accumulated labels contain duplicate physical states")
    if any(row.get("split") != "train" for row in rows):
        raise ValueError("prior search contains non-TRAIN labels")
    identity = {
        "root": str(root),
        "summary_file_sha256": file_sha256(root / "summary.json"),
        "accumulated_labels_file_sha256": file_sha256(root / "accumulated_train_labels.json"),
        "protocol_sha256": summary["protocol_sha256"],
        "accumulated_unique_label_count": len(rows),
        "readiness": summary["readiness"],
    }
    return [dict(row) for row in rows], identity


def _collect_duration_candidates(
    anchors: Sequence[TubeBoundaryAnchor],
    output_dir: Path,
    *,
    duration: int,
    env: Any,
    policy: Callable[[Any, Any], Any],
    policy_record: Mapping[str, Any],
    frozen_manifest_sha256: str,
    protocol_seed: int,
    frontier_score_ceiling: float,
    strengths: Sequence[float],
    action_names: Sequence[str],
    signs: Sequence[int],
    excluded_state_hashes: set[str],
) -> dict[str, Any]:
    strengths = validate_strengths(strengths)
    directions = action_basis_directions(action_names=action_names, signs=signs)
    if int(duration) <= 0:
        raise ValueError("downstream refinement duration must be positive")
    if not anchors or any(anchor.phase != "downstream" or anchor.phase_index != 1 for anchor in anchors):
        raise ValueError("downstream refinement requires downstream-only anchors")
    if any(anchor.value_score > float(frontier_score_ceiling) for anchor in anchors):
        raise ValueError("downstream refinement anchor exceeds score ceiling")

    artifact = env.tube_pool.artifact
    support_hashes = {str(row["state_sha256"]) for row in artifact.entries}
    max_interactions = len(anchors) * len(strengths) * len(directions) * int(duration)
    anchor_identities = [
        {
            "phase": anchor.phase,
            "phase_index": anchor.phase_index,
            "entry_index": anchor.entry_index,
            "global_index": anchor.global_index,
            "parent_group_id": anchor.parent_group_id,
            "state_sha256": anchor.state_sha256,
            "bootstrap_value_score": anchor.value_score,
        }
        for anchor in anchors
    ]
    protocol_base = {
        "schema": BOUNDARY_PROTOCOL_SCHEMA,
        "status": "predeclared",
        "purpose": "downstream_terminal_clipped_local_refinement",
        "split": "train",
        "iteration": int(policy_record["iteration"]),
        "policy_name": policy_record["name"],
        "policy_actor_sha256": policy_record["actor_sha256"],
        "policy_payload_sha256": policy_record["payload_sha256"],
        "policy_formal_config_sha256": policy_record["formal_config_sha256"],
        "frozen_unified_manifest_sha256": frozen_manifest_sha256,
        "source_tube_manifest_sha256": artifact.manifest["manifest_sha256"],
        "source_tube_entry_count": int(artifact.manifest["entry_count"]),
        "anchor_count": len(anchors),
        "anchor_selection": {
            "rule": "bootstrap_score_at_or_below_ceiling_parent_group_unique_state_unique",
            "frontier_score_ceiling": float(frontier_score_ceiling),
            "parent_group_unique_per_phase": True,
            "physical_state_unique": True,
            "anchor_identities": anchor_identities,
        },
        "anchor_semantics": "weak_bootstrap_frontier_probe_not_certified_boundary",
        "protocol_seed": int(protocol_seed),
        "action_order": list(ACTION_ORDER),
        "direction_family": "action_basis",
        "selected_action_names": list(action_names),
        "selected_signs": list(signs),
        "strengths": list(strengths),
        "durations": [int(duration)],
        "terminal_clipping": {
            "enabled": True,
            "capture": "last_finite_nonterminal_state_before_terminal",
            "terminal_outcome_is_not_the_continuation_label": True,
        },
        "maximum_environment_interactions": max_interactions,
        "training_transitions": 0,
        "expert_switching_used": False,
        "test_data_used": False,
        "validation_data_used": False,
        "final_evaluation_data_used": False,
        "state_generation": (
            "reset exact downstream TRAIN Tube entry; apply frozen-policy action plus bounded "
            "action-basis perturbation through authoritative env.step; if the perturbation "
            "causes terminal, capture the last finite nonterminal predecessor instead of "
            "treating that terminal outcome as the continuation label"
        ),
        "claim_boundary": {
            "unlabeled_acquisition_only": True,
            "tube_expansion_claim": False,
            "jce_jel_claim": False,
            "certified_safe_set_claim": False,
        },
    }
    protocol_sha = _canonical_sha256(protocol_base)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "protocol.json", {**protocol_base, "protocol_sha256": protocol_sha})
    bank = output / "boundary_bank"
    (bank / "snapshots").mkdir(parents=True, exist_ok=False)

    reset = jax.jit(env.reset_tube_index)
    step = jax.jit(env.step)
    base_key = jax.random.PRNGKey(int(protocol_seed))
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    exclusions = Counter()
    terminal_probe_outcomes = Counter()
    interactions = 0
    attempted = 0
    variant_index = 0

    for anchor in anchors:
        for strength in strengths:
            for direction in directions:
                attempted += 1
                state = reset(np.int32(anchor.phase_index), np.int32(anchor.entry_index))
                jax.block_until_ready(state)
                if _integer(state.info["active_phase"]) != 1:
                    raise ValueError("downstream refinement reset phase mismatch")
                if _truth(state.info["expert_switching_used"]):
                    raise ValueError("downstream refinement reset used expert switching")
                nominal_actions: list[list[float]] = []
                perturbed_actions: list[list[float]] = []
                effective_deltas: list[list[float]] = []
                terminal_clipped = False
                terminal_meta: dict[str, Any] | None = None
                rejected: str | None = None
                snapshot_nonterminal_steps = 0
                current_variant = variant_index
                variant_index += 1

                for perturb_step in range(int(duration)):
                    previous_state = state
                    variant_key = jax.random.fold_in(base_key, int(current_variant))
                    action_key = jax.random.fold_in(variant_key, int(perturb_step))
                    result = policy(previous_state.obs, action_key)
                    nominal = result[0] if isinstance(result, tuple) else result
                    nominal = np.asarray(jax.device_get(nominal), dtype=np.float32).reshape(-1)
                    if nominal.shape != (len(ACTION_ORDER),) or not np.isfinite(nominal).all():
                        raise ValueError("frozen unified policy returned invalid refinement action")
                    requested = nominal + np.asarray(direction["basis_vector"], dtype=np.float32) * np.float32(strength)
                    perturbed = np.clip(requested, -1.0, 1.0).astype(np.float32)
                    stepped = step(previous_state, perturbed)
                    jax.block_until_ready(stepped)
                    interactions += 1
                    nominal_actions.append(nominal.tolist())
                    perturbed_actions.append(perturbed.tolist())
                    effective_deltas.append((perturbed - nominal).tolist())
                    if _truth(stepped.info["expert_switching_used"]):
                        raise ValueError("downstream refinement used expert switching")
                    if not _finite_state(stepped, perturbed):
                        rejected = "nonfinite"
                        break
                    if _integer(stepped.info["active_phase"]) != 1:
                        rejected = "phase_transition"
                        break
                    if _truth(stepped.done):
                        if not _finite_state(previous_state):
                            rejected = "nonfinite_preterminal"
                            break
                        state = previous_state
                        terminal_clipped = True
                        snapshot_nonterminal_steps = int(perturb_step)
                        end_code = _integer(stepped.info["end_code"])
                        terminal_meta = {
                            "end_code": end_code,
                            "end_reason": END_REASONS.get(end_code, f"unknown_{end_code}"),
                            "terminal_success": _truth(stepped.info["success"]),
                            "physical_failure": _truth(stepped.info["physical_failure"]),
                            "timeout": _truth(stepped.info["timeout"]),
                            "executed_interactions": int(perturb_step + 1),
                        }
                        terminal_probe_outcomes[terminal_meta["end_reason"]] += 1
                        break
                    state = stepped
                    snapshot_nonterminal_steps = int(perturb_step + 1)

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
                if state_hash in excluded_state_hashes:
                    exclusions["previously_labeled_state"] += 1
                    continue
                if state_hash in seen:
                    exclusions["duplicate"] += 1
                    continue
                seen.add(state_hash)
                excluded_state_hashes.add(state_hash)
                relative = Path("snapshots") / f"candidate_{len(entries):06d}"
                save_unified_envelope_snapshot(bank / relative, snapshot)
                qpos = np.asarray(snapshot.qpos)
                qvel = np.asarray(snapshot.qvel)
                metrics = state.metrics
                entries.append(
                    {
                        "candidate_id": f"pi{policy_record['iteration']}_downstream_d{duration}_{len(entries):06d}",
                        "candidate_kind": "reachable_unified_frontier_probe",
                        "split": "train",
                        "phase": "downstream",
                        "phase_index": 1,
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
                            "terminal_clipped": bool(terminal_clipped),
                            "snapshot_nonterminal_steps": int(snapshot_nonterminal_steps),
                            "executed_interactions": len(perturbed_actions),
                            "terminal_probe_outcome": terminal_meta,
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

    if interactions > max_interactions:
        raise ValueError("downstream refinement acquisition exceeded interaction ceiling")
    report = {
        "schema": BOUNDARY_CATALOG_SCHEMA,
        "status": "completed",
        "artifact_role": "unlabeled_policy_conditioned_frontier_candidates",
        "split": "train",
        "iteration": int(policy_record["iteration"]),
        "policy_name": policy_record["name"],
        "policy_actor_sha256": policy_record["actor_sha256"],
        "policy_payload_sha256": policy_record["payload_sha256"],
        "frozen_unified_manifest_sha256": frozen_manifest_sha256,
        "source_tube_manifest_sha256": artifact.manifest["manifest_sha256"],
        "protocol_sha256": protocol_sha,
        "frontier_score_ceiling": float(frontier_score_ceiling),
        "anchor_count": len(anchors),
        "attempted_candidate_count": attempted,
        "candidate_count": len(entries),
        "environment_interactions": interactions,
        "maximum_environment_interactions": max_interactions,
        "terminal_clipped_candidate_count": sum(
            int(row["perturbation"]["terminal_clipped"]) for row in entries
        ),
        "terminal_probe_outcomes": dict(sorted(terminal_probe_outcomes.items())),
        "exclusion_counts": dict(sorted(exclusions.items())),
        "training_transitions": 0,
        "expert_switching_used": False,
        "test_data_used": False,
        "validation_data_used": False,
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


def search_downstream_transition_refinement(
    config_path: Path,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    config_path = Path(config_path)
    config = load_downstream_refinement_config(config_path)
    output = Path(config["output_dir"])
    frozen_path = Path(config["frozen_policy"])
    frozen = load_frozen_unified_manifest(frozen_path)
    policy_record = frozen["policy"]
    if int(policy_record["iteration"]) != int(config["iteration"]):
        raise ValueError("downstream refinement policy iteration drift")

    prior_labels, prior_identity = _validate_prior_search(config, policy_record)
    formal = load_unified_formal_config(Path(policy_record["formal_config"]))
    artifact = load_soft_tube(Path(formal.soft_tube_path))
    if artifact.manifest["manifest_sha256"] != config["source_tube_manifest_sha256"]:
        raise ValueError("downstream refinement source Tube identity drift")
    anchors_all, anchor_audit = select_tube_boundary_anchors(
        artifact,
        max_per_phase=int(config["fixed_acquisition"]["anchors_per_phase"]),
        frontier_score_ceiling=float(config["fixed_acquisition"]["frontier_score_ceiling"]),
    )
    anchors = tuple(anchor for anchor in anchors_all if anchor.phase == "downstream")
    if not anchors:
        raise ValueError("downstream refinement selected no downstream anchors")

    grid = tuple(int(x) for x in config["fixed_acquisition"]["duration_grid"])
    directions = len(config["fixed_acquisition"]["action_names"]) * len(
        config["fixed_acquisition"]["signs"]
    )
    strengths = len(config["fixed_acquisition"]["strengths"])
    maximum_acquisition = len(anchors) * directions * strengths * sum(grid)
    maximum_labeling = (
        len(anchors)
        * directions
        * strengths
        * len(grid)
        * int(config["continuation_labeling"]["max_ticks"])
    )
    protocol_base = {
        "schema": DOWNSTREAM_REFINEMENT_PROTOCOL_SCHEMA,
        "status": "predeclared",
        "purpose": "downstream_contiguous_terminal_clipped_transition_band_refinement",
        "repository_head": _repository_head(),
        "config_path": str(config_path),
        "config_file_sha256": file_sha256(config_path),
        "iteration": int(config["iteration"]),
        "policy_name": policy_record["name"],
        "policy_actor_sha256": policy_record["actor_sha256"],
        "policy_payload_sha256": policy_record["payload_sha256"],
        "frozen_policy_file_sha256": file_sha256(frozen_path),
        "source_tube_manifest_sha256": artifact.manifest["manifest_sha256"],
        "prior_search": prior_identity,
        "upstream_transition_band_frozen": True,
        "anchor_audit": anchor_audit,
        "downstream_anchor_identities": [
            {
                "entry_index": a.entry_index,
                "global_index": a.global_index,
                "parent_group_id": a.parent_group_id,
                "state_sha256": a.state_sha256,
                "bootstrap_value_score": a.value_score,
            }
            for a in anchors
        ],
        "fixed_acquisition": config["fixed_acquisition"],
        "continuation_labeling": config["continuation_labeling"],
        "readiness": config["readiness"],
        "maximum_acquisition_environment_interactions": maximum_acquisition,
        "maximum_labeling_environment_interactions": maximum_labeling,
        "training_transitions": 0,
        "expert_switching_used": False,
        "validation_data_used": False,
        "test_data_used": False,
        "final_evaluation_data_used": False,
        "claim_boundary": config["claim_boundary"],
    }
    protocol_sha = _canonical_sha256(protocol_base)
    protocol = {**protocol_base, "protocol_sha256": protocol_sha}

    if output.exists():
        if not resume:
            raise FileExistsError(f"downstream refinement output already exists: {output}")
        existing = _read_json(output / "search_protocol.json")
        if existing != protocol:
            raise ValueError("cannot resume downstream refinement under a different protocol")
        if (output / "summary.json").exists():
            done = _read_json(output / "summary.json")
            if done.get("status") in ("transition_band_ready", "search_exhausted"):
                return done
    else:
        output.mkdir(parents=True, exist_ok=False)
        _write_json(output / "search_protocol.json", protocol)

    if jax.default_backend() != "gpu":
        raise RuntimeError("downstream refinement requires the visible JAX GPU")
    runtime_config, runtime_artifact, env = build_unified_formal_environment(
        Path(policy_record["formal_config"])
    )
    if runtime_config.config_sha256 != formal.config_sha256:
        raise ValueError("downstream refinement runtime config drift")
    if runtime_artifact.manifest["manifest_sha256"] != artifact.manifest["manifest_sha256"]:
        raise ValueError("downstream refinement runtime Tube drift")
    if env._bundle.xml_sha256 != policy_record["xml_sha256"]:
        raise ValueError("downstream refinement runtime XML drift")
    payload = load_checkpoint(
        Path(policy_record["checkpoint"]), expected=checkpoint_identity(runtime_config, env)
    )
    if file_sha256(Path(policy_record["checkpoint"]) / "payload.pkl") != policy_record[
        "payload_sha256"
    ]:
        raise ValueError("downstream refinement checkpoint payload drift")
    policy = jax.jit(make_checkpoint_policy(env, payload, deterministic=True))

    label_sets: list[list[dict[str, Any]]] = [prior_labels]
    excluded_state_hashes = {str(row["state_sha256"]) for row in prior_labels}
    duration_reports: list[dict[str, Any]] = []
    acquisition_interactions = 0
    labeling_interactions = 0

    completed_duration_count = 0
    for duration in grid:
        duration_root = output / f"duration_{duration:02d}"
        labels_path = duration_root / "labels" / "labels.json"
        label_summary_path = duration_root / "labels" / "summary.json"
        if duration_root.exists():
            if not labels_path.exists() or not label_summary_path.exists():
                raise ValueError(f"cannot resume incomplete downstream duration {duration}")
            label_summary = _read_json(label_summary_path)
            if label_summary.get("status") != "completed":
                raise ValueError(f"cannot resume non-completed downstream duration {duration}")
            rows = json.loads(labels_path.read_text(encoding="utf-8"))
            if not isinstance(rows, list) or len(rows) != int(label_summary["label_count"]):
                raise ValueError(f"downstream duration {duration} label count drift")
            catalog = _read_json(duration_root / "acquisition" / "catalog.json")
            acquisition_summary = _read_json(duration_root / "acquisition" / "summary.json")
            label_sets.append([dict(row) for row in rows])
            excluded_state_hashes.update(str(row["state_sha256"]) for row in rows)
            acquisition_interactions += int(acquisition_summary["environment_interactions"])
            labeling_interactions += int(label_summary["environment_interactions"])
            accumulated = unique_label_rows(label_sets)
            readiness = phase_transition_band_readiness(accumulated, config["readiness"])
            duration_reports.append(
                {
                    "duration": duration,
                    "status": "completed",
                    "resumed": True,
                    "candidate_count": len(rows),
                    "positive_count": int(label_summary["positive_count"]),
                    "negative_count": int(label_summary["negative_count"]),
                    "terminal_clipped_candidate_count": int(
                        acquisition_summary.get("terminal_clipped_candidate_count", 0)
                    ),
                    "terminal_probe_outcomes": acquisition_summary.get(
                        "terminal_probe_outcomes", {}
                    ),
                    "downstream_readiness_after_duration": readiness["downstream"],
                }
            )
            completed_duration_count += 1
            if readiness["downstream"]["ready"]:
                break
            continue
        break

    accumulated = unique_label_rows(label_sets)
    readiness = phase_transition_band_readiness(accumulated, config["readiness"])

    if not readiness["downstream"]["ready"]:
        for duration in grid[completed_duration_count:]:
            duration_root = output / f"duration_{duration:02d}"
            if duration_root.exists():
                raise ValueError(f"unexpected incomplete duration directory: {duration_root}")
            acquisition_dir = duration_root / "acquisition"
            acquisition = _collect_duration_candidates(
                anchors,
                acquisition_dir,
                duration=duration,
                env=env,
                policy=policy,
                policy_record=policy_record,
                frozen_manifest_sha256=file_sha256(frozen_path),
                protocol_seed=int(config["fixed_acquisition"]["acquisition_protocol_seed_base"])
                + duration,
                frontier_score_ceiling=float(config["fixed_acquisition"]["frontier_score_ceiling"]),
                strengths=tuple(float(x) for x in config["fixed_acquisition"]["strengths"]),
                action_names=tuple(config["fixed_acquisition"]["action_names"]),
                signs=tuple(int(x) for x in config["fixed_acquisition"]["signs"]),
                excluded_state_hashes=excluded_state_hashes,
            )
            acquisition_interactions += int(acquisition["environment_interactions"])
            if int(acquisition["candidate_count"]) == 0:
                duration_report = {
                    "duration": duration,
                    "status": "no_candidates",
                    "candidate_count": 0,
                    "terminal_clipped_candidate_count": 0,
                    "terminal_probe_outcomes": acquisition.get("terminal_probe_outcomes", {}),
                    "exclusion_counts": acquisition.get("exclusion_counts", {}),
                }
                duration_reports.append(duration_report)
                _write_json(duration_root / "duration_summary.json", duration_report)
                continue

            labels_dir = duration_root / "labels"
            label_report = label_unified_continuations(
                acquisition_dir / "catalog.json",
                labels_dir,
                env=env,
                policy=policy,
                policy_record=policy_record,
                frozen_manifest_sha256=file_sha256(frozen_path),
                max_ticks=int(config["continuation_labeling"]["max_ticks"]),
                protocol_seed=int(config["fixed_acquisition"]["label_protocol_seed_base"])
                + duration,
            )
            labeling_interactions += int(label_report["environment_interactions"])
            rows = json.loads((labels_dir / "labels.json").read_text(encoding="utf-8"))
            label_sets.append([dict(row) for row in rows])
            accumulated = unique_label_rows(label_sets)
            readiness = phase_transition_band_readiness(accumulated, config["readiness"])
            duration_report = {
                "duration": duration,
                "status": "completed",
                "candidate_count": int(label_report["candidate_count"]),
                "positive_count": int(label_report["positive_count"]),
                "negative_count": int(label_report["negative_count"]),
                "terminal_clipped_candidate_count": int(
                    acquisition.get("terminal_clipped_candidate_count", 0)
                ),
                "terminal_probe_outcomes": acquisition.get("terminal_probe_outcomes", {}),
                "exclusion_counts": acquisition.get("exclusion_counts", {}),
                "downstream_readiness_after_duration": readiness["downstream"],
            }
            duration_reports.append(duration_report)
            _write_json(duration_root / "duration_summary.json", duration_report)
            _write_json(
                output / "progress.json",
                {
                    "schema": DOWNSTREAM_REFINEMENT_SUMMARY_SCHEMA,
                    "status": "searching",
                    "protocol_sha256": protocol_sha,
                    "completed_duration_count": len(duration_reports),
                    "readiness": readiness,
                    "durations": duration_reports,
                    "training_transitions": 0,
                },
            )
            if readiness["downstream"]["ready"]:
                break

    accumulated = unique_label_rows(label_sets)
    readiness = phase_transition_band_readiness(accumulated, config["readiness"])
    downstream_ready = bool(readiness["downstream"]["ready"])
    if readiness["upstream"]["ready"] is not True:
        raise ValueError("downstream refinement unexpectedly lost upstream readiness")
    status = "transition_band_ready" if downstream_ready else "search_exhausted"
    _write_json(output / "accumulated_train_labels.json", {"entries": accumulated})
    summary = {
        "schema": DOWNSTREAM_REFINEMENT_SUMMARY_SCHEMA,
        "status": status,
        "protocol_sha256": protocol_sha,
        "iteration": int(config["iteration"]),
        "policy_name": policy_record["name"],
        "policy_actor_sha256": policy_record["actor_sha256"],
        "policy_payload_sha256": policy_record["payload_sha256"],
        "source_tube_manifest_sha256": artifact.manifest["manifest_sha256"],
        "prior_accumulated_unique_label_count": len(prior_labels),
        "accumulated_unique_label_count": len(accumulated),
        "readiness": readiness,
        "durations": duration_reports,
        "acquisition_environment_interactions": acquisition_interactions,
        "labeling_environment_interactions": labeling_interactions,
        "training_transitions": 0,
        "expert_switching_used": False,
        "validation_data_used": False,
        "test_data_used": False,
        "final_evaluation_data_used": False,
        "next_scientific_gate": (
            "freeze both phase TRAIN transition bands and design group-disjoint expansion "
            "validation before fitting/calibrating C_up^0 and C_down^0"
            if downstream_ready
            else "stop and make a new explicit downstream acquisition-method decision"
        ),
        "claim_boundary": config["claim_boundary"],
    }
    _write_json(output / "summary.json", summary)
    return summary
