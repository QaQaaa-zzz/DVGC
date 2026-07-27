"""Build a provisional Descent RSI bank from deterministic MJX lineages."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import jax
import mujoco
import numpy as np

from dvgc.bank import SnapshotBank
from dvgc.config import (
    ACTION_MAPPING_VERSION, STAGE_ID, config_hash, file_sha256, load_config,
)
from dvgc.continuous import DescentSupportMatcher, load_trajectory
from dvgc.env import END_REASON, OrangeBikeDVGC
from dvgc.provisional_descent import (
    FEATURE_NAMES, SCHEMA_VERSION, greedy_clusters, state_identity,
    tolerance_unique, validate_candidate,
)
from dvgc.reset_geometry import GroundSupportSolver
from dvgc.rollout import restore_snapshot
from dvgc.runtime import save_json


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _trajectory_identity(actions: np.ndarray) -> str:
    return _sha_bytes(np.ascontiguousarray(actions, np.float32).tobytes())


def _state_row_identity(qpos, qvel, ctrl, history) -> str:
    return _sha_bytes(b"".join(
        np.ascontiguousarray(value).tobytes()
        for value in (qpos, qvel, ctrl, history)
    ))


def _smoke_inventory(root: Path, tolerance_decimals: int = 5) -> dict[str, Any]:
    files = sorted((root / "trajectories").glob("episode_*.npz"))
    trajectory_file_hashes, state_hashes, tolerance_hashes = [], [], []
    descent_ticks, lengths = [], []
    for path in files:
        trajectory_file_hashes.append(_sha_bytes(path.read_bytes()))
        arrays, _ = load_trajectory(path)
        lengths.append(len(arrays["qpos"]))
        descent_ticks.append(int(np.sum(arrays["pipeline_phase"] == 4)))
        for index in range(len(arrays["qpos"])):
            values = (
                arrays["qpos"][index], arrays["qvel"][index],
                arrays["ctrl"][index], arrays["observation_history"][index],
            )
            state_hashes.append(_state_row_identity(*values))
            tolerance_hashes.append(_state_row_identity(*(
                np.round(value, tolerance_decimals) for value in values
            )))
    return {
        "trajectory_files": len(files),
        "trajectory_byte_unique": len(set(trajectory_file_hashes)),
        "state_rows": len(state_hashes),
        "state_byte_unique": len(set(state_hashes)),
        "state_tolerance_unique": len(set(tolerance_hashes)),
        "trajectory_lengths": sorted(set(lengths)),
        "descent_ticks_per_trajectory": {
            "min": min(descent_ticks, default=0),
            "max": max(descent_ticks, default=0),
            "values": sorted(set(descent_ticks)),
        },
        "evidence_role": (
            "repeatability_only" if len(set(trajectory_file_hashes)) <= 1
            else "repeatability_and_state_diversity"
        ),
    }


def _margins(feature: np.ndarray, cfg, contact: dict[str, Any]) -> dict[str, float]:
    return {
        "roll_margin_rad": float(np.deg2rad(float(cfg.max_roll_deg)) - abs(feature[3])),
        "pitch_margin_rad": float(np.deg2rad(float(cfg.max_pitch_deg)) - abs(feature[4])),
        "roll_rate_margin": float(cfg.recovery_max_angvel - abs(feature[9])),
        "pitch_rate_margin": float(cfg.recovery_max_angvel - abs(feature[10])),
        "nonwheel_clearance_m": float(contact["nonwheel"]),
        "wheel_clearance_m": float(contact["wheel_min"]),
        "minimum_penetration_m": float(contact["minimum_penetration"]),
    }


def _minimum_margin_score(margins: dict[str, float], cfg) -> float:
    scales = {
        "roll_margin_rad": np.deg2rad(float(cfg.max_roll_deg)),
        "pitch_margin_rad": np.deg2rad(float(cfg.max_pitch_deg)),
        "roll_rate_margin": float(cfg.recovery_max_angvel),
        "pitch_rate_margin": float(cfg.recovery_max_angvel),
        "nonwheel_clearance_m": .05,
    }
    return float(min(margins[name] / scale for name, scale in scales.items()))


def _probe_candidate(env, step, row, protocol, seed) -> dict[str, Any]:
    base_action = np.asarray(row["policy_state"]["last_action"], np.float32)
    horizons = tuple(int(value) for value in protocol["probe_horizons"])
    maximum = max(horizons)
    results = {}
    best_survival = 0
    for index, (name, delta) in enumerate(protocol["probe_actions"].items()):
        state = restore_snapshot(env, row, jax.random.PRNGKey(seed + index))
        action = (
            base_action if name == "nominal"
            else np.zeros_like(base_action) if name == "neutral"
            else np.clip(base_action + np.asarray(delta, np.float32), -1.0, 1.0)
        )
        survived = 0
        initial_feature = np.asarray(env._physical_feature(state.data), np.float64)
        best_attitude = float(abs(initial_feature[3]) + abs(initial_feature[4]))
        reason = "active_at_24"
        phases = [int(state.info["estimated_phase"])]
        for tick in range(1, maximum + 1):
            state = step(state, action)
            phases.append(int(state.info["estimated_phase"]))
            feature = np.asarray(env._physical_feature(state.data), np.float64)
            best_attitude = min(best_attitude, float(abs(feature[3]) + abs(feature[4])))
            if not (np.isfinite(np.asarray(state.data.qpos)).all()
                    and np.isfinite(np.asarray(state.data.qvel)).all()):
                reason = "nonfinite"; break
            if float(state.done):
                reason = END_REASON.get(int(state.info["end_code"]), "unknown")
                break
            survived = tick
        best_survival = max(best_survival, survived)
        results[name] = {
            "action": action.tolist(), "survived_ticks": survived,
            "survived_8": survived >= 8, "survived_16": survived >= 16,
            "survived_24": survived >= 24, "termination_reason": reason,
            "best_attitude_abs_sum_rad": best_attitude,
            "attitude_improvement_rad": float(
                abs(initial_feature[3]) + abs(initial_feature[4]) - best_attitude
            ),
            "phase_monotonic": all(
                current >= previous for previous, current in zip(phases, phases[1:])
            ),
        }
    nominal = results["nominal"]
    return {
        "probes": results,
        "maximum_survival_ticks": best_survival,
        "any_survived_8": any(row["survived_8"] for row in results.values()),
        "any_survived_16": any(row["survived_16"] for row in results.values()),
        "any_survived_24": any(row["survived_24"] for row in results.values()),
        "nominal_survival_ticks": nominal["survived_ticks"],
        "action_authority_nonzero": any(
            value["survived_ticks"] > nominal["survived_ticks"]
            or value["attitude_improvement_rad"] > nominal["attitude_improvement_rad"] + 1e-5
            for name, value in results.items() if name not in ("nominal", "neutral")
        ),
    }


def _assess_candidate(env, step, geometry, matcher, source, protocol, cfg, seed):
    row = copy.deepcopy(source)
    feature = np.asarray(row["physical_feature"], np.float64)
    rejection = None
    arrays = [row[name] for name in ("qpos", "qvel", "ctrl", "qacc_warmstart")]
    if not all(np.isfinite(np.asarray(value)).all() for value in arrays):
        rejection = "nonfinite"
    elif int(row.get("oracle_phase", -1)) != STAGE_ID["flight"]:
        rejection = "wrong_oracle_phase"
    elif int(row.get("policy_state", {}).get("filter_phase", -1)) not in (
        STAGE_ID["takeoff"], STAGE_ID["flight"],
    ):
        rejection = "invalid_filtered_phase"
    elif not bool(row.get("apex_seen")) or feature[8] >= 0.0:
        rejection = "not_physical_descent"
    contact = geometry.measure(row["qpos"], row["qvel"], row["ctrl"])
    margins = _margins(feature, cfg, contact)
    if rejection is None and contact["body_contacts"]:
        rejection = "body_terrain_contact"
    elif rejection is None and contact["wheel_contacts"]:
        rejection = "wheel_terrain_contact"
    elif rejection is None and contact["minimum_penetration"] < float(protocol["deep_penetration_m"]):
        rejection = "deep_penetration"
    elif rejection is None and contact["nonwheel"] < float(protocol["minimum_nonwheel_clearance_m"]):
        rejection = "insufficient_body_clearance"
    attitude_floor = np.deg2rad(float(protocol["minimum_attitude_margin_deg"]))
    if rejection is None and min(margins["roll_margin_rad"], margins["pitch_margin_rad"]) <= attitude_floor:
        rejection = "insufficient_attitude_margin"
    if rejection is None and min(margins["roll_rate_margin"], margins["pitch_rate_margin"]) <= float(protocol["minimum_angular_rate_margin"]):
        rejection = "insufficient_angular_rate_margin"

    reset_replay = {
        "qpos_exact": False, "qvel_exact": False, "deterministic": False,
        "shock_free": False,
    }
    reset_replay["saved_filter_phase"] = int(
        row.get("policy_state", {}).get("filter_phase", -1)
    )
    trainability = None
    if rejection is None:
        replay_actions = []
        replay_traces = []
        for repeat in range(2):
            state = restore_snapshot(env, row, jax.random.PRNGKey(seed))
            reset_replay["restored_filter_phase"] = int(state.info["estimated_phase"])
            reset_replay["qpos_exact"] = bool(np.array_equal(
                np.asarray(state.data.qpos, np.float32), np.asarray(row["qpos"], np.float32)
            ))
            reset_replay["qvel_exact"] = bool(np.array_equal(
                np.asarray(state.data.qvel, np.float32), np.asarray(row["qvel"], np.float32)
            ))
            action = np.asarray(row["policy_state"]["last_action"], np.float32)
            trace = []
            for _ in range(int(protocol["reset_shock_steps"])):
                state = step(state, action)
                trace.append(np.concatenate([
                    np.asarray(state.data.qpos), np.asarray(state.data.qvel)
                ]))
                if float(state.done):
                    break
            replay_actions.append(action)
            replay_traces.append(np.asarray(trace))
        reset_replay["deterministic"] = bool(
            np.array_equal(replay_actions[0], replay_actions[1])
            and np.array_equal(replay_traces[0], replay_traces[1])
        )
        reset_replay["shock_free"] = bool(
            len(replay_traces[0]) == int(protocol["reset_shock_steps"])
        )
        if not reset_replay["qpos_exact"] or not reset_replay["qvel_exact"]:
            rejection = "snapshot_restore_mismatch"
        elif reset_replay["restored_filter_phase"] != reset_replay["saved_filter_phase"]:
            rejection = "phase_detector_restore_mismatch"
        elif not reset_replay["deterministic"]:
            rejection = "snapshot_replay_nondeterministic"
        elif not reset_replay["shock_free"]:
            rejection = "reset_shock"
    if rejection is None:
        trainability = _probe_candidate(env, step, row, protocol, seed + 1000)
        if not trainability["probes"]["nominal"]["phase_monotonic"]:
            rejection = "phase_detector_regression"
        elif not trainability["any_survived_8"]:
            rejection = "no_eight_tick_control_window"
        elif not trainability["action_authority_nonzero"] and trainability["maximum_survival_ticks"] < 16:
            rejection = "no_local_action_authority"

    _, legacy_distance = matcher.evaluate(
        restore_snapshot(env, row, jax.random.PRNGKey(seed + 2000)),
        apex_crossed=True,
    )
    label = "rejected_physical"
    if rejection is None:
        label = (
            "provisional_core" if trainability["any_survived_16"]
            else "provisional_frontier"
        )
    row.update({
        "candidate_schema": SCHEMA_VERSION,
        "artifact_role": "proposal_support_bank",
        "formal_tube_member": False,
        "formal_jel_member": False,
        "provisional_label": label,
        "candidate_kind": label,
        "training_only": True,
        "bootstrap_eligible": rejection is None,
        "contact_state": {
            "wheel_terrain_contacts": int(contact["wheel_contacts"]),
            "body_terrain_contacts": int(contact["body_contacts"]),
            "pairs": list(contact["pairs"]),
        },
        "safety_margins": margins,
        "minimum_margin_score": _minimum_margin_score(margins, cfg),
        "reset_replay": reset_replay,
        "short_horizon_trainability": trainability,
        "legacy_support_distance": float(legacy_distance),
        "legacy_matcher_member": bool(legacy_distance <= matcher.radius),
        "legacy_recoverable_reference": None,
        "rejection_reason": rejection,
    })
    if rejection is None:
        validate_candidate(row)
    return row


def _capture(state, env, source_id, source_path, source_kind, tick, apex_tick):
    row = env.snapshot_record(state, "flight")
    feature = np.asarray(row["physical_feature"], np.float64)
    row.update({
        "candidate_source": source_kind,
        "source_trajectory_id": source_id,
        "source_trajectory_path": str(source_path),
        "source_parent_id": source_id,
        "event_relative_tick": int(tick - apex_tick),
        "apex_relative_tick": int(tick - apex_tick),
        "descent_entry_relative_tick": int(tick - apex_tick),
        "terrain_relative_x": float(feature[0] - env._config.step_front_x),
        "terrain_relative_z": float(feature[2] - env._config.step_top_z),
        "apex_seen": 1,
        "oracle_phase": STAGE_ID["flight"],
    })
    row["id"] = state_identity({
        **row,
        "candidate_schema": SCHEMA_VERSION,
        "provisional_label": "provisional_frontier",
        "descent_layer": "early",
        "candidate_source": source_kind,
        "artifact_role": "proposal_support_bank",
        "formal_tube_member": False,
        "formal_jel_member": False,
    })[:32]
    return row


def _replay_actions(env, step, actions, seed, source_path, source_kind,
                    approach_ticks=0, expected=None):
    state = env.reset(jax.random.PRNGKey(seed))
    for _ in range(int(approach_ticks)):
        state = step(state, np.asarray([0., 1., 0., 0.], np.float32))
    if expected is not None and not np.array_equal(
        np.asarray(state.data.qpos, np.float32), np.asarray(expected[0], np.float32)
    ):
        raise RuntimeError(f"initial replay mismatch: {source_path}")
    source_id = _trajectory_identity(np.asarray(actions, np.float32))
    previous_vz = float(state.data.qvel[2])
    positive_seen = previous_vz > 0.0
    apex_tick = None
    captures = []
    for index, action in enumerate(actions, 1):
        state = step(state, np.clip(action, -1.0, 1.0))
        if expected is not None and not np.array_equal(
            np.asarray(state.data.qpos, np.float32), np.asarray(expected[index], np.float32)
        ):
            raise RuntimeError(f"trajectory replay mismatch at {index}: {source_path}")
        vz = float(state.data.qvel[2])
        positive_seen |= vz > 0.0
        if apex_tick is None and positive_seen and previous_vz > 0.0 and vz <= 0.0:
            apex_tick = index
        previous_vz = vz
        if apex_tick is not None and vz < 0.0 and not float(state.done):
            captures.append(_capture(
                state, env, source_id, source_path, source_kind, index, apex_tick
            ))
        if float(state.done):
            break
    terminal_tick = index
    total = len(captures)
    for position, row in enumerate(captures):
        fraction = position / max(total - 1, 1)
        row["descent_layer"] = "early" if fraction < 1 / 3 else "middle" if fraction < 2 / 3 else "late"
        row["ticks_to_source_failure"] = int(terminal_tick - (apex_tick + row["event_relative_tick"]))
    return captures, {
        "source_trajectory_id": source_id, "captured_descent_ticks": total,
        "apex_tick": apex_tick, "terminal_tick": terminal_tick,
        "termination_reason": END_REASON.get(int(state.info["end_code"]), "unknown"),
    }


def _source_candidates(env, step, smoke_root, bridge_root, bridge_seed):
    candidates, lineages, seen_actions = [], [], set()
    smoke_files = sorted((smoke_root / "trajectories").glob("episode_*.npz"))
    for path in smoke_files:
        arrays, _ = load_trajectory(path)
        actions = arrays["action"][1:]
        identity = _trajectory_identity(actions)
        if identity in seen_actions:
            continue
        seen_actions.add(identity)
        marker = smoke_root / "episodes" / f"{path.stem}.json"
        seed = int(json.loads(marker.read_text())["seed"])
        rows, info = _replay_actions(
            env, step, actions, seed, path, "natural_continuous",
            expected=arrays["qpos"],
        )
        candidates.extend(rows); lineages.append(info)
    for path in sorted(bridge_root.glob("local_search_v*/best_natural_trajectory.npz")):
        arrays, _ = load_trajectory(path)
        actions = arrays["action"][1:]
        identity = _trajectory_identity(actions)
        if identity in seen_actions:
            continue
        seen_actions.add(identity)
        report = json.loads((path.parent / "report.json").read_text())
        approach_ticks = int(report["best"]["approach_ticks"])
        rows, info = _replay_actions(
            env, step, actions, bridge_seed, path, "natural_continuous",
            approach_ticks=approach_ticks, expected=arrays["qpos"],
        )
        candidates.extend(rows); lineages.append(info)
    return candidates, lineages


def _local_candidates(env, step, parents, protocol, seed):
    rows = []
    maximum = int(protocol["maximum_candidates"])
    for parent_index, parent in enumerate(parents[:int(protocol["maximum_local_parents"])]):
        for action_name, delta in protocol["local_perturbation_actions"].items():
            state = restore_snapshot(
                env, parent, jax.random.PRNGKey(seed + parent_index * 100 + len(rows))
            )
            base = np.asarray(parent["policy_state"]["last_action"], np.float32)
            action = np.clip(base + np.asarray(delta, np.float32), -1.0, 1.0)
            for tick in range(1, max(protocol["local_perturbation_ticks"]) + 1):
                state = step(state, action)
                if float(state.done):
                    break
                if tick not in protocol["local_perturbation_ticks"]:
                    continue
                feature = np.asarray(env._physical_feature(state.data), np.float64)
                if feature[8] >= 0.0:
                    continue
                row = _capture(
                    state, env,
                    f"local:{parent['id']}:{action_name}",
                    parent["source_trajectory_path"],
                    "local_rsi_perturbation", tick, 0,
                )
                row.update({
                    "source_parent_id": parent["id"],
                    "descent_layer": parent["descent_layer"],
                    "local_perturbation": {
                        "action_name": action_name, "action": action.tolist(),
                        "ticks": tick, "seed": seed + parent_index * 100 + len(rows),
                    },
                    "ticks_to_source_failure": None,
                })
                rows.append(row)
                if len(rows) >= maximum:
                    return rows
    return rows


def _balanced(rows, maximum):
    buckets = defaultdict(list)
    for row in rows:
        buckets[(row["provisional_label"], row["descent_layer"], row["candidate_source"])].append(row)
    chosen = []
    keys = sorted(buckets)
    while keys and len(chosen) < maximum:
        remaining = []
        for key in keys:
            if buckets[key] and len(chosen) < maximum:
                chosen.append(buckets[key].pop(0))
            if buckets[key]:
                remaining.append(key)
        keys = remaining
    return chosen


def _assign_layers(rows):
    """Split each valid lineage's own controllable Descent window in thirds."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["source_trajectory_id"]].append(row)
    for group in groups.values():
        group.sort(key=lambda row: (row["event_relative_tick"], row["id"]))
        for index, row in enumerate(group):
            fraction = index / max(len(group) - 1, 1)
            row["descent_layer"] = (
                "early" if fraction < 1 / 3
                else "middle" if fraction < 2 / 3 else "late"
            )


def _ranges(rows):
    if not rows:
        return {}
    values = np.asarray([row["physical_feature"] for row in rows], np.float64)
    return {
        name: {"min": float(values[:, index].min()),
               "max": float(values[:, index].max()),
               "mean": float(values[:, index].mean())}
        for index, name in enumerate(FEATURE_NAMES)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-root", required=True)
    parser.add_argument("--bridge-root", required=True)
    parser.add_argument("--stage-support", required=True)
    parser.add_argument("--protocol", default="configs/descent_candidate_bank_v1.json")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--output-bank", required=True)
    parser.add_argument("--output-report", required=True)
    args = parser.parse_args()
    for output in (Path(args.output_bank), Path(args.output_report)):
        if output.exists():
            raise SystemExit(f"refusing to overwrite {output}")
    protocol = json.loads(Path(args.protocol).read_text())
    cfg = load_config(args.config, {
        "training_stage": "flight", "use_bank_resets": False,
        "domain_randomization": False, "obs_noise_enable": False,
        "stage_reachability_objective": "", "expert_chain_termination": False,
    })
    support = SnapshotBank.load(args.stage_support)
    env = OrangeBikeDVGC(cfg, snapshot_bank=SnapshotBank(), stage_support_bank=support)
    step = jax.jit(env.step)
    matcher = DescentSupportMatcher(env)
    geometry = GroundSupportSolver(cfg.xml_path)
    smoke_root, bridge_root = Path(args.smoke_root), Path(args.bridge_root)
    inventory = _smoke_inventory(smoke_root)
    natural_raw, lineages = _source_candidates(
        env, step, smoke_root, bridge_root, int(protocol["bridge_replay_seed"])
    )
    rejections = Counter()
    natural_assessed = []
    for index, row in enumerate(natural_raw):
        assessed = _assess_candidate(
            env, step, geometry, matcher, row, protocol, cfg,
            int(protocol["source_seed"]) + index * 10_000,
        )
        if assessed["rejection_reason"]:
            rejections[assessed["rejection_reason"]] += 1
        natural_assessed.append(assessed)
    natural_eligible = [row for row in natural_assessed if row["rejection_reason"] is None]
    _assign_layers(natural_eligible)
    parent_order = sorted(natural_eligible, key=lambda row: (
        row["descent_layer"], -row["short_horizon_trainability"]["maximum_survival_ticks"],
        row["source_trajectory_id"], row["event_relative_tick"],
    ))
    local_raw = _local_candidates(
        env, step, parent_order, protocol, int(protocol["source_seed"]) + 50_000_000
    )
    local_assessed = []
    for index, row in enumerate(local_raw):
        assessed = _assess_candidate(
            env, step, geometry, matcher, row, protocol, cfg,
            int(protocol["source_seed"]) + 60_000_000 + index * 10_000,
        )
        if assessed["rejection_reason"]:
            rejections[assessed["rejection_reason"]] += 1
        local_assessed.append(assessed)
    all_assessed = natural_assessed + local_assessed
    eligible = [row for row in all_assessed if row["rejection_reason"] is None]
    byte_seen, byte_unique = set(), []
    for row in eligible:
        identity = state_identity(row)
        if identity in byte_seen:
            rejections["byte_exact_duplicate"] += 1
            continue
        byte_seen.add(identity); byte_unique.append(row)
    tolerance_rows, tolerance_rejected = tolerance_unique(
        byte_unique, protocol["tolerance_feature_scale"], protocol["tolerance_distance"]
    )
    rejections["tolerance_duplicate"] += tolerance_rejected
    selected = _balanced(tolerance_rows, int(protocol["maximum_candidates"]))
    assignments = greedy_clusters(
        selected, protocol["tolerance_feature_scale"], protocol["cluster_radius"]
    )
    for row, cluster in zip(selected, assignments):
        row["candidate_cluster"] = int(cluster)
    counts = Counter(row["provisional_label"] for row in selected)
    for row in selected:
        mass = (float(protocol["core_reset_mass"])
                if row["provisional_label"] == "provisional_core"
                else float(protocol["frontier_reset_mass"]))
        row.update({
            "reset_source": "flight_curriculum",
            "reset_weight": mass / max(counts[row["provisional_label"]], 1),
        })
    metadata = {
        "artifact_role": "proposal_support_bank",
        "candidate_schema": SCHEMA_VERSION,
        "safe_claim_allowed": False,
        "certified_tube": False,
        "formal_jel": False,
        "legacy_support_is_gate": False,
        "reset_source_protocol": {
            "version": "descent_provisional_rsi_v1",
            "core_mass": protocol["core_reset_mass"],
            "frontier_mass": protocol["frontier_reset_mass"],
            "stratified_by": ["provisional_label", "descent_layer"],
        },
        "xml_sha256": file_sha256(cfg.xml_path),
        "config_sha256": file_sha256(args.config),
        "effective_config_hash": config_hash(cfg),
        "action_mapping_version": ACTION_MAPPING_VERSION,
        "runtime_solver": env._effective_mjx_solver,
        "protocol": protocol,
        "source_hashes": {
            "stage_support": file_sha256(args.stage_support),
            "protocol": file_sha256(args.protocol),
        },
    }
    bank = SnapshotBank(selected, metadata)
    bank.save(args.output_bank)
    source_counts = Counter(row["candidate_source"] for row in selected)
    layer_counts = Counter(row["descent_layer"] for row in selected)
    survival = {
        str(horizon): sum(
            any(probe[f"survived_{horizon}"] for probe in row["short_horizon_trainability"]["probes"].values())
            for row in selected
        ) for horizon in protocol["probe_horizons"]
    }
    report = {
        "status": "PASS" if selected and counts["provisional_core"] and len(set(assignments)) > 1 else "FAIL",
        "artifact_role": "descent_provisional_candidate_bank_construction",
        "old_bridge_gate": "superseded_as_training_gate",
        "old_support_entry_required": False,
        "landing_or_final_required": False,
        "ppo_authorization": False,
        "smoke_inventory": inventory,
        "lineages": lineages,
        "raw_states": len(all_assessed),
        "natural_raw_states": len(natural_assessed),
        "local_raw_states": len(local_assessed),
        "byte_unique_eligible": len(byte_unique),
        "tolerance_unique_eligible": len(tolerance_rows),
        "selected_candidates": len(selected),
        "provisional_labels": dict(counts),
        "rejected": len([row for row in all_assessed if row["rejection_reason"] is not None]),
        "rejection_reasons": dict(sorted(rejections.items())),
        "descent_layers": dict(layer_counts),
        "candidate_sources": dict(source_counts),
        "source_fractions": {name: count / len(selected) for name, count in source_counts.items()},
        "short_horizon_survivability": survival,
        "reset_replay_pass": sum(row["reset_replay"]["deterministic"] and row["reset_replay"]["shock_free"] for row in selected),
        "candidate_clusters": len(set(assignments)),
        "feature_ranges": _ranges(selected),
        "legacy_reference": {
            "matcher_member_count": sum(row["legacy_matcher_member"] for row in selected),
            "distance_min": min((row["legacy_support_distance"] for row in selected), default=None),
            "used_as_gate": False,
        },
        "bank_path": str(Path(args.output_bank).resolve()),
        "bank_sha256": file_sha256(args.output_bank),
        "provenance": metadata,
    }
    save_json(args.output_report, report)
    print(json.dumps({
        "status": report["status"], "raw": report["raw_states"],
        "selected": report["selected_candidates"], "labels": report["provisional_labels"],
        "layers": report["descent_layers"], "sources": report["candidate_sources"],
        "clusters": report["candidate_clusters"], "rejections": report["rejection_reasons"],
    }, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(40)


if __name__ == "__main__":
    main()
