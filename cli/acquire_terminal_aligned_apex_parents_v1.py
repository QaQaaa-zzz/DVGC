"""Outcome-complete pilot for terminal-aligned valid Apex proposals."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import jax
import numpy as np

from cli.acquire_ascent_apex_parents import _local_action
from cli.build_descent_terminal_targets_from_tube_v1 import FEATURES, INDEX
from cli.runtime_gate import source_fingerprint
from cli.stage_label_pilot import sample_from_state
from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256, load_config
from dvgc.env import END_REASON, OrangeBikeDVGC
from dvgc.rollout import restore_snapshot
from dvgc.runtime import save_json
from dvgc.stage_reachability import evaluate_entry
from dvgc.trajectory_mining import canonical_state_byte_hash


ENTRIES = Path("runs/stage_next_reset_v3_seed0_20260723/ascent/independent_parent_acquisition_v2_24/fresh_ascent_entries.pkl")
TERMINAL = Path("runs/apex_bridge_cd_v2_reprobe/terminal_targets_v4.pkl")
DEFAULT_RUN = Path("runs/apex_terminal_aligned_parent_acquisition_v1/pilot_4x24")
SEED = 3_860_000_000


def pilot_specs() -> list[dict]:
    return [{
        "round": "terminal_aligned_pilot", "hip_amplitude": hip,
        "knee_ratio": ratio, "start_tick": start, "duration": duration,
    } for start in (0, 3) for duration in (16, 28, 44)
      for hip in (0.70, 0.85) for ratio in (0.35, 0.50)]


def low_impulse_specs() -> list[dict]:
    """Evidence-bounded refinement below the exhausted pilot impulse range."""
    return [{
        "round": "terminal_aligned_low_impulse", "hip_amplitude": hip,
        "knee_ratio": ratio, "start_tick": start, "duration": duration,
    } for start in (0, 3) for duration in (8, 12)
      for hip in (0.45, 0.55, 0.65) for ratio in (0.35, 0.50)]


def micro_impulse_specs() -> list[dict]:
    """Short, low-amplitude braking probe after one-step overcontrol evidence."""
    return [{
        "round": "terminal_aligned_micro_impulse", "hip_amplitude": hip,
        "knee_ratio": ratio, "start_tick": start, "duration": duration,
    } for start in (0, 1) for duration in (2, 4)
      for hip in (0.10, 0.20, 0.30) for ratio in (0.35, 0.70)]


def feedback_specs() -> list[dict]:
    """Low-dimensional pitch-rate/vertical-speed feedback screening."""
    return [{
        "round": "terminal_aligned_feedback", "controller_kind": "feedback",
        "wy_gain": wy_gain, "vz_gain": vz_gain, "bias": bias,
        "knee_ratio": ratio, "action_limit": .8,
    } for wy_gain in (.01, .02) for vz_gain in (.05, .10)
      for bias in (0., .10) for ratio in (.35, .70)]


def proposal_action(spec: dict, tick: int, feature: np.ndarray) -> jax.Array:
    if spec.get("controller_kind") != "feedback":
        return _local_action(spec, tick)
    hip = (float(spec["bias"]) + float(spec["wy_gain"]) * float(feature[10])
           + float(spec["vz_gain"]) * (float(feature[8]) - .25))
    limit = float(spec["action_limit"])
    hip = float(np.clip(hip, -limit, limit))
    knee = float(np.clip(hip * float(spec["knee_ratio"]), -limit, limit))
    return jax.numpy.asarray([0., 0., hip, knee], jax.numpy.float32)


def select_parent_entries(records: list[dict], count: int) -> list[dict]:
    groups = defaultdict(list)
    for row in records:
        groups[row.get("upstream_source_kind", "unknown")].append(row)
    for rows in groups.values():
        rows.sort(key=lambda row: (row["trajectory_parent_id"], row["id"]))
    selected = []
    keys = sorted(groups)
    while len(selected) < count and any(groups.values()):
        for key in keys:
            if groups[key] and len(selected) < count:
                selected.append(groups[key].pop(0))
    if len(selected) != count:
        raise ValueError(f"insufficient parent entries: {len(selected)}/{count}")
    if len({row["trajectory_parent_id"] for row in selected}) != count:
        raise ValueError("selected entries are not parent-disjoint")
    return selected


def select_nearest_supported_entries(
    records: list[dict], count: int, supported_parent_ids: set[str],
    excluded_parent_ids: set[str],
) -> tuple[list[dict], list[float]]:
    """Rank unseen parents by physical distance to valid-Apex support."""
    supported = [row for row in records
                 if str(row["trajectory_parent_id"]) in supported_parent_ids]
    if not supported:
        raise ValueError("no supported parent entries found")
    features = np.asarray([row["physical_feature"] for row in records], float)
    floor = np.asarray([
        .05, .05, .05, .05, .05, .05, .2, .2, .2,
        .2, .2, .2, .1, .1, .2, .2,
    ])
    scale = np.maximum(np.std(features, axis=0), floor)
    support_features = np.asarray(
        [row["physical_feature"] for row in supported], float,
    )
    ranked = []
    for row in records:
        parent = str(row["trajectory_parent_id"])
        if parent in supported_parent_ids or parent in excluded_parent_ids:
            continue
        feature = np.asarray(row["physical_feature"], float)
        distance = float(np.min(np.linalg.norm(
            (support_features - feature[None, :]) / scale[None, :], axis=1,
        )))
        ranked.append((distance, parent, str(row["id"]), row))
    ranked.sort(key=lambda item: item[:3])
    if len(ranked) < count:
        raise ValueError(f"insufficient unseen parent entries: {len(ranked)}/{count}")
    selected = ranked[:count]
    return [item[3] for item in selected], [item[0] for item in selected]


def terminal_distance(feature: np.ndarray, target: np.ndarray, center: np.ndarray, scale: np.ndarray) -> float:
    query = np.asarray([feature[INDEX[name]] for name in FEATURES], float)
    return float(np.min(np.linalg.norm(target - ((query - center) / scale)[None, :], axis=1)))


def run_spec(env, step, row, spec, seed, target, center, scale, horizon):
    state = restore_snapshot(env, row, jax.random.PRNGKey(seed))
    previous_vz = float(np.asarray(state.data.qvel[2]))
    feedback_feature = np.asarray(row["physical_feature"], float)
    minimum_residual = float("inf")
    for tick in range(horizon):
        action = proposal_action(spec, tick, feedback_feature)
        state = step(state, action)
        sample = sample_from_state(env, state, previous_vz)
        feature = np.asarray(sample["physical_feature"], float)
        feedback_feature = feature
        entry = evaluate_entry("ascent", sample, env._config)
        minimum_residual = min(minimum_residual, abs(feature[8]) + max(0., .4015-feature[2]) + max(0., feature[2]-.7015))
        if entry["valid"]:
            snapshot = env.snapshot_record(state, "flight")
            pose_margin = min(
                np.deg2rad(35.) - abs(feature[3]), np.deg2rad(75.) - abs(feature[4]),
                4. - np.linalg.norm(feature[9:12]),
            )
            return {
                "success": True, "entry_tick": tick + 1, "failure_reason": None,
                "terminal_distance": terminal_distance(feature, target, center, scale),
                "pose_margin": float(pose_margin), "physical_feature": feature.tolist(),
                "minimum_apex_residual": float(minimum_residual), "snapshot": snapshot,
            }
        if float(np.asarray(state.done)) > .5:
            return {
                "success": False, "entry_tick": None,
                "failure_reason": END_REASON.get(int(np.asarray(state.info["end_code"])), "unknown"),
                "terminal_distance": None, "pose_margin": None,
                "minimum_apex_residual": float(minimum_residual), "snapshot": None,
            }
        previous_vz = float(feature[8])
    return {"success": False, "entry_tick": None, "failure_reason": "horizon_exhaustion",
            "terminal_distance": None, "pose_margin": None,
            "minimum_apex_residual": float(minimum_residual), "snapshot": None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entries", default=str(ENTRIES))
    parser.add_argument("--terminal", default=str(TERMINAL))
    parser.add_argument("--run", default=str(DEFAULT_RUN))
    parser.add_argument("--parents", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--selection", choices=("source_round_robin", "nearest_supported", "explicit"),
                        default="source_round_robin")
    parser.add_argument("--support-report")
    parser.add_argument("--exclude-manifest")
    parser.add_argument("--parent-id", action="append", default=[])
    parser.add_argument("--spec-profile", choices=("pilot", "low_impulse", "micro_impulse", "feedback"),
                        default="pilot")
    args = parser.parse_args()
    root = Path(args.run)
    if root.exists():
        raise SystemExit(f"refusing overwrite {root}")
    gate = json.loads(Path("docs/RUNTIME_GATE.json").read_text())
    if gate.get("status") != "PASS" or gate.get("source_fingerprint") != source_fingerprint(Path.cwd()):
        raise SystemExit("runtime gate stale")
    entries_path, terminal_path = Path(args.entries), Path(args.terminal)
    entries = SnapshotBank.load(entries_path)
    terminal = SnapshotBank.load(terminal_path)
    if terminal.metadata.get("artifact_role") != "proposal_terminal_targets_from_certified_tube":
        raise SystemExit("terminal targets are not Tube-locked")
    selection_distances = None
    supported_parent_ids: set[str] = set()
    excluded_parent_ids: set[str] = set()
    if args.selection == "explicit":
        requested = list(map(str, args.parent_id))
        if len(requested) != args.parents or len(set(requested)) != args.parents:
            raise SystemExit("explicit selection requires one unique --parent-id per parent")
        by_parent = {str(row["trajectory_parent_id"]): row for row in entries.records}
        missing = sorted(set(requested) - set(by_parent))
        if missing:
            raise SystemExit(f"explicit parent IDs not found: {missing}")
        selected = [by_parent[parent] for parent in requested]
    elif args.selection == "nearest_supported":
        if not args.support_report:
            raise SystemExit("--support-report is required for nearest_supported")
        support_report = json.loads(Path(args.support_report).read_text())
        supported_parent_ids = set(map(str, support_report["successful_parent_ids"]))
        if args.exclude_manifest:
            exclude_manifest = json.loads(Path(args.exclude_manifest).read_text())
            excluded_parent_ids.update(map(str, exclude_manifest["parents"]))
        selected, selection_distances = select_nearest_supported_entries(
            entries.records, args.parents, supported_parent_ids, excluded_parent_ids,
        )
    else:
        selected = select_parent_entries(entries.records, args.parents)
    profiles = {"pilot": pilot_specs, "low_impulse": low_impulse_specs,
                "micro_impulse": micro_impulse_specs, "feedback": feedback_specs}
    specs = profiles[args.spec_profile]()
    center = np.asarray(terminal.metadata["normalization_center"], float)
    scale = np.asarray(terminal.metadata["normalization_scale"], float)
    target = np.asarray([[(row["physical_feature"][INDEX[name]] - center[i]) / scale[i]
                          for i, name in enumerate(FEATURES)] for row in terminal.records], float)
    cfg = load_config(args.config, {"training_stage": "flight", "use_bank_resets": False,
        "domain_randomization": False, "obs_noise_enable": False, "stage_reachability_objective": ""})
    env = OrangeBikeDVGC(cfg, snapshot_bank=SnapshotBank()); step = jax.jit(env.step)
    root.mkdir(parents=True)
    inputs = {"entries_sha256": file_sha256(entries_path), "terminal_sha256": file_sha256(terminal_path),
              "source_tube_sha256": terminal.metadata["source_tube_sha256"], "xml_sha256": file_sha256(cfg.xml_path),
              "seed": SEED}
    save_json(root / "manifest.json", {"status": "FROZEN_BEFORE_OUTCOMES", "inputs": inputs,
        "parents": [row["trajectory_parent_id"] for row in selected], "specs": specs,
        "selection": args.selection,
        "spec_profile": args.spec_profile,
        "selection_distances": selection_distances,
        "supported_parent_ids": sorted(supported_parent_ids),
        "excluded_parent_ids": sorted(excluded_parent_ids),
        "all_specs_evaluated_after_first_valid_apex": True})
    save_json(root / "cost_estimate.json", {"estimated_seconds": 1800, "parents": len(selected),
        "specs_per_parent": len(specs), "maximum_rollouts": len(selected)*len(specs)+2*len(selected),
        "fraction_of_24x81_grid": len(selected)*len(specs)/(24*81), "PPO_steps": 0})
    outcomes, best_records = [], []
    for parent_index, row in enumerate(selected):
        parent_rows = []
        snapshots = {}
        for spec_index, spec in enumerate(specs):
            seed = SEED + parent_index * 100_000 + spec_index
            result = run_spec(env, step, row, spec, seed, target, center, scale, args.horizon)
            snapshot = result.pop("snapshot")
            record = {"trajectory_parent_id": row["trajectory_parent_id"], "source_ascent_entry_id": row["id"],
                      "spec_index": spec_index, "spec": spec, "seed": seed, **result}
            parent_rows.append(record); outcomes.append(record)
            if snapshot is not None:
                snapshots[spec_index] = snapshot
        valid = [item for item in parent_rows if item["success"]]
        if not valid:
            continue
        best = min(valid, key=lambda item: (item["pose_margin"] < 0, item["terminal_distance"], -item["pose_margin"], item["spec_index"]))
        replay_hashes = []
        for _ in range(2):
            replay = run_spec(env, step, row, best["spec"], best["seed"], target, center, scale, args.horizon)
            if not replay["success"]:
                raise RuntimeError("selected Apex proposal failed exact replay")
            replay_hashes.append(canonical_state_byte_hash(replay["snapshot"]))
        if len(set(replay_hashes)) != 1:
            raise RuntimeError("selected Apex proposal is not exact-replay deterministic")
        snapshot = copy.deepcopy(snapshots[best["spec_index"]])
        identifier = hashlib.sha256(f"terminal-apex:{row['trajectory_parent_id']}:{best['spec_index']}:{SEED}".encode()).hexdigest()[:32]
        snapshot.update({"id": identifier, "candidate_kind": "terminal_aligned_valid_apex_proposal",
            "trajectory_parent_id": row["trajectory_parent_id"], "display_parent": row["trajectory_parent_id"],
            "source_ascent_entry_id": row["id"], "relative_to_apex": 0,
            "generation_spec": best["spec"], "generation_seed": best["seed"],
            "terminal_distance": best["terminal_distance"], "pose_margin": best["pose_margin"],
            "exact_replay_hash": replay_hashes[0], "artifact_role": "proposal_support_bank",
            "safe_claim_allowed": False})
        best_records.append(snapshot)
    bank_path = root / "terminal_aligned_apex_proposals.pkl"
    SnapshotBank(best_records, {"artifact_role": "proposal_support_bank", "safe_claim_allowed": False,
        "inputs": inputs, "terminal_target_features": list(FEATURES)}).save(bank_path)
    distances = [row["terminal_distance"] for row in best_records]
    status = "PASS" if len(best_records) >= 2 and min(distances) < 23.214247689634647 else "FAIL"
    report = {"status": status, "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "parents": len(selected), "specs_per_parent": len(specs),
        "selection": args.selection, "spec_profile": args.spec_profile,
        "selection_distances": selection_distances,
        "valid_apex_outcomes": sum(x["success"] for x in outcomes),
        "successful_parents": len(best_records), "failure_reasons": dict(Counter(x["failure_reason"] for x in outcomes if not x["success"])),
        "best_terminal_distances": distances, "best_pose_margins": [row["pose_margin"] for row in best_records],
        "previous_best_terminal_distance": 23.214247689634647, "proposal_bank": str(bank_path),
        "proposal_bank_sha256": file_sha256(bank_path), "formal_tube_or_matcher": False, "PPO_authorization": False,
        "next": "local_authority_then_bounded_C_D_bridge" if status == "PASS" else "terminal_aligned_parent_pilot_no_progress",
        "outcomes": outcomes}
    save_json(root / "APEX_TERMINAL_ALIGNED_PARENT_ACQUISITION_V1_REPORT.json", report)
    save_json(root / "completed.json", {"status": status, "next": report["next"]})
    print(json.dumps({key: report[key] for key in ("status", "parents", "valid_apex_outcomes", "successful_parents",
        "best_terminal_distances", "best_pose_margins", "previous_best_terminal_distance", "next")}, indent=2))


if __name__ == "__main__":
    main()
