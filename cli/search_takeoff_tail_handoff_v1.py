"""Bounded two-tick Takeoff-tail search for pose-compatible Ascent entries."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import jax
import numpy as np

from cli.stage_label_pilot import sample_from_state
from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256, load_config
from dvgc.env import END_REASON, OrangeBikeDVGC
from dvgc.rollout import restore_snapshot
from dvgc.runtime import save_json
from dvgc.stage_reachability import evaluate_entry
from dvgc.trajectory_mining import canonical_state_byte_hash


FLOOR = np.asarray([
    .05, .05, .05, .05, .05, .05, .2, .2, .2,
    .2, .2, .2, .1, .1, .2, .2,
])


def residual_design(count: int, seed: int, bound: float) -> np.ndarray:
    """Deterministic stratified local design, with the nominal first."""
    if count < 2:
        raise ValueError("count must include nominal plus at least one residual")
    rng = np.random.default_rng(seed)
    rows = np.empty((count - 1, 4), float)
    for dim in range(4):
        strata = (np.arange(count - 1) + rng.random(count - 1)) / (count - 1)
        rows[:, dim] = (2. * strata[rng.permutation(count - 1)] - 1.) * bound
    return np.vstack([np.zeros((1, 4)), rows])


def support_distance(feature: np.ndarray, support: np.ndarray, scale: np.ndarray) -> float:
    return float(np.min(np.linalg.norm(
        (support - feature[None, :]) / scale[None, :], axis=1,
    )))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tail-bank", required=True)
    p.add_argument("--entry-bank", required=True)
    p.add_argument("--support-report", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--proposals", type=int, default=32)
    p.add_argument("--residual-bound", type=float, default=.30)
    p.add_argument("--keep-per-parent", type=int, default=3)
    p.add_argument("--seed", type=int, default=3_870_000_000)
    p.add_argument("--config", default="configs/default.json")
    args = p.parse_args()
    root = Path(args.output_root)
    if root.exists():
        raise SystemExit(f"refusing overwrite {root}")
    tail_path, entry_path = Path(args.tail_bank), Path(args.entry_bank)
    tails, entries = SnapshotBank.load(tail_path), SnapshotBank.load(entry_path)
    report = json.loads(Path(args.support_report).read_text())
    supported_ids = set(map(str, report["successful_parent_ids"]))
    supported = np.asarray([
        row["physical_feature"] for row in entries.records
        if str(row["trajectory_parent_id"]) in supported_ids
    ], float)
    if not len(supported):
        raise SystemExit("support parents are absent from entry bank")
    all_features = np.asarray([row["physical_feature"] for row in entries.records], float)
    scale = np.maximum(np.std(all_features, axis=0), FLOOR)
    cfg = load_config(args.config, {
        "training_stage": "takeoff", "use_bank_resets": False,
        "domain_randomization": False, "obs_noise_enable": False,
        "stage_reachability_objective": "",
    })
    env = OrangeBikeDVGC(cfg, snapshot_bank=SnapshotBank())
    step = jax.jit(env.step)
    grouped = defaultdict(list)
    for row in tails.records:
        grouped[str(row["trajectory_parent_id"])].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["relative_to_flight_confirmation"]))
    root.mkdir(parents=True)
    save_json(root / "cost_estimate.json", {
        "estimated_seconds": 900, "parents": len(grouped),
        "proposals_per_parent": args.proposals, "PPO_steps": 0,
    })
    design = residual_design(args.proposals, args.seed, args.residual_bound)
    outcomes, kept = [], []
    for parent_index, (parent, rows) in enumerate(sorted(grouped.items())):
        if len(rows) != 2:
            raise RuntimeError(f"parent {parent} does not have exactly two tail ticks")
        parent_outcomes = []
        for proposal_index, residual in enumerate(design):
            state = restore_snapshot(
                env, rows[0], jax.random.PRNGKey(args.seed + parent_index * 10000 + proposal_index),
            )
            takeoff_entry_ever = False
            applied = []
            previous_vz = float(np.asarray(rows[0]["qvel"], float)[2])
            sample = None
            for tick, row in enumerate(rows):
                nominal = np.asarray(row["nominal_action"], float)
                action = nominal.copy()
                action[2:] = np.clip(action[2:] + residual[2*tick:2*tick+2], -1., 1.)
                applied.append(action.tolist())
                state = step(state, action)
                sample = sample_from_state(env, state, previous_vz)
                takeoff_entry_ever |= bool(evaluate_entry("takeoff", sample, cfg)["valid"])
                previous_vz = float(sample["physical_feature"][8])
            assert sample is not None
            feature = np.asarray(sample["physical_feature"], float)
            phase = int(np.asarray(state.info["phase"]))
            done = float(np.asarray(state.done)) > .5
            legal = bool(takeoff_entry_ever and phase == 2 and not done)
            distance = support_distance(feature, supported, scale) if legal else float("inf")
            angular_speed = float(np.linalg.norm(feature[9:12]))
            item = {
                "trajectory_parent_id": parent, "proposal_index": proposal_index,
                "residual": residual.tolist(), "applied_actions": applied,
                "legal_ascent_entry": legal, "support_distance": distance,
                "angular_speed": angular_speed, "pitch": float(feature[4]),
                "roll": float(feature[3]),
                "termination_reason": (None if not done else END_REASON.get(
                    int(np.asarray(state.info["end_code"])), "unknown")),
                "physical_feature": feature.tolist(), "state": state,
            }
            parent_outcomes.append(item)
        legal = [item for item in parent_outcomes if item["legal_ascent_entry"]]
        legal.sort(key=lambda item: (item["support_distance"], item["angular_speed"],
                                     item["proposal_index"]))
        for rank, item in enumerate(legal[:args.keep_per_parent]):
            state = item.pop("state")
            snapshot = env.snapshot_record(state, "flight")
            replay_hashes = []
            for replay_index in range(2):
                replay = restore_snapshot(
                    env, rows[0], jax.random.PRNGKey(args.seed + 50_000 + replay_index),
                )
                for action in item["applied_actions"]:
                    replay = step(replay, np.asarray(action, np.float32))
                replay_hashes.append(canonical_state_byte_hash(env.snapshot_record(replay, "flight")))
            if len(set(replay_hashes)) != 1:
                raise RuntimeError("selected tail correction is not exact replay")
            snapshot.update({
                "id": hashlib.sha256(
                    f"takeoff-tail-corrected:{parent}:{item['proposal_index']}".encode()
                ).hexdigest()[:32],
                "candidate_kind": "takeoff_tail_corrected_ascent_entry",
                "trajectory_parent_id": parent, "source_tail_id": rows[0]["id"],
                "source_ascent_entry_id": rows[0]["source_ascent_entry_id"],
                "tail_residual": item["residual"], "tail_applied_actions": item["applied_actions"],
                "support_distance": item["support_distance"], "angular_speed": item["angular_speed"],
                "selection_rank": rank, "exact_replay_hash": replay_hashes[0],
                "artifact_role": "proposal_support_bank", "safe_claim_allowed": False,
            })
            kept.append(snapshot)
            item["selected_rank"] = rank
        for item in parent_outcomes:
            item.pop("state", None)
            outcomes.append(item)
    bank_path = root / "corrected_ascent_entries.pkl"
    SnapshotBank(kept, {
        "artifact_role": "takeoff_tail_corrected_ascent_proposal_support",
        "safe_claim_allowed": False, "tail_bank_sha256": file_sha256(tail_path),
        "entry_bank_sha256": file_sha256(entry_path),
        "support_parent_ids": sorted(supported_ids), "residual_bound": args.residual_bound,
    }).save(bank_path)
    nominal = [row for row in outcomes if row["proposal_index"] == 0]
    report_out = {
        "status": "PASS" if kept else "FAIL", "parents": len(grouped),
        "proposals": len(outcomes), "legal_entries": sum(row["legal_ascent_entry"] for row in outcomes),
        "selected_entries": len(kept),
        "nominal_distances": {row["trajectory_parent_id"]: row["support_distance"] for row in nominal},
        "best_distances": {parent: min((row["support_distance"] for row in outcomes
                                        if row["trajectory_parent_id"] == parent), default=float("inf"))
                           for parent in grouped},
        "termination_reasons": dict(Counter(row["termination_reason"] for row in outcomes
                                             if row["termination_reason"])),
        "bank": str(bank_path.resolve()), "bank_sha256": file_sha256(bank_path),
        "PPO_authorization": False, "outcomes": outcomes,
    }
    save_json(root / "TAKEOFF_TAIL_HANDOFF_SEARCH_V1_REPORT.json", report_out)
    save_json(root / "completed.json", {"status": report_out["status"],
                                         "bank_sha256": report_out["bank_sha256"]})
    print(json.dumps({key: report_out[key] for key in ("status", "parents", "proposals",
        "legal_entries", "selected_entries", "nominal_distances", "best_distances")}, indent=2))


if __name__ == "__main__":
    main()
