"""Capture exact pre-entry Takeoff tails for bounded handoff correction."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import jax
import numpy as np

from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256, load_config
from dvgc.env import OrangeBikeDVGC
from dvgc.rollout import restore_snapshot
from dvgc.runtime import build_inference, load_params, save_json
from dvgc.trajectory_mining import canonical_state_byte_hash


def selected_entries(records: list[dict], parent_ids: list[str]) -> list[dict]:
    by_parent = {str(row["trajectory_parent_id"]): row for row in records}
    missing = sorted(set(parent_ids) - set(by_parent))
    if missing:
        raise ValueError(f"selected parents missing from entry bank: {missing}")
    if len(set(parent_ids)) != len(parent_ids):
        raise ValueError("selected parents are not unique")
    return [by_parent[parent] for parent in parent_ids]


def state_error(state, row: dict) -> float:
    qpos = np.asarray(jax.device_get(state.data.qpos), float)
    qvel = np.asarray(jax.device_get(state.data.qvel), float)
    return float(max(np.max(np.abs(qpos - np.asarray(row["qpos"], float))),
                     np.max(np.abs(qvel - np.asarray(row["qvel"], float)))))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--entries", required=True)
    p.add_argument("--selection-manifest", required=True)
    p.add_argument("--takeoff-bank", required=True)
    p.add_argument("--policy", action="append", default=[], help="name=policy_dir")
    p.add_argument("--output-root", required=True)
    p.add_argument("--tail-ticks", type=int, default=4)
    p.add_argument("--config", default="configs/default.json")
    args = p.parse_args()
    root = Path(args.output_root)
    if root.exists():
        raise SystemExit(f"refusing overwrite {root}")
    entries_path, source_path = Path(args.entries), Path(args.takeoff_bank)
    entries = SnapshotBank.load(entries_path)
    source = SnapshotBank.load(source_path)
    manifest = json.loads(Path(args.selection_manifest).read_text())
    selected = selected_entries(entries.records, list(map(str, manifest["parents"])))
    source_by_id = {str(row["id"]): row for row in source.records}
    cfg = load_config(args.config, {
        "training_stage": "takeoff", "use_bank_resets": False,
        "domain_randomization": False, "obs_noise_enable": False,
        "stage_reachability_objective": "",
    })
    env = OrangeBikeDVGC(cfg, snapshot_bank=SnapshotBank())
    step = jax.jit(env.step)
    policies = {}
    policy_inputs = []
    for item in args.policy:
        name, raw = item.split("=", 1)
        path = Path(raw)
        policies[name] = build_inference(
            env, load_params(path / "params.pkl"), deterministic=True,
        )
        policy_inputs.append({"name": name, "path": str(path.resolve()),
                              "sha256": file_sha256(path / "params.pkl")})
    root.mkdir(parents=True)
    save_json(root / "cost_estimate.json", {
        "estimated_seconds": 600, "parents": len(selected),
        "maximum_seed_offsets_per_parent": 3, "tail_ticks": args.tail_ticks,
        "PPO_steps": 0,
    })
    records, rows = [], []
    for entry in selected:
        controller = str(entry["takeoff_controller"])
        if controller not in policies:
            raise RuntimeError(f"missing frozen controller {controller}")
        source_row = source_by_id[str(entry["takeoff_source_state_id"])]
        base_seed = int(entry["dynamics_seed"])
        reproductions = []
        best = None
        for seed_offset in range(3):
            state = restore_snapshot(env, source_row, jax.random.PRNGKey(base_seed + seed_offset))
            recent = []
            for tick in range(int(entry["flight_confirmation_tick"])):
                action = policies[controller](
                    state.obs, jax.random.PRNGKey(base_seed + seed_offset * 100 + tick),
                )[0]
                recent.append((tick, copy.deepcopy(env.snapshot_record(state, "takeoff")),
                               np.asarray(jax.device_get(action), float)))
                state = step(state, action)
            error = state_error(state, entry)
            reproductions.append({"seed_offset": seed_offset, "max_qpos_qvel_error": error})
            if best is None or error < best[0]:
                best = (error, seed_offset, state, recent)
        assert best is not None
        error, seed_offset, final_state, recent = best
        if error != 0.0:
            raise RuntimeError(
                f"parent {entry['trajectory_parent_id']} is not exact replay: {error}"
            )
        tail = recent[-args.tail_ticks:]
        action_sequence = [action for _, _, action in tail]
        replay_hashes = []
        for replay_index in range(2):
            replay = restore_snapshot(
                env, tail[0][1], jax.random.PRNGKey(base_seed + 50_000 + replay_index),
            )
            for action in action_sequence:
                replay = step(replay, action)
            if state_error(replay, entry) != 0.0:
                raise RuntimeError("tail action replay did not reproduce entry")
            replay_hashes.append(canonical_state_byte_hash(env.snapshot_record(replay, "flight")))
        if len(set(replay_hashes)) != 1:
            raise RuntimeError("tail replay is nondeterministic")
        for relative, (tick, snapshot, action) in enumerate(tail, start=-len(tail)):
            snapshot.update({
                "id": f"takeoff-tail:{entry['trajectory_parent_id']}:{relative}",
                "candidate_kind": "exact_takeoff_pre_ascent_tail",
                "trajectory_parent_id": entry["trajectory_parent_id"],
                "source_ascent_entry_id": entry["id"],
                "takeoff_source_state_id": entry["takeoff_source_state_id"],
                "takeoff_controller": controller,
                "controller_policy_hash": entry.get("takeoff_controller_policy_hash"),
                "seed_offset": seed_offset, "command_tick": tick,
                "relative_to_flight_confirmation": relative,
                "nominal_action": action.tolist(),
                "target_entry_state_hash": replay_hashes[0],
                "artifact_role": "proposal_support_bank", "safe_claim_allowed": False,
            })
            records.append(snapshot)
        rows.append({
            "trajectory_parent_id": entry["trajectory_parent_id"],
            "controller": controller, "seed_offset": seed_offset,
            "reproduction_trials": reproductions, "tail_exact_replay": True,
            "target_entry_state_hash": replay_hashes[0],
        })
    bank_path = root / "takeoff_tail_snapshots.pkl"
    SnapshotBank(records, {
        "artifact_role": "takeoff_pre_ascent_tail_proposal_support",
        "safe_claim_allowed": False, "entries_sha256": file_sha256(entries_path),
        "takeoff_bank_sha256": file_sha256(source_path), "policies": policy_inputs,
        "tail_ticks": args.tail_ticks,
    }).save(bank_path)
    report = {"status": "PASS", "parents": len(selected), "snapshots": len(records),
              "all_exact_replay": all(row["tail_exact_replay"] for row in rows),
              "bank": str(bank_path.resolve()), "bank_sha256": file_sha256(bank_path),
              "rows": rows, "PPO_authorization": False}
    save_json(root / "TAKEOFF_TAIL_CAPTURE_V1_REPORT.json", report)
    save_json(root / "completed.json", {"status": "PASS", "bank_sha256": report["bank_sha256"]})
    print(json.dumps({key: report[key] for key in ("status", "parents", "snapshots",
                                                    "all_exact_replay", "bank_sha256")}, indent=2))


if __name__ == "__main__":
    main()
