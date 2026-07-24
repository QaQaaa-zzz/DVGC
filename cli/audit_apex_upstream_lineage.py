"""Replay real Takeoff-to-Apex lineages and capture contact-aligned states."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
import numpy as np

from cli.acquire_ascent_apex_parents import _local_action
from cli.search_takeoff_actions import SEQUENCES, action_at
from dvgc.bank import SnapshotBank
from dvgc.centroidal import replay_centroidal
from dvgc.config import file_sha256, load_config
from dvgc.env import OrangeBikeDVGC
from dvgc.rollout import restore_snapshot
from dvgc.runtime import build_inference, load_params, save_json


def _diagnostic(model, state, tick, section, action):
    result = replay_centroidal(
        model, np.asarray(state.data.qpos), np.asarray(state.data.qvel),
        np.asarray(state.data.ctrl),
    )
    return {
        "tick": tick, "section": section, "action": np.asarray(action).tolist(),
        "system_com": result["system_com"],
        "com_velocity": result["com_velocity"],
        "centroidal_angular_momentum":
            result["centroidal_angular_momentum"],
        "robot_terrain_contact_count": len(result["robot_terrain_contacts"]),
        "contacts": result["robot_terrain_contacts"],
        "net_terrain_impulse": (
            np.asarray(result["net_terrain_force"]) * .02
        ).tolist(),
        "net_terrain_angular_impulse": (
            np.asarray(result["net_terrain_torque_about_com"]) * .02
        ).tolist(),
        "crosscheck_linf": result["angular_momentum_crosscheck_linf"],
    }


def _controller(name, policies):
    if name.startswith("bounded:"):
        sequence = SEQUENCES[name.split(":", 1)[1]]
        return lambda state, key, tick: action_at(sequence, tick)
    infer = policies[name]
    return lambda state, key, tick: infer(state.obs, key)[0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", required=True)
    p.add_argument("--takeoff-bank", required=True)
    p.add_argument("--output-bank", required=True)
    p.add_argument("--output-report", required=True)
    p.add_argument("--config", default="configs/default.json")
    p.add_argument("--parent", action="append", required=True)
    a = p.parse_args()
    root = Path(a.run_root)
    source = SnapshotBank.load(a.takeoff_bank)
    source_by_id = {row["id"]: row for row in source.records}
    cfg = load_config(a.config, {
        "training_stage": "flight", "use_bank_resets": False,
        "domain_randomization": False, "obs_noise_enable": False,
        "stage_reachability_objective": "",
    })
    env = OrangeBikeDVGC(cfg, snapshot_bank=SnapshotBank())
    step = jax.jit(env.step)
    model = mujoco.MjModel.from_xml_path(str(cfg.xml_path))
    versions = (
        root / "ascent/independent_parent_acquisition_v1",
        root / "ascent/independent_parent_acquisition_v2_24",
    )
    reports = [json.loads((path / "report.json").read_text())
               for path in versions]
    entries = [SnapshotBank.load(path / "fresh_ascent_entries.pkl")
               for path in versions]
    policy_paths = {}
    for report in reports:
        for row in report["inputs"]["policies"]:
            policy_paths[row["id"]] = Path(row["path"])
    policies = {
        name: build_inference(
            env, load_params(path / "params.pkl"), deterministic=True
        )
        for name, path in policy_paths.items()
    }
    selected_records = []
    parent_results = {}
    for requested in a.parent:
        located = None
        for version, (path, report, bank) in enumerate(
                zip(versions, reports, entries)):
            outcome = next((
                row for row in report["search_outcomes"]
                if row.get("success")
                and row["trajectory_parent_id"] == requested
            ), None)
            if outcome is not None:
                entry = next(row for row in bank.records
                             if row["id"] == outcome["source_ascent_entry_id"])
                located = path, outcome, entry, version
                break
        if located is None:
            parent_results[requested] = {
                "status": "upstream_lineage_unavailable",
                "reason": "reference-only parent has no recorded Takeoff source",
            }
            continue
        path, outcome, entry, version = located
        source_row = source_by_id[entry["takeoff_source_state_id"]]
        action_fn = _controller(entry["takeoff_controller"], policies)
        seed = int(entry["dynamics_seed"])
        state = restore_snapshot(env, source_row, jax.random.PRNGKey(seed))
        trace = [_diagnostic(
            model, state, 0, "takeoff", jp.zeros((4,), jp.float32)
        )]
        states = [(0, "takeoff", state)]
        for tick in range(int(entry["flight_confirmation_tick"])):
            action = action_fn(
                state, jax.random.PRNGKey(seed + tick), tick
            )
            state = step(state, action)
            trace.append(_diagnostic(model, state, tick + 1, "takeoff", action))
            states.append((tick + 1, "takeoff", state))
        qpos_error = float(np.max(np.abs(
            np.asarray(state.data.qpos) - np.asarray(entry["qpos"])
        )))
        qvel_error = float(np.max(np.abs(
            np.asarray(state.data.qvel) - np.asarray(entry["qvel"])
        )))
        exact = qpos_error <= 1e-5 and qvel_error <= 1e-5
        # Continue from the immutable saved entry, not a mismatched replay.
        state = restore_snapshot(env, entry, jax.random.PRNGKey(seed))
        offset = int(entry["flight_confirmation_tick"])
        for tick in range(int(outcome["entry_tick"])):
            action = _local_action(outcome["parameters"], tick)
            state = step(state, action)
            trace.append(_diagnostic(
                model, state, offset + tick + 1, "ascent", action
            ))
            states.append((offset + tick + 1, "ascent", state))
        contact_indices = [
            i for i, row in enumerate(trace)
            if row["robot_terrain_contact_count"] > 0
        ]
        last_contact_i = max(contact_indices) if contact_indices else None
        separation_i = next((
            i for i in range((last_contact_i or -1) + 1, len(trace))
            if trace[i]["robot_terrain_contact_count"] == 0
        ), None) if last_contact_i is not None else None
        apex_i = len(trace) - 1
        wanted = {
            "last_support": last_contact_i, "separation": separation_i,
            "apex_minus_8": max(0, apex_i - 8),
            "apex_minus_6": max(0, apex_i - 6),
            "apex_minus_4": max(0, apex_i - 4), "apex_event": apex_i,
        }
        by_tick = {tick: state for tick, _, state in states}
        captured = {}
        for label, index in wanted.items():
            if index is None:
                continue
            diagnostic = trace[index]
            snapshot_state = by_tick[diagnostic["tick"]]
            record = env.snapshot_record(snapshot_state, "flight")
            record.update({
                "id": hashlib.sha256(
                    f"upstream-lineage:{requested}:{label}".encode()
                ).hexdigest()[:32],
                "candidate_kind": "event_aligned_upstream_entry_proposal",
                "trajectory_parent_id": requested,
                "event_label": label,
                "lineage_tick": diagnostic["tick"],
                "source_takeoff_state_id": source_row["id"],
                "takeoff_controller": entry["takeoff_controller"],
                "source_acquisition": str(path),
                "training_only": True,
                "bootstrap_eligible": False,
            })
            selected_records.append(record)
            captured[label] = {
                "snapshot_id": record["id"], **diagnostic,
            }
        airborne_hx = [
            row["centroidal_angular_momentum"][0]
            for row in trace[(separation_i or len(trace)):]
            if row["robot_terrain_contact_count"] == 0
        ]
        parent_results[requested] = {
            "status": "PASS" if exact else "replay_mismatch",
            "source_acquisition": str(path),
            "source_entry_id": entry["id"],
            "source_takeoff_state_id": source_row["id"],
            "takeoff_controller": entry["takeoff_controller"],
            "takeoff_replay_qpos_linf": qpos_error,
            "takeoff_replay_qvel_linf": qvel_error,
            "last_support_tick": (
                trace[last_contact_i]["tick"] if last_contact_i is not None
                else None
            ),
            "separation_tick": (
                trace[separation_i]["tick"] if separation_i is not None else None
            ),
            "apex_tick": trace[apex_i]["tick"],
            "separation_h": (
                trace[separation_i]["centroidal_angular_momentum"]
                if separation_i is not None else None
            ),
            "airborne_hx_relative_span": (
                float(np.ptp(airborne_hx) / max(abs(np.mean(airborne_hx)), .03))
                if len(airborne_hx) >= 2 else None
            ),
            "captured_events": captured,
            "trace": trace,
        }
    SnapshotBank(selected_records, {
        "artifact_role": "event_aligned_takeoff_tail_apex_lineage_proposals",
        "certified_tube": False, "safe_claim_allowed": False,
        "bootstrap_eligible": False,
        "source_takeoff_bank_sha256": file_sha256(a.takeoff_bank),
        "xml_sha256": file_sha256(cfg.xml_path),
    }).save(a.output_bank)
    payload = {
        "status": (
            "PASS" if all(row["status"] in
                          ("PASS", "upstream_lineage_unavailable")
                          for row in parent_results.values()) else "FAIL"
        ),
        "artifact_role": "exact_upstream_contact_lineage_audit",
        "diagnostic_only": True, "apex_ppo_authorized": False,
        "xml_sha256": file_sha256(cfg.xml_path),
        "takeoff_bank_sha256": file_sha256(a.takeoff_bank),
        "parents": parent_results,
        "captured_states": len(selected_records),
        "output_bank": str(Path(a.output_bank).resolve()),
        "output_bank_sha256": file_sha256(a.output_bank),
    }
    save_json(a.output_report, payload)
    print(json.dumps({
        "status": payload["status"], "captured_states": len(selected_records),
        "parents": {key: value["status"]
                    for key, value in parent_results.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
