"""Structured CPU MuJoCo versus MJX Takeoff-tail event audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
import numpy as np

from dvgc.bank import SnapshotBank
from dvgc.centroidal import replay_centroidal
from dvgc.config import file_sha256, load_config
from dvgc.env import OrangeBikeDVGC
from dvgc.rollout import restore_snapshot
from dvgc.runtime import save_json


def _event(trace):
    contacts = [row["tick"] for row in trace if row["contact_count"] > 0]
    last = max(contacts, default=None)
    separation = next((
        row["tick"] for row in trace
        if last is not None and row["tick"] > last
        and row["contact_count"] == 0
    ), None)
    return last, separation


def _spread(traces, key):
    base = np.asarray(traces[0][-1][key], float)
    return float(max(
        np.max(np.abs(np.asarray(trace[-1][key], float) - base))
        for trace in traces[1:]
    )) if len(traces) > 1 else 0.


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--takeoff-bank", required=True)
    p.add_argument("--lineage-report", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--parent", action="append", required=True)
    p.add_argument("--mjx-repeats", type=int, default=3)
    p.add_argument("--config", default="configs/default.json")
    a = p.parse_args()
    bank = SnapshotBank.load(a.takeoff_bank)
    by_id = {row["id"]: row for row in bank.records}
    lineage = json.loads(Path(a.lineage_report).read_text())
    cfg = load_config(a.config, {
        "training_stage": "flight", "use_bank_resets": False,
        "domain_randomization": False, "obs_noise_enable": False,
        "stage_reachability_objective": "",
    })
    env = OrangeBikeDVGC(cfg, snapshot_bank=SnapshotBank())
    step = jax.jit(env.step)
    model = mujoco.MjModel.from_xml_path(str(cfg.xml_path))
    knee_q = int(model.jnt_qposadr[model.joint("knee_joint").id])

    def measure(qpos, qvel, ctrl, tick):
        momentum = replay_centroidal(model, qpos, qvel, ctrl)
        return {
            "tick": tick,
            "contact_count": len(momentum["robot_terrain_contacts"]),
            "qpos": np.asarray(qpos).tolist(),
            "qvel": np.asarray(qvel).tolist(),
            "h": momentum["centroidal_angular_momentum"],
        }

    def cpu_trace(source, actions):
        data = mujoco.MjData(model)
        data.qpos[:] = source["qpos"]
        data.qvel[:] = source["qvel"]
        data.ctrl[:] = source["ctrl"]
        data.qacc_warmstart[:] = source["qacc_warmstart"]
        mujoco.mj_forward(model, data)
        rows = []
        for tick, action in enumerate(actions, 1):
            data.ctrl[:] = np.asarray(env._action_to_ctrl(
                jp.asarray(action), jp.asarray(data.qpos[knee_q])
            ), float)
            for _ in range(env.n_substeps):
                mujoco.mj_step(model, data)
            rows.append(measure(
                data.qpos, data.qvel, data.ctrl, tick
            ))
        return rows

    def mjx_trace(source, actions, seed):
        state = restore_snapshot(env, source, jax.random.PRNGKey(seed))
        rows = []
        for tick, action in enumerate(actions, 1):
            state = step(state, np.asarray(action, np.float32))
            rows.append(measure(
                np.asarray(state.data.qpos), np.asarray(state.data.qvel),
                np.asarray(state.data.ctrl), tick,
            ))
        return rows

    parents = {}
    gate_pass = True
    for parent in a.parent:
        info = lineage["parents"][parent]
        source = by_id[info["source_takeoff_state_id"]]
        actions = [
            row["action"] for row in info["trace"] if row["tick"] > 0
        ]
        cpu = [cpu_trace(source, actions) for _ in range(2)]
        entry_bank = SnapshotBank.load(
            Path(info["source_acquisition"]) / "fresh_ascent_entries.pkl"
        )
        entry = next(row for row in entry_bank.records
                     if row["id"] == info["source_entry_id"])
        seed = int(entry["dynamics_seed"])
        mjx = [mjx_trace(source, actions, seed)
               for _ in range(a.mjx_repeats)]
        cpu_last, cpu_sep = _event(cpu[0])
        mjx_events = [_event(trace) for trace in mjx]
        cpu_sep_row = next((
            row for row in cpu[0] if row["tick"] == cpu_sep
        ), cpu[0][-1])
        mjx_sep_rows = [
            next((row for row in trace if row["tick"] == event[1]),
                 trace[-1])
            for trace, event in zip(mjx, mjx_events)
        ]
        event_delta = (
            None if cpu_sep is None or mjx_events[0][1] is None
            else abs(cpu_sep - mjx_events[0][1])
        )
        h_delta = float(np.max(np.abs(
            np.asarray(cpu_sep_row["h"])
            - np.asarray(mjx_sep_rows[0]["h"])
        )))
        comparable = (
            event_delta is not None and event_delta <= 1 and h_delta <= .03
        )
        gate_pass &= comparable
        parents[parent] = {
            "source_takeoff_state_id": source["id"],
            "actions_replayed": len(actions),
            "cpu": {
                "last_contact_tick": cpu_last,
                "separation_tick": cpu_sep,
                "separation_h": cpu_sep_row["h"],
                "repeat_qpos_linf": _spread(cpu, "qpos"),
                "repeat_qvel_linf": _spread(cpu, "qvel"),
                "repeat_h_linf": _spread(cpu, "h"),
            },
            "mjx": {
                "events": [{
                    "last_contact_tick": event[0],
                    "separation_tick": event[1],
                } for event in mjx_events],
                "separation_h": [row["h"] for row in mjx_sep_rows],
                "repeat_qpos_linf": _spread(mjx, "qpos"),
                "repeat_qvel_linf": _spread(mjx, "qvel"),
                "repeat_h_linf": _spread(mjx, "h"),
            },
            "cross_engine": {
                "separation_tick_delta": event_delta,
                "separation_h_linf": h_delta,
                "comparable": comparable,
            },
            "historical_composite": {
                "last_contact_tick": info["last_support_tick"],
                "separation_tick": info["separation_tick"],
                "separation_h": info["separation_h"],
            },
        }
    payload = {
        "status": "PASS" if gate_pass else "FAIL",
        "artifact_role": "takeoff_tail_cross_engine_runtime_gate",
        "blocker": None if gate_pass else "takeoff_tail_cross_engine_mismatch",
        "authority_or_discovery_allowed": bool(gate_pass),
        "ppo_authorization": False,
        "comparison_rule": {
            "separation_tick_tolerance": 1,
            "centroidal_h_linf_tolerance": .03,
            "both_required": True,
        },
        "parents": parents,
        "xml_sha256": file_sha256(cfg.xml_path),
        "config_sha256": file_sha256(a.config),
        "takeoff_bank_sha256": file_sha256(a.takeoff_bank),
        "lineage_report_sha256": file_sha256(a.lineage_report),
    }
    save_json(a.output, payload)
    print(json.dumps({
        "status": payload["status"], "blocker": payload["blocker"],
        "parents": {
            parent: row["cross_engine"] for parent, row in parents.items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
