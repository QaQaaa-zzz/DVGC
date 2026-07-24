"""Finite-difference authority of real Takeoff contact-tail windows."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import jax
import mujoco
import numpy as np

from cli.stage_label_pilot import sample_from_state
from dvgc.bank import SnapshotBank
from dvgc.centroidal import replay_centroidal
from dvgc.config import file_sha256, load_config
from dvgc.env import OrangeBikeDVGC
from dvgc.rollout import restore_snapshot
from dvgc.runtime import save_json


OUTPUTS = (
    "hx", "hy", "hz", "roll", "pitch", "wx", "wy", "vx", "vz",
    "hip", "knee", "joint_margin",
)
OUTPUT_FLOORS = np.asarray([
    .03, .03, .02, np.deg2rad(2), np.deg2rad(2), .2, .2, .1, .1,
    .04, .04, .015,
])


def _joint_addresses(model):
    result = {}
    for name in ("hip_joint", "knee_joint"):
        joint = model.joint(name)
        result[name] = (
            int(model.jnt_qposadr[joint.id]),
            int(model.jnt_dofadr[joint.id]),
            np.asarray(model.jnt_range[joint.id], float),
        )
    return result


def _measure(model, state, env, previous_vz, addresses):
    sample = sample_from_state(env, state, previous_vz)
    feature = np.asarray(sample["physical_feature"], float)
    qpos = np.asarray(state.data.qpos, float)
    momentum = replay_centroidal(
        model, qpos, np.asarray(state.data.qvel),
        np.asarray(state.data.ctrl),
    )
    margins = [
        min(qpos[address] - limits[0], limits[1] - qpos[address])
        for address, _, limits in addresses.values()
    ]
    vector = np.asarray([
        *momentum["centroidal_angular_momentum"],
        feature[3], feature[4], feature[9], feature[10], feature[6],
        feature[8], feature[13], feature[14], min(margins),
    ], float)
    return vector, {
        "physical_feature": feature.tolist(),
        "system_com": momentum["system_com"],
        "centroidal_angular_momentum":
            momentum["centroidal_angular_momentum"],
        "robot_terrain_contact_count": len(
            momentum["robot_terrain_contacts"]
        ),
        "net_terrain_force": momentum["net_terrain_force"],
        "net_terrain_torque_about_com":
            momentum["net_terrain_torque_about_com"],
        "crosscheck_linf": momentum["angular_momentum_crosscheck_linf"],
        "joint_margin": float(min(margins)),
    }


def _window_start(separation_tick, requested):
    available = int(separation_tick)
    if requested == "full":
        return 0, available, True
    width = int(requested)
    return max(0, available - width), min(width, available), available >= width


def _rank(matrix, scale):
    if matrix.size == 0:
        return [], 0
    singular = np.linalg.svd(matrix / scale[:, None], compute_uv=False)
    threshold = max(.1, .1 * singular[0]) if len(singular) else np.inf
    return singular.tolist(), int(np.sum(singular > threshold))


def _ranges(values):
    values = np.asarray(values, float)
    return {
        "min": float(np.min(values)), "p05": float(np.percentile(values, 5)),
        "p50": float(np.median(values)), "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--takeoff-bank", required=True)
    p.add_argument("--lineage-report", required=True)
    p.add_argument("--terminal-bank", required=True)
    p.add_argument("--terminal-report", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--parent", action="append", required=True)
    p.add_argument("--amplitude", type=float, default=.12)
    p.add_argument("--config", default="configs/default.json")
    a = p.parse_args()
    source = SnapshotBank.load(a.takeoff_bank)
    source_by_id = {row["id"]: row for row in source.records}
    lineage = json.loads(Path(a.lineage_report).read_text())
    terminal = SnapshotBank.load(a.terminal_bank)
    terminal_report = json.loads(Path(a.terminal_report).read_text())
    cfg = load_config(a.config, {
        "training_stage": "flight", "use_bank_resets": False,
        "domain_randomization": False, "obs_noise_enable": False,
        "stage_reachability_objective": "",
    })
    env = OrangeBikeDVGC(cfg, snapshot_bank=SnapshotBank())
    step = jax.jit(env.step)
    model = mujoco.MjModel.from_xml_path(str(cfg.xml_path))
    addresses = _joint_addresses(model)

    terminal_rows = []
    for record in terminal.records:
        previous_vz = float(np.asarray(record["qvel"])[2])
        state = restore_snapshot(env, record, jax.random.PRNGKey(0))
        vector, diagnostic = _measure(
            model, state, env, previous_vz, addresses
        )
        terminal_rows.append({
            "id": record["id"],
            "cluster": int(record["descent_terminal_cluster"]),
            "runtime_label": record["runtime_replay_label"],
            "vector": vector, "diagnostic": diagnostic,
        })
    terminal_matrix = np.asarray([row["vector"] for row in terminal_rows])
    terminal_center = np.median(terminal_matrix, axis=0)
    terminal_scale = np.maximum(
        np.percentile(terminal_matrix, 75, axis=0)
        - np.percentile(terminal_matrix, 25, axis=0),
        OUTPUT_FLOORS,
    )
    cluster_rows = []
    for cluster in sorted({row["cluster"] for row in terminal_rows}):
        rows = [row for row in terminal_rows if row["cluster"] == cluster]
        matrix = np.asarray([row["vector"] for row in rows])
        source_metrics = next(
            row for row in terminal_report["clusters"]
            if int(row["cluster"]) == cluster
        )
        cluster_rows.append({
            "cluster": cluster, "states": len(rows),
            "all_contact_free": all(
                row["diagnostic"]["robot_terrain_contact_count"] == 0
                for row in rows
            ),
            "contact_counts": dict(Counter(
                row["diagnostic"]["robot_terrain_contact_count"]
                for row in rows
            )),
            "center": np.median(matrix, axis=0).tolist(),
            "scale": np.maximum(
                np.percentile(matrix, 75, axis=0)
                - np.percentile(matrix, 25, axis=0),
                OUTPUT_FLOORS,
            ).tolist(),
            "ranges": {
                name: _ranges(matrix[:, index])
                for index, name in enumerate(OUTPUTS)
            },
            "descent_success_rate":
                source_metrics["descent_controller_success_rate"],
            "landing_final_rate":
                source_metrics["landing_final_recovery_rate"],
        })

    def rollout(source_row, actions, seed):
        state = restore_snapshot(env, source_row, jax.random.PRNGKey(seed))
        previous_vz = float(np.asarray(state.data.qvel[2]))
        integrated_torque = np.zeros(3)
        history = []
        for tick, action in enumerate(actions):
            state = step(state, np.asarray(action, np.float32))
            vector, diagnostic = _measure(
                model, state, env, previous_vz, addresses
            )
            integrated_torque += np.asarray(
                diagnostic["net_terrain_torque_about_com"]
            ) * .02
            history.append({
                "tick": tick + 1, "vector": vector,
                "diagnostic": diagnostic,
            })
            previous_vz = vector[OUTPUTS.index("vz")]
        return state, vector, diagnostic, integrated_torque, history

    parents = {}
    selected_candidates = []
    for parent_index, parent in enumerate(a.parent):
        info = lineage["parents"].get(parent)
        if not info or info["status"] != "PASS":
            parents[parent] = {
                "status": "lineage_unavailable",
                "reason": None if not info else info.get("reason"),
            }
            continue
        separation_tick = int(info["separation_tick"])
        # Contact support can persist beyond the software controller handoff.
        # The authority window is a physical event interval, so retain every
        # recorded action before separation regardless of its section label.
        takeoff_actions = [
            row["action"] for row in info["trace"]
            if int(row["tick"]) > 0
            and int(row["tick"]) <= separation_tick
        ]
        if len(takeoff_actions) != separation_tick:
            raise RuntimeError(
                f"{parent}: incomplete action lineage to separation"
            )
        source_row = source_by_id[info["source_takeoff_state_id"]]
        seed = 12_000_000 + parent_index * 100_000
        _, base_vector, base_diag, base_torque, base_history = rollout(
            source_row, takeoff_actions, seed
        )
        expected = np.asarray(info["separation_h"], float)
        exact_h_error = float(np.max(np.abs(base_vector[:3] - expected)))
        windows = {}
        for requested in ("4", "8", "12", "full"):
            start, width, complete = _window_start(
                separation_tick, requested
            )
            columns = []
            column_meta = []
            invalid_pairs = 0
            max_changes = np.zeros(len(OUTPUTS))
            torque_changes = []
            for tick in range(start, separation_tick):
                for action_dim in range(4):
                    plus = np.asarray(takeoff_actions, float).copy()
                    minus = plus.copy()
                    plus[tick, action_dim] = np.clip(
                        plus[tick, action_dim] + a.amplitude, -1., 1.
                    )
                    minus[tick, action_dim] = np.clip(
                        minus[tick, action_dim] - a.amplitude, -1., 1.
                    )
                    denominator = (
                        plus[tick, action_dim] - minus[tick, action_dim]
                    )
                    if denominator <= 1e-9:
                        invalid_pairs += 1
                        continue
                    _, plus_vector, plus_diag, plus_torque, _ = rollout(
                        source_row, plus, seed
                    )
                    _, minus_vector, minus_diag, minus_torque, _ = rollout(
                        source_row, minus, seed
                    )
                    if (plus_diag["robot_terrain_contact_count"] != 0
                            or minus_diag["robot_terrain_contact_count"] != 0):
                        invalid_pairs += 1
                        continue
                    derivative = (
                        plus_vector - minus_vector
                    ) / denominator
                    columns.append(derivative)
                    max_changes = np.maximum(
                        max_changes,
                        np.maximum(
                            np.abs(plus_vector - base_vector),
                            np.abs(minus_vector - base_vector),
                        ),
                    )
                    torque_delta = max(
                        np.linalg.norm(plus_torque - base_torque),
                        np.linalg.norm(minus_torque - base_torque),
                    )
                    torque_changes.append(torque_delta)
                    column_meta.append({
                        "relative_action_tick": tick - separation_tick,
                        "absolute_action_tick": tick,
                        "action_dimension": action_dim,
                        "contact_torque_integral_change": torque_delta,
                        "hx_derivative": float(derivative[0]),
                        "hy_derivative": float(derivative[1]),
                    })
            matrix = (
                np.asarray(columns, float).T
                if columns else np.zeros((len(OUTPUTS), 0))
            )
            singular, rank = _rank(matrix, terminal_scale)
            linearized_box = (
                np.sum(np.abs(matrix), axis=1) * a.amplitude
                if matrix.size else np.zeros(len(OUTPUTS))
            )
            cluster_costs = []
            for cluster in cluster_rows:
                center = np.asarray(cluster["center"])
                scale = np.asarray(cluster["scale"])
                residual = np.maximum(
                    np.abs(center - base_vector) - linearized_box, 0.
                ) / scale
                dynamic_cost = float(np.linalg.norm(residual))
                cluster_costs.append({
                    "cluster": cluster["cluster"],
                    "linearized_dynamic_reachability_cost": dynamic_cost,
                    "residual_by_feature": dict(
                        zip(OUTPUTS, residual.tolist())
                    ),
                    "landing_final_rate": cluster["landing_final_rate"],
                    "descent_success_rate": cluster[
                        "descent_success_rate"
                    ],
                })
            preferred = min(cluster_costs, key=lambda row: (
                row["linearized_dynamic_reachability_cost"],
                -row["landing_final_rate"],
            ))
            significant = [
                row for row in column_meta
                if abs(row["hx_derivative"]) * a.amplitude >= .01
                or abs(row["hy_derivative"]) * a.amplitude >= .01
            ]
            windows[requested] = {
                "requested_ticks": requested,
                "actual_ticks": width,
                "complete_requested_history": complete,
                "start_relative_to_separation": start - separation_tick,
                "valid_columns": matrix.shape[1],
                "invalid_or_nonseparating_pairs": invalid_pairs,
                "normalized_response_rank": rank,
                "normalized_singular_values": singular,
                "maximum_single_perturbation_change": dict(
                    zip(OUTPUTS, max_changes.tolist())
                ),
                "linearized_bounded_box_half_width": dict(
                    zip(OUTPUTS, linearized_box.tolist())
                ),
                "maximum_contact_torque_integral_change": float(
                    max(torque_changes, default=0.)
                ),
                "first_significant_momentum_response_relative_tick": (
                    min((row["relative_action_tick"] for row in significant),
                        default=None)
                ),
                "last_significant_momentum_response_relative_tick": (
                    max((row["relative_action_tick"] for row in significant),
                        default=None)
                ),
                "preferred_terminal_cluster": preferred["cluster"],
                "preferred_cluster_cost": preferred[
                    "linearized_dynamic_reachability_cost"
                ],
                "cluster_costs": cluster_costs,
                "columns": column_meta,
            }
        full = windows["full"]
        parents[parent] = {
            "status": "PASS",
            "source_takeoff_state_id": source_row["id"],
            "separation_tick": separation_tick,
            "available_contact_tail_ticks": separation_tick,
            "baseline_separation": dict(
                zip(OUTPUTS, base_vector.tolist())
            ),
            "baseline_separation_contact_count": base_diag[
                "robot_terrain_contact_count"
            ],
            "baseline_external_torque_integral":
                base_torque.tolist(),
            "exact_lineage_h_linf": exact_h_error,
            "windows": windows,
            "full_contact_trace": [{
                "tick": row["tick"],
                "contact_count": row["diagnostic"][
                    "robot_terrain_contact_count"
                ],
                "net_terrain_torque_about_com": row["diagnostic"][
                    "net_terrain_torque_about_com"
                ],
                "centroidal_angular_momentum":
                    row["diagnostic"]["centroidal_angular_momentum"],
            } for row in base_history],
        }
        selected_candidates.append({
            "parent": parent,
            "shortest_available_windows": [
                requested for requested in ("4", "8", "12", "full")
                if windows[requested]["complete_requested_history"]
            ],
            "full_contact_preferred_cluster":
                full["preferred_terminal_cluster"],
            "full_contact_cost": full["preferred_cluster_cost"],
        })

    payload = {
        "status": "PASS",
        "artifact_role": "takeoff_tail_window_control_authority_diagnostic",
        "diagnostic_only": True, "ppo_authorization": False,
        "action_order": ["steer", "drive", "hip", "knee"],
        "finite_difference_amplitude": a.amplitude,
        "output_order": list(OUTPUTS),
        "output_scale": terminal_scale.tolist(),
        "coordinate_contract": {
            "frame": "MuJoCo world frame",
            "com": "whole-system mass-weighted body inertial COM",
            "angular_momentum_origin": "whole-system COM",
            "terminal_and_separation_implementation_identical": True,
        },
        "terminal_comparability": {
            "states": len(terminal_rows),
            "all_contact_free": all(
                row["diagnostic"]["robot_terrain_contact_count"] == 0
                for row in terminal_rows
            ),
            "contact_counts": dict(Counter(
                row["diagnostic"]["robot_terrain_contact_count"]
                for row in terminal_rows
            )),
            "external_impulse_between_unrelated_snapshots_assumed": False,
            "ballistic_compatibility_requires_constant_total_h": True,
        },
        "terminal_clusters": cluster_rows,
        "parents": parents,
        "parent_window_summary": selected_candidates,
        "xml_sha256": file_sha256(cfg.xml_path),
        "config_sha256": file_sha256(a.config),
        "takeoff_bank_sha256": file_sha256(a.takeoff_bank),
        "lineage_report_sha256": file_sha256(a.lineage_report),
        "terminal_bank_sha256": file_sha256(a.terminal_bank),
    }
    save_json(a.output, payload)
    print(json.dumps({
        "status": payload["status"],
        "terminal_contact_free":
            payload["terminal_comparability"]["all_contact_free"],
        "parents": {
            key: {
                "status": value["status"],
                "available_ticks": value.get("available_contact_tail_ticks"),
            } for key, value in parents.items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
