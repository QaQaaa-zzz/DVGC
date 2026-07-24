"""Centroidal/contact and authority-normalized pre-Apex audit."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import mujoco
import numpy as np

from dvgc.bank import SnapshotBank
from dvgc.centroidal import replay_centroidal
from dvgc.config import file_sha256, load_config
from dvgc.runtime import save_json


FEATURE_INDEX = {
    "roll": 3, "pitch": 4, "vz": 8, "wx": 9, "wy": 10,
    "hip": 13, "knee": 14,
}
RECOVERY_FEATURES = (
    "roll", "pitch", "wx", "wy", "hx", "hy", "vz", "hip", "knee",
    "joint_margin",
)
FLOORS = np.asarray([
    np.deg2rad(2), np.deg2rad(2), .2, .2, .03, .03, .1, .04, .04, .015,
])


def _joint_margin(model, qpos):
    margins = []
    for name in ("hip_joint", "knee_joint"):
        joint = model.joint(name)
        value = qpos[int(model.jnt_qposadr[joint.id])]
        margins.append(min(
            value - model.jnt_range[joint.id, 0],
            model.jnt_range[joint.id, 1] - value,
        ))
    return float(min(margins))


def _vector(record, centroidal, model):
    feature = np.asarray(record["physical_feature"], float)
    qpos = np.asarray(record["qpos"], float)
    momentum = centroidal["centroidal_angular_momentum"]
    values = {
        name: float(feature[index]) for name, index in FEATURE_INDEX.items()
    }
    values.update({
        "hx": float(momentum[0]), "hy": float(momentum[1]),
        "joint_margin": _joint_margin(model, qpos),
    })
    return np.asarray([values[name] for name in RECOVERY_FEATURES]), values


def _ranges(values):
    values = np.asarray(values, float)
    return {
        "min": float(np.min(values)), "p05": float(np.percentile(values, 5)),
        "p50": float(np.median(values)), "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _terminal_calibration(vectors):
    center = np.median(vectors, axis=0)
    scale = np.maximum(
        np.percentile(vectors, 75, axis=0)
        - np.percentile(vectors, 25, axis=0),
        FLOORS,
    )
    normalized = (vectors - center) / scale
    if len(vectors) > 1:
        distance = np.linalg.norm(
            normalized[:, None, :] - normalized[None, :, :], axis=2
        )
        np.fill_diagonal(distance, np.inf)
        threshold = float(np.percentile(np.min(distance, axis=1), 95))
    else:
        threshold = 1.
    return center, scale, max(threshold, 1.)


def _candidate_vector(terminal):
    omega = terminal["angular_velocity"]
    momentum = terminal["centroidal_angular_momentum"]
    return np.asarray([
        terminal["roll"], terminal["pitch"], omega[0], omega[1],
        momentum[0], momentum[1], terminal["vz"], terminal["hip"],
        terminal["knee"], terminal["joint_margin"],
    ], float)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--authority-bank", required=True)
    p.add_argument("--authority-report", required=True)
    p.add_argument("--terminal-bank", required=True)
    p.add_argument("--terminal-report", required=True)
    p.add_argument("--horizon-report", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--config", default="configs/default.json")
    a = p.parse_args()
    cfg = load_config(a.config)
    model = mujoco.MjModel.from_xml_path(str(cfg.xml_path))
    authority_bank = SnapshotBank.load(a.authority_bank)
    authority_report = json.loads(Path(a.authority_report).read_text())
    terminal_bank = SnapshotBank.load(a.terminal_bank)
    terminal_report = json.loads(Path(a.terminal_report).read_text())
    horizon_report = json.loads(Path(a.horizon_report).read_text())

    authority_rows = []
    by_parent = defaultdict(list)
    for record in authority_bank.records:
        centroidal = replay_centroidal(
            model, record["qpos"], record["qvel"], record.get("ctrl")
        )
        vector, values = _vector(record, centroidal, model)
        row = {
            "snapshot_id": record["id"],
            "parent": record["display_parent"],
            "parent_id": record["trajectory_parent_id"],
            "relative_to_apex": int(record["relative_to_apex"]),
            "recovery_features": values,
            "system_com": centroidal["system_com"],
            "com_velocity": centroidal["com_velocity"],
            "centroidal_angular_momentum":
                centroidal["centroidal_angular_momentum"],
            "body_contributions": centroidal["body_contributions"],
            "robot_terrain_contacts": centroidal["robot_terrain_contacts"],
            "net_terrain_impulse": (
                np.asarray(centroidal["net_terrain_force"]) * .02
            ).tolist(),
            "net_terrain_angular_impulse": (
                np.asarray(
                    centroidal["net_terrain_torque_about_com"]
                ) * .02
            ).tolist(),
            "crosscheck_linf":
                centroidal["angular_momentum_crosscheck_linf"],
            "_vector": vector,
        }
        authority_rows.append(row)
        by_parent[row["parent"]].append(row)

    terminal_rows = []
    for record in terminal_bank.records:
        centroidal = replay_centroidal(
            model, record["qpos"], record["qvel"], record.get("ctrl")
        )
        vector, values = _vector(record, centroidal, model)
        terminal_rows.append({
            "snapshot_id": record["id"],
            "cluster": int(record["descent_terminal_cluster"]),
            "runtime_label": record["runtime_replay_label"],
            "recovery_features": values,
            "centroidal_angular_momentum":
                centroidal["centroidal_angular_momentum"],
            "_vector": vector,
        })
    terminal_vectors = np.asarray([row["_vector"] for row in terminal_rows])
    center, scale, threshold = _terminal_calibration(terminal_vectors)

    cluster_results = []
    for cluster in sorted({row["cluster"] for row in terminal_rows}):
        group = [row for row in terminal_rows if row["cluster"] == cluster]
        cluster_results.append({
            "cluster": cluster, "states": len(group),
            "runtime_labels": dict(Counter(x["runtime_label"] for x in group)),
            "centroidal_momentum_ranges": {
                axis: _ranges([
                    row["centroidal_angular_momentum"][index] for row in group
                ])
                for index, axis in enumerate(("hx", "hy", "hz"))
            },
            "source_cluster_metrics": next(
                row for row in terminal_report["clusters"]
                if int(row["cluster"]) == cluster
            ),
        })

    parent_results = {}
    for parent, rows in by_parent.items():
        rows.sort(key=lambda row: row["relative_to_apex"])
        contact_rows = [row for row in rows if row["robot_terrain_contacts"]]
        last_contact = contact_rows[-1] if contact_rows else None
        airborne = [row for row in rows if not row["robot_terrain_contacts"]]
        hx = [row["centroidal_angular_momentum"][0] for row in airborne]
        conservation_span = (
            float(np.ptp(hx) / max(abs(np.mean(hx)), .03))
            if len(hx) >= 2 else None
        )
        info = authority_report["parent_results"][parent]
        parent_results[parent] = {
            "parent_id": rows[0]["parent_id"],
            "sampled_offsets": [row["relative_to_apex"] for row in rows],
            "last_sampled_contact_offset": (
                last_contact["relative_to_apex"] if last_contact else None
            ),
            "first_sampled_airborne_offset": (
                airborne[0]["relative_to_apex"] if airborne else None
            ),
            "separation_hx": (
                airborne[0]["centroidal_angular_momentum"][0]
                if airborne else None
            ),
            "airborne_hx_relative_span": conservation_span,
            "latest_effective_hx_correction_offset":
                info["latest_effective_relative_to_apex"],
            "event_h": rows[-1]["centroidal_angular_momentum"],
            "event_pose": {
                name: rows[-1]["recovery_features"][name]
                for name in ("roll", "pitch", "wx", "wy")
            },
        }

    recoverability = []
    selected_entries = []
    for outcome in horizon_report["outcomes"]:
        candidates = outcome["plans"][0]["candidate_terminals"]
        vectors = np.asarray([
            _candidate_vector(row["terminal"]) for row in candidates
            if not row["terminal"]["done"]
        ])
        if len(vectors):
            normalized = (vectors[:, None, :] - terminal_vectors[None, :, :])
            normalized /= scale[None, None, :]
            distances = np.linalg.norm(normalized, axis=2)
            minimum = float(np.min(distances))
        else:
            minimum = None
        parent_line = parent_results[outcome["parent"]]
        contact_supported = (
            parent_line["last_sampled_contact_offset"] is not None
            and outcome["actual_start_relative_to_apex"]
            <= parent_line["last_sampled_contact_offset"]
        )
        if minimum is not None and minimum <= threshold:
            label = "recoverable_with_local_feedback"
        elif contact_supported:
            label = "requires_contact_supported_shaping"
        else:
            label = "outside_recoverable_under_current_authority"
        row = {
            "parent": outcome["parent"],
            "start_snapshot_id": outcome["start_snapshot_id"],
            "start_relative_to_apex":
                outcome["actual_start_relative_to_apex"],
            "prediction_horizon": outcome["prediction_horizon"],
            "minimum_terminal_normalized_distance": minimum,
            "calibrated_threshold": threshold,
            "contact_supported_start": contact_supported,
            "diagnostic_label": label,
            "stable_16_ticks": outcome["stable_16_ticks"],
            "formal_descent_support_entry":
                outcome["formal_descent_support_entry"],
            "termination_reason": outcome["termination_reason"],
        }
        recoverability.append(row)
    # Parent-disjoint, at most two neighbouring snapshots per parent.
    for parent in sorted(by_parent):
        eligible = [
            row for row in recoverability
            if row["parent"] == parent
            and row["diagnostic_label"] !=
            "outside_recoverable_under_current_authority"
        ]
        eligible.sort(key=lambda row: (
            row["minimum_terminal_normalized_distance"]
            if row["minimum_terminal_normalized_distance"] is not None
            else float("inf"),
            abs(row["start_relative_to_apex"]),
        ))
        seen = set()
        for row in eligible:
            if row["start_snapshot_id"] in seen or len(seen) >= 2:
                continue
            selected_entries.append(row)
            seen.add(row["start_snapshot_id"])

    terminal_hx = terminal_vectors[:, RECOVERY_FEATURES.index("hx")]
    hx_lo = float(np.percentile(terminal_hx, 5))
    hx_hi = float(np.percentile(terminal_hx, 95))
    hx_iqr = float(np.percentile(terminal_hx, 75)
                   - np.percentile(terminal_hx, 25))
    outside = []
    conserved = []
    for parent, row in parent_results.items():
        hx = row["separation_hx"]
        if hx is not None and (hx < hx_lo - hx_iqr or hx > hx_hi + hx_iqr):
            outside.append(parent)
        if (row["airborne_hx_relative_span"] is not None
                and row["airborne_hx_relative_span"] <= .15):
            conserved.append(parent)
    robust = {"reference:131",
              "89ff1a0e3cb74319b16742932c97decf38be3a39a100d49e855613162e23fcf0"}
    if robust.intersection(outside) and robust.intersection(conserved):
        blocker = "takeoff_tail_centroidal_momentum_blocker"
    elif horizon_report["summary"]["stable_16_ticks"]:
        blocker = (
            "descent_support_coverage_gap"
            if not horizon_report["summary"]["formal_descent_support_entry"]
            else "downstream_controller_gap"
        )
    else:
        blocker = "ballistic_morphology_feedback_blocker"

    for row in authority_rows:
        row.pop("_vector")
    for row in terminal_rows:
        row.pop("_vector")
    payload = {
        "status": "PASS",
        "artifact_role": "apex_centroidal_contact_recoverability_diagnostic",
        "diagnostic_only": True, "not_physical_unreachability_claim": True,
        "apex_ppo_authorized": False,
        "xml_sha256": file_sha256(cfg.xml_path),
        "authority_bank_sha256": file_sha256(a.authority_bank),
        "terminal_bank_sha256": file_sha256(a.terminal_bank),
        "horizon_report_sha256": file_sha256(a.horizon_report),
        "recovery_feature_order": list(RECOVERY_FEATURES),
        "recovery_center": center.tolist(),
        "recovery_scale": scale.tolist(),
        "data_derived_nearest_terminal_threshold": threshold,
        "terminal_hx_reference": {
            "p05": hx_lo, "p95": hx_hi, "iqr": hx_iqr,
        },
        "parent_results": parent_results,
        "terminal_clusters": cluster_results,
        "authority_normalized_recoverability": recoverability,
        "selected_event_aligned_entry_proposals": selected_entries,
        "selection_parent_cap": 2,
        "separation_hx_outside_parents": outside,
        "airborne_hx_conserved_parents": conserved,
        "blocker_classification": blocker,
        "authority_snapshots": authority_rows,
        "terminal_snapshots": terminal_rows,
    }
    save_json(a.output, payload)
    print(json.dumps({
        "blocker_classification": blocker,
        "separation_hx_outside_parents": outside,
        "airborne_hx_conserved_parents": conserved,
        "selected_entries": len(selected_entries),
    }, indent=2))


if __name__ == "__main__":
    main()
