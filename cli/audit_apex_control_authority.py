"""Finite-difference hip/knee control authority around physical Apex."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
import numpy as np

from cli.acquire_ascent_apex_parents import _local_action
from cli.search_takeoff_actions import SEQUENCES, action_at
from cli.stage_label_pilot import sample_from_state
from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256, load_config
from dvgc.env import END_REASON, OrangeBikeDVGC
from dvgc.reset_geometry import GroundSupportSolver
from dvgc.rollout import restore_snapshot
from dvgc.runtime import save_json
from dvgc.stage_reachability import evaluate_entry


OUTPUT_NAMES = (
    "roll", "pitch", "wx", "wy", "wz", "vx", "vz", "hip", "knee",
    "hip_velocity", "knee_velocity",
)


def _parent_id(record):
    return record.get(
        "independent_trajectory_parent_id",
        record.get("source_parent_id", record.get("trajectory_parent_id")),
    )


def _load_parent_specs(run_root: Path, apex_bank: SnapshotBank):
    grouped = defaultdict(list)
    for record in apex_bank.records:
        if record.get("candidate_kind") == "apex_dynamically_reached":
            grouped[_parent_id(record)].append(record)
    specs = []
    reference_bank = SnapshotBank.load(
        run_root / "ascent/reverse_diagnostic_v4_6.pkl"
    )
    for parent, records in grouped.items():
        if any(row.get("source_reference_index") == 131 for row in records):
            specs.append({
                "parent_id": parent, "display_parent": "reference:131",
                "row": next(row for row in reference_bank.records
                            if row.get("reference_index") == 131),
                "parameters": {
                    "round": "known", "hip_amplitude": 1.,
                    "knee_ratio": .5, "start_tick": 0, "duration": 50,
                },
                "event_tick": 15, "action_mode": "reference",
                "source_artifact": str(
                    run_root / "ascent/reverse_diagnostic_v4_6.pkl"
                ),
            })
            continue
        found = None
        for version in ("independent_parent_acquisition_v1",
                        "independent_parent_acquisition_v2_24"):
            root = run_root / "ascent" / version
            report_path = root / "report.json"
            entry_path = root / "fresh_ascent_entries.pkl"
            if not report_path.exists() or not entry_path.exists():
                continue
            report = json.loads(report_path.read_text())
            outcome = next((
                row for row in report["search_outcomes"]
                if row["success"] and row["trajectory_parent_id"] == parent
            ), None)
            if outcome is None:
                continue
            entries = SnapshotBank.load(entry_path)
            entry = next(row for row in entries.records
                         if row["id"] == outcome["source_ascent_entry_id"])
            found = {
                "parent_id": parent, "display_parent": parent,
                "row": entry, "parameters": outcome["parameters"],
                "event_tick": int(outcome["entry_tick"]),
                "action_mode": "local", "source_artifact": str(entry_path),
            }
            break
        if found is None:
            raise RuntimeError(f"missing trajectory lineage for parent {parent}")
        specs.append(found)
    return sorted(specs, key=lambda row: row["display_parent"])


def _nominal_action(spec, tick):
    if spec["action_mode"] == "reference":
        return action_at(SEQUENCES["hip_full_knee_half"], tick)
    return _local_action(spec["parameters"], tick)


def _offset_ticks(event_tick):
    candidates = {
        max(1, event_tick - 12), max(1, event_tick - 8),
        max(1, event_tick - 4), event_tick,
    }
    while len(candidates) < 4:
        candidates.add(max(1, event_tick - len(candidates)))
        if len(candidates) == 1 and event_tick == 1:
            break
    return sorted(candidates)


def _capture_parent(env, step, cfg, spec, seed):
    wanted = _offset_ticks(spec["event_tick"])
    state = restore_snapshot(env, spec["row"], jax.random.PRNGKey(seed))
    previous_vz = float(np.asarray(state.data.qvel[2]))
    captured = []
    for tick in range(max(wanted)):
        action = _nominal_action(spec, tick)
        state = step(state, action)
        sample = sample_from_state(env, state, previous_vz)
        if tick + 1 in wanted:
            snapshot = env.snapshot_record(state, "flight")
            snapshot.update({
                "id": hashlib.sha256(
                    f"apex-control:{spec['parent_id']}:{tick+1}:{seed}".encode()
                ).hexdigest()[:32],
                "candidate_kind": "pre_apex_control_authority_snapshot",
                "trajectory_parent_id": spec["parent_id"],
                "display_parent": spec["display_parent"],
                "nominal_trajectory_tick": tick + 1,
                "nominal_apex_tick": spec["event_tick"],
                "relative_to_apex": tick + 1 - spec["event_tick"],
                "nominal_action": np.asarray(action).tolist(),
                "trajectory_source_artifact": spec["source_artifact"],
                "trajectory_parameters": spec["parameters"],
            })
            captured.append(snapshot)
        if float(np.asarray(state.done)) > .5:
            code = int(np.asarray(state.info["end_code"]))
            raise RuntimeError(
                f"nominal parent {spec['parent_id']} failed before capture: "
                f"{END_REASON.get(code, code)}"
            )
        previous_vz = float(sample["physical_feature"][8])
    return captured


def _measure(model, geometry, state, env, previous_vz):
    sample = sample_from_state(env, state, previous_vz)
    feature = np.asarray(sample["physical_feature"], float)
    hip_id = model.joint("hip_joint").id
    knee_id = model.joint("knee_joint").id
    hip_q = int(model.jnt_qposadr[hip_id]); knee_q = int(model.jnt_qposadr[knee_id])
    hip_v = int(model.jnt_dofadr[hip_id]); knee_v = int(model.jnt_dofadr[knee_id])
    qpos = np.asarray(state.data.qpos); qvel = np.asarray(state.data.qvel)
    contact = geometry.measure(qpos, qvel, np.asarray(state.data.ctrl))
    return {
        "roll": float(feature[3]), "pitch": float(feature[4]),
        "wx": float(feature[9]), "wy": float(feature[10]),
        "wz": float(feature[11]), "vx": float(feature[6]),
        "vz": float(feature[8]), "hip": float(qpos[hip_q]),
        "knee": float(qpos[knee_q]), "hip_velocity": float(qvel[hip_v]),
        "knee_velocity": float(qvel[knee_v]),
        "roll_gate_margin": float(math.radians(35.) - abs(feature[3])),
        "pitch_gate_margin": float(math.radians(75.) - abs(feature[4])),
        "wheel_contacts": int(contact["wheel_contacts"]),
        "body_contacts": int(contact["body_contacts"]),
        "wheel_clearance": float(contact["wheel_min"]),
        "phase": int(np.asarray(state.info["phase"])),
        "done": bool(float(np.asarray(state.done)) > .5),
    }


def _pulse_action(name, amplitude):
    table = {
        "neutral": (0., 0.), "hip_positive": (amplitude, 0.),
        "hip_negative": (-amplitude, 0.), "knee_positive": (0., amplitude),
        "knee_negative": (0., -amplitude),
        "same_positive": (amplitude, amplitude),
        "same_negative": (-amplitude, -amplitude),
        "opposite_positive": (amplitude, -amplitude),
        "opposite_negative": (-amplitude, amplitude),
    }
    hip, knee = table[name]
    return jp.asarray([0., 0., hip, knee], jp.float32)


def _run_pulse(env, step, model, geometry, snapshot, seed, name, amplitude):
    state = restore_snapshot(env, snapshot, jax.random.PRNGKey(seed))
    previous_vz = float(np.asarray(state.data.qvel[2]))
    outputs = {}
    action = _pulse_action(name, amplitude)
    for tick in range(8):
        applied = action if tick < 2 else jp.zeros((4,), jp.float32)
        state = step(state, applied)
        measurement = _measure(model, geometry, state, env, previous_vz)
        if tick + 1 in (1, 2, 4, 8):
            outputs[str(tick + 1)] = measurement
        if measurement["done"]:
            break
        previous_vz = measurement["vz"]
    return outputs


def _response_matrix(responses, amplitude, horizon):
    columns = []
    for positive, negative in (
        ("hip_positive", "hip_negative"),
        ("knee_positive", "knee_negative"),
    ):
        plus = responses[positive][str(horizon)]
        minus = responses[negative][str(horizon)]
        columns.append([
            (plus[name] - minus[name]) / (2 * amplitude)
            for name in OUTPUT_NAMES
        ])
    return np.asarray(columns, float).T


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--apex-bank", required=True)
    p.add_argument("--output-bank", required=True)
    p.add_argument("--output-report", required=True)
    p.add_argument("--run-root", required=True)
    p.add_argument("--config", default="configs/default.json")
    p.add_argument("--seed", type=int, default=11_400_000)
    a = p.parse_args()

    bank = SnapshotBank.load(a.apex_bank)
    run_root = Path(a.run_root)
    cfg = load_config(a.config, {
        "training_stage": "flight", "use_bank_resets": False,
        "domain_randomization": False, "obs_noise_enable": False,
        "stage_reachability_objective": "",
    })
    env = OrangeBikeDVGC(cfg, snapshot_bank=SnapshotBank())
    step = jax.jit(env.step)
    model = mujoco.MjModel.from_xml_path(str(cfg.xml_path))
    geometry = GroundSupportSolver(cfg.xml_path)
    parents = _load_parent_specs(run_root, bank)
    snapshots = []
    for pi, spec in enumerate(parents):
        snapshots.extend(_capture_parent(
            env, step, cfg, spec, a.seed + pi * 10_000
        ))
    amplitudes = (.12, .25)
    pulse_names = (
        "neutral", "hip_positive", "hip_negative", "knee_positive",
        "knee_negative", "same_positive", "same_negative",
        "opposite_positive", "opposite_negative",
    )
    rows = []
    for si, snapshot in enumerate(snapshots):
        amplitude_results = {}
        for amplitude in amplitudes:
            responses = {
                name: _run_pulse(
                    env, step, model, geometry, snapshot,
                    a.seed + 1_000_000 + si * 1000 + int(amplitude * 100) * 10,
                    name, amplitude,
                )
                for name in pulse_names
            }
            matrices = {}
            for horizon in (1, 2, 4, 8):
                if all(str(horizon) in responses[name]
                       for name in ("hip_positive", "hip_negative",
                                    "knee_positive", "knee_negative")):
                    matrix = _response_matrix(responses, amplitude, horizon)
                    pose_matrix = matrix[[0, 1, 2, 3], :]
                    normalized = pose_matrix / np.asarray(
                        [.02, .02, .2, .2]
                    )[:, None]
                    singular = np.linalg.svd(normalized, compute_uv=False)
                    matrices[str(horizon)] = {
                        "matrix_outputs_by_hip_knee": matrix.tolist(),
                        "pose_singular_values_normalized": singular.tolist(),
                        "effective_rank": int(np.sum(
                            singular > max(.1, .1 * singular[0])
                        )),
                    }
            amplitude_results[str(amplitude)] = {
                "responses": responses, "finite_difference": matrices,
            }
        small = amplitude_results["0.12"]["finite_difference"].get("4")
        if small is None:
            roll_authority = pitch_authority = 0.; rank = 0
        else:
            matrix = np.asarray(small["matrix_outputs_by_hip_knee"], float)
            roll_authority = float(max(
                np.max(np.abs(matrix[0])) * .25 / .02,
                np.max(np.abs(matrix[2])) * .25 / .2,
            ))
            pitch_authority = float(max(
                np.max(np.abs(matrix[1])) * .25 / .02,
                np.max(np.abs(matrix[3])) * .25 / .2,
            ))
            rank = int(small["effective_rank"])
        rows.append({
            "snapshot_id": snapshot["id"],
            "parent_id": snapshot["trajectory_parent_id"],
            "display_parent": snapshot["display_parent"],
            "relative_to_apex": snapshot["relative_to_apex"],
            "nominal_trajectory_tick": snapshot["nominal_trajectory_tick"],
            "roll_authority_normalized": roll_authority,
            "pitch_authority_normalized": pitch_authority,
            "effective_pose_rank": rank,
            "amplitudes": amplitude_results,
        })
    parent_results = {}
    for parent in parents:
        group = [row for row in rows if row["parent_id"] == parent["parent_id"]]
        effective = [
            row for row in group
            if row["roll_authority_normalized"] >= 1.
            and row["effective_pose_rank"] >= 2
        ]
        event = max(group, key=lambda row: row["relative_to_apex"])
        if (event["roll_authority_normalized"] >= 1.
                and event["effective_pose_rank"] >= 2):
            classification = "apex_local_response_detected"
        elif effective:
            classification = "pre_apex_correction_required"
        else:
            classification = "upstream_entry_shaping_required"
        latest = max(
            (row["relative_to_apex"] for row in effective), default=None
        )
        parent_results[parent["display_parent"]] = {
            "parent_id": parent["parent_id"], "classification": classification,
            "latest_effective_relative_to_apex": latest,
            "event_roll_authority_normalized": event["roll_authority_normalized"],
            "event_pitch_authority_normalized": event["pitch_authority_normalized"],
            "event_effective_pose_rank": event["effective_pose_rank"],
            "offsets": [{
                "relative_to_apex": row["relative_to_apex"],
                "roll_authority_normalized": row["roll_authority_normalized"],
                "pitch_authority_normalized": row["pitch_authority_normalized"],
                "effective_pose_rank": row["effective_pose_rank"],
            } for row in group],
        }
    SnapshotBank(snapshots, {
        "artifact_role": "pre_apex_control_authority_snapshots",
        "certified_tube": False, "safe_claim_allowed": False,
        "source_apex_bank_sha256": file_sha256(a.apex_bank),
        "generation_seed": a.seed,
    }).save(a.output_bank)
    payload = {
        "status": "PASS",
        "artifact_role": "apex_pre_post_control_authority_diagnostic",
        "not_global_controllability_claim": True,
        "apex_bank_sha256": file_sha256(a.apex_bank),
        "xml_sha256": file_sha256(cfg.xml_path),
        "parents": len(parents), "snapshots": len(snapshots),
        "pulse_amplitudes": list(amplitudes), "pulse_duration_ticks": 2,
        "measurement_horizons": [1, 2, 4, 8],
        "output_order": list(OUTPUT_NAMES),
        "classification_rule": (
            "effective if a 0.25 normalized action can change roll or roll-rate "
            "by the declared physical resolution within four ticks and the "
            "normalized [roll,pitch,wx,wy] response has rank two; this detects "
            "a local response and is not a closed-loop correctability claim"
        ),
        "parent_results": parent_results,
        "rows": rows,
        "output_bank": str(Path(a.output_bank).resolve()),
        "output_bank_sha256": file_sha256(a.output_bank),
    }
    save_json(a.output_report, payload)
    print(json.dumps({k: v for k, v in payload.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
