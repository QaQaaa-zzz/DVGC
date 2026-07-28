"""Targeted four-actuator authority smoke at two frozen Apex parents."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from cli.runtime_gate import source_fingerprint
from cli.stage_label_pilot import sample_from_state
from dvgc.bank import SnapshotBank
from dvgc.config import ACTION_MAPPING_VERSION, file_sha256, load_config
from dvgc.env import END_REASON, OrangeBikeDVGC
from dvgc.rollout import restore_snapshot
from dvgc.runtime import save_json


STARTS = Path("runs/stage_next_reset_v3_seed0_20260723/apex/feedback_bridge_v1/bridge_start_selection.pkl")
DEFAULT_RUN = Path("runs/apex_full_action_authority_v1/two_parent_smoke")
CHANNELS = ("steer", "rear_wheel_drive", "hip", "knee")
OUTPUTS = ("roll", "pitch", "wx", "wy", "wz", "vx", "vz")
OUTPUT_INDICES = (3, 4, 9, 10, 11, 6, 8)
OUTPUT_RESOLUTION = np.asarray((0.02, 0.02, 0.2, 0.2, 0.2, 0.08, 0.08), float)


def pulse_action(channel: int, amplitude: float) -> jnp.ndarray:
    return jnp.zeros((4,), jnp.float32).at[channel].set(amplitude)


def normalized_authority(plus: np.ndarray, minus: np.ndarray, amplitude: float) -> np.ndarray:
    derivative = (np.asarray(plus, float) - np.asarray(minus, float)) / (2 * amplitude)
    return np.abs(derivative) * 0.25 / OUTPUT_RESOLUTION


def rollout_pulse(env, step, record: dict, seed: int, channel: int, amplitude: float) -> dict:
    state = restore_snapshot(env, record, jax.random.PRNGKey(seed))
    previous_vz = float(np.asarray(state.data.qvel[2]))
    outputs = {}
    action = pulse_action(channel, amplitude)
    for tick in range(8):
        state = step(state, action if tick < 2 else jnp.zeros((4,), jnp.float32))
        sample = sample_from_state(env, state, previous_vz)
        feature = np.asarray(sample["physical_feature"], float)
        if tick + 1 in (1, 2, 4, 8):
            outputs[str(tick + 1)] = {
                "values": [float(feature[index]) for index in OUTPUT_INDICES],
                "done": bool(float(np.asarray(state.done)) > 0.5),
                "end_reason": END_REASON.get(int(np.asarray(state.info["end_code"])), "unknown"),
                "roll_margin": float(np.deg2rad(35.0) - abs(feature[3])),
                "pitch_margin": float(np.deg2rad(75.0) - abs(feature[4])),
            }
        if float(np.asarray(state.done)) > 0.5:
            break
        previous_vz = float(feature[8])
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--starts", default=str(STARTS))
    parser.add_argument("--run", default=str(DEFAULT_RUN))
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--seed", type=int, default=3_840_000_000)
    args = parser.parse_args()
    starts_path, root = Path(args.starts), Path(args.run)
    if root.exists():
        raise SystemExit(f"refusing overwrite {root}")
    gate = json.loads(Path("docs/RUNTIME_GATE.json").read_text())
    if gate.get("status") != "PASS" or gate.get("source_fingerprint") != source_fingerprint(Path.cwd()):
        raise SystemExit("runtime gate stale")
    cfg = load_config(args.config, {
        "training_stage": "flight", "use_bank_resets": False,
        "domain_randomization": False, "obs_noise_enable": False,
        "stage_reachability_objective": "",
    })
    if cfg.action_mapping_version != ACTION_MAPPING_VERSION:
        raise SystemExit("action mapping mismatch")
    starts = SnapshotBank.load(starts_path)
    if len(starts.records) != 2:
        raise SystemExit(f"expected two frozen Apex parents, got {len(starts.records)}")
    root.mkdir(parents=True)
    inputs = {
        "starts_sha256": file_sha256(starts_path), "xml_sha256": file_sha256(cfg.xml_path),
        "action_mapping_version": cfg.action_mapping_version, "seed": args.seed,
    }
    save_json(root / "manifest.json", {
        "status": "FROZEN_BEFORE_OUTCOMES", "inputs": inputs,
        "parents": [row["display_parent"] for row in starts.records],
        "channels": CHANNELS, "amplitudes": [0.12, 0.25], "pulse_ticks": 2,
        "measurement_ticks": [1, 2, 4, 8],
    })
    save_json(root / "cost_estimate.json", {
        "estimated_seconds": 180, "parents": 2, "channels": 4,
        "signed_pulses": 2, "amplitudes": 2, "horizon": 8, "PPO_steps": 0,
    })
    env = OrangeBikeDVGC(cfg, snapshot_bank=SnapshotBank())
    step = jax.jit(env.step)
    rows = []
    for parent_index, record in enumerate(starts.records):
        channel_rows = {}
        for channel, name in enumerate(CHANNELS):
            amplitude_rows = {}
            for amplitude in (0.12, 0.25):
                base_seed = args.seed + parent_index * 100_000 + channel * 10_000 + int(amplitude * 100)
                plus = rollout_pulse(env, step, record, base_seed, channel, amplitude)
                minus = rollout_pulse(env, step, record, base_seed, channel, -amplitude)
                response = {}
                for horizon in (1, 2, 4, 8):
                    key = str(horizon)
                    if key not in plus or key not in minus:
                        continue
                    authority = normalized_authority(plus[key]["values"], minus[key]["values"], amplitude)
                    response[key] = {
                        "normalized_authority": dict(zip(OUTPUTS, authority.tolist(), strict=True)),
                        "plus": plus[key], "minus": minus[key],
                    }
                amplitude_rows[str(amplitude)] = response
            max_roll = max(
                row["normalized_authority"]["roll"] for amplitude in amplitude_rows.values() for row in amplitude.values()
            )
            max_roll_rate = max(
                row["normalized_authority"]["wx"] for amplitude in amplitude_rows.values() for row in amplitude.values()
            )
            max_pitch = max(
                row["normalized_authority"]["pitch"] for amplitude in amplitude_rows.values() for row in amplitude.values()
            )
            channel_rows[name] = {
                "max_roll_authority": max_roll, "max_roll_rate_authority": max_roll_rate,
                "max_pitch_authority": max_pitch,
                "roll_effective": bool(max(max_roll, max_roll_rate) >= 1.0),
                "responses": amplitude_rows,
            }
        rows.append({"parent": record["display_parent"], "channels": channel_rows})
    non_leg_effective = {
        name: sum(row["channels"][name]["roll_effective"] for row in rows)
        for name in ("steer", "rear_wheel_drive")
    }
    status = "PASS" if max(non_leg_effective.values()) > 0 else "FAIL"
    report = {
        "status": status, "artifact_role": "apex_full_action_local_authority_diagnostic",
        "not_controllability_or_tube_evidence": True, "inputs": inputs,
        "rows": rows, "non_leg_roll_effective_parents": non_leg_effective,
        "PPO_authorization": False,
        "next": "extend_bounded_feedback_action_set" if status == "PASS" else "non_leg_channels_not_locally_effective",
    }
    save_json(root / "APEX_FULL_ACTION_AUTHORITY_V1_REPORT.json", report)
    save_json(root / "completed.json", {"status": status, "next": report["next"]})
    print(json.dumps({key: report[key] for key in ("status", "non_leg_roll_effective_parents", "next")}, indent=2))


if __name__ == "__main__":
    main()
