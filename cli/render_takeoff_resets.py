"""Render corrected Takeoff resets with joint/contact telemetry overlays."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jp
import mediapy as media
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from dvgc.bank import SnapshotBank
from dvgc.config import load_config
from dvgc.env import OrangeBikeDVGC
from dvgc.reset_geometry import GroundSupportSolver
from dvgc.rollout import restore_snapshot
from dvgc.runtime import save_json


def decorate(frame, lines):
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rounded_rectangle((12, 12, 565, 164), radius=8, fill=(0, 0, 0, 175))
    font = ImageFont.load_default(size=19)
    for index, line in enumerate(lines):
        draw.text((25, 22 + 25 * index), line, font=font, fill=(255, 255, 255, 255))
    return np.asarray(image)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--action-search", default="")
    parser.add_argument("--seed", type=int, default=9_940_000)
    parser.add_argument("--config", default="configs/default.json")
    args = parser.parse_args()
    bank = SnapshotBank.load(args.bank)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config, {
        "training_stage": "takeoff", "use_bank_resets": False,
        "domain_randomization": False, "obs_noise_enable": False,
    })
    env = OrangeBikeDVGC(cfg, snapshot_bank=SnapshotBank())
    step_fn = jax.jit(env.step)
    support = GroundSupportSolver(cfg.xml_path)
    model = env.mj_model
    hip_q = int(model.jnt_qposadr[model.joint("hip_joint").id])
    knee_q = int(model.jnt_qposadr[model.joint("knee_joint").id])
    root_q = int(model.jnt_qposadr[model.joint("floating_base_joint").id])
    action_traces = {}
    if args.action_search:
        search = json.loads(Path(args.action_search).read_text())
        successful = sorted(
            (item for item in search["outcomes"] if item["success"]),
            key=lambda item: (int(item["entry_tick"]), item["candidate_id"]),
        )
        for item in successful:
            action_traces.setdefault(
                item["candidate_id"], [point["action"] for point in item["trace"]]
            )
        chosen = [row for row in bank.records if row["id"] in action_traces][:3]
        if len(chosen) < 3:
            raise SystemExit("Action search has fewer than three successful reset states")
    else:
        chosen = [
            next(row for row in bank.records if row["candidate_kind"] == "canonical_compressed"),
            next(row for row in bank.records if row["candidate_kind"] == "reference_aligned_compressed"),
            next(row for row in reversed(bank.records) if row["candidate_kind"] == "reference_aligned_compressed"),
        ]
    reports = []
    for video_index, row in enumerate(chosen, 1):
        state = restore_snapshot(env, row, jax.random.PRNGKey(args.seed + video_index))
        states = [state]
        actions = [jp.zeros(4, jp.float32)]
        selected_actions = action_traces.get(row["id"], [])
        rollout_steps = len(selected_actions) if selected_actions else 24
        for tick in range(rollout_steps):
            action = (
                jp.asarray(selected_actions[tick], jp.float32)
                if selected_actions
                else jp.zeros(4, jp.float32) if tick < 5
                else jp.asarray([0, 0, 1, 1], jp.float32)
            )
            state = step_fn(state, action)
            states.append(state)
            actions.append(action)
            if float(np.asarray(jax.device_get(state.done))) > 0.5:
                break
        renderer = mujoco.Renderer(model, height=540, width=960)
        data = mujoco.MjData(model)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.azimuth, camera.elevation, camera.distance = 90, -10, 1.8
        frames = []
        telemetry = []
        for tick, (current, action) in enumerate(zip(states, actions)):
            qpos = np.asarray(jax.device_get(current.data.qpos))
            qvel = np.asarray(jax.device_get(current.data.qvel))
            ctrl = np.asarray(jax.device_get(current.data.ctrl))
            contact = support.measure(qpos, qvel, ctrl)
            data.qpos[:] = qpos
            data.qvel[:] = qvel
            data.ctrl[:] = ctrl
            mujoco.mj_forward(model, data)
            camera.lookat[:] = [qpos[root_q] + 0.25, qpos[root_q + 1], 0.22]
            renderer.update_scene(data, camera=camera)
            frame = renderer.render()
            phase = int(np.asarray(jax.device_get(current.info["phase"])))
            lines = [
                f"corrected Takeoff reset | {row['candidate_kind']}",
                f"step={tick:02d} action hip/knee={float(action[2]):+.1f}/{float(action[3]):+.1f}",
                f"hip={qpos[hip_q]:+.3f} rad   knee={qpos[knee_q]:+.3f} rad",
                f"root z={qpos[root_q+2]:.3f} m   vz={qvel[2]:+.3f} m/s",
                f"phase={phase} wheel contacts={contact['wheel_contacts']} body contacts={contact['body_contacts']}",
                f"wheel min clearance={contact['wheel_min']*1000:+.2f} mm",
            ]
            frames.append(decorate(frame, lines))
            telemetry.append({
                "step": tick, "hip": float(qpos[hip_q]), "knee": float(qpos[knee_q]),
                "root_z": float(qpos[root_q + 2]), "vertical_velocity": float(qvel[2]),
                "phase": phase, "wheel_contacts": int(contact["wheel_contacts"]),
                "body_contacts": int(contact["body_contacts"]),
                "wheel_clearance_min_m": float(contact["wheel_min"]),
            })
        renderer.close()
        frames = [frames[0]] * 15 + frames + [frames[-1]] * 12
        path = output / f"corrected_takeoff_reset_{video_index}_{row['candidate_kind']}.mp4"
        media.write_video(path, frames, fps=25, codec="h264", crf=18)
        reports.append({
            "video": str(path.resolve()), "candidate_id": row["id"],
            "candidate_kind": row["candidate_kind"], "reference_index": row["reference_index"],
            "bounded_action_success": bool(selected_actions),
            "telemetry": telemetry,
        })
    save_json(output / "video_manifest.json", {
        "status": "PASS", "bank": str(Path(args.bank).resolve()),
        "reset_protocol_sha256": bank.metadata["reset_protocol_sha256"],
        "playback_speed": "0.5x", "videos": reports,
    })
    print(json.dumps([{k: v for k, v in row.items() if k != "telemetry"} for row in reports], indent=2))


if __name__ == "__main__":
    main()
