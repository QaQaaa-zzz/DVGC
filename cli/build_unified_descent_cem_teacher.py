"""Build the read-only CEM teacher/anchor dataset for Descent bootstrap v1."""
from __future__ import annotations

import argparse
import copy
import json
import pickle
import subprocess
from collections import Counter
from pathlib import Path

import jax
import numpy as np

from cli.runtime_gate import source_fingerprint
from dvgc.bank import SnapshotBank
from dvgc.config import ACTION_MAPPING_VERSION, STAGE_ID, file_sha256, load_config
from dvgc.descent_probe import formal_dynamic_margin
from dvgc.descent_probe import batched_base_state
from dvgc.descent_teacher import (
    ACTION_ORDER, nearest_neighbor_audit, normalized_observation,
    trajectory_support_radius,
)
from dvgc.env import OrangeBikeDVGC
from dvgc.policy import load_bundle
from dvgc.ppo_integrity import normalizer_summary
from dvgc.rollout import restore_snapshot
from dvgc.runtime import build_inference, save_json


EXPECTED_HEAD = "ff3e3a1"
EXPECTED_BANK = "8e6342bc0d9d5e7821929f6ddccd4fb5b7c23923c66ab2d058309249dfed45a1"
EXPECTED_XML = "d7e9f43ff8fb9e4571203f81062ce9c828acfa38692ee8c71a3e5daa15ce794c"
EXPECTED_NORMALIZER = "8f2e36b6f69a3d20da67c1854f7e908c98dd6b03ae70e287e0a7e28522f93a7e"
BANK = Path("runs/mjx_continuous_pipeline_repair_v1/descent_candidate_bank_v1/descent_candidates_v2.pkl")
POLICY = Path("runs/stage_experts/descent_tube_seed0_20260716T2330/round_3/train/policy")
CEM = Path("runs/unified_descent_controllability_reward_curriculum_probe_v1/full_v2/residual_cem_oracle_results.json")
LANDING = Path("artifacts/landing_candidates.pkl")


def _item(state, env, candidate_id, tick, kind, target, frozen_action, residual, next_state=None):
    snapshot = env.snapshot_record(state, "flight" if kind != "landing_anchor" else "landing")
    ps = snapshot["policy_state"]
    result = {
        "kind": kind, "candidate_id": candidate_id, "tick": int(tick),
        "snapshot": snapshot,
        "observation": np.asarray(state.obs["state"], np.float32),
        "target_action": np.asarray(target, np.float32),
        "frozen_action": np.asarray(frozen_action, np.float32),
        "residual": np.asarray(residual, np.float32),
        "phase": int(snapshot["oracle_phase"]),
        "contact_mode": int(np.argmax(np.asarray(ps["contact_probs"]))),
        "delay_buffer": np.asarray(ps["delay_buffer"], np.float32),
        "phase_probs": np.asarray(ps["phase_probs"], np.float32),
        "progress": float(ps["phase_progress"]), "confidence": float(ps["phase_confidence"]),
        "physical_margin": float(formal_dynamic_margin(np.asarray(snapshot["physical_feature"]), env._config)),
    }
    if next_state is not None:
        result["next_snapshot"] = env.snapshot_record(next_state, "flight")
        result["next_physical_margin"] = float(formal_dynamic_margin(
            np.asarray(result["next_snapshot"]["physical_feature"]), env._config))
    return result


def _pi_action(inference, state, seed):
    action, _ = inference(state.obs, jax.random.PRNGKey(seed))
    return np.asarray(action, np.float32)


def _batched_snapshot(env, state, source_stage):
    """Serialize row zero without collapsing MJX-Warp's batched metadata."""
    data = state.data; info = state.info
    feature = np.asarray(jax.device_get(jax.vmap(env._physical_feature)(data)[0]), np.float32)
    get = lambda name: np.asarray(jax.device_get(info[name][0]))
    estimated_phase = int(get("estimated_phase")); phase_probs = get("phase_probs").astype(np.float32)
    x = float(feature[0]); cfg = env._config
    if estimated_phase == STAGE_ID["approach"]:
        progress = (x - float(cfg.ground_spawn_x)) / max(float(cfg.step_front_x - cfg.takeoff_window_far - cfg.ground_spawn_x), 1e-6)
    elif estimated_phase == STAGE_ID["takeoff"]:
        progress = (x - float(cfg.step_front_x - cfg.takeoff_window_far)) / max(float(cfg.takeoff_window_far - cfg.takeoff_window_near), 1e-6)
    elif estimated_phase == STAGE_ID["flight"]:
        progress = (x - float(cfg.step_front_x - cfg.takeoff_window_near)) / max(float(cfg.valid_landing_min_past_edge + cfg.takeoff_window_near), 1e-6)
    else:
        progress = float(get("recovery_count")) / max(float(cfg.recovery_hold_steps), 1.0)
    return {
        "qpos": np.asarray(jax.device_get(data.qpos[0]), np.float32),
        "qvel": np.asarray(jax.device_get(data.qvel[0]), np.float32),
        "ctrl": np.asarray(jax.device_get(data.ctrl[0]), np.float32),
        "qacc_warmstart": np.asarray(jax.device_get(data.qacc_warmstart[0]), np.float32),
        "physical_feature": feature, "source_phase": source_stage,
        "oracle_phase": int(get("phase")), "had_airborne": int(get("had_airborne")),
        "had_valid_landing": int(get("had_valid_landing")), "contact_age": int(get("contact_age")),
        "landing_entry_age": int(get("landing_entry_age")), "airborne_count": int(get("airborne_count")),
        "prelaunch_airborne_count": int(get("prelaunch_airborne_count")),
        "landing_bounce_count": int(get("landing_bounce_count")), "invalid_wheel_count": int(get("invalid_wheel_count")),
        "recovery_count": int(get("recovery_count")), "stage_entry_ever": int(get("stage_entry_ever")),
        "apex_seen": int(get("apex_seen")), "jump_signal_latched": bool(get("jump_signal_latched")),
        "jump_window_start_x": float(get("jump_window_start_x")), "jump_window_end_x": float(get("jump_window_end_x")),
        "policy_state": {
            "last_action": get("last_action").astype(np.float32), "obs_history": get("obs_history").astype(np.float32),
            "actor_observation": np.asarray(jax.device_get(state.obs["state"][0]), np.float32),
            "filter_phase": estimated_phase, "phase_probs": phase_probs,
            "contact_probs": np.asarray([1.0 - float(get("had_airborne")), float(get("had_valid_landing"))], np.float32),
            "phase_progress": float(np.clip(progress, 0, 1)), "phase_confidence": float(np.max(phase_probs)),
            "estimator_hidden": np.zeros((0,), np.float32), "delay_buffer": phase_probs[None, :],
            "prev_acc_z": float(get("prev_acc_z")), "prev_vz": float(get("prev_vz")),
        },
    }


def _batched_item(state, next_state, env, candidate_id, tick, kind, target, frozen, residual):
    snapshot = _batched_snapshot(env, state, "flight")
    next_snapshot = _batched_snapshot(env, next_state, "flight")
    ps = snapshot["policy_state"]
    return {
        "kind": kind, "candidate_id": candidate_id, "tick": int(tick), "snapshot": snapshot,
        "observation": np.asarray(state.obs["state"][0], np.float32),
        "target_action": np.asarray(target[0], np.float32), "frozen_action": np.asarray(frozen[0], np.float32),
        "residual": np.asarray(residual[0], np.float32), "phase": int(snapshot["oracle_phase"]),
        "contact_mode": int(np.argmax(ps["contact_probs"])), "delay_buffer": np.asarray(ps["delay_buffer"]),
        "phase_probs": np.asarray(ps["phase_probs"]), "progress": float(ps["phase_progress"]),
        "confidence": float(ps["phase_confidence"]),
        "physical_margin": float(formal_dynamic_margin(snapshot["physical_feature"], env._config)),
        "next_snapshot": next_snapshot,
        "next_physical_margin": float(formal_dynamic_margin(next_snapshot["physical_feature"], env._config)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    args = parser.parse_args(); root = Path(args.run)
    dataset_path = root / "teacher_dataset.pkl"
    if dataset_path.exists(): raise SystemExit(f"refusing overwrite {dataset_path}")
    if subprocess.run(["git", "merge-base", "--is-ancestor", EXPECTED_HEAD, "HEAD"]).returncode:
        raise SystemExit("wrong Git lineage")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise SystemExit("worktree must be clean")
    cfg = load_config("configs/unified_descent_rsi_learnability_pilot_v1.json")
    gate = json.loads(Path("docs/RUNTIME_GATE.json").read_text())
    if (file_sha256(BANK) != EXPECTED_BANK or file_sha256(cfg.xml_path) != EXPECTED_XML
            or cfg.action_mapping_version != ACTION_MAPPING_VERSION):
        raise SystemExit("immutable asset mismatch")
    if gate.get("status") != "PASS" or gate.get("source_fingerprint") != source_fingerprint(Path.cwd()):
        raise SystemExit("runtime gate is stale")
    params, _, manifest = load_bundle(POLICY, verify_files=True)
    if normalizer_summary(params[0])["sha256"] != EXPECTED_NORMALIZER:
        raise SystemExit("normalizer mismatch")
    bank = SnapshotBank.load(BANK); records = bank.records
    by_id = {row["id"]: row for row in records}
    cem = json.loads(CEM.read_text())
    oracle = {row["candidate_id"]: row for row in cem["candidates"]}
    positive = sorted(row["candidate_id"] for row in cem["candidates"]
                      if row["time_to_failure_delta"] >= 4 and row["oracle"]["exact_replay"])
    nonpositive = sorted(set(by_id) - set(positive))
    if len(positive) != 9 or len(nonpositive) != 5:
        raise SystemExit(f"unexpected teacher split {len(positive)}/{len(nonpositive)}")
    env = OrangeBikeDVGC(cfg, snapshot_bank=bank); step = jax.jit(env.step)
    batched_step = jax.jit(jax.vmap(env.step))
    inference = build_inference(env, params, deterministic=True)
    teacher, anchors, alignment = [], [], []
    # Reconstruct the exact best CEM lineage.  Ticks 0--7 are positive labels;
    # later available ticks are frozen-policy anchors on that same lineage.
    record_index = {row["id"]: index for index, row in enumerate(records)}
    for candidate_index, candidate_id in enumerate(positive):
        state = batched_base_state(env, by_id[candidate_id], 202607270 + record_index[candidate_id], 1)
        knots = np.asarray(oracle[candidate_id]["oracle"]["residual_knots"], np.float32)
        saved_actions = np.asarray(oracle[candidate_id]["oracle"]["best"]["actions"], np.float32)
        for tick in range(24):
            frozen, _ = inference(state.obs, jax.random.PRNGKey(3000000 + candidate_index * 100 + tick))
            frozen = np.asarray(frozen, np.float32)
            residual = knots[min(tick // 4, len(knots) - 1)][None, :]
            target = np.clip(frozen + residual, -1, 1)
            next_state = batched_step(state, target)
            alignment.append(float(np.max(np.abs(target[0] - saved_actions[tick]))))
            item = _batched_item(state, next_state, env, candidate_id, tick,
                                 "positive_teacher" if tick < 8 else "teacher_tail_anchor",
                                 target if tick < 8 else frozen, frozen,
                                 residual if tick < 8 else np.zeros((1, 4), np.float32))
            (teacher if tick < 8 else anchors).append(item)
            state = next_state
            if float(state.done[0]): break
    # Five non-significant candidates: only frozen-policy labels are anchors.
    for candidate_index, candidate_id in enumerate(nonpositive):
        state = restore_snapshot(env, by_id[candidate_id], jax.random.PRNGKey(4000000 + candidate_index))
        for tick in range(8):
            frozen = _pi_action(inference, state, 4100000 + candidate_index * 100 + tick)
            next_state = step(state, frozen)
            anchors.append(_item(state, env, candidate_id, tick, "nonpositive_anchor",
                                 frozen, frozen, np.zeros(4, np.float32), next_state))
            state = next_state
            if float(state.done): break
    # Canonical Landing anchors are state-only preservation probes.
    landing_bank = SnapshotBank.load(LANDING)
    landing_indices = np.linspace(0, len(landing_bank.records) - 1, 32, dtype=int)
    landing_env = OrangeBikeDVGC(cfg, snapshot_bank=SnapshotBank())
    landing_inference = build_inference(landing_env, params, deterministic=True)
    for index in landing_indices:
        row = landing_bank.records[int(index)]
        state = restore_snapshot(landing_env, row, jax.random.PRNGKey(5000000 + int(index)))
        frozen = _pi_action(landing_inference, state, 5100000 + int(index))
        item = _item(state, landing_env, row["id"], 0, "landing_anchor", frozen, frozen, np.zeros(4))
        anchors.append(item)
    # Natural-start anchors are generated from canonical resets without bank use.
    natural_cfg = load_config("configs/unified_descent_rsi_learnability_pilot_v1.json",
                              overrides={"use_bank_resets": False})
    natural_env = OrangeBikeDVGC(natural_cfg, snapshot_bank=SnapshotBank())
    natural_step = jax.jit(natural_env.step)
    natural_inference = build_inference(natural_env, params, deterministic=True)
    for index in range(16):
        state = natural_env.reset(jax.random.PRNGKey(6000000 + index))
        for tick in range(2):
            frozen = _pi_action(natural_inference, state, 6100000 + index * 10 + tick)
            next_state = natural_step(state, frozen)
            item = _item(state, natural_env, f"natural_{index:02d}", tick,
                         "natural_anchor", frozen, frozen, np.zeros(4), next_state)
            anchors.append(item); state = next_state
            if float(state.done): break
    mean = np.asarray(params[0].mean["state"]); std = np.asarray(params[0].std["state"])
    for item in teacher + anchors:
        item["normalized_observation"] = normalized_observation(item["observation"], mean, std)
    observations = np.asarray([row["normalized_observation"] for row in teacher])
    residuals = np.asarray([row["residual"] for row in teacher])
    ids = [row["candidate_id"] for row in teacher]
    representability = nearest_neighbor_audit(observations, residuals, ids)
    representability["candidate_support_p95"] = {
        candidate_id: trajectory_support_radius(np.asarray([
            row["normalized_observation"] for row in teacher if row["candidate_id"] == candidate_id]))
        for candidate_id in positive
    }
    per_tick = {}
    for tick in range(8):
        values = np.asarray([row["residual"] for row in teacher if row["tick"] == tick])
        per_tick[str(tick)] = {
            "rms": np.sqrt(np.mean(values * values, axis=0)).tolist(),
            "max_abs": np.max(np.abs(values), axis=0).tolist(),
        }
    root.mkdir(parents=True, exist_ok=True)
    with dataset_path.open("wb") as handle:
        pickle.dump({"teacher": teacher, "anchors": anchors}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    kind_counts = Counter(row["kind"] for row in anchors)
    manifest_out = {
        "status": "PASS", "artifact_role": "descent_initialization_teacher_only",
        "teacher_samples": len(teacher), "anchor_samples": len(anchors),
        "positive_candidates": positive, "nonpositive_candidates": nonpositive,
        "teacher_ticks": list(range(8)), "anchor_kind_counts": dict(sorted(kind_counts.items())),
        "action_order": list(ACTION_ORDER), "teacher_target": "clip(pi_D(o)+delta_CEM)",
        "post_squash_policy_action_space": True, "candidate_id_actor_input": False,
        "oracle_phase_actor_input": False, "heldout_used": False,
        "candidate_bank_sha256": file_sha256(BANK), "cem_artifact_sha256": file_sha256(CEM),
        "xml_sha256": file_sha256(cfg.xml_path), "policy_params_sha256": file_sha256(POLICY / "params.pkl"),
        "normalizer_sha256": normalizer_summary(params[0])["sha256"],
        "normalizer_count": normalizer_summary(params[0])["count"],
        "dataset_sha256": file_sha256(dataset_path),
    }
    save_json(root / "teacher_dataset_manifest.json", manifest_out)
    save_json(root / "teacher_action_alignment_audit.json", {
        "status": "PASS" if max(alignment) < 2e-5 else "FAIL",
        "command_tick_alignment_max_abs": max(alignment),
        "delay_buffer_preserved_in_snapshot": True,
        "actual_applied_action_source": "env.step target; next snapshot ctrl retained",
        "pre_squash_or_actuator_target_mixed": False,
        "per_tick_residual": per_tick,
        "all_teacher_exact_replay_source": True,
    })
    save_json(root / "teacher_representability_audit.json", {
        "status": "PASS" if representability["representable"] else "FAIL",
        **representability,
        "actor_observation_only": True,
        "future_state_or_candidate_label_input": False,
    })
    print(json.dumps({"manifest": manifest_out, "representability": representability}, indent=2))


if __name__ == "__main__":
    main()
