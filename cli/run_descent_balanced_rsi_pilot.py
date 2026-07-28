"""Run one conservative 3,200-step balanced Descent RSI block."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pickle
import subprocess
import time
from collections import Counter
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from cli.freeze_descent_predecessor_assets import verify_frozen_assets
from cli.run_backward_descent_nominal_pilot import C_L, EXPECTED, PI_D, PI_L, _load_record, _restore
from cli.run_backward_descent_rsi_pilot import certify_policy
from cli.runtime_gate import source_fingerprint
from dvgc.backward_search import make_descent_landing_rollout
from dvgc.bank import SnapshotBank
from dvgc.config import ACTION_MAPPING_VERSION, file_sha256, load_config, save_config
from dvgc.descent_balanced import iterative_balanced_weights, marginal_masses
from dvgc.descent_supervised import build_actor_tools, train_supervised
from dvgc.env import OrangeBikeDVGC
from dvgc.policy import load_bundle, save_bundle
from dvgc.runtime import (
    build_inference, make_ppo_train_fn, ppo_effective_timesteps, save_json,
    validate_ppo_batch_layout,
)


ROOT = Path("runs/descent_diverse_p1_predecessor_recovery_v1")
DATASET = ROOT / "descent_recovery_teacher_dataset_v1_balanced.pkl"
PILOT = ROOT / "descent_balanced_rsi_pilot_v1"
PRIOR = Path("runs/backward_recovery_tube_fast_track_v1/descent_cem3_tier2/descent_cem_pilot_report.json")
BLOCK_STEPS = 3200


def _params_hash(params):
    digest = hashlib.sha256()
    for leaf in jax.tree.leaves(params):
        value = np.asarray(leaf); digest.update(str(value.shape).encode()); digest.update(value.tobytes())
    return digest.hexdigest()


def _load_node_record(node, harvested):
    physical = node["physical_state"]; artifact = Path(physical["source_artifact"])
    if artifact == ROOT / "trajectory_harvested_snapshots.pkl":
        return copy.deepcopy(harvested[int(physical["source_index"])]["snapshot_v4"])
    return copy.deepcopy(_load_record(physical))


def _strip_training_labels(record):
    for name in ("final", "chain", "tube_version", "policy_version", "estimator_version",
                 "certified_safe", "tube_metrics_eligible", "safe_claim_allowed"):
        record.pop(name, None)
    record.update({"artifact_role": "proposal_support_bank", "training_eligible": True})
    return record


def build_training_records(balanced, p0_frontier, c_l_safe, harvested):
    rows = []
    balance_weights = iterative_balanced_weights(balanced, iterations=500)
    for node, weight in zip(balanced, balance_weights, strict=True):
        record = _strip_training_labels(_load_node_record(node, harvested))
        record.update({"reset_source": "flight_curriculum", "reset_weight": .60 * float(weight),
                       "reset_parent_id": node["node_id"], "backward_tube_label": "P1",
                       "bootstrap_group": "balanced_P1", "descent_layer": node["region"]})
        rows.append(record)
    candidates = sorted({node["candidate_id"] for node in p0_frontier})
    for candidate in candidates:
        group = [node for node in p0_frontier if node["candidate_id"] == candidate]
        for node in group:
            record = _strip_training_labels(_load_node_record(node, harvested))
            record.update({"reset_source": "flight_curriculum", "reset_weight": .20 / len(candidates) / len(group),
                           "reset_parent_id": node["node_id"], "backward_tube_label": "P0",
                           "bootstrap_group": "non_dominant_P0_frontier", "descent_layer": node["region"]})
            rows.append(record)
    for record in c_l_safe:
        item = _strip_training_labels(copy.deepcopy(record))
        item.update({"reset_source": "canonical_entry_rehearsal", "reset_weight": .20 / len(c_l_safe),
                     "reset_parent_id": record["id"], "backward_tube_label": "C_L_interface",
                     "bootstrap_group": "downstream_C_L_retention"})
        rows.append(item)
    if not np.isclose(sum(float(row["reset_weight"]) for row in rows), 1., atol=1e-8):
        raise ValueError("reset weights do not sum to one")
    return rows, {
        "balanced_P1_marginals": {field: marginal_masses(balanced, balance_weights, field)
                                  for field in ("candidate_id", "layer", "region")},
        "group_mass": {name: sum(row["reset_weight"] for row in rows if row["bootstrap_group"] == name)
                       for name in ("balanced_P1", "non_dominant_P0_frontier", "downstream_C_L_retention")},
    }


def _cross_action_audit(original_tool, current_tool, original_policy, current_policy, observations, targets=None):
    obs = jnp.asarray(observations); before = np.asarray(original_tool(original_policy, obs)); after = np.asarray(current_tool(current_policy, obs))
    delta = after - before; result = {"delta_rms": float(np.sqrt(np.mean(delta * delta))),
        "delta_max": float(np.max(np.abs(delta))), "saturation_fraction": float(np.mean(np.abs(after) >= .95))}
    if targets is not None:
        error = after - np.asarray(targets); result["imitation_rms"] = float(np.sqrt(np.mean(error * error)))
    return result


def _cert_summary(cert, fixed_ids):
    by_id = {row["node_id"]: row for row in cert["rows"]}
    return {"P0": sum(by_id[node]["P0"]["pass"] for node in fixed_ids),
            "P1": sum(by_id[node]["P1"]["pass"] for node in fixed_ids),
            "P0_ids": sorted(node for node in fixed_ids if by_id[node]["P0"]["pass"]),
            "P1_ids": sorted(node for node in fixed_ids if by_id[node]["P1"]["pass"])}


def _landing_retention(env, lparams, records):
    rollout = make_descent_landing_rollout(env, lparams, lparams, horizon=200, residual_ticks=8)
    total = success = 0
    for start in range(0, len(records), 32):
        group = records[start:start+32]
        states = [_restore(env, record, jax.random.PRNGKey(70_000_000 + start + i)) for i, record in enumerate(group)]
        batched = jax.tree.map(lambda *values: jnp.stack(values), *states)
        raw = jax.device_get(rollout(batched, jnp.zeros((len(group), 2, 4), jnp.float32), jax.random.PRNGKey(0)))
        total += len(group); success += int(np.sum(np.asarray(raw["final_recovery"])))
    return {"states": total, "final_recovery": success, "rate": success / total,
            "gate": success / total >= .80}


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--run", default=str(PILOT))
    args = parser.parse_args(); root = Path(args.run)
    if root.exists(): raise SystemExit(f"refusing overwrite {root}")
    valid, failed = verify_frozen_assets(ROOT)
    if not valid: raise SystemExit(f"frozen asset identity failure: {failed}")
    launch = json.loads((ROOT / "balanced_p1_launch_freeze_manifest.json").read_text())
    if launch["status"] != "PASS" or launch["PPO_authorization"] is not True: raise SystemExit("balanced launch not authorized")
    gate = json.loads(Path("docs/RUNTIME_GATE.json").read_text()); fingerprint = source_fingerprint(Path.cwd())
    if gate.get("status") != "PASS" or gate.get("source_fingerprint") != fingerprint: raise SystemExit("runtime gate stale")
    if not DATASET.exists() or file_sha256(DATASET) != json.loads((ROOT / "descent_recovery_teacher_dataset_v1_balanced_audit.json").read_text())["dataset_sha256"]:
        raise SystemExit("balanced teacher dataset identity failure")
    dparams, policy_cfg, dmanifest = load_bundle(PI_D, verify_files=True); lparams, _, _ = load_bundle(PI_L, verify_files=True)
    cfg = load_config("configs/backward_descent_rsi_pilot_v1.json")
    if file_sha256(cfg.xml_path) != EXPECTED["xml"] or cfg.action_mapping_version != ACTION_MAPPING_VERSION: raise SystemExit("runtime model mismatch")
    balanced = json.loads((ROOT / "balanced_p1_launch_subset_v1_frozen.json").read_text())["nodes"]
    full_p1 = json.loads((ROOT / "full_p1_bank_v1.json").read_text())["nodes"]
    old_all = json.loads(PRIOR.read_text())["nodes"]
    new_nodes = json.loads((ROOT / "source_a_certification_results.json").read_text())["nodes"]
    all_nodes = old_all + new_nodes
    dominant = Counter(node["candidate_id"] for node in full_p1).most_common(1)[0][0]
    p0_frontier = [node for node in all_nodes if node["p0"] and not node["p1"] and node["candidate_id"] != dominant]
    harvested = pickle.loads((ROOT / "trajectory_harvested_snapshots.pkl").read_bytes())
    entry = SnapshotBank.load(C_L); c_l_safe = [row for row in entry.records if row["final"]["label"] == "safe"]
    records, sampling = build_training_records(balanced, p0_frontier, c_l_safe, harvested)
    root.mkdir(parents=True); save_config(cfg, root / "effective_config.json")
    bank = SnapshotBank(records, {"artifact_role": "proposal_support_bank", "formal_tube_or_jel": False,
        "reset_source_protocol": "balanced_P1_60__non_dominant_P0_20__C_L_20",
        "balanced_launch_sha256": file_sha256(ROOT / "balanced_p1_launch_subset_v1_frozen.json"),
        "teacher_dataset_sha256": file_sha256(DATASET)})
    bank.save(root / "training_bank.pkl"); save_json(root / "sampling_preflight.json", sampling | {
        "records": len(records), "weight_sum": sum(row["reset_weight"] for row in records),
        "bank_sha256": file_sha256(root / "training_bank.pkl")})
    train_env = OrangeBikeDVGC(cfg, snapshot_bank=bank, cert_bank=entry)
    eval_cfg = load_config("configs/backward_descent_rsi_pilot_v1.json", {"use_bank_resets": False, "expert_chain_termination": False})
    eval_env = OrangeBikeDVGC(eval_cfg, snapshot_bank=SnapshotBank(), cert_bank=entry)
    # Model/reset/step/snapshot/deterministic-inference gates specific to this bank.
    reset = train_env.reset(jax.random.PRNGKey(0)); infer = build_inference(train_env, dparams, deterministic=True)
    a1, _ = infer(reset.obs, jax.random.PRNGKey(1)); a2, _ = infer(reset.obs, jax.random.PRNGKey(1))
    next1 = jax.jit(train_env.step)(reset, a1); next2 = jax.jit(train_env.step)(reset, a2)
    preflight = {"model_load": True, "reset_step_finite": bool(np.isfinite(np.asarray(next1.data.qpos)).all()),
        "deterministic_inference": bool(np.array_equal(np.asarray(a1), np.asarray(a2))),
        "deterministic_step": bool(np.array_equal(np.asarray(next1.data.qpos), np.asarray(next2.data.qpos))),
        "snapshot_restore": bool(np.isfinite(np.asarray(_restore(eval_env, _load_node_record(balanced[0], harvested), jax.random.PRNGKey(2)).data.qpos)).all()),
        "short_PPO_runtime_gate": "covered_by_current_docs_RUNTIME_GATE"}
    if not all(value is True or isinstance(value, str) for value in preflight.values()): raise SystemExit(f"training preflight failed: {preflight}")
    save_json(root / "training_preflight.json", preflight)
    dataset = pickle.loads(DATASET.read_bytes())
    teacher_obs = np.asarray([row["observation"] for row in dataset["teacher"]]); teacher_y = np.asarray([row["target_action"] for row in dataset["teacher"]])
    anchor_obs = np.asarray([row["observation"] for row in dataset["anchors"]]); anchor_y = np.asarray([row["target_action"] for row in dataset["anchors"]])
    _, actor_tool, _ = build_actor_tools(eval_env, dparams)
    supervised = []
    for steps in (25, 50, 100):
        policy, history = train_supervised(base_policy=dparams[1], actor_action=actor_tool,
            teacher_observation=teacher_obs, teacher_target=teacher_y,
            anchor_observation=anchor_obs, anchor_target=anchor_y,
            learning_rate=3e-5, steps=steps, mode="head")
        anchor_audit = _cross_action_audit(actor_tool, actor_tool, dparams[1], policy, anchor_obs)
        teacher_audit = _cross_action_audit(actor_tool, actor_tool, dparams[1], policy, teacher_obs, teacher_y)
        supervised.append({"steps": steps, "policy": policy, "history": history,
                           "anchor": anchor_audit, "teacher": teacher_audit,
                           "anchor_gate": anchor_audit["delta_rms"] <= .02 and anchor_audit["delta_max"] <= .05})
    eligible = [row for row in supervised if row["anchor_gate"]]
    if not eligible: raise SystemExit("behavior anchor hard gate rejected all preregistered head fits")
    selected = min(eligible, key=lambda row: (row["teacher"]["imitation_rms"], row["steps"]))
    prefit_params = (dparams[0], selected["policy"], dparams[2])
    save_json(root / "behavior_anchor_selection.json", {"selection_without_physical_results": True,
        "selected_steps": selected["steps"], "candidates": [{k: row[k] for k in ("steps", "history", "anchor", "teacher", "anchor_gate")} for row in supervised]})
    save_bundle(root / "checkpoint_prefit", params=prefit_params, config=cfg, xml_path=cfg.xml_path,
                candidate_bank=root / "training_bank.pkl", downstream_bank=C_L,
                policy_version="descent-balanced-rsi-prefit", extra={"artifact_role": "bounded_phase_local_rsi_pilot", "effective_steps": 0})
    loader = lambda node: _load_node_record(node, harvested)
    baseline = certify_policy(eval_env, dparams, lparams, all_nodes, 71_000_000, record_loader=loader)
    prefit = certify_policy(eval_env, prefit_params, lparams, all_nodes, 72_000_000, record_loader=loader)
    fixed_ids = [row["node_id"] for row in full_p1]; balanced_ids = [row["node_id"] for row in balanced]
    base_fixed, prefit_fixed = _cert_summary(baseline, fixed_ids), _cert_summary(prefit, fixed_ids)
    base_balanced, prefit_balanced = _cert_summary(baseline, balanced_ids), _cert_summary(prefit, balanced_ids)
    save_json(root / "prefit_physical_gate.json", {"baseline_fixed": base_fixed, "prefit_fixed": prefit_fixed,
        "baseline_balanced": base_balanced, "prefit_balanced": prefit_balanced})
    if not set(base_balanced["P1_ids"]) <= set(prefit_balanced["P1_ids"]):
        raise SystemExit("behavior prefit reduced balanced P1 support")
    num_envs, batch_size, num_minibatches, num_evals = 50, 25, 2, 3
    validate_ppo_batch_layout(num_envs=num_envs, batch_size=batch_size, num_minibatches=num_minibatches)
    effective = ppo_effective_timesteps(BLOCK_STEPS, unroll_length=32, batch_size=batch_size,
                                        num_minibatches=num_minibatches, num_evals=num_evals)
    if effective != BLOCK_STEPS: raise SystemExit(f"unexpected effective budget {effective}")
    save_json(root / "pilot_config.json", {"seed": 0, "effective_steps": BLOCK_STEPS,
        "reset_masses": [0.60, 0.20, 0.20], "learning_rate": 3e-5, "teacher_prefit_steps": selected["steps"],
        "post_PPO_teacher_rehearsal_steps": 25, "initial_policy": EXPECTED["pi_D"],
        "rejected_6400_checkpoint_used": False, "heldout_used": False, "delay": False})
    progress = []
    def progress_fn(step, metrics):
        progress.append({"effective_steps": int(step), **{key: float(value) for key, value in metrics.items() if np.asarray(value).shape == ()}})
        save_json(root / "training_progress.json", {"status": "running", "progress": progress})
    train_fn = make_ppo_train_fn(timesteps=BLOCK_STEPS, episode_length=64, num_envs=num_envs,
        num_eval_envs=16, num_evals=num_evals, seed=0, learning_rate=3e-5,
        entropy_cost=float(dmanifest["ppo_hyperparameters"]["entropy_cost"]), reward_scaling=.1,
        checkpoint_dir=root / "orbax", unroll_length=32, batch_size=batch_size,
        num_minibatches=num_minibatches, num_updates_per_batch=2, discounting=.995,
        gae_lambda=.97, clipping_epsilon=.10, max_grad_norm=.75, restore_params=prefit_params, full_reset=True)
    started = time.time()
    try: _, raw_params, final_metrics = train_fn(environment=train_env, progress_fn=progress_fn, eval_env=train_env)
    except BaseException as exc:
        save_json(root / "training_integrity_report.json", {"status": "FAIL", "error": str(exc), "error_type": type(exc).__name__}); raise
    # Fixed post-block teacher rehearsal; no physical result is consulted.
    _, current_actor, _ = build_actor_tools(eval_env, raw_params)
    rehearsed_policy, rehearsal_history = train_supervised(base_policy=raw_params[1], actor_action=current_actor,
        teacher_observation=teacher_obs, teacher_target=teacher_y, anchor_observation=anchor_obs,
        anchor_target=anchor_y, learning_rate=3e-5, steps=25, mode="head")
    final_params = (raw_params[0], rehearsed_policy, raw_params[2])
    _, final_actor, _ = build_actor_tools(eval_env, final_params)
    anchor_drift = _cross_action_audit(actor_tool, final_actor, dparams[1], final_params[1], anchor_obs)
    teacher_drift = _cross_action_audit(actor_tool, final_actor, dparams[1], final_params[1], teacher_obs, teacher_y)
    save_bundle(root / "checkpoint_3200", params=final_params, config=cfg, xml_path=cfg.xml_path,
                candidate_bank=root / "training_bank.pkl", downstream_bank=C_L,
                policy_version="descent-balanced-rsi-3200", extra={"artifact_role": "bounded_phase_local_rsi_pilot",
                "effective_steps": BLOCK_STEPS, "teacher_rehearsal_steps": 25})
    final = certify_policy(eval_env, final_params, lparams, all_nodes, 73_000_000, record_loader=loader)
    final_fixed, final_balanced = _cert_summary(final, fixed_ids), _cert_summary(final, balanced_ids)
    retention = _landing_retention(eval_env, lparams, c_l_safe)
    nonfinite = any(not np.isfinite(value) for row in progress for value in row.values() if isinstance(value, float))
    checks = {
        "all_balanced_P1_retained": final_balanced["P1"] == len(balanced_ids),
        "full_P1_pointwise_final": final_fixed["P0"] == len(fixed_ids),
        "baseline_P1_not_reduced": set(base_fixed["P1_ids"]) <= set(final_fixed["P1_ids"]),
        "candidate_coverage_not_reduced": len({row["candidate_id"] for row in full_p1 if row["node_id"] in final_fixed["P1_ids"]}) >= len({row["candidate_id"] for row in full_p1 if row["node_id"] in base_fixed["P1_ids"]}),
        "layer_coverage_not_reduced": len({row["layer"] for row in full_p1 if row["node_id"] in final_fixed["P1_ids"]}) >= len({row["layer"] for row in full_p1 if row["node_id"] in base_fixed["P1_ids"]}),
        "anchor_drift": anchor_drift["delta_rms"] <= .02 and anchor_drift["delta_max"] <= .05,
        "landing_retention": retention["gate"], "finite": not nonfinite,
        "measurable_improvement": final["P1"] > baseline["P1"] or final["P0"] > baseline["P0"],
    }
    accepted = all(checks.values())
    integrity = {"status": "PASS" if not nonfinite else "FAIL", "effective_steps": BLOCK_STEPS,
        "nonfinite": nonfinite, "oom": False, "timeout": False, "elapsed_seconds": time.time()-started,
        "runtime_fingerprint": fingerprint, "frozen_assets_unchanged": verify_frozen_assets(ROOT)[0]}
    save_json(root / "training_integrity_report.json", integrity)
    report = {"status": "ACCEPT" if accepted else "REJECT", "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "baseline": {"all": {"P0": baseline["P0"], "P1": baseline["P1"]}, "fixed": base_fixed, "balanced": base_balanced},
        "prefit": {"all": {"P0": prefit["P0"], "P1": prefit["P1"]}, "fixed": prefit_fixed, "balanced": prefit_balanced},
        "final": {"all": {"P0": final["P0"], "P1": final["P1"]}, "fixed": final_fixed, "balanced": final_balanced},
        "anchor_drift": anchor_drift, "teacher_drift": teacher_drift,
        "rehearsal_history": rehearsal_history, "landing_retention": retention,
        "acceptance_checks": checks, "integrity": integrity, "final_metrics": final_metrics,
        "PPO_authorization": "block_2" if accepted else False, "heldout_used": False, "delay": False,
        "formal_tube_or_jel": False}
    save_json(root / "DESCENT_BALANCED_RSI_PILOT_BLOCK1_REPORT.json", report)
    save_json(root / "completed.json", {"status": report["status"], "effective_steps": BLOCK_STEPS,
        "block_2_authorized": accepted})
    print(json.dumps({"status": report["status"], "baseline": report["baseline"], "final": report["final"],
                      "anchor_drift": anchor_drift, "retention": retention, "checks": checks}, indent=2))


if __name__ == "__main__": main()
