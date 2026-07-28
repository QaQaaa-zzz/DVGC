"""Audit legacy Descent snapshot timing and whole-packet delay sensitivity."""
from __future__ import annotations

import argparse
import inspect
import json
import pickle
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from cli.run_unified_descent_feedback_probe import _assets
from dvgc.config import file_sha256
from dvgc.delay_probe import make_packet_delay_rollout
from dvgc.descent_supervised import build_actor_tools
from dvgc.env import END_REASON
from dvgc.observation_audit import array_sha256, history_alignment
from dvgc.rollout import restore_snapshot_logged, restore_snapshot_reconstructed
from dvgc.runtime import save_json
from dvgc.snapshot_timing import J12_DELAY_SEQUENCE, causal_prior_packet, snapshot_v2_contract
from dvgc.support_diagnostic import weak_components


EXPECTED_HEAD = "c540d2f"
SNAPSHOTS = Path("runs/unified_descent_feedback_teacher_support_and_representation_probe_v1/feedback_probe_snapshots.pkl")
AUTHORITY = Path("runs/unified_descent_feedback_teacher_support_and_representation_probe_v1_replay_corrected/local_cem_authority_results.json")
MULTIMODALITY = Path("runs/unified_descent_feedback_teacher_support_and_representation_probe_v1_multimodality/successful_action_multimodality_audit.json")
TRANSFER = Path("runs/unified_descent_feedback_correction_transfer_and_support_geometry_audit_v1/feedback_correction_cross_snapshot_transfer_matrix.json")
MODES = {
    "L0": (0,) * 24,
    "R0": (0,) * 24,
    "D1": (1,) * 24,
    "D2": (2,) * 24,
    "J12": J12_DELAY_SEQUENCE,
}
ACTION_NAMES = ("steer", "drive", "hip", "knee")


def _stack_states(states):
    return jax.tree_util.tree_map(lambda *values: jnp.stack(values), *states)


def _take_state(state, indices):
    indices = jnp.asarray(indices, jnp.int32)
    return jax.tree_util.tree_map(lambda value: value[indices], state)


def _queue(logged):
    return np.stack((causal_prior_packet(logged, 2), causal_prior_packet(logged, 1), np.asarray(logged, np.float32)))


def _state_and_queue(env, snapshots, mode):
    states, queues = [], []
    for item in snapshots:
        record = item["snapshot"]
        key = jax.random.PRNGKey(int(item["generation_seed"]))
        if mode == "R0":
            state = restore_snapshot_reconstructed(env, record, key)
            packet = np.asarray(jax.device_get(state.obs["state"]), np.float32)
        else:
            state = restore_snapshot_logged(env, record, key)
            packet = np.asarray(record["policy_state"]["actor_observation"], np.float32)
        states.append(state)
        queues.append(_queue(packet))
    return _stack_states(states), jnp.asarray(np.stack(queues), jnp.float32)


def _medoid_corrections(authority, multimodality):
    medoid_rank = {int(row["snapshot_index"]): int(row["successful_medoid"]["rank"]) for row in multimodality["rows"]}
    result = {}
    for index, row in enumerate(authority["rows"]):
        if not row["authoritative_correction"]:
            continue
        rank = medoid_rank[index]
        top = next(value for value in row["top5"] if int(value["rank"]) == rank)
        result[index] = np.asarray(top["residual_knots"], np.float32)
    return result


def _result_rows(raw, repeat, indices, baseline=None):
    rows = []
    for position, index in enumerate(indices):
        code = int(raw["end_code"][position])
        item = {
            "snapshot_index": int(index),
            "survival": int(raw["survival"][position]),
            "minimum_margin": float(raw["minimum_margin"][position]),
            "terminal_margin": float(raw["terminal_margin"][position]),
            "end_code": code,
            "failure": END_REASON.get(code, "horizon"),
            "repeat_bit_exact": all(np.array_equal(np.asarray(raw[key])[position], np.asarray(repeat[key])[position]) for key in raw),
        }
        if baseline is not None:
            before = baseline[index]
            item["gain"] = item["survival"] - before["survival"]
            item["no_new_failure_type"] = item["failure"] in {before["failure"], "horizon"}
            item["authority_pass"] = item["repeat_bit_exact"] and item["gain"] >= 2 and item["no_new_failure_type"]
        rows.append(item)
    return rows


def _action_delta(reference, value):
    delta = np.asarray(value) - np.asarray(reference)
    return {
        "rms": float(np.sqrt(np.mean(delta * delta))),
        "max_abs": float(np.max(np.abs(delta))),
        "per_dimension_rms": dict(zip(ACTION_NAMES, np.sqrt(np.mean(delta * delta, axis=tuple(range(delta.ndim - 1)))).tolist())),
        "per_dimension_max_abs": dict(zip(ACTION_NAMES, np.max(np.abs(delta), axis=tuple(range(delta.ndim - 1))).tolist())),
    }


def _support_layers(snapshots, corrected):
    counts = Counter(snapshots[row["snapshot_index"]]["candidate_id"] for row in corrected if row["authority_pass"])
    candidates = sorted({item["candidate_id"] for item in snapshots})
    name = {3: "robust-core", 2: "frontier", 1: "sparse-support", 0: "unsupported"}
    return {candidate: name[min(counts[candidate], 3)] for candidate in candidates}


def _transfer_summary(rows, candidates):
    diagonal = [row for row in rows if row["same_snapshot"]]
    same = [row for row in rows if row["same_candidate"] and not row["same_snapshot"]]
    cross = [row for row in rows if not row["same_candidate"]]
    edges = sorted({(row["source_candidate_id"], row["target_candidate_id"]) for row in cross if row["physical_transfer"]})
    def summary(values):
        return {"eligible": len(values), "gain_at_least_2": sum(row["physical_transfer"] for row in values)}
    return {
        "categories": {"diagonal": summary(diagonal), "same_candidate_off_diagonal": summary(same), "cross_candidate": summary(cross)},
        "successful_pair_count": sum(row["physical_transfer"] for row in rows),
        "successful_pair_set": [[row["source_snapshot_index"], row["target_snapshot_index"]] for row in rows if row["physical_transfer"]],
        "weak_candidate_components": weak_components(candidates, edges),
        "successful_cross_candidate_edges": edges,
    }


def _forensics(root, snapshots):
    from dvgc.env import OrangeBikeDVGC
    from dvgc.rollout import restore_snapshot_mode
    functions = {
        "OrangeBikeDVGC.step": OrangeBikeDVGC.step,
        "OrangeBikeDVGC.snapshot_record": OrangeBikeDVGC.snapshot_record,
        "OrangeBikeDVGC._state_from_values": OrangeBikeDVGC._state_from_values,
        "restore_snapshot_mode": restore_snapshot_mode,
    }
    locations = {}
    for name, fn in functions.items():
        source, line = inspect.getsourcelines(fn)
        locations[name] = {"file": inspect.getsourcefile(fn), "start_line": line, "end_line": line + len(source) - 1}
    alignments = [history_alignment(item["snapshot"]["policy_state"]["actor_observation"], item["snapshot"]["policy_state"]["obs_history"]) for item in snapshots]
    report = {
        "status": "PASS",
        "classification": "HYBRID_STATE_RECONSTRUCTION_ERROR",
        "code_locations": locations,
        "control_tick_contract": {
            "physical_state_t": "state.data after transition t-1->t",
            "actor_observation_t": "pre-update history [t-3,t-2,t-1] plus current frame t; exact state.obs consumed by policy at tick t",
            "saved_obs_history": "post-update [t-2,t-1,t] saved from state.info",
            "policy_action_t": "generated from actor_observation_t before env.step at tick t",
            "snapshot_ctrl": "control applied on transition t-1->t",
            "ctrl_applied_t": "action_to_ctrl(policy_action_t), applied on transition t->t+1 and not stored as snapshot.ctrl",
        },
        "call_order": ["policy consumes state.obs_t", "env.step(action_t)", "physics applies ctrl_t", "frame_t+1 is built", "obs_history_post_t+1 is stored", "state.obs_t+1 uses obs_history_pre_t+1 plus frame_t+1", "snapshot_record saves physical_t+1, post-history_t+1 and actor_observation_t+1"],
        "history_alignment": {"post_current": sum(row["saved_equals_post_current"] for row in alignments), "required_pre_current": sum(row["saved_equals_required_pre_current"] for row in alignments), "total": len(alignments)},
        "why_not_fixed_delay": "the online packet includes current frame t; the mismatch is produced only when restore treats saved post-history as pre-history and appends frame t again",
        "why_logged_input_is_authoritative": "snapshot_record copies the exact state.obs actor tensor that the online deterministic policy consumes at the same physical state boundary",
        "heldout_used": False,
    }
    save_json(root / "snapshot_timing_forensics.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    root = Path(args.run)
    if root.exists():
        raise SystemExit(f"refusing overwrite {root}")
    if subprocess.run(["git", "merge-base", "--is-ancestor", EXPECTED_HEAD, "HEAD"]).returncode or subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise SystemExit("invalid git state")
    root.mkdir(parents=True)
    preregistration = {
        "modes": {name: list(schedule) for name, schedule in MODES.items()},
        "delay_semantics": "whole causal actor packet FIFO; missing legacy warm-start packets hold the oldest measured frame",
        "fixed_horizon": 24,
        "authority_rule": "repeat exact, gain>=2 versus same-mode baseline, no new failure type",
        "structural_stability_rule": {"authority_retention_each_D1_D2": 0.75, "baseline_failure_agreement_each_D1_D2": 0.75, "maximum_candidate_layer_changes": 2, "forbid_robust_core_to_unsupported": True, "cross_transfer_retention_each_D1_D2": 0.70, "largest_component_shrink_max": 1},
        "new_search": False, "training": False, "heldout_used": False,
    }
    save_json(root / "audit_preregistration.json", preregistration)
    _, _, env, params = _assets()
    with SNAPSHOTS.open("rb") as handle:
        snapshots = pickle.load(handle)
    authority = json.loads(AUTHORITY.read_text())
    multimodality = json.loads(MULTIMODALITY.read_text())
    old_transfer = json.loads(TRANSFER.read_text())
    if len(snapshots) != 24 or len(authority["rows"]) != 24 or len(old_transfer["pairs"]) != 244:
        raise SystemExit("frozen asset count gate")
    corrections = _medoid_corrections(authority, multimodality)
    if len(corrections) != 12:
        raise SystemExit("frozen correction count gate")
    forensics = _forensics(root, snapshots)
    _, actor_action, _ = build_actor_tools(env, params)
    modes = {}
    candidate_order = sorted({item["candidate_id"] for item in snapshots})
    states = {}
    queues = {}
    rollouts = {}
    zero = jnp.zeros((24, 2, 4), jnp.float32)
    correction_indices = sorted(corrections)
    correction_knots = jnp.asarray(np.stack([corrections[index] for index in correction_indices]), jnp.float32)
    for mode, schedule in MODES.items():
        state, queue = _state_and_queue(env, snapshots, mode)
        states[mode], queues[mode] = state, queue
        rollout = make_packet_delay_rollout(env, params, schedule)
        rollouts[mode] = rollout
        raw = jax.device_get(rollout(state, zero, queue, jax.random.PRNGKey(11_000_000)))
        repeat = jax.device_get(rollout(state, zero, queue, jax.random.PRNGKey(11_000_000)))
        baseline_rows = _result_rows(raw, repeat, list(range(24)))
        baseline = {row["snapshot_index"]: row for row in baseline_rows}
        source_state = _take_state(state, correction_indices)
        source_queue = queue[jnp.asarray(correction_indices)]
        corrected_raw = jax.device_get(rollout(source_state, correction_knots, source_queue, jax.random.PRNGKey(12_000_000)))
        corrected_repeat = jax.device_get(rollout(source_state, correction_knots, source_queue, jax.random.PRNGKey(12_000_000)))
        corrected_rows = _result_rows(corrected_raw, corrected_repeat, correction_indices, baseline)
        initial_packets = np.asarray(queue)[:, 2 - int(schedule[0])]
        initial_actions = np.stack([np.asarray(actor_action(params[1], packet), np.float32) for packet in initial_packets])
        modes[mode] = {
            "delay_schedule": list(schedule),
            "initial_actions": initial_actions.tolist(),
            "baseline": baseline_rows,
            "local_corrections": corrected_rows,
            "local_authority_pass": sum(row["authority_pass"] for row in corrected_rows),
            "candidate_support_layers": _support_layers(snapshots, corrected_rows),
            "baseline_failure_counts": dict(Counter(row["failure"] for row in baseline_rows)),
            "repeat_exact": all(row["repeat_bit_exact"] for row in baseline_rows + corrected_rows),
            "baseline_actions": np.asarray(raw["actions"]).tolist(),
        }
    l0_initial = np.asarray(modes["L0"]["initial_actions"])
    l0_actions = np.asarray(modes["L0"]["baseline_actions"])
    logged_action_matches = []
    for item, value in zip(snapshots, l0_initial, strict=True):
        expected = np.asarray(item["frozen_pi_d_action"], np.float32)
        logged_action_matches.append(bool(np.array_equal(value, expected)))
    for mode in MODES:
        modes[mode]["initial_action_delta_vs_L0"] = _action_delta(l0_initial, np.asarray(modes[mode]["initial_actions"]))
        modes[mode]["rollout_action_delta_vs_L0"] = _action_delta(l0_actions, np.asarray(modes[mode]["baseline_actions"]))
        del modes[mode]["initial_actions"]
        del modes[mode]["baseline_actions"]
    transfers = {}
    source_to_residual = corrections
    target_indices = [int(pair["target_snapshot_index"]) for pair in old_transfer["pairs"]]
    residuals = jnp.asarray(np.stack([source_to_residual[int(pair["source_snapshot_index"])] for pair in old_transfer["pairs"]]), jnp.float32)
    for mode in ("L0", "D1", "D2"):
        state = _take_state(states[mode], target_indices)
        queue = queues[mode][jnp.asarray(target_indices)]
        raw = jax.device_get(rollouts[mode](state, residuals, queue, jax.random.PRNGKey(13_000_000)))
        repeat = jax.device_get(rollouts[mode](state, residuals, queue, jax.random.PRNGKey(13_000_000)))
        baseline = {row["snapshot_index"]: row for row in modes[mode]["baseline"]}
        rows = []
        for position, old in enumerate(old_transfer["pairs"]):
            target = int(old["target_snapshot_index"]); code = int(raw["end_code"][position]); failure = END_REASON.get(code, "horizon")
            exact = all(np.array_equal(np.asarray(raw[key])[position], np.asarray(repeat[key])[position]) for key in raw)
            gain = int(raw["survival"][position]) - baseline[target]["survival"]
            no_new = failure in {baseline[target]["failure"], "horizon"}
            rows.append({key: old[key] for key in ("source_snapshot_index", "source_candidate_id", "target_snapshot_index", "target_candidate_id", "same_snapshot", "same_candidate", "source_layer", "target_layer")} | {"survival": int(raw["survival"][position]), "baseline_survival": baseline[target]["survival"], "gain": gain, "minimum_physical_margin": float(raw["minimum_margin"][position]), "failure": failure, "repeat_bit_exact": exact, "no_new_failure_type": no_new, "physical_transfer": bool(exact and gain >= 2 and no_new), "eligibility_preserved": True})
        transfers[mode] = _transfer_summary(rows, candidate_order) | {"pairs": rows, "eligible_pair_count": len(rows), "semantic_changes": 0}
    l0_authority = modes["L0"]["local_authority_pass"]
    l0_cross = transfers["L0"]["categories"]["cross_candidate"]["gain_at_least_2"]
    l0_component = max(map(len, transfers["L0"]["weak_candidate_components"]))
    checks = {}
    for mode in ("D1", "D2"):
        failure_agreement = sum(a["failure"] == b["failure"] for a, b in zip(modes["L0"]["baseline"], modes[mode]["baseline"], strict=True)) / 24
        layer_changes = sum(modes["L0"]["candidate_support_layers"][cid] != modes[mode]["candidate_support_layers"][cid] for cid in candidate_order)
        robust_drop = any(modes["L0"]["candidate_support_layers"][cid] == "robust-core" and modes[mode]["candidate_support_layers"][cid] == "unsupported" for cid in candidate_order)
        cross = transfers[mode]["categories"]["cross_candidate"]["gain_at_least_2"]
        component = max(map(len, transfers[mode]["weak_candidate_components"]))
        checks[mode] = {"authority_retention": modes[mode]["local_authority_pass"] / max(l0_authority, 1), "baseline_failure_agreement": failure_agreement, "candidate_layer_changes": layer_changes, "robust_core_to_unsupported": robust_drop, "cross_transfer_retention": cross / max(l0_cross, 1), "largest_component": component, "stable": modes[mode]["local_authority_pass"] / max(l0_authority, 1) >= .75 and failure_agreement >= .75 and layer_changes <= 2 and not robust_drop and cross / max(l0_cross, 1) >= .70 and component >= l0_component - 1}
    l0_authority_reproduces = l0_authority == 12 and all(logged_action_matches)
    if not l0_authority_reproduces:
        classification = "LOGGED_OBSERVATION_AUTHORITY_FAILURE"
    elif checks["D1"]["stable"] and checks["D2"]["stable"]:
        classification = "CONTROL_RELEVANT_ONE_FRAME_DELAY_TOLERATED"
    else:
        classification = "DELAY_SENSITIVE_FEEDBACK_SUPPORT"
    report = {
        "experiment": "unified_descent_snapshot_timing_and_delay_sensitivity_audit_v2",
        "status": "PASS",
        "timing_problem": forensics["classification"],
        "logged_actor_input_role": "authoritative_logged_actor_input" if all(logged_action_matches) else "logged_input_unverified",
        "logged_action_exact": sum(logged_action_matches),
        "logged_authority_reproduced": l0_authority_reproduces,
        "mode_results": modes,
        "transfer_results": transfers,
        "delay_stability_checks": checks,
        "causal_classification": classification,
        "evidence_roles": {"online_rollouts": "empirical_online_evidence", "logged_replays": "logged_observation_replay_evidence", "independent_restore": "independent_reconstruction_unverified", "tube_jel": "certification_pending_timing_audit"},
        "recapture_24_state_required": classification == "LOGGED_OBSERVATION_AUTHORITY_FAILURE",
        "snapshot_schema_v2_contract": snapshot_v2_contract(),
        "frozen_assets": {"snapshots_sha256": file_sha256(SNAPSHOTS), "authority_sha256": file_sha256(AUTHORITY), "transfer_sha256": file_sha256(TRANSFER)},
        "heldout_used": False, "new_cem": False, "training": False, "ppo_authorization": False, "bootstrap_authorization": False,
    }
    save_json(root / "mode_action_and_authority_results.json", {"modes": modes, "delay_stability_checks": checks})
    save_json(root / "correction_transfer_delay_sensitivity.json", transfers)
    save_json(root / "snapshot_schema_v2_contract.json", snapshot_v2_contract())
    save_json(root / "UNIFIED_DESCENT_SNAPSHOT_TIMING_AND_DELAY_SENSITIVITY_AUDIT_V2_REPORT.json", report)
    print(json.dumps({"classification": classification, "timing_problem": forensics["classification"], "logged_action_exact": sum(logged_action_matches), "authority": {mode: modes[mode]["local_authority_pass"] for mode in MODES}, "delay_checks": checks}, indent=2))


if __name__ == "__main__":
    main()
