"""Recapture the frozen 24-state Descent probe with v4 timing and re-audit delay."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pickle
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from cli.run_unified_descent_feedback_probe import BANK, POLICY, STUDENT, _assets
from cli.runtime_gate import source_fingerprint
from dvgc.bank import SnapshotBank
from dvgc.config import ACTION_MAPPING_VERSION, file_sha256
from dvgc.continuous import load_trajectory
from dvgc.delay_probe import active_prefix_repeat_comparison, make_packet_delay_rollout
from dvgc.descent_supervised import build_actor_tools
from dvgc.env import END_REASON
from dvgc.observation_audit import array_sha256
from dvgc.policy import load_bundle
from dvgc.ppo_integrity import normalizer_summary
from dvgc.rollout import restore_snapshot_mode
from dvgc.runtime import save_json
from dvgc.snapshot_timing import (
    J12_DELAY_SEQUENCE, SNAPSHOT_SCHEMA_NAME, snapshot_v4_contract,
    validate_snapshot_v4, validate_transfer_eligibility,
)
from dvgc.support_diagnostic import weak_components


EXPECTED_START = "0f87f06"
LEGACY = Path("runs/unified_descent_feedback_teacher_support_and_representation_probe_v1/feedback_probe_snapshots.pkl")
AUTHORITY = Path("runs/unified_descent_feedback_teacher_support_and_representation_probe_v1_replay_corrected/local_cem_authority_results.json")
MULTIMODALITY = Path("runs/unified_descent_feedback_teacher_support_and_representation_probe_v1_multimodality/successful_action_multimodality_audit.json")
TRANSFER = Path("runs/unified_descent_feedback_correction_transfer_and_support_geometry_audit_v1/feedback_correction_cross_snapshot_transfer_matrix.json")
CONFIG = Path("configs/unified_descent_rsi_learnability_pilot_v1.json")
MODES = {
    "L0": (0,) * 24,
    "D1": (1,) * 24,
    "D2": (2,) * 24,
    "J12": J12_DELAY_SEQUENCE,
}
ACTION_NAMES = ("steer", "drive", "hip", "knee")
BRIDGE_SEED = 12_330_000


def _json_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=float).encode()
    return hashlib.sha256(raw).hexdigest()


def _physical_hash(row: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in ("qpos", "qvel", "ctrl", "qacc_warmstart"):
        value = np.ascontiguousarray(row[name])
        digest.update(name.encode()); digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode()); digest.update(value.tobytes())
    return digest.hexdigest()


def _v4_physical_hash(row: dict[str, Any]) -> str:
    p = row["physical_state_t"]
    return _physical_hash({"qpos": p["qpos"], "qvel": p["qvel"],
                           "ctrl": p["ctrl_previous"],
                           "qacc_warmstart": p["qacc_warmstart"]})


def _semantic_hash_legacy(row: dict[str, Any]) -> str:
    ps = row["policy_state"]
    payload = {
        "physical": _physical_hash(row),
        "actor": array_sha256(ps["actor_observation"]),
        "last": array_sha256(ps["last_action"]),
        "phase": int(row["oracle_phase"]),
        "estimated_phase": int(ps["filter_phase"]),
        "phase_probs": array_sha256(ps["phase_probs"]),
        "contact_age": int(row["contact_age"]),
        "had_valid_landing": int(row["had_valid_landing"]),
    }
    return _json_hash(payload)


def _semantic_hash_v4(row: dict[str, Any]) -> str:
    est = row["estimator_state_pre_t"]
    payload = {
        "physical": _v4_physical_hash(row),
        "actor": array_sha256(row["actor_observation_t"]),
        "last": array_sha256(row["last_normalized_command_t"]),
        "phase": int(est["phase"]),
        "estimated_phase": int(est["estimated_phase"]),
        "phase_probs": array_sha256(est["phase_probs"]),
        "contact_age": int(est["contact_age"]),
        "had_valid_landing": int(est["had_valid_landing"]),
    }
    return _json_hash(payload)


def _approach_ticks(path: Path) -> int:
    report = path.parent / "report.json"
    return int(json.loads(report.read_text())["best"]["approach_ticks"])


def _natural_candidate_states(env, rows: dict[str, dict[str, Any]], wanted: set[str]):
    """Replay immutable natural lineages and recover the eight candidate states."""
    step = jax.jit(env.step)
    found = {}
    grouped: dict[Path, list[dict[str, Any]]] = {}
    for cid in sorted(wanted):
        row = rows[cid]
        if row.get("candidate_source") != "natural_continuous":
            raise RuntimeError(f"v4 recapture requires natural lineage: {cid}")
        grouped.setdefault(Path(row["source_trajectory_path"]), []).append(row)
    for path, targets in grouped.items():
        arrays, _ = load_trajectory(path)
        state = env.reset(jax.random.PRNGKey(BRIDGE_SEED))
        for _ in range(_approach_ticks(path)):
            state = step(state, jnp.asarray([0., 1., 0., 0.], jnp.float32))
        if not np.array_equal(np.asarray(state.data.qpos), np.asarray(arrays["qpos"][0])):
            raise RuntimeError(f"natural lineage initial mismatch: {path}")
        for index, action in enumerate(arrays["action"][1:], 1):
            state = step(state, jnp.asarray(np.clip(action, -1., 1.), jnp.float32))
            if not np.array_equal(np.asarray(state.data.qpos), np.asarray(arrays["qpos"][index])):
                raise RuntimeError(f"natural lineage divergence: {path}:{index}")
            for target in targets:
                cid = target["id"]
                if cid not in found and np.array_equal(np.asarray(state.data.qpos), np.asarray(target["qpos"])) and np.array_equal(np.asarray(state.data.qvel), np.asarray(target["qvel"])):
                    found[cid] = state
    missing = sorted(wanted - set(found))
    if missing:
        raise RuntimeError(f"natural candidate states not recovered: {missing}")
    return found


def _provenance(cfg, params) -> dict[str, str]:
    return {
        "xml_sha256": file_sha256(cfg.xml_path),
        "config_sha256": file_sha256(CONFIG),
        "action_mapping_version": ACTION_MAPPING_VERSION,
        "policy_params_sha256": file_sha256(POLICY / "params.pkl"),
        "policy_config_sha256": file_sha256(POLICY / "config.json"),
        "policy_manifest_sha256": file_sha256(POLICY / "manifest.json"),
        "normalizer_sha256": normalizer_summary(params[0])["sha256"],
        "source_fingerprint": source_fingerprint(Path.cwd()),
    }


def _validator(env, params, record, provenance, frozen_action):
    _, actor_action, _ = build_actor_tools(env, params)
    independent = jax.jit(lambda key: restore_snapshot_mode(
        env, record, key,
        observation_mode="timing_explicit_independent_reconstruction",
    ))
    return validate_snapshot_v4(
        record,
        expected_shapes={
            "qpos": (env.mj_model.nq,), "qvel": (env.mj_model.nv,),
            "act": (env.mj_model.na,), "ctrl_previous": (env.mj_model.nu,),
            "qacc_warmstart": (env.mj_model.nv,),
            "sensordata": (env.mj_model.nsensordata,),
        },
        expected_hashes=provenance,
        actor_action_fn=lambda obs: np.asarray(actor_action(params[1], jnp.asarray(obs))),
        ctrl_from_action_fn=lambda action: np.asarray(env._action_to_ctrl(
            jnp.asarray(action), jnp.asarray(record["physical_state_t"]["qpos"])[env._joint_qpos["knee_joint"]]
        )),
        current_frame_fn=lambda _: np.asarray(independent(jax.random.PRNGKey(0)).obs["state"]).reshape(env._actor_history_steps, env._actor_frame_dim)[-1],
    )


def _recapture(root: Path, cfg, bank, env, params, legacy):
    records = {row["id"]: row for row in bank.records}
    candidate_ids = {item["candidate_id"] for item in legacy}
    base = _natural_candidate_states(env, records, candidate_ids)
    with STUDENT.open("rb") as handle:
        student = pickle.load(handle)
    _, student_action, _ = build_actor_tools(env, (params[0], student, params[2]))
    _, frozen_action, _ = build_actor_tools(env, params)
    provenance = _provenance(cfg, params)
    by_candidate = {cid: sorted((item for item in legacy if item["candidate_id"] == cid), key=lambda x: x["tick"])
                    for cid in sorted(candidate_ids)}
    captured, comparisons = [], []
    step = jax.jit(env.step)
    for cid, items in by_candidate.items():
        state = base[cid]
        selected = {int(item["tick"]): item for item in items}
        for tick in range(max(selected) + 1):
            if tick in selected:
                old = selected[tick]
                action = np.asarray(frozen_action(params[1], state.obs["state"]), np.float32)
                v4 = env.snapshot_record_v4(state, "flight", jnp.asarray(action), provenance)
                validation = _validator(env, params, v4, provenance, action)
                independent = jax.jit(lambda key: restore_snapshot_mode(
                    env, v4, key,
                    observation_mode="timing_explicit_independent_reconstruction",
                ))(jax.random.PRNGKey(int(old["generation_seed"])))
                logged = jax.jit(lambda key: restore_snapshot_mode(
                    env, v4, key,
                    observation_mode="timing_explicit_logged_replay",
                ))(jax.random.PRNGKey(int(old["generation_seed"])))
                old_snap = old["snapshot"]
                checks = {
                    "validator": bool(validation["valid"]),
                    "physical_hash": _physical_hash(old_snap) == _v4_physical_hash(v4),
                    "semantic_hash": _semantic_hash_legacy(old_snap) == _semantic_hash_v4(v4),
                    "actor_observation": np.array_equal(np.asarray(old_snap["policy_state"]["actor_observation"]), np.asarray(v4["actor_observation_t"])),
                    "post_history": np.array_equal(np.asarray(old_snap["policy_state"]["obs_history"]), np.asarray(v4["obs_history_post_t"])),
                    "frozen_action": np.array_equal(np.asarray(old["frozen_pi_d_action"], np.float32), action) and np.array_equal(action, np.asarray(v4["policy_action_t"])),
                    "candidate_tick": old["candidate_id"] == cid and int(old["tick"]) == tick,
                    "independent_actor": np.array_equal(np.asarray(independent.obs["state"]), np.asarray(v4["actor_observation_t"])),
                    "logged_actor": np.array_equal(np.asarray(logged.obs["state"]), np.asarray(v4["actor_observation_t"])),
                    "fifo_real": int(v4["actor_packet_fifo_valid"]) == 3,
                }
                comparisons.append({
                    "index": len(captured), "candidate_id": cid, "tick": tick,
                    "checks": checks, "exact": all(checks.values()),
                    "validator_failed": validation["failed"],
                    "legacy_snapshot_hash": old["snapshot_hash"],
                    "legacy_semantic_hash": _semantic_hash_legacy(old_snap),
                    "v4_semantic_hash": _semantic_hash_v4(v4),
                })
                captured.append({**{k: old[k] for k in old if k != "snapshot"},
                                 "legacy_snapshot": old_snap, "snapshot_v4": v4})
            command = np.asarray(student_action(student, state.obs["state"]), np.float32)
            state = step(state, jnp.asarray(command))
    passed = sum(row["exact"] for row in comparisons)
    save_json(root / "v4_recapture_identity.json", {
        "status": "PASS" if passed == 24 else "FAIL",
        "classification": None if passed == 24 else "V4_RECAPTURE_IDENTITY_FAILURE",
        "exact": passed, "total": 24, "rows": comparisons,
        "heldout_used": False, "training": False, "new_cem": False,
    })
    if passed != 24:
        return None
    temporary = root / ".timing_explicit_snapshots.pkl.partial"
    with temporary.open("wb") as handle:
        pickle.dump(captured, handle, pickle.HIGHEST_PROTOCOL)
        handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, root / "timing_explicit_snapshots.pkl")
    save_json(root / "timing_explicit_snapshots_manifest.json", {
        "schema": SNAPSHOT_SCHEMA_NAME, "records": 24,
        "sha256": file_sha256(root / "timing_explicit_snapshots.pkl"),
        "candidate_count": 8, "per_candidate": dict(Counter(x["candidate_id"] for x in captured)),
        "heldout_used": False,
    })
    return captured


def _medoid_corrections(authority, multimodality):
    ranks = {int(row["snapshot_index"]): int(row["successful_medoid"]["rank"])
             for row in multimodality["rows"]}
    corrections = {}
    for index, row in enumerate(authority["rows"]):
        if row["authoritative_correction"]:
            top = next(x for x in row["top5"] if int(x["rank"]) == ranks[index])
            corrections[index] = np.asarray(top["residual_knots"], np.float32)
    return corrections


def _result(raw, repeat, snapshot_index, baseline=None):
    exact = active_prefix_repeat_comparison(raw, repeat)["exact"]
    code = int(np.asarray(raw["end_code"])[0])
    row = {
        "snapshot_index": int(snapshot_index), "survival": int(np.asarray(raw["survival"])[0]),
        "minimum_margin": float(np.asarray(raw["minimum_margin"])[0]),
        "terminal_margin": float(np.asarray(raw["terminal_margin"])[0]),
        "termination_tick": int(np.asarray(raw["termination_tick"])[0]),
        "end_code": code, "failure": END_REASON.get(code, "horizon"),
        "landing_entry": bool(np.asarray(raw["landing_entry"])[0]),
        "chain": bool(np.asarray(raw["chain"])[0]),
        "recovery_success": bool(np.asarray(raw["recovery_success"])[0]),
        "final_recovery": bool(np.asarray(raw["final_recovery"])[0]),
        "repeat_bit_exact": bool(exact),
    }
    if baseline is not None:
        row["gain"] = row["survival"] - baseline["survival"]
        row["no_new_failure_type"] = row["failure"] in {baseline["failure"], "horizon"}
        row["authority_pass"] = exact and row["gain"] >= 2 and row["no_new_failure_type"]
    return row


def _batched_state(env, record, seed):
    return jax.jit(jax.vmap(lambda key: restore_snapshot_mode(
        env, record, key,
        observation_mode="timing_explicit_independent_reconstruction",
    )))(jax.random.split(jax.random.PRNGKey(seed), 1))


def _action_delta(reference, value):
    delta = np.asarray(value) - np.asarray(reference)
    axes = tuple(range(delta.ndim - 1))
    return {"rms": float(np.sqrt(np.mean(delta * delta))),
            "max_abs": float(np.max(np.abs(delta))),
            "per_dimension_rms": dict(zip(ACTION_NAMES, np.sqrt(np.mean(delta * delta, axis=axes)).tolist())),
            "per_dimension_max_abs": dict(zip(ACTION_NAMES, np.max(np.abs(delta), axis=axes).tolist()))}


def _support_layers(captured, rows):
    count = Counter(captured[row["snapshot_index"]]["candidate_id"] for row in rows if row["authority_pass"])
    name = {0: "unsupported", 1: "sparse-support", 2: "frontier", 3: "robust-core"}
    return {cid: name[min(count[cid], 3)] for cid in sorted({x["candidate_id"] for x in captured})}


def _transfer_summary(rows, candidates):
    categories = {
        "diagonal": [r for r in rows if r["same_snapshot"]],
        "same_candidate_off_diagonal": [r for r in rows if r["same_candidate"] and not r["same_snapshot"]],
        "cross_candidate": [r for r in rows if not r["same_candidate"]],
    }
    edges = sorted({(r["source_candidate_id"], r["target_candidate_id"])
                    for r in categories["cross_candidate"] if r["physical_transfer"]})
    return {
        "categories": {k: {"eligible": len(v), "gain_at_least_2": sum(r["physical_transfer"] for r in v)} for k, v in categories.items()},
        "successful_pair_count": sum(r["physical_transfer"] for r in rows),
        "successful_pair_set": [[r["source_snapshot_index"], r["target_snapshot_index"]] for r in rows if r["physical_transfer"]],
        "successful_cross_candidate_edges": edges,
        "weak_candidate_components": weak_components(candidates, edges),
        "candidate_grouped": {cid: {"targets": sum(r["target_candidate_id"] == cid for r in rows),
                                    "success": sum(r["target_candidate_id"] == cid and r["physical_transfer"] for r in rows)} for cid in candidates},
    }


def _reaudit(root, env, params, captured, authority, multimodality, old_transfer):
    corrections = _medoid_corrections(authority, multimodality)
    if len(corrections) != 12 or len(old_transfer["pairs"]) != 244:
        raise RuntimeError("frozen correction/pair count gate")
    _, actor_action, _ = build_actor_tools(env, params)
    states = [_batched_state(env, x["snapshot_v4"], 20_000_000 + i) for i, x in enumerate(captured)]
    queues = [jnp.asarray(x["snapshot_v4"]["actor_packet_fifo_t"][None], jnp.float32) for x in captured]
    modes, raw_cache, rollout_fns = {}, {}, {}
    zero = jnp.zeros((1, 2, 4), jnp.float32)
    for mode, schedule in MODES.items():
        rollout = make_packet_delay_rollout(env, params, schedule)
        rollout_fns[mode] = rollout
        baseline_rows, correction_rows, first_actions = [], [], []
        raw_cache[mode] = {}
        for index, (state, queue) in enumerate(zip(states, queues, strict=True)):
            seed = 21_000_000 + index
            raw = jax.device_get(rollout(state, zero, queue, jax.random.PRNGKey(seed)))
            repeat = jax.device_get(rollout(state, zero, queue, jax.random.PRNGKey(seed)))
            raw_cache[mode][index] = raw
            baseline_rows.append(_result(raw, repeat, index))
            packet = np.asarray(queue)[0, 2 - int(schedule[0])]
            first_actions.append(np.asarray(actor_action(params[1], jnp.asarray(packet)), np.float32))
        baseline = {row["snapshot_index"]: row for row in baseline_rows}
        for index in sorted(corrections):
            knots = jnp.asarray(corrections[index][None], jnp.float32)
            seed = 22_000_000 + index
            raw = jax.device_get(rollout(states[index], knots, queues[index], jax.random.PRNGKey(seed)))
            repeat = jax.device_get(rollout(states[index], knots, queues[index], jax.random.PRNGKey(seed)))
            correction_rows.append(_result(raw, repeat, index, baseline[index]))
        modes[mode] = {
            "delay_schedule": list(schedule), "baseline": baseline_rows,
            "local_corrections": correction_rows,
            "local_authority_pass": sum(row["authority_pass"] for row in correction_rows),
            "candidate_support_layers": _support_layers(captured, correction_rows),
            "baseline_failure_counts": dict(Counter(row["failure"] for row in baseline_rows)),
            "repeat_exact": all(row["repeat_bit_exact"] for row in baseline_rows + correction_rows),
            "initial_actions": np.stack(first_actions),
        }
    l0_actions = modes["L0"]["initial_actions"]
    l0_expected = np.stack([np.asarray(x["frozen_pi_d_action"], np.float32) for x in captured])
    l0_action_exact = int(sum(np.array_equal(a, b) for a, b in zip(l0_actions, l0_expected, strict=True)))
    for mode in MODES:
        modes[mode]["initial_action_delta_vs_L0"] = _action_delta(l0_actions, modes[mode]["initial_actions"])
        del modes[mode]["initial_actions"]

    old_baseline_failure = {}
    for pair in old_transfer["pairs"]:
        target = int(pair["target_snapshot_index"])
        old_baseline_failure.setdefault(target, pair["failure"] if int(pair["gain"]) == 0 else None)
    # The old artifact stores corrected failure.  Target baseline semantics are
    # asserted by the frozen legacy L0 report where available, otherwise by the
    # freshly reproduced L0 baseline after 24/24 identity.
    old_v2 = json.loads(Path("runs/unified_descent_snapshot_timing_and_delay_sensitivity_audit_v2_retry1/UNIFIED_DESCENT_SNAPSHOT_TIMING_AND_DELAY_SENSITIVITY_AUDIT_V2_REPORT.json").read_text())
    old_l0 = {int(r["snapshot_index"]): r["failure"] for r in old_v2["mode_results"]["L0"]["baseline"]}
    expected_rows, actual_rows = [], []
    for pair in old_transfer["pairs"]:
        source, target = int(pair["source_snapshot_index"]), int(pair["target_snapshot_index"])
        base = {
            "source_snapshot_index": source, "target_snapshot_index": target,
            "correction_sha256": array_sha256(corrections[source]),
            "phase": int(captured[target]["legacy_snapshot"]["oracle_phase"]),
            "contact_mode": f"valid={int(captured[target]['legacy_snapshot']['had_valid_landing'])};age={int(captured[target]['legacy_snapshot']['contact_age'])}",
            "failure_precursor": old_l0[target],
            "delay_semantics": "v4_complete_packet_fifo_t_minus_2_to_t",
        }
        expected_rows.append(base | {"source_snapshot_hash": _semantic_hash_legacy(captured[source]["legacy_snapshot"]), "target_snapshot_hash": _semantic_hash_legacy(captured[target]["legacy_snapshot"])})
        est = captured[target]["snapshot_v4"]["estimator_state_pre_t"]
        actual_rows.append(base | {
            "source_snapshot_hash": _semantic_hash_v4(captured[source]["snapshot_v4"]),
            "target_snapshot_hash": _semantic_hash_v4(captured[target]["snapshot_v4"]),
            "phase": int(est["phase"]),
            "contact_mode": f"valid={int(est['had_valid_landing'])};age={int(est['contact_age'])}",
            "failure_precursor": modes["L0"]["baseline"][target]["failure"],
        })
    eligibility = validate_transfer_eligibility(
        expected_rows, actual_rows,
        expected_artifact_sha256=file_sha256(TRANSFER),
        actual_artifact_sha256=file_sha256(TRANSFER),
    )
    if not eligibility["valid"]:
        save_json(root / "transfer_eligibility_failure.json", eligibility)
        raise RuntimeError(f"transfer eligibility failed: {eligibility['failed']}")

    candidates = sorted({x["candidate_id"] for x in captured})
    transfers = {}
    for mode in ("L0", "D1", "D2", "J12"):
        baseline = {r["snapshot_index"]: r for r in modes[mode]["baseline"]}
        rows = []
        for position, old in enumerate(old_transfer["pairs"]):
            source, target = int(old["source_snapshot_index"]), int(old["target_snapshot_index"])
            knots = jnp.asarray(corrections[source][None], jnp.float32)
            seed = 23_000_000 + position
            raw = jax.device_get(rollout_fns[mode](states[target], knots, queues[target], jax.random.PRNGKey(seed)))
            repeat = jax.device_get(rollout_fns[mode](states[target], knots, queues[target], jax.random.PRNGKey(seed)))
            result = _result(raw, repeat, target, baseline[target])
            rows.append({k: old[k] for k in ("source_snapshot_index", "source_candidate_id", "target_snapshot_index", "target_candidate_id", "same_snapshot", "same_candidate", "source_layer", "target_layer")} | result | {"physical_transfer": result["authority_pass"]})
        transfers[mode] = _transfer_summary(rows, candidates) | {"pairs": rows, "eligibility": eligibility}
    l0_set = {tuple(x) for x in transfers["L0"]["successful_pair_set"]}
    for mode in MODES:
        current = {tuple(x) for x in transfers[mode]["successful_pair_set"]}
        union = l0_set | current
        transfers[mode]["success_set_vs_L0"] = {
            "intersection": len(l0_set & current), "lost": len(l0_set - current),
            "gained": len(current - l0_set), "jaccard": 1.0 if not union else len(l0_set & current) / len(union),
        }
    l0_authority = modes["L0"]["local_authority_pass"]
    l0_cross = transfers["L0"]["categories"]["cross_candidate"]["gain_at_least_2"]
    l0_component = max(map(len, transfers["L0"]["weak_candidate_components"]))
    stability = {}
    for mode in ("D1", "D2"):
        failure_agreement = sum(a["failure"] == b["failure"] for a, b in zip(modes["L0"]["baseline"], modes[mode]["baseline"], strict=True)) / 24
        changes = sum(modes["L0"]["candidate_support_layers"][c] != modes[mode]["candidate_support_layers"][c] for c in candidates)
        robust_drop = any(modes["L0"]["candidate_support_layers"][c] == "robust-core" and modes[mode]["candidate_support_layers"][c] == "unsupported" for c in candidates)
        cross = transfers[mode]["categories"]["cross_candidate"]["gain_at_least_2"]
        component = max(map(len, transfers[mode]["weak_candidate_components"]))
        values = {"authority_retention": modes[mode]["local_authority_pass"] / max(l0_authority, 1), "baseline_failure_agreement": failure_agreement, "candidate_layer_changes": changes, "robust_core_to_unsupported": robust_drop, "cross_transfer_retention": cross / max(l0_cross, 1), "largest_component": component}
        values["stable"] = values["authority_retention"] >= .75 and failure_agreement >= .75 and changes <= 2 and not robust_drop and values["cross_transfer_retention"] >= .70 and component >= l0_component - 1
        stability[mode] = values
    classification = "CONTROL_RELEVANT_ONE_FRAME_DELAY_TOLERATED" if stability["D1"]["stable"] and stability["D2"]["stable"] else "DELAY_SENSITIVE_FEEDBACK_SUPPORT"
    save_json(root / "mode_action_and_authority_results.json", {"L0_action_exact": l0_action_exact, "modes": modes, "stability": stability})
    save_json(root / "correction_transfer_delay_sensitivity.json", transfers)
    return {"classification": classification, "L0_action_exact": l0_action_exact,
            "modes": modes, "transfers": transfers, "stability": stability,
            "eligibility": eligibility}


def _call_site_audit(root: Path):
    output = subprocess.check_output(["rg", "-n", r"restore_snapshot(_logged|_reconstructed|_mode)?\(", "--glob", "*.py", "--glob", "!tests/**", "."], text=True)
    authority = {"cli/certify.py", "cli/audit.py", "cli/certify_descent_entries.py", "cli/certify_stable_descent_shard.py", "cli/certify_descent_construction_shard.py", "cli/certify_expert_provisional.py", "cli/runtime_gate.py", "cli/run_timing_explicit_snapshot_schema_v4_delay_reaudit.py"}
    rows = []
    for line in output.splitlines():
        path = line.split(":", 1)[0].removeprefix("./")
        rows.append({"call_site": line, "authority_sensitive": path in authority,
                     "actual_mode": "explicit schema-selected authority mode" if path in authority else "deprecated compatibility/non-authority legacy path"})
    save_json(root / "restore_call_site_audit.json", {"total": len(rows), "rows": rows})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    args = parser.parse_args(); root = Path(args.run)
    if root.exists():
        allowed = {"preregistration.json", "restore_call_site_audit.json"}
        unexpected = sorted(path.name for path in root.iterdir() if path.name not in allowed)
        if unexpected or not (root / "preregistration.json").is_file():
            raise SystemExit(f"refusing overwrite {root}: {unexpected}")
    if subprocess.run(["git", "merge-base", "--is-ancestor", EXPECTED_START, "HEAD"]).returncode:
        raise SystemExit("unexpected git history")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise SystemExit("worktree must be clean")
    root.mkdir(parents=True, exist_ok=True)
    cfg, bank, env, params = _assets()
    legacy = pickle.loads(LEGACY.read_bytes())
    authority = json.loads(AUTHORITY.read_text()); multimodality = json.loads(MULTIMODALITY.read_text())
    old_transfer = json.loads(TRANSFER.read_text())
    if len(legacy) != 24 or file_sha256(LEGACY) != "d151f9e142a8160e3ea832e0243267600e338b88a138e2f79ea3f8667252cdf3":
        raise SystemExit("legacy asset gate")
    save_json(root / "snapshot_schema_v4_contract.json", snapshot_v4_contract())
    _call_site_audit(root)
    captured = _recapture(root, cfg, bank, env, params, legacy)
    if captured is None:
        print(json.dumps({"classification": "V4_RECAPTURE_IDENTITY_FAILURE", "exact": json.loads((root / "v4_recapture_identity.json").read_text())["exact"]}))
        return
    results = _reaudit(root, env, params, captured, authority, multimodality, old_transfer)
    report = {
        "experiment": "timing_explicit_snapshot_schema_v4_and_delay_reaudit_v1",
        "status": "PASS", "schema": SNAPSHOT_SCHEMA_NAME,
        "recapture_identity": "24/24 bit-exact", "old_corrections_inherited": 12,
        "old_pairs_inherited": 244, **results,
        "frozen_assets": {"legacy_snapshots_sha256": file_sha256(LEGACY), "authority_sha256": file_sha256(AUTHORITY), "multimodality_sha256": file_sha256(MULTIMODALITY), "old_transfer_sha256": file_sha256(TRANSFER)},
        "heldout_used": False, "training": False, "new_cem": False,
        "ppo_authorization": False, "bootstrap_authorization": False,
    }
    save_json(root / "TIMING_EXPLICIT_SNAPSHOT_SCHEMA_V4_AND_DELAY_REAUDIT_V1_REPORT.json", report)
    print(json.dumps({"classification": results["classification"], "recapture": "24/24", "L0_action_exact": results["L0_action_exact"], "authority": {m: results["modes"][m]["local_authority_pass"] for m in MODES}, "transfer": {m: results["transfers"][m]["successful_pair_count"] for m in MODES}}, indent=2))


if __name__ == "__main__":
    main()
