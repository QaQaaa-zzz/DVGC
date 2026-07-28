"""Strict v4 complete-packet delay re-audit for the frozen Descent policy."""
from __future__ import annotations

import argparse
import json
import pickle
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from cli.run_timing_explicit_snapshot_schema_v4_delay_reaudit import (
    AUTHORITY, CONFIG, MODES, MULTIMODALITY, POLICY, TRANSFER,
    _assets, _batched_state, _medoid_corrections, _semantic_hash_legacy,
    _semantic_hash_v4, _support_layers, _transfer_summary,
)
from cli.runtime_gate import source_fingerprint
from dvgc.config import ACTION_MAPPING_VERSION, file_sha256
from dvgc.delay_probe import active_prefix_repeat_comparison, make_packet_delay_rollout
from dvgc.env import END_REASON
from dvgc.observation_audit import array_sha256
from dvgc.ppo_integrity import normalizer_summary
from dvgc.runtime import save_json
from dvgc.snapshot_timing import validate_transfer_eligibility


EXPERIMENT = "unified_descent_timing_explicit_packet_delay_reaudit_v1"
EXPECTED_START = "948fb24564d1d2a524f4dbfa51d93ce2964b1a1f"
V4 = Path("runs/v4_current_frame_independent_reconstruction_localization_v1/timing_explicit_snapshots.pkl")
V4_GATE = Path("runs/v4_current_frame_independent_reconstruction_localization_v1/postfix_24_state_identity_gate.json")
PRECURSOR = Path("runs/unified_descent_snapshot_timing_and_delay_sensitivity_audit_v2_retry1/UNIFIED_DESCENT_SNAPSHOT_TIMING_AND_DELAY_SENSITIVITY_AUDIT_V2_REPORT.json")
EXPECTED_HASHES = {
    "v4_snapshots": "9bd00d3e8e06f440e2338c7a5e1c61ac2e376f0712c960393b8ae7738798ae28",
    "v4_gate": "d28cce478fd532ad8a288f486702566061452aebed51a771a223b51a686bee73",
    "authority": "9cf20b84c79a71994b07bdd9060881bad430329f2fed83618319afcecd0c2b07",
    "multimodality": "3559aca02fad392b2805344e236145b1440039254e0d01c015d5558fe87562da",
    "transfer": "fc6a883cab5db3a85a7fb79e3082f81ab16b9bed3e121db184316dec485b5345",
    "precursor": "84064ee503cc6dd83414a68182afb90f0967cbbe508b5ff05ae4d98befb4fd76",
    "policy_params": "52721668eed0cc78b41a45ad7c319e687f43add8977f2b4bdfcad8208c4353f2",
    "policy_config": "20e3fd3aaf57569ff6d921b8e5a0f389020db36589562326c3ee98b71f7d8d3d",
    "policy_manifest": "5679b73af86664cd780a6ee50d07186ac2eb9e34147e6317ca403e243e586d4f",
    "normalizer": "8f2e36b3591dcb90f3808b0c1f7c9cd62dfd72a101445716171163d34c93a7e",
    "xml": "d7e9f43ff8fb9e4571203f81062ce9c828acfa38692ee8c71a3e5daa15ce794c",
    "experiment_config": "cc246788e4931bbc84c97cbdb0749ae88cc5f75108d85a0b3a618b08a11a24ff",
}
DELAY_SEMANTICS = "v4_complete_packet_fifo_t_minus_2_to_t"


def _load(path: Path) -> Any:
    return json.loads(path.read_text())


def _reason(code: int) -> str:
    return END_REASON.get(code, "horizon")


def _selected(authority: dict, multimodality: dict) -> dict[int, dict]:
    ranks = {int(row["snapshot_index"]): int(row["successful_medoid"]["rank"])
             for row in multimodality["rows"]}
    selected = {}
    for row in authority["rows"]:
        index = int(row["snapshot_index"])
        if row["authoritative_correction"]:
            selected[index] = next(x for x in row["top5"] if int(x["rank"]) == ranks[index])
    return selected


def _asset_gate(root: Path, prereg: dict) -> tuple[list, dict, dict, dict, dict]:
    paths = {
        "v4_snapshots": V4, "v4_gate": V4_GATE, "authority": AUTHORITY,
        "multimodality": MULTIMODALITY, "transfer": TRANSFER,
        "precursor": PRECURSOR, "policy_params": POLICY / "params.pkl",
        "policy_config": POLICY / "config.json", "policy_manifest": POLICY / "manifest.json",
        "experiment_config": CONFIG,
    }
    checks: dict[str, bool] = {name: file_sha256(path) == EXPECTED_HASHES[name]
                               for name, path in paths.items()}
    captured = pickle.loads(V4.read_bytes())
    authority, multimodality, transfer, precursor = map(_load, (AUTHORITY, MULTIMODALITY, TRANSFER, PRECURSOR))
    corrections = _medoid_corrections(authority, multimodality)
    selected = _selected(authority, multimodality)
    gate = _load(V4_GATE)
    checks.update({
        "v4_gate_pass": gate.get("status") == "PASS" and gate.get("exact") == 24,
        "snapshot_count": len(captured) == 24,
        "candidate_tick_unique": len({(x["candidate_id"], int(x["tick"])) for x in captured}) == 24,
        "correction_count": len(corrections) == 12 and len(selected) == 12,
        "correction_sources": set(corrections) == {int(x["snapshot_index"]) for x in multimodality["rows"]},
        "pair_count": len(transfer["pairs"]) == 244,
        "pair_keys_unique": len({(int(x["source_snapshot_index"]), int(x["target_snapshot_index"])) for x in transfer["pairs"]}) == 244,
        "pair_indices_valid": all(int(x[k]) in range(24) for x in transfer["pairs"] for k in ("source_snapshot_index", "target_snapshot_index")),
        "pair_sources_have_correction": all(int(x["source_snapshot_index"]) in corrections for x in transfer["pairs"]),
        "correction_content_exact": all(array_sha256(corrections[i]) == array_sha256(np.asarray(selected[i]["residual_knots"], np.float32)) for i in corrections),
        "semantic_identity_24": all(_semantic_hash_legacy(x["legacy_snapshot"]) == _semantic_hash_v4(x["snapshot_v4"]) for x in captured),
        "snapshot_identity": all(x["candidate_id"] == authority["rows"][i]["candidate_id"] and int(x["tick"]) == int(authority["rows"][i]["tick"]) for i, x in enumerate(captured)),
        "action_mapping": ACTION_MAPPING_VERSION == prereg["frozen_assets"]["action_mapping_version"],
    })
    old_l0 = {int(row["snapshot_index"]): row["failure"] for row in precursor["mode_results"]["L0"]["baseline"]}
    expected_rows, actual_rows = [], []
    for pair in transfer["pairs"]:
        source, target = int(pair["source_snapshot_index"]), int(pair["target_snapshot_index"])
        legacy, v4 = captured[target]["legacy_snapshot"], captured[target]["snapshot_v4"]
        common = {
            "source_snapshot_index": source, "target_snapshot_index": target,
            "correction_sha256": array_sha256(corrections[source]),
            "failure_precursor": old_l0[target], "delay_semantics": DELAY_SEMANTICS,
        }
        expected_rows.append(common | {
            "source_snapshot_hash": _semantic_hash_legacy(captured[source]["legacy_snapshot"]),
            "target_snapshot_hash": _semantic_hash_legacy(legacy),
            "phase": int(legacy["oracle_phase"]),
            "contact_mode": f"valid={int(legacy['had_valid_landing'])};age={int(legacy['contact_age'])}",
        })
        est = v4["estimator_state_pre_t"]
        actual_rows.append(common | {
            "source_snapshot_hash": _semantic_hash_v4(captured[source]["snapshot_v4"]),
            "target_snapshot_hash": _semantic_hash_v4(v4),
            "phase": int(est["phase"]),
            "contact_mode": f"valid={int(est['had_valid_landing'])};age={int(est['contact_age'])}",
        })
    eligibility = validate_transfer_eligibility(
        expected_rows, actual_rows,
        expected_artifact_sha256=file_sha256(TRANSFER),
        actual_artifact_sha256=file_sha256(TRANSFER),
    )
    checks["pair_eligibility"] = bool(eligibility["valid"])
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "classification": None if all(checks.values()) else "INHERITED_ASSET_IDENTITY_FAILURE",
        "checks": checks, "failed": sorted(k for k, v in checks.items() if not v),
        "file_hashes": {name: file_sha256(path) for name, path in paths.items()},
        "correction_hashes": {str(i): array_sha256(value) for i, value in corrections.items()},
        "pair_eligibility": eligibility, "result_fields_inherited": False,
        "heldout_used": False,
    }
    save_json(root / "inherited_asset_identity_gate.json", report)
    if report["status"] != "PASS":
        raise RuntimeError(f"INHERITED_ASSET_IDENTITY_FAILURE: {report['failed']}")
    return captured, authority, multimodality, transfer, corrections


def _result(raw: dict, repeat: dict, index: int, baseline: dict | None = None) -> dict:
    comparison = active_prefix_repeat_comparison(raw, repeat, index)
    tick = int(np.asarray(raw["termination_tick"])[index])
    code = int(np.asarray(raw["end_code"])[index])
    actions = np.asarray(raw["actions"])[:tick, index]
    ctrls = np.asarray(raw["ctrls"])[:tick, index]
    packets = np.asarray(raw["packets"])[:tick, index]
    row = {
        "survival": int(np.asarray(raw["survival"])[index]),
        "absolute_survival": int(np.asarray(raw["survival"])[index]),
        "minimum_margin": float(np.asarray(raw["minimum_margin"])[index]),
        "terminal_margin": float(np.asarray(raw["terminal_margin"])[index]),
        "termination_tick": tick, "horizon": tick == 24,
        "end_code": code, "failure": _reason(code),
        "landing_entry": bool(np.asarray(raw["landing_entry"])[index]),
        "chain": bool(np.asarray(raw["chain"])[index]),
        "recovery_success": bool(np.asarray(raw["recovery_success"])[index]),
        "final_recovery": bool(np.asarray(raw["final_recovery"])[index]),
        "active_prefix_repeat_exact": bool(comparison["exact"]),
        "repeat_mismatch_fields": comparison["failed_fields"],
        "active_action_sha256": array_sha256(actions),
        "active_ctrl_sha256": array_sha256(ctrls),
        "active_packet_sha256": array_sha256(packets),
        "packet_delay_trace": np.asarray(raw["packet_delay_trace"])[:tick, index].tolist(),
        "phase_trace_summary": dict(Counter(map(str, np.asarray(raw["phase_trace"])[:tick, index].tolist()))),
        "contact_trace_summary": dict(Counter(map(str, np.asarray(raw["contact_age_trace"])[:tick, index].tolist()))),
    }
    if baseline is not None:
        row["baseline_absolute_survival"] = baseline["absolute_survival"]
        row["gain"] = row["survival"] - baseline["survival"]
        row["no_new_failure_type"] = row["failure"] in {baseline["failure"], "horizon"}
        row["authority_pass"] = bool(row["active_prefix_repeat_exact"] and row["gain"] >= 2 and row["no_new_failure_type"])
        row["authority_fail_reason"] = ([] if row["authority_pass"] else
            (["repeat_mismatch"] if not row["active_prefix_repeat_exact"] else []) +
            (["gain_below_2"] if row["gain"] < 2 else []) +
            (["new_failure_type"] if not row["no_new_failure_type"] else []))
    return row


def _run_twice(rollout, state, residual, queue, seed):
    key = jax.random.PRNGKey(seed)
    return tuple(jax.device_get(rollout(state, residual, queue, key)) for _ in range(2))


def _historical_l0_check(index: int, baseline: dict, corrected: dict, raw: dict,
                         authority_row: dict, selected: dict) -> dict:
    expected_base = authority_row["baseline"]
    replay = selected.get("exact_replay", {}).get("replay", selected)
    expected_actions = np.asarray(replay.get("actions", selected["actions"]), np.float32)
    prefix = corrected["termination_tick"]
    actual_actions = np.asarray(raw["actions"])[:prefix, 0]
    expected_corrected = {
        "survival": int(replay.get("survival", selected["survival"])),
        "minimum_margin": float(replay.get("minimum_margin", selected["minimum_margin"])),
        "terminal_margin": float(replay.get("terminal_margin", selected["terminal_margin"])),
        "end_code": int(replay.get("end_code", selected["end_code"])),
    }
    checks = {
        "baseline_survival": baseline["survival"] == int(expected_base["survival"]),
        "baseline_minimum_margin": baseline["minimum_margin"] == float(expected_base["minimum_margin"]),
        "baseline_terminal_margin": baseline["terminal_margin"] == float(expected_base["terminal_margin"]),
        "baseline_end_code": baseline["end_code"] == int(expected_base["end_code"]),
        "corrected_survival": corrected["survival"] == expected_corrected["survival"],
        "corrected_minimum_margin": corrected["minimum_margin"] == expected_corrected["minimum_margin"],
        "corrected_terminal_margin": corrected["terminal_margin"] == expected_corrected["terminal_margin"],
        "corrected_end_code": corrected["end_code"] == expected_corrected["end_code"],
        "corrected_action_active_prefix": np.array_equal(actual_actions, expected_actions[:prefix]),
        "authority_decision": corrected["authority_pass"] == bool(authority_row["authoritative_correction"]),
        "action_repeat_bit_exact": corrected["active_prefix_repeat_exact"],
        "ctrl_repeat_bit_exact": "ctrls" not in corrected["repeat_mismatch_fields"],
    }
    return {"snapshot_index": index, "checks": checks,
            "exact": all(checks.values()), "failed": [k for k, v in checks.items() if not v],
            "expected_corrected": expected_corrected}


def _mode_local(mode: str, rollout, states, queues, corrections, authority, selected):
    zero = jnp.zeros((1, 2, 4), jnp.float32)
    baseline_rows, corrected_rows, raw_corrected = [], [], {}
    for index in range(24):
        raw, repeat = _run_twice(rollout, states[index], zero, queues[index], 31_000_000 + index)
        baseline_rows.append({"snapshot_index": index} | _result(raw, repeat, 0))
    baseline = {x["snapshot_index"]: x for x in baseline_rows}
    for index in sorted(corrections):
        residual = jnp.asarray(corrections[index][None], jnp.float32)
        raw, repeat = _run_twice(rollout, states[index], residual, queues[index], 32_000_000 + index)
        raw_corrected[index] = raw
        corrected_rows.append({"snapshot_index": index} | _result(raw, repeat, 0, baseline[index]))
    layers = _support_layers(
        [{"candidate_id": row["candidate_id"]} for row in authority["rows"]], corrected_rows
    )
    return {
        "mode": mode, "baseline": baseline_rows, "local_corrections": corrected_rows,
        "local_authority_pass": sum(x["authority_pass"] for x in corrected_rows),
        "candidate_support_layers": layers,
        "baseline_failure_counts": dict(Counter(x["failure"] for x in baseline_rows)),
        "repeat_exact": all(x["active_prefix_repeat_exact"] for x in baseline_rows + corrected_rows),
        "event_counts": {key: sum(x[key] for x in baseline_rows + corrected_rows)
                         for key in ("horizon", "landing_entry", "chain", "recovery_success", "final_recovery")},
    }, raw_corrected


def _transfer_mode(mode, rollout, states, queues, corrections, pairs, baselines, candidates):
    rows = []
    for position, old in enumerate(pairs):
        source, target = int(old["source_snapshot_index"]), int(old["target_snapshot_index"])
        residual = jnp.asarray(corrections[source][None], jnp.float32)
        raw, repeat = _run_twice(rollout, states[target], residual, queues[target], 33_000_000 + position)
        outcome = _result(raw, repeat, 0, baselines[target])
        rows.append({
            "source_snapshot_index": source, "source_candidate_id": old["source_candidate_id"],
            "target_snapshot_index": target, "target_candidate_id": old["target_candidate_id"],
            "same_snapshot": bool(old["same_snapshot"]), "same_candidate": bool(old["same_candidate"]),
            **outcome, "physical_transfer": outcome["authority_pass"],
        })
    summary = _transfer_summary(rows, candidates)
    summary["pairs"] = rows
    summary["event_counts"] = {key: sum(x[key] for x in rows)
                               for key in ("horizon", "landing_entry", "chain", "recovery_success", "final_recovery")}
    summary["repeat_exact"] = all(x["active_prefix_repeat_exact"] for x in rows)
    return summary


def _changes(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    a = np.asarray(values, np.float64)
    return {"count": len(values), "mean": float(a.mean()), "median": float(np.median(a)),
            "min": float(a.min()), "max": float(a.max())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    args = parser.parse_args(); root = Path(args.run)
    prereg_path = root / "preregistration.json"
    if not prereg_path.is_file():
        raise SystemExit("frozen preregistration is required")
    if subprocess.run(["git", "merge-base", "--is-ancestor", EXPECTED_START, "HEAD"]).returncode:
        raise SystemExit("unexpected git history")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise SystemExit("worktree must be clean")
    prereg = _load(prereg_path)
    captured, authority, multimodality, transfer_asset, corrections = _asset_gate(root, prereg)

    cfg, _, env, params = _assets()
    runtime_checks = {
        "xml": file_sha256(cfg.xml_path) == EXPECTED_HASHES["xml"],
        "normalizer": normalizer_summary(params[0])["sha256"] == EXPECTED_HASHES["normalizer"],
        "source_fingerprint": source_fingerprint(Path.cwd()) == prereg["frozen_assets"]["runtime_source_fingerprint"],
    }
    save_json(root / "runtime_asset_gate.json", {"status": "PASS" if all(runtime_checks.values()) else "FAIL", "checks": runtime_checks})
    if not all(runtime_checks.values()):
        raise RuntimeError(f"runtime asset gate failed: {runtime_checks}")

    states = [_batched_state(env, row["snapshot_v4"], 30_000_000 + i) for i, row in enumerate(captured)]
    queues = [jnp.asarray(row["snapshot_v4"]["actor_packet_fifo_t"][None], jnp.float32) for row in captured]
    selected = _selected(authority, multimodality)
    rollouts = {name: make_packet_delay_rollout(env, params, schedule) for name, schedule in MODES.items()}

    l0, l0_raw = _mode_local("L0", rollouts["L0"], states, queues, corrections, authority, selected)
    l0_rows = {x["snapshot_index"]: x for x in l0["local_corrections"]}
    l0_base = {x["snapshot_index"]: x for x in l0["baseline"]}
    historical = [_historical_l0_check(i, l0_base[i], l0_rows[i], l0_raw[i], authority["rows"][i], selected[i]) for i in sorted(corrections)]
    packet_checks = [
        np.array_equal(np.asarray(queues[i])[0, 2], np.asarray(captured[i]["snapshot_v4"]["actor_observation_t"]))
        for i in range(24)
    ]
    l0_gate = {
        "status": "PASS" if all(x["exact"] for x in historical) and all(packet_checks) else "FAIL",
        "classification": None if all(x["exact"] for x in historical) and all(packet_checks) else "L0_CORRECTION_REPLAY_IDENTITY_FAILURE",
        "corrections_exact": sum(x["exact"] for x in historical), "total": 12,
        "packet_t_identity": sum(packet_checks), "packet_total": 24,
        "rows": historical, "local_results": l0,
    }
    save_json(root / "l0_correction_replay_identity_gate.json", l0_gate)
    if l0_gate["status"] != "PASS":
        print(json.dumps({"classification": l0_gate["classification"], "exact": l0_gate["corrections_exact"]}))
        return

    modes = {"L0": l0}
    for mode in ("D1", "D2", "J12"):
        modes[mode], _ = _mode_local(mode, rollouts[mode], states, queues, corrections, authority, selected)
    candidates = sorted({x["candidate_id"] for x in captured})
    transfers = {}
    for mode in ("L0", "D1", "D2", "J12"):
        baseline = {x["snapshot_index"]: x for x in modes[mode]["baseline"]}
        transfers[mode] = _transfer_mode(mode, rollouts[mode], states, queues, corrections,
                                         transfer_asset["pairs"], baseline, candidates)

    l0_set = {tuple(x) for x in transfers["L0"]["successful_pair_set"]}
    l0_pair = {(x["source_snapshot_index"], x["target_snapshot_index"]): x for x in transfers["L0"]["pairs"]}
    stability = {}
    l0_authority = modes["L0"]["local_authority_pass"]
    l0_cross = transfers["L0"]["categories"]["cross_candidate"]["gain_at_least_2"]
    l0_components = transfers["L0"]["weak_candidate_components"]
    l0_component = max((len(x) for x in l0_components), default=0)
    for mode in ("D1", "D2", "J12"):
        current = {tuple(x) for x in transfers[mode]["successful_pair_set"]}
        union = l0_set | current
        successful_l0_rows = [x for x in transfers[mode]["pairs"] if (x["source_snapshot_index"], x["target_snapshot_index"]) in l0_set]
        survival_delta = [x["survival"] - l0_pair[(x["source_snapshot_index"], x["target_snapshot_index"])]["survival"] for x in successful_l0_rows]
        margin_delta = [x["minimum_margin"] - l0_pair[(x["source_snapshot_index"], x["target_snapshot_index"])]["minimum_margin"] for x in successful_l0_rows]
        failure_agreement = sum(a["failure"] == b["failure"] for a, b in zip(modes["L0"]["baseline"], modes[mode]["baseline"], strict=True)) / 24
        layer_changes = sum(modes["L0"]["candidate_support_layers"][c] != modes[mode]["candidate_support_layers"][c] for c in candidates)
        robust_drop = any(modes["L0"]["candidate_support_layers"][c] == "robust-core" and modes[mode]["candidate_support_layers"][c] == "unsupported" for c in candidates)
        cross = transfers[mode]["categories"]["cross_candidate"]["gain_at_least_2"]
        component = max((len(x) for x in transfers[mode]["weak_candidate_components"]), default=0)
        values = {
            "authority_retention": modes[mode]["local_authority_pass"] / max(l0_authority, 1),
            "baseline_failure_agreement": failure_agreement,
            "candidate_layer_changes": layer_changes, "robust_core_to_unsupported": robust_drop,
            "cross_transfer_retention": cross / max(l0_cross, 1),
            "largest_component": component,
            "success_intersection": len(l0_set & current), "lost": len(l0_set - current),
            "gained": len(current - l0_set), "union": len(union),
            "jaccard": 1.0 if not union else len(l0_set & current) / len(union),
            "l0_success_absolute_survival_change": _changes(survival_delta),
            "l0_success_minimum_margin_change": _changes(margin_delta),
        }
        values["stable"] = bool(values["authority_retention"] >= .75 and failure_agreement >= .75 and layer_changes <= 2 and not robust_drop and values["cross_transfer_retention"] >= .70 and component >= l0_component - 1)
        stability[mode] = values

    all_stable = all(stability[x]["stable"] for x in ("D1", "D2", "J12"))
    classification = "PACKET_DELAY_ROBUST_PROVISIONAL_FEEDBACK_SUPPORT" if all_stable else "DELAY_SENSITIVE_FEEDBACK_SUPPORT"
    no_final = not any(transfers[m]["event_counts"]["final_recovery"] for m in transfers)
    suggest = bool(stability["D2"]["stable"] is False or stability["J12"]["stable"] is False) and no_final
    save_json(root / "local_authority_packet_delay_results.json", {"modes": modes})
    save_json(root / "correction_transfer_packet_delay_results.json", {"modes": transfers, "stability": stability})
    report = {
        "experiment": EXPERIMENT, "status": "PASS", "classification": classification,
        "preregistration_sha256": file_sha256(prereg_path),
        "L0_identity_gate": "PASS", "local_authority": {m: modes[m]["local_authority_pass"] for m in modes},
        "events": {m: transfers[m]["event_counts"] for m in transfers},
        "candidate_layers": {m: modes[m]["candidate_support_layers"] for m in modes},
        "transfer_categories": {m: transfers[m]["categories"] for m in transfers},
        "stability": stability, "new_failure_types": sorted({x["failure"] for m in modes for x in modes[m]["local_corrections"]} - {x["failure"] for x in modes["L0"]["local_corrections"]}),
        "pi_D_delay_aware_v1_recommended": suggest,
        "evidence_role": "provisional_feedback_support_only",
        "tube_or_jel_formed": False, "training": False, "ppo": False,
        "bootstrap": False, "new_cem": False, "heldout_used": False,
    }
    save_json(root / "UNIFIED_DESCENT_TIMING_EXPLICIT_PACKET_DELAY_REAUDIT_V1_REPORT.json", report)
    save_json(root / "completed.json", {"status": "PASS", "report": str(root / "UNIFIED_DESCENT_TIMING_EXPLICIT_PACKET_DELAY_REAUDIT_V1_REPORT.json")})
    print(json.dumps({"classification": classification, "authority": report["local_authority"], "recommended": suggest}, indent=2))


if __name__ == "__main__":
    main()
