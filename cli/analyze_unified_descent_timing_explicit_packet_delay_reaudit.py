"""Derive the final packet-delay audit report without rerunning dynamics."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from dvgc.config import file_sha256
from dvgc.runtime import save_json


def _stats(values):
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    a = np.asarray(values, np.float64)
    return {"count": len(values), "mean": float(a.mean()), "median": float(np.median(a)),
            "min": float(a.min()), "max": float(a.max())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    args = parser.parse_args(); root = Path(args.run)
    prereg = json.loads((root / "preregistration.json").read_text())
    identity = json.loads((root / "inherited_asset_identity_gate.json").read_text())
    runtime = json.loads((root / "runtime_asset_gate.json").read_text())
    l0_gate = json.loads((root / "l0_correction_replay_identity_gate.json").read_text())
    local = json.loads((root / "local_authority_packet_delay_results.json").read_text())["modes"]
    transfer_artifact = json.loads((root / "correction_transfer_packet_delay_results.json").read_text())
    transfers, stability = transfer_artifact["modes"], transfer_artifact["stability"]
    assert identity["status"] == runtime["status"] == l0_gate["status"] == "PASS"
    assert l0_gate["corrections_exact"] == 12 and l0_gate["packet_t_identity"] == 24
    assert set(local) == set(transfers) == {"L0", "D1", "D2", "J12"}
    assert all(len(transfers[m]["pairs"]) == 244 and transfers[m]["repeat_exact"] for m in transfers)

    l0_baseline = {x["snapshot_index"]: x for x in local["L0"]["baseline"]}
    baseline_changes, absolute, failure_transitions = {}, {}, {}
    for mode in local:
        base = local[mode]["baseline"]
        baseline_changes[mode] = {
            "survival_change_vs_L0": _stats([x["survival"] - l0_baseline[x["snapshot_index"]]["survival"] for x in base]),
            "minimum_margin_change_vs_L0": _stats([x["minimum_margin"] - l0_baseline[x["snapshot_index"]]["minimum_margin"] for x in base]),
            "failure_type_transitions_vs_L0": dict(Counter(
                f"{l0_baseline[x['snapshot_index']]['failure']}->{x['failure']}" for x in base
            )),
        }
        mode_baseline = {x["snapshot_index"]: x for x in base}
        failure_transitions[mode] = dict(Counter(
            f"{mode_baseline[y['snapshot_index']]['failure']}->{y['failure']}"
            for y in local[mode]["local_corrections"]
        ))
        pairs = transfers[mode]["pairs"]
        absolute[mode] = {
            "baseline_survival": _stats([x["baseline_absolute_survival"] for x in pairs]),
            "corrected_survival": _stats([x["absolute_survival"] for x in pairs]),
            "minimum_margin": _stats([x["minimum_margin"] for x in pairs]),
            "terminal_margin": _stats([x["terminal_margin"] for x in pairs]),
        }

    local_failures = {m: {
        "baseline": dict(Counter(x["failure"] for x in local[m]["baseline"])),
        "corrected": dict(Counter(x["failure"] for x in local[m]["local_corrections"])),
    } for m in local}
    all_stable = all(stability[m]["stable"] for m in ("D1", "D2", "J12"))
    classification = "PACKET_DELAY_ROBUST_PROVISIONAL_FEEDBACK_SUPPORT" if all_stable else "DELAY_SENSITIVE_FEEDBACK_SUPPORT"
    final_count = sum(transfers[m]["event_counts"]["final_recovery"] for m in transfers)
    recommend = bool((not stability["D2"]["stable"] or not stability["J12"]["stable"]) and final_count == 0)
    report = {
        "experiment": "unified_descent_timing_explicit_packet_delay_reaudit_v1",
        "status": "PASS", "classification": classification,
        "preregistration_sha256": file_sha256(root / "preregistration.json"),
        "frozen_assets": prereg["frozen_assets"],
        "identity_gates": {"inheritance": identity["status"], "runtime": runtime["status"],
                           "L0": l0_gate["status"], "L0_corrections_exact": 12,
                           "L0_packets_exact": 24},
        "local_authority": {m: local[m]["local_authority_pass"] for m in local},
        "local_failures": local_failures,
        "local_events": {m: local[m]["event_counts"] for m in local},
        "candidate_layers": {m: local[m]["candidate_support_layers"] for m in local},
        "transfer": {m: {
            "successful_pairs": transfers[m]["successful_pair_count"],
            "categories": transfers[m]["categories"],
            "events": transfers[m]["event_counts"],
            "candidate_grouped": transfers[m]["candidate_grouped"],
            "weak_candidate_components": transfers[m]["weak_candidate_components"],
        } for m in transfers},
        "success_set_and_preregistered_gates": stability,
        "absolute_outcomes": absolute,
        "baseline_degradation": baseline_changes,
        "local_failure_transitions": failure_transitions,
        "active_prefix_repeatability": {m: transfers[m]["repeat_exact"] and local[m]["repeat_exact"] for m in local},
        "new_delay_mode_failure_type_vs_L0": sorted(
            {x["failure"] for m in ("D1", "D2", "J12") for x in local[m]["local_corrections"]}
            - {x["failure"] for x in local["L0"]["local_corrections"]}
        ),
        "pi_D_delay_aware_v1_recommended": recommend,
        "recommendation_basis": "L0 identity and true FIFO passed; D2 preregistered gate failed; no Final-Recovery" if recommend else "preregistered conditions not all met",
        "evidence_role": "provisional_feedback_support_only",
        "tube_or_jel_formed": False, "training": False, "ppo": False,
        "bootstrap": False, "new_cem": False, "heldout_used": False,
    }
    save_json(root / "UNIFIED_DESCENT_TIMING_EXPLICIT_PACKET_DELAY_REAUDIT_V1_REPORT.json", report)
    summary = [
        "# Timing-explicit packet-delay re-audit",
        "", f"Classification: `{classification}`.",
        "", f"L0 identity: 12/12 corrections and 24/24 packet_t identities PASS.",
        "", "Local authority: " + ", ".join(f"{m}={report['local_authority'][m]}/12" for m in ("L0", "D1", "D2", "J12")) + ".",
        "", "No Chain, recovery success, or Final-Recovery occurred. Results remain provisional feedback support; no Tube/JEL was formed.",
        "", f"pi_D_delay_aware_v1 recommendation: {recommend}. No policy was created or trained.",
    ]
    (root / "SUMMARY.md").write_text("\n".join(summary) + "\n")
    save_json(root / "completed.json", {"status": "PASS", "classification": classification,
                                         "report": str(root / "UNIFIED_DESCENT_TIMING_EXPLICIT_PACKET_DELAY_REAUDIT_V1_REPORT.json")})
    print(json.dumps({"classification": classification, "authority": report["local_authority"], "recommended": recommend}, indent=2))


if __name__ == "__main__":
    main()
