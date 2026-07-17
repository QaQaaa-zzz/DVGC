"""Analyze independent exact-state C_D audit without distance generalization."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from dvgc.bank import SnapshotBank, beta_posterior, posterior_label
from dvgc.certification import assert_disjoint_branch_seeds
from dvgc.config import file_sha256, load_config
from dvgc.discrete_tube import ExactTubeMembership, snapshot_identity
from dvgc.runtime import save_json
from dvgc.seed_registry import seed_set_sha256


def calibration_metrics(prediction, observation, bins=5):
    p, y = np.asarray(prediction, np.float64), np.asarray(observation, np.float64)
    brier = float(np.mean((p-y)**2))
    ece = 0.0
    edges = np.linspace(0, 1, bins+1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p <= hi if hi == 1 else p < hi)
        if mask.any(): ece += float(mask.mean()) * abs(float(p[mask].mean()-y[mask].mean()))
    return {"brier": brier, "ece": float(ece), "bins": bins}


def audit_label(successes, branches, cfg):
    posterior = beta_posterior(successes, branches-successes, alpha0=cfg.beta_alpha0,
                              beta0=cfg.beta_beta0, q_low=cfg.posterior_q_low,
                              q_high=cfg.posterior_q_high)
    label = posterior_label(posterior, branches, min_branches=cfg.min_branches,
                            safe_threshold=cfg.safe_threshold, dead_threshold=cfg.dead_threshold,
                            boundary_max_width=cfg.boundary_max_width)
    return label, posterior


def grouped_summary(rows, key):
    grouped=defaultdict(list)
    for row in rows:grouped[str(row.get(key))].append(row)
    result={}
    for name,values in grouped.items():
        successes=sum(row["audit_successes"] for row in values);branches=sum(row["audit_branches"] for row in values)
        members=[row for row in values if row["pointwise_member"]]
        result[name]={"states":len(values),"exact_members":len(members),"audit_safe":sum(row["audit_label"]=="safe" for row in values),
                      "final_rate":successes/branches if branches else 0.0,
                      "parents":len({row["parent"] for row in values})}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-bank", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--construction-report", required=True)
    parser.add_argument("--audit-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/default.json")
    args = parser.parse_args()
    out = Path(args.output)
    if out.exists(): raise SystemExit(f"Output exists: {out}")
    cfg = load_config(args.config)
    bank = SnapshotBank.load(args.candidate_bank)
    manifest_data = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    membership = ExactTubeMembership.from_manifest(manifest_data)
    construction = json.loads(Path(args.construction_report).read_text(encoding="utf-8"))
    audit = json.loads(Path(args.audit_report).read_text(encoding="utf-8"))
    if manifest_data["certification_hash"] != file_sha256(args.construction_report):
        raise SystemExit("Exact Tube certification manifest hash mismatch")
    if manifest_data["policy_hash"] != audit.get("descent_policy_hash"):
        raise SystemExit("Exact Tube policy and audit policy mismatch")
    if audit["candidate_bank_sha256"] != file_sha256(args.candidate_bank):
        raise SystemExit("Pointwise audit candidate bank hash mismatch")
    construction_evidence = [ev for row in construction["rows"] for ev in row["branch_evidence"]]
    audit_evidence = [ev for row in audit["rows"] for ev in row["branch_evidence"]]
    assert_disjoint_branch_seeds(construction_evidence, [ev["branch_seed"] for ev in audit_evidence])
    indices=sorted(int(row["candidate_index"]) for row in audit["rows"])
    if indices!=list(range(len(indices))):raise SystemExit("Pointwise audit global indices are incomplete or duplicated")
    if any(int(row["branches"])!=int(cfg.max_branches) or len(row["branch_evidence"])!=int(cfg.max_branches) for row in audit["rows"]):
        raise SystemExit("Pointwise audit branch budget is incomplete")
    audit_seeds=[int(ev["branch_seed"]) for ev in audit_evidence]
    if len(audit_seeds)!=len(set(audit_seeds)):raise SystemExit("Pointwise audit branch seeds are not unique")
    audit_root=Path(args.audit_report).parent;proof_path=audit_root/"seed_intersection_proof.json";audit_manifest_path=audit_root/"pointwise_audit_manifest.json"
    registry_proof=None
    if proof_path.exists():
        registry_proof=json.loads(proof_path.read_text())
        if registry_proof.get("status")!="PASS" or int(registry_proof.get("intersection_count",-1))!=0:
            raise SystemExit("Global seed-registry intersection proof failed")
        if registry_proof.get("candidate_seed_set_sha256")!=seed_set_sha256(audit_seeds):
            raise SystemExit("Pointwise audit seeds do not match the registered complete seed grid")
    if audit_manifest_path.exists():
        audit_manifest=json.loads(audit_manifest_path.read_text())
        checks=(
            (int(audit_manifest["seed"])==int(audit["seed"]),"seed"),
            (audit_manifest["policy_hash"]==audit["descent_policy_hash"],"policy"),
            (audit_manifest["candidate_bank_sha256"]==audit["candidate_bank_sha256"],"candidate"),
            (audit_manifest["landing_entry_set_sha256"]==audit["landing_entry_set_sha256"],"C_L"),
            (audit_manifest["landing_policy_hash"]==audit["landing_policy_hash"],"pi_L"),
            (audit_manifest["xml_sha256"]==file_sha256("assets/orange_bike_4kg_horizontal.xml"),"XML"),
        )
        failed=[name for ok,name in checks if not ok]
        if failed:raise SystemExit(f"Pointwise audit manifest provenance mismatch: {failed}")
    rows = bank.records_for_phase("flight", include_training_only=False)
    by_id = {row["id"]: row for row in rows}
    if set(by_id) != {row["id"] for row in audit["rows"]}:
        raise SystemExit("Pointwise audit ids do not exactly cover D_all_unique")
    evaluated = []
    for result in audit["rows"]:
        record = by_id[result["id"]]
        label, posterior = audit_label(int(result["final"]), int(result["branches"]), cfg)
        predicted = membership.contains(record, policy_hash=manifest_data["policy_hash"],
                                        certification_hash=manifest_data["certification_hash"])
        evaluated.append({"id": record["id"], "snapshot_sha256": snapshot_identity(record),
                          "construction_label": record["final"]["label"], "construction_probability": record["final"]["posterior"]["mean"],
                          "parent": record.get("entry_source_id", record.get("parent_candidate_id")),
                          "layer": record.get("descent_layer"), "pointwise_member": predicted,
                          "audit_successes": int(result["final"]), "audit_branches": int(result["branches"]),
                          "audit_rate": float(result["final_rate"]), "audit_label": label,
                          "audit_posterior": posterior})
    truth = [row["audit_label"] == "safe" for row in evaluated]
    pred = [row["pointwise_member"] for row in evaluated]
    tp = sum(p and t for p,t in zip(pred,truth)); fp = sum(p and not t for p,t in zip(pred,truth))
    fn = sum(not p and t for p,t in zip(pred,truth)); tn = len(pred)-tp-fp-fn
    precision = tp/(tp+fp) if tp+fp else 1.0; recall = tp/(tp+fn) if tp+fn else 0.0
    member_rows = [row for row in evaluated if row["pointwise_member"]]
    audit_rates = [row["audit_rate"] for row in evaluated]
    prediction = [row["construction_probability"] for row in evaluated]
    branches = int(audit["terminal_summary"]["branches"])
    physical = int(audit["terminal_summary"]["physical_failures"])
    terminal_causes=Counter(str(ev["terminal_cause"]) for ev in audit_evidence)
    end_reasons=Counter(str(ev["end_reason"]) for ev in audit_evidence if ev.get("end_reason") is not None)
    nonfinite=end_reasons.get("nonfinite") if end_reasons else None
    if nonfinite is not None and any(ev.get("end_reason")=="nonfinite" and ev.get("terminal_cause")!="physical_failure" for ev in audit_evidence):
        raise SystemExit("A nonfinite branch was not counted as physical failure")
    physical_end_reasons = audit["terminal_summary"].get("physical_end_reasons", {})
    report = {
        "status": "PASS" if precision >= .95 and len(member_rows) >= 4 else "FAIL",
        "gate": {"minimum_precision": .95, "minimum_unique_safe": 4,
                 "parent_diversity_is_acquisition_priority_not_activation_gate":True},
        "pointwise": {"precision": precision, "recall": recall, "candidate_mass_coverage": sum(pred)/len(pred),
                      "confusion": {"tp":tp,"fp":fp,"fn":fn,"tn":tn}},
        "raw_rate_threshold_precision": (sum(row["audit_rate"] >= cfg.safe_threshold for row in member_rows)/len(member_rows)
                                         if member_rows else 1.0),
        "aggregate_final_rate": sum(row["audit_successes"] for row in evaluated)/sum(row["audit_branches"] for row in evaluated),
        "calibration": calibration_metrics(prediction, audit_rates),
        "physical_failure_rate": physical/branches if branches else 0.0,
        "timeout_rate": audit["terminal_summary"]["timeouts"]/branches if branches else 0.0,
        "horizon_rate": audit["terminal_summary"]["horizon_exhaustions"]/branches if branches else 0.0,
        "audit_seed": audit["seed"], "audit_seed_namespace": audit["seed_namespace"],
        "construction_seed_namespace": construction["seed_namespace"], "seed_isolation_pass": True,
        "global_seed_registry_proof":registry_proof,
        "branch_validation":{"states":len(evaluated),"branches_per_state":int(cfg.max_branches),
                             "unique_branch_seeds":len(set(audit_seeds)),"global_indices_complete":True},
        "terminal_causes":dict(terminal_causes),
        "termination_subcauses":{"available":bool(audit["terminal_summary"].get("physical_end_reasons_available", end_reasons)),
                                  "counts":dict(end_reasons),
                                  "physical_end_reasons":physical_end_reasons,
                                  "pitch":physical_end_reasons.get("pitch"),
                                  "roll":physical_end_reasons.get("roll"),
                                  "nonfinite":physical_end_reasons.get("nonfinite", nonfinite)},
        "provenance":{"consistent":True,"policy_hash":audit["descent_policy_hash"],
                      "candidate_bank_sha256":audit["candidate_bank_sha256"],
                      "xml_sha256":audit_manifest.get("xml_sha256") if audit_manifest_path.exists() else None,
                      "landing_entry_set_sha256":audit["landing_entry_set_sha256"],
                      "landing_policy_hash":audit["landing_policy_hash"],
                      "frozen_manifest_policy_hash":manifest_data["policy_hash"]},
        "member_results": member_rows, "audit_label_distribution": dict(Counter(row["audit_label"] for row in evaluated)),
        "member_parent_count":len({row["parent"] for row in member_rows}),
        "by_parent":grouped_summary(evaluated,"parent"),"by_layer":grouped_summary(evaluated,"layer"),
        "rows": evaluated,
    }
    save_json(out, report)
    print(json.dumps({k:v for k,v in report.items() if k != "rows"}, indent=2))


if __name__ == "__main__": main()
