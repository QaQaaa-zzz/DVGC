"""Normalize the passed compact-expert audit into a standard immutable Tube."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from dvgc.bank import SnapshotBank, beta_posterior, posterior_label
from dvgc.config import file_sha256, load_config
from dvgc.descent_entry import DESCENT_ENTRY_FEATURE_NAMES, descent_entry_feature
from dvgc.runtime import save_json


DEFAULT_ROOT=Path("runs/descent_diverse_p1_predecessor_recovery_v5_p1_core_adapter/independent_audit_v1")


def certified_outcome(successes,branches,cfg):
    posterior=beta_posterior(successes,branches-successes)
    return {"successes":int(successes),"failures":int(branches-successes),"branches":int(branches),
        "posterior":posterior,"label":posterior_label(posterior,branches,min_branches=int(cfg.min_branches),
        safe_threshold=float(cfg.safe_threshold),dead_threshold=float(cfg.dead_threshold),
        boundary_max_width=float(cfg.boundary_max_width))}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--audit-root",default=str(DEFAULT_ROOT));args=parser.parse_args()
    root=Path(args.audit_root);source=root/"descent_tube.pkl";audit_path=root/"report.json"
    output=root/"descent_tube_v2.pkl";handoff=root/"canonical_descent_exact_entry_v1.pkl";report_path=root/"tube_v2_finalize_report.json"
    if output.exists() or handoff.exists() or report_path.exists():raise SystemExit("refusing overwrite finalized Tube artifacts")
    audit=json.loads(audit_path.read_text());bank=SnapshotBank.load(source);cfg=load_config("configs/default.json")
    if audit["status"]!="PASS" or audit["artifact_role"]!="certified_tube":raise SystemExit("source audit not passed")
    by_id={row["id"]:row for row in audit["rows"]}
    if set(by_id)!={row["id"] for row in bank.records}:raise SystemExit("audit/Tube state identities differ")
    version="descent-compact-"+hashlib.sha256((file_sha256(audit_path)+audit["policy_version"]).encode()).hexdigest()[:12]
    records=[];seeds=[]
    for record in bank.records:
        row=by_id[record["id"]];branches=copy.deepcopy(row["branches"]);seeds.extend(int(branch["branch_seed"]) for branch in branches)
        final=sum(bool(branch["final_recovery"]) for branch in branches);chain=sum(bool(branch["chain_success"]) for branch in branches)
        item=copy.deepcopy(record);item.update({"source_phase":"flight","origin_phase":"descent",
            "entry_feature":descent_entry_feature(item["physical_feature"],cfg).astype("float32"),
            "descent_layer":row["layer"],"descent_region":row["region"],"chain":certified_outcome(chain,len(branches),cfg),
            "final":certified_outcome(final,len(branches),cfg),"policy_version":audit["policy_version"],
            "estimator_version":"event_filter_v1","tube_version":version,"certification_branches":branches,
            "artifact_role":"certified_tube","certified_safe":row["audit_label"]=="safe",
            "tube_metrics_eligible":True,"safe_claim_allowed":True})
        records.append(item)
    if len(seeds)!=len(set(seeds)):raise SystemExit("audit branch seed reuse")
    if any(record["final"]["label"]!="safe" for record in records):raise SystemExit("not all v2 Tube members are Final-safe")
    metadata=copy.deepcopy(bank.metadata);metadata.update({"artifact_role":"certified_tube","phase":"descent",
        "last_tube_version":version,"last_policy_version":audit["policy_version"],"certification_report":str(audit_path.resolve()),
        "certification_report_sha256":file_sha256(audit_path),"branches_per_state":32,"continuous_matcher_active":False,
        "standard_record_certification_fields":True,"supersedes":str(source.resolve())})
    SnapshotBank(records,metadata).save(output)
    handoff_meta=copy.deepcopy(metadata);handoff_meta.update({"entry_bank_role":"canonical_descent_exact_handoff_set",
        "membership_type":"exact_snapshot_identity","continuous_matcher_active":False,
        "entry_feature_names":DESCENT_ENTRY_FEATURE_NAMES,"radius":None,"training_labels_from_audit":False})
    SnapshotBank(copy.deepcopy(records),handoff_meta).save(handoff)
    result={"status":"PASS","tube_version":version,"policy_version":audit["policy_version"],"states":len(records),
        "safe_states":sum(record["final"]["label"]=="safe" for record in records),"branches":len(seeds),
        "unique_branch_seeds":len(set(seeds)),"tube_sha256":file_sha256(output),"tube_path":str(output),
        "exact_handoff_sha256":file_sha256(handoff),"exact_handoff_path":str(handoff),"continuous_matcher_active":False,
        "entry_feature_names":DESCENT_ENTRY_FEATURE_NAMES,"v1_status":"SUPERSEDED_SCHEMA_ONLY_DO_NOT_USE_AS_STANDARD_BANK",
        "scientific_outcomes_changed":False,"audit_labels_used_for_training":False}
    save_json(report_path,result);print(json.dumps(result,indent=2))


if __name__=="__main__":main()
