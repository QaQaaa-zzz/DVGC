"""Build a non-certified training/search support bank from legal empirical states."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256
from dvgc.research_semantics import PROPOSAL_SUPPORT_BANK, proposal_training_weight
from dvgc.runtime import save_json


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-bank", required=True)
    p.add_argument("--stable-report", required=True)
    p.add_argument("--physical-audit", required=True)
    p.add_argument("--output-bank", required=True)
    p.add_argument("--output-report", required=True)
    a = p.parse_args()
    out, report_path = Path(a.output_bank), Path(a.output_report)
    if out.exists() or report_path.exists():
        raise SystemExit("Proposal support output already exists")
    source = SnapshotBank.load(a.source_bank)
    stable = json.loads(Path(a.stable_report).read_text())
    audit = json.loads(Path(a.physical_audit).read_text())
    if stable.get("candidate_bank_sha256") != file_sha256(a.source_bank):
        raise SystemExit("Stable report/source bank hash mismatch")
    if audit.get("status") != "PASS":
        raise SystemExit("Physical candidate audit is not PASS")
    audit_hash = audit.get("candidate_bank_sha256", audit.get("bank_sha256"))
    if audit_hash != file_sha256(a.source_bank):
        raise SystemExit("Physical audit/source bank hash mismatch")
    stable_rows = {row["id"]: row for row in stable["rows"]}
    retained, excluded = [], {"dead": 0, "unknown": 0, "missing": 0}
    for record in source.records_for_phase("flight", include_training_only=False):
        evidence = stable_rows.get(record["id"])
        if evidence is None:
            excluded["missing"] += 1
            continue
        label = str(evidence["label"])
        posterior = evidence["combined"]["posterior"]
        weight = proposal_training_weight(label, posterior["lower"])
        if label == "dead":
            excluded["dead"] += 1
            continue
        item = copy.deepcopy(record)
        item.update({
            "artifact_role": PROPOSAL_SUPPORT_BANK,
            "empirical_label": label,
            "empirical_posterior": copy.deepcopy(posterior),
            "proposal_training_weight": weight,
            "proposal_active_sampling_only": label == "unknown",
            "certified_safe": False,
            "source_stable_report_sha256": file_sha256(a.stable_report),
        })
        if label == "unknown":
            excluded["unknown"] += 1
            item["training_only"] = True
        retained.append(item)
    metadata = copy.deepcopy(source.metadata)
    metadata.update({
        "artifact_role": PROPOSAL_SUPPORT_BANK,
        "safe_claim_allowed": False,
        "tube_metrics_eligible": False,
        "source_bank": str(Path(a.source_bank).resolve()),
        "source_bank_sha256": file_sha256(a.source_bank),
        "source_stable_report": str(Path(a.stable_report).resolve()),
        "source_stable_report_sha256": file_sha256(a.stable_report),
        "physical_audit": str(Path(a.physical_audit).resolve()),
        "physical_audit_sha256": file_sha256(a.physical_audit),
    })
    SnapshotBank(retained, metadata).save(out)
    labels = {name: sum(row["empirical_label"] == name for row in retained)
              for name in ("safe", "boundary", "unknown")}
    report = {
        "status": "PASS", "artifact_role": PROPOSAL_SUPPORT_BANK,
        "safe_claim_allowed": False, "tube_metrics_eligible": False,
        "states": len(retained), "labels": labels, "excluded": excluded,
        "source_bank_sha256": file_sha256(a.source_bank),
        "stable_report_sha256": file_sha256(a.stable_report),
        "physical_audit_sha256": file_sha256(a.physical_audit),
        "output_bank_sha256": file_sha256(out),
    }
    save_json(report_path, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
