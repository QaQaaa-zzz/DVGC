"""Independently verify a frozen final shared-policy JEL artifact contract."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256
from dvgc.policy import load_bundle
from dvgc.runtime import save_json


STAGES = ("takeoff", "ascent", "apex", "descent", "landing")


def verify_rows(records: list[dict], policy_hash: str) -> tuple[dict, list[str]]:
    reasons = []; phases = Counter()
    all_seeds = []
    for row in records:
        phase = row.get("phase_rsi_stage"); phases[str(phase)] += 1
        final = row.get("final", {}); branches = row.get("certification_branches", [])
        construction = row.get("construction_final_branches", [])
        audit = row.get("independent_audit_final_branches", [])
        if row.get("artifact_role") != "final_shared_policy_jel":
            reasons.append(f"{row.get('id')}: role")
        if not (row.get("certified_safe") is True and row.get("safe_claim_allowed") is True
                and row.get("formal_shared_policy_jel") is True
                and row.get("training_only") is False):
            reasons.append(f"{row.get('id')}: safety flags")
        if row.get("policy_params_sha256") != policy_hash:
            reasons.append(f"{row.get('id')}: policy identity")
        if not (final.get("label") == "safe" and int(final.get("successes", -1)) == 64
                and int(final.get("failures", -1)) == 0 and int(final.get("branches", -1)) == 64):
            reasons.append(f"{row.get('id')}: explicit Final outcome")
        if len(branches) != 64 or len(construction) != 32 or len(audit) != 32:
            reasons.append(f"{row.get('id')}: branch counts")
        if any(branch.get("final_recovery") is not True for branch in branches):
            reasons.append(f"{row.get('id')}: non-Final branch")
        if ({branch.get("certification_round") for branch in construction} != {"construction"}
                or {branch.get("certification_round") for branch in audit} != {"independent_audit"}):
            reasons.append(f"{row.get('id')}: round identity")
        all_seeds.extend(branch.get("seed") for branch in branches)
    if None in all_seeds or len(all_seeds) != len(set(all_seeds)):
        reasons.append("branch seeds absent or duplicated across frozen JEL")
    if set(phases) != set(STAGES) or any(phases[stage] <= 0 for stage in STAGES):
        reasons.append("all five phases are not represented")
    return {stage: phases.get(stage, 0) for stage in STAGES}, reasons


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jel-bank", required=True)
    parser.add_argument("--jel-report", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--root-candidate-bank", required=True)
    parser.add_argument("--canonical-entry-bank", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise SystemExit("refusing to overwrite final JEL verification")
    bank = SnapshotBank.load(args.jel_bank); report = json.loads(Path(args.jel_report).read_text())
    _, _, manifest = load_bundle(args.policy, verify_files=True)
    policy_hash = file_sha256(Path(args.policy) / "params.pkl")
    phase_counts, row_reasons = verify_rows(bank.records, policy_hash)
    checks = {
        "bank_role": bank.metadata.get("artifact_role") == "final_shared_policy_jel",
        "formal_and_independent": (bank.metadata.get("formal_shared_policy_jel") is True
                                   and bank.metadata.get("independent_audit") is True
                                   and bank.metadata.get("certified_tube") is True),
        "policy_identity": bank.metadata.get("policy_params_sha256") == policy_hash,
        "xml_identity": bank.metadata.get("xml_sha256") == manifest.get("xml_sha256"),
        "action_mapping_identity": (bank.metadata.get("action_mapping_version")
                                    == manifest.get("action_mapping_version")),
        "root_candidate_identity": (bank.metadata.get("root_source_candidate_bank_sha256")
                                    == file_sha256(args.root_candidate_bank)),
        "canonical_entry_identity": (bank.metadata.get("canonical_entry_bank_sha256")
                                     == file_sha256(args.canonical_entry_bank)),
        "report_status": report.get("status") == "PASS",
        "report_bank_identity": report.get("output_bank_sha256") == file_sha256(args.jel_bank),
        "report_safe_count": int(report.get("safe_states", -1)) == len(bank.records),
        "report_phase_counts": report.get("phase_safe_counts") == phase_counts,
        "coverage_denominator": (int(report.get("root_source_states", -1))
                                 == len(SnapshotBank.load(args.root_candidate_bank).records)),
        "row_contract": not row_reasons,
    }
    payload = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "artifact_role": "final_shared_policy_jel_independent_structure_verification",
        "checks": checks, "row_reasons": row_reasons, "safe_states": len(bank.records),
        "phase_safe_counts": phase_counts, "jel_bank_sha256": file_sha256(args.jel_bank),
        "jel_report_sha256": file_sha256(args.jel_report), "policy_params_sha256": policy_hash,
        "formal_jel_verified": all(checks.values()),
    }
    save_json(output, payload); print(json.dumps(payload, indent=2))
    if payload["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
