"""Build the provenance-locked phase-balanced reset bank for unified Tube-RSI."""
from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from cli.train_stage_reachability_model import parent_key
from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256
from dvgc.runtime import save_json


STAGES = ("takeoff", "ascent", "apex", "descent", "landing")
LOCAL_STAGES = {"takeoff", "ascent", "apex"}
FORMAL_STAGES = {"descent", "landing"}


def eligible_records(stage: str, bank: SnapshotBank) -> list[dict]:
    if stage in LOCAL_STAGES:
        metadata = bank.metadata
        if (metadata.get("artifact_role") != "stage_entry_certified_proposal_support"
                or metadata.get("evidence_scope") != "local_next_stage"
                or metadata.get("independent_audit") is not True
                or int(metadata.get("branch_count", 0)) != 32
                or metadata.get("certified_tube") is not False):
            raise ValueError(f"{stage} input is not isolated 32-branch local support")
        return list(bank.records)
    if stage == "descent":
        if (bank.metadata.get("artifact_role") != "certified_tube"
                or bank.metadata.get("independent_audit") is not True
                or int(bank.metadata.get("branches_per_state", 0)) != 32):
            raise ValueError("Descent input is not the independently audited Final-safe Tube")
        rows = [row for row in bank.records
                if row.get("final", {}).get("label") == "safe" and row.get("certified_safe") is True]
    elif stage == "landing":
        rows = [row for row in bank.records if row.get("final", {}).get("label") == "safe"]
    else:
        raise ValueError(f"unsupported stage {stage}")
    if not rows:
        raise ValueError(f"{stage} has no Final-safe states")
    return rows


def build_balanced_records(banks: dict[str, SnapshotBank], source_hashes: dict[str, str]):
    if set(banks) != set(STAGES):
        raise ValueError(f"exact stages required: {STAGES}")
    stage_mass = 1.0 / len(STAGES)
    output, counts, parent_counts = [], {}, {}
    for stage in STAGES:
        rows = eligible_records(stage, banks[stage])
        groups: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            groups[parent_key(row)].append(row)
        counts[stage] = len(rows); parent_counts[stage] = len(groups)
        for parent in sorted(groups):
            weight = stage_mass / len(groups) / len(groups[parent])
            for row in groups[parent]:
                item = copy.deepcopy(row)
                origin_id = str(item["id"])
                for key in ("final", "chain", "certification_branches", "certified_safe",
                            "safe_claim_allowed", "tube_metrics_eligible"):
                    item.pop(key, None)
                item.update({
                    "id": f"phase-rsi:{stage}:{origin_id}",
                    "origin_record_id": origin_id,
                    "origin_artifact_sha256": source_hashes[stage],
                    "origin_artifact_role": banks[stage].metadata.get("artifact_role", "legacy_certified_tube"),
                    "phase_rsi_stage": stage,
                    "origin_phase": row.get("source_phase"),
                    "artifact_role": "proposal_support_bank",
                    "training_only": True,
                    "reset_source": "flight_curriculum",
                    "reset_parent_id": f"{stage}:{parent}",
                    "reset_weight": weight,
                })
                output.append(item)
    total = sum(float(row["reset_weight"]) for row in output)
    if not np.isclose(total, 1.0, atol=1e-8):
        raise ValueError(f"reset weights sum to {total}")
    return output, counts, parent_counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for stage in STAGES:
        parser.add_argument(f"--{stage}-bank", required=True)
    parser.add_argument("--landing-completion-analysis", required=True)
    parser.add_argument("--output-bank", required=True)
    parser.add_argument("--output-report", required=True)
    args = parser.parse_args()
    output_bank, output_report = Path(args.output_bank), Path(args.output_report)
    if output_bank.exists() or output_report.exists():
        raise SystemExit("refusing overwrite phase-balanced Tube-RSI bank")
    paths = {stage: Path(getattr(args, f"{stage}_bank")) for stage in STAGES}
    hashes = {stage: file_sha256(path) for stage, path in paths.items()}
    landing_analysis = json.loads(Path(args.landing_completion_analysis).read_text())
    final_landing = landing_analysis.get("final_recertification", {})
    audit_landing = landing_analysis.get("independent_audit", {})
    if (landing_analysis.get("status") != "LANDING_COMPLETE_AUDITED"
            or final_landing.get("tube_sha256") != hashes["landing"]
            or audit_landing.get("status") != "PASS_INDEPENDENT_AUDIT"):
        raise SystemExit("Landing Tube completion/audit identity is not current")
    banks = {stage: SnapshotBank.load(path) for stage, path in paths.items()}
    rows, counts, parents = build_balanced_records(banks, hashes)
    output_bank.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "artifact_role": "phase_balanced_tube_rsi_reset_bank",
        "formal_tube_or_jel": False,
        "reset_source_protocol": {
            "name": "phase_balanced_tube_rsi_v1",
            "bank_phase_masses": {stage: 1.0 / len(STAGES) for stage in STAGES},
            "within_phase": "equal parent mass then equal state mass",
            "natural_reset_probability": "configured separately by unified PPO",
        },
        "source_bank_sha256s": hashes,
        "source_roles": {stage: banks[stage].metadata.get("artifact_role", "legacy_certified_tube")
                         for stage in STAGES},
        "local_support_is_not_formal_tube": sorted(LOCAL_STAGES),
        "formal_final_safe_sources": sorted(FORMAL_STAGES),
    }
    SnapshotBank(rows, metadata).save(output_bank)
    report = {
        "status": "PASS", **metadata,
        "records": len(rows), "stage_state_counts": counts,
        "stage_parent_counts": parents,
        "stage_weight_mass": {stage: sum(float(row["reset_weight"]) for row in rows
                                         if row["phase_rsi_stage"] == stage) for stage in STAGES},
        "source_phase_counts": dict(Counter(str(row.get("origin_phase")) for row in rows)),
        "weight_sum": sum(float(row["reset_weight"]) for row in rows),
        "output_bank": str(output_bank), "output_bank_sha256": file_sha256(output_bank),
        "landing_completion_analysis_sha256": file_sha256(args.landing_completion_analysis),
    }
    save_json(output_report, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
