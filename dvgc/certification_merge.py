"""Strict merge of globally indexed frozen-policy certification chunks."""
from __future__ import annotations

import copy
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from .bank import SnapshotBank
from .certification import summarize_branches


LABEL_FIELDS = (
    "chain", "final", "policy_version", "estimator_version",
    "certification_branches", "label_timestamp", "label_age",
    "seed_namespace", "connection_flag",
)


def merge_certification_parts(
    banks: Sequence[SnapshotBank], reports: Sequence[Mapping[str, Any]]
) -> tuple[SnapshotBank, dict[str, Any]]:
    if not banks or len(banks)!=len(reports): raise ValueError("Certification banks and reports must be non-empty and paired")
    parts=[dict(report) for report in reports]; common=("phase","policy_version","estimator_version","seed_namespace","construction_seed","candidate_bank_sha256","downstream_bank","downstream_bank_sha256","total_bank_states")
    expected={key:parts[0].get(key) for key in common}
    for report in parts[1:]:
        for key,value in expected.items():
            if report.get(key)!=value: raise ValueError(f"Certification parts disagree on {key}")
    total=int(expected["total_bank_states"]); phase=str(expected["phase"])
    source_ids=[[row["id"] for row in bank.records_for_phase(phase,include_training_only=False)] for bank in banks]
    if any(ids!=source_ids[0] for ids in source_ids[1:]) or len(source_ids[0])!=total: raise ValueError("Certification parts do not share the same ordered candidate states")
    covered=[]
    for report in parts:
        start=int(report["state_index_start"]); stop=int(report["state_index_end_exclusive"])
        indices=[int(row["state_index"]) for row in report.get("results",[])]
        if start<0 or stop>total or start>=stop or indices!=list(range(start,stop)): raise ValueError("Certification part range does not match its results")
        covered.extend(indices)
    if sorted(covered)!=list(range(total)) or len(covered)!=len(set(covered)): raise ValueError("Certification parts must cover every global state exactly once")

    merged=copy.deepcopy(banks[0]); merged.invalidate_phase(phase,reason="merge globally indexed certification parts")
    merged_rows=merged.records_for_phase(phase,include_training_only=False); all_branches=[]; results=[]
    for bank,report in zip(banks,parts):
        part_rows=bank.records_for_phase(phase,include_training_only=False)
        for result in report["results"]:
            index=int(result["state_index"]); source=part_rows[index]; target=merged_rows[index]
            if source["id"]!=target["id"] or int(source["final"]["branches"])<=0: raise ValueError("Certification part contains missing or misindexed evidence")
            for key in LABEL_FIELDS: target[key]=copy.deepcopy(source[key])
            all_branches.extend(target["certification_branches"])
            merged_result=copy.deepcopy(result)
            merged_result.setdefault("candidate_kind",target.get("candidate_kind","unknown"))
            merged_result.setdefault("branch_evidence",copy.deepcopy(target["certification_branches"]))
            results.append(merged_result)
    seeds=[int(row["branch_seed"]) for row in all_branches]
    if len(seeds)!=len(set(seeds)): raise ValueError("Certification parts reuse branch seeds")
    tube_version=f"{phase}-{uuid.uuid4().hex[:10]}"
    for row in merged_rows:
        row["tube_version"]=tube_version
    merged.metadata.update({
        "last_policy_version":expected["policy_version"], "last_tube_version":tube_version,
        "downstream_bank":expected["downstream_bank"],
        "construction_seed":int(expected["construction_seed"]),
        "construction_seed_namespace":expected["seed_namespace"],
        "dynamics_variants":banks[0].metadata.get("dynamics_variants",[]),
    })
    results.sort(key=lambda row:int(row["state_index"]))
    report={
        **expected,"tube_version":tube_version,"states":total,
        "state_index_start":0,"state_index_end_exclusive":total,
        "merged_part_ranges":[[int(p["state_index_start"]),int(p["state_index_end_exclusive"])] for p in parts],
        "summary":merged.summary(),"terminal_summary":summarize_branches(all_branches),"results":results,
    }
    report["grouped_candidate_metrics"]={}
    for kind in sorted({row.get("candidate_kind","unknown") for row in results}):
        subset=[row for row in results if row.get("candidate_kind","unknown")==kind]
        evidence=[branch for row in subset for branch in row["branch_evidence"]]
        report["grouped_candidate_metrics"][kind]={"states":len(subset),"terminal_summary":summarize_branches(evidence)}
    return merged,report
