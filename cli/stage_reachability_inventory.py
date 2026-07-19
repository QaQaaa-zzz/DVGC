"""Build a deduplicated inventory for next-stage reachability without replay."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256, load_config
from dvgc.runtime import save_json
from dvgc.stage_reachability import protocol_payload

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BANKS = (
    ("flight_candidates", "artifacts/flight_candidates_augmented_v1.pkl"),
    ("landing_candidates", "artifacts/landing_candidates.naconmax512_v1.pkl"),
    ("landing_tube", "artifacts/landing_tube.pkl"),
    ("descent_unique", "runs/stage_experts/descent_tube_seed0_20260716T2330/round_3/frozen/D_all_unique.pkl"),
    ("trajectory_mined", "runs/stage_experts/trajectory_mining_resume_seed0_20260718T194327/trajectory_mining_corrected/candidate_pool.pkl"),
    ("handoff_selected", "runs/jump_envelope_seed0_20260719/handoff/fast_route/h1_selected.pkl"),
)
EXISTING_REPORTS = (
    "runs/stage_experts/trajectory_mining_resume_seed0_20260718T194327/cycle_4/stable/report.json",
    "runs/stage_experts/trajectory_mining_resume_seed0_20260718T194327/cycle_5/stable/report.json",
)


def state_hash(row: dict[str, Any]) -> str:
    h=hashlib.sha256()
    for key in ("qpos","qvel","ctrl","qacc_warmstart"):
        value=np.ascontiguousarray(np.asarray(row.get(key,[]),np.float32))
        h.update(key.encode());h.update(value.shape.__repr__().encode());h.update(value.tobytes())
    return h.hexdigest()


def infer_stage(row: dict[str, Any], source: str) -> str:
    sub=str(row.get("flight_subinterval","")).lower()
    if sub in ("ascent","apex","descent"):
        return sub
    if "descent" in source or row.get("descent_layer"):
        return "descent"
    phase=str(row.get("source_phase","")).lower()
    if phase=="takeoff": return "takeoff"
    if phase=="landing": return "landing"
    if phase=="flight": return "descent" if row.get("had_valid_landing") else "unknown_flight"
    return "unknown"


def coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "unique_states":len(rows),
        "reference_indices":len({str(r.get("reference_index",r.get("entry_source_reference_index"))) for r in rows if r.get("reference_index",r.get("entry_source_reference_index")) is not None}),
        "trajectory_parents":len({str(r.get("trajectory_parent_id",r.get("parent_candidate_id",r.get("entry_source_parent")))) for r in rows if r.get("trajectory_parent_id",r.get("parent_candidate_id",r.get("entry_source_parent"))) is not None}),
        "dynamics_variants":sorted({str(r.get("dynamics_variant",r.get("entry_source_dynamics_variant"))) for r in rows if r.get("dynamics_variant",r.get("entry_source_dynamics_variant")) is not None}),
        "candidate_kinds":dict(Counter(str(r.get("candidate_kind","missing")) for r in rows)),
    }


def run(output: Path, relabel_output: Path, config: str="configs/default.json") -> tuple[dict,dict]:
    cfg=load_config(config); protocol=protocol_payload(cfg)
    by_stage: dict[str,dict[str,dict]]=defaultdict(dict);sources=[];duplicates=Counter()
    for name,relative in DEFAULT_BANKS:
        path=ROOT/relative
        if not path.exists():
            sources.append({"name":name,"path":relative,"exists":False});continue
        bank=SnapshotBank.load(path);counts=Counter()
        for row in bank.records:
            stage=infer_stage(row,name);digest=state_hash(row);counts[stage]+=1
            if digest in by_stage[stage]:
                duplicates[stage]+=1
                existing=by_stage[stage][digest]
                # A later certified view of the same immutable state enriches
                # labels/provenance; it does not create another training row.
                if row.get("final",{}).get("label") in ("safe","boundary","dead"):
                    existing["final"]=row["final"]
                    existing["certified_inventory_source"]=name
                continue
            item=dict(row);item["state_byte_hash"]=digest;item["inventory_source"]=name
            by_stage[stage][digest]=item
        sources.append({"name":name,"path":relative,"exists":True,"sha256":file_sha256(path),"records":len(bank.records),"stage_counts":dict(counts),"artifact_role":bank.metadata.get("artifact_role",bank.metadata.get("entry_bank_role"))})
    # Existing composite reports can supply branch outcomes but not valid
    # next-entry event snapshots/time. They are reusable controller evidence,
    # never silently converted to next-stage labels.
    report_evidence=[]
    for relative in EXISTING_REPORTS:
        path=ROOT/relative
        if not path.exists():continue
        data=json.loads(path.read_text());rows=data.get("rows",[]);branches=sum(len(r.get("branch_evidence",[])) for r in rows)
        fields=set().union(*(set(e) for r in rows for e in r.get("branch_evidence",[]))) if rows else set()
        directly_reusable={"next_stage_reached","entry_snapshot","time_to_next_stage"}.issubset(fields)
        report_evidence.append({"path":relative,"sha256":file_sha256(path),"states":len(rows),"branches":branches,"branch_fields":sorted(fields),"direct_next_stage_labels":directly_reusable,"reuse":"controller_outcome_only" if not directly_reusable else "direct"})
    stages={stage:coverage(list(by_stage.get(stage,{}).values())) for stage in ("takeoff","ascent","apex","descent","landing")}
    reference_counts={}
    reference_csv=ROOT/"docs/reference_phase_envelopes.csv"
    if reference_csv.exists():
        import csv
        for row in csv.DictReader(reference_csv.open()): reference_counts[row["phase"]]=int(row["count"])
    inventory={"status":"PASS","artifact_role":"stage_reachability_coverage_inventory","protocol_sha256":protocol["protocol_sha256"],"sources":sources,"stages":stages,"cross_source_duplicates_by_stage":dict(duplicates),"reference_rows_not_snapshots":reference_counts,"existing_branch_reports":report_evidence,
               "interpretation":"Branch count is evidence cost, not unique-state coverage. Missing next-entry snapshots/time require minimal replay only.",
               "proposal_support_set_semantics":"physical legal training/acquisition support; never certified safe","certified_tube_semantics":"frozen-policy high-confidence branch certification plus independent audit"}
    # Landing Final labels are the only existing labels that have exactly the
    # new successor semantics (Landing -> Stable). Other stages stay unknown
    # until an event-aligned next-entry detector is replayed.
    label_counts={stage:Counter() for stage in stages}
    records=[]
    for stage,items in by_stage.items():
        if stage not in label_counts:continue
        for digest,row in items.items():
            final=row.get("final",{});old=final.get("label") if stage=="landing" else None
            label={"safe":"high_confidence_positive","boundary":"boundary","dead":"negative_under_current_controller_bank","unknown":"unknown"}.get(old,"unknown")
            label_counts[stage][label]+=1
            records.append({"state_byte_hash":digest,"candidate_id":row.get("id"),"stage":stage,"next_stage":{"takeoff":"ascent","ascent":"apex","apex":"descent","descent":"landing","landing":"stable"}[stage],"label":label,"label_reuse":"direct_existing_final" if stage=="landing" and old else "requires_event_aligned_rollout","reference_index":row.get("reference_index",row.get("entry_source_reference_index")),"trajectory_parent":row.get("trajectory_parent_id",row.get("parent_candidate_id")),"source":row["inventory_source"]})
    relabel={"status":"PASS","artifact_role":"existing_data_stage_relabel_plan","protocol_sha256":protocol["protocol_sha256"],"label_counts":{k:dict(v) for k,v in label_counts.items()},"records":records,"missing_fields_requiring_minimal_replay":["next_stage_reached","time_to_next_stage","entry_snapshot","entry_quality"],"old_final_labels_not_reused_for_intermediate_stages":True}
    save_json(output,inventory);save_json(relabel_output,relabel);return inventory,relabel


def main():
    p=argparse.ArgumentParser();p.add_argument("--output",required=True);p.add_argument("--relabel-output",required=True);p.add_argument("--config",default="configs/default.json");a=p.parse_args()
    run(Path(a.output),Path(a.relabel_output),a.config)

if __name__=="__main__":main()
