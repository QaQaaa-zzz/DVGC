"""Build a provenance-keyed, label-free proposal index for backward Tube search."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

from dvgc.bank import SnapshotBank
from dvgc.config import ACTION_MAPPING_VERSION, file_sha256, load_config
from dvgc.entry import entry_feature_from_physical, normalized_nearest
from dvgc.observation_audit import array_sha256
from dvgc.runtime import save_json


C_L = Path("runs/stage_experts/flight_seed0_20260715T2045/bridge_recovery/entry_set_bridge.pkl")
SUPPORT = Path("runs/stage_next_bootstrap_seed0_20260720/support_v2/descent_proposal_support_v1.pkl")
DESCENT = Path("runs/mjx_continuous_pipeline_repair_v1/descent_candidate_bank_v1/descent_candidates_v2.pkl")
V4 = Path("runs/v4_current_frame_independent_reconstruction_localization_v1/timing_explicit_snapshots.pkl")
AUTHORITY = Path("runs/unified_descent_feedback_teacher_support_and_representation_probe_v1_replay_corrected/local_cem_authority_results.json")
TRANSFER = Path("runs/unified_descent_timing_explicit_packet_delay_reaudit_v1/correction_transfer_packet_delay_results.json")
RUNTIME_RANKING = Path("runs/stage_next_reset_v3_seed0_20260723/apex/interface_v5/descent_support_runtime_audit.json")
EXPECTED = {
    C_L: "185164da04380291946d02ff6556f867dfd8532f409fcab5f8c61873de82aa41",
    SUPPORT: "1c39f9e3c29bfa1aa2da523cfd5c2a16847bc05a6d94d813d64c456f6e744dde",
    DESCENT: "8e6342bc0d9d5e7821929f6ddccd4fb5b7c23923c66ab2d058309249dfed45a1",
    V4: "9bd00d3e8e06f440e2338c7a5e1c61ac2e376f0712c960393b8ae7738798ae28",
    AUTHORITY: "9cf20b84c79a71994b07bdd9060881bad430329f2fed83618319afcecd0c2b07",
    RUNTIME_RANKING: "1ab3c395e83e44366e66a2c7b987d3354782be3bfcd5adaba09ba09280060df4",
}


def state_hash(qpos, qvel, ctrl, qacc) -> str:
    digest = hashlib.sha256()
    for name, value in (("qpos",qpos),("qvel",qvel),("ctrl",ctrl),("qacc_warmstart",qacc)):
        a=np.ascontiguousarray(value);digest.update(name.encode());digest.update(str(a.dtype).encode());digest.update(str(a.shape).encode());digest.update(a.tobytes())
    return digest.hexdigest()


def _row(source: Path, source_index: int, record: dict, feature, cfg, entries, downstream_ids, center, scale,
         historical: dict, correction: dict | None, snapshot_hash: str | None = None,
         runtime_ranking: dict | None = None):
    valid=bool(record.get("had_valid_landing",False)); support=bool(record.get("contact_age",0)>0)
    entry=entry_feature_from_physical(feature,valid_landing=valid,support=support,contact_age=int(record.get("contact_age",0)),cfg=cfg)
    distance,nearest,contribution=normalized_nearest(entry,entries,center,scale)
    qpos=record["qpos"];qvel=record["qvel"];ctrl=record.get("ctrl",np.zeros(4,np.float32));qacc=record.get("qacc_warmstart",np.zeros_like(qvel))
    identifier=str(record.get("id",f"v4-{source_index}")); hist=historical.get(source_index,{})
    candidate = (record.get("candidate_id") or record.get("origin_parent")
                 or record.get("source_parent_id") or identifier)
    return {
        "proposal_id": hashlib.sha256(f"{source}:{identifier}:{source_index}".encode()).hexdigest()[:32],
        "phase": "descent", "candidate_id": str(candidate),
        "source_record_id": identifier,
        "tick": int(record.get("tick",record.get("event_relative_tick",record.get("reference_index",source_index)))),
        "physical_state_sha256": state_hash(qpos,qvel,ctrl,qacc),
        "snapshot_sha256": snapshot_hash or str(record.get("state_byte_hash",state_hash(qpos,qvel,ctrl,qacc))),
        "source_artifact": str(source), "source_artifact_sha256": file_sha256(source), "source_index": int(source_index),
        "height": float(feature[2]), "vertical_velocity": float(feature[8]), "pitch": float(feature[4]), "roll": float(feature[3]),
        "angular_velocity": np.asarray(feature[9:12]).tolist(),
        "contact_state": {"had_valid_landing":valid,"contact_age":int(record.get("contact_age",0)),"invalid_wheel_count":int(record.get("invalid_wheel_count",0))},
        "nearest_downstream_node_id": str(downstream_ids[nearest]),
        "distance_to_nearest_downstream_safe_node": distance,
        "distance_contributions": np.asarray(contribution).tolist(),
        "historical_best_survival": hist.get("survival"), "historical_landing_entry": hist.get("landing_entry"),
        "historical_full_chain_successes_for_ranking_only": (runtime_ranking or {}).get(identifier,0),
        "available_correction": correction,
        "labels_inherited": False,
    }


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",required=True);args=parser.parse_args()
    for path,expected in EXPECTED.items():
        if file_sha256(path)!=expected: raise SystemExit(f"asset hash mismatch: {path}")
    cfg=load_config("configs/unified_descent_rsi_learnability_pilot_v1.json")
    if file_sha256(cfg.xml_path)!="d7e9f43ff8fb9e4571203f81062ce9c828acfa38692ee8c71a3e5daa15ce794c" or ACTION_MAPPING_VERSION!=cfg.action_mapping_version:
        raise SystemExit("runtime provenance mismatch")
    downstream=SnapshotBank.load(C_L);safe=[row for row in downstream.records if row["final"]["label"]=="safe"]
    entries=np.asarray([row["entry_feature"] for row in safe],np.float64);downstream_ids=[row["id"] for row in safe];matcher=downstream.metadata["entry_matcher"]
    center=np.asarray(matcher["center"]);scale=np.asarray(matcher["scale"])
    authority=json.loads(AUTHORITY.read_text()); corrections={int(row["snapshot_index"]):row for row in authority["rows"] if row["authoritative_correction"]}
    transfer=json.loads(TRANSFER.read_text())["modes"]["L0"]["pairs"]
    runtime_rows=json.loads(RUNTIME_RANKING.read_text())["rows"]
    runtime_ranking={row["candidate_id"]:sum(bool(branch["final_landing_recovery"]) for branch in row["branches"]) for row in runtime_rows}
    historical={}
    for row in transfer:
        target=int(row["target_snapshot_index"]); old=historical.get(target)
        if old is None or int(row["survival"])>int(old["survival"]): historical[target]={"survival":int(row["survival"]),"landing_entry":bool(row["landing_entry"])}
    proposals=[]
    for path in (SUPPORT,DESCENT):
        bank=SnapshotBank.load(path)
        for index,record in enumerate(bank.records):
            proposals.append(_row(path,index,record,np.asarray(record["physical_feature"]),cfg,entries,downstream_ids,center,scale,{},None,runtime_ranking=runtime_ranking))
    captured=pickle.loads(V4.read_bytes())
    for index,item in enumerate(captured):
        record=dict(item["snapshot_v4"]);record.update({"candidate_id":item["candidate_id"],"tick":item["tick"]})
        correction=None
        if index in corrections:
            correction={"artifact":str(AUTHORITY),"snapshot_index":index,"authority_row_sha256":hashlib.sha256(json.dumps(corrections[index],sort_keys=True).encode()).hexdigest(),"original_cem_residual_sha256":array_sha256(np.asarray(item["original_cem_residual"],np.float32))}
        proposals.append(_row(V4,index,record,np.asarray(record["physical_feature"]),cfg,entries,downstream_ids,center,scale,historical,correction,item["snapshot_hash"]))
    unique={}
    for proposal in sorted(proposals,key=lambda row:(row["distance_to_nearest_downstream_safe_node"],row["proposal_id"])):
        unique.setdefault(proposal["physical_state_sha256"],proposal)
    rows=list(unique.values()); distances=np.asarray([x["distance_to_nearest_downstream_safe_node"] for x in rows]);quantiles=np.quantile(distances,[.25,.5,.75])
    for row in rows:
        distance=row["distance_to_nearest_downstream_safe_node"]
        row["shell_layer"]=1+sum(distance>value for value in quantiles)
        row["region"]="late" if row["shell_layer"]==1 else "middle" if row["shell_layer"] in (2,3) else "early"
        row["search_priority"]=[-int(row["historical_full_chain_successes_for_ranking_only"]),distance,-int(bool(row["historical_landing_entry"])),-int(row["historical_best_survival"] or 0)]
    rows.sort(key=lambda row:(row["search_priority"],row["proposal_id"]))
    output=Path(args.output)
    save_json(output,{"status":"PASS","artifact_role":"backward_tube_proposal_index","safe_labels_inherited":False,"canonical_C_L_safe_count":len(safe),"canonical_C_L_sha256":file_sha256(C_L),"proposal_count":len(rows),"source_counts":{str(path):sum(x["source_artifact"]==str(path) for x in rows) for path in (SUPPORT,DESCENT,V4)},"shell_counts":{str(i):sum(x["shell_layer"]==i for x in rows) for i in range(1,5)},"rows":rows,"provenance":{"xml_sha256":file_sha256(cfg.xml_path),"action_mapping_version":ACTION_MAPPING_VERSION,"authority_sha256":file_sha256(AUTHORITY),"transfer_sha256":file_sha256(TRANSFER)}})
    print(json.dumps({"proposals":len(rows),"C_L_safe":len(safe),"output_sha256":file_sha256(output)},indent=2))


if __name__=="__main__":main()
