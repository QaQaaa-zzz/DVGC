"""Calibrate C_D matcher from certified descent support and independent audit."""
from __future__ import annotations

import argparse, copy, json
from pathlib import Path

from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256, load_config
from dvgc.descent_entry import DESCENT_ENTRY_FEATURE_NAMES, matcher_audit
from dvgc.entry import calibrate_radius, robust_normalization
from cli.build_descent_entries import snapshot_identity


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--certified-bank",required=True); p.add_argument("--audit-report",required=True); p.add_argument("--output-bank",required=True); p.add_argument("--config",default="configs/default.json"); a=p.parse_args(); out=Path(a.output_bank)
    if out.exists(): raise SystemExit(f"Output exists: {out}")
    cfg=load_config(a.config); bank=SnapshotBank.load(a.certified_bank); raw_rows=bank.records_for_phase("flight",include_training_only=False); unique={}
    for row in raw_rows: unique.setdefault(snapshot_identity(row),row)
    rows=list(unique.values()); safe=[r for r in rows if r["final"]["label"]=="safe"]
    if len(safe)<int(cfg.tube_activation_min_safe): raise SystemExit(f"Insufficient C_D safe entries: {len(safe)}")
    labels=[r["final"]["label"] for r in rows]; features=[r["entry_feature"] for r in rows]; center,scale=robust_normalization([r["entry_feature"] for r in safe],cfg.descent_entry_scale_floors)
    cal=calibrate_radius(features,labels,center,scale,float(cfg.descent_entry_minimum_calibration_precision)); audit=json.loads(Path(a.audit_report).read_text()); audit_by={r["id"]:r for r in audit["rows"]}
    truth=[float(audit_by[r["id"]]["final_rate"])>=float(cfg.safe_threshold) for r in rows]; matcher={"version":"descent_entry_task_relative_v1","feature_names":DESCENT_ENTRY_FEATURE_NAMES,"center":center.tolist(),"scale":scale.tolist(),"scale_floors":list(cfg.descent_entry_scale_floors),"radius":cal["radius"],"calibration_precision":cal["precision"],"calibration_recall":cal["recall"],"certified_bank_sha256":file_sha256(a.certified_bank),"audit_report":str(Path(a.audit_report).resolve()),"audit_seed":audit["seed_namespace"]}
    independent=matcher_audit(rows,safe,matcher,truth); matcher.update({f"independent_audit_{k}":v for k,v in independent.items()})
    result=SnapshotBank(copy.deepcopy(rows),copy.deepcopy(bank.metadata)); result.metadata.update({"entry_bank_role":"canonical_descent_handoff_set","entry_set_version":bank.metadata["last_tube_version"],"entry_matcher":matcher,"deduplicated_from_records":len(raw_rows)}); result.save(out)
    report={"status":"PASS" if independent["precision"]>=float(cfg.descent_entry_minimum_calibration_precision) else "FAIL","states":len(rows),"deduplicated_from_records":len(raw_rows),"safe_entries":len(safe),"labels":{k:labels.count(k) for k in ("safe","boundary","dead","unknown")},"radius":cal["radius"],"calibration_precision":cal["precision"],"calibration_recall":cal["recall"],"independent_audit":independent,"bank_sha256":file_sha256(out)}; out.with_suffix(".calibration.json").write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
    if report["status"]!="PASS": raise SystemExit(2)


if __name__=="__main__": main()
