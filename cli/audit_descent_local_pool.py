"""Audit physical quality, diversity, and provenance of a local descent pool."""
from __future__ import annotations

import argparse,json
from collections import Counter
from pathlib import Path

import numpy as np

from cli.audit_candidates import _contact_audit,_rollout_audit
from cli.build_descent_entries import snapshot_identity
from dvgc.bank import SnapshotBank
from dvgc.config import config_hash,file_sha256,load_config
from dvgc.runtime import save_json


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--bank",required=True); p.add_argument("--output",required=True); p.add_argument("--config",default="configs/default.json"); a=p.parse_args(); out=Path(a.output)
    if out.exists(): raise SystemExit(f"Output exists: {out}")
    cfg=load_config(a.config,{"training_stage":"flight","domain_randomization":False,"obs_noise_enable":False,"use_bank_resets":False}); bank=SnapshotBank.load(a.bank); rows=bank.records_for_phase("flight",include_training_only=False); eligible=[r for r in rows if r.get("local_bootstrap_eligible")]; children=[r for r in eligible if r.get("candidate_kind")=="descent_local_proposal"]
    contact=_contact_audit(eligible,cfg.xml_path,-0.002); rollout=_rollout_audit(eligible,cfg,int(cfg.descent_local_validation_steps)); child_contact=_contact_audit(children,cfg.xml_path,-0.002); child_rollout=_rollout_audit(children,cfg,int(cfg.descent_local_validation_steps))
    base=[r for r in rows if r.get("candidate_kind")!="descent_local_proposal"]; base_ids=[snapshot_identity(r) for r in base]; child_ids=[snapshot_identity(r) for r in children]; parent_counts=Counter(r.get("parent_candidate_id") for r in children); groups=Counter(r.get("bootstrap_group") for r in eligible); layers=Counter(r.get("descent_layer") for r in eligible)
    eligible_ids=[snapshot_identity(r) for r in eligible]; flags={"all_finite":all(np.isfinite(np.asarray(r["qpos"])).all() and np.isfinite(np.asarray(r["qvel"])).all() for r in rows),"all_flight_phase":all(int(r.get("oracle_phase",-1))==2 for r in rows),"eligible_state_unique":len(eligible_ids)==len(set(eligible_ids)),"children_state_unique":len(child_ids)==len(set(child_ids)) and not (set(child_ids)&set(base_ids)),"children_no_contact":child_contact["records_with_robot_terrain_contact"]==0,"children_no_deep_penetration":child_contact["records_with_deep_penetration"]==0,"children_short_physical_failure_zero":child_rollout["short_horizon_physical_failure_rate"]==0,"children_nonfinite_zero":child_rollout["nonfinite_records"]==0,"parent_cap":max(parent_counts.values(),default=0)<=int(cfg.descent_local_max_children_per_parent),"group_support":all(groups[g]>0 for g in ("provisional_safe","boundary","successful_anchor")),"provenance_current":bank.metadata.get("xml_sha256")==file_sha256(cfg.xml_path) and bank.metadata.get("candidate_config_hash")==config_hash(cfg)}
    report={"status":"PASS" if all(flags.values()) else "FAIL","bank_sha256":file_sha256(a.bank),"records":len(rows),"eligible":len(eligible),"children":len(children),"diagnostic_base_duplicate_count":len(base_ids)-len(set(base_ids)),"groups":dict(groups),"layers":dict(layers),"unique_source_parents":len({r.get("entry_source_id") for r in eligible}),"unique_child_parents":len(parent_counts),"maximum_children_per_parent":max(parent_counts.values(),default=0),"overall_contact":contact,"overall_rollout":rollout,"child_contact":child_contact,"child_rollout":child_rollout,"quality_flags":flags,"provenance":{"xml_sha256":bank.metadata.get("xml_sha256"),"c_l_sha256":bank.metadata.get("landing_entry_set_sha256"),"source_certified_bank_sha256":bank.metadata.get("source_certified_bank_sha256"),"candidate_seed":bank.metadata.get("local_candidate_seed")}}
    save_json(out,report); print(json.dumps(report,indent=2));
    if report["status"]!="PASS": raise SystemExit(2)


if __name__=="__main__": main()
