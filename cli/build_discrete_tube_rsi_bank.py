"""Build a parent-balanced exact safe/boundary reset bank for Tube-RSI."""
from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path

from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256, load_config
from dvgc.runtime import save_json


def parent_key(row):
    return str(row.get("entry_source_id", row.get("parent_candidate_id", row["id"])))


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stable-bank",required=True);parser.add_argument("--output-bank",required=True)
    parser.add_argument("--output-report",required=True);parser.add_argument("--config",default="configs/default.json")
    args=parser.parse_args()
    if Path(args.output_bank).exists() or Path(args.output_report).exists():raise SystemExit("Tube-RSI output exists")
    cfg=load_config(args.config);source=SnapshotBank.load(args.stable_bank)
    groups={"safe":source.records_for_phase("flight",final_labels=("safe",),include_training_only=False),
            "boundary":source.records_for_phase("flight",final_labels=("boundary",),include_training_only=False)}
    if not groups["safe"] or not groups["boundary"]:raise SystemExit("Tube-RSI requires safe and boundary support")
    masses={"safe":float(cfg.discrete_tube_rsi_safe_mass),"boundary":float(cfg.discrete_tube_rsi_boundary_mass)}
    records=[];parent_mass={}
    for label,rows in groups.items():
        parents=defaultdict(list)
        for row in rows:parents[parent_key(row)].append(row)
        per_parent=masses[label]/len(parents)
        for parent,items in parents.items():
            parent_mass[f"{label}:{parent}"]=per_parent
            for row in items:
                item=copy.deepcopy(row);item.update({"reset_source":"descent_tube_rsi","origin_phase":"flight",
                    "bootstrap_group":label,"reset_parent_id":parent,"reset_weight":per_parent/len(items),
                    "original_bank_sha256":file_sha256(args.stable_bank)})
                records.append(item)
    metadata=copy.deepcopy(source.metadata);metadata.update({"bank_role":"discrete_descent_tube_rsi",
        "source_bank_sha256":file_sha256(args.stable_bank),"reset_masses":masses})
    SnapshotBank(records,metadata).save(args.output_bank)
    report={"status":"PASS","source_bank_sha256":file_sha256(args.stable_bank),
            "output_bank_sha256":file_sha256(args.output_bank),"records":len(records),
            "counts":{key:len(value) for key,value in groups.items()},"expected_reset_ratio":masses,
            "parents":{key:len({parent_key(row) for row in value}) for key,value in groups.items()},
            "parent_reset_weights":parent_mass,"dead_unknown_excluded":True}
    save_json(args.output_report,report);print(json.dumps(report,indent=2))


if __name__=="__main__":main()
