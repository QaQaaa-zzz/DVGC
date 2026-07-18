"""Build the fixed parent-balanced reset bank for the single roll-targeted block."""
from __future__ import annotations

import argparse,copy,json
from pathlib import Path

from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256,load_config
from dvgc.descent_local import build_candidate_bootstrap_bank
from dvgc.runtime import save_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--stable-bank",required=True);p.add_argument("--output-bank",required=True)
    p.add_argument("--output-report",required=True);p.add_argument("--config",default="configs/default.json");a=p.parse_args()
    if Path(a.output_bank).exists() or Path(a.output_report).exists():raise SystemExit("Targeted reset output exists")
    source=SnapshotBank.load(a.stable_bank);records=[]
    for row in source.records_for_phase("flight",include_training_only=False):
        group=None
        if row.get("stable_safe"):group="provisional_safe"
        elif row["final"]["label"]=="boundary":group="boundary"
        elif row.get("candidate_kind")=="successful_trajectory_snapshot":group="successful_anchor"
        if group:
            item=copy.deepcopy(row);item.update({"bootstrap_group":group,"local_bootstrap_eligible":True});records.append(item)
    bank=SnapshotBank(records,copy.deepcopy(source.metadata));training,report=build_candidate_bootstrap_bank(bank,a.stable_bank,load_config(a.config))
    training.save(a.output_bank);report.update({"status":"PASS","output_bank_sha256":file_sha256(a.output_bank),
        "roll_targeted_single_block":True,"dead_as_positive_rsi":False})
    save_json(a.output_report,report);print(json.dumps(report,indent=2))


if __name__=="__main__":main()
