"""Create a fixed-budget re-labelling plan from label age and uncertainty."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from dvgc.bank import SnapshotBank

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--bank",required=True); p.add_argument("--phase",required=True); p.add_argument("--budget",type=int,default=64); p.add_argument("--policy-kl",type=float,default=0.0); p.add_argument("--eval-drop",type=float,default=0.0); p.add_argument("--max-age",type=int,default=4); p.add_argument("--output",required=True); a=p.parse_args()
    bank=SnapshotBank.load(a.bank); rows=bank.records_for_phase(a.phase,include_training_only=False); trigger=(a.policy_kl>.08 or a.eval_drop>.08 or any(int(r.get("label_age",0))>=a.max_age for r in rows)); selected=bank.prioritized_for_relabel(a.phase,a.budget) if trigger else []
    report={"triggered":trigger,"policy_kl":a.policy_kl,"eval_drop":a.eval_drop,"selected_ids":[r["id"] for r in selected],"priority_inputs":["final Beta width","boundary/unknown","model uncertainty","label age","reset use","connection state"]}; Path(a.output).write_text(json.dumps(report,indent=2),encoding="utf-8"); print(json.dumps(report,indent=2))
if __name__=="__main__": main()
