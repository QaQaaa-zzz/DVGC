from __future__ import annotations
import argparse, json
from pathlib import Path
from dvgc.bank import SnapshotBank
from dvgc.viability import ViabilityEnsemble

def main():
    p=argparse.ArgumentParser(description="Fit the secondary Physical-Belief viability ensemble."); p.add_argument("--bank",required=True); p.add_argument("--output",required=True); p.add_argument("--report",required=True); p.add_argument("--members",type=int,default=5); p.add_argument("--seed",type=int,default=0); a=p.parse_args()
    bank=SnapshotBank.load(a.bank); model=ViabilityEnsemble(a.members,64,a.seed); report=model.fit(bank.records); model.save(a.output); bank.save(a.bank); Path(a.report).write_text(json.dumps(report,indent=2),encoding="utf-8"); print(json.dumps(report,indent=2))
if __name__=="__main__": main()
