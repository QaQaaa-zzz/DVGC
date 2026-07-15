"""Evaluate a frozen expert stack on a fixed candidate bank."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256
from dvgc.expert_training import evaluate_flight_composite
from dvgc.experts import StageExpertRegistry
from dvgc.policy import load_bundle
from dvgc.runtime import save_json

def main():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument("--registry",required=True); p.add_argument("--bank",required=True); p.add_argument("--entry-set",required=True); p.add_argument("--output",required=True); p.add_argument("--seed",type=int,default=8100000); a=p.parse_args(); out=Path(a.output)
 if out.exists(): raise SystemExit(f"Output exists: {out}")
 registry=StageExpertRegistry.load(a.registry); flight=registry.specs["flight"]; landing=registry.specs["landing"]
 if file_sha256(a.entry_set)!=flight.downstream_entry_set_sha256: raise SystemExit("Entry-set provenance mismatch")
 fp,fc,_=load_bundle(flight.checkpoint_path,verify_files=True); lp,_,_=load_bundle(landing.checkpoint_path,verify_files=True); bank=SnapshotBank.load(a.bank)
 report=evaluate_flight_composite(fp,fc,lp,bank.records_for_phase("flight",include_training_only=False),a.entry_set,seed=a.seed,controller_stack_hash=flight.controller_stack_hash)
 report.update({"policy_hashes":{"flight":flight.policy_hash,"landing":landing.policy_hash},"candidate_bank_sha256":file_sha256(a.bank),"entry_set_sha256":file_sha256(a.entry_set),"registry_hash":registry.registry_hash,"runtime_source_fingerprint":registry.runtime_source_fingerprint,"evaluation_seed":a.seed})
 save_json(out,report); print(json.dumps({k:v for k,v in report.items() if k!="rows"},indent=2))

if __name__=="__main__": main()
