"""Merge globally indexed C_D audit shards with strict seed uniqueness."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dvgc.certification import summarize_branches
from dvgc.runtime import save_json


def merge_reports(shards):
    keys=("seed","seed_namespace","candidate_bank_sha256","landing_entry_set_sha256","descent_policy_hash","landing_policy_hash","total_states")
    for key in keys:
        if len({json.dumps(s.get(key),sort_keys=True) for s in shards})!=1: raise ValueError(f"Audit shard {key} mismatch")
    rows=sorted((row for shard in shards for row in shard["rows"]),key=lambda row:row["candidate_index"]); indices=[r["candidate_index"] for r in rows]
    total=int(shards[0]["total_states"])
    if indices!=list(range(total)): raise ValueError(f"Audit candidate indices are not complete and unique: {indices}")
    evidence=[ev for row in rows for ev in row["branch_evidence"]]; seed_keys=[(ev["seed_namespace"],ev["branch_seed"]) for ev in evidence]
    if len(seed_keys)!=len(set(seed_keys)): raise ValueError("Audit branch seeds are not globally unique")
    report={key:shards[0][key] for key in keys}; report.update({"status":"PASS","audit_only":True,"states":len(rows),"shards":[{"start_index":s["start_index"],"end_index":s["end_index"]} for s in shards],"terminal_summary":summarize_branches(evidence),"rows":rows})
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--shard",action="append",required=True); p.add_argument("--output",required=True); a=p.parse_args(); out=Path(a.output)
    if out.exists(): raise SystemExit(f"Output exists: {out}")
    shards=[json.loads(Path(path).read_text()) for path in a.shard]
    try: report=merge_reports(shards)
    except ValueError as exc: raise SystemExit(str(exc)) from exc
    save_json(out,report); print(json.dumps({k:v for k,v in report.items() if k!="rows"},indent=2))


if __name__=="__main__": main()
