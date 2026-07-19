"""Combine independently complete Cycle-4/Cycle-5 handoff decompositions."""
from __future__ import annotations
import argparse, copy, json
from pathlib import Path
import numpy as np
from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256
from dvgc.research_semantics import summarize_handoff
from dvgc.runtime import save_json

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',action='append',required=True);p.add_argument('--proposal-bank-input',action='append',required=True);p.add_argument('--entry-bank',required=True);p.add_argument('--output',required=True);p.add_argument('--proposal-bank',required=True);p.add_argument('--dedup-distance',type=float,default=.15);a=p.parse_args()
    out=Path(a.output);bank_out=Path(a.proposal_bank)
    if out.exists() or bank_out.exists():raise SystemExit('Combined handoff output exists')
    reports=[json.loads(Path(x).read_text()) for x in a.report]
    if any(r.get('status')!='PASS' for r in reports):raise SystemExit('Input handoff report is not PASS')
    rows=[row for r in reports for row in r['rows']]
    if len({(r['policy_label'],r['event_index']) for r in rows})!=len(rows):raise SystemExit('Duplicate policy/event key')
    entry=SnapshotBank.load(a.entry_bank);matcher=entry.metadata['entry_matcher'];scale=np.asarray(matcher['scale'],np.float64)
    proposals=[copy.deepcopy(row) for path in a.proposal_bank_input for row in SnapshotBank.load(path).records];unique=[]
    for row in proposals:
        feature=np.asarray(row['entry_feature'],np.float64)
        if any(np.linalg.norm((feature-np.asarray(old['entry_feature'],np.float64))/scale)<a.dedup_distance for old in unique):continue
        unique.append(row)
    parents={row.get('entry_source_parent') or row['entry_source_id'] for row in unique}
    metadata={'artifact_role':'pending_entry_proposals','active_for_matching':False,'safe_claim_allowed':False,'entry_bank_sha256':file_sha256(a.entry_bank),'entry_matcher_radius':matcher['radius'],'entry_matcher_unchanged':True,'dedup_distance':a.dedup_distance,'proposal_count_before_dedup':len(proposals),'unique_parents':len(parents)}
    SnapshotBank(unique,metadata).save(bank_out)
    payload={'status':'PASS','artifact_role':'combined_handoff_decomposition','summary':summarize_handoff(rows),'by_policy':{r['policy_label']:r['summary'] for r in reports},'rows':rows,'events':len(rows),'pending_entry_proposals':len(unique),'pending_entry_parents':len(parents),'new_entry_extension_eligible':len(unique)>=4 and len(parents)>=2,'entry_bank_sha256':file_sha256(a.entry_bank),'entry_matcher_radius':matcher['radius'],'entry_matcher_unchanged':True,'proposal_bank':str(bank_out.resolve()),'proposal_bank_sha256':file_sha256(bank_out)}
    save_json(out,payload);print(json.dumps({k:v for k,v in payload.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
