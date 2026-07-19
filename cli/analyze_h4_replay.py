"""Compare H4 events with an exact formal-code replay of their candidates."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from collections import Counter
from dvgc.config import file_sha256
from dvgc.runtime import save_json

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--handoff-shard',action='append',required=True);p.add_argument('--formal-replay',required=True);p.add_argument('--source-stable-report',required=True);p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output)
 if out.exists():raise SystemExit('H4 analysis output exists')
 handoff=[r for path in a.handoff_shard for r in json.loads(Path(path).read_text())['rows'] if r['handoff_class']=='H4']
 replay=json.loads(Path(a.formal_replay).read_text());by_key={(r['id'],b['branch_index']):b for r in replay['rows'] for b in r['branch_evidence']}
 source=json.loads(Path(a.source_stable_report).read_text());original={(r['id'],b['branch_index']):b for r in source['rows'] for b in r['branch_evidence']}
 rows=[]
 for row in handoff:
  key=(row['candidate_id'],row['branch_index']);old=original[key];fresh=by_key[key]
  exact=all(old.get(k)==fresh.get(k) for k in ('terminal_cause','end_code','end_reason','raw_composite_final_recovery','chain_success','final_recovery','steps'))
  rows.append({'candidate_id':key[0],'candidate_index':row['candidate_index'],'branch_index':key[1],'formal_branch_seed':row['formal_branch_seed'],'original_terminal':old['terminal_cause'],'formal_replay_terminal':fresh['terminal_cause'],'instrumented_terminal':row['termination_reason'],'formal_replay_exact':exact,'instrumentation_reproduced_formal':row['replay_final']==bool(fresh.get('raw_composite_final_recovery')) and row['replay_chain']==bool(fresh.get('chain_success'))})
 payload={'status':'PASS','artifact_role':'h4_exact_replay_analysis','h4_branches':len(rows),'formal_replay_exact':sum(r['formal_replay_exact'] for r in rows),'instrumentation_exact':sum(r['instrumentation_reproduced_formal'] for r in rows),'classification':('instrumentation_divergence' if all(r['formal_replay_exact'] for r in rows) and not all(r['instrumentation_reproduced_formal'] for r in rows) else 'formal_nondeterminism' if not all(r['formal_replay_exact'] for r in rows) else 'event_semantics'),'terminal_pairs':dict(Counter(f"{r['original_terminal']}->{r['formal_replay_terminal']}" for r in rows)),'formal_replay_sha256':file_sha256(a.formal_replay),'rows':rows}
 save_json(out,payload);print(json.dumps({k:v for k,v in payload.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
