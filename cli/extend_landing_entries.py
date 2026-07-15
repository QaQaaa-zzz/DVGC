"""Extend canonical C_L with independently certified Final-safe proposals."""
from __future__ import annotations
import argparse,copy,hashlib,json
from pathlib import Path
from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256

def main():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument('--base-bank',required=True); p.add_argument('--certified-proposals',required=True); p.add_argument('--output-bank',required=True); a=p.parse_args()
 out=Path(a.output_bank)
 if out.exists(): raise SystemExit(f'Output exists: {out}')
 base=SnapshotBank.load(a.base_bank); extension=SnapshotBank.load(a.certified_proposals)
 matcher=copy.deepcopy(base.metadata['entry_matcher']); safe=[copy.deepcopy(r) for r in extension.records if r['final']['label']=='safe' and not r.get('training_only',False)]
 if not safe: raise SystemExit('No independently certified Final-safe extension entries')
 records=copy.deepcopy(base.records)+safe; digest=hashlib.sha256(''.join(sorted(r['id'] for r in records if r['final']['label']=='safe')).encode()).hexdigest()[:12]; version=f'landing-entry-{digest}'
 for row in records:
  if int(row['final']['branches'])>0:
   row['source_tube_version']=row.get('tube_version'); row['tube_version']=version
 metadata=copy.deepcopy(base.metadata); metadata.update({'entry_bank_role':'canonical_certified_landing_entry_set_extended','entry_set_version':version,'last_policy_version':base.metadata['last_policy_version'],'last_tube_version':version,'base_bank':str(Path(a.base_bank).resolve()),'base_bank_sha256':file_sha256(a.base_bank),'extension_bank':str(Path(a.certified_proposals).resolve()),'extension_bank_sha256':file_sha256(a.certified_proposals),'extension_safe_count':len(safe),'entry_matcher':matcher})
 result=SnapshotBank(records,metadata); result.save(out)
 reloaded=SnapshotBank.load(out)
 if reloaded.metadata['entry_matcher']!=base.metadata['entry_matcher']: raise SystemExit('Matcher changed during entry extension')
 report={'status':'PASS','entry_set_version':version,'base_states':len(base.records),'extension_safe_count':len(safe),'states':len(records),'final_safe_count':sum(r['final']['label']=='safe' for r in records),'matcher_radius':matcher['radius'],'matcher_unchanged':True,'bank_sha256':file_sha256(out)}
 out.with_suffix('.extension.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
