"""Freeze and self-check the Apex->Descent entry protocol before policy use."""
from __future__ import annotations
import argparse, json
from collections import Counter
from pathlib import Path
import numpy as np
from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256, load_config
from dvgc.runtime import save_json
from dvgc.stage_reachability import evaluate_entry, protocol_payload

def main():
 p=argparse.ArgumentParser();p.add_argument('--support',required=True);p.add_argument('--output',required=True);p.add_argument('--config',default='configs/default.json');a=p.parse_args()
 cfg=load_config(a.config);bank=SnapshotBank.load(a.support);meta=dict(bank.metadata);meta['support_features']=[r['physical_feature'] for r in bank.records];protocol=protocol_payload(cfg,meta)
 valid=[];reject=Counter()
 for row in bank.records:
  sample=dict(row);sample.update(canonical_phase='flight',apex_seen=True,dual_wheel_airborne=True,physical_failure=False,nonfinite=False,prohibited_contact=False,body_terrain_contact=False,deep_penetration=False,invalid_wheel_contact=False)
  result=evaluate_entry('apex',sample,cfg,meta)
  if result['valid']:valid.append(row)
  else:reject.update(result['reasons'])
 if not valid:raise SystemExit('Reference/observed Descent support cannot pass detector')
 base=dict(valid[0]);base.update(canonical_phase='flight',apex_seen=True,dual_wheel_airborne=True,physical_failure=False,nonfinite=False,prohibited_contact=False,body_terrain_contact=False,deep_penetration=False,invalid_wheel_contact=False)
 negatives={}
 for name,updates in {'no_apex_latch':{'apex_seen':False},'rolling_fall':{'physical_feature':np.asarray(base['physical_feature']).copy()},'body_contact':{'body_terrain_contact':True},'wheel_contact':{'dual_wheel_airborne':False},'nonfinite':{'nonfinite':True}}.items():
  sample=dict(base);sample.update(updates)
  if name=='rolling_fall':sample['physical_feature'][3]=np.deg2rad(60.)
  negatives[name]=not evaluate_entry('apex',sample,cfg,meta)['valid']
 flags={'support_reference_pass':len(valid)>0,'detector_rejects_all_declared_negatives':all(negatives.values()),'protocol_hash_frozen':len(protocol['protocol_sha256'])==64}
 payload={'status':'PASS' if all(flags.values()) else 'FAIL','artifact_role':'frozen_stage_entry_protocol','protocol':protocol,'protocol_sha256':protocol['protocol_sha256'],'support_bank':str(Path(a.support).resolve()),'support_bank_sha256':file_sha256(a.support),'support_records':len(bank.records),'support_records_passing':len(valid),'support_rejection_reasons':dict(reject),'negative_controls':negatives,'quality_flags':flags,'detector_fire_semantics':'stage_entry_ever latch makes the terminal event fire once'}
 save_json(a.output,payload);print(json.dumps({k:payload[k] for k in ('status','protocol_sha256','support_records','support_records_passing','negative_controls')},indent=2));raise SystemExit(0 if payload['status']=='PASS' else 2)
if __name__=='__main__':main()
