"""Calibrate canonical Landing-entry matcher without Flight outcomes."""
from __future__ import annotations
import argparse, copy, json
from pathlib import Path
import numpy as np
from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256, load_config
from dvgc.entry import ENTRY_FEATURE_NAMES, calibrate_radius, normalized_nearest, robust_normalization

def main():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument('--certified-bank',required=True); p.add_argument('--audit-report',required=True); p.add_argument('--output-bank',required=True); p.add_argument('--config',default='configs/default.json'); a=p.parse_args()
 out=Path(a.output_bank)
 if out.exists(): raise SystemExit(f'Output exists: {out}')
 cfg=load_config(a.config); bank=SnapshotBank.load(a.certified_bank); rows=bank.records_for_phase('landing',include_training_only=False); labels=[r['final']['label'] for r in rows]; features=np.asarray([r['entry_feature'] for r in rows])
 safe=[r for r in rows if r['final']['label']=='safe']
 if len(safe)<cfg.tube_activation_min_safe: raise SystemExit('Insufficient certified Landing entries')
 center,scale=robust_normalization([r['entry_feature'] for r in safe],cfg.landing_entry_scale_floors)
 cal=calibrate_radius(features,labels,center,scale,cfg.landing_entry_minimum_calibration_precision)
 audit=json.loads(Path(a.audit_report).read_text()); audit_by={r['id']:r for r in audit['rows']}; safe_features=[r['entry_feature'] for r in safe]
 predicted=[]; truth=[]
 for row in rows:
  predicted.append(normalized_nearest(row['entry_feature'],safe_features,center,scale)[0]<=cal['radius'])
  truth.append(float(audit_by[row['id']]['audit_final'])>=float(cfg.safe_threshold))
 tp=sum(p and t for p,t in zip(predicted,truth)); fp=sum(p and not t for p,t in zip(predicted,truth)); fn=sum(not p and t for p,t in zip(predicted,truth))
 precision=tp/(tp+fp) if tp+fp else 1.; recall=tp/(tp+fn) if tp+fn else 0.
 result=copy.deepcopy(bank); result.metadata['entry_matcher']={'version':'landing_entry_task_relative_v1','feature_names':ENTRY_FEATURE_NAMES,'center':center.tolist(),'scale':scale.tolist(),'scale_floors':list(cfg.landing_entry_scale_floors),'radius':cal['radius'],'entry_window_steps':cfg.landing_entry_window_steps,'calibration_precision':cal['precision'],'calibration_recall':cal['recall'],'independent_audit_precision':precision,'independent_audit_recall':recall,'certified_bank_sha256':file_sha256(a.certified_bank),'audit_report':str(Path(a.audit_report).resolve()),'audit_seed':audit.get('seed_namespace')}
 result.metadata['entry_bank_role']='canonical_certified_landing_entry_set'; result.save(out)
 report={'status':'PASS' if precision>=cfg.landing_entry_minimum_calibration_precision else 'FAIL','states':len(rows),'safe_entries':len(safe),'labels':{k:labels.count(k) for k in ('safe','boundary','dead','unknown')},'radius':cal['radius'],'calibration_precision':cal['precision'],'calibration_recall':cal['recall'],'independent_audit_precision':precision,'independent_audit_recall':recall,'false_positives':fp,'false_negatives':fn,'bank_sha256':file_sha256(out)}
 out.with_suffix('.calibration.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
 if report['status']!='PASS': raise SystemExit(2)
if __name__=='__main__': main()
