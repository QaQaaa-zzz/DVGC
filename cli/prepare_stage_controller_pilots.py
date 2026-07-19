"""Prepare immutable six-state banks/configs for bounded stage controllers."""
from __future__ import annotations
import argparse,copy
from pathlib import Path
from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256,load_config,save_config
from dvgc.runtime import save_json
from cli.stage_label_pilot import evenly

OBJECTIVES={"takeoff":"takeoff_to_ascent","ascent":"ascent_to_apex","apex":"apex_to_descent"}

def main():
 p=argparse.ArgumentParser();p.add_argument('--takeoff-bank',required=True);p.add_argument('--flight-bank',required=True);p.add_argument('--output-root',required=True);p.add_argument('--config',default='configs/default.json');a=p.parse_args();root=Path(a.output_root);root.mkdir(parents=True,exist_ok=True)
 takeoff=SnapshotBank.load(a.takeoff_bank);flight=SnapshotBank.load(a.flight_bank)
 sources={"takeoff":takeoff.records_for_phase('takeoff'),"ascent":[r for r in flight.records if r.get('flight_subinterval')=='ascent'],"apex":[r for r in flight.records if r.get('flight_subinterval')=='apex']};report={"status":"PASS","artifact_role":"stage_controller_pilot_inputs","stages":{}}
 for stage,rows in sources.items():
  chosen=evenly(rows,6)
  if len(chosen)!=6:raise SystemExit(f'{stage} has {len(chosen)} pilot states')
  stage_root=root/stage;stage_root.mkdir(exist_ok=True);bank_path=stage_root/'reset_bank.pkl';config_path=stage_root/'config.json'
  metadata={"artifact_role":"proposal_support_set_stage_controller_pilot","stage":stage,"objective":OBJECTIVES[stage],"source_bank_sha256":file_sha256(a.takeoff_bank if stage=='takeoff' else a.flight_bank),"certified_tube":False}
  SnapshotBank([copy.deepcopy(r) for r in chosen],metadata).save(bank_path)
  cfg=load_config(a.config,{"training_stage":"takeoff" if stage=='takeoff' else "flight","stage_reachability_objective":OBJECTIVES[stage],"use_bank_resets":True,"domain_randomization":False,"obs_noise_enable":False,"stage_curriculum_scale":0.0});save_config(cfg,config_path)
  report['stages'][stage]={"states":6,"ids":[r['id'] for r in chosen],"reference_indices":[r.get('reference_index') for r in chosen],"bank":str(bank_path),"bank_sha256":file_sha256(bank_path),"config":str(config_path),"objective":OBJECTIVES[stage]}
 save_json(root/'inputs.json',report)
if __name__=='__main__':main()
