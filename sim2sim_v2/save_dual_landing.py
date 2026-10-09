"""Freeze the verified checkpoint and its paired evaluation evidence for delivery."""
from pathlib import Path
import json,hashlib,shutil
ROOT=Path(__file__).resolve().parent
source=ROOT/'runs/no_speed_dual_landing';out=ROOT/'runs/deliveries/dual_landing_01966080_20261009'
out.mkdir(parents=True,exist_ok=False)
id=json.loads((source/'paired_identity.json').read_text());checkpoint=Path(id['actor']).parent
shutil.copytree(checkpoint,out/'checkpoint')
shutil.copytree(ROOT/'model',out/'physical_models/physx_conversion_source')
xml=Path('/home/qy/DVGC/assets/orange_bike_4kg_horizontal.xml')
expected=json.loads((source/'mjx_warm01966080/runtime.json').read_text())['model_sha256']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(xml)==expected
shutil.copytree(xml.parent,out/'physical_models/mjx_assets')
for engine,name in [('mjx','mjx_warm01966080'),('physx','physx_warm01966080')]:shutil.copytree(source/name,out/'evidence'/engine)
for name in ['paired_identity.json','paired_results.csv','paired_landing.png','REPORT.md']:shutil.copy2(source/name,out/name)
# Original report's relative evidence paths belong to its original source folder.
(out/'SOURCE_REPORT_PATH.txt').write_text(str(source/'REPORT.md')+'\n')
(out/'REPORT.md').unlink()
manifest=dict(checkpoint='transition_01966080',actor_sha256=id['sha256'],source_checkpoint=str(checkpoint),source_report=str(source/'REPORT.md'),purpose='same frozen actor in nominal MJX-Warp and calibrated PhysX, natural landing; no minimum forward speed',source_config_evaluation_overrides={'sim2sim_randomization':None,'sim2sim_low_speed_failure':None},sim_dt=.005,ctrl_dt=.02,files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()})
assert sha(out/'checkpoint/actor.npz')==id['sha256'];(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
print(out)
