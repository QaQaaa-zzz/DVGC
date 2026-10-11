"""T0 frozen TRAIN gradient contract applied to saved T1 endpoints; zero updates."""
import json,pickle
from pathlib import Path
from jit_dvgc.retention_t0_diagnostics import SourceRegistry,load_inputs,make_predictors,train_gradients,tree_sha
r=Path('/home/qy/DVGC/JIT/runs/experiments/retention_next_20261010/T1_paired_ppo_002');out=r/'saved_review_002';p=json.loads((r/'plan.json').read_text());reg=SourceRegistry()
_,_,_,_,corpora,_=load_inputs('/home/qy/DVGC/JIT/runs/experiments/retention_next_20261010/B_keep_coverage_003',Path(p['parent']).parent,reg)
models={}
for arm in ('PPO_keep','PPO_demo_keep'):
 for step in (0,16000,32000):
  node=reg.json(r/'training'/arm/f'checkpoint_{step:06d}.json');path=reg.register(node['inference'],node['inference_sha256'])
  with path.open('rb') as f:params=pickle.load(f)
  assert tree_sha(params[1])==node['actor_sha256'];models[f'{arm}_{step}']=params
predict,production=make_predictors();corpora[2].actions=predict(models['PPO_keep_0'],corpora[2].observations,independent=True)
result=train_gradients(models,corpora,production);reg.verify_unchanged()
(out/'TRAIN_frozen_gradients.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');(out/'gradient_sources.json').write_text(json.dumps(reg.files,indent=2)+'\n')
print('6 frozen endpoints, fixed TRAIN only, zero optimizer/physics/TEST')
