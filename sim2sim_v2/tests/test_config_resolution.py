import copy,json,sys
from pathlib import Path
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from resolved_task_config import ResolvedTaskConfig,FROZEN_SOURCE,frozen_imports
frozen_imports()
from jit_dvgc import rewards
rewards.jp=np

def spec():
 return json.loads((ROOT.parent/'docs/evidence/sim2sim_ppo_20261009/ppo/warmstart_kd5/config.json').read_text())

class ConfigTests(unittest.TestCase):
 def test_legacy_exact(self):
  r=ResolvedTaskConfig(spec());self.assertEqual(r.raw,json.loads((ROOT/'policy/resolved_config.json').read_text()));self.assertEqual(r.effective['substeps'],4)
 def test_conflict_rejected(self):
  s=spec();s['reward']['height_coeff']+=1
  with self.assertRaisesRegex(ValueError,'reward'):ResolvedTaskConfig(s)
 def test_v2_requires_explicit(self):
  s=spec();s['task_config_mode']='jump_landing_v2'
  with self.assertRaisesRegex(ValueError,'explicit'):ResolvedTaskConfig(s)
 def test_v2_effective_change(self):
  s=spec();s['task_config_mode']='jump_landing_v2';s['resolved_task_config']=ResolvedTaskConfig(spec()).raw;s['reward']['height_coeff']=7;s['resolved_task_config']['reward']=s['reward']
  self.assertEqual(ResolvedTaskConfig(s).cfg.reward.height_coeff,7)
 def test_frozen_import(self):
  self.assertTrue(Path(rewards.__file__).resolve().is_relative_to(FROZEN_SOURCE))
 def test_reward_sensitivity(self):
  base=ResolvedTaskConfig(spec());r=base.raw['reward'];state=rewards.RewardState(*[0.,0.,.42,.1,.12,.1,1.5,.1,.1,.2,.3,.2,.5,.6,1.,2.]);action=np.array([.2,.3,.4,.5]);previous=np.zeros(4)
  fields=('jump_signal','first_apex_success','illegal_contact','physical_failure_transition','roll_pitch_failure_transition','jump_zone_missed_transition','stuck_transition','yaw_limit_transition','timeout_transition')
  for name in r:
   if name=='failed_episode_return':continue # applied after reward in task; verified separately
   changed=copy.deepcopy(spec());changed['task_config_mode']='jump_landing_v2';changed['resolved_task_config']=copy.deepcopy(base.raw)
   changed['resolved_task_config']['reward'][name]=r[name]+(.01 if name in ('jump_reward_min_height','low_height_penalty_full_at','max_beneficial_height','peak_reward_height') else 1.)
   changed['reward']=changed['resolved_task_config']['reward'];other=ResolvedTaskConfig(changed)
   sensitive=False
   for height in [.1,.2,.345,.355,.42,.495,.505,.795,.805]:
    for flag in [None,*fields]:
     flags={f:np.asarray(f in ('jump_signal',flag)) for f in fields}
     inp=rewards.RewardInputs(state.replace(z=height),action,previous,**flags)
     a=rewards.phase_u_reward(inp,base.cfg.reward,base.cfg.physical_limits);b=rewards.phase_u_reward(inp,other.cfg.reward,other.cfg.physical_limits)
     # Verify real term or clipped output sensitivity; clipping can hide a correct component.
     sensitive |= not np.allclose([a.total,*a.components.values()],[b.total,*b.components.values()])
   self.assertTrue(sensitive,name)
 def test_observation_legacy_parity(self):
  import importlib.util,mujoco
  from policy_runtime import Observation
  oldpath=ROOT.parent/'docs/evidence/sim2sim_ppo_20261009/models/physx/runtime/policy_runtime.py'
  loader=importlib.util.spec_from_file_location('archived_policy_runtime',oldpath);old=importlib.util.module_from_spec(loader);loader.loader.exec_module(old)
  model=mujoco.MjModel.from_xml_path(str(ROOT/'model/source.xml'));names=[model.joint(i).name for i in range(1,model.njnt)]
  a=old.Observation(model);r=ResolvedTaskConfig(spec());b=Observation(model,r.raw,r.observation)
  np.testing.assert_array_equal(a.initial(2.6),b.initial(2.6))
  for x in [2.6,3.5,4.1,2.6]:
   args=([x,0,.4],[1,0,0,0],model.key_qpos[0,7:].copy(),np.arange(5)*.1,names,np.array([2.,.1,.2]),np.array([.1,.2,.3]),np.array([1.,2.,3.]),np.array([.1,.2,.3,.4]))
   np.testing.assert_array_equal(a.advance(*args),b.advance(*args))
 def test_unimplemented_action_semantics_rejected(self):
  from policy_runtime import control
  s=spec();s['task_config_mode']='jump_landing_v2';s['resolved_task_config']=ResolvedTaskConfig(spec()).raw
  s['resolved_task_config']['action']['joint_target_semantics']='incremental_knee'
  with self.assertRaisesRegex(ValueError,'action semantics'):ResolvedTaskConfig(s)
  with self.assertRaisesRegex(ValueError,'action semantics'):control(np.zeros(4),None,s['resolved_task_config'])
 def test_failed_return_effect(self):
  # Task's post-reward terminal replacement intentionally returns target minus accumulated return.
  s=spec();r=ResolvedTaskConfig(s);before=17.;old=r.cfg.reward.failed_episode_return-before
  s['task_config_mode']='jump_landing_v2';s['resolved_task_config']=r.raw;s['reward']['failed_episode_return']=-321;s['resolved_task_config']['reward']=s['reward']
  self.assertNotEqual(old,ResolvedTaskConfig(s).cfg.reward.failed_episode_return-before)
if __name__=='__main__':unittest.main()
