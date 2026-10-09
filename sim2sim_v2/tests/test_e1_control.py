import unittest,sys,json,tempfile,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from e1_training_control import authorized_spec,resolve_learner_ppo,physical_rank
from resolved_task_config import ResolvedTaskConfig
from train_physx_brax import checkpoint_schedule
ROOT=Path(__file__).resolve().parents[1]
class E1(unittest.TestCase):
 def test_config_axes_and_budget(self):
  s=json.loads((ROOT/'configs/e1_stable_learner.json').read_text());old=json.loads(Path(s['e1_scope']['baseline_config']).read_text())
  for k in ['reward','reset','events','action','physical_limits','physics_dt','control_dt','target_drive_gains','contact_required']:
   self.assertEqual(s.get(k),old.get(k),k)
  p=resolve_learner_ppo(s);self.assertEqual(p['num_parallel_envs']*p['unroll_length']*20,491520);self.assertEqual(p['learning_rate'],3e-5)
  self.assertEqual(ResolvedTaskConfig(s).sha256,s['expected_resolved_task_sha256'])
  self.assertEqual(checkpoint_schedule(s,491520),{0,245760,491520})
  self.assertFalse(s['launch_ready'])
 def test_conflict_rejected(self):
  s=json.loads((ROOT/'configs/e1_stable_learner.json').read_text());s['learner_overrides']['learning_rate']=9
  with self.assertRaises(ValueError):resolve_learner_ppo(s)
 def test_authorization_required(self):
  cfg=ROOT/'configs/e1_stable_learner.json'
  with tempfile.TemporaryDirectory() as d:
   gate=Path(d)/'gate.json';gate.write_text('{}')
   with self.assertRaises(ValueError):authorized_spec(cfg,gate)
   gate.write_text(json.dumps({'config_sha256':hashlib.sha256(cfg.read_bytes()).hexdigest(),'P0_complete':True,'P1_complete':True,'root_authorized_for_E1':True}))
   self.assertTrue(authorized_spec(cfg,gate)['launch_ready'])
 def test_selection_excludes_return(self):
  good=[dict(stable_landing=True,stable_hold_s=1,takeoff_s=.5,return_=-1000)]
  bad=[dict(stable_landing=False,stable_hold_s=0,takeoff_s=.5,return_=100000)]
  self.assertGreater(physical_rank(good),physical_rank(bad))
if __name__=='__main__':unittest.main()
