import jax.numpy as jp
from dvgc.config import default_config
from dvgc.rewards import compute_stage_next_entry_reward

def test_event_dominates_bounded_dense_stage_reward():
 cfg=default_config();f=jp.zeros(16);a=jp.zeros(4)
 x=compute_stage_next_entry_reward(cfg=cfg,objective='apex_to_descent',feature=f,previous_feature=f,action=a,previous_action=a,next_entry=jp.asarray(True),hard_failure=jp.asarray(False),jump_latched=jp.asarray(True),window_active=jp.asarray(True),joint_energy=jp.asarray(0.0))
 assert float(x['event'])==8. and -.25<=float(x['shaping'])<=.2

def test_takeoff_success_prefers_low_rate_handoff_without_changing_event():
 cfg=default_config();a=jp.zeros(4);good=jp.zeros(16);bad=good.at[4].set(.5).at[10].set(3.5)
 def reward(feature):
  return compute_stage_next_entry_reward(cfg=cfg,objective='takeoff_to_ascent',feature=feature,previous_feature=feature,action=a,previous_action=a,next_entry=jp.asarray(True),hard_failure=jp.asarray(False),jump_latched=jp.asarray(True),window_active=jp.asarray(True),joint_energy=jp.asarray(0.0))
 g,b=reward(good),reward(bad)
 assert float(g['event'])==float(b['event'])==8.
 assert 0.<float(b['handoff_bonus'])<float(g['handoff_bonus'])<=1.
 assert float(g['reward'])>float(b['reward'])

def test_takeoff_handoff_bonus_requires_success_event():
 cfg=default_config();f=jp.zeros(16);a=jp.zeros(4)
 x=compute_stage_next_entry_reward(cfg=cfg,objective='takeoff_to_ascent',feature=f,previous_feature=f,action=a,previous_action=a,next_entry=jp.asarray(False),hard_failure=jp.asarray(False),jump_latched=jp.asarray(True),window_active=jp.asarray(True),joint_energy=jp.asarray(0.0))
 assert float(x['handoff_bonus'])==0.

def test_ascent_rewards_upward_height_progress():
 cfg=default_config();p=jp.zeros(16);up=p.at[2].set(.05);down=p.at[2].set(-.05);a=jp.zeros(4)
 def r(f):return compute_stage_next_entry_reward(cfg=cfg,objective='ascent_to_apex',feature=f,previous_feature=p,action=a,previous_action=a,next_entry=jp.asarray(False),hard_failure=jp.asarray(False),jump_latched=jp.asarray(True),window_active=jp.asarray(True),joint_energy=jp.asarray(0.0))['reward']
 assert float(r(up))>float(r(down))

def test_apex_support_potential_has_no_positive_stall_loop():
 cfg=default_config();f=jp.zeros(16);a=jp.zeros(4)
 def progress(cur,prev):
  return compute_stage_next_entry_reward(cfg=cfg,objective='apex_to_descent',feature=f,previous_feature=f,action=a,previous_action=a,next_entry=jp.asarray(False),hard_failure=jp.asarray(False),jump_latched=jp.asarray(True),window_active=jp.asarray(True),joint_energy=jp.asarray(0.0),current_support_distance=jp.asarray(cur),previous_support_distance=jp.asarray(prev))['progress']
 assert float(progress(1.,2.))>0
 assert float(progress(2.,1.))<0
 assert float(progress(1.,1.))<=.01
