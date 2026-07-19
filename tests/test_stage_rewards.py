import jax.numpy as jp
from dvgc.config import default_config
from dvgc.rewards import compute_stage_next_entry_reward

def test_event_dominates_bounded_dense_stage_reward():
 cfg=default_config();f=jp.zeros(16);a=jp.zeros(4)
 x=compute_stage_next_entry_reward(cfg=cfg,objective='apex_to_descent',feature=f,previous_feature=f,action=a,previous_action=a,next_entry=jp.asarray(True),hard_failure=jp.asarray(False),jump_latched=jp.asarray(True),window_active=jp.asarray(True),joint_energy=jp.asarray(0.0))
 assert float(x['event'])==8. and -.25<=float(x['shaping'])<=.2

def test_ascent_rewards_upward_height_progress():
 cfg=default_config();p=jp.zeros(16);up=p.at[2].set(.05);down=p.at[2].set(-.05);a=jp.zeros(4)
 def r(f):return compute_stage_next_entry_reward(cfg=cfg,objective='ascent_to_apex',feature=f,previous_feature=p,action=a,previous_action=a,next_entry=jp.asarray(False),hard_failure=jp.asarray(False),jump_latched=jp.asarray(True),window_active=jp.asarray(True),joint_energy=jp.asarray(0.0))['reward']
 assert float(r(up))>float(r(down))
