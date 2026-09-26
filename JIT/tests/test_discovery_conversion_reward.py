import numpy as np
import pytest
from jit_dvgc.pulse_exploration import pulse_feedback
W=dict(novelty=.05,success=0.,failure=.5,pulse_failure=5.,conversion=2.,repeat=.1)
def row(cell,initial=1,final=1,trained=False,**kw):
 return dict(cell=cell,pulse_cells=[cell],initial_label=initial,label=final,learning_attempted=trained,pulse_applied_steps=3,**kw)
def test_conversion_dominates_repeat_and_known_success_has_no_quality_bonus():
 r=[row('old',0,1,True),row('old'),row('fresh',0,0,True)]
 reward,mask,seen,parts=pulse_feedback(r,['old'],W,quality_mode='discovery_conversion')
 np.testing.assert_allclose(reward,[1.9,-.1,-.45]);assert mask.all();assert seen==['fresh','old']
def test_batch_duplicates_split_novelty_without_order_bias():
 r=[row('x'),row('x')]
 a,_,_,_=pulse_feedback(r,[],W,quality_mode='discovery_conversion')
 np.testing.assert_allclose(a,[.025,.025])
def test_whole_pulse_window_not_just_endpoint():
 r=[row('old',pulse_cells_override=True)]
 r[0]['pulse_cells']=['new','old','old']
 a,_,seen,_=pulse_feedback(r,['old'],W,quality_mode='discovery_conversion')
 np.testing.assert_allclose(a,[.025]);assert 'new' in seen

def test_unknown_not_penalized_and_untrained_failure_not_conversion():
 r=[row('a',0,None,True),row('b',0,0,False),row('c',1,1,True)]
 a,m,_,_=pulse_feedback(r,[],W,quality_mode='discovery_conversion')
 assert list(m)==[False,False,True];np.testing.assert_allclose(a,[0,0,.05])
def test_direct_physical_failure_overrides_even_inconsistent_success_label():
 r=[row('x',0,1,True,prefix_terminal=True,prefix_physical_failure=True)]
 a,_,_,_=pulse_feedback(r,[],W,quality_mode='discovery_conversion')
 np.testing.assert_allclose(a,[-5])
def test_missing_window_provenance_rejected():
 r=row('x');del r['pulse_cells']
 with pytest.raises(ValueError,match='pulse_cells'):pulse_feedback([r],[],W,quality_mode='discovery_conversion')

def test_adoption_bonus_requires_final_decision_and_keeps_local_credit():
 weights={**W,'conversion':.2,'adoption_bonus':1.8}
 rows=[row('a',0,1,True,successor_adopted=False),row('b',0,1,True,successor_adopted=True),row('c',0,0,True,successor_adopted=True)]
 rewards,mask,_,_=pulse_feedback(rows,[],weights,quality_mode='discovery_conversion')
 np.testing.assert_allclose(rewards,[.25,2.05,-.45]);assert mask.all()

def test_adoption_bonus_never_guesses_missing_final_decision():
 with pytest.raises(ValueError,match='adoption'):
  pulse_feedback([row('a',0,1,True)],[],{**W,'adoption_bonus':1.8},quality_mode='discovery_conversion')

def test_physical_failure_never_gets_adoption_bonus():
 rewards,_,_,_=pulse_feedback([row('a',0,1,True,prefix_terminal=True,prefix_physical_failure=True)],[],{**W,'adoption_bonus':1.8},quality_mode='discovery_conversion')
 np.testing.assert_allclose(rewards,[-5.])
