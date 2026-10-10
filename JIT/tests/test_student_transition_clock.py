import numpy as np
import pytest

def test_demo_weight_moves_with_transitions_when_normalizer_is_fixed():
    from jit_dvgc.generative_bridge.student_clock import demo_weight
    assert float(demo_weight(0,128000,.2,.05))==pytest.approx(.2)
    assert float(demo_weight(64000,128000,.2,.05))==pytest.approx(.125)
    assert float(demo_weight(128000,128000,.2,.05))==pytest.approx(.05)

def test_frozen_trainer_keeps_normalizer_and_injects_actual_completed_rollout_clock():
    import jax.numpy as jp
    from brax.training import types
    from brax.training.acme import running_statistics
    from jit_dvgc.generative_bridge.student_clock import instrument_frozen_normalizer
    n=running_statistics.init_state({'state':jp.zeros(2)})
    data=types.Transition(observation={'state':jp.ones((2,3,2))},action=jp.zeros((2,3,1)),reward=jp.zeros((2,3)),
        discount=jp.ones((2,3)),next_observation={'state':jp.ones((2,3,2))},extras={'policy_extras':{}})
    from types import SimpleNamespace
    state=SimpleNamespace(normalizer_params=n,env_steps=types.UInt64(hi=0,lo=32000))
    out,d=instrument_frozen_normalizer(trainer_fixture)(state,data)
    for a,b in zip(__import__('jax').tree.leaves(n),__import__('jax').tree.leaves(out)):np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(d.extras['policy_extras']['completed_training_transitions'],np.full((2,3),32006))

def trainer_fixture(training_state,data):
    # Match the pinned trainer's two normalization branches with real statistics.
    from brax.training.acme import running_statistics
    env_step_per_training_step=6
    normalizer_params=training_state.normalizer_params
    normalizer_params=running_statistics.update(normalizer_params,data.observation)
    normalizer_params=running_statistics.update(normalizer_params,data.observation)
    return normalizer_params,data

def test_installed_brax_layout_and_full_learner_composition_are_supported():
    from brax.training.agents.ppo.train import train
    from jit_dvgc.generative_bridge.student_clock import instrument_frozen_normalizer
    from jit_dvgc.generative_bridge.learner_continuation import instrument_trainer
    assert callable(instrument_frozen_normalizer(train))
    continued=instrument_trainer(train,lambda s,k:(s,k),lambda *a:None)
    assert callable(instrument_frozen_normalizer(continued))

def test_freezing_rejects_normalizer_count_clock():
    from jit_dvgc.generative_bridge.student import make_joint_student_trainer
    with pytest.raises(ValueError,match='explicit completed'):
        make_joint_student_trainer(lambda **k:None,None,retention=None,keep_coefficient=0,freeze_actor_normalizer=True)
