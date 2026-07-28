import jax.numpy as jnp
import numpy as np

from dvgc.descent_supervised import extract_trainable, replace_trainable, train_supervised


def policy():
    return {"params": {
        "neutral_loc":{"kernel":jnp.zeros((2,4)),"bias":jnp.zeros(4)},
        "scale_parameter":jnp.ones(4),
        "trunk":{
            "hidden_0":{"kernel":jnp.ones((2,2)),"bias":jnp.ones(2)},
            "hidden_1":{"kernel":jnp.ones((2,2))*2,"bias":jnp.ones(2)*2},
            "hidden_2":{"kernel":jnp.ones((2,2))*3,"bias":jnp.ones(2)*3},
        },
    }}


def test_head_mode_changes_only_deterministic_mean_head():
    base=policy();trainable=extract_trainable(base,"head")
    trainable["neutral_loc"]["bias"]=jnp.ones(4)
    updated=replace_trainable(base,trainable,"head")
    np.testing.assert_array_equal(updated["params"]["neutral_loc"]["bias"],np.ones(4))
    np.testing.assert_array_equal(updated["params"]["scale_parameter"],base["params"]["scale_parameter"])
    np.testing.assert_array_equal(updated["params"]["trunk"]["hidden_2"]["kernel"],base["params"]["trunk"]["hidden_2"]["kernel"])


def test_last_block_mode_keeps_earlier_blocks_and_log_std_frozen():
    base=policy();trainable=extract_trainable(base,"last_block")
    trainable["hidden_2"]["bias"]=jnp.zeros(2)
    updated=replace_trainable(base,trainable,"last_block")
    np.testing.assert_array_equal(updated["params"]["trunk"]["hidden_0"]["kernel"],base["params"]["trunk"]["hidden_0"]["kernel"])
    np.testing.assert_array_equal(updated["params"]["trunk"]["hidden_1"]["kernel"],base["params"]["trunk"]["hidden_1"]["kernel"])
    np.testing.assert_array_equal(updated["params"]["scale_parameter"],base["params"]["scale_parameter"])
    np.testing.assert_array_equal(updated["params"]["trunk"]["hidden_2"]["bias"],np.zeros(2))


def test_teacher_weight_must_be_strict_probability():
    import pytest
    with pytest.raises(ValueError, match="teacher_weight"):
        train_supervised(base_policy=policy(), actor_action=lambda p, x: x,
                         teacher_observation=np.zeros((1, 4)), teacher_target=np.zeros((1, 4)),
                         anchor_observation=np.zeros((1, 4)), anchor_target=np.zeros((1, 4)),
                         learning_rate=1e-3, steps=1, teacher_weight=0.0)
