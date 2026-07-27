from pathlib import Path


def test_rsi_smoke_is_interface_only_and_jitted():
    text = Path("cli/smoke_descent_candidate_rsi.py").read_text()
    assert "StratifiedRSISampler" in text
    assert "jax.jit(jax.vmap(env.reset))" in text
    assert "jax.jit(jax.vmap(env.step))" in text
    assert "batch_state_crosstalk_absent" in text
    assert '"ppo_authorization": False' in text
    assert '"formal_tube_or_jel": False' in text
    assert "sampler_resume_exact" in text
    assert "single_invocation_control_and_counterfactual_pairs" in text
    assert "quad_keys = jp.concatenate([keys, keys, keys, keys], axis=0)" in text
    assert "same_action_control_linf" in text
    assert "jax.block_until_ready" in text
