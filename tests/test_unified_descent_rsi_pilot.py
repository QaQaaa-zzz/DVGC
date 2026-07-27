from pathlib import Path

from dvgc.config import load_config
from dvgc.runtime import ppo_effective_timesteps


def test_pilot_budget_and_reward_contract():
    cfg = load_config("configs/unified_descent_rsi_learnability_pilot_v1.json")
    assert cfg.episode_length == 24
    assert cfg.descent_rsi_pilot_reward_enable
    assert not cfg.descent_local_reward_enable
    assert cfg.natural_prob_flight == 0.0
    assert ppo_effective_timesteps(
        6400, unroll_length=32, batch_size=25, num_minibatches=2, num_evals=5,
    ) == 6400


def test_pilot_runner_keeps_formal_claims_and_ppo_bounded():
    text = Path("cli/run_unified_descent_rsi_pilot.py").read_text()
    assert 'a.timesteps!=6400' in text
    assert '"formal_tube_or_jel":False' in text
    assert '"ppo_authorization":False' in text
    assert "old_matcher\":False" in text
