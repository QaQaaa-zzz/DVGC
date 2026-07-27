from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from dvgc.descent_probe import exact_replay_matches, formal_dynamic_margin, lexicographic_order


def test_formal_dynamic_margin_uses_only_hard_roll_and_pitch_limits():
    cfg = SimpleNamespace(max_roll_deg=30.0, max_pitch_deg=40.0)
    feature = jnp.zeros((2, 16))
    feature = feature.at[0, 3].set(jnp.deg2rad(20.0))
    feature = feature.at[0, 4].set(jnp.deg2rad(10.0))
    feature = feature.at[0, 9].set(1.0e6)
    feature = feature.at[0, 10].set(-1.0e6)
    feature = feature.at[1, 3].set(jnp.deg2rad(31.0))

    margin = np.asarray(formal_dynamic_margin(feature, cfg))

    np.testing.assert_allclose(margin[0], np.deg2rad(10.0), atol=1e-6)
    np.testing.assert_allclose(margin[1], -np.deg2rad(1.0), atol=1e-6)


def test_lexicographic_order_prioritizes_physics_then_effort():
    result = {
        "survival": np.array([10, 11, 11, 11]),
        "minimum_margin": np.array([10.0, 0.1, 0.2, 0.2]),
        "terminal_margin": np.array([10.0, 1.0, 0.5, 0.5]),
        "residual_rms": np.array([0.0, 0.0, 0.2, 0.1]),
    }

    assert lexicographic_order(result).tolist() == [3, 2, 1, 0]


def test_exact_replay_must_also_match_selected_cem_summary():
    replay = {
        "survival": np.array([12]), "minimum_margin": np.array([.1]),
        "terminal_margin": np.array([.2]), "end_code": np.array([5]),
        "actions": np.zeros((2, 1, 4)), "features": np.zeros((2, 1, 16)),
    }
    summary = {
        "survival": 12, "minimum_margin": .1, "terminal_margin": .2,
        "end_code": 5, "actions": np.zeros((2, 4)), "features": np.zeros((2, 16)),
    }
    assert exact_replay_matches(replay, replay, summary)
    wrong = dict(summary, survival=14)
    assert not exact_replay_matches(replay, replay, wrong)
