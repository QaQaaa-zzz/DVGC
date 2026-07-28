import numpy as np

from cli.audit_apex_full_action_authority_v1 import normalized_authority, pulse_action


def test_pulse_action_uses_declared_four_channel_order():
    for channel in range(4):
        action = np.asarray(pulse_action(channel, 0.25))
        assert action[channel] == 0.25
        assert np.count_nonzero(action) == 1


def test_normalized_authority_is_symmetric_and_finite():
    plus = np.asarray([0.02, 0.01, 0.2, 0.1, 0.0, 0.08, 0.04])
    authority = normalized_authority(plus, -plus, 0.25)
    assert authority.shape == (7,)
    assert np.isfinite(authority).all()
    assert np.allclose(authority[:3], [1.0, 0.5, 1.0])
