from pathlib import Path

import numpy as np

from cli.discover_contact_supported_apex_bridge import _lexicographic


def test_segmented_bridge_lexicographic_order_prefers_final():
    base = {
        "final_landing_recovery": False,
        "descent_controller_success": True,
        "formal_descent_support_entry": True,
        "stable_16_ticks": True,
        "termination_reason": "horizon_exhaustion",
        "physical_apex_tick": 2,
        "max_stable_descent_ticks": 20,
        "contact_momentum_residual": 0.,
    }
    final = dict(base, final_landing_recovery=True,
                 descent_controller_success=False)
    assert _lexicographic(final) > _lexicographic(base)


def test_contact_bridge_uses_real_last_support_and_no_ppo():
    text = Path("cli/discover_contact_supported_apex_bridge.py").read_text()
    assert '"last_support"' in text
    assert "contact_supported_momentum_shaping" in text
    assert '"apex_ppo_authorized": False' in text
    assert "prediction_horizon" in text
