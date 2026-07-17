from collections import defaultdict
from types import SimpleNamespace

from cli.build_discrete_tube_rsi_bank import parent_key


def test_parent_key_prefers_original_entry_source():
    assert parent_key({"id":"x","parent_candidate_id":"child","entry_source_id":"root"})=="root"


def test_declared_tube_rsi_masses_and_limits_are_fixed():
    from dvgc.config import default_config
    cfg=default_config()
    assert cfg.discrete_tube_rsi_safe_mass==.70
    assert cfg.discrete_tube_rsi_boundary_mass==.30
    assert cfg.descent_acquisition_max_rounds==2
    assert cfg.descent_acquisition_max_proposals==64
