import jax
import pytest

from dvgc.rollout import _logged_replay_sidecars, inferred_apex_seen, restore_snapshot_mode
from dvgc.snapshot_timing import SNAPSHOT_SCHEMA_NAME, authority_replay_mode


def test_dynamic_snapshot_with_none_reference_index_uses_explicit_apex_latch():
    assert inferred_apex_seen({"reference_index": None, "apex_seen": 1}) == 1
    assert inferred_apex_seen({"reference_index": None, "apex_seen": 0}) == 0


def test_apex_latch_fallback_is_none_safe():
    assert inferred_apex_seen({"reference_index": None, "source_index": None}) == 0
    assert inferred_apex_seen({"reference_index": 220}) == 1
    assert inferred_apex_seen({"source_index": 219}) == 0


def test_logged_mode_requires_logged_actor_input_before_env_access():
    row={"source_phase":"flight","oracle_phase":2,"policy_state":{}}
    with pytest.raises(ValueError,match="actor_observation"):
        restore_snapshot_mode(object(),row,jax.random.PRNGKey(0),observation_mode="legacy_logged_replay")


def test_restore_mode_is_never_implicit_for_authority_api():
    with pytest.raises(ValueError,match="observation_mode"):
        restore_snapshot_mode(object(),{},jax.random.PRNGKey(0),observation_mode="auto")


def test_authority_mode_never_independently_reconstructs_legacy_records():
    assert authority_replay_mode({"schema_name": SNAPSHOT_SCHEMA_NAME}) == "timing_explicit_independent_reconstruction"
    assert authority_replay_mode({"schema_name": "dvgc_physical_policy_state_v3_warmstart"}) == "legacy_logged_replay"


def test_independent_mode_does_not_read_current_frame_or_actor_sidecars():
    class Guarded(dict):
        def __getitem__(self, key):
            if key in {"actor_observation_t", "current_frame_t"}:
                raise AssertionError(f"independent mode read {key}")
            return super().__getitem__(key)
    assert _logged_replay_sidecars(Guarded(), False) == (None, None)
