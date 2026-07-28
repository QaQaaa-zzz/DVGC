import numpy as np
import pytest

from dvgc.descent_predecessor import (
    active_predecessor_ticks, expand_residual_knots, predecessor_priority,
    remaining_residual_suffix, require_forward_lineage,
)


def test_residual_suffix_preserves_exact_remaining_ticks():
    knots = np.asarray([[1, 2, 3, 4], [5, 6, 7, 8]], np.float32)
    expanded = expand_residual_knots(knots)
    assert np.array_equal(expanded[:4], np.repeat(knots[:1], 4, axis=0))
    suffix = remaining_residual_suffix(expanded, 3)
    assert np.array_equal(suffix[:5], expanded[3:])
    assert np.array_equal(suffix[5:], np.zeros((3, 4), np.float32))


def test_terminal_and_fifo_filter_prevents_synthetic_or_post_entry_capture():
    assert active_predecessor_ticks(5, [1, 2, 3, 3, 3, 3], 4) == [2, 3]


def test_priority_prefers_non_dominant_low_support_and_near_entry():
    base = {"source_was_p1": True, "source_tick": 3, "physical_state_hash": "x"}
    rare = base | {"candidate_id": "rare", "relative_tick_to_downstream_entry": -2}
    main = base | {"candidate_id": "main", "relative_tick_to_downstream_entry": -1}
    assert predecessor_priority(rare, "main", {"rare": 1, "main": 7}) < predecessor_priority(main, "main", {"rare": 1, "main": 7})


def test_no_state_splicing_negative_gate():
    good = {"construction_method": "forward_mjx_active_prefix", "state_splicing": False,
            "reverse_integration": False, "source_tick": 2, "downstream_entry_tick": 3,
            "actor_packet_fifo_valid": 3}
    require_forward_lineage(good)
    with pytest.raises(ValueError, match="splicing"):
        require_forward_lineage(good | {"state_splicing": True})


def test_artifact_index_is_stable_without_array_dictionary_equality():
    proposals = [{"array": np.asarray([1., 2.])}, {"array": np.asarray([3., 4.])}]
    for index, proposal in enumerate(proposals):
        proposal["_artifact_index"] = index
    proposals.reverse()
    assert [row["_artifact_index"] for row in proposals] == [1, 0]
