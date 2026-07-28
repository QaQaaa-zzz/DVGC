import pytest

from cli.normalize_stage_entry_bank import normalize
from cli.stage_label_pilot import annotate_entry_snapshot


def test_new_entry_snapshot_is_directly_reusable_by_next_flight_substage():
    row = {"id": "upstream", "trajectory_parent_id": "parent"}
    snapshot = {}
    annotate_entry_snapshot(snapshot, row, "ascent", "entry", "policy", 7, 4, {})
    assert snapshot["entry_to_stage"] == "apex"
    assert snapshot["flight_subinterval"] == "apex"
    assert snapshot["upstream_candidate_id"] == "upstream"


def test_existing_entry_bank_normalization_preserves_state_and_adds_provenance():
    records = [{"id": "entry", "entry_from_stage": "ascent",
                "entry_to_stage": "apex", "qpos": [1, 2]}]
    evaluation = {"labels": [{"candidate_id": "upstream", "branches": [
        {"entry_snapshot_id": "entry"}]}]}
    normalized = normalize(records, evaluation)
    assert normalized[0]["qpos"] == [1, 2]
    assert normalized[0]["flight_subinterval"] == "apex"
    assert normalized[0]["upstream_candidate_id"] == "upstream"


def test_normalization_rejects_unprovenanced_entry():
    with pytest.raises(ValueError, match="absent"):
        normalize([{"id": "x", "entry_from_stage": "ascent",
                    "entry_to_stage": "apex"}], {"labels": []})
