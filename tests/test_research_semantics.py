from dvgc.research_semantics import (
    classify_handoff, proposal_training_weight, summarize_handoff,
)
from cli.handoff_decomposition_shard import _events


def test_handoff_classes_are_disjoint_and_do_not_redefine_success():
    assert classify_handoff(has_contact=True, event_order_valid=True, matched_c_l=False, landing_label="safe") == "H1"
    assert classify_handoff(has_contact=True, event_order_valid=True, matched_c_l=True, landing_label="dead") == "H2"
    assert classify_handoff(has_contact=True, event_order_valid=True, matched_c_l=False, landing_label="boundary") == "H3"
    assert classify_handoff(has_contact=False, event_order_valid=True, matched_c_l=False, landing_label="safe") == "H4"
    assert classify_handoff(has_contact=True, event_order_valid=False, matched_c_l=True, landing_label="safe") == "H4"
    assert classify_handoff(has_contact=True, event_order_valid=True, matched_c_l=True, landing_label="safe") == "matched_recoverable"


def test_proposal_weights_never_admit_dead_and_keep_unknown_low():
    assert proposal_training_weight("safe", .8) == 1.0
    assert proposal_training_weight("boundary", .65) > proposal_training_weight("boundary", .4)
    assert 0 < proposal_training_weight("unknown", .9) < proposal_training_weight("boundary", .4)
    assert proposal_training_weight("dead", .9) == 0.0


def test_handoff_summary_reports_branch_state_parent_and_layer_support():
    rows = [
        {"handoff_class": "H1", "candidate_id": "a", "parent_id": "p", "layer": "late"},
        {"handoff_class": "H1", "candidate_id": "b", "parent_id": "p", "layer": "middle"},
        {"handoff_class": "H3", "candidate_id": "a", "parent_id": "p", "layer": "late"},
    ]
    report = summarize_handoff(rows)
    assert report["branches"] == 3
    assert report["classes"]["H1"] == {
        "branches": 2, "branch_fraction": 2 / 3, "states": 2,
        "parents": 1, "layers": {"late": 1, "middle": 1},
    }


def test_event_selector_keeps_handoff_and_marks_physical_to_handoff_pairs():
    before = {"rows": [{"id": "s", "candidate_index": 0, "branch_evidence": [
        {"branch_index": 0, "terminal_cause": "physical_failure"},
        {"branch_index": 1, "terminal_cause": "final_recovery"},
    ]}]}
    after = {"rows": [{"id": "s", "candidate_index": 0, "parent": "p", "layer": "late", "branch_evidence": [
        {"branch_index": 0, "terminal_cause": "handoff_missed_final"},
        {"branch_index": 1, "terminal_cause": "final_recovery"},
    ]}]}
    selected = _events(after, before)
    assert len(selected) == 1
    assert selected[0]["physical_to_handoff"] is True
