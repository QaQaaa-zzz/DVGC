import pytest

from cli.pilot_descent_ranked_reachability_v1 import validate_selection


def _rows():
    rows = []
    for index, region in enumerate(["early"] + ["middle"] * 7 + ["late"] * 4):
        rows.append({
            "proposal_id": f"p{index}", "candidate_id": f"c{index}", "region": region,
            "shell_layer": 1, "physical_state_sha256": f"h{index}",
            "source_artifact": "bank.pkl", "source_index": index, "reachability_score": 0.5,
        })
    return rows


def test_selection_requires_proposal_only_hash_matched_parent_disjoint_artifact():
    rows = _rows()
    ranking = {
        "status": "PASS", "artifact_role": "proposal_only_reachability_ranking",
        "formal_tube_or_matcher": False, "independent_audit_labels_used": False,
        "model_sha256": "model", "selected": rows,
    }
    assert validate_selection(ranking, rows, "model") == rows
    bad = dict(ranking, selected=rows[:-1] + [dict(rows[-1], candidate_id="c0")])
    with pytest.raises(ValueError, match="parent-disjoint"):
        validate_selection(bad, rows, "model")


def test_selection_rejects_audit_feedback_or_index_drift():
    rows = _rows()
    ranking = {
        "status": "PASS", "artifact_role": "proposal_only_reachability_ranking",
        "formal_tube_or_matcher": False, "independent_audit_labels_used": True,
        "model_sha256": "model", "selected": rows,
    }
    with pytest.raises(ValueError, match="evidence boundary"):
        validate_selection(ranking, rows, "model")
