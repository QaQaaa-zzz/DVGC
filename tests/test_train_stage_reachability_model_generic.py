import json
from pathlib import Path

from cli.train_stage_reachability_model import parent_key


def test_stage_model_cli_is_generic_and_keeps_predictions_non_authoritative():
    text = Path("cli/train_stage_reachability_model.py").read_text()
    assert "choices=('takeoff','ascent','apex','descent','landing')" in text
    assert "'stage':a.stage" in text
    assert "'not_certified_tube':True" in text
    assert "'not_safe_labels':True" in text
    assert "negative means negative under the frozen controller bank" in text


def test_parent_identity_prefers_real_trajectory_lineage():
    assert parent_key({"trajectory_parent_id": "trajectory", "reference_index": 4}) == "trajectory"
    assert parent_key({"parent_candidate_id": "candidate", "reference_index": 4}) == "candidate"
    assert parent_key({"reference_index": 4}) == "reference:4"
