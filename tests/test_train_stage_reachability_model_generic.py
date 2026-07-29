import json
from pathlib import Path


def test_stage_model_cli_is_generic_and_keeps_predictions_non_authoritative():
    text = Path("cli/train_stage_reachability_model.py").read_text()
    assert "choices=('takeoff','ascent','apex','descent','landing')" in text
    assert "'stage':a.stage" in text
    assert "'not_certified_tube':True" in text
    assert "'not_safe_labels':True" in text
    assert "negative means negative under the frozen controller bank" in text
