from __future__ import annotations


def test_gate_diagnosis_separates_auc_recall_and_parent_failures():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "cli" / "analyze_iterative_calibration_failure.py"
    spec = importlib.util.spec_from_file_location("analyze_iterative_calibration_failure", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    base = {
        "contract": {"minimum_roc_auc": 0.70, "minimum_positive_recall": 0.20},
        "metrics": {"roc_auc": 0.80},
        "positive_recall_at_threshold": 0.30,
        "accepted_negative_count": 0,
        "parent_support": {
            "g0": {
                "candidate_count": 4,
                "positive_count": 3,
                "negative_count": 1,
                "accepted_positive_count": 1,
            },
            "g1": {
                "candidate_count": 4,
                "positive_count": 3,
                "negative_count": 1,
                "accepted_positive_count": 1,
            },
        },
    }
    passed = module._gate_diagnosis(base)
    assert passed["failed_checks"] == []

    auc = {**base, "metrics": {"roc_auc": 0.60}}
    assert module._gate_diagnosis(auc)["failure_category"] == "ranking_generalization_failure"

    recall = {**base, "positive_recall_at_threshold": 0.10}
    assert module._gate_diagnosis(recall)["failure_category"] == "conservative_threshold_recall_failure"

    parent = {
        **base,
        "parent_support": {
            **base["parent_support"],
            "g1": {
                **base["parent_support"]["g1"],
                "accepted_positive_count": 0,
            },
        },
    }
    result = module._gate_diagnosis(parent)
    assert "accepted_positive_in_every_parent" in result["failed_checks"]
    assert result["failure_category"] == "parent_local_threshold_coverage_failure"
