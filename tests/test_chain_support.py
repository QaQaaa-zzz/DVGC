import pytest

from cli.audit_chain_support import evaluation_rows


def test_evaluation_rows_accepts_legacy_and_composite_reports():
    rows = [{"candidate_id": "x", "final": 1}]
    assert evaluation_rows({"rows": rows}) is rows
    assert evaluation_rows({"composite": {"rows": rows}}) is rows


def test_evaluation_rows_rejects_summary_only_report():
    with pytest.raises(ValueError, match="no candidate rows"):
        evaluation_rows({"composite": {"episodes": 160}})
