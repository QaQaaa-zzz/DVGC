from pathlib import Path


def test_stage_reset_audit_requires_time_aligned_anchors_and_t0_gate():
    text = Path("cli/audit_stage_reset_alignment.py").read_text()
    assert "aligned_reference_anchors" in text
    assert "next_stage_at_t0" in text
    assert "training_authorized" in text
    assert '"ascent"' in text and '"apex"' in text
