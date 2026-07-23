from pathlib import Path


def test_takeoff_reset_builder_has_two_sources_and_authenticity_gates():
    text = Path("cli/build_takeoff_reset_bank.py").read_text()
    assert "canonical_compressed" in text
    assert "reference_aligned_compressed" in text
    assert "adjacent_pairs" in text
    assert "complete XML key pose/velocity" in text
    assert "GroundSupportSolver" in text
    assert "premature_next_stage" in text
    assert "shock_probe_steps" in text
    assert "reset_protocol_sha256" in text
    assert "model.joint(\"hip_joint\")" in text
    assert "model.joint(\"knee_joint\")" in text
