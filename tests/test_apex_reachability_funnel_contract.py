from pathlib import Path


def test_apex_funnel_is_exact_and_never_promotes_local_support_to_formal_tube():
    text = Path("scripts/run_apex_reachability_funnel.sh").read_text()
    assert "--required-branches 4 --next-branches 8" in text
    assert "--required-branches 8 --next-branches 32" in text
    assert "--evidence-scope local_next_stage" in text
    assert "--branches 32" in text
    assert "cli.build_phase_balanced_tube_rsi_bank" in text
    assert "descent_tube_v5.pkl" in text
    assert "artifacts/landing_tube.pkl" in text
    assert "cli.train" not in text
    assert "rm " not in text
