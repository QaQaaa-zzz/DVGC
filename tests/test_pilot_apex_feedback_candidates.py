from pathlib import Path


def test_pilot_keeps_local_formal_and_final_outcomes_separate():
    text = Path("cli/pilot_apex_feedback_candidates.py").read_text()
    assert '"local_successes"' in text
    assert '"formal_successes"' in text
    assert '"final_successes"' in text
    assert '"safe_claim_allowed": False' in text
