from dvgc.research_semantics import CERTIFIED_TUBE, PROPOSAL_SUPPORT_BANK, validate_artifact_role


def test_certified_and_proposal_roles_are_not_interchangeable():
    validate_artifact_role({"artifact_role": CERTIFIED_TUBE}, expected=CERTIFIED_TUBE)
    validate_artifact_role({"artifact_role": PROPOSAL_SUPPORT_BANK}, expected=PROPOSAL_SUPPORT_BANK)
    try:
        validate_artifact_role({"artifact_role": PROPOSAL_SUPPORT_BANK}, expected=CERTIFIED_TUBE)
    except ValueError:
        pass
    else:
        raise AssertionError("proposal support bank was accepted as a certified Tube")
