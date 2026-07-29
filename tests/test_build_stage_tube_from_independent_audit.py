from pathlib import Path

from cli.build_stage_tube_from_independent_audit import exact_safe, final_safe


def test_exact_safe_requires_all_preregistered_branches():
    success = [{"success": True}] * 32
    assert exact_safe({"label": "positive", "s": 32, "n": 32, "branches": success}, 32)
    assert not exact_safe({"label": "positive", "s": 31, "n": 32, "branches": success}, 32)
    assert not exact_safe({"label": "boundary", "s": 32, "n": 32, "branches": success}, 32)
    assert not exact_safe({"label": "positive", "s": 8, "n": 8,
                           "branches": [{"success": True}] * 8}, 32)
    assert not exact_safe({"label": "positive", "s": 32, "n": 32,
                           "branches": [{"success": True}] * 31 + [{"success": False}]}, 32)
    assert not exact_safe({"label": "positive", "s": 32, "n": 32}, 32)


def test_frozen_support_source_preserves_controller_provenance_contract():
    text = __import__("pathlib").Path("cli/build_stage_tube_from_independent_audit.py").read_text()
    assert '"controller_descriptors"' in text
    assert '"certifying_controller_bank"' in text
    assert '"certified_teacher_action_evidence"' in text
    assert '"independent audit branch seeds are not globally unique"' in text


def test_final_safe_requires_explicit_full_stack_outcomes():
    assert final_safe({"branches": [{"final_recovery": True}] * 32}, 32)
    assert not final_safe({"branches": [{"success": True}] * 32}, 32)
    assert not final_safe({"branches": [{"final_recovery": True}] * 31}, 32)


def test_seed_validation_accepts_legacy_branch_seed_and_new_seed_fields():
    text = Path("cli/build_stage_tube_from_independent_audit.py").read_text()
    assert 'branch.get("seed", branch.get("branch_seed"))' in text


def test_local_entry_support_is_not_declared_a_tube():
    text = Path("cli/build_stage_tube_from_independent_audit.py").read_text()
    assert '"certified_tube": args.evidence_scope == "final_recovery"' in text
    assert '"stage_entry_certified_proposal_support"' in text
