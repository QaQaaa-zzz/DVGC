from cli.audit_final_shared_policy_candidates import exact_final_label
from cli.freeze_final_shared_policy_jel import calibration_metrics, exact_ids
from pathlib import Path


def test_final_candidate_label_requires_every_branch_to_recover():
    good = [{"final_recovery": True}] * 32
    assert exact_final_label(good)["label"] == "positive"
    mixed = good[:-1] + [{"final_recovery": False}]
    assert exact_final_label(mixed)["label"] == "boundary"
    assert exact_final_label([{"final_recovery": False}] * 32)["label"] == "negative"


def test_final_jel_freezer_requires_explicit_32_branch_final_evidence():
    good = {"candidate_id": "s", "s": 32, "n": 32, "label": "positive",
            "branches": [{"final_recovery": True}] * 32}
    assert exact_ids({"labels": [good]}, 32) == {"s"}
    bad = dict(good); bad["branches"] = [{"final_recovery": True}] * 31
    assert exact_ids({"labels": [bad]}, 32) == set()
    bad = dict(good); bad["branches"] = good["branches"][:-1] + [{"final_recovery": False}]
    assert exact_ids({"labels": [bad]}, 32) == set()


def test_final_audit_accepts_only_provenance_derived_survivor_banks():
    audit_text = Path("cli/audit_final_shared_policy_candidates.py").read_text()
    select_text = Path("cli/select_exact_branch_survivors.py").read_text()
    assert '"root_source_bank_sha256"' in select_text
    assert '"root_source_state_count"' in select_text
    assert '"root_phase_state_counts"' in select_text
    assert 'candidates.metadata.get("root_source_bank_sha256", input_hash)' in audit_text


def test_final_jel_requires_all_five_phases_and_root_coverage_denominator():
    text = Path("cli/freeze_final_shared_policy_jel.py").read_text()
    assert "final shared-policy JEL lacks independently safe support in phases" in text
    assert '"root_source_states": root_count' in text
    assert '"phase_coverage"' in text
    assert '"independent_audit_precision"' in text


def test_final_calibration_uses_independent_branch_outcomes():
    construction = {
        "a": {"s": 32}, "b": {"s": 16},
    }
    audit = {
        "a": {"branches": [{"final_recovery": True}] * 32},
        "b": {"branches": ([{"final_recovery": True}] * 16
                              + [{"final_recovery": False}] * 16)},
    }
    metrics = calibration_metrics(construction, audit, 32)
    assert metrics["brier"] == 0.125
    assert metrics["ece_10_bin"] == 0.0
    assert metrics["audit_branches"] == 64
