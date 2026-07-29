from types import SimpleNamespace

from cli.normalize_descent_tube_v6 import normalize_records
from cli.verify_descent_tube_v5 import verify_records
from dvgc.bank import beta_posterior, posterior_label


CFG = SimpleNamespace(
    min_branches=8, safe_threshold=.7, dead_threshold=.3,
    boundary_max_width=.35,
)


def _outcome(successes):
    posterior = beta_posterior(successes, 32 - successes)
    return {
        "successes": successes, "failures": 32 - successes, "branches": 32,
        "posterior": posterior,
        "label": posterior_label(
            posterior, 32, min_branches=8, safe_threshold=.7,
            dead_threshold=.3, boundary_max_width=.35,
        ),
    }


def _legacy_row():
    evidence = [{
        "branch_index": index, "branch_seed": 1000 + index,
        "seed_namespace": "independent", "final_recovery": index < 30,
        "chain_success": index < 11, "end_reason": "recovery" if index < 30 else "roll_limit",
    } for index in range(32)]
    return {
        "id": "state", "state_byte_hash": "physical-state",
        "artifact_role": "certified_tube", "certified_safe": True,
        "safe_claim_allowed": True, "training_only": False,
        "tube_metrics_eligible": True, "independent_audit": {"label": "safe"},
        "policy_version": "policy", "final": _outcome(30),
        # Legacy extension accidentally stored Chain AND Final (9) rather than
        # the independently latched chain_ever count (11).
        "chain": _outcome(9), "certification_branches": evidence,
    }


def test_normalization_preserves_final_membership_and_repairs_chain_semantics():
    source = _legacy_row()
    rows, changes = normalize_records(
        [source], policy_identity="policy", cfg=CFG, tube_version="v6",
    )
    row = rows[0]
    assert row["final"]["successes"] == source["final"]["successes"] == 30
    assert row["final"]["label"] == "safe"
    assert row["chain"]["successes"] == 11
    assert row["independent_audit"] is True
    assert row["independent_audit_legacy"] == {"label": "safe"}
    assert changes == {
        "chain_records_corrected": 1,
        "legacy_independent_audit_fields_normalized": 1,
    }
    summary, reasons = verify_records(
        rows, policy_identity="policy", branches=32, min_branches=8,
        safe_threshold=.7, dead_threshold=.3, boundary_max_width=.35,
    )
    assert reasons == []
    assert summary["posterior_safe_with_failures_states"] == 1
