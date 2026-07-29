from pathlib import Path

from cli.screen_descent_network_candidates_with_frozen_policy_v1 import (
    PILOTS,
    collect_final_safe_candidates,
)


def test_candidate_collection_uses_only_final_safe_unique_states():
    records = collect_final_safe_candidates(list(PILOTS))
    assert len(records) == 3
    assert len({row["id"] for row in records}) == 3
    assert len({row["candidate_id"] for row in records}) == 3
    assert all(row["artifact_role"] == "proposal_support_bank" for row in records)
    assert all(not row["safe_claim_allowed"] for row in records)
    assert all("independent_audit" not in str(Path(row["source_artifact"])).lower() for row in records)
