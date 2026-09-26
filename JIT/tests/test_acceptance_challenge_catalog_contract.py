from __future__ import annotations

import importlib.util
from pathlib import Path

from jit_dvgc.unified_continuation_labels import validate_unified_boundary_catalog


def _load_cli():
    path = Path(__file__).resolve().parents[1] / "cli" / "acceptance_challenge_repair.py"
    spec = importlib.util.spec_from_file_location("acceptance_challenge_repair_catalog", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _phase_catalog(*, phase: str, protocol_sha: str, state_sha: str) -> dict:
    phase_index = 0 if phase == "upstream" else 1
    actor = "a" * 64
    payload = "b" * 64
    frozen = "c" * 64
    tube = "d" * 64
    return {
        "schema": "jit_unified_boundary_catalog_v1",
        "status": "completed",
        "artifact_role": "unlabeled_policy_conditioned_frontier_candidates",
        "split": "train",
        "iteration": 1,
        "policy_name": "pi_1",
        "policy_actor_sha256": actor,
        "policy_payload_sha256": payload,
        "frozen_unified_manifest_sha256": frozen,
        "source_tube_manifest_sha256": tube,
        "protocol_sha256": protocol_sha,
        "anchor_count": 1,
        "attempted_candidate_count": 1,
        "candidate_count": 1,
        "phase_attempted_candidate_counts": {phase: 1},
        "phase_candidate_counts": {phase: 1},
        "phase_exclusion_counts": {phase: {}},
        "environment_interactions": 2,
        "maximum_environment_interactions": 2,
        "training_transitions": 0,
        "expert_switching_used": False,
        "test_data_used": False,
        "validation_data_used": False,
        "final_evaluation_data_used": False,
        "exclusion_counts": {},
        "claim_boundary": {
            "unlabeled_acquisition_only": True,
            "tube_expansion_claim": False,
            "jce_jel_claim": False,
            "certified_safe_set_claim": False,
        },
        "entries": [
            {
                "candidate_id": f"pi1_{phase}_000000",
                "candidate_kind": "reachable_unified_frontier_probe",
                "split": "train",
                "phase": phase,
                "phase_index": phase_index,
                "snapshot": "snapshots/candidate_000000",
                "source_bank": "boundary_bank",
                "state_sha256": state_sha,
                "parent_group_id": f"{phase}_group",
                "parent_state_sha256": ("e" if phase == "upstream" else "f") * 64,
                "policy_iteration": 1,
                "policy_actor_sha256": actor,
                "policy_payload_sha256": payload,
                "protocol_sha256": protocol_sha,
            }
        ],
    }


def test_v3c_merged_catalog_stays_compatible_with_continuation_label_validator(tmp_path: Path) -> None:
    cli = _load_cli()
    actor = "a" * 64
    payload = "b" * 64
    frozen = "c" * 64
    tube = "d" * 64
    challenge = {
        "source_plan": "frontier_plan.json",
        "challenge_plan_sha256": "1" * 64,
        "phase_probe_panels": {
            "upstream": dict(cli.TWO_AXIS_PANEL),
            "downstream": dict(cli.TWO_AXIS_PANEL),
        },
        "seeds": {"acquisition": 1234, "labeling": 5678},
    }
    source_plan = {"plan_sha256": "2" * 64}
    phase_catalogs = {
        "upstream": _phase_catalog(
            phase="upstream", protocol_sha="3" * 64, state_sha="4" * 64
        ),
        "downstream": _phase_catalog(
            phase="downstream", protocol_sha="5" * 64, state_sha="6" * 64
        ),
    }

    report = cli._merge_phase_acquisitions(
        challenge_plan_path=tmp_path / "challenge_plan.json",
        challenge=challenge,
        source_plan=source_plan,
        phase_catalogs=phase_catalogs,
        acquisition_dir=tmp_path / "acquisition",
    )

    assert report["artifact_role"] == "unlabeled_policy_conditioned_frontier_candidates"
    assert report["claim_boundary"] == {
        "unlabeled_acquisition_only": True,
        "tube_expansion_claim": False,
        "jce_jel_claim": False,
        "certified_safe_set_claim": False,
    }
    assert [row["phase"] for row in report["entries"]] == ["upstream", "downstream"]
    assert all(row["protocol_sha256"] == report["protocol_sha256"] for row in report["entries"])
    assert report["entries"][0]["source_bank"].startswith("phase_upstream/")
    assert report["entries"][1]["source_bank"].startswith("phase_downstream/")

    validate_unified_boundary_catalog(
        report,
        policy_record={
            "iteration": 1,
            "name": "pi_1",
            "actor_sha256": actor,
            "payload_sha256": payload,
        },
        frozen_manifest_sha256=frozen,
    )
    assert report["source_tube_manifest_sha256"] == tube
