"""Zero-interaction preflight for the locked expansion-validation runtime.

The protocol audit proves source identity and TRAIN/validation separation. This
additional gate rebuilds the exact 76-D actor observation that frozen ``pi_0``
will see from every legacy held-out snapshot under unified phase semantics and
uses that observation for the runtime leakage audit. The legacy cached
observation is retained only as provenance: a mismatch is expected for old
post-Apex snapshots whose cached Phase-U task bit differs from the unified
downstream task bit. It performs no environment step, policy rollout, labeling,
or validation-outcome inspection.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .expansion_validation_protocol import (
    audit_expansion_validation_protocol,
    audit_group_disjointness,
    load_expansion_validation_protocol_config,
)
from .expansion_validation_runtime import (
    _expected_unified_actor_observation,
    enumerate_validation_attempts,
    load_validation_anchor_snapshots,
)
from .iteration_train_evidence import load_frozen_iteration_train_evidence


def unified_actor_observation_from_legacy_snapshot(snapshot: Any, *, phase: str) -> np.ndarray:
    """Compatibility wrapper around the single runtime observation authority."""
    return _expected_unified_actor_observation(snapshot, phase=phase)


def audit_expansion_validation_runtime_preflight(config_path: Path) -> dict[str, Any]:
    """Validate runtime observation semantics and leakage with zero interactions."""
    config_path = Path(config_path)
    protocol_audit = audit_expansion_validation_protocol(config_path)
    config = load_expansion_validation_protocol_config(config_path)
    protocol = config["protocol"]
    train_manifest, train_rows = load_frozen_iteration_train_evidence(
        Path(str(protocol["frozen_train_evidence"]))
    )
    anchors = load_validation_anchor_snapshots(protocol)

    validation_rows: list[dict[str, Any]] = []
    cached_match_count = 0
    cached_mismatch_count = 0
    cached_mismatch_count_by_phase = {"upstream": 0, "downstream": 0}
    for phase in ("upstream", "downstream"):
        for anchor_index, anchor in enumerate(protocol["sources"][phase]["anchors"]):
            snapshot = anchors[(phase, anchor_index)]
            actual = unified_actor_observation_from_legacy_snapshot(snapshot, phase=phase)
            cached = np.asarray(snapshot.observation, dtype=np.float32)
            if cached.shape != actual.shape or not np.isfinite(cached).all():
                raise ValueError(f"validation {phase} anchor cached observation is invalid")
            if np.allclose(cached, actual, rtol=0.0, atol=1.0e-5):
                cached_match_count += 1
            else:
                cached_mismatch_count += 1
                cached_mismatch_count_by_phase[phase] += 1
            validation_rows.append(
                {
                    "phase": phase,
                    "parent_group_id": str(anchor["parent_group_id"]),
                    "state_sha256": str(anchor["state_sha256"]),
                    "actor_observation": actual,
                }
            )

    leakage = audit_group_disjointness(
        train_rows,
        validation_rows,
        observation_atol=float(protocol["near_duplicate_audit"]["actor_observation_atol"]),
    )
    attempts = enumerate_validation_attempts(protocol)
    if len(attempts) != int(protocol["interaction_budget"]["attempt_count"]):
        raise ValueError("validation runtime attempt schedule count drift")

    expected_acquisition = 0
    for phase in ("upstream", "downstream"):
        panel = protocol["panels"][phase]
        anchors_count = len(protocol["sources"][phase]["anchors"])
        directions = len(panel["action_names"]) * len(panel["signs"])
        expected_acquisition += (
            anchors_count
            * directions
            * len(panel["strengths"])
            * sum(int(value) for value in panel["durations"])
        )
    if expected_acquisition != int(
        protocol["interaction_budget"]["maximum_acquisition_environment_interactions"]
    ):
        raise ValueError("validation runtime acquisition budget drift")

    return {
        "schema": "jit_expansion_validation_runtime_preflight_v1",
        "status": "runtime_preflight_ready",
        "iteration": int(protocol["iteration"]),
        "policy_name": str(protocol["policy_name"]),
        "scientific_protocol_sha256": str(protocol_audit["protocol_sha256"]),
        "frozen_train_manifest_sha256": str(train_manifest["manifest_sha256"]),
        "validation_anchor_count": len(validation_rows),
        "validation_anchor_unified_observation_count": len(validation_rows),
        "legacy_cached_observation_match_count": cached_match_count,
        "legacy_cached_observation_mismatch_count": cached_mismatch_count,
        "legacy_cached_observation_mismatch_count_by_phase": cached_mismatch_count_by_phase,
        "validation_parent_group_count_by_phase": dict(
            protocol_audit["validation_parent_group_count_by_phase"]
        ),
        "attempt_count": len(attempts),
        "maximum_acquisition_environment_interactions": expected_acquisition,
        "maximum_labeling_environment_interactions": int(
            protocol["interaction_budget"]["maximum_labeling_environment_interactions"]
        ),
        **leakage,
        "environment_interactions": 0,
        "training_transitions": 0,
        "expert_switching_used": False,
        "validation_outcomes_inspected": False,
        "test_data_used": False,
        "final_evaluation_data_used": False,
        "claim_boundary": dict(protocol["claim_boundary"]),
        "next_gate": "GPU restore contract and full preflight before formal validation launch",
    }
