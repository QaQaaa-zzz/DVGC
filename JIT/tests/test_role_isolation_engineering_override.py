from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from jit_dvgc.iterative_frontier_protocol import canonical_sha256


CLI = Path(__file__).parents[1] / "cli" / "audit_iterative_role_isolation.py"
spec = importlib.util.spec_from_file_location("audit_iterative_role_isolation", CLI)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def _manifest(role: str, sha_char: str) -> dict:
    return {
        "iteration": 1,
        "plan_sha256": "p" * 64,
        "role_manifest_sha256": sha_char * 64,
        "role": role,
    }


def _summary(pair_count: int, *, same_parent: int = 0) -> dict:
    return {
        "pair_count": pair_count,
        "same_phase_pair_count": pair_count,
        "cross_phase_pair_count": 0,
        "same_parent_group_pair_count": same_parent,
        "phase_pair_counts": {"upstream->upstream": pair_count} if pair_count else {},
        "minimum_max_abs_diff": 0.001 if pair_count else None,
        "maximum_max_abs_diff": 0.005 if pair_count else None,
        "examples": [],
    }


def _diagnostic(tmp_path: Path, *, train_acceptance_pairs: int) -> tuple[Path, dict, dict, dict, dict]:
    tm = _manifest("train", "a")
    cm = _manifest("calibration", "b")
    am = _manifest("acceptance", "c")
    summaries = {
        "train_calibration": _summary(140),
        "train_acceptance": _summary(train_acceptance_pairs),
        "calibration_acceptance": _summary(157),
    }
    exact = {
        "train_calibration": 0,
        "train_acceptance": 0,
        "calibration_acceptance": 0,
    }
    payload = {
        "schema": module.NEAR_DIAGNOSTIC_SCHEMA,
        "status": "completed_read_only_failure_diagnostic",
        "iteration": 1,
        "plan_sha256": tm["plan_sha256"],
        "train_role_manifest_sha256": tm["role_manifest_sha256"],
        "calibration_role_manifest_sha256": cm["role_manifest_sha256"],
        "acceptance_role_manifest_sha256": am["role_manifest_sha256"],
        "actor_observation_atol": 0.01,
        "parent_group_counts": {},
        "exact_overlap_counts": exact,
        "near_overlap": summaries,
        "decision": "automatic_iteration_remains_stopped",
        "claim_boundary": {
            "diagnostic_only": True,
            "isolation_gate_relaxed": False,
            "observation_atol_changed": False,
            "rows_removed_or_reassigned": False,
            "new_environment_interactions": False,
            "test_or_final_data_used": False,
        },
        "training_transitions": 0,
        "environment_interactions": 0,
        "test_data_used": False,
        "final_evaluation_data_used": False,
    }
    payload["diagnostic_sha256"] = canonical_sha256(payload)
    path = tmp_path / "diagnostic.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, tm, cm, am, summaries


def test_engineering_override_allows_calibration_proximity_when_train_acceptance_clean(
    tmp_path: Path,
) -> None:
    path, tm, cm, am, summaries = _diagnostic(tmp_path, train_acceptance_pairs=0)
    result = module.validate_engineering_override(
        path,
        tm=tm,
        cm=cm,
        am=am,
        atol=0.01,
        summaries=summaries,
        exact_counts={
            "train_calibration": 0,
            "train_acceptance": 0,
            "calibration_acceptance": 0,
        },
    )
    assert result["diagnostic_sha256"]


def test_engineering_override_rejects_train_acceptance_near_overlap(tmp_path: Path) -> None:
    path, tm, cm, am, summaries = _diagnostic(tmp_path, train_acceptance_pairs=1)
    with pytest.raises(ValueError, match="TRAIN/ACCEPTANCE near overlap"):
        module.validate_engineering_override(
            path,
            tm=tm,
            cm=cm,
            am=am,
            atol=0.01,
            summaries=summaries,
            exact_counts={
                "train_calibration": 0,
                "train_acceptance": 0,
                "calibration_acceptance": 0,
            },
        )
