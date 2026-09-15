from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_cli():
    path = Path(__file__).resolve().parents[1] / "cli" / "analyze_frontier_role_balance.py"
    spec = importlib.util.spec_from_file_location("analyze_frontier_role_balance", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _role(root: Path, *, logical_role: str, upstream_score: float, downstream_score: float, upstream_labels: list[int], downstream_labels: list[int]) -> None:
    acquisition = root / "acquisition"
    labels_dir = root / "labels"
    acquisition.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    phase_protocols = {
        "upstream": {
            "panel": {
                "action_names": ["steer", "rear_wheel_drive", "hip", "knee"],
                "signs": [-1, 1],
                "strengths": [0.025, 0.05, 0.1],
                "durations": [1, 2, 4, 8],
                "active_action_dimensions": 1,
            }
        },
        "downstream": {
            "panel": {
                "action_names": ["steer", "rear_wheel_drive", "hip", "knee"],
                "signs": [-1, 1],
                "strengths": [0.15, 0.3, 0.5],
                "durations": [2, 4, 8],
                "active_action_dimensions": 2,
            }
        },
    }
    protocol = {
        "status": "predeclared",
        "logical_role": logical_role,
        "phase_protocols": phase_protocols,
    }

    candidates = []
    labels = []
    for phase, score, values in (
        ("upstream", upstream_score, upstream_labels),
        ("downstream", downstream_score, downstream_labels),
    ):
        for index, label in enumerate(values):
            candidate_id = f"{logical_role}-{phase}-{index}"
            parent = f"{logical_role}-{phase}-g{index % 2}"
            perturbation = (
                {
                    "action_name": "hip",
                    "sign": 1,
                    "strength": 0.1,
                    "duration": 8,
                }
                if phase == "upstream"
                else {
                    "action_names": ["hip", "knee"],
                    "signs": [1, -1],
                    "basis_vector": [0.0, 0.0, 1.0, -1.0],
                    "strength": 0.5,
                    "duration": 8,
                }
            )
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "phase": phase,
                    "parent_group_id": parent,
                    "parent_state_sha256": ("a" if phase == "upstream" else "b") * 64,
                    "anchor_value_score": score + 0.01 * (index % 2),
                    "anchor_global_index": index % 2,
                    "perturbation": perturbation,
                }
            )
            labels.append(
                {
                    "candidate_id": candidate_id,
                    "label": label,
                    "outcome_class": "success" if label else "physical_failure",
                }
            )

    catalog = {
        "status": "completed",
        "candidate_count": len(candidates),
        "iteration": 1,
        "policy_actor_sha256": "c" * 64,
        "policy_payload_sha256": "d" * 64,
        "protocol_sha256": "e" * 64,
        "phase_attempted_candidate_counts": {
            "upstream": len(upstream_labels) + 2,
            "downstream": len(downstream_labels) + 3,
        },
        "phase_candidate_counts": {
            "upstream": len(upstream_labels),
            "downstream": len(downstream_labels),
        },
        "phase_exclusion_counts": {
            "upstream": {"terminal": 2},
            "downstream": {"terminal": 3},
        },
        "entries": candidates,
    }
    summary = {
        "status": "completed",
        "candidate_count": len(labels),
        "protocol_sha256": "f" * 64,
    }
    (acquisition / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    (acquisition / "catalog.json").write_text(json.dumps(catalog), encoding="utf-8")
    (labels_dir / "labels.json").write_text(json.dumps(labels), encoding="utf-8")
    (labels_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


def test_role_balance_reads_phase_specific_panels_and_compares_scores(tmp_path: Path) -> None:
    cli = _load_cli()
    train = tmp_path / "train"
    calibration = tmp_path / "calibration"
    _role(
        train,
        logical_role="train",
        upstream_score=0.10,
        downstream_score=0.20,
        upstream_labels=[1, 0, 1, 0],
        downstream_labels=[1, 0, 1, 0],
    )
    _role(
        calibration,
        logical_role="calibration",
        upstream_score=0.80,
        downstream_score=0.70,
        upstream_labels=[1, 1, 1, 1],
        downstream_labels=[1, 0, 1, 0],
    )

    result = cli.analyze([("train", train), ("calibration", calibration)])
    assert result["status"] == "completed"
    assert result["roles"]["calibration"]["by_phase"]["upstream"]["negative_count"] == 0
    assert result["roles"]["train"]["by_phase"]["downstream"]["panel"]["active_action_dimensions"] == 2
    train_score = result["comparisons"]["upstream"]["role_anchor_value_score_summaries"]["train"]["mean"]
    calibration_score = result["comparisons"]["upstream"]["role_anchor_value_score_summaries"]["calibration"]["mean"]
    assert calibration_score > train_score
    assert result["comparisons"]["upstream"]["parent_group_overlap"]["train__calibration"]["parent_group_overlap_count"] == 0
