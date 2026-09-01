"""Canonical namespace for formal unified PPO orchestration.

The production CLI enters through this module. Before constructing MJX or
spending an environment interaction, it performs a static Soft-Tube snapshot
preflight so schema/plotting incompatibilities fail at zero interactions.

The wrapper also reconciles a newly failed run from persisted TRAIN-panel
reports.  The shared callback controller can only account a panel after the
panel function returns; if persistence/plotting fails after rollout, the report
already contains the true interaction count and must remain the accounting
authority for that failed run.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from ..soft_tube import load_soft_tube
from ..unified_diagnostic import _tube_points
from ..unified_formal import *  # noqa: F401,F403
from ..unified_formal import run_unified_formal as _run_unified_formal


def preflight_unified_formal_tube(config_path: Path) -> dict[str, Any]:
    """Validate the full configured Tube support without constructing an env."""
    config = load_unified_formal_config(Path(config_path))
    artifact = load_soft_tube(Path(config.soft_tube_path))
    actual_manifest = artifact.manifest.get("manifest_sha256")
    if actual_manifest != config.soft_tube_manifest_sha256:
        raise ValueError("formal preflight Soft-Tube manifest identity mismatch")
    points = _tube_points(artifact)
    if len(points) != len(artifact.entries):
        raise ValueError("formal preflight Tube point count mismatch")
    return {
        "soft_tube_manifest_sha256": str(actual_manifest),
        "entry_count": len(points),
        "environment_interactions": 0,
        "training_transitions": 0,
    }


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def _persisted_train_panel_interactions(run_dir: Path) -> int:
    """Return interactions proven by completed persisted TRAIN-panel reports."""
    panel_root = Path(run_dir) / "train_panels"
    if not panel_root.is_dir():
        return 0
    total = 0
    seen_transitions: set[int] = set()
    for report_path in sorted(panel_root.glob("transition_*/report.json")):
        report = _read_json_object(report_path)
        transition = int(report.get("training_checkpoint_transition", -1))
        interactions = int(report.get("environment_interactions", -1))
        if transition < 0 or interactions <= 0:
            raise ValueError(
                f"invalid persisted TRAIN-panel accounting report: {report_path}"
            )
        if transition in seen_transitions:
            raise ValueError(
                f"duplicate persisted TRAIN-panel transition: {transition}"
            )
        seen_transitions.add(transition)
        total += interactions
    return total


def _reconcile_failed_train_panel_accounting(run_dir: Path) -> int:
    """Repair only a newly closed engineering-error run from persisted evidence.

    Historical run artifacts are never scanned or rewritten by this function;
    the caller passes exactly the run directory created by the current launch.
    """
    status_path = Path(run_dir) / "status.json"
    if not status_path.is_file():
        return 0
    status = _read_json_object(status_path)
    if status.get("status") != "engineering_error":
        return 0
    accounting = status.get("interaction_accounting")
    if not isinstance(accounting, dict):
        raise ValueError("failed unified run status has no interaction accounting")
    persisted = _persisted_train_panel_interactions(run_dir)
    recorded = int(accounting.get("diagnostic", 0))
    if persisted <= recorded:
        return recorded
    accounting = dict(accounting)
    accounting["diagnostic"] = persisted
    status = dict(status)
    status["interaction_accounting"] = accounting
    status["environment_transitions"] = sum(int(value) for value in accounting.values())
    status["accounting_reconciliation"] = {
        "source": "persisted_train_panel_reports",
        "previous_diagnostic_interactions": recorded,
        "reconciled_diagnostic_interactions": persisted,
    }
    _atomic_json(status_path, status)
    return persisted


def run_unified_formal(config_path: Path, run_id: str, **kwargs: Any) -> dict[str, Any]:
    """Run formal unified PPO after a zero-interaction static Tube preflight."""
    preflight_unified_formal_tube(config_path)
    root = Path(
        kwargs.get("run_root")
        or os.environ.get("JIT_RUN_ROOT", "JIT/runs/pi_unified")
    )
    run_dir = root / run_id
    try:
        return _run_unified_formal(config_path, run_id, **kwargs)
    except Exception:
        _reconcile_failed_train_panel_accounting(run_dir)
        raise
