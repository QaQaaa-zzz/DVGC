"""Canonical namespace for formal unified PPO orchestration.

The production CLI enters through this module. Before constructing MJX or
spending an environment interaction, it performs a static Soft-Tube snapshot
preflight so schema/plotting incompatibilities fail at zero interactions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

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


def run_unified_formal(config_path: Path, run_id: str, **kwargs: Any) -> dict[str, Any]:
    """Run formal unified PPO after a zero-interaction static Tube preflight."""
    preflight_unified_formal_tube(config_path)
    return _run_unified_formal(config_path, run_id, **kwargs)
