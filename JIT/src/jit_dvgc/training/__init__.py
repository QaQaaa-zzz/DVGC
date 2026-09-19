"""Stable unified training and freeze API.

The package root is the production import surface. Historical flat modules stay
loadable only for frozen-artifact and external-import compatibility.
"""

from .formal import (
    FORMAL_SCHEMA,
    FORMAL_TARGET,
    build_unified_formal_environment,
    load_unified_formal_config,
    preflight_unified_formal_tube,
    run_unified_formal,
)
from ..unified_policy_freeze import freeze_unified_policy, load_frozen_unified_manifest
from ..unified_training import (
    PILOT_SCHEMA,
    checkpoint_identity,
    load_unified_pilot_config,
    read_json,
    run_unified_pilot,
)

__all__ = [
    "FORMAL_SCHEMA",
    "FORMAL_TARGET",
    "build_unified_formal_environment",
    "load_unified_formal_config",
    "preflight_unified_formal_tube",
    "run_unified_formal",
    "freeze_unified_policy",
    "load_frozen_unified_manifest",
    "PILOT_SCHEMA",
    "checkpoint_identity",
    "load_unified_pilot_config",
    "read_json",
    "run_unified_pilot",
]
