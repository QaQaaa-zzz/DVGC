"""Stable TRAIN-diagnostic API for unified policies."""

from .paired_policy_gate import (
    load_paired_policy_gate_config,
    run_paired_policy_gate,
    summarize_paired_gate_records,
)
from ..unified_diagnostic import run_unified_fixed_panel
from ..unified_natural_evaluation import run_canonical_natural_evaluation

__all__ = [
    "load_paired_policy_gate_config",
    "run_paired_policy_gate",
    "summarize_paired_gate_records",
    "run_unified_fixed_panel",
    "run_canonical_natural_evaluation",
]
