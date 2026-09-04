"""Stable analysis API for unified-policy JIT evidence."""

from .capability_progression import (
    analyze_capability_progression,
    analyze_capability_progression_file,
)
from .capability_tube import (
    build_capability_tube_geometry,
    load_projected_capability_entries,
    physical_coordinates_from_arrays,
    quantize_coordinates,
    resolution_contract,
)
from .paired_policy_gate import (
    load_paired_policy_gate_config,
    prepare_locked_bank_gate_config,
    run_paired_policy_gate,
    summarize_paired_gate_records,
)
from ..unified_diagnostics import run_unified_fixed_panel
from ..unified_natural_evaluation import run_canonical_natural_evaluation

__all__ = [
    "analyze_capability_progression",
    "analyze_capability_progression_file",
    "build_capability_tube_geometry",
    "load_projected_capability_entries",
    "physical_coordinates_from_arrays",
    "quantize_coordinates",
    "resolution_contract",
    "load_paired_policy_gate_config",
    "prepare_locked_bank_gate_config",
    "run_paired_policy_gate",
    "summarize_paired_gate_records",
    "run_unified_fixed_panel",
    "run_canonical_natural_evaluation",
]
