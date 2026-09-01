"""Compatibility tests for the categorized JIT import surface."""

from __future__ import annotations


def test_canonical_facades_preserve_legacy_authorities():
    from jit_dvgc import core_retaining_tube_iteration as legacy_iteration
    from jit_dvgc import policy_conditioned_continuation_field as legacy_field
    from jit_dvgc import snapshot_pool as legacy_pool
    from jit_dvgc import soft_tube as legacy_soft
    from jit_dvgc import tube_rsi_smoke as legacy_smoke
    from jit_dvgc import unified_boundary as legacy_boundary
    from jit_dvgc import unified_continuation_labels as legacy_labels
    from jit_dvgc import unified_diagnostic as legacy_diagnostic
    from jit_dvgc import unified_policy_freeze as legacy_freeze
    from jit_dvgc import unified_training as legacy_training

    from jit_dvgc.acquisition import boundary
    from jit_dvgc.analysis import unified as analysis_unified
    from jit_dvgc.continuation import field, labels
    from jit_dvgc.snapshots import pool
    from jit_dvgc.training import freeze, unified
    from jit_dvgc.tube import iteration, smoke, soft

    assert soft.load_soft_tube is legacy_soft.load_soft_tube
    assert smoke.run_tube_rsi_smoke is legacy_smoke.run_tube_rsi_smoke
    assert iteration.build_core_retaining_tube is legacy_iteration.build_core_retaining_tube
    assert pool.SnapshotPool is legacy_pool.SnapshotPool
    assert unified.checkpoint_identity is legacy_training.checkpoint_identity
    assert freeze.freeze_unified_policy is legacy_freeze.freeze_unified_policy
    assert boundary.collect_unified_boundary_candidates is legacy_boundary.collect_unified_boundary_candidates
    assert labels.label_unified_continuations is legacy_labels.label_unified_continuations
    assert field.fit_policy_conditioned_continuation_fields is legacy_field.fit_policy_conditioned_continuation_fields
    assert analysis_unified.run_unified_fixed_panel is legacy_diagnostic.run_unified_fixed_panel


def test_formal_namespace_adds_zero_interaction_preflight_without_redefining_contract():
    from jit_dvgc import unified_formal as legacy_formal
    from jit_dvgc.training import formal

    assert formal.FORMAL_SCHEMA == legacy_formal.FORMAL_SCHEMA
    assert formal.FORMAL_TARGET == legacy_formal.FORMAL_TARGET
    assert formal.load_unified_formal_config is legacy_formal.load_unified_formal_config
    assert formal.run_unified_formal is not legacy_formal.run_unified_formal
    assert callable(formal.preflight_unified_formal_tube)
