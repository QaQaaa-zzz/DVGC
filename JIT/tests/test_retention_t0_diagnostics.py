"""T0 must reject mismatched execution, data leakage, and mutable outputs."""
import json
from pathlib import Path

import numpy as np
import pytest


def test_calibration_is_pi0_only_and_nonfinite_fails_closed():
    from jit_dvgc.retention_t0_diagnostics import calibration_tolerance
    assert calibration_tolerance([0., 2e-7]) == pytest.approx(2e-6)
    assert calibration_tolerance([0.]) == 1e-6
    with pytest.raises(ValueError, match='finite'):
        calibration_tolerance([float('nan')])
    with pytest.raises(ValueError, match='nonempty'):
        calibration_tolerance([])


def test_chain_mismatch_blocks_t1_without_relaxing_tolerance():
    from jit_dvgc.retention_t0_diagnostics import chain_gate
    gate = chain_gate([{'maxabs': 2e-5, 'name': 'wrong_actor'}], 1e-6)
    assert gate['phase'] == 'blocked'
    assert gate['t1_allowed'] is False
    assert gate['tolerance'] == 1e-6
    assert gate['blocking_failures'][0]['name'] == 'wrong_actor'


def test_nonfinite_or_missing_chain_cannot_pass():
    from jit_dvgc.retention_t0_diagnostics import chain_gate
    assert not chain_gate([{'maxabs': float('nan'), 'name': 'bad'}], 1e-6)['t1_allowed']
    assert not chain_gate([], 1e-6)['t1_allowed']
    assert chain_gate([{'maxabs': 2e-7, 'name': 'matched'}], 1e-6)['t1_allowed']


def test_role_guard_rejects_dev_before_arrays_are_loaded():
    from jit_dvgc.retention_t0_diagnostics import require_train
    require_train({'role': 'TRAIN'})
    with pytest.raises(ValueError, match='TRAIN'):
        require_train({'role': 'DEV', 'path': '/must/not/open'})
    with pytest.raises(ValueError, match='TRAIN'):
        require_train({'role': 'TRAIN', 'DEV_used': True})


def test_same_state_error_is_separate_from_trajectory_drift():
    from jit_dvgc.retention_t0_diagnostics import error_statistics
    result = error_statistics(np.array([[1., 0., 0., 0.], [0., 0., 0., 0.]]), np.zeros((2, 4)))
    assert result['n'] == 2
    assert result['mse_by_action'] == [.5, 0., 0., 0.]
    assert result['mse'] == .125
    assert result['maxabs'] == 1.


def test_read_registry_detects_changed_source_and_no_overwrite(tmp_path):
    from jit_dvgc.retention_t0_diagnostics import SourceRegistry, create_output
    source = tmp_path / 'source.json'
    source.write_text('{"a": 1}')
    registry = SourceRegistry()
    assert registry.json(source) == {'a': 1}
    source.write_text('{"a": 2}')
    with pytest.raises(ValueError, match='changed'):
        registry.verify_unchanged()
    out = create_output(tmp_path / 'report', [tmp_path / 'original'])
    with pytest.raises(FileExistsError):
        create_output(out, [tmp_path / 'original'])
    with pytest.raises(ValueError, match='source'):
        create_output(tmp_path / 'original' / 'report', [tmp_path / 'original'])


def test_actor_reconstruction_normalizes_once_and_uses_first_four_logits():
    from jit_dvgc.retention_t0_diagnostics import independent_forward
    actor = {'params': {'hidden_0': {'kernel': np.eye(2, dtype=np.float32), 'bias': np.zeros(2)},
                        'hidden_1': {'kernel': np.array([[1., 0., 0., 0., 50., 50., 50., 50.],
                                                       [0., 1., 0., 0., 50., 50., 50., 50.]]),
                                     'bias': np.zeros(8)}}}
    got = independent_forward(actor, np.array([1., 1.]), np.array([2., 2.]), np.array([[3., 1.]]))
    expected = np.array([[np.tanh(1. / (1. + np.exp(-1.))), 0., 0., 0.]])
    np.testing.assert_allclose(got, expected, atol=1e-6)


def test_gradient_diagnostics_preserve_opposition_without_updates():
    from jit_dvgc.retention_t0_diagnostics import gradient_cosine
    assert gradient_cosine(np.array([1., 2.]), np.array([-1., -2.])) == pytest.approx(-1.)
    assert gradient_cosine(np.zeros(2), np.ones(2)) is None


def test_physical_drift_does_not_use_calibrated_action_gate_tolerance():
    from jit_dvgc.retention_t0_diagnostics import drift_statistics
    a = np.zeros((3, 2)); b = a.copy(); b[1, 0] = 2e-6; b[2, 0] = .01
    result = drift_statistics(a, b, 'qvel')
    assert result['first_tick'] == 1
    assert result['threshold_per_component'] == 1e-6
    assert result['gate_used'] is False
    assert 'm/s' in result['units']


def test_native_cpu_gap_is_separated_from_independent_reconstruction_error():
    from jit_dvgc.retention_t0_diagnostics import reconstruction_statistics
    own = np.full((2, 4), .1)
    native = np.full((2, 4), .1000001)
    saved = np.full((2, 4), .1005)
    result = reconstruction_statistics(own, native, saved)
    assert result['maxabs'] == pytest.approx(.0005)
    assert result['native_cpu_vs_saved_maxabs'] == pytest.approx(.0004999)
    assert result['independent_cpu_vs_native_maxabs'] == pytest.approx(.0000001)
