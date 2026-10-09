"""Successful offline work must commit before bypassing native teardown."""
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest


def test_successful_process_avoids_faulty_native_finalizer(tmp_path):
    script = '''
import atexit, os
from jit_dvgc.generative_bridge.worker_lifecycle import run_generator_process
from jit_dvgc.generative_bridge import worker
import jit_dvgc.generative_bridge.worker_lifecycle as lifecycle
atexit.register(os.abort)
worker.run_generator = lambda config: None
lifecycle.commit_generator_result = lambda config: print('COMMITTED', flush=True)
run_generator_process({})
'''
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout == 'COMMITTED\n'


def test_failed_work_never_commits_or_exits(monkeypatch):
    from jit_dvgc.generative_bridge import worker, worker_lifecycle as lifecycle
    calls = []
    def fail(config):
        raise RuntimeError('training failure')
    monkeypatch.setattr(worker, 'run_generator', fail)
    monkeypatch.setattr(lifecycle, 'commit_generator_result', lambda c: calls.append('commit'))
    monkeypatch.setattr(os, '_exit', lambda code: calls.append(code))
    with pytest.raises(RuntimeError, match='training failure'):
        lifecycle.run_generator_process({})
    assert calls == []


def test_failed_validation_never_exits(monkeypatch):
    from jit_dvgc.generative_bridge import worker, worker_lifecycle as lifecycle
    calls = []
    monkeypatch.setattr(worker, 'run_generator', lambda config: None)
    def fail(config):
        raise ValueError('checkpoint corrupt')
    monkeypatch.setattr(lifecycle, 'commit_generator_result', fail)
    monkeypatch.setattr(os, '_exit', lambda code: calls.append(code))
    with pytest.raises(ValueError, match='checkpoint corrupt'):
        lifecycle.run_generator_process({})
    assert calls == []


def _fixture(tmp_path):
    from jit_dvgc.generative_bridge.diffusion import create_train_state, save_state
    from jit_dvgc.generative_bridge.contracts import file_sha
    from jit_dvgc.generative_bridge.protocol import atomic_json
    import jax.numpy as jp
    state = create_train_state({'w': jp.ones(2)}, jp.array([1,2], dtype=jp.uint32),
                               {'mean':jp.zeros(76),'std':jp.ones(76)})
    root = tmp_path/'generator'
    checkpoint = root/'attempt_0000/update_0002'
    save_state(tmp_path/'incumbent', state, {})
    incumbent_manifest = tmp_path/'incumbent/manifest.json'
    incumbent = dict(checkpoint_manifest=str(incumbent_manifest),
        checkpoint_manifest_sha256=file_sha(incumbent_manifest), state_updates=0)
    state['updates'] = jp.array(2)
    save_state(checkpoint, state, {})
    from jit_dvgc.handoff_bank import pytree_sha256
    manifest = checkpoint/'manifest.json'
    result = dict(status='completed', checkpoint_manifest=str(manifest),
        checkpoint_manifest_sha256=file_sha(manifest), updates=2, state_updates=2,
        generator_update_policy='last_valid', inference_parameters='ema', corpus_sha256='corpus',
        initial_state_updates=0,charged_updates_before=10,charged_updates=2,total_charged_updates=12,
        selected_state_sha256=pytree_sha256(state))
    atomic_json(root/'completed.json', result)
    atomic_json(root/'attempt_0000/cost_receipt.json',dict(status='completed',charged_updates=2))
    atomic_json(root/'attempt_0000/generator_selection.json',dict(status='completed',updates=2,selected='update_0002'))
    config = dict(mode='incremental',output=str(root),result=str(tmp_path/'result.json'),updates=2,
        generator_update_policy='last_valid', corpus={'sha256':'corpus'},
        incumbent=incumbent, charged_updates_before=10)
    atomic_json(config['result'], result)
    return config, result


def test_commit_checks_and_syncs_real_full_state(tmp_path):
    from jit_dvgc.generative_bridge.worker_lifecycle import commit_generator_result
    config, result = _fixture(tmp_path)
    receipt = commit_generator_result(config)
    assert receipt['status'] == 'committed'
    assert receipt['state_updates'] == 2
    assert Path(config['output'], 'process_commit.json').is_file()


@pytest.mark.parametrize('failure', ['missing_completed', 'result_mismatch', 'corrupt_state', 'incomplete_cost'])
def test_commit_rejects_partial_or_changed_result(tmp_path, failure):
    from jit_dvgc.generative_bridge.worker_lifecycle import commit_generator_result
    config, result = _fixture(tmp_path)
    root = Path(config['output'])
    if failure == 'missing_completed':
        (root/'completed.json').unlink()
    elif failure == 'result_mismatch':
        (root/'completed.json').write_text('{}')
    elif failure == 'corrupt_state':
        Path(result['checkpoint_manifest']).with_name('state.msgpack').write_bytes(b'bad')
    else:
        (root/'attempt_0000/cost_receipt.json').write_text('{"status":"failed","charged_updates":2}')
    with pytest.raises((ValueError, OSError)):
        commit_generator_result(config)
    assert not (root/'process_commit.json').exists()


@pytest.mark.parametrize('failure', ['incumbent_age', 'incumbent_hash', 'charged_before', 'selected_hash', 'total_charge'])
def test_commit_rejects_false_training_lineage(tmp_path, failure):
    from jit_dvgc.generative_bridge.worker_lifecycle import commit_generator_result
    from jit_dvgc.generative_bridge.protocol import atomic_json
    config, result = _fixture(tmp_path)
    if failure == 'incumbent_age':
        config['incumbent']['state_updates'] = 100
    elif failure == 'incumbent_hash':
        config['incumbent']['checkpoint_manifest_sha256'] = 'bad'
    elif failure == 'charged_before':
        config['charged_updates_before'] = 100
    elif failure == 'selected_hash':
        result['selected_state_sha256'] = 'bad'
    else:
        result['total_charged_updates'] = 13
    atomic_json(config['result'], result)
    atomic_json(Path(config['output'])/'completed.json', result)
    with pytest.raises((ValueError, OSError)):
        commit_generator_result(config)
    assert not (Path(config['output'])/'process_commit.json').exists()


def test_explicit_retry_preserves_prior_charges(tmp_path):
    from jit_dvgc.generative_bridge.worker_lifecycle import commit_generator_result
    from jit_dvgc.generative_bridge.protocol import atomic_json
    config, result = _fixture(tmp_path)
    root = Path(config['output'])
    (root/'attempt_0000').rename(root/'attempt_0001')
    atomic_json(root/'attempt_0000/cost_receipt.json',dict(status='failed',charged_updates=1))
    result['checkpoint_manifest']=str(root/'attempt_0001/update_0002/manifest.json')
    result['charged_updates']=3
    result['total_charged_updates']=13
    config['updates']=3
    atomic_json(config['result'], result)
    atomic_json(root/'completed.json', result)
    assert commit_generator_result(config)['state_updates']==2
