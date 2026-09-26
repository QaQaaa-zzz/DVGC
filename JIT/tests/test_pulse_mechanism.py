import hashlib
import json
from pathlib import Path

import pytest

from jit_dvgc.pulse_mechanism import prepare_mechanism


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


@pytest.fixture
def preparation(tmp_path, monkeypatch):
    import jit_dvgc.unified_policy_freeze as freeze

    repo = tmp_path / 'repo'
    write(repo / 'JIT/src/jit_dvgc/fixture.py', {})
    write(repo / 'JIT/cli/run_pulse_exploration.py', {})
    down = write(repo / 'JIT/configs/down.json', {
        'descent': {'continuous_stability': True, 'recovery_ticks': 25,
                    'min_post_contact_forward_progress': 0.0},
        'model': {}})
    up = write(repo / 'JIT/configs/up.json', {'model': {}})
    config = write(repo / 'JIT/configs/source.json', {
        'success_criterion': 'stable_forward_recovery',
        'inputs': {'up_config_path': str(up), 'down_config_path': str(down)}})
    checkpoint = repo / 'JIT/runs/source/checkpoint'
    write(checkpoint / 'identity.json', {})
    (checkpoint / 'payload.pkl').write_bytes(b'frozen actor payload')
    record = {'name': 'source', 'xml_sha256': 'a' * 64,
              'actor_sha256': '1' * 64, 'normalizer_sha256': '2' * 64,
              'formal_config': str(config), 'checkpoint': str(checkpoint)}
    frozen = write(repo / 'JIT/runs/source/frozen.json', {'policy': record})
    task_template = write(repo / 'JIT/runs/history/bank.json', {
        'task': {'xml_sha256': 'a' * 64, 'start_contract_sha256': 'b' * 64,
                 'centerline_sha256': 'c' * 64, 'resolution_sha256': 'd' * 64,
                 'success_criterion': 'first_valid_landing',
                 'continuation_start_semantics': 'fresh_continuation_v1'},
        'members': [{'name': 'forbidden_legacy_policy'}],
        'support': 'do-not-read.json', 'success_count': 99})
    # Frozen-policy loading verifies checkpoint internals in its own tests;
    # this test exercises preparation, file locking and independent routing.
    monkeypatch.setattr(freeze, 'load_frozen_unified_manifest',
                        lambda path: json.loads(Path(path).read_text()))
    return dict(initial_frozen_policy=frozen, task_template_bank=task_template,
                jump_start_state_sha256='e' * 64, output=repo / 'JIT/runs/new',
                seed=11, repo=repo, python='/usr/bin/python3')


def test_prepared_arms_share_only_frozen_source_and_cross_event_amplitude(preparation):
    result = prepare_mechanism(**preparation)
    root = preparation['output']
    specs = [json.loads((root / arm / 'spec.json').read_text())
             for arm in ('learned', 'fixed_random')]
    assert [s['controller_mode'] for s in specs] == ['learned_residual', 'fixed_random']
    for spec in specs:
        bank = json.loads(Path(spec['bank']).read_text())
        assert [m['name'] for m in bank['members']] == ['source']
        assert bank['members'][0]['frozen_policy'] == str(preparation['initial_frozen_policy'])
        assert bank['task']['success_criterion'] == 'stable_forward_recovery'
        assert 'support' not in bank and 'success_count' not in bank
        assert spec['training_mode'] == 'fixed_policy'
        assert spec['quality_mode'] == 'current_policy'
        assert spec['iteration_mode'] == 'current_policy_only_v1'
        assert spec['maximum_actual_interactions'] == 2_500_000
        assert not any(k in spec for k in ('resume_run', 'witnessed_support', 'reuse_results'))
        pairs = list(zip(spec['pulse_event_schedule'], spec['delta_limit_schedule']))
        assert len(pairs) == 8
        assert {(event, tuple(amp)) for event, amp in pairs} == {
            (event, (amplitude,) * 4)
            for event in ('start', 'liftoff', 'apex', 'descent')
            for amplitude in (.1, .15)}
    assert specs[0]['bank'] != specs[1]['bank']
    assert result['max_interactions'] == 5_000_000
    assert [s['name'] for s in result['stages']] == ['learned', 'fixed_random']
    assert all(s['env']['JAX_PLATFORMS'] == 'cpu' for s in result['stages'])
    assert all(s['gate']['kind'] == 'gpu_idle' for s in specs)
    assert not (root / 'execution').exists()
    assert not (root / 'learned/lineage').exists()


def test_preparation_locks_configs_payload_and_runtime_and_refuses_overwrite(preparation):
    from jit_dvgc.gated_execution import _verify_locks

    plan = prepare_mechanism(**preparation)
    path = preparation['output'] / 'plan.json'
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    _verify_locks(plan, path, digest)
    config = preparation['repo'] / 'JIT/configs/down.json'
    config.write_text(config.read_text() + '\n')
    with pytest.raises(ValueError, match='hash mismatch'):
        _verify_locks(plan, path, digest)
    with pytest.raises(FileExistsError):
        prepare_mechanism(**preparation)


def test_preparation_rejects_wrong_recovery_window_before_creating_output(preparation):
    config = preparation['repo'] / 'JIT/configs/down.json'
    value = json.loads(config.read_text())
    value['descent']['recovery_ticks'] = 100
    write(config, value)
    with pytest.raises(ValueError, match='recovery'):
        prepare_mechanism(**preparation)
    assert not preparation['output'].exists()


@pytest.mark.parametrize('amplitudes', [[], [0], [float('nan')], [.1, .1], [1.1]])
def test_preparation_rejects_ambiguous_or_invalid_amplitude_schedule(preparation, amplitudes):
    with pytest.raises(ValueError, match='amplitude'):
        prepare_mechanism(**preparation, amplitudes=amplitudes)
    assert not preparation['output'].exists()
