import json
from pathlib import Path

from jump_planning.runner import execute_cases
import pytest


def config():
    path = Path(__file__).resolve().parents[1] / 'configs/qualification.json'
    value = json.loads(path.read_text())
    value['policies'] = value['policies'][:1]
    value['cases'] = value['cases'][:2]
    return value


def test_exception_preserves_unknown_and_conservative_cost(tmp_path):
    def fail(*args):
        raise RuntimeError('deliberate fixture error')
    cases, accounting = execute_cases(config(), tmp_path, tmp_path, case_runner=fail)
    assert len(cases) == 2
    assert [row['status'] for row in cases] == ['unknown', 'unknown']
    assert cases[0]['charged_control_steps'] == 400
    assert cases[1]['charged_control_steps'] == 0
    assert accounting['charged_control_steps'] == 400
    assert accounting['engineering_errors'] == 1
    assert cases[1]['reason'] == 'not_executed_after_engineering_error'


def test_partial_control_and_trace_kept(tmp_path):
    def finite(*args):
        return ({'status': 'failure', 'reason': 'body_contact', 'qualified': False,
                 'charged_control_steps': 1, 'physics_steps': 2},
                [{'time': .005, 'finite': True}, {'time': .01, 'finite': False, 'vx': float('nan')}])
    cases, accounting = execute_cases(config(), tmp_path, tmp_path, case_runner=finite)
    assert accounting['charged_control_steps'] == 2
    assert accounting['physics_steps'] == 4
    rows = [json.loads(line) for line in (tmp_path / cases[0]['trajectory']).read_text().splitlines()]
    assert len(rows) == 2 and rows[1]['vx'] is None and rows[1]['finite'] is False


def test_invalid_configuration_has_terminal_error_receipt(tmp_path):
    import jump_planning.runner as runner
    broken = tmp_path / 'broken.json'
    broken.write_text('{broken JSON')
    output = tmp_path / 'JUMP/runs/invalid'
    with pytest.raises(ValueError):
        runner.run_declared(broken, tmp_path / 'unused.json', tmp_path, output, tmp_path)
    status = json.loads((output / 'status.json').read_text())
    assert status['phase'] == 'error'
    assert status['charged_control_steps'] == 0
    assert (output / 'ACTIVE_RUN.json').exists()
    assert not (tmp_path / 'JUMP/runs/campaign_ledger.json').exists()
