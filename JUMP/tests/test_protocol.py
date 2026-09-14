import copy
import json
from pathlib import Path
import pytest

from jump_planning.protocol import validate_config, reserve_budget, close_budget

ROOT = Path(__file__).resolve().parents[1]


def test_declared_first_batch_is_bounded():
    config = json.loads((ROOT / 'configs/qualification.json').read_text())
    assert validate_config(config) == 4800
    config['budget']['max_control_steps'] = 4799
    with pytest.raises(ValueError, match='budget'):
        validate_config(config)


def test_timing_and_undefined_numbers_rejected():
    base = json.loads((ROOT / 'configs/qualification.json').read_text())
    for section, key, value in [('timing', 'sim_dt', .001), ('scene', 'height', float('nan')), ('limits', 'min_forward_speed', -1)]:
        config = copy.deepcopy(base)
        config[section][key] = value
        with pytest.raises(ValueError):
            validate_config(config)


def test_budget_reserves_inflight_and_accounts_actuals(tmp_path):
    path = tmp_path / 'ledger.json'
    campaign = {'total_control_steps': 100, 'stages': {'qualification_engineering': 100}}
    reserve_budget(path, campaign, 'first', 'qualification_engineering', 80)
    with pytest.raises(ValueError, match='budget'):
        reserve_budget(path, campaign, 'second', 'qualification_engineering', 30)
    close_budget(path, 'first', 20, 'completed')
    reserve_budget(path, campaign, 'second', 'qualification_engineering', 80)
    with pytest.raises(ValueError, match='reservation'):
        close_budget(path, 'second', 81, 'completed')
    assert json.loads(path.read_text())['runs']['first']['charged_control_steps'] == 20


def test_duplicate_run_does_not_reset_spent_budget(tmp_path):
    path = tmp_path / 'ledger.json'
    campaign = {'total_control_steps': 100, 'stages': {'qualification_engineering': 100}}
    reserve_budget(path, campaign, 'same', 'qualification_engineering', 80)
    close_budget(path, 'same', 80, 'completed')
    with pytest.raises(ValueError, match='already'):
        reserve_budget(path, campaign, 'same', 'qualification_engineering', 80)
