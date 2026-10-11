"""Report saved artifacts only; do not re-enter learner verification or physics."""
import json
from pathlib import Path

import pytest


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    from jit_dvgc import retention_t1 as m
    root = tmp_path / 'run'
    plan = root / 'plan.json'
    write(plan, {'output': str(root), 'required_train_roots': ['r0', 'r1', 'r2', 'r3', 'r4'],
                 'train_roots': str(root / 'train_roots.json'),
                 'solver_roots': str(root / 'solver_roots.json'),
                 'budget': {'total': 700200, 'prior_charged': 1122852, 'hard_cap': 2000000}})
    write(root / 'train_roots.json', [{'root': {'root_id': f'r{i}'}} for i in range(8)])
    write(root / 'solver_roots.json', [{'root': {'root_id': f's{i}'}} for i in range(7)])
    write(root / 'status.json', {'phase': 'prepared', 'charged_interactions': 0})
    # Input/code lock checking has separate behavioral tests. This fixture isolates
    # downstream reporting with a pre-audited declaration, not a synthetic vehicle.
    monkeypatch.setattr(m, 'audit', lambda path: {'phase': 'passed'})
    return m, plan, root


def report(m, plan):
    assert hasattr(m, 'report'), 'saved-only T1 report is not implemented'
    return m.report(plan)


def complete_artifacts(m, root):
    labels = lambda n: {g: [1] * n + [0] * (16 - n) for g in 'ABCD'}
    arms = {}
    for arm, n in zip(m.ARMS, (4, 5)):
        train = {f'r{i}': int(i < n) for i in range(8)}
        arms[arm] = dict(checkpoint_transition=32000, DEV=[labels(n + 4) for _ in range(3)],
                         TRAIN=[dict(train) for _ in range(3)],
                         SOLVER_DEV={f's{i}': int(i < 3) for i in range(7)})
        write(root / 'training' / arm / 'status.json',
              {'phase': 'completed', 'completed_training_transitions': 32000})
        write(root / 'training' / arm / 'verification.json',
              {'phase': 'passed', 'completed_training_transitions': 32000,
               'checks': {'normalizer_frozen': True, 'full_state_restore_exact': True}})
    write(root / 'evaluation_results.json', {'source_DEV': [labels(8) for _ in range(3)], 'arms': arms})


def test_prepared_report_is_partial_without_touching_missing_worker_outputs(prepared):
    m, plan, root = prepared
    out = report(m, plan)
    assert out['execution_status'] == 'prepared'
    assert out['evaluation_status'] == 'partial'
    assert out['t1_passed'] is None
    assert not (root / 'training').exists()
    assert out['new_physics_control_steps'] == 0
    assert out['published_policy'] is None


def test_complete_labels_without_verified_final_learners_cannot_pass(prepared):
    m, plan, root = prepared
    complete_artifacts(m, root)
    (root / 'training' / m.ARMS[1] / 'verification.json').unlink()
    out = report(m, plan)
    assert out['evaluation_status'] == 'partial'
    assert out['t1_passed'] is None
    assert out['missing_or_invalid_artifacts']


def test_verified_fixed_endpoint_report_uses_all_repeats_and_does_not_adopt(prepared):
    m, plan, root = prepared
    complete_artifacts(m, root)
    out = report(m, plan)
    assert out['evaluation_status'] == 'complete'
    assert out['t1_passed'] is True
    assert out['arms'][m.ARMS[1]]['stable_train_success'] == 5
    assert out['automatic_next_stage'] is False
    assert out['published_policy'] is None
    # No rerun of verify_saved_training, whose restore hook creates this directory.
    assert not list(root.rglob('restore_verification'))


def test_failed_saved_verification_is_not_hidden_by_complete_labels(prepared):
    m, plan, root = prepared
    complete_artifacts(m, root)
    write(root / 'training' / m.ARMS[0] / 'verification.json',
          {'phase': 'failed', 'completed_training_transitions': 32000,
           'checks': {'normalizer_frozen': False}})
    out = report(m, plan)
    assert out['evaluation_status'] == 'partial'
    assert out['t1_passed'] is None


def test_both_arms_same_wrong_solver_ids_do_not_pass(prepared):
    m, plan, root = prepared
    complete_artifacts(m, root)
    path = root / 'evaluation_results.json'
    evaluation = json.loads(path.read_text())
    for arm in m.ARMS:
        evaluation['arms'][arm]['SOLVER_DEV'] = {f'wrong{i}': int(i < 3) for i in range(7)}
    write(path, evaluation)
    out = report(m, plan)
    assert out['evaluation_status'] == 'partial'
    assert out['t1_passed'] is None
    assert 'solver7' in out['missing_or_invalid_artifacts'][0]
