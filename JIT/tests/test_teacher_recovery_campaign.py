import json
from pathlib import Path
import pytest


def test_approved_campaign_parameters_and_budget():
    from jit_dvgc.generative_bridge.recovery_campaign import campaign_parameters
    c = campaign_parameters()
    assert c['rounds'] == 100
    assert c['teacher_query_count'] == c['explorer']['num_envs'] == 128
    assert c['explorer']['learning_rate'] == c['explorer']['max_learning_rate'] == 1e-4
    assert c['explorer']['entropy_coefficient'] == .001
    assert c['student']['student_transitions'] == 32000
    assert c['teacher_reviews'] == 2
    assert c['student']['ppo_coefficient'] == .1
    assert c['student']['retention_batch_size'] == 512
    assert c['physics_cap_per_round'] >= 128*400*(1+17*3)+32000+9*32*400


def test_snapshot_reset_time_is_locked_original_clock(tmp_path):
    from jit_dvgc.generative_bridge.recovery_campaign import snapshot_entry
    from jit_dvgc.generative_bridge.contracts import file_sha
    snapshot = tmp_path/'snapshot'
    snapshot.mkdir()
    p = snapshot/'identity.json'
    p.write_text(json.dumps({'episode_step': 18}))
    row = dict(snapshot=str(snapshot), snapshot_context_sha256='ctx')
    entry = snapshot_entry(row, 3)
    assert entry['snapshot_time'] == pytest.approx(.36)
    assert entry['snapshot_time_source']['sha256'] == file_sha(p)
    assert entry['snapshot_time_source']['control_step_field'] == 'episode_step'
    assert entry['verified_success'] is True


def test_failed_or_unknown_teachers_are_not_reset_successes():
    from jit_dvgc.generative_bridge.recovery_campaign import verified_teacher_roots
    rows = [{'root_id': 'a'}, {'root_id': 'b'}, {'root_id': 'c'}]
    teachers = {'a': {'teacher_status': 'searched_no_solution'},
        'b': {'teacher_status': 'verified_solution', 'verification': {'repeat_labels':[1,0]}},
        'c': {'teacher_status': 'verified_solution', 'verification': {'repeat_labels':[1,1]}, 'demo':{}}}
    assert [r['root_id'] for r in verified_teacher_roots(rows, teachers)] == ['c']
