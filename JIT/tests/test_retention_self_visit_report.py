"""Read-only report contracts for the bounded self-visited demo stage."""
import numpy as np
import pytest

from jit_dvgc.retention_b import DEMO_SCHEMA, self_visit_demo_budget
from jit_dvgc.retention_b_report import analyze, build_report
from jit_dvgc.retention_repair import read, write


def prepared(tmp_path):
    parent = tmp_path / 'parent'
    parent.mkdir()
    source_report = parent / 'coverage_report_0002'
    source_report.mkdir()
    write(source_report / 'TRAIN_repeat_labels.json', [])
    write(parent / 'plan.json', dict(output=str(parent)))
    write(tmp_path / 'demo.json', dict(old_count=525, new_count=228, count=753, eligible_query_count=4))
    write(tmp_path / 'keep.json', dict(summary=dict(observations=dict(nominal=1566, random=4723))))
    write(tmp_path / 'train.json', [])
    write(tmp_path / 'solver.json', [])
    plan = dict(schema=DEMO_SCHEMA, output=str(tmp_path), source_B=str(parent / 'plan.json'),
                source_report=str(source_report), budget=self_visit_demo_budget(975315),
                original_D2_started_unix=10, demo_started_unix=100,
                BC=dict(updates=1000, demo_coefficient=1., keep_coefficient=1., checkpoints=[100, 500, 1000]),
                only_changed_factor='TRAIN_student_visited_demo_support_at_matched_updates',
                best_dev_candidate=dict(reference='R5'), demo_manifest=str(tmp_path / 'demo.json'),
                keep_receipt=str(tmp_path / 'keep.json'), train_roots=str(tmp_path / 'train.json'),
                solver_roots=str(tmp_path / 'solver.json'), keep_collection=None)
    write(tmp_path / 'plan.json', plan)
    write(tmp_path / 'status.json', dict(phase='prepared', charged_interactions=0, updated_unix=120,
                                         completed_supervised_updates=0))
    return plan


def row(train=0, solver=0, lost=0):
    return dict(train=[], solver=[], train_success=train, solver_success=solver, eligible=True,
                cells={g:dict(N01=0, N10=lost if g=='A' else 0, success=0) for g in 'ABCD'})


def test_prepared_routes_demo_schema_and_describes_only_actual_intervention(tmp_path):
    p = prepared(tmp_path)
    out = build_report(tmp_path / 'plan.json')
    s = read(out / 'summary.json')
    text = (out / 'INDEX.md').read_text()
    assert out.name == 'self_visit_demo_report_0001'
    assert s['evaluation_status'] == 'not_started'
    assert s['new_keep_collection'] == 'not_collected_fixed_source_reused'
    assert s['demo_support'] == dict(old_frames=525, added_frames=228, total_frames=753, added_queries=4)
    assert s['fixed_keep_frames'] == 6289
    assert s['stage_deadline_unix'] == 10900
    assert s['source_report'] == p['source_report']
    assert s['reliable_recovery_improvement'] == 'NOT_ESTABLISHED'
    assert 'BC1000' in text and 'BC2000' not in text
    assert '525 + 228 = 753' in text and '6289' in text
    assert 'demo_started_unix=100' in text
    assert 'frame counts are not sampling proportions' in text
    assert 'PENDING until physical collection' not in text
    assert 'improved' not in text.lower()


def test_partial_reports_only_available_matched_nodes_without_selection_claim(tmp_path):
    p = prepared(tmp_path)
    write(tmp_path / 'status.json', dict(phase='failed', updated_unix=160, completed_supervised_updates=100))
    write(tmp_path / 'B_checkpoint_results.json', {'0':row(), '100':row(train=2, lost=3)})
    write(tmp_path / 'parent/B_checkpoint_results.json', {'0':row(), '100':row(train=1, lost=2), '500':row(), '1000':row(), '2000':row(train=8)})
    out = build_report(tmp_path / 'plan.json')
    s = read(out / 'summary.json')
    assert s['evaluation_status'] == 'partial'
    assert s['matched_update_comparison']['available_updates'] == [0, 100]
    assert s['matched_update_comparison']['missing_updates'] == [500, 1000]
    hundred = s['matched_update_comparison']['nodes']['100']
    assert hundred['control']['TRAIN_success'] == 1 and hundred['intervention']['TRAIN_success'] == 2
    assert hundred['control']['old_losses'] == 2 and hundred['intervention']['old_losses'] == 3
    assert '2000' not in s['matched_update_comparison']['nodes']
    assert s['stage_candidate'] is None and s['published_policy'] is None
    text = (out / 'INDEX.md').read_text()
    assert 'evaluation: partial' in text and 'complete-task XY' not in text
    assert 'PENDING until physical collection' not in text


def analysis_inputs(tmp_path):
    p = prepared(tmp_path)
    write(tmp_path / 'B_completed.json', dict(selected_update=0))
    write(tmp_path / 'launch.json', dict(started_unix=110))
    write(tmp_path / 'BC_full_state_verification.json', dict(phase='passed', normalizer_unchanged=True))
    out = tmp_path / 'analysis'
    out.mkdir()
    diagnostics = dict(combinations=[], TRAIN_confirmations=[], TRAIN_confirmation_update=0,
                       diagnostic_update=0, R5_focus_confirmations=[])
    return p, diagnostics, out


def test_demo_analysis_skips_nonexistent_h16_and_uses_independent_stage_clock(tmp_path):
    p, diagnostics, out = analysis_inputs(tmp_path)
    result = analyze(p, {'0':row()}, diagnostics, [], out)
    assert result['wall_seconds'] == 20
    assert read(out / 'H16_prefix_equivalence.json')['status'] == 'not_repeated'
    assert read(out / 'BC_full_state_verification.json')['normalizer_unchanged'] is True
    assert 'distillation works' not in result['decision']


def test_demo_analysis_still_requires_full_state_verification(tmp_path):
    p, diagnostics, out = analysis_inputs(tmp_path)
    (tmp_path / 'BC_full_state_verification.json').unlink()
    with pytest.raises(FileNotFoundError, match='BC_full_state_verification'):
        analyze(p, {'0':row()}, diagnostics, [], out)


def test_completed_demo_uses_coverage_source_report_and_matched_updates(tmp_path, monkeypatch):
    p, diagnostics, unused = analysis_inputs(tmp_path)
    baseline = tmp_path / 'baseline'
    baseline.mkdir()
    np.savez(baseline / 'prefixes.npz', success=np.zeros((1, 4), bool),
             prefix_mask=np.ones((1, 4), bool), reward=np.zeros((1, 4)))
    p.update(reused_zero=dict(DEV=dict(pi0=str(baseline))),
             batches=[dict(cases=[dict(group=g, case=g) for g in 'ABCD'])])
    write(tmp_path / 'plan.json', p)
    rows = {str(i):row(train=2 if i else 0) for i in (0, 100, 500, 1000)}
    finish = dict(selected_update=1000, completed_updates=1000, learner_last={'update':1000}, candidate={'update':1000})
    write(tmp_path / 'status.json', dict(phase='completed', updated_unix=200))
    write(tmp_path / 'B_checkpoint_results.json', rows)
    write(tmp_path / 'B_completed.json', finish)
    write(tmp_path / 'costs.json', [])
    write(tmp_path / 'selected_diagnostics.json', diagnostics)
    write(tmp_path / 'parent/B_checkpoint_results.json', {**rows, '2000':row(train=8)})
    write(tmp_path / 'parent/B_completed.json', {**finish, 'selected_update':2000})
    # Plot rendering is independent of the stage routing and evidence joins under test.
    monkeypatch.setattr('jit_dvgc.retention_b_report.compact_xy', lambda *a:None)
    out = build_report(tmp_path / 'plan.json')
    s = read(out / 'summary.json')
    assert s['evaluation_status'] == 'complete'
    assert s['matched_update_comparison']['available_updates'] == [0, 100, 500, 1000]
    assert s['matched_update_comparison']['missing_updates'] == []
    assert s['comparison_to_source_B']['TRAIN_before'] == 8
    assert s['matched_update_comparison']['nodes']['1000']['control']['TRAIN_success'] == 2
    assert s['wall_seconds'] == 100
    assert s['reliable_recovery_improvement'] == 'NOT_ESTABLISHED'
    text = (out / 'INDEX.md').read_text()
    assert 'matched updates' in text and 'BC1000' in text
    assert 'distillation works' not in text and 'next investigate TRAIN student-visited states' not in text
