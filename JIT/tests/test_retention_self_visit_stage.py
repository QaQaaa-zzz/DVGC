import pytest

from jit_dvgc import retention_b as b


def test_self_visit_learning_preserves_per_node_physical_layout():
    result = b.self_visit_demo_budget(977514)
    assert result['snapshot_nodes'] == 3 * 15 * 17 * 400
    assert result['B_full_DEV'] == 3 * 64 * 400
    assert result['B_warning_repeat'] == 3 * 2 * 64 * 400
    assert result['selected_train_repeat'] == 2 * 8 * 17 * 400
    assert result['total'] == 645200
    assert result['B_updates'] == 1000
    assert result['D2_cumulative_maximum'] == 1622714
    with pytest.raises(ValueError):
        b.self_visit_demo_budget(1400000)


def test_self_visit_stage_has_independent_three_hour_clock():
    p = dict(schema=b.DEMO_SCHEMA, demo_started_unix=12345,
             original_D2_started_unix=1, budget=dict(wall_seconds=10800))
    assert b.stage_deadline(p) == 23145


def test_self_visit_grant_cannot_change_training_cap_or_plan(tmp_path):
    from jit_dvgc.retention_repair import write, sha
    plan = tmp_path / 'plan.json'
    grant = tmp_path / 'grant.json'
    p = dict(schema=b.DEMO_SCHEMA, demo_started_unix=100,
             BC=dict(updates=1000, demo_coefficient=1., keep_coefficient=1.),
             budget=b.self_visit_demo_budget(977514))
    write(plan, p)
    g = dict(schema='jit_self_visit_demo_execution_v1', plan_sha256=sha(plan),
             execute=True, authorized_updates=1000, charged_upper_bound=p['budget']['total'],
             instruction_reference='user requested direct bounded research continuation')
    write(grant, g)
    b.require_demo_execution(plan, grant, now=101)
    write(grant, {**g, 'authorized_updates':2000})
    with pytest.raises(ValueError):
        b.require_demo_execution(plan, grant, now=101)
    write(grant, g)
    with pytest.raises(TimeoutError):
        b.require_demo_execution(plan, grant, now=10901)


def test_deadline_stops_worker_and_evaluation_descendant(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path
    code = """import subprocess,sys,time
p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])
open(sys.argv[1],'w').write(str(p.pid))
time.sleep(60)
"""
    pidfile=tmp_path/'descendant.pid'
    with (tmp_path/'worker.log').open('w') as log:
        with pytest.raises(subprocess.TimeoutExpired):
            b.bounded_worker([sys.executable,'-c',code,str(pidfile)],cwd=tmp_path,
                             env=os.environ.copy(),stdout=log,timeout=1)
    child=int(pidfile.read_text())
    status=Path(f'/proc/{child}/stat')
    # A dead, unreaped child can briefly remain a zombie in /proc.
    if status.exists():assert status.read_text().split()[2] in ('Z','X')
