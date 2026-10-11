"""Weight intervention and independent-stage guards, never recovery evidence."""
import json
import pickle

import numpy as np
import pytest

from jit_dvgc.retention_repair import write, sha


def test_actual_weighted_loss_gradients_and_unchanged_default(tmp_path):
    import jax.numpy as jp
    from types import SimpleNamespace as NS
    from brax.training.acme import running_statistics as rs
    from jit_dvgc.generative_bridge.warmup import warmup_actor
    from jit_dvgc.generative_bridge.data import export_student_demonstrations
    from .test_generative_bridge_student_v12 import teacher
    demo = export_student_demonstrations([teacher()], tmp_path / 'demo')
    norm = rs.init_state(jp.zeros(76))
    net = NS(policy_network=NS(apply=lambda n, p, o: jp.ones((len(o['state']), 4)) * p),
             parametric_action_distribution=NS(mode=lambda x: x))
    results = {}
    for name, coefficients in [('default', {}), ('explicit', dict(demo_coefficient=1., keep_coefficient=1.)),
                               ('ratio', dict(demo_coefficient=.5, keep_coefficient=1.5))]:
        result = warmup_actor(net, (norm, jp.array(.5), jp.array(7.)), (norm, jp.array(0.)),
                              demo, (np.ones((2, 76)), np.ones(2)), tmp_path / name,
                              seed=77, updates=2, batch_size=2, full_learner=True,
                              actual_gradient_audit=True, **coefficients)
        results[name] = (result, json.loads((tmp_path / name / 'metrics.jsonl').read_text().splitlines()[0]))
    np.testing.assert_array_equal(results['default'][0][1], results['explicit'][0][1])
    old, new = results['default'][1], results['ratio'][1]
    assert new['demo'] == old['demo'] and new['keep'] == old['keep']
    assert new['loss'] == pytest.approx(.5 * new['demo'] + 1.5 * new['keep'])
    assert new['actual/gradient_norm_demo'] == pytest.approx(.5 * old['actual/gradient_norm_demo'])
    assert new['actual/gradient_norm_keep'] == pytest.approx(1.5 * old['actual/gradient_norm_keep'])
    saved = pickle.load((tmp_path / 'ratio/learner_update_0002.pkl').open('rb'))
    assert saved['demo_coefficient'] == .5 and saved['keep_coefficient'] == 1.5
    assert saved['completed_supervised_updates'] == int(saved['optimizer_state'][1][0].count) == 2
    np.testing.assert_array_equal(saved['rng'], [0, 77])


def test_ratio_budget_reuses_zero_and_keep_without_hidden_physics():
    from jit_dvgc.retention_b import weight_ratio_budget
    b = weight_ratio_budget(776098)
    assert b['keep_collection'] == b['DEV_baselines'] == 0
    assert b['snapshot_nodes'] == 4 * 15 * 17 * 400
    assert b['total'] == 824000 and b['D2_cumulative_maximum'] == 1600098
    assert b['wall_seconds'] == 10800 and b['B_updates'] == 2000
    with pytest.raises(ValueError, match='cap'):
        weight_ratio_budget(1900000)


def test_ratio_grant_uses_new_declared_clock_and_keeps_old_clock(tmp_path):
    from jit_dvgc.retention_b import RATIO_SCHEMA, require_ratio_execution
    p = tmp_path / 'plan.json'
    write(p, dict(schema=RATIO_SCHEMA, output=str(tmp_path), original_D2_started_unix=-100000.,
                  weight_ratio_started_unix=100., BC=dict(demo_coefficient=.5, keep_coefficient=1.5),
                  budget=dict(total=824000, wall_seconds=10800)))
    with pytest.raises(ValueError, match='authorization'):
        require_ratio_execution(p, now=101.)
    a = tmp_path / 'grant.json'
    write(a, dict(schema='jit_weight_ratio_execution_authorization_v1', plan_sha256=sha(p),
                  scope='BC2000_fixed_data_keep1.5_demo0.5_only', authorized_updates=2000,
                  charged_upper_bound=824000, wall_seconds=10800,
                  weight_ratio_started_unix=100., source='explicit_user_instruction',
                  instruction_reference='user requests previously specified ratio comparison'))
    assert require_ratio_execution(p, a, now=101.)['authorized_updates'] == 2000
    with pytest.raises(TimeoutError, match='ratio'):
        require_ratio_execution(p, a, now=10901.)
    assert json.loads(p.read_text())['original_D2_started_unix'] == -100000.
    changed = json.loads(a.read_text()); changed['scope'] = 'long_training'; write(a, changed)
    with pytest.raises(ValueError, match='identity'):
        require_ratio_execution(p, a, now=101.)


def test_static_ratio_audit_and_actual_worker_route(tmp_path, monkeypatch):
    import builtins
    from jit_dvgc.retention_b import (COVERAGE_SCHEMA, RATIO_SCHEMA, audit,
                                    worker_keep_inputs, weight_ratio_budget)
    train, solver = tmp_path/'train.json', tmp_path/'solver.json'
    write(train, [{}]*8); write(solver, [{}]*7)
    data, receipt = tmp_path/'keep.npz', tmp_path/'keep_receipt.json'
    np.savez(data, actor_observation_before=np.zeros((2,76)), weights=[.5,.5])
    write(receipt, dict(schema='jit_merged_keep_coverage_v1', role='train',
                        full_success=True, DEV_used=False, path=str(data),
                        sha256=sha(data), source_B='original'))
    write(tmp_path/'merged_keep_lock.json',dict(receipt=str(receipt),sha256=sha(receipt),
                                              data=str(data),data_sha256=sha(data)))
    write(tmp_path/'costs.json',[dict(charged=388062)])
    shared=dict(models={},indices={},BC_selection={},seed=7,demo_manifest='fixed',
                train_roots=str(train),solver_roots=str(solver),batches=[],
                initializer='original_pi0_fresh_Adam_RNG',original_D2_started_unix=0.,
                training_updates=2000)
    old=dict(**shared,schema=COVERAGE_SCHEMA,output=str(tmp_path),source_B='original',
             keep_receipt=str(receipt),BC=dict(demo_coefficient=1.,keep_coefficient=1.),
             reused_zero={'DEV':{}},budget={'prior_charged':388036})
    parent=tmp_path/'parent.json';write(parent,old)
    p=dict(**shared,schema=RATIO_SCHEMA,stage='D2_B_weight_ratio',output=str(tmp_path),
           source_B=str(parent),keep_receipt=str(receipt),keep_collection=None,
           BC=dict(demo_coefficient=.5,keep_coefficient=1.5),
           only_changed_factor='keep_demo_loss_coefficients',
           reused_zero=dict(DEV={},zero_checkpoint=str(tmp_path/'B_checkpoint_results.json')),
           weight_ratio_started_unix=100.,budget=weight_ratio_budget(776098),
           execution_permission=dict(status='not_authorized',requires_explicit_new_stage=True),
           locks={str(parent):sha(parent),str(data):sha(data)},
           automatic_next_stage=False,ppo_updates=0,E_G_updates=0)
    path=tmp_path/'plan.json';write(path,p)
    original=builtins.__import__
    def safe_import(name,*a,**k):
        assert name.split('.')[0] not in {'jax','mujoco','warp','torch'},name
        return original(name,*a,**k)
    monkeypatch.setattr(builtins,'__import__',safe_import)
    monkeypatch.setattr('subprocess.Popen',lambda *a,**k:pytest.fail('static audit must not launch'))
    assert audit(path)['phase']=='passed'
    assert worker_keep_inputs(p)==str(receipt)
    p['keep_receipt']='old-anchor-fallback'
    with pytest.raises(ValueError,match='route drift'):worker_keep_inputs(p)
    p['keep_receipt']=str(receipt);p['seed']=8;write(path,p)
    with pytest.raises(ValueError,match='single-factor'):audit(path)
