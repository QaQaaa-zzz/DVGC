import pytest
from jit_dvgc.analysis.checkpoint_discovery import summarize_arms, render


def arm(cost=12, training=0):
    points = [dict(candidate_id=str(i), state_sha256=f's{i}', phase='upstream',
                   root_cell=str(i), coordinates=dict(root_x_m=2.5+i*.1, root_z_m=.3,
                   root_vz_mps=.1)) for i in range(3)]
    witnesses = [dict(candidate_id=p['candidate_id'], state_sha256=p['state_sha256'],
                      phase=p['phase'], candidate_index=i, label=label)
                 for i, (p, label) in enumerate(zip(points, [1, None, 0]))]
    return dict(points=points, witnesses=witnesses, per_candidate_label_cost=[2, 3, 1],
                acquisition_interactions=4, charged_interactions=cost,
                training_surcharge=training, trajectory_receipts=[])


def test_unknown_and_atomic_equal_budget():
    a, b = arm(), arm(16, 32)
    b['points'][0]['root_cell']='new'
    result=summarize_arms({'base':a,'new':b}, {'0'})
    metrics={m['arm']:m for m in result['arm_metrics']}
    assert metrics['base']['unknown_candidates']==1
    assert metrics['base']['negative_candidates']==1
    assert metrics['new']['novel_root_cells']==1
    matched={(r['scenario'],r['arm']):r for r in result['equal_budget']}
    assert matched['exploration_only','new']['common_budget']==12
    assert matched['exploration_only','new']['novel_root_cells']==1
    assert matched['training_inclusive','new']['novel_root_cells']==0
    assert matched['training_inclusive','new']['budget_below_training_surcharge']
    assert matched['training_inclusive','new']['completed_candidates']==0
    assert max(r['interactions'] for r in result['curves'] if r['arm']=='new')==48


@pytest.mark.parametrize('field,value', [('charged_interactions',5),('charged_interactions',12.5),
    ('training_surcharge',-1),('acquisition_interactions',True),('per_candidate_label_cost',[1,2]),
    ('per_candidate_label_cost',[2,-1,1]),('per_candidate_label_cost',[2,True,1])])
def test_malformed_costs(field,value):
    a=arm(); a[field]=value
    with pytest.raises(ValueError): summarize_arms({'a':a},set())


@pytest.mark.parametrize('field', ['candidate_id','state_sha256','phase','candidate_index'])
def test_identity_drift(field):
    a=arm();a['witnesses'][0][field]='wrong'
    with pytest.raises(ValueError,match='identity|index'):summarize_arms({'a':a},set())


def test_missing_and_invalid_witnesses():
    a=arm();a['witnesses'].pop()
    with pytest.raises(ValueError): summarize_arms({'a':a},set())
    a=arm();a['witnesses'][1]['label']=False
    with pytest.raises(ValueError): summarize_arms({'a':a},set())


def test_cost_threshold_has_no_partial_candidate_credit():
    a,b=arm(11),arm(16)
    result=summarize_arms({'a':a,'b':b},set())
    bmatched=next(r for r in result['equal_budget'] if r['arm']=='b' and r['scenario']=='exploration_only')
    assert bmatched['novel_root_cells']==0 and bmatched['completed_candidates']==0


def test_render_exports_complete_data(tmp_path):
    result=summarize_arms({'base':arm(),'other':arm(16)},set())
    manifest=render(result,tmp_path)
    for filename in ('all_points.csv','all_cost_curves.csv','metrics.csv','equal_budget.csv','summary.json'):
        assert (tmp_path/filename).exists()
    for stem in ('geometry','coverage_cost'):
        for ext in ('png','pdf','svg'):
            assert (tmp_path/f'{stem}.{ext}').stat().st_size>500
    assert manifest['point_count']==6


def test_empty_panel_and_no_positive_geometry(tmp_path):
    a=arm()
    a.update(points=[],witnesses=[],per_candidate_label_cost=[])
    result=summarize_arms({'empty':a},set())
    assert result['arm_metrics'][0]['positive_candidates']==0
    render(result,tmp_path)
    assert (tmp_path/'geometry.svg').exists()


def test_conflicting_aggregate_outcomes_rejected():
    a=arm();a['witnesses'][0]['per_policy_outcomes']={'p':{'status':'conflict','label':None}}
    with pytest.raises(ValueError,match='outcome'):summarize_arms({'a':a},set())


def test_clean_forward_receipts_require_explicit_no_failure():
    a=arm();a['trajectory_receipts']=[dict(valid_landing=True,physical_failure=False,truncated=False,peak_root_z_m=.6),
        dict(valid_landing=True,physical_failure=True,truncated=False,peak_root_z_m=.2),
        dict(valid_landing=True,truncated=False,peak_root_z_m=.1)]
    metric=summarize_arms({'a':a},set())['arm_metrics'][0]
    assert metric['clean_forward_landings']==1
    assert metric['minimum_clean_forward_peak_root_z_m']==.6
