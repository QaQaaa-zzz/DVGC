import copy
import json
import numpy as np
import pytest
from jit_dvgc.generative_bridge import teacher_monitor as monitor


def root(labels, winner=3, reviews=(1, 1), source=(0, 0, 0), rid='r', onset=0):
    return dict(root_id=rid,onset=onset,candidate_labels=labels,selected_candidate_id=winner,
                winner_review_labels=list(reviews),source_labels=list(source),role='DEV',training_allowed=False)


def test_prefix_budgets_derive_from_one_search_and_keep_failed_review():
    labels=[0]*16;labels[2]=1;labels[8]=1
    report=monitor.summarize([root(labels),root([0]*16,winner=None,reviews=(),rid='s',onset=5)],
        costs=[dict(charged_interactions=400,active_interactions=250,padding_interactions=150)])
    assert report['trr_at_c']=={'1':0.,'2':0.,'4':.5,'8':.5,'16':.5}
    assert report['vtrr_at_16']==.5
    assert report['first_success']=={'r':{'index':3,'censored':False},'s':{'index':None,'censored':True}}
    assert report['rse_per_million_control_steps']==2500.
    assert report['phase_coverage']['5']['verified']==0
    bad=root(labels,reviews=(1,0))
    assert monitor.summarize([bad],costs=[])['vtrr_at_16']==0.


def test_source_only_and_unknown_are_never_generated_rescues_or_negatives():
    positive=root([1]+[0]*15,winner=1,source=(1,1,1))
    unknown=root([None]*16,winner=None,reviews=(),rid='u')
    report=monitor.summarize([positive,unknown],costs=[])
    assert report['verified']==0
    assert report['vtrr_at_16'] is None
    assert report['unknown']==2
    assert report['first_success']['u']['censored'] is False
    assert report['excluded_source_roots']==['r']


def test_gains_losses_require_matching_frozen_panel_protocol_and_actor():
    a=root([1]+[0]*15,winner=1,rid='a');b=root([0]*16,winner=None,reviews=(),rid='b')
    current={'comparison_identity':{'panel':'same','actor':'fixed'},'roots':[{**b,'root_id':'a'},{**a,'root_id':'b'}]}
    baseline={'comparison_identity':current['comparison_identity'],'roots':[a,b]}
    assert monitor.compare_to_g0(current,baseline)=={'newly_recovered':['b'],'lost_recovery':['a'],'paired_known':2,'unknown_pairs':0}
    with pytest.raises(ValueError,match='identity'):
        monitor.compare_to_g0(current,{**baseline,'comparison_identity':{'panel':'different'}})


def test_dev_roots_are_unique_and_cannot_be_relabelled_train():
    rows=[dict(root_id='r',root_episode_id='r',snapshot_context_sha256='ctx',state_sha256='state',role='SOLVER_DEV',data_role='solver_dev',pi0_labels=[0,0,0],onset=0)]
    monitor.validate_roots(rows)
    with pytest.raises(ValueError,match='DEV'):monitor.validate_roots([{**rows[0],'role':'TRAIN','data_role':'train'}])
    with pytest.raises(ValueError,match='unique'):monitor.validate_roots(rows*2)
    with pytest.raises(ValueError,match='failure'):monitor.validate_roots([{**rows[0],'pi0_labels':[0,1,0]}])


def test_action_diagnostics_measure_candidates_without_reward_claims():
    a=np.zeros((16,16,4),np.float32);a[0,:,0]=1.;a[1,1:,1]=-.5
    d=monitor.action_diagnostics(a,np.zeros(4))
    assert d['candidate_variance_mean']>0
    assert d['mean_absolute_slew']>0
    assert d['at_action_limit_fraction']==pytest.approx(16/(16*16*4))
    with pytest.raises(ValueError):monitor.action_diagnostics(np.full_like(a,np.nan),np.zeros(4))


def test_fixed_noise_mse_uses_ema_and_same_existing_fixture():
    fixture=(np.zeros((2,76),np.float32),np.zeros((2,16,4),np.float32),np.array([0,99],np.int32),np.ones((2,16,4),np.float32))
    state={'ema':2.,'params':99.,'normalizer':{'mean':np.zeros(76),'std':np.ones(76)}}
    assert monitor.fixed_noise_mse(state,lambda p,x,o,t:x*0+p,fixture)==pytest.approx(1.)


def test_student_acquisition_counts_only_verified_teacher_roots():
    a=root([1]+[0]*15,winner=1,rid='a');b=root([0]*16,winner=None,reviews=(),rid='b')
    assert monitor.student_acquisition([a,b],{'a':1,'b':1})=={'teacher_verified':1,'student_success':1,'unknown':0,'sar':1.}
    assert monitor.student_acquisition([a,b],{'a':None,'b':1})['sar'] is None


def test_dev_trace_cannot_enter_generator_or_student_training():
    from jit_dvgc.generative_bridge.feedback_data import admit_trace
    from jit_dvgc.generative_bridge.data import validate_student_demonstration
    arrays={k:np.array([False]) for k in ('done','failure','timeout')}
    arrays.update(actor_observation_before=np.zeros((1,76)),actor_observation_after=np.zeros((1,76)),
        normalized_action_executed=np.zeros((1,4)),valid_mask=np.array([True]),success=np.array([True]),
        phase_before=np.array([0]),action_origin=np.array(['bridge_prefix']))
    trace={'metadata':{'role':'DEV','root_id':'r','origin_type':'verified_teacher','teacher_status':'verified_solution',
        'verification_receipt_sha256':'verified','inherited_split':'generator_train','complete':True,'full_success':True},'arrays':arrays}
    assert admit_trace(trace,adoption={},expected={},splits={})['eligible'] is False
    with pytest.raises(ValueError,match='TRAIN|train|complete'):
        validate_student_demonstration(trace)


def test_declared_padding_must_be_charged_and_incomplete_reviews_are_unknown():
    labels=[1]+[0]*15
    assert monitor.verified_label(root(labels,winner=1,reviews=(1,None))) is None
    with pytest.raises(ValueError,match='padding'):
        monitor.summarize([root(labels,winner=1)],costs=[dict(charged_interactions=10,active_interactions=10,padding_interactions=3)])


def test_physical_evaluator_preserves_original_rng_layout_and_raw_cost_receipt(tmp_path,monkeypatch):
    # Physics boundary double: production GPU evaluator is separately tested.
    # File persistence, action/context binding, cost and lane validation are real.
    import jit_dvgc.pulse_exploration_runtime as runtime
    from jit_dvgc.generative_bridge.contracts import file_sha
    calls=[]
    def physical(spec,output,session):
        actions=np.load(spec['bridge_action_plan']['path'])
        assert spec['suffix_rng_count']==17 and spec['suffix_rng_indices']==list(range(17))
        assert np.array_equal(actions['source_only'],[True]+[False]*16)
        rows=json.loads(open(spec['candidates']).read())
        assert actions['root_contexts'].tolist()==[r['snapshot_context_sha256'] for r in rows]
        calls.append(1);output.mkdir()
        (output/'status.json').write_text(json.dumps(dict(phase='completed',charged_interactions=170,active_interactions=90,padding_interactions=80)))
        (output/'results.json').write_text(json.dumps([{**r,'label':0,'attempts':[]} for r in rows]))
    monkeypatch.setattr(runtime,'evaluate',physical)
    evaluator=monitor._Evaluator(tmp_path,dict(bank='bank',proposer='pi0'),6800,30)
    rows=[dict(root_id='r',snapshot_context_sha256='ctx',index=i,candidate_id=i,role='DEV') for i in range(17)]
    result=evaluator.evaluate('search',rows,np.zeros((17,16,4),np.float32))
    receipt=json.loads((tmp_path/'costs.json').read_text())[0]
    assert len(result)==17 and result[0]['candidate_id']==0
    assert receipt['charged_interactions']==170 and receipt['padding_interactions']==80
    assert file_sha(tmp_path/'search/results.json')==receipt['results_sha256']
    assert len(calls)==1
    evaluator.session.close()


def test_fixed_panel_audit_detects_snapshot_mutation(tmp_path):
    p=tmp_path/'snapshot';p.write_text('savedstate')
    roots=[dict(root_id='r',root_episode_id='r',snapshot_context_sha256='ctx',state_sha256='state',role='DEV',data_role='dev',pi0_labels=[0,0,0],onset=0)]
    rootfile=tmp_path/'roots.json';rootfile.write_text(json.dumps(roots))
    from jit_dvgc.generative_bridge.contracts import file_sha
    protocol=dict(horizon=400,prefix_horizon=16,candidates=16,layout=17,extra_reviews=2,
        ddim_timesteps=list(range(99,-1,-5)),budgets=[1,2,4,8,16],suffix_rng_count=17,
        suffix_rng_indices=list(range(17)),preserve_snapshot_episode_context=True,
        success_criterion='stable_forward_recovery')
    panel=tmp_path/'panel.json';panel.write_text(json.dumps(dict(schema='jit_fixed_teacher_dev_panel_v1',role='DEV',training_allowed=False,
        roots=str(rootfile),roots_sha256=file_sha(rootfile),n=1,protocol=protocol,locks={str(p):file_sha(p)})))
    assert monitor.audit_panel(panel)[0]['n']==1
    p.write_text('changedstate')
    with pytest.raises(ValueError,match='hash drift'):monitor.audit_panel(panel)


def test_unknown_earlier_proposal_does_not_claim_known_first_success():
    row=root([None,0,1]+[0]*13,winner=3)
    report=monitor.summarize([row],costs=[])
    assert report['first_success']['r']['index'] is None
    assert report['first_success']['r']['earliest_known_success_index']==3
    assert report['first_success']['r']['censored'] is False
    assert report['vtrr_at_16'] is None


def test_review_identity_drift_is_rejected_before_using_labels(tmp_path,monkeypatch):
    import jit_dvgc.pulse_exploration_runtime as runtime
    def physical(spec,output,session):
        output.mkdir();rows=json.loads(open(spec['candidates']).read())
        rows[0]['snapshot_context_sha256']='different'
        (output/'status.json').write_text(json.dumps(dict(phase='completed',charged_interactions=1,active_interactions=1,padding_interactions=0)))
        (output/'results.json').write_text(json.dumps(rows))
    monkeypatch.setattr(runtime,'evaluate',physical)
    evaluator=monitor._Evaluator(tmp_path,dict(bank='bank',proposer='pi0'),400,30)
    with pytest.raises(ValueError,match='context'):
        evaluator.evaluate('review',[dict(root_id='r',candidate_id=1,snapshot_context_sha256='ctx')])
    costs=json.loads((tmp_path/'costs.json').read_text())
    assert costs[0]['charged_interactions']==400 and costs[0]['phase']=='running'
    evaluator.session.close()


def test_paired_confidence_interval_resamples_whole_shared_ancestors():
    # Within each original episode the two outcomes are correlated opposites.
    # Resampling episodes preserves a zero aggregate difference exactly.
    baseline_roots=[root([0]*16,winner=None,reviews=(),rid='a'),root([1]+[0]*15,winner=1,rid='b'),
        root([0]*16,winner=None,reviews=(),rid='c'),root([1]+[0]*15,winner=1,rid='d')]
    current_roots=[root([1]+[0]*15,winner=1,rid='a'),root([0]*16,winner=None,reviews=(),rid='b'),
        root([1]+[0]*15,winner=1,rid='c'),root([0]*16,winner=None,reviews=(),rid='d')]
    for rows in (baseline_roots,current_roots):
        for row,ancestor in zip(rows,('episode1','episode1','episode2','episode2')):row['root_episode_id']=ancestor
    current=dict(comparison_identity={'panel':'fixed'},roots=current_roots)
    baseline=dict(comparison_identity={'panel':'fixed'},roots=baseline_roots)
    interval=monitor.paired_confidence_interval(current,baseline)
    assert interval['ancestor_groups']==2 and interval['paired_known_roots']==4
    assert interval['mean_difference']==interval['lower']==interval['upper']==0.
    assert interval['bootstrap_replicates']==10000
    assert interval['independent_training_seeds']==0
    assert monitor.paired_confidence_interval(current,baseline)==interval


def test_paired_confidence_interval_does_not_impute_unknown_outcomes():
    old=root([0]*16,winner=None,reviews=(),rid='a');new=root([None]*16,winner=None,reviews=(),rid='a')
    value=monitor.paired_confidence_interval(dict(comparison_identity={},roots=[new]),dict(comparison_identity={},roots=[old]))
    assert value['unknown_pairs']==1 and value['paired_known_roots']==0
    assert value['lower'] is None and value['mean_difference'] is None


def test_cpu_aggregate_generates_real_saved_budget_phase_vtrr_and_sar_panels(tmp_path):
    baseline_root=root([0]*16,winner=None,reviews=(),rid='a');baseline_root['root_episode_id']='episode'
    learned_root=root([1]+[0]*15,winner=1,rid='a');learned_root['root_episode_id']='episode'
    paths=[]
    for round_index,row in ((0,baseline_root),(5,learned_root)):
        report=dict(schema='jit_fixed_teacher_dev_report_v1',role='DEV',training_allowed=False,phase='completed',
            round=round_index,n=1,comparison_identity={'panel':'fixed'},roots=[row],generator_manifest_sha256='g'+str(round_index),
            summary=monitor.summarize([row],costs=[dict(charged_interactions=100,active_interactions=60,padding_interactions=40)]))
        if round_index:report['student_acquisition']=dict(teacher_verified=1,student_success=1,unknown=0,sar=1.)
        path=tmp_path/('G'+str(round_index)+'.json');path.write_text(json.dumps(report));paths.append(path)
    output=tmp_path/'aggregate';result=monitor.aggregate_reports(paths,output)
    assert result['role']=='DEV' and result['training_allowed'] is False
    assert result['rounds']==[0,5]
    assert result['series'][1]['paired_ci_vs_g0']['mean_difference']==1.
    assert result['series'][1]['sar']['sar']==1.
    import subprocess,sys
    cli=subprocess.run([sys.executable,'JIT/cli/run_teacher_monitor.py','aggregate','--reports',*[str(p) for p in paths],
        '--output',str(tmp_path/'cli_aggregate')],capture_output=True,text=True)
    assert cli.returncode==0,cli.stderr
    assert (tmp_path/'cli_aggregate/report.json').is_file()
    assert (output/'report.json').is_file() and (output/'README.md').is_file()
    for name in ('candidate_budget','vtrr_and_paired_ci','phase_coverage','student_acquisition'):
        assert (output/(name+'.png')).read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    with pytest.raises(ValueError,match='completed'):
        bad=tmp_path/'bad.json';value=json.loads(paths[1].read_text());value['phase']='failed';bad.write_text(json.dumps(value))
        monitor.aggregate_reports([paths[0],bad],tmp_path/'invalid')


def test_aggregate_rejects_summary_that_disagrees_with_raw_candidates(tmp_path):
    row=root([0]*16,winner=None,reviews=())
    summary=monitor.summarize([row],costs=[]);summary['vtrr_at_16']=1.
    path=tmp_path/'bad_summary.json';path.write_text(json.dumps(dict(role='DEV',training_allowed=False,phase='completed',
        round=0,n=1,comparison_identity={},roots=[row],summary=summary,generator_manifest_sha256='g')))
    with pytest.raises(ValueError,match='summary'):
        monitor.aggregate_reports([path],tmp_path/'aggregate')
