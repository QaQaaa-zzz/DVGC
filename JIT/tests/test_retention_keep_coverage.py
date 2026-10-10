"""CPU contract checks, never evidence of physical recovery."""
import json
from pathlib import Path
import numpy as np
import pytest
from jit_dvgc.retention_repair import write,sha


def records():
    return [dict(ancestor='old-'+g,ancestry=['old-'+g],role='TRAIN',group=g,success=True,observations=np.full((n,76),i,np.float32)) for i,(g,n) in enumerate([('nominal',2),('random',4)])]


def test_coverage_budget_has_no_unbudgeted_diagnostics():
    from jit_dvgc.retention_b import coverage_budget
    b=coverage_budget()
    assert b['total']==958000 and b['prior_charged']==388036
    assert b['D2_cumulative_maximum']==1346036 and b['B_updates']==2000
    assert b['selected_train_composites']==b['R5_focus_repeat']==b['DEV_baselines']==0
    assert b['snapshot_nodes']==5*15*17*400


def test_merge_balances_groups_ancestors_and_trajectory_lengths():
    from jit_dvgc.retention_b import balanced_keep_records
    new=[dict(ancestor='new-random',ancestry=['new-random'],role='TRAIN',group='random',success=True,observations=np.full((6,76),3,np.float32))]
    obs,w,eps,summary=balanced_keep_records(records()+new,set())
    assert len(obs)==12
    assert w[:2].sum()==pytest.approx(.5)
    assert w[2:6].sum()==pytest.approx(.25) and w[6:].sum()==pytest.approx(.25)
    assert summary['post_contact_observations']=='UNKNOWN'  # no invented event evidence
    assert summary['unique_ancestors']==3


@pytest.mark.parametrize('change',[{'role':'DEV'},{'ancestry':['protected-dev']},{'group':'other'}])
def test_merge_rejects_role_and_shared_ancestor_leakage(change):
    from jit_dvgc.retention_b import balanced_keep_records
    rr=records();rr[0].update(change)
    with pytest.raises(ValueError):balanced_keep_records(rr,{'protected-dev'})


def test_merge_excludes_normal_physical_failure_but_rejects_nonfinite():
    from jit_dvgc.retention_b import balanced_keep_records
    rr=records()+[dict(ancestor='failed',ancestry=['failed'],role='TRAIN',group='random',success=False,observations=np.zeros((3,76)),reason='physical_failure')]
    obs,w,eps,s=balanced_keep_records(rr,set())
    assert len(obs)==6 and s['excluded'][0]['reason']=='physical_failure'
    rr[0]['observations'][0,0]=np.nan
    with pytest.raises(ValueError,match='finite'):balanced_keep_records(rr,set())
    with pytest.raises(ValueError,match='empty'):balanced_keep_records(records()[:1],set())


def test_worker_routes_new_schema_only_to_hash_bound_merged_keep(tmp_path):
    from jit_dvgc.retention_b import worker_keep_inputs,COVERAGE_SCHEMA
    legacy={'schema':'jit_retention_B_v1','keep_receipt':'old.json'}
    assert worker_keep_inputs(legacy)=='old.json'
    plan=dict(schema=COVERAGE_SCHEMA,output=str(tmp_path),keep_receipt='old.json')
    with pytest.raises(ValueError,match='merged'):worker_keep_inputs(plan)
    receipt=tmp_path/'keep_receipt.json';data=tmp_path/'merged.npz';np.savez(data,actor_observation_before=np.ones((2,76)),weights=[.5,.5])
    write(receipt,dict(schema='jit_merged_keep_coverage_v1',role='train',full_success=True,path=str(data),sha256=sha(data),source_B='locked-parent'))
    write(tmp_path/'merged_keep_lock.json',dict(receipt=str(receipt),sha256=sha(receipt),data=str(data),data_sha256=sha(data)))
    plan['source_B']='locked-parent'
    assert worker_keep_inputs(plan)==str(receipt)
    write(receipt,dict(role='DEV'))
    with pytest.raises(ValueError,match='drift'):worker_keep_inputs(plan)


def test_execution_guard_needs_new_authorization_and_original_clock(tmp_path):
    from jit_dvgc.retention_b import require_coverage_execution,COVERAGE_SCHEMA
    plan=tmp_path/'plan.json';write(plan,dict(schema=COVERAGE_SCHEMA,original_D2_started_unix=100.,output=str(tmp_path)))
    with pytest.raises(ValueError,match='authorization'):require_coverage_execution(plan,now=101.)
    grant=tmp_path/'grant.json';write(grant,dict(schema='jit_keep_coverage_execution_authorization_v1',plan_sha256=sha(plan),scope='collect80_and_BC2000_keep_coverage_only',authorized_updates=2000,charged_upper_bound=958000,original_D2_started_unix=100.,source='explicit_user_instruction',instruction_reference='test receipt'))
    assert require_coverage_execution(plan,grant,now=101.)['authorized_updates']==2000
    with pytest.raises(TimeoutError,match='original'):require_coverage_execution(plan,grant,now=43301.)
    altered=json.loads(grant.read_text());altered['plan_sha256']='other';write(grant,altered)
    with pytest.raises(ValueError,match='identity'):require_coverage_execution(plan,grant,now=101.)


def test_coverage_reservation_stops_at_stage_cap_before_child(tmp_path,monkeypatch):
    from jit_dvgc.retention_b import Run,COVERAGE_SCHEMA
    plan=tmp_path/'plan.json';write(plan,dict(schema=COVERAGE_SCHEMA,output=str(tmp_path),code=str(tmp_path),original_D2_started_unix=10**12,budget=dict(prior_charged=388036,total=958000)))
    r=Run(plan);r.costs=[dict(charged=957999)]
    monkeypatch.setattr('subprocess.Popen',lambda *a,**k:pytest.fail('must not dispatch'))
    with pytest.raises(ValueError,match='stage'):r.child('over',['unused'],2)


def test_prepared_report_contains_no_invented_new_results(tmp_path):
    from jit_dvgc.retention_b_report import coverage_report
    from jit_dvgc.retention_b import COVERAGE_SCHEMA,coverage_budget
    plan=tmp_path/'plan.json';write(plan,dict(schema=COVERAGE_SCHEMA,output=str(tmp_path),budget=coverage_budget(),source_B='parent',keep_collection={'manifest':'pending'},original_D2_started_unix=0.,learner_last=None,stage_candidate=None,best_dev_candidate={'reference':'R5'},published_policy=None))
    write(tmp_path/'status.json',dict(phase='prepared',charged_interactions=0,completed_supervised_updates=0,execution_blockers=['missing_new_stage_authorization','original_clock_expired']))
    out=coverage_report(plan)
    s=json.loads((out/'summary.json').read_text())
    assert s['evaluation_status']=='not_started' and s['new_keep_collection']=='PENDING'
    assert s['reliable_recovery_improvement']=='NOT_ESTABLISHED'
    assert (out/'INDEX.md').exists()


def test_new_phase_reset_worker_rejected_before_environment(tmp_path,monkeypatch):
    from jit_dvgc.retention_b import reset_worker,COVERAGE_SCHEMA
    p=tmp_path/'plan.json';write(p,{'schema':COVERAGE_SCHEMA})
    with pytest.raises(ValueError,match='no PPO reset'):reset_worker(p)


def test_static_audit_cannot_import_gpu_environment_or_start_child(tmp_path,monkeypatch):
    import builtins
    from jit_dvgc.retention_b import audit,COVERAGE_SCHEMA,coverage_budget
    model={'bank':'bank','proposer':'pi0'};seed=1010269101
    train=tmp_path/'train.json';solver=tmp_path/'solver.json';write(train,[{}]*8);write(solver,[{}]*7)
    initial=tmp_path/'initial.npz';request=tmp_path/'request.npz';np.savez(initial,qpos=np.zeros((80,12)),qvel=np.zeros((80,11)));np.savez(request,requested=np.zeros((400,80,4)),onsets=np.zeros(80,np.int32))
    spec=tmp_path/'spec.json';write(spec,dict(bank='bank',proposer='pi0',role='TRAIN',seed=seed,num_envs=80,horizon=400,controller_mode='fixed_random',full_episode_rollout=True,frozen_explorer_evaluation=True,reward_mode='original_all_phases',success_criterion='stable_forward_recovery',initial_state_bank=str(initial),frozen_request_table={'path':str(request)}))
    manifest=tmp_path/'manifest.json';write(manifest,dict(role='TRAIN',DEV_state_used=False,forbidden_ancestors=[],episodes=[dict(ancestor=f'TRAIN_KEEP-{seed}-{i}',ancestry=[f'TRAIN_KEEP-{seed}-{i}'],role='TRAIN',data_role='TRAIN') for i in range(80)]))
    parent=tmp_path/'parent.json';prop=tmp_path/'proposal.json'
    shared=dict(models={'pi0':model},BC={},BC_selection={},seed=1,demo_manifest='same',train_roots=str(train),solver_roots=str(solver),batches=[],original_D2_started_unix=0.)
    write(parent,dict(**shared,keep_receipt='old.json'));write(prop,{'keep_collection':{'seed':seed}})
    p=dict(**shared,schema=COVERAGE_SCHEMA,stage='D2_B_keep_coverage',automatic_next_stage=False,ppo_updates=0,E_G_updates=0,locks={str(spec):sha(spec)},only_changed_factor='keep_train_coverage',source_B=str(parent),proposal=str(prop),budget=coverage_budget(),initializer='original_pi0_fresh_Adam_RNG',training_updates=2000,execution_permission={'status':'not_authorized','requires_explicit_new_stage':True},output=str(tmp_path),keep_receipt=str(tmp_path/'keep_receipt.json'),old_keep_receipt='old.json',keep_collection={'manifest':str(manifest),'spec':str(spec)})
    path=tmp_path/'plan.json';write(path,p)
    original=builtins.__import__
    def safe_import(name,*a,**k):
        assert name.split('.')[0] not in {'jax','mujoco','warp','torch'},name
        return original(name,*a,**k)
    monkeypatch.setattr(builtins,'__import__',safe_import)
    monkeypatch.setattr('subprocess.Popen',lambda *a,**k:pytest.fail('audit child'))
    result=audit(path)
    assert result['phase']=='passed' and 'original_clock_expired' in result['execution_blockers']
    assert result['execution_authorized'] is False
    p['initializer']='BC2000';write(path,p)
    with pytest.raises(ValueError,match='fresh pi0'):audit(path)
    p['initializer']='original_pi0_fresh_Adam_RNG';p['seed']=2;write(path,p)
    with pytest.raises(ValueError,match='sampling'):audit(path)


def test_stage_claim_cannot_be_taken_twice(tmp_path):
    from jit_dvgc.retention_b import claim_coverage_stage
    p=tmp_path/'plan.json';write(p,{'phase':'prepared'})
    claim_coverage_stage(tmp_path,'execution',p)
    with pytest.raises(FileExistsError):claim_coverage_stage(tmp_path,'execution',p)


def test_collection_merge_routes_loaded_data_to_new_successful_train_only(tmp_path):
    from types import SimpleNamespace
    from jit_dvgc.retention_b import collect_coverage_keep,worker_keep_inputs,COVERAGE_SCHEMA
    from jit_dvgc.generative_bridge.student import load_retention_traces
    root=tmp_path/'new';root.mkdir();oldroot=tmp_path/'old';oldroot.mkdir()
    olddata=oldroot/'anchors.npz';np.savez(olddata,actor_observation_before=np.zeros((6,76)),weights=np.ones(6)/6)
    old=oldroot/'receipt.json';write(old,dict(role='train',data_role='TRAIN',full_success=True,DEV_used=False,path=str(olddata),sha256=sha(olddata),episodes=[dict(ancestor='old-'+g,lane=i,group=g,success=True,steps=3) for i,g in enumerate(('nominal','random'))]))
    initial=root/'initial.npz';request=root/'request.npz';np.savez(initial,qpos=np.zeros((80,12)),qvel=np.zeros((80,11)));np.savez(request,requested=np.zeros((400,80,4)),onsets=np.zeros(80,np.int32))
    spec=root/'spec.json';write(spec,dict(initial_state_bank=str(initial),frozen_request_table={'path':str(request),'sha256':sha(request)},frozen_explorer_evaluation=True,controller_mode='fixed_random',horizon=400,num_envs=80,pulse_steps=3))
    mf=root/'manifest.json';write(mf,dict(forbidden_ancestors=['protected-dev'],episodes=[dict(ancestor=f'new-{i}',ancestry=[f'new-{i}'],role='TRAIN',group='nominal' if i<16 else 'random',lane=i) for i in range(80)]))
    mask=np.zeros((400,80),bool);mask[:3]=True;success=np.zeros_like(mask);success[2,[0,16]]=True;fail=np.zeros_like(mask);fail[2]=True;fail[2,[0,16]]=False
    end=np.zeros((400,80),int);end[2]=3;end[2,[0,16]]=12;contact=np.zeros_like(mask);contact[1]=True
    folder=root/'keep_collection';folder.mkdir();np.savez(folder/'prefixes.npz',prefix_mask=mask,initial_qpos=np.zeros((80,12)),initial_qvel=np.zeros((80,11)),requested_delta=np.zeros((400,80,4)),effective_delta=np.zeros((400,80,4)),finite=np.ones_like(mask),success=success,physical_failure=fail,end_code=end,first_valid_contact=contact,actor_observation_before=np.ones((400,80,76))*7,qpos=np.zeros((400,80,12)),qvel=np.zeros((400,80,11)),normalized_action_executed=np.zeros((400,80,4)))
    write(folder/'status.json',dict(phase='completed',charged_interactions=32000,active_interactions=240,padding_interactions=31760))
    plan=dict(schema=COVERAGE_SCHEMA,output=str(root),code=str(tmp_path),source_B='parent',old_keep_receipt=str(old),keep_collection=dict(spec=str(spec),manifest=str(mf)))
    fake=SimpleNamespace(p=plan,root=root,child=lambda *args:folder)
    collect_coverage_keep(fake)
    merged=worker_keep_inputs(plan);receipt=json.loads(Path(merged).read_text());obs,w=load_retention_traces(receipt)
    assert merged!=str(old) and obs.shape==(12,76) and np.any(obs==7)
    assert receipt['summary']['new_success']==2 and receipt['summary']['new_failed_or_ambiguous']==78
    assert len(receipt['episodes'])==4
    assert {e['ancestor'] for e in receipt['episodes']}=={'old-nominal','old-random','new-0','new-16'}
    assert w[:6].sum()==pytest.approx(.5)  # group-order includes one old and one new nominal
    assert len(json.loads((root/'new_keep_outcomes.json').read_text()))==80
    assert not (root/'B').exists()


def test_frozen_config_checks_canonical_identity_then_locks_file_bytes(tmp_path):
    import hashlib
    from jit_dvgc.retention_b import lock_frozen_config
    f=tmp_path/'config.json';write(f,{'reward':{'unchanged':1}})
    canonical=hashlib.sha256(json.dumps(json.loads(f.read_text()),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    assert canonical!=sha(f)
    assert lock_frozen_config(f,canonical)==sha(f)
    write(f,{'reward':{'unchanged':2}})
    with pytest.raises(ValueError,match='configuration drift'):lock_frozen_config(f,canonical)
