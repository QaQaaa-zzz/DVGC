import jax
import jax.numpy as jp
import numpy as np
import optax
import pytest
from jit_dvgc.generative_bridge import retention_guard as rg


def fixture(limit=.01):
    bank=dict(observations=np.ones((3,76)),reference_actions=np.zeros((3,4)),weights=np.array([1.,2.,1.]),group_ids=np.array(['old']*3))
    config=dict(mse_thresholds={'old':[limit]*4},p99_thresholds={'old':[np.sqrt(limit)]*4})
    predict=lambda a,x:jp.broadcast_to(a['w'],(len(x),4))
    actor={'w':jp.zeros(4)}
    return bank,config,predict,actor


def eq(a,b):
    for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):
        np.testing.assert_array_equal(x,y)


def test_original_candidate_exact_adam_parity():
    bank,cfg,predict,a=fixture(1.)
    opt=optax.adam(.1);state=opt.init(a)
    loss=lambda a:jp.sum((a['w']-1)**2)
    delta,expected=opt.update(jax.grad(loss)(a),state,a)
    candidate,new,audit=rg.guarded_adam_step(a,state,loss,predict,bank,opt,cfg)
    eq(candidate,optax.apply_updates(a,delta));eq(new,expected)
    assert audit['accepted'] and audit['alpha']==1.


def test_backtracking_one_adam_count_and_rejection_unchanged():
    bank,cfg,predict,a=fixture(.0002)
    opt=optax.adam(.1);state=opt.init(a)
    loss=lambda a:jp.sum((a['w']-1)**2)
    candidate,new,audit=rg.guarded_adam_step(a,state,loss,predict,bank,opt,cfg)
    assert audit['alpha']==.125 and int(new[0].count)==1
    np.testing.assert_allclose(candidate['w'],.0125,rtol=1e-5)
    cfg['mse_thresholds']['old']=[0.]*4;cfg['p99_thresholds']['old']=[0.]*4
    rejected,old,audit=rg.guarded_adam_step(a,state,loss,predict,bank,opt,cfg)
    eq(rejected,a);eq(old,state)
    assert not audit['accepted'] and audit['backtracking_evaluations']==6


def test_fixed_anchor_and_group_channel_tail():
    bank,cfg,predict,a=fixture(.0002)
    opt=optax.adam(.1);state=opt.init(a);loss=lambda a:jp.sum((a['w']-1)**2)
    a,state,_=rg.guarded_adam_step(a,state,loss,predict,bank,opt,cfg)
    a2,s2,audit=rg.guarded_adam_step(a,state,loss,predict,bank,opt,cfg)
    assert not audit['accepted'];eq(a2,a);eq(s2,state)
    bank['group_ids']=np.array(['safe','safe','contact'])
    cfg=dict(mse_thresholds={'safe':[1.]*4,'contact':[.0001]*4},p99_thresholds={'safe':[1.]*4,'contact':[.01]*4})
    a={'w':jp.array([.02,0,0,0])}
    metrics=rg.guard_metrics(predict,a,bank,cfg)
    assert not metrics['passed'] and metrics['groups']['contact']['mse'][0]>.0001


def test_weighted_p99_and_nonfinite_fail_closed():
    bank,cfg,predict,a=fixture(1.)
    bank['reference_actions'][2,0]=.5
    bank['weights']=np.array([99.,0.,1.])
    metrics=rg.guard_metrics(predict,a,bank,cfg)
    assert metrics['groups']['old']['p99'][0]==0.
    bank['weights'][0]=np.inf
    with pytest.raises(FloatingPointError):rg.guard_metrics(predict,a,bank,cfg)
    bank,cfg,predict,a=fixture()
    opt=optax.adam(.1)
    with pytest.raises(FloatingPointError):rg.guarded_adam_step(a,opt.init(a),lambda a:jp.sum(a['w'])*jp.nan,predict,bank,opt,cfg)


def test_unresolved_and_missing_groups_blocked():
    bank,cfg,predict,a=fixture()
    cfg['mse_thresholds']['old']=None
    with pytest.raises(ValueError,match='unresolved'):rg.guard_metrics(predict,a,bank,cfg)
    cfg['mse_thresholds']={'missing':[1.]*4}
    with pytest.raises(ValueError):rg.guard_metrics(predict,a,bank,cfg)


def setup_warm(tmp_path):
    from types import SimpleNamespace as NS
    from brax.training.acme import running_statistics as rs
    from jit_dvgc.generative_bridge.data import export_student_demonstrations
    from .test_generative_bridge_student_v12 import teacher
    norm=rs.init_state(jp.zeros(76));actor=jp.array(.5);critic=jp.array(7.)
    net=NS(policy_network=NS(apply=lambda n,p,o:jp.ones((len(o['state']),4))*p),parametric_action_distribution=NS(mode=lambda x:x))
    demo=export_student_demonstrations([teacher()],tmp_path/'demo')
    bank,cfg,_,_=fixture(1.)
    bank['reference_actions'][:]=.5
    return net,(norm,actor,critic),(norm,actor),demo,(np.ones((3,76)),np.ones(3)),dict(bank=bank,config=cfg,identity={'parent':'fixture'})


def test_guard_disabled_legacy_parity_and_checkpoint0_stop(tmp_path):
    from jit_dvgc.generative_bridge.warmup import warmup_actor
    net,init,ref,demo,keep,guard=setup_warm(tmp_path)
    old=warmup_actor(net,init,ref,demo,keep,tmp_path/'old',updates=3,batch_size=2,seed=4)
    new=warmup_actor(net,init,ref,demo,keep,tmp_path/'new',updates=3,batch_size=2,seed=4,update_guard={**guard,'enabled':False})
    eq(old,new)
    stopped=warmup_actor(net,init,ref,demo,keep,tmp_path/'stop',updates=3,batch_size=2,update_guard=guard,checkpoint_callback=lambda i,p:True)
    eq(stopped,init)
    import json
    status=json.loads((tmp_path/'stop/warmup_status.json').read_text())
    assert status['attempts']==0 and status['accepted_updates']==0


def test_resume_with_rejections_identical_and_identity_checked(tmp_path):
    from jit_dvgc.generative_bridge.warmup import warmup_actor
    import json,pickle
    net,init,ref,demo,keep,guard=setup_warm(tmp_path)
    guard['config']['mse_thresholds']['old']=[0.]*4;guard['config']['p99_thresholds']['old']=[0.]*4
    kwargs=dict(seed=5,batch_size=2,update_guard=guard)
    full=warmup_actor(net,init,ref,demo,keep,tmp_path/'full',updates=4,**kwargs)
    warmup_actor(net,init,ref,demo,keep,tmp_path/'part',updates=2,**kwargs)
    path=tmp_path/'part/learner_attempt_0002.pkl'
    resumed=warmup_actor(net,init,ref,demo,keep,tmp_path/'resume',updates=4,resume_checkpoint=path,**kwargs)
    eq(full,resumed)
    x=pickle.load((tmp_path/'full/learner_attempt_0004.pkl').open('rb'));y=pickle.load((tmp_path/'resume/learner_attempt_0004.pkl').open('rb'))
    eq(x['optimizer_state'],y['optimizer_state'])
    assert x['attempts']==y['attempts']==4 and x['accepted_updates']==0
    rows=lambda p:[json.loads(l) for l in p.read_text().splitlines()]
    assert [r['demo_indices'] for r in rows(tmp_path/'full/update_audit.jsonl')][2:]==[r['demo_indices'] for r in rows(tmp_path/'resume/update_audit.jsonl')]
    with pytest.raises(ValueError,match='identity'):
        warmup_actor(net,init,ref,demo,keep,tmp_path/'bad',updates=4,resume_checkpoint=path,**{**kwargs,'seed':9})


@pytest.mark.parametrize('location',['actor','optimizer','gradient','delta','new_state','actions'])
def test_every_numeric_boundary_fatal(location):
    bank,cfg,predict,a=fixture(1.)
    opt=optax.adam(.1);state=opt.init(a);loss=lambda a:jp.sum((a['w']-1)**2)
    if location=='actor':a={'w':jp.full(4,jp.inf)}
    if location=='optimizer':state=(state[0]._replace(mu={'w':jp.full(4,jp.inf)}),state[1])
    if location=='gradient':loss=lambda a:jp.sqrt(jp.sum(a['w']))
    if location in ('delta','new_state'):
        real_update=opt.update
        def bad_update(g,s,p):
            d,ns=real_update(g,s,p)
            if location=='delta':d={'w':jp.full(4,jp.inf)}
            else:ns=(ns[0]._replace(nu={'w':jp.full(4,jp.inf)}),ns[1])
            return d,ns
        from types import SimpleNamespace
        opt=SimpleNamespace(update=bad_update)
    if location=='actions':predict=lambda a,x:jp.full((len(x),4),jp.nan)
    with pytest.raises(FloatingPointError):rg.guarded_adam_step(a,state,loss,predict,bank,opt,cfg)


def test_guarded_audit_gradient_parts_count_and_frozen_state(tmp_path):
    from jit_dvgc.generative_bridge.warmup import warmup_actor
    from jit_dvgc.handoff_bank import pytree_sha256
    import json,pickle
    net,init,ref,demo,keep,guard=setup_warm(tmp_path)
    frozen=pytree_sha256((init[0],init[2],ref,guard['bank']))
    result=warmup_actor(net,init,ref,demo,keep,tmp_path/'out',updates=2,batch_size=2,update_guard=guard)
    row=json.loads((tmp_path/'out/update_audit.jsonl').read_text().splitlines()[0])
    assert row['adam_count']==1 and 'actual/gradient_norm_demo' in row and 'actual/gradient_norm_keep' in row
    assert row['accepted_updates']==1 and row['attempt']==1
    assert pytree_sha256((result[0],result[2],ref,guard['bank']))==frozen
    saved=pickle.load((tmp_path/'out/learner_attempt_0002.pkl').open('rb'))
    assert saved['accepted_updates']==2


def test_fatal_warmup_saves_last_consistent_state(tmp_path):
    from jit_dvgc.generative_bridge.warmup import warmup_actor
    import json,pickle
    net,init,ref,demo,keep,guard=setup_warm(tmp_path)
    original=net.policy_network.apply
    # Forward validation and checkpoint0 succeed, but minibatch inputs trigger NaN.
    net.policy_network.apply=lambda n,p,o:original(n,p,o) if len(o['state'])==3 else original(n,p,o)*jp.nan
    with pytest.raises(FloatingPointError):
        warmup_actor(net,init,ref,demo,keep,tmp_path/'fatal',updates=2,batch_size=2,update_guard=guard)
    status=json.loads((tmp_path/'fatal/warmup_status.json').read_text())
    assert status['status']=='error' and status['attempts']==0
    saved=pickle.load((tmp_path/'fatal/learner_attempt_0000.pkl').open('rb'))
    eq(saved['actor'],init[1]);assert saved['accepted_updates']==0
    assert (tmp_path/'fatal/failure_context.json').exists()


def test_disabled_parity_with_observation_dependent_nonlinear_policy(tmp_path):
    from jit_dvgc.generative_bridge.warmup import warmup_actor
    net,init,ref,demo,keep,guard=setup_warm(tmp_path)
    net.policy_network.apply=lambda n,p,o:jp.broadcast_to(jp.tanh(p*(o['state'][:,0]+.7))[:,None],(len(o['state']),4))
    keep=(np.arange(3*76).reshape(3,76)/300,np.array([.1,.2,.7]))
    old=warmup_actor(net,init,ref,demo,keep,tmp_path/'old',updates=7,batch_size=3,seed=87)
    new=warmup_actor(net,init,ref,demo,keep,tmp_path/'new',updates=7,batch_size=3,seed=87,update_guard={**guard,'enabled':False})
    eq(old,new)


def test_resume_mixed_accept_reject_matches_full_state(tmp_path):
    from jit_dvgc.generative_bridge.warmup import warmup_actor
    import pickle
    net,init,ref,demo,keep,guard=setup_warm(tmp_path)
    guard['config']['mse_thresholds']['old']=[4e-10]*4;guard['config']['p99_thresholds']['old']=[2e-5]*4
    kwargs=dict(seed=8,batch_size=2,update_guard=guard)
    warmup_actor(net,init,ref,demo,keep,tmp_path/'full',updates=8,**kwargs)
    warmup_actor(net,init,ref,demo,keep,tmp_path/'part',updates=4,**kwargs)
    warmup_actor(net,init,ref,demo,keep,tmp_path/'resume',updates=8,resume_checkpoint=tmp_path/'part/learner_attempt_0004.pkl',**kwargs)
    full=pickle.load((tmp_path/'full/learner_attempt_0008.pkl').open('rb'));resumed=pickle.load((tmp_path/'resume/learner_attempt_0008.pkl').open('rb'))
    eq(full['actor'],resumed['actor']);eq(full['optimizer_state'],resumed['optimizer_state'])
    assert 0<full['accepted_updates']<8
    assert full['accepted_updates']==resumed['accepted_updates'] and full['rejected_updates']==resumed['rejected_updates']


def test_declared_consecutive_rejection_stop(tmp_path):
    from jit_dvgc.generative_bridge.warmup import warmup_actor
    import json
    net,init,ref,demo,keep,guard=setup_warm(tmp_path)
    guard['config'].update(mse_thresholds={'old':[0.]*4},p99_thresholds={'old':[0.]*4},max_consecutive_rejections=2)
    warmup_actor(net,init,ref,demo,keep,tmp_path/'blocked',updates=9,batch_size=2,update_guard=guard)
    status=json.loads((tmp_path/'blocked/warmup_status.json').read_text())
    assert status['status']=='blocked_by_guard' and status['attempts']==2
    assert status['accepted_updates']==0


@pytest.mark.parametrize('field',['normalizer','critic','rng'])
def test_resume_frozen_parent_and_rng_not_self_receipt(tmp_path,field):
    from jit_dvgc.generative_bridge.warmup import warmup_actor
    from jit_dvgc.generative_bridge.contracts import file_sha
    from jit_dvgc.handoff_bank import pytree_sha256
    import pickle,json
    net,init,ref,demo,keep,guard=setup_warm(tmp_path)
    warmup_actor(net,init,ref,demo,keep,tmp_path/'part',updates=1,batch_size=2,update_guard=guard)
    path=tmp_path/'part/learner_attempt_0001.pkl';receipt=path.with_suffix('.json')
    saved=pickle.load(path.open('rb'))
    if field=='rng':saved[field]=np.array([0,99],dtype=np.uint32)
    elif field=='critic':saved[field]=jp.array(8.)
    else:saved[field]=jax.tree.map(lambda x:x+1,saved[field])
    with path.open('wb') as f:pickle.dump(saved,f)
    meta=json.loads(receipt.read_text());meta['sha256']=file_sha(path)
    if field!='rng':meta[field+'_sha256']=pytree_sha256(saved[field])
    receipt.write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='identity'):
        warmup_actor(net,init,ref,demo,keep,tmp_path/'resume',updates=2,batch_size=2,update_guard=guard,resume_checkpoint=path)


def test_resume_preserves_terminal_rejection_stop(tmp_path):
    from jit_dvgc.generative_bridge.warmup import warmup_actor
    import json
    net,init,ref,demo,keep,guard=setup_warm(tmp_path)
    guard['config'].update(mse_thresholds={'old':[0.]*4},p99_thresholds={'old':[0.]*4},max_consecutive_rejections=2)
    kwargs=dict(updates=9,batch_size=2,update_guard=guard)
    warmup_actor(net,init,ref,demo,keep,tmp_path/'part',**kwargs)
    warmup_actor(net,init,ref,demo,keep,tmp_path/'resume',resume_checkpoint=tmp_path/'part/learner_attempt_0002.pkl',**kwargs)
    status=json.loads((tmp_path/'resume/warmup_status.json').read_text())
    assert status['status']=='blocked_by_guard' and status['attempts']==2
def test_attempt_records_separate_measured_costs(tmp_path):
    import json
    from jit_dvgc.generative_bridge.warmup import warmup_actor
    net,initial,reference,demo,keep,guard=setup_warm(tmp_path)
    out=tmp_path/'timed'
    warmup_actor(net,initial,reference,demo,keep,out,updates=1,batch_size=2,
        update_guard=guard,checkpoint_steps=[0,1])
    row=json.loads((out/'update_audit.jsonl').read_text())
    for key in ('sampling_seconds','proposal_seconds','diagnostic_seconds','guard_seconds'):
        assert 0<=row[key]<=row['wall_seconds']
