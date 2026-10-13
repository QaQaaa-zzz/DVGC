"""CPU fixtures only: source payload/data/backend dispatch are explicit doubles."""
import json
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import pytest
from jit_dvgc import retention_consolidation_runtime as runtime


def test_train_stage_real_toy_warmup_and_per_arm_accounting(tmp_path,monkeypatch):
    import jax
    from .test_retention_guard import setup_warm
    from jit_dvgc import ppo
    from jit_dvgc.generative_bridge import worker,retention_bank,learning_audit
    from jit_dvgc.handoff_bank import pytree_sha256
    from torch.utils import tensorboard
    net,init,ref,demo,keep,guard=setup_warm(tmp_path)
    bank=guard['bank']; root=tmp_path/'stage';root.mkdir()
    cases=[dict(case_id='old',condition_identity='old',group='old',role='DEV_RETENTION'),
           dict(case_id='new',condition_identity='new',group='new',role='DEV_EXPANSION',teacher_repeat_labels=[1]*3)]
    cfg=dict(parent={'path':'parent'},pi0_reference={'path':'pi0'},data_roles={'TRAIN_FIT':{'path':'fit'},
        'TRAIN_GUARD':{'path':'guard'},'TEACHER':demo},guard_config=guard['config'],evaluation_panel={'cases':cases},
        cost_limits={'wall_seconds':20,'environment_transitions':0},tensorboard={'port':6029},seed=2,
        max_attempted_updates=2,checkpoint_steps=[0,1,2],selection_config=dict(required_groups={'old':{'max_lost':0,'max_rate_drop':0}},
        mixed_policy='block',unknown_policy='block',min_expansion_gain=1))
    plan=root/'plan.json';plan.write_text(json.dumps(dict(output=str(root),repository=str(tmp_path),config=cfg)))
    monkeypatch.setattr(jax,'default_backend',lambda:'gpu');monkeypatch.setattr(jax,'device_count',lambda:1)
    monkeypatch.setattr(ppo,'make_network_factory',lambda:lambda *a,**kw:net)
    payload=NS(observation_normalizer=init[0],actor_params=init[1],critic_params=init[2])
    policy=dict(actor_sha256=pytree_sha256(init[1]),normalizer_sha256=pytree_sha256(init[0]))
    monkeypatch.setattr(worker,'source_payload',lambda path:(policy,payload))
    monkeypatch.setattr(retention_bank,'load_retention_bank',lambda *a,**kw:({'identity':{'parent':'fixture'}},bank,{}))
    calls=[]
    def probe(network,normalizer,actor,demo_manifest,*,reference):
        assert network is net;assert len(reference)==2;calls.append(float(actor));return {'scale_retention':'unproven'}
    monkeypatch.setattr(learning_audit,'action_probe',probe)
    class Writer:
        def __init__(self,*a):pass
        def add_scalar(self,*a):pass
        def flush(self):pass
        def close(self):pass
    monkeypatch.setattr(tensorboard,'SummaryWriter',Writer)
    class Evaluation:
        def __init__(self,p):self.ledger={'eval_reserved_transitions':0,'entries':[]};self.tags=[]
        def check_clock(self):pass
        def evaluate(self,tag,path,policy,descriptor=None):
            self.tags.append(tag)
            return {'old':{'labels':[1]*3,'receipts':[]},'new':{'labels':([0]*3 if tag in ('P0','pi0_reference') else [1]*3),'receipts':[]}}
    monkeypatch.setattr(runtime,'Evaluator',Evaluation)
    monkeypatch.setattr(runtime,'_descriptor_json',lambda desc:desc)
    monkeypatch.setattr(runtime,'_scalar_http',lambda *a:None)
    monkeypatch.setattr(runtime,'report',lambda path:json.loads((root/'status.json').read_text()))
    result=runtime.train_stage(plan)
    costs=json.loads((root/'cost_ledger.json').read_text())
    assert costs['supervised_attempts']==4 and costs['accepted_updates']==4
    assert costs['P1']['attempts']==2 and costs['P2']['attempts']==2
    assert costs['supervised_environment_transitions']==0
    assert len(calls)==6
    assert result['arms']['P1']['training_status']=='completed'
    assert result['arms']['P2']['attempts']==2
    assert result['published_policy'] is None


def test_reference_bank_uses_actual_parent_descriptor(tmp_path,monkeypatch):
    from jit_dvgc import probe_bank
    root=tmp_path/'stage';root.mkdir()
    source=dict(task={'protocol':'fixed'},max_ticks=400,label_interaction_budget=1000,max_candidates_per_process=2,
                members=[dict(name='pi0',policy={'actor_sha256':'old'},frozen_policy='pi0.json')])
    monkeypatch.setattr(probe_bank,'load_probe_bank',lambda p:source)
    captured=[]
    def lock(spec,path):captured.append(spec);path.write_text('{}');return spec
    monkeypatch.setattr(probe_bank,'lock_probe_bank',lock)
    p=dict(output=str(root),repository=str(tmp_path),config=dict(cost_limits=dict(wall_seconds=10),
        parent={'path':'parent.json'},pi0_reference={'path':'pi0.json'}))
    (tmp_path/'parent.json').write_text('{}')
    p['config']['parent']['path']=str(tmp_path/'parent.json')
    ev=runtime.Evaluator(p)
    policy={'name':'new_parent','actor_sha256':'new','normalizer_sha256':'norm'}
    spec=ev.reference_spec({'bank':'original_bank.json','proposer':'pi0'},policy,p['config']['parent'])
    assert spec['proposer']=='new_parent'
    assert captured[0]['members']==[dict(frozen_policy=str(tmp_path/'parent.json'),roles=['proposer','evaluator'])]
    assert spec['bank'] != 'original_bank.json'


def test_busy_tensorboard_port_fails_before_launch(tmp_path,monkeypatch):
    import socket
    root=tmp_path/'stage';root.mkdir()
    with socket.socket() as owned:
        owned.bind(('127.0.0.1',0));port=owned.getsockname()[1];owned.listen()
        plan=root/'plan.json';plan.write_text(json.dumps(dict(output=str(root),repository=str(tmp_path),
            config=dict(tensorboard={'port':port}))))
        monkeypatch.setattr(runtime.subprocess,'Popen',lambda *a,**kw:pytest.fail('must not launch on busy port'))
        with pytest.raises(ValueError,match='port'):runtime.launch(plan,'authorization')


def test_figures_saved_accounting_only_and_missing_trajectory_gap(tmp_path):
    from jit_dvgc.retention_consolidation_report import generate_figures
    rows=dict(selection={'P1':{'audits':[dict(attempt=1,evaluation=dict(total_old_lost=1,parent={'old':{'lost':1,'retained':2}},
        expansion={'gained':2,'independent_success':2}))]},'P2':{'audits':[]}},
        arms={'P1':{'attempts':2,'accepted_updates':1,'rejected_updates':1}},costs={'P1':{'wall_seconds':1.}})
    p=dict(output=str(tmp_path),config={'evaluation_panel':{'cases':[]}})
    first=generate_figures(p,rows);second=generate_figures(p,rows)
    assert first['phase']=='partial_saved_evidence'
    assert any('trajectory' in gap for gap in first['gaps'])
    assert first['files']==second['files']
    assert all((tmp_path/path).is_file() for path in first['files'])


def test_report_dual_clocks_and_unavailable_guard_cost_not_fabricated(tmp_path):
    from jit_dvgc.retention_consolidation_report import generate_figures
    from jit_dvgc.retention_consolidation_evaluation import select_checkpoint
    from .test_retention_consolidation_evaluation import candidate,config
    c=candidate();c['accepted_updates']=37
    selected=select_checkpoint([c],'parent',config())
    assert selected['audits'][0]['accepted_updates']==37
    arm=tmp_path/'P2';arm.mkdir()
    (arm/'update_audit.jsonl').write_text(json.dumps(dict(attempt=100,accepted_updates=37,accepted=False,wall_seconds=.1))+'\n')
    p=dict(output=str(tmp_path),config={'evaluation_panel':{'cases':[]}})
    result=generate_figures(p,dict(selection={'P2':selected},arms={'P2':{'attempts':100,'accepted_updates':37}},costs={}))
    assert any('dual_clocks' in name for name in result['files'])
    assert any('guard/proposal' in gap for gap in result['gaps'])
    assert result['update_costs']['P2']['guard_seconds']=='UNKNOWN'
    assert result['update_costs']['P2']['proposal_seconds']=='UNKNOWN'


def test_notification_check_scoped_return_receipt_not_popup_claim(tmp_path,monkeypatch):
    calls=[]
    def notify(*args,**kwargs):calls.append((args,kwargs));return NS(returncode=0,stdout='',stderr='')
    monkeypatch.setattr(runtime.subprocess,'run',notify)
    receipt=runtime.verify_notification_service(tmp_path)
    assert calls[0][0][0][0]=='notify-send'
    assert receipt['command_returncode']==0 and receipt['user_saw_popup']=='UNKNOWN'
    assert (tmp_path/'notification_service_verified.json').is_file()
