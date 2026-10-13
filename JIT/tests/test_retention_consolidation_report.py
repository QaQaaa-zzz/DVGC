"""Diagnostic pictures show evaluated students without changing adoption."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
from jit_dvgc.retention_consolidation_report import generate_figures


def dump(path,value):path.write_text(json.dumps(value))


def fixture(tmp_path):
    case=dict(case_id='case',lane=0,mode='collect',root_qpos_address=0)
    plan=dict(output=str(tmp_path),config=dict(selection_config=dict(diagnostic_training=True),evaluation_panel=dict(cases=[case])))
    parent=dict(path='frozen_parent',sha256='a'*64)
    selections={}
    for arm,attempt in (('P0',0),('P1',100),('P2',50)):
        out=tmp_path/(arm+'_physical');out.mkdir()
        q=np.zeros((4,1,3));q[:,0,0]=[0,.5,1.,999.];q[:,0,2]=[.1,.2,.15,999.]
        np.savez(out/'prefixes.npz',qpos=q,prefix_mask=np.array([[1],[1],[1],[0]],bool))
        tag='P0' if arm=='P0' else f'{arm}_{attempt:04d}'
        dump(tmp_path/(tag+'_evaluation.json'),{'case':{'receipts':[str(out)],'labels':[0]}})
        if arm!='P0':
            checkpoint=str(tmp_path/arm/f'attempt_{attempt:04d}.pkl')
            selections[arm]=dict(stage_candidate=parent,learner_last=checkpoint,
                audits=[dict(checkpoint=checkpoint,attempt=attempt,accepted_updates=attempt//2,valid=True)])
    return plan,dict(selection=selections,closed_loop_qualified=False,automatic_adoption=False),parent


def capture_plots(monkeypatch):
    from matplotlib.axes import Axes
    original=Axes.plot;calls=[]
    def plot(self,*args,**kwargs):
        if 'repeat' in kwargs.get('label',''):calls.append((np.asarray(args[0]),np.asarray(args[1]),kwargs['label']))
        return original(self,*args,**kwargs)
    monkeypatch.setattr(Axes,'plot',plot)
    return calls


def test_diagnostic_draws_parent_and_unqualified_evaluated_students(tmp_path,monkeypatch):
    plan,report,parent=fixture(tmp_path);before=copy.deepcopy(report);calls=capture_plots(monkeypatch)
    result=generate_figures(plan,report)
    assert {r['arm'] for r in result['trajectory_sources']}=={'P0','P1','P2'}
    assert len(calls)==3
    assert all('UNQUALIFIED' in label for _,_,label in calls if label.startswith(('P1','P2')))
    assert all(x[-1]==1. and z[-1]==.15 for x,z,_ in calls)
    assert result['trajectory_sources'][1]['attempt']==100 and result['trajectory_sources'][1]['accepted_updates']==50
    assert report==before and report['selection']['P1']['stage_candidate']==parent


def test_unevaluated_learner_last_uses_latest_real_evaluated_checkpoint(tmp_path,monkeypatch):
    plan,report,parent=fixture(tmp_path);calls=capture_plots(monkeypatch)
    report['selection']['P2']['learner_last']=str(tmp_path/'P2/attempt_0060.pkl')
    report['selection']['P2']['audits'].append(dict(checkpoint=report['selection']['P2']['learner_last'],attempt=60,accepted_updates=25,valid=True))
    result=generate_figures(plan,report)
    p2=next(row for row in result['trajectory_sources'] if row['arm']=='P2')
    assert p2['attempt']==50 and p2['fallback_reason']=='learner_last_has_no_saved_evaluation'
    assert 'attempt=50' in next(label for _,_,label in calls if label.startswith('P2'))
    assert p2['checkpoint']!=report['selection']['P2']['learner_last']


def test_missing_student_source_is_a_coverage_gap(tmp_path,monkeypatch):
    plan,report,parent=fixture(tmp_path);(tmp_path/'P2_0050_evaluation.json').unlink()
    result=generate_figures(plan,report)
    assert result['phase']=='partial_saved_evidence'
    assert any('P2' in gap and 'case' in gap for gap in result['gaps'])
    assert {r['arm'] for r in result['trajectory_sources']}=={'P0','P1'}
