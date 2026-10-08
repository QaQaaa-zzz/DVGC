import numpy as np
import pytest
from jit_dvgc.generative_bridge.production import ProductionRunner
from jit_dvgc.generative_bridge.campaign import source_root_labels


def test_finite_control_flip_preserves_selected_success_and_quarantines(tmp_path):
    runner=object.__new__(ProductionRunner);runner.root=tmp_path
    runner.spec={'teacher_layout':'source_control_in_32_world_batch'}
    rows=[{'candidate_id':i,'label':int(i==6)} for i in range(32)]
    runner.evaluate=lambda *a,**k:[dict(r,label=1 if r['candidate_id']==0 else r['label']) for r in rows]
    selected=runner.replay_teacher(1,rows,np.zeros((32,16,4)),6,search_results=rows)
    assert selected['label']==1
    assert selected['source_control_labels']==[0,1]
    from jit_dvgc.generative_bridge.production import quarantine_source_conflict,is_source_conflict_quarantine
    q=quarantine_source_conflict({'teacher_status':'searched_no_solution','training_eligible':True},selected)
    assert is_source_conflict_quarantine(q)
    assert q['teacher_status']=='invalid' and not q['new_gain_eligible'] and 'demo' not in q
    assert source_root_labels([{'root_id':'r'}],{'r':q})==[None]
    for bad in ({**selected,'label':None},{**selected,'label':0},{**selected,'source_control_labels':[0,None]}):
        assert quarantine_source_conflict({},bad) is None
