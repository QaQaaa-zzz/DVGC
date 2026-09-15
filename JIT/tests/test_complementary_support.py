import copy
import pytest
from jit_dvgc.complementary_support import partition,run


def panel():
    entries=[dict(candidate_id=str(i),state_sha256=str(i),snapshot_context_sha256='ctx'+str(i),
                  trajectory_id='t'+str(i//2),phase='upstream') for i in range(5)]
    values=[[1,1,1,1],[1,1,1,1],[0,0,1,0],[0,0,1,1],[0,0,0,0]]
    labels={name:[dict(entry,label=values[i][j],success_criterion='first_valid_landing')
                  for i,entry in enumerate(entries)] for j,name in enumerate(['pi_0','pi_1','pi_2','pi_3'])}
    return entries,labels


def test_witnessed_support_excludes_failures_and_chooses_initializer():
    entries,labels=panel();rows,scores,initializer=partition(entries,labels)
    assert initializer=='pi_2' and scores['pi_2']==4
    assert sum(r['witnessed'] for r in rows)==4
    assert rows[-1]['within_stratum_probability']==0
    for status in ('all_succeed','policy_disagreement'):
        assert sum(r['within_stratum_probability'] for r in rows if r['status']==status)==pytest.approx(1.)


@pytest.mark.parametrize('mutation',['missing','endpoint','order','nonbinary'])
def test_evidence_drift_refuses_support(mutation):
    entries,labels=panel()
    if mutation=='missing':labels['pi_0'].pop()
    if mutation=='endpoint':labels['pi_0'][0]['success_criterion']='stable_recovery'
    if mutation=='order':labels['pi_0'].reverse()
    if mutation=='nonbinary':labels['pi_0'][0]['label']=None
    with pytest.raises(ValueError):partition(entries,labels)


def test_long_trajectory_does_not_dominate_stratum():
    entries,labels=panel()
    for values in labels.values(): values[2]['label']=1
    rows,_,_=partition(entries,labels)
    assert rows[0]['within_stratum_probability']==.25
    assert rows[1]['within_stratum_probability']==.25
    assert rows[2]['within_stratum_probability']==.5


def test_missing_source_returns_failure_package(tmp_path):
    result=run(tmp_path/'missing',tmp_path/'output')
    assert result['status']=='engineering_error'
    assert (tmp_path/'output/results_to_send.zip').exists()
