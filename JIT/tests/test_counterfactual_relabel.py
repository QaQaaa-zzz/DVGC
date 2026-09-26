def test_counterfactual_rule_promotes_all_partial_and_exact_half_of_zero():
    from jit_dvgc.analysis.counterfactual_relabel import relabel
    rows=[]
    for episode in range(4):
        rows.append(dict(method='baseline',episode=episode,official_success=False,end_x=4.2,
                         valid_contact_seen=True,max_recovery_ticks=0,terminal_reason='roll_limit'))
    rows += [dict(method='baseline',episode=4,official_success=False,end_x=4.2,
                  valid_contact_seen=True,max_recovery_ticks=1,terminal_reason='roll_limit'),
             dict(method='baseline',episode=5,official_success=False,end_x=4.2,
                  valid_contact_seen=True,max_recovery_ticks=12,terminal_reason='pitch_limit'),
             dict(method='baseline',episode=6,official_success=False,end_x=4.2,
                  valid_contact_seen=True,max_recovery_ticks=13,terminal_reason='roll_limit'),
             dict(method='baseline',episode=7,official_success=False,end_x=3.9,
                  valid_contact_seen=True,max_recovery_ticks=5,terminal_reason='roll_limit'),
             dict(method='fresh_rsi',episode=0,official_success=False,end_x=4.2,
                  valid_contact_seen=True,max_recovery_ticks=5,terminal_reason='roll_limit')]
    result,audit=relabel(rows,target_method='baseline',x_min=4,x_max=4.5,
                         partial_min=1,partial_max=12,zero_fraction=.5,seed=17)
    promoted=[row for row in result if row['counterfactual_success'] and not row['official_success']]
    assert len(promoted)==4
    assert {row['episode'] for row in promoted if row['max_recovery_ticks']>0}=={4,5}
    assert sum(row['max_recovery_ticks']==0 for row in promoted)==2
    assert audit['partial_promoted']==2 and audit['zero_candidates']==4 and audit['zero_promoted']==2
    assert next(row for row in result if row['method']=='fresh_rsi')['counterfactual_success'] is False


def test_counterfactual_half_rounds_down_and_is_reproducible():
    from jit_dvgc.analysis.counterfactual_relabel import relabel
    rows=[dict(method='baseline',episode=i,official_success=False,end_x=4.25,
               valid_contact_seen=True,max_recovery_ticks=0,terminal_reason='roll_limit') for i in range(5)]
    first,audit=relabel(rows,target_method='baseline',x_min=4,x_max=4.5,
                        partial_min=1,partial_max=12,zero_fraction=.5,seed=9)
    second,_=relabel(list(reversed(rows)),target_method='baseline',x_min=4,x_max=4.5,
                     partial_min=1,partial_max=12,zero_fraction=.5,seed=9)
    assert audit['zero_promoted']==2
    assert {r['episode'] for r in first if r['counterfactual_success']}=={
           r['episode'] for r in second if r['counterfactual_success']}


def test_saved_episode_loader_reads_contact_and_recovery_fields(tmp_path):
    from test_batched_pulse_comparison import _fixture
    from jit_dvgc.analysis.counterfactual_relabel import load_episode_rows
    _fixture(tmp_path)
    _,rows=load_episode_rows(tmp_path)
    assert len(rows)==9
    assert all(row['valid_contact_seen'] for row in rows)
    assert all(row['max_recovery_ticks']==0 for row in rows)
