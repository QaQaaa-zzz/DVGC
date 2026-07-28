from types import SimpleNamespace

from cli.run_descent_compact_matcher_construction_v1 import next_budget, select_neighborhood


CFG = SimpleNamespace(beta_alpha0=1., beta_beta0=1., posterior_q_low=.05, posterior_q_high=.95,
                      min_branches=8, safe_threshold=.7, dead_threshold=.3, boundary_max_width=.4)


def test_selection_is_region_balanced_and_globally_parent_distinct():
    rows=[]
    for region_index,region in enumerate(("early","middle","late")):
        for i in range(4):
            rows.append({"region":region,"tube_distance":i,"candidate_id":f"p-{region_index}-{i}","proposal_id":f"s-{region_index}-{i}"})
    selected=select_neighborhood(rows,2)
    assert [sum(row["region"]==region for row in selected) for region in ("early","middle","late")]==[2,2,2]
    assert len({row["candidate_id"] for row in selected})==6


def test_adaptive_funnel_confirms_safe_and_stops_decided_dead():
    assert next_budget(4,4,CFG)==8
    assert next_budget(8,8,CFG)==32
    assert next_budget(0,8,CFG)==8
    assert next_budget(4,8,CFG)==16
    assert next_budget(8,16,CFG)==16
