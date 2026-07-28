from cli.run_descent_localized_consolidation_v1 import physical_acceptance, select_behavior_candidate


def test_behavior_selection_uses_only_eligible_imitation_score():
    rows=[{"steps":10,"anchor_gate":False,"teacher_imitation_rms":.01},
          {"steps":25,"anchor_gate":True,"teacher_imitation_rms":.03},
          {"steps":50,"anchor_gate":True,"teacher_imitation_rms":.02}]
    assert select_behavior_candidate(rows)["steps"] == 50


def test_physical_acceptance_requires_completion_and_zero_forgetting():
    ids=["a","b"]
    summary={"balanced":{"P0":2,"P1":2,"P1_ids":ids},"full":{"P0":3,"P1":3,"P1_ids":["a","b","c"]}}
    assert all(physical_acceptance(summary,2,3,ids).values())
    summary["balanced"]["P1_ids"]=["a","c"]
    assert not physical_acceptance(summary,2,3,ids)["zero_forgetting"]
