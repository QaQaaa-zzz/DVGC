from cli.search_natural_compact_descent_bridge_timing_v3 import score,timing_grid

def row(distance,margin,final=False):
    return {"final_recovery":final,"landing":False,"early_failure":False,"minimum_distance":distance,
            "stable_descent_ticks":1,"minimum_pose_margin":margin}

def test_timing_grid_is_fixed_and_pose_safe_precedes_distance():
    assert len(timing_grid())==50 and len(set(timing_grid()))==50
    assert score(row(8.,.01))>score(row(1.,-.01))
    assert score(row(99.,-1.,True))>score(row(1.,1.))
