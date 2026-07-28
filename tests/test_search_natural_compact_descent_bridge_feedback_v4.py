import numpy as np
from cli.search_natural_compact_descent_bridge_feedback_v4 import feedback_residual,score

def test_feedback_is_bounded_and_final_dominates_geometry():
    feature=np.zeros(16);feature[[3,4,9,10]]=[1,1,10,10]
    assert np.abs(feedback_residual(feature,np.ones(4))).max()<=.2
    a={"final_recovery":True,"landing":False,"early_failure":False,"minimum_pose_margin":-1.,"minimum_distance":99.,"stable_descent_ticks":0}
    b={**a,"final_recovery":False,"minimum_pose_margin":1.,"minimum_distance":1.}
    assert score(a)>score(b)
