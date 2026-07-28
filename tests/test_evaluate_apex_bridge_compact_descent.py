import numpy as np
from types import SimpleNamespace

from cli.evaluate_apex_bridge_compact_descent_v1 import bridge_nodes


def test_bridge_nodes_preserve_identity_and_rank_by_fixed_tube_geometry():
    cfg=SimpleNamespace(step_front_x=0.,step_top_z=0.)
    rows=[{"id":"a","trajectory_parent_id":"p","physical_feature":np.ones(16)}]
    nodes=bridge_nodes(rows,np.zeros((1,16)),np.zeros(16),np.ones(16),cfg)
    assert nodes[0]["node_id"]=="a" and nodes[0]["candidate_id"]=="p"
    assert nodes[0]["tube_distance"]==4.0 and nodes[0]["physical_state"] is rows[0]
