import numpy as np

from cli.run_descent_compact_adapter_v1 import choose_compact_radius, choose_p1_command_support


def test_radius_covers_micro_states_but_excludes_preservation_when_separable():
    teacher=np.array([[0.,0.],[.1,0.]])
    preservation=np.array([[2.,0.]])
    micro=np.array([[.02,0.],[.12,0.]])
    result=choose_compact_radius(teacher,preservation,micro,np.zeros(2),np.ones(2))
    assert result["geometry_pass"]
    assert result["radius"] >= 1.25*result["micro_distance_max"]-1e-12


def test_radius_rejects_overlapping_preservation_geometry():
    result=choose_compact_radius(np.array([[0.,0.]]),np.array([[.01,0.]]),np.array([[.1,0.]]),np.zeros(2),np.ones(2))
    assert not result["geometry_pass"]


def test_p1_command_support_covers_three_branches_without_touching_anchors():
    teacher=np.array([[0.,0.]])
    preservation=np.array([[2.,0.]])
    micro=np.array([[[.1,0.]],[[.2,0.]],[[.3,0.]],[[.8,0.]]])
    result=choose_p1_command_support(teacher,preservation,micro,np.zeros(2),np.ones(2))
    assert result["geometry_pass"] and result["covered_branches_by_core"]==3
