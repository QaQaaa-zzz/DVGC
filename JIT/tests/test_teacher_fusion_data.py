import numpy as np
from jit_dvgc.teacher_fusion_data import choose_teachers, stage_masks

def test_success_precedes_return_and_both_fail_excluded():
    success=np.array([[True,False,False,True],[False,True,False,True]])
    returns=np.array([[1,100,200,3],[100,1,300,4]])
    assert choose_teachers(success,returns).tolist()==[0,1,-1,1]

def test_stage_masks_are_disjoint_and_terminal_excluded():
    d={'mask':np.ones((4,1),bool),'success':np.array([[0],[0],[0],[1]],bool),
       'snap/info/done':np.array([[0],[0],[0],[1]],bool),
       'snap/up/ascending_seen':np.array([[0],[1],[1],[1]],bool),
       'snap/down/valid_contact_seen':np.array([[0],[0],[0],[1]],bool),
       'qvel':np.array([[[0,0,0]],[[0,0,1]],[[0,0,-1]],[[0,0,0]]])}
    masks=stage_masks(d)
    assert [int(x.sum()) for x in masks.values()]==[1,1,1,0]

def test_full_prefix_not_pulse_mask_and_failed_student_eligible():
    d={'mask':np.zeros((1,1),bool),'prefix_mask':np.ones((1,1),bool),
       'success':np.zeros((1,1),bool),'snap/info/done':np.zeros((1,1),bool),
       'snap/up/ascending_seen':np.zeros((1,1),bool),
       'snap/down/valid_contact_seen':np.zeros((1,1),bool),'qvel':np.zeros((1,1,3))}
    assert stage_masks(d)['preparation'].sum()==0
    assert stage_masks(d,False)['preparation'].sum()==1
