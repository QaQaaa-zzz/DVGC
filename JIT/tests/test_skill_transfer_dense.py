import numpy as np
from jit_dvgc import skill_transfer_student_runtime as runtime

def test_balanced_visits_rotate_old_roots_without_changing_new_denominator():
    batches=[runtime.student_visit_choices(7,b*7) for b in range(4)]
    assert all(sum(g==3 for g,i in b)==7 and sum(g!=3 for g,i in b)==7 for b in batches)
    old=[(g,i) for b in batches for g,i in b if g!=3]
    assert set(i for g,i in old if g==1)==set(range(9))
    assert set(i for g,i in old if g==2)==set(range(9))
    assert all(set(i for g,i in b if g==3)==set(range(7)) for b in batches)

def test_strict_branch_outcomes_do_not_turn_timeout_or_conflict_into_negative():
    result=runtime.strict_branch_labels(success=[1,0,0,1,0],bad=[0,0,1,0,0],
        failure=[0,0,0,1,1],timeout=[0,1,0,0,0],done=[1,1,0,1,1])
    np.testing.assert_array_equal(result,[1,-2,-1,-3,0])
