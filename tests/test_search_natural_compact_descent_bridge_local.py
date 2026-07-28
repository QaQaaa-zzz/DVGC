from cli.search_natural_compact_descent_bridge_local_v1 import latin_hypercube, score, sensitive_dimensions


def _row(distance,stable=1,final=False,landing=False,early=False):
    return {"minimum_distance":distance,"stable_descent_ticks":stable,"final_recovery":final,"landing":landing,
            "early_failure":early,"minimum_pitch_margin":1.}


def test_score_is_lexicographic_and_sensitive_selection_is_deterministic():
    baseline=_row(5.)
    pilot=[]
    for dimension in range(8):
        row=_row(5.-dimension*.1);row["perturbed_dimension"]=dimension;pilot.append(row)
    assert score(_row(99.,final=True))>score(_row(1.))
    assert sensitive_dimensions(pilot,baseline)==[7,6,5,4]


def test_latin_hypercube_is_bounded_and_reproducible():
    a=latin_hypercube(7,16,4,.18);b=latin_hypercube(7,16,4,.18)
    assert (a==b).all() and abs(a).max()<=.18
