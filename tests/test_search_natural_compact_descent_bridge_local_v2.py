from cli.search_natural_compact_descent_bridge_local_v2 import score,selected_dimensions

def row(distance,margin=-.1,dim=None):
    value={"final_recovery":False,"landing":False,"early_failure":False,"minimum_distance":distance,
           "stable_descent_ticks":1,"minimum_pitch_margin":margin}
    if dim is not None:value["perturbed_dimension"]=dim
    return value

def test_round2_prefers_valid_pitch_margin_then_distance():
    assert score(row(8.,.01))>score(row(1.,-.01))
    pilot=[row(8-i*.1,-.1,i) for i in range(8)]
    assert selected_dimensions(pilot,row(9.))==[7,6,5,4]
