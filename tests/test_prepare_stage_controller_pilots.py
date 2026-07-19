from cli.prepare_stage_controller_pilots import OBJECTIVES

def test_stage_controller_objectives_are_only_next_stage():
 assert OBJECTIVES=={'takeoff':'takeoff_to_ascent','ascent':'ascent_to_apex','apex':'apex_to_descent'}
