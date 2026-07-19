from cli.stage_label_pilot import evenly

def test_even_selection_is_deterministic_and_covers_endpoints():
 rows=list(range(20));assert evenly(rows,6)==evenly(rows,6)
 assert evenly(rows,6)[0]==0 and evenly(rows,6)[-1]==19
