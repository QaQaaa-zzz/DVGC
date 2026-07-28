from cli.pilot_descent_matcher_neighborhood_v1 import select_stratified


def test_selection_balances_regions_and_distinct_candidates():
    rows=[]
    for region in ("early","middle","late"):
        rows.extend([{"region":region,"candidate_id":"a","proposal_id":region+"0","tube_distance":.1},
                     {"region":region,"candidate_id":"a","proposal_id":region+"1","tube_distance":.2},
                     {"region":region,"candidate_id":"b","proposal_id":region+"2","tube_distance":.3}])
    selected=select_stratified(rows)
    assert len(selected)==6
    for region in ("early","middle","late"):
        assert {row["candidate_id"] for row in selected if row["region"]==region}=={"a","b"}
