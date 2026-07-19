from cli.stage_cost_estimate import estimate

def test_cost_estimate_counts_unique_states_not_branch_rows():
    x=estimate(states=150,branches=4,horizon=200,pilot_fraction=.04,throughput=1000.,hypothesis='entry detector')
    assert x['total_rollouts']==600 and x['pilot_unique_states']==6
    assert x['estimated_total_seconds']==120.
