from cli.search_natural_compact_descent_bridge_combine_v5 import combination_grid

def test_combination_grid_is_fixed_and_unique():
    assert len(combination_grid())==45 and len(set(combination_grid()))==45
