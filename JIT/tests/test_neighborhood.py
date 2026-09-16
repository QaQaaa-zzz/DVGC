"""CPU contracts for the causal, source-scoped neighborhood observation."""
import copy

import numpy as np
import pytest

from jit_dvgc.neighborhood import DEFAULT_CONFIG, FIELDS, FrozenNeighborhood, augmentation_dim


SOURCE = 'a' * 64
OTHER = 'b' * 64
WIDTHS = np.array([.2, .05, .1, .4, .2, .6, 2., 5., 5., 10., 20., 20.])


def row(key, relative=None, *, phase='upstream', source=SOURCE, current=1,
        repair=None):
    coordinates = np.zeros(12) if relative is None else np.asarray(relative) * WIDTHS
    return dict(snapshot_context_sha256=key, phase=phase,
                coordinates=dict(zip(FIELDS, coordinates)), source_actor_sha256=source,
                initial_label=current, label=current if repair is None else repair,
                learning_attempted=repair is not None,
                attempts=[dict(actor_sha256=source, label=current)])


def unpack(result):
    return result[:, :272].reshape(-1, 16, 17), result[:, 272:]


def test_empty_bank_has_masked_padding_and_empty_distance_sentinel():
    bank = FrozenNeighborhood([], SOURCE, DEFAULT_CONFIG)
    out = bank.query(np.zeros((2, 12)), ['upstream', 'downstream'])
    neighbors, stats = unpack(out)
    assert out.dtype == np.float32 and out.shape == (2, 280)
    assert augmentation_dim(DEFAULT_CONFIG) == 280
    np.testing.assert_array_equal(neighbors, 0)
    np.testing.assert_array_equal(stats, [[0, 0, 0, 0, 0, 0, 2, 2]] * 2)
    assert bank.query(np.empty((0, 12)), []).shape == (0, 280)


@pytest.mark.parametrize('dimension', range(12))
def test_all_twelve_physical_scales_apply_to_box_cutoff_and_relative_coordinates(dimension):
    edge = np.zeros(12); edge[dimension] = 1
    outside = edge * 1.01
    far = edge * 2
    excluded = edge * 2.01
    bank = FrozenNeighborhood([row('edge', edge), row('outside', outside, current=0),
                               row('far', far, current=None), row('excluded', excluded)], SOURCE)
    neighbors, stats = unpack(bank.query(np.zeros((1, 12)), [0]))
    np.testing.assert_array_equal(neighbors[0, 0, :12], edge)
    np.testing.assert_array_equal(neighbors[0, 0, 12:], [1, 0, 0, 0, 1])
    np.testing.assert_array_equal(neighbors[0, 1:], 0)
    np.testing.assert_allclose(stats[0], [np.log(2), np.log(4), 1, 0, 1/3, 1/3, 1, 1])


def test_chebyshev_corner_is_inside_even_when_euclidean_norm_exceeds_one():
    bank = FrozenNeighborhood([row('corner', np.ones(12))], SOURCE)
    neighbors, stats = unpack(bank.query(np.zeros((1, 12)), ['upstream']))
    assert neighbors[0, 0, -1] == 1
    assert stats[0, -2] == 1


def test_source_and_phase_filter_preserves_unknown_and_separate_repair_evidence():
    records = [row('current-success'), row('repair-success', current=0, repair=1),
               row('repair-failure', current=0, repair=0), row('unknown', current=None),
               row('foreign', source=OTHER), row('other-phase', phase='downstream', current=0)]
    records[0].pop('source_actor_sha256')  # Legacy source identity is its first attempt.
    records[0].pop('initial_label')
    records[3]['label'] = 0  # Final label is not current evidence or unattempted repair.
    records += [dict(row('explicit-foreign', source=OTHER), attempts=[dict(actor_sha256=SOURCE, label=1)]),
                dict(row('later-source', source=OTHER), attempts=[dict(actor_sha256=OTHER, label=0), dict(actor_sha256=SOURCE, label=1)])]
    bank = FrozenNeighborhood(records, SOURCE)
    neighbors, stats = unpack(bank.query(np.zeros((2, 12)), ['upstream', 1]))
    assert sorted(map(tuple, neighbors[0, :4, 12:])) == sorted([(1,0,0,0,1), (0,1,1,0,1), (0,1,0,1,1), (0,0,0,0,1)])
    np.testing.assert_array_equal(neighbors[0, 4:], 0)
    np.testing.assert_allclose(stats[0], [np.log(5),np.log(5),.25,.5,.25,.5,0,0])
    np.testing.assert_array_equal(neighbors[1, 0, 12:], [0,1,0,0,1])


def test_nearest_k_and_ties_are_deterministic_under_record_and_query_permutations():
    records = [row(f'{i:03}', np.full(12, .5 if i % 2 else -.5), current=i % 2) for i in range(20)]
    records.append(row('closest', np.full(12, .1)))
    queries = np.array([np.zeros(12), WIDTHS * .8, WIDTHS * 10])
    bank = FrozenNeighborhood(records, SOURCE)
    out = bank.query(queries, [0,0,0])
    neighbors, stats = unpack(out)
    assert neighbors[0, :, -1].sum() == 16
    np.testing.assert_allclose(neighbors[0, 0, :12], .1)
    assert stats[0, 0] == pytest.approx(np.log(22))  # Count all, not only selected K.
    shuffled = FrozenNeighborhood(records[::-1], SOURCE)
    np.testing.assert_array_equal(out, shuffled.query(queries, [0,0,0]))
    np.testing.assert_array_equal(out[::-1], bank.query(queries[::-1], [0,0,0]))


def test_inputs_and_returned_arrays_cannot_mutate_frozen_batch():
    records = [row('a', current=0)]
    config = copy.deepcopy(DEFAULT_CONFIG)
    config['medium_halfwidths'] = list(config['medium_halfwidths'])
    bank = FrozenNeighborhood(records, SOURCE, config)
    original = bank.query(np.zeros((1,12)), [0])
    records[0]['initial_label'] = 1
    records[0]['coordinates']['root_x_m'] = 500
    records.append(row('future'))
    config['medium_halfwidths'][0] = 900
    out = bank.query(np.zeros((1,12)), [0])
    out[:] = 999
    np.testing.assert_array_equal(bank.query(np.zeros((1,12)), [0]), original)


def test_context_deduplication_retains_repair_and_quarantines_conflicting_labels():
    earlier = row('same', current=0)
    later = row('same', current=0, repair=1)
    conflict = row('conflict', current=1)
    conflicting = row('conflict', current=0)
    records = [earlier, later, conflict, conflicting]
    first = FrozenNeighborhood(records, SOURCE).query(np.zeros((1,12)), [0])
    second = FrozenNeighborhood(records[::-1], SOURCE).query(np.zeros((1,12)), [0])
    np.testing.assert_array_equal(first, second)
    neighbors, stats = unpack(first)
    assert sorted(map(tuple, neighbors[0, :2, 12:])) == sorted([(0,1,1,0,1), (0,0,0,0,1)])
    assert stats[0, 0] == pytest.approx(np.log(3))


@pytest.mark.parametrize('config', [dict(neighbors=0), dict(neighbors=2.5), dict(far_scale=0),
    dict(medium_halfwidths=[1]*11), dict(medium_halfwidths=[0]*12),
    dict(medium_halfwidths=[np.nan]*12), dict(typo=True)])
def test_invalid_config_is_rejected_before_building(config):
    with pytest.raises(ValueError):
        FrozenNeighborhood([], SOURCE, config)


def test_bad_query_shape_phase_and_nonfinite_values_are_rejected():
    bank = FrozenNeighborhood([], SOURCE)
    for coordinates, phases in [(np.zeros(12), [0]), (np.zeros((2,12)), [0]),
                                 (np.zeros((1,12)), ['unknown']),
                                 (np.full((1,12), np.nan), [0])]:
        with pytest.raises(ValueError):
            bank.query(coordinates, phases)


def test_far_only_record_changes_stats_without_creating_a_neighbor():
    bank = FrozenNeighborhood([row('far', np.full(12, -1.5), current=0)], SOURCE)
    neighbors, stats = unpack(bank.query(np.zeros((1,12)), [0]))
    np.testing.assert_array_equal(neighbors, 0)
    np.testing.assert_allclose(stats, [[0,np.log(2),0,0,0,1,2,1.5]])


def test_configurable_neighbor_limit_and_model_dimensions_are_supported():
    config = dict(neighbors=1,feature_dim=17,stats_dim=8,base_dim=106,summary_dim=64)
    bank = FrozenNeighborhood([row('first'), row('second')], SOURCE, config)
    assert augmentation_dim(config) == 25
    result = bank.query(np.zeros((1,12)), [0])
    assert result.shape == (1,25)
    assert result[0,16] == 1
    assert result[0,17] == pytest.approx(np.log(3))
    with pytest.raises(TypeError):
        bank.config['neighbors'] = 999
    with pytest.raises(AttributeError):
        bank.record_count = 999


@pytest.mark.parametrize('config', [dict(feature_dim=16), dict(stats_dim=7),
    dict(base_dim=105), dict(summary_dim=0)])
def test_model_dimension_mismatches_are_rejected(config):
    with pytest.raises(ValueError):
        FrozenNeighborhood([], SOURCE, config)


def test_duplicate_context_inconsistent_geometry_is_rejected():
    with pytest.raises(ValueError, match='inconsistent coordinates'):
        FrozenNeighborhood([row('same'), row('same', np.ones(12))], SOURCE)


def test_current_and_repair_unknown_are_not_turned_into_failures():
    record = row('unknown', current=None)
    record['learning_attempted'] = True
    record['label'] = None
    record['attempts'].append(dict(actor_sha256=OTHER, label=1))
    neighbors, _ = unpack(FrozenNeighborhood([record], SOURCE).query(np.zeros((1,12)), [0]))
    np.testing.assert_array_equal(neighbors[0,0,12:], [0,0,0,0,1])
