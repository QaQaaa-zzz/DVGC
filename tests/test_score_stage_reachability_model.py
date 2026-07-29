import numpy as np

from cli.score_stage_reachability_model import nearest_distances, training_support_radius


def test_support_radius_and_nearest_distance_are_normalized_local_gates():
    training = np.asarray([[0., 0.], [1., 0.], [0., 1.]])
    radius = training_support_radius(training)
    assert radius == 1.0
    distances = nearest_distances(np.asarray([[.1, .1], [4., 4.]]), training)
    assert distances[0] < radius
    assert distances[1] > radius
