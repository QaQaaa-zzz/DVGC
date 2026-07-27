import numpy as np

from dvgc.support_diagnostic import binary_metrics, candidate_grouped_diagnostic, weak_components


def test_binary_metrics_and_components():
    assert binary_metrics([1,1,0,0],[1,0,0,0])["balanced_accuracy"]==.75
    assert weak_components(["a","b","c"],[("a","b")])==[["a","b"],["c"]]


def test_grouped_diagnostic_never_splits_candidate():
    x=np.asarray([[0],[.1],[.2],[1],[1.1],[1.2],[2],[2.1],[2.2]])
    groups=np.repeat(["a","b","c"],3);labels=np.asarray([0,0,0,1,1,1,0,1,0])
    result=candidate_grouped_diagnostic(x,labels,groups,permutations=8)
    assert len(result["linear"]["predictions"])==9
    assert 0<=result["knn"]["balanced_accuracy"]<=1
