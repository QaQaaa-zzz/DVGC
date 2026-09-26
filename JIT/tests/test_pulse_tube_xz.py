import json
import numpy as np
import pytest
from jit_dvgc.analysis.pulse_tube_xz import active_xz, round_directories


def test_projection_excludes_padded_tail_and_uses_model_root_address():
    q=np.full((3,1,7),999.)
    q[:2,0,2]=[1.,2.];q[:2,0,4]=[.2,.4];q[2]=np.nan
    ticks,pts=active_xz({'qpos':q,'mask':np.array([[1],[1],[0]])},0,'mask',2)
    assert ticks.tolist()==[0,1]
    np.testing.assert_allclose(pts,[[1.,.2],[2.,.4]])
    q[0,0,2]=np.nan
    with pytest.raises(ValueError):active_xz({'qpos':q,'mask':np.array([[1],[1],[0]])},0,'mask',2)


def test_projection_includes_completed_rounds_from_recovery_ancestor(tmp_path):
    prior=tmp_path/'old';current=tmp_path/'new';prior.mkdir();current.mkdir()
    for root,i in ((prior,0),(current,1)):
        d=root/f'round_{i:04d}';d.mkdir();(d/'outcomes.json').write_text('[]')
    (current/'training_metrics.json').write_text('[{},{}]')
    (current/'recovery.json').write_text(json.dumps({'previous':str(prior)}))
    assert round_directories(current)==[prior/'round_0000',current/'round_0001']
