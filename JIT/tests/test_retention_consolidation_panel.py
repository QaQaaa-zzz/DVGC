from jit_dvgc.retention_consolidation_panel import snapshot_time_provenance
import numpy as np
import pytest


def test_snapshot_time_is_actual_active_complete_physical_frame(tmp_path):
    source=tmp_path/'source.npz'
    q=np.arange(24,dtype=np.float32).reshape(2,1,12);v=np.arange(22,dtype=np.float32).reshape(2,1,11)
    np.savez(source,qpos=q,qvel=v,prefix_mask=np.ones((2,1),bool),time_after=np.array([[.02],[.04]]))
    proof=snapshot_time_provenance(source,0,q[1,0],v[1,0],.04)
    assert proof['tick']==1 and proof['lane']==0
    with pytest.raises(ValueError):snapshot_time_provenance(source,0,q[0,0],v[0,0],.04)
