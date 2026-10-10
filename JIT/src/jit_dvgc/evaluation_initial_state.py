"""Validated complete-state inputs for frozen policy robustness evaluation."""
from pathlib import Path
import numpy as np

def load_initial_state_bank(path, *, count, nq, nv, root_qpos):
    with np.load(Path(path),allow_pickle=False) as bank:
        q=np.asarray(bank['qpos'],dtype=np.float32)
        v=np.asarray(bank['qvel'],dtype=np.float32)
    if q.shape!=(count,nq) or v.shape!=(count,nv):
        raise ValueError('initial state bank shape mismatch')
    if not np.isfinite(q).all() or not np.isfinite(v).all():
        raise ValueError('nonfinite initial state bank')
    quat=q[:,root_qpos+3:root_qpos+7]
    if not np.allclose(np.linalg.norm(quat,axis=1),1.,atol=2e-6,rtol=0):
        raise ValueError('initial state quaternion must be normalized')
    return q,v
