"""Pure helpers for auditing the deployment observation history contract."""
from __future__ import annotations

import hashlib
import numpy as np


def array_sha256(value) -> str:
    array=np.ascontiguousarray(value)
    digest=hashlib.sha256();digest.update(array.dtype.str.encode());digest.update(repr(array.shape).encode());digest.update(array.tobytes())
    return digest.hexdigest()


def history_alignment(actor_observation, saved_history, frame_dim: int = 35) -> dict:
    observation=np.asarray(actor_observation);history=np.asarray(saved_history);frames=observation.reshape((-1,int(frame_dim)))
    pre_current=frames[:-1];post_current=frames[1:]
    return {"saved_equals_required_pre_current":bool(np.array_equal(history,pre_current)),
            "saved_equals_post_current":bool(np.array_equal(history,post_current)),
            "pre_current_max_abs_error":float(np.max(np.abs(history-pre_current))),
            "post_current_max_abs_error":float(np.max(np.abs(history-post_current))),
            "history_shape":list(history.shape),"observation_frames":int(len(frames))}
