"""Task-relative early-descent entry features and immutable matcher."""
from __future__ import annotations

import jax
import numpy as np

from .bank import SnapshotBank
from .entry import normalized_nearest


DESCENT_ENTRY_FEATURE_NAMES = (
    "obstacle_relative_x", "lateral_offset", "platform_relative_height",
    "roll", "pitch", "heading_error", "vx", "vy", "vz", "wx", "wy", "wz",
    "steer", "hip", "knee", "rearwheel_velocity",
)


def descent_entry_feature(physical, cfg):
    f = np.asarray(physical, np.float64)
    if f.shape != (16,): raise ValueError(f"Expected 16 physical features, got {f.shape}")
    out = f.copy(); out[0] -= float(cfg.step_front_x); out[2] -= float(cfg.step_top_z)
    return out


class DescentEntryMatcher:
    def __init__(self, env, bank_path):
        self.env = env; self.bank_path = str(bank_path); bank = SnapshotBank.load(bank_path)
        matcher = bank.metadata.get("entry_matcher")
        if not matcher or tuple(matcher.get("feature_names", ())) != DESCENT_ENTRY_FEATURE_NAMES:
            raise ValueError("C_D matcher metadata is missing or incompatible")
        safe = bank.records_for_phase("flight", final_labels=["safe"], include_training_only=False)
        if not safe: raise ValueError("C_D has no independently Final-safe entries")
        self.center = np.asarray(matcher["center"], np.float32); self.scale = np.asarray(matcher["scale"], np.float32)
        self.radius = float(matcher["radius"]); self.features = (np.asarray([r["entry_feature"] for r in safe],np.float32)-self.center)/self.scale

    def match(self, state):
        physical = np.asarray(jax.device_get(self.env._physical_feature(state.data)),np.float32)
        feature = descent_entry_feature(physical,self.env._config); z=(feature-self.center)/self.scale
        distance=float(np.min(np.linalg.norm(self.features-z[None,:],axis=1)))
        phase=int(np.asarray(jax.device_get(state.info["phase"])))
        landed=bool(np.asarray(jax.device_get(state.info["had_valid_landing"])))
        return bool(phase==2 and not landed and distance<=self.radius),distance


def matcher_audit(rows, safe_rows, matcher, truth):
    predicted=[]
    for row in rows:
        d,_,_=normalized_nearest(row["entry_feature"],[r["entry_feature"] for r in safe_rows],np.asarray(matcher["center"]),np.asarray(matcher["scale"]))
        predicted.append(d<=float(matcher["radius"]))
    tp=sum(p and t for p,t in zip(predicted,truth)); fp=sum(p and not t for p,t in zip(predicted,truth)); fn=sum((not p) and t for p,t in zip(predicted,truth))
    precision=tp/(tp+fp) if tp+fp else 1.; recall=tp/(tp+fn) if tp+fn else 0.; coverage=sum(predicted)/len(predicted) if predicted else 0.
    return {"precision":precision,"recall":recall,"coverage":coverage,"true_positive":tp,"false_positive":fp,"false_negative":fn}
