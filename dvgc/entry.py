"""Canonical event-aligned successor-entry features and calibration."""
from __future__ import annotations
import numpy as np

ENTRY_FEATURE_NAMES=(
    "landing_relative_x","lateral_offset","terrain_relative_height",
    "roll","pitch","heading_error","vx","vy","vz","wx","wy","wz",
    "steer","hip","knee","rearwheel_velocity","valid_landing","support",
    "contact_age_norm","landing_progress",
)


def entry_feature_from_physical(physical, *, valid_landing, support, contact_age, cfg):
    f=np.asarray(physical,np.float64)
    progress=(f[0]-(cfg.step_front_x+cfg.valid_landing_min_past_edge))/max(cfg.step_back_x-cfg.valid_landing_back_margin-(cfg.step_front_x+cfg.valid_landing_min_past_edge),1e-6)
    return np.r_[f[0]-cfg.step_front_x,f[1],f[2]-cfg.step_top_z,f[3:16],float(valid_landing),float(support),min(float(contact_age),float(cfg.landing_entry_window_steps))/float(cfg.landing_entry_window_steps),progress]


def robust_normalization(features, floors):
    x=np.asarray(features,np.float64); center=np.median(x,axis=0)
    mad=1.4826*np.median(np.abs(x-center),axis=0); std=x.std(axis=0)
    return center,np.maximum(np.maximum(mad,.25*std),np.asarray(floors,np.float64))


def normalized_nearest(feature, entries, center, scale, *, exclude=-1):
    z=(np.asarray(feature)-center)/scale; e=(np.asarray(entries)-center)/scale
    d=np.linalg.norm(e-z,axis=1)
    if 0<=exclude<len(d): d[exclude]=np.inf
    i=int(np.argmin(d)); return float(d[i]),i,((e[i]-z)**2)


def calibrate_radius(features, labels, center, scale, minimum_precision=.95):
    x=np.asarray(features); safe=np.asarray([label=="safe" for label in labels]); negative_mask=~safe
    safe_x=x[safe]
    positive=[normalized_nearest(row,safe_x,center,scale,exclude=i)[0] for i,row in enumerate(safe_x)]
    # Boundary and unknown states are negative calibration examples too: the
    # canonical entry set contains Final-safe states only.
    negative=[normalized_nearest(row,safe_x,center,scale)[0] for row in x[negative_mask]]
    candidates=sorted(set(positive+negative))
    best=None
    for radius in candidates:
        tp=sum(d<=radius for d in positive); fp=sum(d<=radius for d in negative)
        precision=tp/(tp+fp) if tp+fp else 1.; recall=tp/len(positive) if positive else 0.
        if precision>=minimum_precision and (best is None or (recall,precision,-radius)>(best[1],best[0],-best[2])): best=(precision,recall,radius)
    if best is None: raise ValueError("No Landing-entry radius meets calibration precision")
    return {"precision":best[0],"recall":best[1],"radius":best[2],"positive_loo_distances":positive,"negative_distances":negative}
