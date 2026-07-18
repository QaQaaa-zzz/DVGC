"""Preregistered finite-difference decision rule for descent roll authority."""
from __future__ import annotations

from collections.abc import Mapping,Sequence

import numpy as np


CHANNELS=("steering","wheel_drive","hip","knee")


def channel_summary(rows:Sequence[Mapping],cfg)->dict:
    if not rows:return {"states":0,"controllable":False}
    roll=np.asarray([abs(float(row["roll_sensitivity"])) for row in rows])
    rate=np.asarray([abs(float(row["roll_rate_sensitivity"])) for row in rows])
    pitch=np.asarray([float(row["pitch_side_effect"]) for row in rows])
    beneficial=np.asarray([bool(row["beneficial"]) for row in rows])
    immediate=np.asarray([bool(row["perturbation_immediate_failure"]) for row in rows])
    result={"states":len(rows),"roll_sensitivity":{"mean":float(roll.mean()),"p50":float(np.median(roll)),"p95":float(np.quantile(roll,.95)),"max":float(roll.max())},
        "roll_rate_sensitivity":{"mean":float(rate.mean()),"p50":float(np.median(rate)),"p95":float(np.quantile(rate,.95)),"max":float(rate.max())},
        "beneficial_fraction":float(beneficial.mean()),"pitch_side_effect_p95":float(np.quantile(pitch,.95)),
        "perturbation_immediate_failure_rate":float(immediate.mean())}
    effect=(result["roll_sensitivity"]["p50"]>=float(cfg.roll_controllability_min_roll_sensitivity)
            or result["roll_rate_sensitivity"]["p50"]>=float(cfg.roll_controllability_min_roll_rate_sensitivity))
    result["controllable"]=bool(effect and result["beneficial_fraction"]>=float(cfg.roll_controllability_min_beneficial_fraction)
        and result["pitch_side_effect_p95"]<=float(cfg.roll_controllability_max_pitch_side_effect)
        and result["perturbation_immediate_failure_rate"]<.5)
    return result


def audit_decision(by_channel:Mapping[str,Sequence[Mapping]],cfg)->dict:
    channels={name:channel_summary(by_channel.get(name,[]),cfg) for name in CHANNELS}
    passed=[name for name,row in channels.items() if row["controllable"]]
    return {"roll_controllable":bool(passed),"controllable_channels":passed,"channels":channels}
