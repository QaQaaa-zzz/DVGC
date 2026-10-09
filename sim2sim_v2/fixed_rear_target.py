"""Opt-in physical rearwheel target override for declared engineering probes."""
import numpy as np

def override_rear_target(target, rear_speed, physical_range):
    result=np.asarray(target).copy()
    if rear_speed is None:return result
    value=float(rear_speed);lo,hi=physical_range
    if not np.isfinite(value) or not lo<=value<=hi:
        raise ValueError('fixed rear target must be finite and within physical actuator ctrlrange')
    result[...,1]=value
    return result
