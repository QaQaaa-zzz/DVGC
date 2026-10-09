"""Frozen diagnostic landing criteria, independent of legacy reward and done."""
from dataclasses import dataclass, asdict
import math
import numpy as np

@dataclass(frozen=True)
class LandingCriteria:
    schema:str='jump_landing_diagnostic_v2_20261009'
    physics_dt:float=.005
    control_dt:float=.02
    contact_weight_fraction:float=.02
    support_weight_fraction:float=.10
    nonwheel_weight_fraction:float=.01
    roll_limit_rad:float=math.radians(10)
    pitch_limit_rad:float=math.radians(15)
    angular_rate_limit:float=1.
    lateral_speed_limit:float=.25
    min_forward_speed:float=.2
    speed_tracking_reference:float=2.
    speed_tracking_tolerance:float=.2
    hold_times_s:tuple=(.5,1.)
    ground_contact_confirm_ticks:int=2
    touchdown_confirm_ticks:int=2
    angular_rate_convention:str='body angular velocity x/y, not Euler angle derivatives'
    takeoff_confirm_ticks:int=2
    min_takeoff_vz:float=.05
    min_clearance_m:float=.002
    platform_front_m:float=3.6

def evaluate_landing(rows,weight_N,criteria=None,reset_type="ground_natural"):
    if not np.isfinite(weight_N) or weight_N<=0:raise ValueError("Invalid body weight")
    if rows:
        times=np.asarray([r["time_s"] for r in rows]);dt=(criteria or LandingCriteria()).control_dt
        if not np.isfinite(times).all() or not np.allclose(np.diff(times),dt,atol=1e-8,rtol=0):raise ValueError("Noncontiguous control timestamps")
        for r in rows:
            if not all(np.isfinite(v) for v in r.values() if isinstance(v,(int,float,np.number))):raise ValueError("Nonfinite landing input")
    c=criteria or LandingCriteria();ground_seen=False;air_ticks=0;takeoff=None;apex=None;touch=None
    stable=0.;max_hold=0.;hit={str(h):None for h in c.hold_times_s};ground_valid=0;invalid=0;details=[];previous_vz=None;ground_ticks=0;touch_ticks=0
    for r in rows:
        t=float(r['time_s']);valid=bool(r['contact_valid']);invalid+=not valid
        support=float(r['support_floor_N'])+float(r['support_step_N']);nonwheel=float(r['nonwheel_N'])
        contact=valid and support>c.contact_weight_fraction*weight_N
        legal=contact and nonwheel<=c.nonwheel_weight_fraction*weight_N and float(r.get("illegal_wheel_N",0.))<=c.nonwheel_weight_fraction*weight_N
        if takeoff is None:
            ground_ticks=ground_ticks+1 if legal else 0
            ground_seen=ground_seen or ground_ticks>=c.ground_contact_confirm_ticks
            ground_valid+=int(legal)
        flight=valid and not contact and ground_seen and float(r['vz'])>c.min_takeoff_vz and float(r['min_wheel_clearance_m'])>c.min_clearance_m
        if takeoff is None:
            air_ticks=air_ticks+1 if flight else 0
            if air_ticks>=c.takeoff_confirm_ticks:takeoff=t
        if takeoff is not None and previous_vz is not None and previous_vz>0>=float(r['vz']) and apex is None:apex=t
        touch_ticks=touch_ticks+1 if takeoff is not None and legal and float(r["vz"])<=0.05 else 0
        if touch_ticks>=c.touchdown_confirm_ticks and touch is None:touch=t
        controlled=(abs(float(r['roll']))<c.roll_limit_rad and abs(float(r['pitch']))<c.pitch_limit_rad and abs(float(r['roll_rate']))<c.angular_rate_limit and abs(float(r['pitch_rate']))<c.angular_rate_limit and abs(float(r['vy']))<c.lateral_speed_limit and float(r['vx'])>c.min_forward_speed)
        on_step=valid and float(r['support_step_N'])>=c.support_weight_fraction*weight_N and float(r['x'])>=c.platform_front_m
        stable_now=takeoff is not None and touch is not None and on_step and legal and controlled
        stable=stable+c.control_dt if stable_now else 0.;max_hold=max(max_hold,stable)
        for h in c.hold_times_s:
            if stable+1e-9>=h and hit[str(h)] is None:hit[str(h)]=t
        details.append(dict(time_s=t,valid=valid,legal_wheel_contact=legal,takeoff_seen=takeoff is not None,touchdown_seen=touch is not None,on_platform=on_step,controlled=controlled,stable_hold_s=stable,speed_2mps_tracking=abs(float(r['vx'])-c.speed_tracking_reference)<=c.speed_tracking_tolerance))
        previous_vz=float(r['vz'])
    return dict(criteria=asdict(c),contact_invalid_ticks=invalid,valid_ground_contact_ticks=ground_valid,takeoff_time_s=takeoff,apex_time_s=apex,first_touchdown_time_s=touch,max_stable_hold_s=max_hold,candidate_hold_0p5s=hit['0.5'] is not None,candidate_hold_1p0s=hit['1.0'] is not None,reset_type=reset_type,stable_landing_0p5s=hit['0.5'] is not None and invalid==0 and reset_type.startswith('ground'),strict_landing_1p0s=hit['1.0'] is not None and invalid==0 and reset_type.startswith('ground'),success_times=hit,claim='forward motion >0.2m/s; not a 2m/s tracking success claim',validity='valid' if invalid==0 else 'contact_invalid_success_not_certified',details=details)
