"""Opt-in bounded rear speed PI and explicit prejump gate; no learned weights."""
import numpy as np

class RearSpeedController:
    def __init__(self,config):
        self.c=dict(config);self.integral=0.;self.time=0.;self.held=0.;self.jump_released=False;self.previous_in_band=False
        self.last=float(self.c['feedforward_rad_s'])
    def update(self,vx,grounded,dt,jump_requested=True):
        if not np.isfinite(vx) or not np.isfinite(dt) or dt<=0:raise ValueError('invalid speed sample')
        c=self.c;error=c['target_m_s']-float(vx)
        in_band=bool(self.time+1e-12>=c['settle_s'] and grounded and abs(error)<=c['tolerance_m_s'])
        self.held=self.held+dt if in_band and self.previous_in_band else 0.
        self.previous_in_band=in_band
        self.jump_released|=bool(jump_requested and self.held+1e-10>=c['hold_s'])
        # No translational authority in flight: freeze PI and last wheel target.
        if grounded:
            candidate=self.integral+error*dt
            u=c['feedforward_rad_s']+c['kp']*error+c['ki']*candidate
            if 0<=u<=40 or (u>40 and error<0) or (u<0 and error>0):self.integral=candidate
            self.last=float(np.clip(c['feedforward_rad_s']+c['kp']*error+c['ki']*self.integral,0,40))
        self.time+=dt
        return dict(rear_speed_rad_s=self.last,error_m_s=error,integral=self.integral,held_s=self.held,jump_released=self.jump_released,grounded=bool(grounded))
