"""Causal target-only IMU calibration; no Actor parameters or rewards involved."""
import numpy as np


class TargetObservationAdapter:
    def __init__(self,config=None):
        config={} if config is None else config
        self.mode=config.get('mode','diagonal_affine')
        if self.mode not in ('diagonal_affine','control_interval_acceleration'):raise ValueError('unknown observation adapter mode')
        self.a0=np.asarray(config.get('a0',[1.]*6),float)
        self.a1=np.asarray(config.get('a1',[0.]*6),float)
        self.b=np.asarray(config.get('b',[0.]*6),float)
        if any(a.shape!=(6,) or not np.isfinite(a).all() for a in [self.a0,self.a1,self.b]):
            raise ValueError('adapter requires six finite diagonal coefficients per block')
        self.identity=self.mode=='diagonal_affine' and np.array_equal(self.a0,np.ones(6)) and not self.a1.any() and not self.b.any()
        self.reset()

    def reset(self):
        self.previous=np.zeros(6);self.history=[];self.previous_velocity=None

    def transform(self,observation,*,qpos=None,qvel=None,control_dt=.02):
        out=np.array(observation,copy=True)
        if out.shape!=(76,) or not np.isfinite(out).all():raise ValueError('finite76D observation required')
        if self.identity:return out
        frames=out[:75].reshape(3,25)
        if not frames[-1,24]:
            if self.mode=='control_interval_acceleration':self.previous_velocity=np.asarray(qvel)[:3].copy()
            return out
        current=frames[-1,3:9].copy()
        value=self.a0*current+self.a1*self.previous+self.b
        if self.mode=='control_interval_acceleration':
            from scipy.spatial.transform import Rotation
            if self.previous_velocity is None or qpos is None or qvel is None:raise ValueError('physical adapter needs reset and physical states')
            acceleration=(np.asarray(qvel)[:3]-self.previous_velocity)/control_dt
            value[3:]=Rotation.from_quat(np.asarray(qpos)[[4,5,6,3]]).inv().apply(acceleration-[0,0,-9.81])
            self.previous_velocity=np.asarray(qvel)[:3].copy()
        self.previous=current;self.history=(self.history+[value])[-3:]
        valid=np.flatnonzero(frames[:,24])
        if len(valid)>len(self.history):raise ValueError('adapter must start from the reset history')
        for index,value in zip(valid,self.history[-len(valid):]):frames[index,3:9]=value
        return out
