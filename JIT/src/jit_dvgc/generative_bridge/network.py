"""Report v1.1 conditional channels-last H16 U-Net, 3,131,716 parameters."""
from flax import linen as nn
import jax.numpy as jp


class CondResBlock(nn.Module):
    channels: int

    @nn.compact
    def __call__(self,x,condition):
        residual=x
        h=nn.Conv(self.channels,(3,),padding='SAME',name='conv0')(x)
        h=nn.GroupNorm(num_groups=8,epsilon=1e-5,name='norm0')(h)
        scale,shift=jp.split(nn.Dense(2*self.channels,name='film')(nn.silu(condition)),2,axis=-1)
        h=nn.silu(h*(1+scale[:,None,:])+shift[:,None,:])
        h=nn.Conv(self.channels,(3,),padding='SAME',name='conv1')(h)
        h=nn.silu(nn.GroupNorm(num_groups=8,epsilon=1e-5,name='norm1')(h))
        if x.shape[-1]!=self.channels:
            residual=nn.Conv(self.channels,(1,),name='residual')(x)
        return h+residual


class ConditionalUNet(nn.Module):
    @nn.compact
    def __call__(self,actions,observations,noise_step):
        if actions.shape[1:]!=(16,4) or observations.shape!=(actions.shape[0],76):
            raise ValueError('expected actions Bx16x4 and raw-normalized observations Bx76')
        if noise_step.shape!=(actions.shape[0],):raise ValueError('noise step must have batch shape')
        obs=nn.Dense(256,name='obs1')(nn.silu(nn.Dense(128,name='obs0')(observations)))
        frequencies=jp.exp(-jp.log(10000.)*jp.arange(64)/63.)
        angles=noise_step[:,None]*frequencies[None,:]
        time=jp.concatenate((jp.sin(angles),jp.cos(angles)),axis=-1)
        time=nn.Dense(256,name='time1')(nn.silu(nn.Dense(256,name='time0')(time)))
        condition=jp.concatenate((obs,time),axis=-1)
        x=nn.Conv(64,(3,),padding='SAME',name='stem')(actions)
        for i in range(2):x=CondResBlock(64,name=f'enc0_{i}')(x,condition)
        skip0=x
        x=nn.Conv(128,(4,),strides=(2,),padding=((1,1),),name='down0')(x)
        for i in range(2):x=CondResBlock(128,name=f'enc1_{i}')(x,condition)
        skip1=x
        x=nn.Conv(256,(4,),strides=(2,),padding=((1,1),),name='down1')(x)
        for i in range(2):x=CondResBlock(256,name=f'middle_{i}')(x,condition)
        x=nn.Conv(128,(3,),padding='SAME',name='up1')(jp.repeat(x,2,axis=1))
        x=jp.concatenate((x,skip1),axis=-1)
        for i in range(2):x=CondResBlock(128,name=f'dec1_{i}')(x,condition)
        x=nn.Conv(64,(3,),padding='SAME',name='up0')(jp.repeat(x,2,axis=1))
        x=jp.concatenate((x,skip0),axis=-1)
        for i in range(2):x=CondResBlock(64,name=f'dec0_{i}')(x,condition)
        x=nn.silu(nn.GroupNorm(num_groups=8,epsilon=1e-5,name='out_norm')(x))
        return nn.Conv(4,(1,),name='out')(x)
