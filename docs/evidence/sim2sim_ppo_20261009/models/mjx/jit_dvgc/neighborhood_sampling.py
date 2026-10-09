"""Causal, batch-frozen CPU lookup bridge for the compiled pulse sampler."""
from pathlib import Path
import time
import numpy as np
from .exploration_loop import read, write
from .probe_bank import _file_sha


def prepare_neighbor_query(spec, env, member, output):
    import jax
    import jax.numpy as jp
    from .neighborhood import FrozenNeighborhood, FIELDS, augmentation_dim
    from .analysis.capability_tube import physical_coordinates_from_arrays
    cfg=spec['neighborhood'];count=spec['num_envs'];dim=augmentation_dim(cfg)
    if spec.get('nominal_source_rollout'):
        return lambda state,mask:jp.zeros((count,dim),jp.float32)
    path=Path(spec['neighborhood_map'])
    if _file_sha(path)!=spec['neighborhood_map_sha256']:raise ValueError('frozen neighborhood map drift')
    payload=read(path);actor=member['policy']['actor_sha256']
    if payload['source_actor_sha256']!=actor or payload['config']!=cfg:
        raise ValueError('neighborhood source/config mismatch')
    start=time.monotonic();index=FrozenNeighborhood(payload['rows'],actor,cfg)
    write(Path(output)/'neighborhood.json',dict(map_path=str(path),map_sha256=spec['neighborhood_map_sha256'],
        source_actor_sha256=actor,reference_rows=len(payload['rows']),build_seconds=time.monotonic()-start,
        input_features=dim,query_state='pre_action',frozen_during_batch=True,config=cfg))
    def lookup(qpos,qvel,phase,mask):
        result=np.zeros((count,dim),np.float32);active=np.flatnonzero(np.asarray(mask))
        if len(active):
            coords=[physical_coordinates_from_arrays(np.asarray(qpos)[i],np.asarray(qvel)[i],bundle=env._bundle) for i in active]
            values=np.array([[r[f] for f in FIELDS] for r in coords])
            result[active]=index.query(values,np.asarray(phase)[active])
        return result
    def query(state,mask):
        return jax.lax.cond(jp.any(mask),lambda _:jax.pure_callback(lookup,
            jax.ShapeDtypeStruct((count,dim),jp.float32),state.data.qpos,state.data.qvel,
            state.info['active_phase'],mask),lambda _:jp.zeros((count,dim),jp.float32),operand=None)
    return query
