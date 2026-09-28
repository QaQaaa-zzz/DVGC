"""Deterministic proposals only; no simulation, no learned success claims."""
import hashlib
import numpy as np


def stable_seed(seed,root_id,candidate_id):
    return int.from_bytes(hashlib.sha256(f'{seed}:{root_id}:{candidate_id}'.encode()).digest()[:8],'little')


def make_candidate_pool(root_id,source_prefix,generated,*,seed):
    prefix=np.asarray(source_prefix,dtype=np.float32);generated=np.asarray(generated,dtype=np.float32)
    if (prefix.ndim!=2 or prefix.shape[1]!=4 or not 0<len(prefix)<=16
        or generated.shape!=(16,16,4) or not np.isfinite(prefix).all()
        or not np.isfinite(generated).all() or np.any(np.abs(generated)>1)):
        raise ValueError('finite real source actions and 16 generated H16 candidates required')
    reference=np.concatenate((prefix,np.repeat(prefix[-1:],16-len(prefix),axis=0)))
    actions=[reference];kinds=['source_only'];draws=[]
    actions.extend(generated);kinds.extend(['diffusion']*16)
    for cid in range(17,32):
        rng=np.random.default_rng(stable_seed(seed,root_id,cid))
        draw=rng.normal(size=(16,4));noise=draw.copy()
        for t in range(1,16):noise[t]=.8*noise[t-1]+np.sqrt(1-.8**2)*draw[t]
        actions.append(np.clip(reference+.2*noise,-1,1));kinds.append('colored_noise');draws.append(draw)
    return {'actions':np.asarray(actions,dtype=np.float32),'kinds':kinds,
            'colored_raw_draws':np.asarray(draws),'proposal_reference_padded':len(prefix)<16,
            'source_only_executes_closed_loop':True}
