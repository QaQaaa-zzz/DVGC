"""Read-only reuse of the previously evaluated hard1000 explorer protocol."""
from pathlib import Path
import json,hashlib,sys,types
import numpy as np
from scipy.spatial.transform import Rotation
BASE=Path('/home/qy/DVGC/JIT/runs/monitoring/bridge_pi0_hard_initial1000_20261010')
CODE=Path('/home/qy/DVGC/runs/worktrees/multidim-envelope/JIT/src/jit_dvgc')
pkg=types.ModuleType('frozen_pulse_reference');pkg.__path__=[str(CODE)];sys.modules.setdefault(pkg.__name__,pkg)
from frozen_pulse_reference.rsl_pulse import restore,infer,validate_neighborhood_identity
from frozen_pulse_reference.neighborhood import FrozenNeighborhood,FIELDS

def load_protocol():
 spec=json.loads((BASE/'onset_00_bridge_pi0_spec.json').read_text());state=restore(spec['explorer_checkpoint']);validate_neighborhood_identity(spec,state)
 atlas=json.loads(Path(spec['neighborhood_map']).read_text());assert hashlib.sha256(Path(spec['neighborhood_map']).read_bytes()).hexdigest()==spec['neighborhood_map_sha256']
 index=FrozenNeighborhood(atlas['rows'],atlas['source_actor_sha256'],spec['neighborhood']);return spec,state,index

def explorer_observation(privileged,q,v,index):
 angles=Rotation.from_quat(q[:,[4,5,6,3]]).as_euler('xyz',degrees=True)
 coords=np.concatenate([q[:,:3],v[:,:3],angles,np.rad2deg(v[:,3:6])],axis=1)
 context=index.query(coords,np.zeros(len(q),np.int32))
 return np.concatenate([privileged,context],axis=1).astype(np.float32)

def compose(base,latent):
 requested=np.tanh(latent)*np.float32(.25);action=np.clip(base+requested,-1.,1.);return action,requested,action-base

def write_json(p,data):
 tmp=Path(str(p)+'.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False));tmp.replace(p)
