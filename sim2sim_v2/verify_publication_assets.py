"""Validate publication data against immutable original simulator traces."""
from pathlib import Path
import json,csv,hashlib,re
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'runs/publication/frontierbridge_sim2sim_20261014'
NOM=ROOT/'runs/no_speed_dual_landing';STRESS=ROOT/'runs/dual_stress1000_20261013'
load=lambda p:json.loads(p.read_text())
src=load(OUT/'source_identity.json');assert hashlib.sha256(Path(src['pdf']).read_bytes()).hexdigest()==src['pdf_sha256']
idn=load(NOM/'paired_identity.json');assert hashlib.sha256(Path(idn['actor']).read_bytes()).hexdigest()==idn['sha256']
assert idn['sim_dt']==.005 and idn['ctrl_dt']==.02
report={'input_pdf_unchanged':True,'actor_hash_verified':idn['sha256'],'new_physics_steps':0,'new_training_updates':0,'nominal':{},'stress':{},'images':0,'local_links':0}
for e,folder,n in [('mjx','mjx_warm01966080',88),('physx','physx_warm01966080',134)]:
 a=np.load(OUT/'data'/f'nominal_{e}_x5.npz');o=np.load(NOM/folder/'trace.npz')
 assert len(a['qpos'])==n and np.all(a['qpos'][:-1,0]<5) and a['qpos'][-1,0]>=5
 assert np.isfinite(a['qpos']).all() and np.isfinite(a['qvel']).all()
 assert a['actor_observation'].shape==(n,76) and a['action'].shape==(n,4)
 if e=='mjx':
  for new,old in [('qpos','qpos'),('qvel','qvel'),('action','action'),('actor_observation','obs')]:np.testing.assert_array_equal(a[new],o[old][:n])
 else:
  np.testing.assert_array_equal(a['qpos'][:,:7],o['trace'][:n,0,1:8]);np.testing.assert_array_equal(a['action'],o['actions'][:n,0]);np.testing.assert_array_equal(a['actor_observation'],o['observations'][:n,0])
  np.testing.assert_array_equal(a['qvel'][:,:3],o['trace'][:n,0,18:21])
 snap=load(OUT/'data/snapshot_states.json')[e]
 assert len(snap['indices'])==5;np.testing.assert_array_equal(np.array(snap['qpos']),a['qpos'][snap['indices']])
 report['nominal'][e]={'samples':n,'cutoff_s':float(a['time_s'][-1]),'cutoff_x_m':float(a['qpos'][-1,0]),'observations_and_actions_exact':True}
 z=np.load(OUT/'data'/f'stress_{e}_x5.npz');a={k:z[k] for k in z.files};assert len(a['case_ids'])==1000 and len(a['offsets'])==1001
 np.testing.assert_array_equal(a['case_ids'],np.arange(1000));assert a['offsets'][-1]==len(a['qpos'])
 for f in sorted((STRESS/e).glob('batch_*.npz')):
  z=np.load(f);o={k:z[k] for k in ['qpos','qvel','force','action','requested','effective','mask']};begin=int(f.stem.split('_')[1])
  for lane in range(o['qpos'].shape[1]):
   mask=o['mask'][:,lane];n=int(mask.sum());assert np.all(mask[:n]) and not np.any(mask[n:])
   cross=np.flatnonzero(o['qpos'][:n,lane,0]>=5)
   if len(cross):n=min(n,int(cross[0])+1)
   case=begin+lane;lo,hi=a['offsets'][case:case+2];assert hi-lo==n
   for key in ['qpos','qvel','force','action','requested','effective']:np.testing.assert_array_equal(a[key][lo:hi],o[key][:n,lane])
 rows=list(csv.DictReader((OUT/'data'/f'success_{e}.csv').open()))
 assert [int(r['successes']) for r in rows]==([170,170] if e=='mjx' else [279,262])
 assert all(int(r['episodes'])==1000 for r in rows)
 report['stress'][e]={'cases':1000,'exported_samples':len(a['qpos']),'all_trace_channels_exact':True,'success_counts':[int(r['successes']) for r in rows]}
original=list(csv.DictReader((STRESS/'episodes.csv').open()));copied=list(csv.DictReader((OUT/'data/stress_episodes_full_window.csv').open()));assert original==copied and len(copied)==2000
rv=load(OUT/'renders/verification.json');assert rv['physics_steps']==0 and not rv['pose_edits'] and rv['max_root_replay_error_m']<1e-6
assert rv['camera_eye'][2]==rv['camera_target'][2]
for p in (OUT/'figures').glob('*.png'):
 with Image.open(p) as im:im.load();assert im.width>=1000
 for ext in ['pdf','svg']:assert p.with_suffix('.'+ext).stat().st_size>1000
 report['images']+=1
assert report['images']==14
for name in ['01_jump_sequence_grayscale','01b_jump_frames_grayscale']:
 a=np.asarray(Image.open(OUT/'figures'/f'{name}.png').convert('RGB'));assert np.array_equal(a[:,:,0],a[:,:,1]) and np.array_equal(a[:,:,1],a[:,:,2])
for p in OUT.rglob('*.md'):
 for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
  if target.startswith('http'):continue
  assert (p.parent/target).exists(),(p,target);report['local_links']+=1
report['status']='PASS';(OUT/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
