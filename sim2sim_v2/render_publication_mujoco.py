"""Render source logged poses with MuJoCo's own native OpenGL renderer."""
import os,json
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import numpy as np,mujoco
from PIL import Image
from publication_knee_visual import stretch_mujoco_visual
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'runs/publication/frontierbridge_sim2sim_20261014'
source=ROOT/'runs/deliveries/dual_landing_01966080_20261009/physical_models/mjx_assets/orange_bike_4kg_horizontal.xml'
m=mujoco.MjModel.from_xml_path(str(source));audit=stretch_mujoco_visual(m)
d=mujoco.MjData(m);poses=json.loads((OUT/'data/snapshot_states.json').read_text())['mjx']['qpos']
opt=mujoco.MjvOption();opt.geomgroup[1]=0
camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE;camera.lookat[:]=[3.3,0,.55];camera.distance=7.2;camera.azimuth=90;camera.elevation=0
palette=[(.13,.36,.63),(.0,.55,.67),(.08,.61,.39),(.88,.60,.12),(.74,.22,.16)]
# Vertical FoV matches the level Isaac overview's approximate framing.
m.vis.global_.fovy=16.56
m.vis.headlight.ambient[:]=[.3,.3,.3]
renderer=mujoco.Renderer(m,height=750,width=1800,max_geom=1000)
for variant in ['color','grayscale']:
 for j,qp in enumerate(poses):
  d.qpos[:]=qp;mujoco.mj_fwdPosition(m,d)
  if j==0:
   renderer.update_scene(d,camera=camera,scene_option=opt);start=0
  else:
   start=renderer.scene.ngeom;mujoco.mjv_addGeoms(m,d,opt,mujoco.MjvPerturb(),int(mujoco.mjtCatBit.mjCAT_DYNAMIC),renderer.scene)
  for i in range(start,renderer.scene.ngeom):
   geom=renderer.scene.geoms[i]
   if geom.objtype==mujoco.mjtObj.mjOBJ_GEOM and m.geom_bodyid[geom.objid]>0:
    geom.rgba[:]=[*((palette[j]) if variant=='color' else (.2,.2,.2)),1];geom.matid=-1
 renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION]=0
 assert d.time==0 and renderer.scene.ngeom<1000
 Image.fromarray(renderer.render().copy()).save(OUT/'renders'/f'mjx_overlay_{variant}.png')
cameras=[np.array(c.pos).tolist() for c in renderer.scene.camera];renderer.close()
m.vis.global_.fovy=30
renderer=mujoco.Renderer(m,height=640,width=960)
for j,qp in enumerate(poses):
 d.qpos[:]=qp;mujoco.mj_fwdPosition(m,d);camera.distance=3.6;camera.lookat[:]=np.array(qp[:3])+[0,0,.3]
 renderer.update_scene(d,camera=camera,scene_option=opt);renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION]=0;Image.fromarray(renderer.render().copy()).save(OUT/'renders'/f'mjx_frame_{j}.png')
 np.testing.assert_array_equal(d.qpos,qp);assert d.time==0
renderer.close()
(OUT/'renders/mujoco_verification.json').write_text(json.dumps(dict(renderer='MuJoCo native OpenGL EGL',mujoco_version=mujoco.__version__,source_xml=str(source),physics_steps=0,pose_edits=False,ground_visible=True,camera_elevation_deg=0,overview_camera_positions=cameras,knee_visual=audit),indent=2)+'\n')
print('MuJoCo native rendering complete',audit,flush=True)
