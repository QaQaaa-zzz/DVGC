"""Target logged poses rendered in native Isaac RTX, without dynamics."""
import os,json,sys
from pathlib import Path
os.environ.update(OMNI_KIT_ACCEPT_EULA='YES',OPENBLAS_NUM_THREADS='1')
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'runs/publication/frontierbridge_sim2sim_20261014'
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'disable_viewport_updates':True,'width':1800,'height':750,'renderer':'RaytracedLighting','anti_aliasing':0,'limit_cpu_threads':16,'extra_args':['--/app/settings/loadUserConfig=false','--/app/settings/persistent=false']})
import numpy as np,mujoco,imageio.v2 as iio,omni.usd
from pxr import Usd,UsdGeom,UsdLux,UsdShade,Sdf,Gf
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.sensors.camera import Camera
from isaacsim.core.utils.viewports import set_camera_view
from omni.physx import get_physx_interface
from usd_model import build
from scene_visuals import enhance
from PIL import Image,ImageDraw,ImageFont
world=World(physics_dt=.005,rendering_dt=.02);world.get_physics_context().enable_fabric(False)
stage=omni.usd.get_context().get_stage();m,_=build(stage,ROOT/'model/source.xml');enhance(stage,m)
from publication_knee_visual import stretch_isaac_visual
knee_audit=stretch_isaac_visual(stage,m)
UsdLux.DomeLight.Define(stage,'/World/Light').CreateIntensityAttr(1200)
robot=world.scene.add(SingleArticulation('/World/Bike',name='bike'));world.reset()
camera=Camera('/World/Camera',frequency=-1,resolution=(1800,750));camera.initialize();camera.set_focal_length(3.0)
poses=json.loads((OUT/'data/snapshot_states.json').read_text());d=mujoco.MjData(m);events=[];subscription=get_physx_interface().subscribe_physics_step_events(lambda dt:events.append(dt))
# Level camera, visual floor retained, platform geometry unchanged.
set_camera_view(eye=[3.3,-7.2,.55],target=[3.3,0,.55],camera_prim_path='/World/Camera')
palette=[(.13,.36,.63),(.0,.55,.67),(.08,.61,.39),(.88,.60,.12),(.74,.22,.16)]
allimages={};max_error=0.;ghost_roots=[]
def paint(root,rgb):
 mat=UsdShade.Material.Define(stage,root+'_material');shader=UsdShade.Shader.Define(stage,root+'_material/shader');shader.CreateIdAttr('UsdPreviewSurface');shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb));shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.8)
 mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface');UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(root)).Bind(mat,UsdShade.Tokens.strongerThanDescendants)
try:
 for engine in ['physx']:
  # Hide prior row's ghosts, build canonical bodies at actual recorded qpos.
  for root in ghost_roots:UsdGeom.Imageable(stage.GetPrimAtPath(root)).MakeInvisible()
  ghosts=[]
  for j,qp in enumerate(poses[engine]['qpos']):
   d.qpos[:]=qp;mujoco.mj_fwdPosition(m,d);root=f'/World/{engine}_pose_{j}';UsdGeom.Xform.Define(stage,root)
   for b in range(1,m.nbody):
    name=m.body(b).name;target=root+'/'+name;Sdf.CopySpec(stage.GetRootLayer(),'/World/Bike/'+name,stage.GetRootLayer(),target)
    p=stage.GetPrimAtPath(target);p.GetAttribute('xformOp:translate').Set(Gf.Vec3d(*map(float,d.xpos[b])));p.GetAttribute('xformOp:orient').Set(Gf.Quatf(float(d.xquat[b,0]),Gf.Vec3f(*map(float,d.xquat[b,1:]))))
    for child in Usd.PrimRange(p):
     for schema in list(child.GetAppliedSchemas()):
      if 'Physics' in schema or 'Physx' in schema:child.RemoveAppliedSchema(schema)
   ghosts.append(root);ghost_roots.append(root)
  UsdGeom.Imageable(stage.GetPrimAtPath('/World/Bike')).MakeInvisible()
  for variant in ['color','grayscale']:
   for j,root in enumerate(ghosts):paint(root,palette[j] if variant=='color' else (.20,.20,.20))
   for _ in range(12):world.render()
   rgb=camera.get_rgba()[:,:,:3].astype(np.uint8);assert rgb.shape==(750,1800,3) and rgb.std()>2
   Image.fromarray(rgb).save(OUT/'renders'/f'{engine}_overlay_{variant}.png');allimages[(engine,variant)]=Image.fromarray(rgb)
  # Five separate frames, same fixed level view and actual recorded states.
  for root in ghosts:UsdGeom.Imageable(stage.GetPrimAtPath(root)).MakeInvisible()
  UsdGeom.Imageable(stage.GetPrimAtPath('/World/Bike')).MakeVisible()
  camera.set_resolution((960,640))
  qa=[int(m.joint(n).qposadr[0]) for n in robot.dof_names]
  for j,qp in enumerate(poses[engine]['qpos']):
   qp=np.asarray(qp);robot.set_world_pose(qp[:3],qp[3:7]);robot.set_joint_positions(qp[qa]);world.physics_sim_view.update_articulations_kinematic();get_physx_interface().update_transformations(False,True,True)
   set_camera_view(eye=qp[:3]+[0,-3.6,.3],target=qp[:3]+[0,0,.3],camera_prim_path='/World/Camera')
   for _ in range(12):world.render()
   max_error=max(max_error,float(abs(robot.get_world_pose()[0]-qp[:3]).max()))
   Image.fromarray(camera.get_rgba()[:,:,:3].astype(np.uint8)).save(OUT/'renders'/f'{engine}_frame_{j}.png')
  camera.set_resolution((1800,750));set_camera_view(eye=[3.3,-7.2,.55],target=[3.3,0,.55],camera_prim_path='/World/Camera')
  print('RENDERED',engine,flush=True)
 assert not events
 (OUT/'renders/verification.json').write_text(json.dumps(dict(physics_steps=0,max_root_replay_error_m=max_error,pose_edits=False,cutoff='first sample rootx>=5m',source_states=str(OUT/'data/snapshot_states.json'),camera_eye=[3.3,-7.2,.55],camera_target=[3.3,0,.55],ground_visible=True,renderer='Isaac native RTX',knee_visual=knee_audit),indent=2))
except Exception:
 import traceback
 (OUT/'renders/error.log').write_text(traceback.format_exc());raise
finally:subscription.unsubscribe();app.close()
