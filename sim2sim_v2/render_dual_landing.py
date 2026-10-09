"""Native MuJoCo and Isaac RTX replay of immutable, verified physical states."""
import os,argparse,json,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--engine',choices=['mjx','physx'],required=True);p.add_argument('--delivery',type=Path,required=True);a=p.parse_args()
ROOT=Path(__file__).resolve().parent;D=a.delivery.resolve();E=D/'evidence'/a.engine;O=D/'videos';O.mkdir(exist_ok=True)
os.environ.update(OMNI_KIT_ACCEPT_EULA='YES',MUJOCO_GL='egl',OPENBLAS_NUM_THREADS='1')
if a.engine=='physx':
 from isaacsim import SimulationApp
 app=SimulationApp({'headless':True,'disable_viewport_updates':True,'width':960,'height':640,'renderer':'RaytracedLighting','anti_aliasing':0,'limit_cpu_threads':16,'extra_args':['--/app/settings/loadUserConfig=false','--/app/settings/persistent=false']})
import numpy as np,imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
manifest=json.loads((D/'manifest.json').read_text());land=json.loads((E/('landing.json' if a.engine=='mjx' else 'landing_lane_0.json')).read_text());tr=np.load(E/'trace.npz')
physics_events=[];err=0.;jerr=0.
if a.engine=='physx':
 import omni.usd
 from pxr import UsdLux
 from isaacsim.core.api import World
 from isaacsim.core.prims import SingleArticulation
 from isaacsim.sensors.camera import Camera
 from isaacsim.core.utils.viewports import set_camera_view
 from omni.physx import get_physx_interface
 from usd_model import build
 from scene_visuals import enhance
 world=World(physics_dt=.005,rendering_dt=.02);world.get_physics_context().enable_fabric(False);stage=omni.usd.get_context().get_stage()
 model,_=build(stage,D/'physical_models/physx_conversion_source/source.xml');enhance(stage,model);light=UsdLux.DomeLight.Define(stage,'/World/Light');light.CreateIntensityAttr(1100)
 robot=world.scene.add(SingleArticulation('/World/Bike',name='bike'));world.reset()
 camera=Camera('/World/Camera',frequency=-1,resolution=(960,640));camera.initialize();camera.set_focal_length(3.)
 sub=get_physx_interface().subscribe_physics_step_events(lambda dt:physics_events.append(float(dt)))
 raw=tr['trace'][:,0];t=(np.arange(len(raw))+1)*.02;pos=raw[:,1:4];quat=raw[:,4:8];names=json.loads((E/'audit.json').read_text())['dof_names'];joint=raw[:,8:13][:,[names.index(n) for n in robot.dof_names]];velocity=raw[:,18:21]
else:
 import mujoco
 model=mujoco.MjModel.from_xml_path(str(D/'physical_models/mjx_assets/orange_bike_4kg_horizontal.xml'));model.vis.global_.offwidth=960;model.vis.global_.offheight=640
 data=mujoco.MjData(model);renderer=mujoco.Renderer(model,height=640,width=960);camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE;camera.distance=3.5;camera.azimuth=90;camera.elevation=-8
 t=tr['time_s'];pos=tr['qpos'][:,:3];velocity=tr['qvel'][:,:3]
frames=0
with imageio.get_writer(O/f'{a.engine}_realtime.mp4',fps=50,codec='libx264',quality=8,macro_block_size=16) as writer:
 for i,time_s in enumerate(t):
  if a.engine=='physx':
   robot.set_world_pose(pos[i],quat[i]);robot.set_joint_positions(joint[i]);world.physics_sim_view.update_articulations_kinematic();get_physx_interface().update_transformations(False,True,True)
   set_camera_view(eye=pos[i]+[.8,-3.8,.50],target=pos[i]+[.25,0,.35],camera_prim_path='/World/Camera')
   for _ in range(30 if i==0 else 3):world.render()
   rgb=camera.get_rgba()[:,:,:3].astype(np.uint8);err=max(err,float(np.max(abs(robot.get_world_pose()[0]-pos[i]))));jerr=max(jerr,float(np.max(abs(robot.get_joint_positions()-joint[i]))))
  else:
   data.qpos[:]=tr['qpos'][i];data.qvel[:]=tr['qvel'][i];mujoco.mj_forward(model,data);camera.lookat[:]=pos[i]+[.25,0,.3];renderer.update_scene(data,camera=camera);rgb=renderer.render().copy();assert data.time==0
   err=max(err,float(np.max(abs(data.qpos-tr['qpos'][i]))))
  assert rgb.shape==(640,960,3) and rgb.std()>1
  im=Image.fromarray(rgb);draw=ImageDraw.Draw(im);draw.rectangle((0,0,960,85),fill=(13,20,29))
  reached=land['success_times']['1.0'] is not None and time_s+1e-8>=land['success_times']['1.0']
  lines=[('Isaac / PhysX RTX' if a.engine=='physx' else 'MJX-Warp states / MuJoCo renderer')+' | transition_01966080',f't={time_s:.2f}s | vx={velocity[i,0]:.2f} m/s (diagnostic) | sim/control=5/20 ms',('Stable landing 1s verified' if reached else 'Natural takeoff / landing sequence')+' | recorded-state replay, no pose edits']
  for k,line in enumerate(lines):draw.text((12,5+26*k),line,font=font,fill='white' if k<2 else '#9DE4B5')
  writer.append_data(np.asarray(im));frames+=1
  if i in [0,int(np.argmin(abs(t-land['apex_time_s']))),int(np.argmin(abs(t-land['success_times']['1.0']))),len(t)-1]:im.save(O/f'{a.engine}_{i:04d}.png')
  if i%50==0:print(a.engine,'FRAME',i,flush=True)
assert not physics_events
verification=dict(engine=a.engine,renderer='Isaac RTX' if a.engine=='physx' else 'MuJoCo EGL',video_kind='recorded-state replay; no new dynamics rollout or manually changed pose',actor_sha256=manifest['actor_sha256'],trace_sha256=hashlib.sha256((E/'trace.npz').read_bytes()).hexdigest(),frames=frames,fps=50,duration_s=frames/50,physics_steps_during_render=0,max_pose_replay_error=err,max_joint_replay_error=jerr)
(O/f'{a.engine}_verification.json').write_text(json.dumps(verification,indent=2));print(verification,flush=True)
if a.engine=='physx':sub.unsubscribe();app.close()
else:renderer.close()
