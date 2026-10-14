"""Historical knee stretch applied only to renderer-local visual geometry."""
import numpy as np
from scipy.spatial.transform import Rotation

def stretch_parameters(m):
 b=m.body('uparm').id
 ids=[g for g in range(m.ngeom) if m.geom_bodyid[g]==b and m.geom_rgba[g,3]>0 and not(m.geom_contype[g] or m.geom_conaffinity[g])]
 g=next(g for g in ids if int(m.geom_type[g])==7);mid=int(m.geom_dataid[g])
 adr=int(m.mesh_vertadr[mid]);count=int(m.mesh_vertnum[mid]);v=np.array(m.mesh_vert[adr:adr+count])
 r=Rotation.from_quat(m.geom_quat[g][[1,2,3,0]]);body_v=r.apply(v)+m.geom_pos[g]
 z0=float(body_v[:,2].min());z1=float(body_v[:,2].max());anchor=float(m.jnt_pos[m.joint('knee_joint').id,2]);scale=(anchor-z0)/(z1-z0)
 stretched=body_v.copy();stretched[:,2]=z0+(stretched[:,2]-z0)*scale
 points=r.inv().apply(stretched-m.geom_pos[g])
 audit=dict(variant='stretch',stretch_factor=scale,stretch_anchor_z_m=z0,mesh_end_before_m=z1,mesh_end_after_m=float(stretched[:,2].max()),knee_anchor_z_m=anchor,visual_geom_ids=ids,pose_edits=False)
 assert abs(audit['mesh_end_after_m']-anchor)<1e-9
 return ids,g,mid,points,z0,scale,audit

def stretch_mujoco_visual(m):
 ids,g,mid,points,z0,scale,audit=stretch_parameters(m)
 # Compiled mass/inertia stay frozen: no mj_setConst, no model recompilation.
 before={k:np.array(getattr(m,k),copy=True) for k in ['body_mass','body_inertia','body_ipos','body_iquat','jnt_pos','jnt_axis','geom_friction','geom_contype','geom_conaffinity','actuator_gainprm','actuator_biasprm']}
 collider=np.flatnonzero(m.geom_contype|m.geom_conaffinity)
 collision_before={k:np.array(getattr(m,k)[collider],copy=True) for k in ['geom_pos','geom_quat','geom_size']}
 adr=int(m.mesh_vertadr[mid]);m.mesh_vert[adr:adr+len(points)]=points
 for gi in ids:
  if gi!=g:
   m.geom_pos[gi,2]=z0+(m.geom_pos[gi,2]-z0)*scale;m.geom_size[gi,2]*=scale
 for k,v in before.items():np.testing.assert_array_equal(v,getattr(m,k))
 for k,v in collision_before.items():np.testing.assert_array_equal(v,getattr(m,k)[collider])
 audit['physical_properties_unchanged']=True
 return audit

def stretch_isaac_visual(stage,m):
 from pxr import Usd,UsdGeom,UsdPhysics,Gf
 def physical_snapshot():
  return {str(p.GetPath()):([str(s) for s in p.GetAppliedSchemas() if 'Physics' in s or 'Physx' in s],{a.GetName():str(a.Get()) for a in p.GetAttributes() if a.GetName().startswith(('physics:','physx'))},{r.GetName():str(r.GetTargets()) for r in p.GetRelationships() if r.GetName().startswith(('physics:','physx'))}) for p in stage.Traverse() if any('Physics' in s or 'Physx' in s for s in p.GetAppliedSchemas())}
 before=physical_snapshot();ids,g,mid,points,z0,scale,audit=stretch_parameters(m)
 for gi in ids:
  name=m.geom(gi).name or f'geom_{gi}';p=stage.GetPrimAtPath(f'/World/Bike/uparm/{name}_g{gi}')
  assert p and not p.HasAPI(UsdPhysics.CollisionAPI)
  if gi==g:UsdGeom.Mesh(p).GetPointsAttr().Set(points.tolist())
  else:
   attr=p.GetAttribute('xformOp:translate');pos=np.array(attr.Get());pos[2]=z0+(pos[2]-z0)*scale;attr.Set(Gf.Vec3d(*map(float,pos)))
   attr=p.GetAttribute('xformOp:scale');size=np.array(attr.Get());size[2]*=scale;attr.Set(Gf.Vec3f(*map(float,size)))
 assert before==physical_snapshot()
 audit['physical_properties_unchanged']=True;audit['physical_prim_count']=len(before)
 return audit
