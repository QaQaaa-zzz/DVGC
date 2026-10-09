"""Read active USD materials and bindings; do not confuse them with solver readback."""
def summarize_materials(materials):
    wheels=[(x['static_friction'],x['dynamic_friction']) for n,x in materials.items() if 'wheel' in n]
    modes={x['combine'] for x in materials.values()}
    return dict(materials=materials,
        wheel_static_dynamic_friction=list(wheels[0]) if wheels and len(set(wheels))==1 else None,
        all_material_combine=next(iter(modes)) if len(modes)==1 else 'mixed' if modes else None)


def read_contact_audit(stage,model):
    from pxr import UsdPhysics,PhysxSchema,UsdShade
    materials={};bindings=[]
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.MaterialAPI):
            api=UsdPhysics.MaterialAPI(prim);name=prim.GetName()
            g=int(name.split('_')[-1]) if name.startswith('material_') else -1
            key=model.geom(g).name if 0<=g<model.ngeom else str(prim.GetPath())
            materials[key]=dict(path=str(prim.GetPath()),static_friction=api.GetStaticFrictionAttr().Get(),
                dynamic_friction=api.GetDynamicFrictionAttr().Get(),
                combine=PhysxSchema.PhysxMaterialAPI(prim).GetFrictionCombineModeAttr().Get())
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            mat,_=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(materialPurpose='physics')
            pc=PhysxSchema.PhysxCollisionAPI(prim)
            bindings.append(dict(collider=str(prim.GetPath()),material=str(mat.GetPath()),
                contact_offset=pc.GetContactOffsetAttr().Get(),rest_offset=pc.GetRestOffsetAttr().Get()))
    result=summarize_materials(materials)
    result.update(bindings=bindings,evidence='active USD stage after reset; not a solver material readback')
    return result
