"""Explicit in-memory terrain ablation; never modifies the authoritative XML."""
def move_platform_out_of_scene(model):
    g=model.geom('step').id
    original=model.geom_pos[g].copy()
    model.geom_pos[g,2]=-100.
    return dict(platform_removed=True,implementation='move named platform below ground before engine conversion; original XML untouched',
                original_position=original.tolist(),active_position=model.geom_pos[g].tolist(),
                active_top_z=float(model.geom_pos[g,2]+model.geom_size[g,2]),
                virtual_jump_zone_and_front_x_preserved=True)
