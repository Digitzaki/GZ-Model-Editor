from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("output")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else [])

    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=str(source), automatic_bone_orientation=False)
    scene = bpy.context.scene
    for armature in (obj for obj in scene.objects if obj.type == "ARMATURE"):
        armature.data.pose_position = "POSE"
        if armature.animation_data is not None:
            armature.animation_data.action = None
        for pose_bone in armature.pose.bones:
            pose_bone.matrix_basis.identity()
    scene.frame_set(0)
    bpy.context.view_layer.update()
    bpy.ops.export_scene.fbx(
        filepath=str(output),
        use_selection=False,
        apply_unit_scale=True,
        bake_space_transform=False,
        add_leaf_bones=False,
        bake_anim=True,
        bake_anim_use_all_bones=False,
        bake_anim_use_nla_strips=False,
        bake_anim_use_all_actions=True,
        bake_anim_force_startend_keying=False,
        bake_anim_step=1.0,
        bake_anim_simplify_factor=0.0,
        path_mode="AUTO",
    )


if __name__ == "__main__":
    main()
