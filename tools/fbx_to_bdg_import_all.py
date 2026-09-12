from pathlib import Path
import argparse, collections, json, shutil, sys, zipfile

TOOL_DIR=Path(__file__).resolve().parent
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0,str(TOOL_DIR))

from fbx_to_bdg_import import (
    clean_windows_folder_arg, find_case_insensitive, patch_textures,
    patch_mesh_from_fbx, patch_skeleton_from_fbx, patch_raw_anims,
    patch_deleted_bone_animation_locks, patch_mesh_for_skeleton_position_edits,
    patch_type3_skeleton_from_report, patch_type4_skeleton_pose,
)

def import_one(
    root: Path,
    extracted: Path,
    force=False,
    patch_unchanged=False,
    with_skeleton=False,
    keep_mesh_in_place=False,
):
    log_path=extracted/'import_log.json'
    manifest=json.loads(log_path.read_text(encoding='utf-8'))
    base=extracted.name[:-len('-Kaiju-Extracted')] if extracted.name.endswith('-Kaiju-Extracted') else extracted.name
    out=root/f'{base}-Kaiju-Reimported'
    if out.exists():
        if force:
            shutil.rmtree(out)
        else:
            raise ValueError(f'output exists: {out}')
    out.mkdir(parents=True)

    shape_name=manifest.get('source_shapes') or manifest.get('source')
    anim_name=manifest.get('source_anim') or manifest.get('animation_source')
    shape_src=find_case_insensitive(root, shape_name or '')
    anim_src=find_case_insensitive(root, anim_name or '') if anim_name else None
    if shape_src is None or (anim_name and anim_src is None):
        raise ValueError(f'missing original BDGs: {shape_name}, {anim_name}')

    staged_shape=out/shape_src.name
    shutil.copy2(shape_src, staged_shape)
    staged_anim=None
    if anim_src:
        staged_anim=out/anim_src.name
        shutil.copy2(anim_src, staged_anim)

    copied=[]
    pvm_names=manifest.get('pvms') or [p.name for p in root.glob('*.pvm') if p.name.lower().startswith(base.lower())]
    for pvm_name in pvm_names:
        p=find_case_insensitive(root, pvm_name)
        if p and not (out/p.name).exists():
            shutil.copy2(p, out/p.name); copied.append(p.name)
    for p in sorted(extracted.glob('*.pvm')):
        if not (out/p.name).exists():
            shutil.copy2(p, out/p.name); copied.append(p.name)

    report={
        'source_extracted':extracted.name,
        'output':out.name,
        'copied':[staged_shape.name]+(([staged_anim.name] if staged_anim else []))+sorted(set(copied)),
        'texture_patches':[],
        'animation_resource_patches':[],
        'mesh_patch':{'status':'not_run'},
        'skeleton_patch':{'status':'not_run'},
        'animation_action_import':[],
        'limits':[
            'Mesh import requires the same FBX polygon order/topology as the extracted mesh.',
            'BDG vertex streams have fixed influence limits; extra FBX weights are reduced and reported.',
            'New FBX leaf bones are added before mesh import and may receive weights during their first GUI import.',
            'Geometry positions/normals/UVs/weights import from same-topology FBX.',
            'Changed BDG skeleton local positions import from FBX bone nodes; deleted/missing bone nodes preserve the original skeleton record.',
            'Texture PNGs import back into Shapes.BDG when encoded format is supported.',
            'Edited FBX Actions are rebuilt as native Type 4 clips when the extraction contains an Action baseline.',
            'Exact native animation BIN resources may still be supplied in animations_import/ or animations_raw/.',
        ],
    }

    shape=bytearray(staged_shape.read_bytes())
    anim=bytearray(staged_anim.read_bytes()) if staged_anim else bytearray()

    patch_textures(shape, extracted, manifest, report, patch_unchanged=patch_unchanged)
    try:
        patch_mesh_from_fbx(shape, extracted, manifest, report, patch_unchanged=patch_unchanged)
    except Exception as e:
        report['mesh_patch']={'status':f'error: {type(e).__name__}: {e}'}
    if with_skeleton:
        try:
            patch_skeleton_from_fbx(shape, extracted, manifest, report, patch_unchanged=patch_unchanged)
            if staged_anim:
                patch_type3_skeleton_from_report(anim, report, 'animation')
            patch_type4_skeleton_pose(shape, report, 'shapes')
            if staged_anim:
                patch_type4_skeleton_pose(anim, report, 'animation')
            if keep_mesh_in_place:
                report['mesh_skeleton_bake']={
                    'status':'skipped_keep_mesh_in_place',
                    'vertices_moved':0,
                }
            else:
                patch_mesh_for_skeleton_position_edits(shape, manifest, report)
        except Exception as e:
            report['skeleton_patch']={'status':f'error: {type(e).__name__}: {e}'}
    else:
        report['skeleton_patch']={'status':'skipped_by_default_use_--with-skeleton'}
    if staged_anim:
        if manifest.get('animation_action_baseline'):
            from bdg_animation_import import import_bdg_actions

            fbx_path=extracted/manifest.get('fbx', f'{base}.fbx')
            anim, _action_results=import_bdg_actions(bytes(anim), fbx_path, manifest, report)
            anim=bytearray(anim)
        else:
            report['animation_action_import'].append({'status':'skipped_missing_action_baseline'})
        patch_raw_anims(anim, extracted, manifest, report, patch_unchanged=patch_unchanged)
        patch_deleted_bone_animation_locks(anim, extracted, manifest, report)
    else:
        patch_deleted_bone_animation_locks(anim, extracted, manifest, report)

    staged_shape.write_bytes(shape)
    if staged_anim:
        staged_anim.write_bytes(anim)

    # Keep the game-ready .zip wrappers in sync with the patched loose BDGs.
    zip_outputs=[]
    for bdg in [staged_shape] + ([staged_anim] if staged_anim else []):
        zpath=out/(bdg.stem + '.zip')
        with zipfile.ZipFile(zpath, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(bdg, arcname=bdg.name)
        zip_outputs.append(zpath.name)
    report['zip_outputs']=zip_outputs
    (out/'import_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report

def main():
    ap=argparse.ArgumentParser(description='Import every *-Kaiju-Extracted folder with real same-topology writeback where supported.')
    ap.add_argument('folder',nargs='?',default='.')
    ap.add_argument('--all',action='store_true')
    ap.add_argument('--force',action='store_true')
    ap.add_argument('--patch-unchanged',action='store_true')
    ap.add_argument('--with-skeleton', action='store_true', help='Also import FBX rest-pose bone transforms. Off by default because Blender FBX axis conversion can rotate monsters in-game.')
    ap.add_argument('--keep-mesh-in-place', action='store_true', help='Do not automatically move weighted vertices with edited rest-bone positions.')
    args=ap.parse_args()
    root=Path(clean_windows_folder_arg(args.folder)).resolve()
    folders=sorted([p for p in root.iterdir() if p.is_dir() and p.name.endswith('-Kaiju-Extracted')])
    if not folders:
        raise SystemExit('No *-Kaiju-Extracted folders found.')
    if len(folders)>1 and not args.all:
        raise SystemExit('More than one extracted folder found. Pass --all.')
    reports=[]; errors=[]
    for f in folders:
        print(f'== Importing {f.name} ==')
        try:
            r=import_one(
                root,
                f,
                force=args.force,
                patch_unchanged=args.patch_unchanged,
                with_skeleton=args.with_skeleton,
                keep_mesh_in_place=args.keep_mesh_in_place,
            )
            reports.append({
                'folder':f.name,
                'status':'ok',
                'output':r['output'],
                'textures':dict(collections.Counter(x.get('status','?') for x in r['texture_patches'])),
                'mesh':r['mesh_patch'].get('status'),
                'skeleton':r['skeleton_patch'].get('status'),
                'raw_anims':dict(collections.Counter(x.get('status','?') for x in r['animation_resource_patches'])),
            })
            print('   ok:', r['output'], 'mesh=', r['mesh_patch'].get('status'), 'skeleton=', r['skeleton_patch'].get('status'))
        except Exception as e:
            errors.append({'folder':f.name,'status':'error','error':str(e)})
            print('   ERROR:',e)
    (root/'all_kaiju_import_report.json').write_text(json.dumps({'reports':reports,'errors':errors},indent=2,default=str),encoding='utf-8')
    if errors:
        print(f'Finished with {len(errors)} errors. See all_kaiju_import_report.json')
        raise SystemExit(1)
    print('Finished imports. See all_kaiju_import_report.json')

if __name__=='__main__':
    main()
