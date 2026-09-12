from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
TOOLS = ROOT / "tools"
FROZEN_TOOL_FLAG = "--gz-bridge-tool"


def clean_path(value: str | Path) -> Path:
    text = str(value).strip()
    while text and text[-1] in ('"', "'"):
        text = text[:-1].rstrip()
    while text and text[0] in ('"', "'"):
        text = text[1:].lstrip()
    return Path(text).expanduser().resolve()


def kaiju_base_from_shapes(path: Path) -> str:
    base = re.sub(r"_Shapes\.BDG$", "", path.name, flags=re.I)
    if base == path.name:
        raise SystemExit(f"Expected a *_Shapes.BDG file, got: {path.name}")
    return base


def find_case_insensitive(folder: Path, name: str) -> Path | None:
    direct = folder / name
    if direct.exists():
        return direct
    lname = name.lower()
    for child in folder.iterdir():
        if child.name.lower() == lname:
            return child
    return None


def cmg_zip_member(zip_path: Path) -> str | None:
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            cmgs = [n for n in zf.namelist() if not n.endswith("/") and n.lower().endswith(".cmg")]
    except zipfile.BadZipFile as exc:
        raise SystemExit(f"Invalid zip file: {zip_path}") from exc
    if not cmgs:
        return None
    preferred = [n for n in cmgs if Path(n).stem.lower() == zip_path.stem.lower()]
    return sorted(preferred or cmgs, key=lambda n: (len(Path(n).parts), n.lower()))[0]


def should_zip_damm_cmg(cmg_name: str) -> bool:
    return re.search(r"_[012]\.cmg$", cmg_name, flags=re.I) is not None


def write_single_file_zip(zip_path: Path, file_path: Path, arcname: str | None = None) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(file_path, arcname or file_path.name)


def find_shapes(input_path: Path) -> tuple[str, Path]:
    if input_path.is_dir():
        shapes_files = sorted(p for p in input_path.iterdir() if p.is_file() and p.name.lower().endswith("_shapes.bdg"))
        if not shapes_files:
            raise SystemExit(f"No *_Shapes.BDG found in {input_path}")
        if len(shapes_files) > 1:
            listed = "\n  ".join(p.name for p in shapes_files)
            raise SystemExit(f"More than one *_Shapes.BDG found. Pass the file path instead:\n  {listed}")
        shapes = shapes_files[0]
    else:
        shapes = input_path
    if not shapes.exists():
        raise SystemExit(f"Missing Shapes BDG: {shapes}")
    return kaiju_base_from_shapes(shapes), shapes


def copy_optional_pvms(source_folder: Path, stage: Path, base: str) -> None:
    for pvm in sorted(source_folder.iterdir()):
        if pvm.is_file() and pvm.suffix.lower() == ".pvm" and pvm.name.lower().startswith(base.lower()):
            shutil.copy2(pvm, stage / pvm.name)


def looks_like_export_project(path: Path) -> bool:
    return path.is_dir() and any(path.glob("*.fbx"))


def bdg_import_log_path(project: Path) -> Path:
    path = project / "import_log.json"
    if path.exists():
        return path
    raise SystemExit(f"Missing import_log.json beside edited FBX: {project}")


def merge_tree(src: Path, dst: Path, skip_file=None) -> None:
    dst = dst.resolve()
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.rglob("*"):
        rel = item.relative_to(src)
        target = dst / rel
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            if skip_file and skip_file(item):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)


def character_output_folder(out_arg: str | None, base: str) -> Path:
    if out_arg:
        out = clean_path(out_arg)
        return out if out.name.lower() == base.lower() else (out / base).resolve()
    return (Path.cwd() / base).resolve()


def run_tool_in_process(script: str, *args: str) -> None:
    tool_path = TOOLS / script
    module_name = f"_gz_converter_{Path(script).stem}"
    spec = importlib.util.spec_from_file_location(module_name, tool_path)
    if not spec or not spec.loader:
        raise SystemExit(f"Missing bundled tool: {tool_path}")
    module = importlib.util.module_from_spec(spec)
    old_argv = sys.argv[:]
    old_sys_path = sys.path[:]
    try:
        # Dynamically loaded bundled tools import helpers from the same folder.
        # PyInstaller extracts those source files as data, so the tools folder
        # must be an explicit import root in the isolated worker process.
        tools_path = str(TOOLS)
        if tools_path not in sys.path:
            sys.path.insert(0, tools_path)
        sys.argv = [script, *args]
        spec.loader.exec_module(module)
        result = module.main()
        if result:
            raise SystemExit(result)
    except SystemExit:
        raise
    except BaseException as exc:
        raise SystemExit(f"{script} failed: {exc}") from exc
    finally:
        sys.argv = old_argv
        sys.path[:] = old_sys_path


def run_tool(script: str, *args: str) -> None:
    if getattr(sys, "frozen", False):
        # Keep CPU-heavy FBX parsing outside the Tk process. In a windowed
        # PyInstaller build sys.executable is this same EXE; bridge_gui handles
        # this private flag before constructing the GUI.
        cmd = [sys.executable, FROZEN_TOOL_FLAG, script, *args]
    else:
        cmd = [sys.executable, str(TOOLS / script), *args]
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(ROOT) if not existing_pythonpath else str(ROOT) + os.pathsep + existing_pythonpath
    print(f"Running {script}...", flush=True)
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
    return_code = process.wait()
    if return_code:
        raise SystemExit(return_code)
    print(f"Finished {script}.", flush=True)


def export_bdg(args: argparse.Namespace) -> int:
    input_path = clean_path(args.input)
    if input_path.suffix.lower() == ".cmg":
        return export_cmg(args, input_path)
    if input_path.suffix.lower() == ".cmp":
        return export_cmp(args, input_path)

    base, shapes = find_shapes(input_path)
    out = (
        clean_path(args.out)
        if getattr(args, "flat_out", False) and args.out
        else character_output_folder(args.out, base)
    )

    with tempfile.TemporaryDirectory(prefix="bdg_bridge_export_") as tmp:
        stage = Path(tmp)
        shutil.copy2(shapes, stage / shapes.name)
        anim_arg = getattr(args, "anim", None)
        if anim_arg:
            anim = clean_path(anim_arg)
            if not anim.exists():
                raise SystemExit(f"Missing Character.BDG: {anim}")
        else:
            anim = None
        if anim:
            shutil.copy2(anim, stage / f"{base}.BDG")
        copy_optional_pvms(shapes.parent, stage, base)

        export_args = [str(stage), "--force"]
        if getattr(args, "no_fbx_animations", False):
            export_args.append("--no-fbx-animations")
        run_tool("bdg_to_fbx_extract_all.py", *export_args)
        extracted = stage / f"{base}-Kaiju-Extracted"
        merge_tree(
            extracted,
            out,
            skip_file=lambda p: p.suffix.lower() == ".pvm" or p.name.lower() == "skeleton.txt",
        )
        for stale_name in ("mesh_debug_obj", "animations_raw"):
            stale_debug = out / stale_name
            if stale_debug.exists() and stale_debug.is_dir():
                shutil.rmtree(stale_debug)

    fbx_files = sorted(out.glob("*.fbx"))
    print(f"Exported FBX project: {out}")
    print(f"FBX: {fbx_files[0] if fbx_files else out}")
    return 0


def export_cmg(args: argparse.Namespace, cmg: Path) -> int:
    if not cmg.exists():
        raise SystemExit(f"Missing CMG: {cmg}")
    out = (
        clean_path(args.out)
        if getattr(args, "flat_out", False) and args.out
        else character_output_folder(args.out, cmg.stem)
    )
    out.mkdir(parents=True, exist_ok=True)
    fbx = out / f"{cmg.stem}.fbx"
    run_tool("cmg_probe.py", str(cmg), "--fbx", str(fbx), "--scale", "10")
    print(f"Exported CMG FBX project: {out}")
    print(f"FBX: {fbx}")
    return 0


def export_cmp(args: argparse.Namespace, cmp: Path) -> int:
    if not cmp.exists():
        raise SystemExit(f"Missing CMP: {cmp}")
    out = (
        clean_path(args.out)
        if getattr(args, "flat_out", False) and args.out
        else character_output_folder(args.out, cmp.stem)
    )
    out.mkdir(parents=True, exist_ok=True)
    fbx = out / f"{cmp.stem}.fbx"
    run_tool("cmp_probe.py", str(cmp), "--fbx", str(fbx), "--scale", "10")
    print(f"Exported CMP FBX project: {out}")
    print(f"FBX: {fbx}")
    return 0


def import_fbx(args: argparse.Namespace) -> int:
    fbx = clean_path(args.fbx)
    if not fbx.exists():
        raise SystemExit(f"Missing FBX: {fbx}")

    original = clean_path(args.original)
    if original.suffix.lower() == ".cmg":
        return import_cmg(args, fbx, original)
    if original.suffix.lower() == ".cmp":
        return import_cmp(args, fbx, original)
    if original.suffix.lower() == ".zip":
        member = cmg_zip_member(original)
        if member:
            return import_cmg_zip(args, fbx, original, member)
        raise SystemExit(f"Original zip does not contain a CMG: {original.name}")

    bundle_mode = getattr(args, "bundle_mode", None)
    if bundle_mode == "CMP/CMG":
        raise SystemExit(
            "CMP/CMG mode requires a .CMP, .CMG, or CMG .ZIP in Original file; "
            f"got: {original.name}"
        )
    if original.suffix.lower() != ".bdg":
        raise SystemExit(
            "Original file must be a .CMP, .CMG, CMG .ZIP, or *_Shapes.BDG; "
            f"got: {original.name}"
        )

    base_from_original, shapes = find_shapes(original)
    base = base_from_original
    project = clean_path(args.project) if getattr(args, "project", None) else fbx.parent
    out = clean_path(args.out) if args.out else (Path.cwd() / f"{base}-Kaiju-Reimported").resolve()

    anim_arg = getattr(args, "anim", None)
    if anim_arg:
        anim = clean_path(anim_arg)
        if not anim.exists():
            raise SystemExit(f"Missing Character.BDG: {anim}")
        anim_name = anim.name
    else:
        anim = None

    with tempfile.TemporaryDirectory(prefix="bdg_bridge_import_") as tmp:
        stage = Path(tmp)
        shutil.copy2(shapes, stage / shapes.name)
        if anim:
            shutil.copy2(anim, stage / anim.name)
        copy_optional_pvms(shapes.parent, stage, base_from_original)

        # Grow the native BDG skeletons before extracting the fresh import map.
        # The refreshed manifest then contains the new bone indices and all
        # mesh offsets after the Type 3/Type 4 resources changed size, allowing
        # new vertex weights to import on this same GUI pass.
        add_bone_args = [
            str(stage / shapes.name),
            str(fbx),
            str(stage / shapes.name),
            "--scale",
            "10",
        ]
        if anim:
            add_bone_args.extend(
                [
                    "--character",
                    str(stage / anim.name),
                    "--output-character",
                    str(stage / anim.name),
                ]
            )
        run_tool("bdg_add_bones.py", *add_bone_args)

        # Build a fresh import map from the donor files. The edited FBX is the
        # user artifact; import_log.json is internal metadata and is recreated
        # here so Wii imports work like the direct CMP/CMG paths.
        run_tool("bdg_to_fbx_extract_all.py", str(stage), "--force", "--no-fbx-animations")
        staged_project = stage / f"{base}-Kaiju-Extracted"
        manifest_path = staged_project / "import_log.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        # Preserve optional project-side texture and exact native animation
        # edits without importing a stale manifest or a stale FBX.
        if project.is_dir():
            for folder_name in ("textures", "animations_import", "animations_raw"):
                source_assets = project / folder_name
                if source_assets.is_dir():
                    merge_tree(source_assets, staged_project / folder_name)
            project_manifest_path = project / "import_log.json"
            if project_manifest_path.exists():
                project_manifest = json.loads(project_manifest_path.read_text(encoding="utf-8"))
                baseline_name = project_manifest.get("animation_action_baseline")
                baseline_path = project / str(baseline_name) if baseline_name else None
                if baseline_path and baseline_path.exists():
                    shutil.copy2(baseline_path, staged_project / baseline_path.name)
                    manifest["animation_action_baseline"] = baseline_path.name
                    manifest["fbx_bone_translation_baseline"] = project_manifest.get(
                        "fbx_bone_translation_baseline"
                    )
                    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            for pvm in sorted(project.glob("*.pvm")):
                shutil.copy2(pvm, staged_project / pvm.name)

        target_fbx_name = manifest.get("fbx") or fbx.name
        target_fbx = staged_project / target_fbx_name
        if not target_fbx.parent.exists():
            target_fbx.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fbx, target_fbx)

        import_args = [str(stage), "--force", "--with-skeleton"]
        if getattr(args, "keep_mesh_in_place", False):
            import_args.append("--keep-mesh-in-place")
        run_tool("fbx_to_bdg_import_all.py", *import_args)
        reimported = stage / f"{base}-Kaiju-Reimported"
        merge_tree(reimported, out)

    print(f"Imported FBX into BDG copies: {out}")
    return 0


def import_cmg_zip(args: argparse.Namespace, fbx: Path, original_zip: Path, member: str) -> int:
    with tempfile.TemporaryDirectory(prefix="cmg_bridge_zip_import_") as tmp:
        stage = Path(tmp)
        with zipfile.ZipFile(original_zip, "r") as zf:
            zf.extract(member, stage)
        original_cmg = stage / member
        original_cmg = original_cmg.resolve()
        cmg_name = Path(member).name
        out_root = clean_path(args.out) if args.out else (Path.cwd() / f"{Path(cmg_name).stem}-CMG-Reimported").resolve()
        out_root.mkdir(parents=True, exist_ok=True)
        out_cmg = out_root / cmg_name
        import_args = [str(fbx), str(original_cmg), "--out", str(out_cmg), "--scale", "10"]
        if getattr(args, "keep_mesh_in_place", False):
            import_args.append("--keep-mesh-in-place")
        run_tool("cmg_fbx_import.py", *import_args)
        if should_zip_damm_cmg(cmg_name):
            out_zip = out_root / f"{Path(cmg_name).stem}.zip"
            write_single_file_zip(out_zip, out_cmg, cmg_name)
            out_cmg.unlink(missing_ok=True)
            print(f"Imported FBX into CMG zip: {out_zip}")
        else:
            print(f"Imported FBX into CMG copy: {out_cmg}")
    return 0


def import_cmg(args: argparse.Namespace, fbx: Path, original: Path) -> int:
    if not original.exists():
        raise SystemExit(f"Missing original CMG: {original}")
    out_root = clean_path(args.out) if args.out else (Path.cwd() / f"{original.stem}-CMG-Reimported").resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    out_cmg = out_root / original.name
    import_args = [str(fbx), str(original), "--out", str(out_cmg), "--scale", "10"]
    if getattr(args, "keep_mesh_in_place", False):
        import_args.append("--keep-mesh-in-place")
    run_tool("cmg_fbx_import.py", *import_args)
    if should_zip_damm_cmg(original.name):
        out_zip = out_root / f"{original.stem}.zip"
        write_single_file_zip(out_zip, out_cmg, original.name)
        out_cmg.unlink(missing_ok=True)
        print(f"Imported FBX into CMG zip: {out_zip}")
    else:
        print(f"Imported FBX into CMG copy: {out_cmg}")
    return 0


def import_cmp(args: argparse.Namespace, fbx: Path, original: Path) -> int:
    if not original.exists():
        raise SystemExit(f"Missing original CMP: {original}")
    out_root = clean_path(args.out) if args.out else (Path.cwd() / f"{original.stem}-CMP-Reimported").resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    out_cmp = out_root / original.name
    run_tool("cmp_add_bones.py", str(original), str(fbx), str(out_cmp), "--scale", "10")
    import_args = [str(fbx), str(out_cmp), "--out", str(out_cmp), "--scale", "10"]
    if getattr(args, "keep_mesh_in_place", False):
        import_args.append("--keep-mesh-in-place")
    run_tool("cmp_fbx_import.py", *import_args)
    print(f"Imported FBX into CMP copy: {out_cmp}")
    return 0


def quick_export_import(args: argparse.Namespace) -> int:
    """Export the native files produced by a completed import back to FBX."""
    original = clean_path(args.original)
    suffix = original.suffix.lower()

    if suffix == ".cmp":
        out_root = clean_path(args.out) if args.out else (Path.cwd() / f"{original.stem}-CMP-Reimported").resolve()
        imported = out_root / original.name
        quick_root = out_root / "FBX-Reexport"
        export_args = argparse.Namespace(out=str(quick_root), flat_out=True)
        result = export_cmp(export_args, imported)
        print(f"Quick re-export saved under: {quick_root}")
        return result

    if suffix in (".cmg", ".zip"):
        if suffix == ".zip":
            member = cmg_zip_member(original)
            if not member:
                raise SystemExit(f"No CMG found in zip: {original}")
            cmg_name = Path(member).name
        else:
            cmg_name = original.name

        cmg_stem = Path(cmg_name).stem
        out_root = clean_path(args.out) if args.out else (Path.cwd() / f"{cmg_stem}-CMG-Reimported").resolve()
        quick_root = out_root / "FBX-Reexport"
        imported_cmg = out_root / cmg_name
        export_args = argparse.Namespace(out=str(quick_root), flat_out=True)

        if imported_cmg.exists():
            result = export_cmg(export_args, imported_cmg)
        else:
            imported_zip = out_root / f"{cmg_stem}.zip"
            if not imported_zip.exists():
                raise SystemExit(f"Missing imported CMG or zip in: {out_root}")
            with tempfile.TemporaryDirectory(prefix="cmg_bridge_quick_export_") as tmp:
                with zipfile.ZipFile(imported_zip, "r") as zf:
                    imported_member = cmg_zip_member(imported_zip)
                    if not imported_member:
                        raise SystemExit(f"No CMG found in imported zip: {imported_zip}")
                    zf.extract(imported_member, tmp)
                result = export_cmg(export_args, (Path(tmp) / imported_member).resolve())

        print(f"Quick re-export saved under: {quick_root}")
        return result

    base, shapes = find_shapes(original)
    out_root = clean_path(args.out) if args.out else (Path.cwd() / f"{base}-Kaiju-Reimported").resolve()
    imported_shapes = out_root / shapes.name
    imported_anim = None
    anim_arg = getattr(args, "anim", None)
    if anim_arg:
        imported_anim = out_root / clean_path(anim_arg).name
        if not imported_anim.exists():
            raise SystemExit(f"Missing imported Character.BDG: {imported_anim}")

    quick_root = out_root / "FBX-Reexport"
    export_args = argparse.Namespace(
        input=str(imported_shapes),
        anim=str(imported_anim) if imported_anim else None,
        out=str(quick_root),
        force=True,
        no_fbx_animations=False,
        flat_out=True,
    )
    result = export_bdg(export_args)
    print(f"Quick re-export saved under: {quick_root}")
    return result


def custom_model(args: argparse.Namespace) -> int:
    replacement = clean_path(args.fbx)
    template = clean_path(args.template)
    if not replacement.exists():
        raise SystemExit(f"Missing replacement model: {replacement}")
    if not template.exists():
        raise SystemExit(f"Missing template model: {template}")
    if template.suffix.lower() == ".bdg":
        if replacement.suffix.lower() not in (".fbx", ".bdg"):
            raise SystemExit("A BDG custom model requires a replacement FBX or donor *_Shapes.BDG")
        out_root = clean_path(args.out) if args.out else (Path.cwd() / f"{template.stem}-BDG-Custom").resolve()
        out_root.mkdir(parents=True, exist_ok=True)
        out_bdg = out_root / template.name
        tool_args = [str(replacement), str(template), "--out", str(out_bdg), "--scale", "10"]
        if replacement.suffix.lower() == ".fbx" and getattr(args, "bind_unweighted_root", False):
            tool_args.append("--bind-unweighted-root")
        run_tool("bdg_custom_model.py", *tool_args)
        print(f"Built custom model into template BDG copy: {out_bdg}")
        return 0
    if template.suffix.lower() != ".cmp":
        raise SystemExit(f"Custom Model requires a CMP or *_Shapes.BDG template, got: {template.name}")
    if replacement.suffix.lower() != ".fbx":
        raise SystemExit("A CMP custom model requires a replacement FBX")
    out_root = clean_path(args.out) if args.out else (Path.cwd() / f"{template.stem}-CMP-Custom").resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    out_cmp = out_root / template.name
    tool_args = [str(replacement), str(template), "--out", str(out_cmp), "--scale", "10"]
    if getattr(args, "bind_unweighted_root", False):
        tool_args.append("--bind-unweighted-root")
    run_tool("cmp_custom_model.py", *tool_args)
    print(f"Built custom FBX model into CMP copy: {out_cmp}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal Blender FBX <-> BDG/CMG/CMP bridge.")
    sub = parser.add_subparsers(dest="command", required=True)

    exp = sub.add_parser("export", help="Export a *_Shapes.BDG, .CMG, or .CMP to Blender-ready FBX")
    exp.add_argument("input", help="Input folder containing one *_Shapes.BDG, direct *_Shapes.BDG path, .CMG path, or .CMP path")
    exp.add_argument("--anim", help="Optional Character.BDG to include with BDG exports for animation data")
    exp.add_argument("--out", help="Output parent/project folder. BDG exports are placed in ./<character> by default.")
    exp.add_argument("--force", action="store_true", help="Accepted for old scripts; export overwrites same-name files but never deletes folders")
    exp.add_argument("--no-fbx-animations", action="store_true", help=argparse.SUPPRESS)
    exp.set_defaults(func=export_bdg)

    imp = sub.add_parser("import", help="Patch a Blender-edited FBX back into BDG/CMG/CMP copies")
    imp.add_argument("fbx", help="Edited FBX")
    imp.add_argument("--project", help="Optional folder containing edited textures or native animation bins")
    imp.add_argument("--original", required=True, help="Original *_Shapes.BDG, .CMG, or .CMP path")
    imp.add_argument("--anim", help="Optional Character.BDG to include with BDG imports for animation data")
    imp.add_argument("--out", help="Output folder")
    imp.add_argument("--force", action="store_true")
    imp.add_argument(
        "--keep-mesh-in-place",
        action="store_true",
        help="Import rest-bone positions without automatically moving their weighted vertices",
    )
    imp.set_defaults(func=import_fbx)

    custom = sub.add_parser("custom-model", help="Replace template CMP or BDG geometry with a replacement model")
    custom.add_argument("fbx", help="Replacement FBX, or legacy donor *_Shapes.BDG")
    custom.add_argument("--template", required=True, help="Template CMP or *_Shapes.BDG providing skeleton and materials")
    custom.add_argument("--out", help="Output folder")
    custom.add_argument(
        "--bind-unweighted-root",
        action="store_true",
        help="Rigidly bind unweighted vertices to the template's shallowest usable skin bone",
    )
    custom.set_defaults(func=custom_model)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
