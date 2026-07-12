#!/usr/bin/env python3
"""Small FBX/BDG bridge for Blender workflows."""
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
    return path.is_dir() and (path / "import_log.json").exists() and any(path.glob("*.fbx"))


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


def run_tool(script: str, *args: str) -> None:
    if getattr(sys, "frozen", False):
        tool_path = TOOLS / script
        module_name = f"_gz_converter_{Path(script).stem}"
        spec = importlib.util.spec_from_file_location(module_name, tool_path)
        if not spec or not spec.loader:
            raise SystemExit(f"Missing bundled tool: {tool_path}")
        module = importlib.util.module_from_spec(spec)
        old_argv = sys.argv[:]
        try:
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
        return

    cmd = [sys.executable, str(TOOLS / script), *args]
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(ROOT) if not existing_pythonpath else str(ROOT) + os.pathsep + existing_pythonpath
    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc


def export_bdg(args: argparse.Namespace) -> int:
    input_path = clean_path(args.input)
    if input_path.suffix.lower() == ".cmg":
        return export_cmg(args, input_path)
    if input_path.suffix.lower() == ".cmp":
        raise SystemExit("CMP export tools were not recoverable after the bridge folder was overwritten.")

    base, shapes = find_shapes(input_path)
    out = character_output_folder(args.out, base)

    with tempfile.TemporaryDirectory(prefix="bdg_bridge_export_") as tmp:
        stage = Path(tmp)
        shutil.copy2(shapes, stage / shapes.name)
        copy_optional_pvms(shapes.parent, stage, base)

        run_tool("bdg_to_fbx_extract_all.py", str(stage), "--force")
        extracted = stage / f"{base}-Kaiju-Extracted"
        merge_tree(
            extracted,
            out,
            skip_file=lambda p: p.suffix.lower() == ".pvm" or p.name.lower() == "skeleton.txt",
        )
        stale_debug = out / "mesh_debug_obj"
        if stale_debug.exists() and stale_debug.is_dir():
            shutil.rmtree(stale_debug)

    fbx_files = sorted(out.glob("*.fbx"))
    print(f"Exported FBX project: {out}")
    print(f"FBX: {fbx_files[0] if fbx_files else out}")
    return 0


def export_cmg(args: argparse.Namespace, cmg: Path) -> int:
    if not cmg.exists():
        raise SystemExit(f"Missing CMG: {cmg}")
    out = character_output_folder(args.out, cmg.stem)
    out.mkdir(parents=True, exist_ok=True)
    fbx = out / f"{cmg.stem}.fbx"
    run_tool("cmg_probe.py", str(cmg), "--fbx", str(fbx), "--scale", "10")
    print(f"Exported CMG FBX project: {out}")
    print(f"FBX: {fbx}")
    return 0


def import_fbx(args: argparse.Namespace) -> int:
    fbx = clean_path(args.fbx)
    if not fbx.exists():
        raise SystemExit(f"Missing FBX: {fbx}")

    original = clean_path(args.original)
    if original.suffix.lower() == ".cmg":
        return import_cmg(args, fbx, original)
    if original.suffix.lower() == ".zip":
        member = cmg_zip_member(original)
        if member:
            return import_cmg_zip(args, fbx, original, member)

    project = clean_path(args.project) if getattr(args, "project", None) else fbx.parent
    manifest_path = bdg_import_log_path(project)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    base_from_original, shapes = find_shapes(original)
    if project.name.endswith("-Kaiju-Extracted"):
        base = project.name[:-len("-Kaiju-Extracted")]
    else:
        source_name = manifest.get("source_shapes") or manifest.get("source") or shapes.name
        base = re.sub(r"_Shapes\.BDG$", "", str(source_name), flags=re.I)
        if base == str(source_name):
            base = base_from_original

    out = clean_path(args.out) if args.out else (Path.cwd() / f"{base}-Kaiju-Reimported").resolve()

    anim_name = manifest.get("source_anim") or manifest.get("animation_source") or (base_from_original + ".BDG")
    anim = find_case_insensitive(shapes.parent, anim_name) if anim_name else None

    with tempfile.TemporaryDirectory(prefix="bdg_bridge_import_") as tmp:
        stage = Path(tmp)
        staged_project = stage / f"{base}-Kaiju-Extracted"
        shutil.copytree(project, staged_project)

        target_fbx_name = manifest.get("fbx") or fbx.name
        target_fbx = staged_project / target_fbx_name
        if not target_fbx.parent.exists():
            target_fbx.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fbx, target_fbx)

        shutil.copy2(shapes, stage / shapes.name)
        if anim:
            shutil.copy2(anim, stage / anim.name)
        copy_optional_pvms(shapes.parent, stage, base_from_original)

        run_tool("fbx_to_bdg_import_all.py", str(stage), "--force")
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
        run_tool("cmg_fbx_import.py", str(fbx), str(original_cmg), "--out", str(out_cmg), "--scale", "10")
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
    run_tool("cmg_fbx_import.py", str(fbx), str(original), "--out", str(out_cmg), "--scale", "10")
    if should_zip_damm_cmg(original.name):
        out_zip = out_root / f"{original.stem}.zip"
        write_single_file_zip(out_zip, out_cmg, original.name)
        out_cmg.unlink(missing_ok=True)
        print(f"Imported FBX into CMG zip: {out_zip}")
    else:
        print(f"Imported FBX into CMG copy: {out_cmg}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal Blender FBX <-> BDG/CMG bridge.")
    sub = parser.add_subparsers(dest="command", required=True)

    exp = sub.add_parser("export", help="Export a *_Shapes.BDG or .CMG to Blender-ready FBX")
    exp.add_argument("input", help="Input folder containing one *_Shapes.BDG, direct *_Shapes.BDG path, or .CMG path")
    exp.add_argument("--out", help="Output parent/project folder. BDG exports are placed in ./<character> by default.")
    exp.add_argument("--force", action="store_true", help="Accepted for old scripts; export overwrites same-name files but never deletes folders")
    exp.set_defaults(func=export_bdg)

    imp = sub.add_parser("import", help="Patch a Blender-edited FBX back into BDG/CMG copies")
    imp.add_argument("fbx", help="Edited FBX")
    imp.add_argument("--project", help="Project folder created by export")
    imp.add_argument("--original", required=True, help="Original *_Shapes.BDG or .CMG path")
    imp.add_argument("--out", help="Output folder")
    imp.add_argument("--force", action="store_true")
    imp.set_defaults(func=import_fbx)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
