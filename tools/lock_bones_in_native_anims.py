from __future__ import annotations

import argparse
import fnmatch
import json
import math
import shutil
import struct
from pathlib import Path


def clean_arg(value: str) -> str:
    text = str(value).strip()
    while text and text[-1] in ('"', "'"):
        text = text[:-1].rstrip()
    while text and text[0] in ('"', "'"):
        text = text[1:].lstrip()
    return text or "."


def quat_xyz_to_i16(q):
    return tuple(max(-32767, min(32767, int(round(float(v) * 32767.0)))) for v in q[:3])


def find_project(path: Path) -> Path:
    if path.is_file():
        path = path.parent
    if (path / "import_log.json").exists():
        return path
    folders = sorted(p for p in path.iterdir() if p.is_dir() and (p / "import_log.json").exists())
    if len(folders) == 1:
        return folders[0]
    if not folders:
        raise SystemExit(f"No extracted project with import_log.json found in {path}")
    raise SystemExit("More than one extracted project found. Pass the exact *-Kaiju-Extracted folder.")


def match_bones(manifest: dict, patterns: list[str]) -> dict[int, dict]:
    bones = {int(b["idx"]): b for b in manifest.get("bones", [])}
    selected = {}
    lowered = [p.lower() for p in patterns]
    for idx, bone in bones.items():
        name = str(bone.get("name", ""))
        lname = name.lower()
        for pat in lowered:
            if fnmatch.fnmatch(lname, pat) or pat in lname or pat == str(idx):
                selected[idx] = bone
                break
    return selected


def first_record_xyz(records, layout: str):
    if not records:
        return None
    rec = records[0]
    if layout == "explicit_qxyz_time" and len(rec) >= 3:
        return tuple(int(x) for x in rec[:3])
    if layout == "continuation_time_qxyz" and len(rec) >= 4:
        return tuple(int(x) for x in rec[1:4])
    return None


def patch_track(payload: bytearray, track: dict, xyz: tuple[int, int, int]) -> int:
    rel = int(str(track["track_rel"]), 16)
    count = int(track["record_count"])
    layout = str(track["layout"])
    changed = 0
    if layout == "explicit_qxyz_time":
        for i in range(count):
            off = rel + 4 + i * 8
            if off + 6 > len(payload):
                break
            struct.pack_into(">hhh", payload, off, *xyz)
            changed += 1
    elif layout == "continuation_time_qxyz":
        for i in range(count):
            off = rel + i * 8 + 2
            if off + 6 > len(payload):
                break
            struct.pack_into(">hhh", payload, off, *xyz)
            changed += 1
    return changed


def safe_raw_name(anim: dict) -> str:
    if anim.get("safe_filename"):
        return str(anim["safe_filename"])
    name = str(anim.get("name", ""))
    return "".join(c if c.isalnum() or c in "_.-" else "_" for c in name)[:120] + ".bin"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Freeze selected native BDG animation bone tracks in an extracted bridge project."
    )
    ap.add_argument("project", help="Path to a *-Kaiju-Extracted folder, or a folder containing one.")
    ap.add_argument("--bone", action="append", required=True, help="Bone id/name/pattern. Repeat for children.")
    ap.add_argument(
        "--mode",
        choices=("first-key", "rest"),
        default="first-key",
        help="first-key freezes each clip at its first keyed pose; rest freezes to the Shapes.BDG bind/rest pose.",
    )
    ap.add_argument("--clip", action="append", help="Optional animation clip name/pattern. Repeat to include more.")
    ap.add_argument("--dry-run", action="store_true", help="Report tracks that would be patched without writing files.")
    args = ap.parse_args()

    project = find_project(Path(clean_arg(args.project)).resolve())
    manifest = json.loads((project / "import_log.json").read_text(encoding="utf-8"))
    raw_dir = project / "animations_raw"
    native_manifest = raw_dir / "animation_native_tracks_v11.json"
    if not native_manifest.exists():
        raise SystemExit(f"Missing native track manifest: {native_manifest}")

    selected = match_bones(manifest, args.bone)
    if not selected:
        raise SystemExit("No bones matched: " + ", ".join(args.bone))

    clip_patterns = [p.lower() for p in (args.clip or ["*"])]
    tracks_manifest = json.loads(native_manifest.read_text(encoding="utf-8"))
    report = {
        "project": project.name,
        "mode": args.mode,
        "bones": [{"idx": idx, "name": b.get("name")} for idx, b in sorted(selected.items())],
        "clips": [],
        "dry_run": bool(args.dry_run),
    }
    patched_files = 0
    patched_tracks = 0
    patched_records = 0

    raw_entries = {int(a["resource_id"]): a for a in manifest.get("animation_resource_locations", []) if "resource_id" in a}
    for anim in tracks_manifest:
        clip_name = str(anim.get("name", ""))
        if not any(fnmatch.fnmatch(clip_name.lower(), pat) or pat in clip_name.lower() for pat in clip_patterns):
            continue
        raw_entry = raw_entries.get(int(anim.get("resource_id", -1)), {})
        raw_name = safe_raw_name(raw_entry or anim)
        raw_path = raw_dir / raw_name
        if not raw_path.exists():
            report["clips"].append({"name": clip_name, "status": "missing_raw_bin", "raw": raw_name})
            continue
        payload = bytearray(raw_path.read_bytes())
        clip_report = {"name": clip_name, "raw": raw_name, "tracks": []}
        for tr in anim.get("tracks", []):
            bone_id = int(tr.get("bone_id", -1))
            if bone_id not in selected:
                continue
            if args.mode == "rest":
                xyz = quat_xyz_to_i16(selected[bone_id].get("local_quaternion_xyzw", (0, 0, 0, 1)))
            else:
                xyz = first_record_xyz(tr.get("records", []), str(tr.get("layout", "")))
            if xyz is None:
                continue
            changed = patch_track(payload, tr, xyz)
            if changed:
                patched_tracks += 1
                patched_records += changed
                clip_report["tracks"].append({
                    "bone_id": bone_id,
                    "bone_name": tr.get("bone_name"),
                    "layout": tr.get("layout"),
                    "records_patched": changed,
                    "xyz_i16": list(xyz),
                })
        if clip_report["tracks"]:
            patched_files += 1
            if not args.dry_run:
                backup = raw_path.with_suffix(raw_path.suffix + ".bak")
                if not backup.exists():
                    shutil.copy2(raw_path, backup)
                raw_path.write_bytes(payload)
            clip_report["status"] = "would_patch" if args.dry_run else "patched"
        else:
            clip_report["status"] = "no_matching_tracks"
        report["clips"].append(clip_report)

    report["summary"] = {
        "raw_files_patched": patched_files,
        "tracks_patched": patched_tracks,
        "records_patched": patched_records,
    }
    out_report = project / "bone_lock_report.json"
    out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"Wrote {out_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
