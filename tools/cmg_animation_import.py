from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

from bdg_animation_import import (
    _deduplicate_native_times,
    _normalize_quat,
    _quat_dot,
    compare_action_to_baseline,
    replace_bundle_main_entries,
)
from cmg_probe import (
    clean_name,
    cmg_quaternion_keys,
    parse_cmg_animation_rotation_tracks,
    parse_cmg_animation_translation_tracks,
    parse_cmg_skeleton,
)
from fbx_animation_data import read_action_baseline, read_fbx_actions
from parser_core import PipeworksParser


def decode_cmg_type4(blob: bytes, bones: list[dict], scale: float) -> dict:
    duration = struct.unpack_from(">f", blob, 0x1C)[0]
    bone_count = struct.unpack_from(">I", blob, 0x2C)[0]
    by_index = {int(bone["idx"]): bone for bone in bones}
    tracks = {index: {"bone": index} for index in by_index if index < bone_count}
    for track in parse_cmg_animation_translation_tracks(blob) or []:
        keys = []
        for key_index in range(int(track["key_count"])):
            time, x, y, z = struct.unpack_from(">Hhhh", blob, int(track["records_pos"]) + key_index * 8)
            value = tuple(
                float(raw) / 32767.0 * float(track["scales"][axis]) * scale
                for axis, raw in enumerate((x, y, z))
            )
            keys.append(((time & 0xFFFE) / 65534.0 * duration, value))
        tracks[int(track["bone"])]["translation_keys"] = keys
    for track in parse_cmg_animation_rotation_tracks(blob):
        bone = int(track["bone"])
        tracks[bone]["rotation_keys"] = cmg_quaternion_keys(
            list(track.get("records") or []), duration, tuple(by_index[bone]["q"])
        )
    return {"duration": duration, "bone_count": bone_count, "tracks": tracks}


def _stored_quaternion(value, previous=None, first: bool = False):
    value = _normalize_quat(value)
    if previous is not None and _quat_dot(value, previous) < 0.0:
        value = tuple(-component for component in value)
    if first and value[3] < 0.0:
        value = tuple(-component for component in value)
    xyz = tuple(max(-32767, min(32767, int(round(-component * 32767.0)))) for component in value[:3])
    return xyz, value


def encode_cmg_type4(original: bytes, action: dict, bones: list[dict], scale: float) -> bytes:
    decoded = decode_cmg_type4(original, bones, scale)
    duration = max(1.0 / 120.0, float(action.get("duration") or decoded["duration"]))
    by_index = {int(bone["idx"]): bone for bone in bones}
    action_tracks = action.get("tracks", {})
    translation = bytearray()
    rotation = bytearray()

    for bone in range(int(decoded["bone_count"])):
        native = decoded["tracks"].get(bone, {})
        imported = action_tracks.get(str(by_index[bone]["name"]), {})
        translation_keys = imported.get("translation_keys") or native.get("translation_keys")
        if translation_keys:
            keys = _deduplicate_native_times(translation_keys, duration)
            values = [tuple(float(value) / scale for value in vector) for _time, vector in keys]
            scales = [max(abs(vector[axis]) for vector in values) for axis in range(3)]
            scales = [value if value > 1.0e-12 else 1.0 for value in scales]
            translation.extend(struct.pack(">fffBBBB", *scales, bone, len(keys), len(keys) - 1, 0))
            for (time, _value), vector in zip(keys, values):
                xyz = [
                    max(-32767, min(32767, int(round(vector[axis] / scales[axis] * 32767.0))))
                    for axis in range(3)
                ]
                translation.extend(struct.pack(">Hhhh", time & 0xFFFE, *xyz))

        rotation_keys = imported.get("rotation_keys") or native.get("rotation_keys")
        if not rotation_keys:
            rotation_keys = [(0.0, tuple(by_index[bone]["q"]))]
        keys = _deduplicate_native_times(rotation_keys, duration, quaternion=True)
        rotation.extend(struct.pack(">BBH", bone, len(keys), 0))
        first_xyz, previous = _stored_quaternion(keys[0][1], first=True)
        rotation.extend(struct.pack(">hhh", *first_xyz))
        for time, value in keys[1:]:
            xyz, previous = _stored_quaternion(value, previous)
            rotation.extend(struct.pack(">Hhhh", (time & 0xFFFE) | (1 if previous[3] < 0.0 else 0), *xyz))

    translation_start = struct.unpack_from(">I", original, 0x38)[0]
    payload = bytearray(original[:translation_start])
    payload.extend(translation)
    rotation_start = len(payload)
    payload.extend(rotation)
    struct.pack_into(">f", payload, 0x1C, duration)
    struct.pack_into(">I", payload, 0x24, len(payload))
    struct.pack_into(">I", payload, 0x3C, rotation_start)
    verified = decode_cmg_type4(bytes(payload), bones, scale)
    if not verified["tracks"]:
        raise ValueError("rebuilt CMG Type 4 clip has no decodable tracks")
    return bytes(payload)


def baseline_path_for_fbx(fbx: Path) -> Path:
    return fbx.with_suffix(".animation_action_baseline_v1.json.gz")


def action_without_rest_bone_offsets(
    action: dict,
    skeleton_patch: dict | None,
    scale: float,
) -> dict:
    offsets = {
        str(bone["name"]): tuple(
            (
                float(bone["new_local_translation"][axis])
                - float(bone["old_local_translation"][axis])
            )
            * scale
            for axis in range(3)
        )
        for bone in (skeleton_patch or {}).get("changed_bones") or []
    }
    if not offsets:
        return action
    adjusted = dict(action)
    adjusted_tracks = {}
    for bone_name, track in action.get("tracks", {}).items():
        adjusted_track = dict(track)
        offset = offsets.get(str(bone_name))
        keys = track.get("translation_keys") or []
        if offset is not None and keys:
            adjusted_track["translation_keys"] = [
                (
                    time,
                    tuple(float(value[axis]) - offset[axis] for axis in range(3)),
                )
                for time, value in keys
            ]
        adjusted_tracks[bone_name] = adjusted_track
    adjusted["tracks"] = adjusted_tracks
    return adjusted


def import_cmg_actions(
    data: bytes,
    entries: list[dict],
    bones: list[dict],
    fbx: Path,
    scale: float,
    skeleton_patch: dict | None = None,
) -> tuple[bytes, list[dict]]:
    baseline_path = baseline_path_for_fbx(fbx)
    if not baseline_path.exists():
        return data, [{"status": "skipped_missing_action_baseline"}]
    candidates = []
    for entry in entries:
        if entry["file_type"] != 4 or entry["is_resource"] or int(entry["size"]) < 0x44:
            continue
        blob = data[int(entry["offset"]) : int(entry["offset"]) + int(entry["size"])]
        if struct.unpack_from(">I", blob, 0x20)[0] == 2 and struct.unpack_from(">I", blob, 0x3C)[0] <= len(blob):
            candidates.append(entry)
    names = [clean_name(str(entry["name"])) for entry in candidates]
    actions = read_fbx_actions(fbx, names)
    baseline = read_action_baseline(baseline_path)
    motion_helper_names = {
        str(bone["name"])
        for bone in bones
        if int(bone.get("parent", -1)) < 0
        or "liftnode" in str(bone["name"]).lower().replace("_", "")
    }
    replacements = {}
    results = []
    for entry, name in zip(candidates, names):
        action = actions.get(name)
        if action is None or name not in baseline:
            results.append({"clip": name, "status": "preserved_missing_action_or_baseline"})
            continue
        comparison_action = action_without_rest_bone_offsets(action, skeleton_patch, scale)
        comparison = compare_action_to_baseline(
            comparison_action,
            baseline[name],
            translation_tolerance=(
                0.30 if (skeleton_patch or {}).get("changed_bones") else 0.25
            ),
        )
        changed_names = set(comparison.get("changed_bones") or [])
        changed_channels = comparison.get("changed_channels") or {}
        helper_translation_only = (
            bool((skeleton_patch or {}).get("changed_bones"))
            and bool(changed_names)
            and changed_names.issubset(motion_helper_names)
            and all(channels == ["translation"] for channels in changed_channels.values())
        )
        if helper_translation_only:
            relaxed = compare_action_to_baseline(
                comparison_action,
                baseline[name],
                translation_tolerance=3.0,
            )
            if not relaxed["changed"]:
                relaxed["comparison"] = "skeleton_edit_motion_helper_tolerance"
                comparison = relaxed
        if not comparison["changed"]:
            results.append({"clip": name, "status": "preserved_byte_for_byte", **comparison})
            continue
        original = data[int(entry["offset"]) : int(entry["offset"]) + int(entry["size"])]
        rebuilt = encode_cmg_type4(original, action, bones, scale)
        replacements[int(entry["file_num"])] = rebuilt
        results.append(
            {
                "clip": name,
                "status": "rebuilt_from_fbx_action",
                "size_before": len(original),
                "size_after": len(rebuilt),
                **comparison,
            }
        )
    return replace_bundle_main_entries(data, replacements, alignment=0x20), results


def main() -> int:
    parser = argparse.ArgumentParser(description="Import edited FBX Actions into a GameCube CMG.")
    parser.add_argument("cmg")
    parser.add_argument("fbx")
    parser.add_argument("output")
    parser.add_argument("--scale", type=float, default=10.0)
    args = parser.parse_args()
    source = Path(args.cmg)
    pipeworks = PipeworksParser(str(source))
    entries = pipeworks.parse()
    bones, _globals = parse_cmg_skeleton(pipeworks, entries, args.scale)
    output, results = import_cmg_actions(pipeworks.file_data or b"", entries, bones, Path(args.fbx), args.scale)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(output)
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
