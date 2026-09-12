from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path

from bdg_animations import parse_rotation_tracks, parse_translation_tracks, quaternion_keys
from bdg_to_fbx_extract_all import (
    quat_to_euler_xyz_degrees,
    sample_quat_keys,
    sample_vector_keys,
    unwrap_eulers,
)
from fbx_animation_data import read_action_baseline, read_fbx_actions
from fbx_to_bdg_import import euler_xyz_degrees_to_quat


def _align(value: int, alignment: int = 0x10) -> int:
    return (int(value) + alignment - 1) & ~(alignment - 1)


def _normalize_quat(value):
    length = math.sqrt(sum(float(component) ** 2 for component in value))
    if length <= 1.0e-12:
        return (0.0, 0.0, 0.0, 1.0)
    return tuple(float(component) / length for component in value)


def _quat_dot(left, right) -> float:
    return sum(float(a) * float(b) for a, b in zip(left, right))


def _quat_angle_degrees(left, right) -> float:
    dot = min(1.0, max(-1.0, abs(_quat_dot(_normalize_quat(left), _normalize_quat(right)))))
    return math.degrees(2.0 * math.acos(dot))


def _quat_slerp(left, right, alpha: float):
    left = _normalize_quat(left)
    right = _normalize_quat(right)
    dot = _quat_dot(left, right)
    if dot < 0.0:
        right = tuple(-value for value in right)
        dot = -dot
    if dot > 0.9995:
        return _normalize_quat(tuple(a + (b - a) * alpha for a, b in zip(left, right)))
    angle = math.acos(min(1.0, max(-1.0, dot)))
    scale = math.sin(angle)
    return tuple(
        (math.sin((1.0 - alpha) * angle) * a + math.sin(alpha * angle) * b) / scale
        for a, b in zip(left, right)
    )


def _sample(keys, time: float, quaternion: bool = False):
    if not keys:
        return None
    if time <= keys[0][0]:
        return keys[0][1]
    if time >= keys[-1][0]:
        return keys[-1][1]
    for index in range(1, len(keys)):
        if time <= keys[index][0]:
            left_time, left_value = keys[index - 1]
            right_time, right_value = keys[index]
            span = right_time - left_time
            alpha = 0.0 if span <= 1.0e-12 else (time - left_time) / span
            if quaternion:
                return _quat_slerp(left_value, right_value, alpha)
            return tuple(a + (b - a) * alpha for a, b in zip(left_value, right_value))
    return keys[-1][1]


def _native_time(seconds: float, duration: float) -> int:
    if duration <= 1.0e-12:
        return 0
    return max(0, min(65534, int(round(float(seconds) / duration * 65534.0)))) & 0xFFFE


def _deduplicate_native_times(keys, duration: float, quaternion: bool = False):
    by_time = {}
    for seconds, value in sorted(keys, key=lambda item: float(item[0])):
        by_time[_native_time(seconds, duration)] = _normalize_quat(value) if quaternion else tuple(value)
    ordered = [(time, value) for time, value in sorted(by_time.items())]
    if len(ordered) <= 255:
        return ordered
    source = [(time / 65534.0 * duration, value) for time, value in ordered]
    result = []
    for index in range(255):
        native = (round(index * 65534 / 254.0) & 0xFFFE) if index else 0
        seconds = native / 65534.0 * duration
        result.append((native, _sample(source, seconds, quaternion=quaternion)))
    return result


def _skeleton_from_manifest(manifest: dict) -> dict[int, dict]:
    result = {}
    for raw in manifest.get("bones", []):
        index = int(raw["idx"])
        result[index] = {
            "name": str(raw["name"]),
            "t": tuple(float(value) for value in raw.get("local_translation", (0, 0, 0))),
            "q": _normalize_quat(
                raw.get("native_local_quaternion_xyzw", raw.get("local_quaternion_xyzw", (0, 0, 0, 1)))
            ),
        }
    return result


def decode_bdg_type4(blob: bytes, skeleton: dict[int, dict], export_scale: float) -> dict:
    duration = struct.unpack_from(">f", blob, 0x1C)[0]
    bone_count = struct.unpack_from(">I", blob, 0x2C)[0]
    tracks = {index: {"bone": index} for index in range(min(bone_count, len(skeleton)))}
    for track in parse_translation_tracks(blob, bone_count):
        keys = []
        scales = track["scales"]
        for index in range(int(track["key_count"])):
            time, x, y, z = struct.unpack_from(">Hhhh", blob, int(track["records_pos"]) + index * 8)
            value = tuple(
                float(raw) / 32767.0 * float(scales[axis]) * export_scale
                for axis, raw in enumerate((x, y, z))
            )
            keys.append(((time & 0xFFFE) / 65534.0 * duration, value))
        tracks[int(track["bone"])]["translation_keys"] = keys
    for track in parse_rotation_tracks(blob, bone_count):
        bone = int(track["bone"])
        tracks[bone]["rotation_keys"] = quaternion_keys(
            track["records"], duration, skeleton[bone]["q"], str(track["layout"])
        )
    return {"duration": duration, "bone_count": bone_count, "tracks": tracks}


def _comparison_times(left, right, duration: float) -> list[float]:
    times = {0.0, max(0.0, float(duration))}
    times.update(float(time) for time, _value in left or [])
    times.update(float(time) for time, _value in right or [])
    ordered = sorted(time for time in times if -1.0e-7 <= time <= duration + 1.0e-7)
    times.update((a + b) * 0.5 for a, b in zip(ordered, ordered[1:]) if b - a > 1.0e-6)
    return sorted(times)


def _usable_translation(value) -> bool:
    return bool(value) and all(
        math.isfinite(float(component)) and abs(float(component)) <= 1.0e7
        for component in value
    )


def compare_action_to_native(
    action: dict,
    native: dict,
    skeleton: dict[int, dict],
    export_scale: float,
    translation_tolerance: float = 0.1,
    rotation_tolerance_degrees: float = 3.5,
) -> dict:
    duration = float(native["duration"])
    maximum_translation_error = 0.0
    maximum_rotation_error = 0.0
    changed_bones = set()
    changed_channels: dict[str, set[str]] = {}
    duration_changed = False
    action_tracks = action.get("tracks", {})
    if abs(float(action.get("duration", duration)) - duration) > 1.0 / 120.0:
        changed_bones.add("<duration>")
        duration_changed = True
    for bone, rest in skeleton.items():
        imported = action_tracks.get(rest["name"], {})
        original = native["tracks"].get(bone, {})
        imported_translation = imported.get("translation_keys") or []
        source_translation = original.get("translation_keys") or []
        if source_translation:
            sampled_times, sampled_values = sample_vector_keys(
                [item[0] for item in source_translation],
                [item[1] for item in source_translation],
                duration,
                fps=60.0,
            )
            expected_translation = list(zip(sampled_times, sampled_values))
            check_times = sorted({float(time) for time, _value in imported_translation}) or [0.0]
            for time in check_times:
                actual = _sample(imported_translation, time) if imported_translation else expected_translation[0][1]
                expected = _sample(expected_translation, time)
                error = math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(actual, expected)))
                maximum_translation_error = max(maximum_translation_error, error)
                if error > translation_tolerance:
                    changed_bones.add(rest["name"])
                    changed_channels.setdefault(rest["name"], set()).add("translation")
                    break

        imported_rotation = imported.get("rotation_keys") or []
        source_rotation = original.get("rotation_keys") or []
        if source_rotation:
            sampled_times, sampled_quats = sample_quat_keys(
                [item[0] for item in source_rotation],
                [item[1] for item in source_rotation],
                duration,
                fps=60.0,
            )
            sampled_eulers = unwrap_eulers([quat_to_euler_xyz_degrees(value) for value in sampled_quats])
            expected_eulers = list(zip(sampled_times, sampled_eulers))
            check_times = sorted({float(time) for time, _value in imported_rotation}) or [0.0]
            for time in check_times:
                actual = _sample(imported_rotation, time, quaternion=True) if imported_rotation else source_rotation[0][1]
                expected_euler = _sample(expected_eulers, time)
                expected = euler_xyz_degrees_to_quat(*expected_euler)
                error = _quat_angle_degrees(actual, expected)
                maximum_rotation_error = max(maximum_rotation_error, error)
                if error > rotation_tolerance_degrees:
                    changed_bones.add(rest["name"])
                    changed_channels.setdefault(rest["name"], set()).add("rotation")
                    break
    return {
        "changed": bool(changed_bones),
        "changed_bones": sorted(changed_bones),
        "changed_channels": {
            bone_name: sorted(channels)
            for bone_name, channels in sorted(changed_channels.items())
        },
        "duration_changed": duration_changed,
        "maximum_translation_error": maximum_translation_error,
        "maximum_rotation_error_degrees": maximum_rotation_error,
    }


def compare_action_to_baseline(
    action: dict,
    baseline: dict,
    translation_tolerance: float = 0.25,
    rotation_tolerance_degrees: float = 10.0,
) -> dict:
    changed_bones = set()
    changed_channels: dict[str, set[str]] = {}
    maximum_translation_error = 0.0
    maximum_rotation_error = 0.0
    duration = float(baseline.get("duration", 0.0))
    duration_changed = abs(float(action.get("duration", duration)) - duration) > 1.0 / 120.0
    if duration_changed:
        changed_bones.add("<duration>")
    imported_tracks = action.get("tracks", {})
    baseline_tracks = baseline.get("tracks", {})
    source_channels = baseline.get("source_channels")
    if source_channels is None:
        source_channels = {
            bone_name: [
                channel
                for channel, key in (("t", "translation_keys"), ("r", "rotation_euler_keys"))
                if track.get(key)
            ]
            for bone_name, track in baseline_tracks.items()
        }
    for bone_name in sorted(set(baseline_tracks) | set(imported_tracks)):
        expected = baseline_tracks.get(bone_name, {})
        imported = imported_tracks.get(bone_name, {})
        expected_translation = expected.get("translation_keys") or []
        imported_translation = imported.get("translation_keys") or []
        if "t" in source_channels.get(bone_name, []):
            imported_translation = imported.get("translation_keys") or []
            if not imported_translation:
                changed_bones.add(bone_name)
                changed_channels.setdefault(bone_name, set()).add("translation")
            for time, reference in expected_translation:
                actual = _sample(imported_translation, float(time)) if imported_translation else reference
                # Some native clips contain sentinel/garbage translation scales that FBX
                # can only preserve approximately. They are not editable transforms.
                if not _usable_translation(reference) or not _usable_translation(actual):
                    continue
                error = math.dist(tuple(actual), tuple(reference))
                maximum_translation_error = max(maximum_translation_error, error)
                if error > translation_tolerance:
                    changed_bones.add(bone_name)
                    changed_channels.setdefault(bone_name, set()).add("translation")
                    break
        elif len(imported_translation) > 1:
            reference = tuple(imported_translation[0][1])
            if any(math.dist(tuple(value), reference) > translation_tolerance for _time, value in imported_translation[1:]):
                changed_bones.add(bone_name)
                changed_channels.setdefault(bone_name, set()).add("translation")
        expected_eulers = expected.get("rotation_euler_keys") or []
        imported_eulers = imported.get("rotation_euler_keys") or []
        if "r" in source_channels.get(bone_name, []):
            if not imported_eulers:
                changed_bones.add(bone_name)
                changed_channels.setdefault(bone_name, set()).add("rotation")
            for time, expected_euler in expected_eulers:
                actual_euler = _sample(imported_eulers, float(time)) if imported_eulers else None
                actual = euler_xyz_degrees_to_quat(*actual_euler) if actual_euler is not None else None
                reference = euler_xyz_degrees_to_quat(*expected_euler)
                if actual is None:
                    continue
                error = _quat_angle_degrees(actual, reference)
                maximum_rotation_error = max(maximum_rotation_error, error)
                if error > rotation_tolerance_degrees:
                    changed_bones.add(bone_name)
                    changed_channels.setdefault(bone_name, set()).add("rotation")
                    break
        elif len(imported_eulers) > 1:
            reference = euler_xyz_degrees_to_quat(*imported_eulers[0][1])
            if any(
                _quat_angle_degrees(euler_xyz_degrees_to_quat(*value), reference)
                > rotation_tolerance_degrees
                for _time, value in imported_eulers[1:]
            ):
                changed_bones.add(bone_name)
                changed_channels.setdefault(bone_name, set()).add("rotation")
    comparison = {
        "changed": bool(changed_bones),
        "changed_bones": sorted(changed_bones),
        "changed_channels": {
            bone_name: sorted(channels)
            for bone_name, channels in sorted(changed_channels.items())
        },
        "duration_changed": duration_changed,
        "maximum_translation_error": maximum_translation_error,
        "maximum_rotation_error_degrees": maximum_rotation_error,
        "comparison": "exported_action_baseline",
    }
    raw_action = baseline.get("raw_action")
    if comparison["changed"] and raw_action:
        raw_baseline = dict(raw_action)
        raw_baseline["source_channels"] = source_channels
        raw_comparison = compare_action_to_baseline(
            action,
            raw_baseline,
            translation_tolerance=translation_tolerance,
            rotation_tolerance_degrees=rotation_tolerance_degrees,
        )
        if not raw_comparison["changed"]:
            raw_comparison["comparison"] = "raw_export_action_baseline"
            return raw_comparison
    return comparison


def _pack_quaternion(value, previous=None):
    value = _normalize_quat(value)
    if previous is not None and _quat_dot(value, previous) < 0.0:
        value = tuple(-component for component in value)
    if previous is None and value[3] < 0.0:
        value = tuple(-component for component in value)
    xyz = [max(-32767, min(32767, int(round(component * 32767.0)))) for component in value[:3]]
    return tuple(xyz), value


def _native_bdg_rotation_entries(blob: bytes, bone_count: int) -> dict[int, bytes] | None:
    section_start = struct.unpack_from(">I", blob, 0x3C)[0]
    section_end = struct.unpack_from(">I", blob, 0x38)[0]
    if not (0x40 <= section_start < section_end <= len(blob)):
        return None
    cursor = section_start
    previous_bone = -1
    entries = {}
    while cursor + 2 <= section_end and any(blob[cursor:section_end]):
        entry_start = cursor
        bone = int(blob[cursor])
        key_count = int(blob[cursor + 1])
        entry_end = cursor + 2 + key_count * 8
        if bone <= previous_bone or bone >= bone_count or entry_end > section_end:
            return None
        if key_count and struct.unpack_from(">H", blob, cursor + 2)[0] != 0:
            return None
        entries[bone] = bytes(blob[entry_start:entry_end])
        previous_bone = bone
        cursor = entry_end
    if any(blob[cursor:section_end]):
        return None
    return entries


def _native_bdg_translation_entries(blob: bytes, bone_count: int) -> dict[int, bytes] | None:
    section_start = struct.unpack_from(">I", blob, 0x38)[0]
    section_end = struct.unpack_from(">I", blob, 0x24)[0]
    if not (0x40 <= section_start <= section_end <= len(blob)):
        return None
    tracks = sorted(parse_translation_tracks(blob, bone_count), key=lambda item: int(item["rel"]))
    cursor = section_start
    entries = {}
    for track in tracks:
        start = int(track["rel"])
        end = min(_align(int(track["end"])), section_end)
        if start != cursor or end < start:
            return None
        entries[int(track["bone"])] = bytes(blob[start:end])
        cursor = end
    if any(blob[cursor:section_end]):
        return None
    return entries


def _encode_bdg_rotation_entry(bone: int, keys, duration: float) -> bytes:
    native_keys = _deduplicate_native_times(keys or [], duration, quaternion=True)
    if not native_keys:
        return bytes((bone, 0))
    entry = bytearray((bone, len(native_keys)))
    first_xyz, previous = _pack_quaternion(native_keys[0][1])
    entry.extend(struct.pack(">Hhhh", 0, *first_xyz))
    for native_time, value in native_keys[1:]:
        xyz, previous = _pack_quaternion(value, previous)
        sign = 1 if previous[3] < 0.0 else 0
        entry.extend(struct.pack(">Hhhh", (native_time & 0xFFFE) | sign, *xyz))
    return bytes(entry)


def _encode_bdg_translation_entry(
    bone: int,
    keys,
    duration: float,
    export_scale: float,
) -> bytes:
    native_keys = _deduplicate_native_times(keys or [], duration)
    if not native_keys:
        return b""
    native_values = [
        tuple(float(value) / export_scale for value in vector)
        for _time, vector in native_keys
    ]
    scales = [max(abs(vector[axis]) for vector in native_values) for axis in range(3)]
    scales = [value if value > 1.0e-12 else 1.0 for value in scales]
    entry = bytearray(struct.pack(">fffBBBB", *scales, bone, len(native_keys), len(native_keys) - 1, 0))
    for (native_time, _value), vector in zip(native_keys, native_values):
        xyz = [
            max(-32767, min(32767, int(round(vector[axis] / scales[axis] * 32767.0))))
            for axis in range(3)
        ]
        entry.extend(struct.pack(">Hhhh", native_time & 0xFFFE, *xyz))
    entry.extend(b"\x00" * (_align(len(entry)) - len(entry)))
    return bytes(entry)


def reverse_bdg_type4_native(
    original: bytes,
    skeleton: dict[int, dict],
    export_scale: float,
) -> bytes:
    """Reverse a Wii Type 4 clip without passing unchanged data through FBX."""
    decoded = decode_bdg_type4(original, skeleton, export_scale)
    duration = float(decoded["duration"])
    bone_count = int(decoded["bone_count"])
    rotation_entries = _native_bdg_rotation_entries(original, bone_count)
    if rotation_entries is None:
        raise ValueError("native reversal requires the sequential Wii rotation layout")

    payload = bytearray(original)
    rotation_start = struct.unpack_from(">I", original, 0x3C)[0]
    translation_start = struct.unpack_from(">I", original, 0x38)[0]
    cursor = rotation_start
    while cursor + 2 <= translation_start and any(original[cursor:translation_start]):
        bone = int(original[cursor])
        key_count = int(original[cursor + 1])
        entry_end = cursor + 2 + key_count * 8
        original_keys = decoded["tracks"].get(bone, {}).get("rotation_keys") or []
        if key_count:
            if len(original_keys) != key_count:
                raise ValueError(
                    f"rotation key count mismatch for bone {bone}: {len(original_keys)} != {key_count}"
                )
            reversed_keys = [
                (duration - float(time), value)
                for time, value in reversed(original_keys)
            ]
            rebuilt = _encode_bdg_rotation_entry(bone, reversed_keys, duration)
            if len(rebuilt) != entry_end - cursor:
                raise ValueError(f"reversed rotation entry size changed for bone {bone}")
            payload[cursor:entry_end] = rebuilt
        cursor = entry_end

    for track in parse_translation_tracks(original, bone_count):
        records_pos = int(track["records_pos"])
        key_count = int(track["key_count"])
        records = [
            original[records_pos + index * 8 : records_pos + (index + 1) * 8]
            for index in range(key_count)
        ]
        reversed_records = []
        for record in reversed(records):
            old_time = struct.unpack_from(">H", record, 0)[0] & 0xFFFE
            new_time = 65534 - old_time
            reversed_records.append(struct.pack(">H", new_time) + record[2:])
        payload[records_pos : records_pos + key_count * 8] = b"".join(reversed_records)

    if len(payload) != len(original):
        raise ValueError("native reversal changed the Type 4 resource size")
    verified = decode_bdg_type4(bytes(payload), skeleton, export_scale)
    if verified["bone_count"] != bone_count or verified["duration"] != duration:
        raise ValueError("native reversal changed the Type 4 header")
    return bytes(payload)


def encode_bdg_type4(
    original: bytes,
    action: dict,
    skeleton: dict[int, dict],
    export_scale: float,
    changed_channels: dict[str, list[str]] | None = None,
    duration_changed: bool = True,
) -> bytes:
    original_decoded = decode_bdg_type4(original, skeleton, export_scale)
    duration = (
        max(1.0 / 120.0, float(action.get("duration") or original_decoded["duration"]))
        if duration_changed
        else float(original_decoded["duration"])
    )
    bone_count = int(original_decoded["bone_count"])
    action_tracks = action.get("tracks", {})
    changed_channels = (
        {str(name): set(channels) for name, channels in changed_channels.items()}
        if changed_channels is not None
        else None
    )
    native_rotation_entries = _native_bdg_rotation_entries(original, bone_count)
    native_translation_entries = _native_bdg_translation_entries(original, bone_count)
    preserve_native = (
        changed_channels is not None
        and native_rotation_entries is not None
        and native_translation_entries is not None
    )

    rotation = bytearray()
    translation = bytearray()
    for bone in range(bone_count):
        rest = skeleton[bone]
        imported = action_tracks.get(rest["name"], {})
        original_track = original_decoded["tracks"].get(bone, {})
        channels = changed_channels.get(rest["name"], set()) if changed_channels is not None else None

        if preserve_native and "rotation" not in channels:
            rotation.extend(native_rotation_entries.get(bone, bytes((bone, 0))))
        else:
            rotation_keys = (
                imported.get("rotation_keys")
                if changed_channels is not None and "rotation" in channels
                else imported.get("rotation_keys") or original_track.get("rotation_keys")
            )
            if rotation_keys and not original_track.get("rotation_keys") and all(
                _quat_angle_degrees(value, rest["q"]) <= 0.05 for _time, value in rotation_keys
            ):
                rotation_keys = []
            rotation.extend(_encode_bdg_rotation_entry(bone, rotation_keys, duration))

        if preserve_native and "translation" not in channels:
            translation.extend(native_translation_entries.get(bone, b""))
        else:
            translation_keys = (
                imported.get("translation_keys")
                if changed_channels is not None and "translation" in channels
                else imported.get("translation_keys") or original_track.get("translation_keys")
            )
            if translation_keys and not original_track.get("translation_keys"):
                rest_value = tuple(value * export_scale for value in rest["t"])
                if all(math.dist(value, rest_value) <= 0.001 for _time, value in translation_keys):
                    translation_keys = []
            translation.extend(
                _encode_bdg_translation_entry(bone, translation_keys, duration, export_scale)
            )

    rotation_start = 0x58
    rotation_end = _align(rotation_start + len(rotation))
    payload = bytearray(original[:rotation_start])
    payload.extend(rotation)
    payload.extend(b"\x00" * (rotation_end - len(payload)))
    translation_start = len(payload)
    payload.extend(translation)

    struct.pack_into(">f", payload, 0x1C, duration)
    struct.pack_into(">I", payload, 0x24, len(payload))
    struct.pack_into(">I", payload, 0x38, translation_start)
    struct.pack_into(">I", payload, 0x3C, rotation_start)
    decoded = decode_bdg_type4(bytes(payload), skeleton, export_scale)
    if decoded["bone_count"] != bone_count:
        raise ValueError("rebuilt Type 4 bone count changed")
    return bytes(payload)


def replace_bundle_main_entries(
    bundle: bytes,
    replacements: dict[int, bytes],
    alignment: int = 0x10,
) -> bytes:
    out = bytearray(bundle)
    endian = ">" if struct.unpack_from("<H", out, 0x2C)[0] == 0 else "<"
    file_count = struct.unpack_from(endian + "H", out, 0x62)[0]
    main_start = struct.unpack_from(endian + "I", out, 0x68)[0]
    for file_num, replacement in replacements.items():
        row = next(
            (
                0x78 + index * 0x12
                for index in range(file_count)
                if struct.unpack_from(endian + "H", out, 0x78 + index * 0x12)[0] == file_num
            ),
            None,
        )
        if row is None:
            raise ValueError(f"bundle has no TOC row for file {file_num}")
        old_relative = struct.unpack_from(endian + "I", out, row + 2)[0]
        old_size = struct.unpack_from(endian + "I", out, row + 6)[0]
        old_absolute = main_start + old_relative
        old_slot_size = _align(old_size, alignment)
        new_slot_size = _align(len(replacement), alignment)
        next_relative = min(
            (
                struct.unpack_from(endian + "I", out, 0x78 + index * 0x12 + 2)[0]
                for index in range(file_count)
                if struct.unpack_from(endian + "I", out, 0x78 + index * 0x12 + 6)[0]
                and struct.unpack_from(endian + "I", out, 0x78 + index * 0x12 + 2)[0] > old_relative
            ),
            default=None,
        )
        if next_relative is not None:
            old_slot_size = min(old_slot_size, next_relative - old_relative)
        padded = replacement + b"\x00" * (new_slot_size - len(replacement))
        out[old_absolute : old_absolute + old_slot_size] = padded
        delta = new_slot_size - old_slot_size
        struct.pack_into(endian + "I", out, row + 6, len(replacement))
        if delta:
            for index in range(file_count):
                other = 0x78 + index * 0x12
                relative = struct.unpack_from(endian + "I", out, other + 2)[0]
                if other != row and relative > old_relative:
                    struct.pack_into(endian + "I", out, other + 2, relative + delta)
            resource_start = struct.unpack_from(endian + "I", out, 0x70)[0]
            struct.pack_into(endian + "I", out, 0x70, resource_start + delta)
        main_size = max(
            struct.unpack_from(endian + "I", out, 0x78 + index * 0x12 + 2)[0]
            + struct.unpack_from(endian + "I", out, 0x78 + index * 0x12 + 6)[0]
            for index in range(file_count)
        )
        struct.pack_into(endian + "I", out, 0x6C, main_size)
    return bytes(out)


def replace_bdg_main_entries(bundle: bytes, replacements: dict[int, bytes]) -> bytes:
    return replace_bundle_main_entries(bundle, replacements, alignment=0x10)


def import_bdg_actions(
    bundle: bytes,
    fbx_path: Path,
    manifest: dict,
    report: dict | None = None,
) -> tuple[bytes, list[dict]]:
    skeleton = _skeleton_from_manifest(manifest)
    export_scale = float(manifest.get("fbx_export_scale") or 1.0)
    locations = manifest.get("animation_resource_locations", [])
    names = [str(item["name"]) for item in locations]
    actions = read_fbx_actions(Path(fbx_path), names)
    baseline = {}
    baseline_name = manifest.get("animation_action_baseline")
    if baseline_name:
        baseline_path = Path(fbx_path).parent / str(baseline_name)
        if baseline_path.exists():
            baseline = read_action_baseline(baseline_path)

    endian = ">" if struct.unpack_from("<H", bundle, 0x2C)[0] == 0 else "<"
    file_count = struct.unpack_from(endian + "H", bundle, 0x62)[0]
    main_start = struct.unpack_from(endian + "I", bundle, 0x68)[0]
    rows = {
        struct.unpack_from(endian + "H", bundle, 0x78 + index * 0x12)[0]: 0x78 + index * 0x12
        for index in range(file_count)
    }
    replacements = {}
    results = []
    for location in locations:
        file_num = int(location["resource_id"])
        name = str(location["name"])
        action = actions.get(name)
        if action is None:
            results.append({"clip": name, "status": "preserved_missing_action"})
            continue
        row = rows.get(file_num)
        if row is None:
            results.append({"clip": name, "status": "preserved_missing_bundle_entry"})
            continue
        relative = struct.unpack_from(endian + "I", bundle, row + 2)[0]
        size = struct.unpack_from(endian + "I", bundle, row + 6)[0]
        original = bytes(bundle[main_start + relative : main_start + relative + size])
        native = decode_bdg_type4(original, skeleton, export_scale)
        if name in baseline:
            comparison = compare_action_to_baseline(action, baseline[name])
        else:
            comparison = compare_action_to_native(action, native, skeleton, export_scale)
        if not comparison["changed"]:
            results.append({"clip": name, "status": "preserved_byte_for_byte", **comparison})
            continue
        rebuilt = encode_bdg_type4(
            original,
            action,
            skeleton,
            export_scale,
            changed_channels=comparison.get("changed_channels"),
            duration_changed=bool(comparison.get("duration_changed")),
        )
        replacements[file_num] = rebuilt
        results.append(
            {
                "clip": name,
                "status": "rebuilt_from_fbx_action",
                "size_before": len(original),
                "size_after": len(rebuilt),
                **comparison,
            }
        )
    output = replace_bdg_main_entries(bundle, replacements)
    if report is not None:
        report.setdefault("animation_action_import", []).extend(results)
    return output, results


def main() -> int:
    parser = argparse.ArgumentParser(description="Import edited FBX Actions into a Wii character BDG.")
    parser.add_argument("bundle")
    parser.add_argument("fbx")
    parser.add_argument("manifest")
    parser.add_argument("output")
    args = parser.parse_args()
    bundle_path = Path(args.bundle)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    output, results = import_bdg_actions(bundle_path.read_bytes(), Path(args.fbx), manifest)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(output)
    print(json.dumps(results, indent=2))
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
