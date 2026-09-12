from __future__ import annotations

import hashlib
import math
import re
import struct
from pathlib import Path

from parser_core import PipeworksParser


def be32(data: bytes, offset: int) -> int:
    return struct.unpack_from(">I", data, offset)[0]


def bef(data: bytes, offset: int) -> float:
    return struct.unpack_from(">f", data, offset)[0]


def normalize_quaternion(values) -> tuple[float, float, float, float]:
    length = math.sqrt(sum(float(value) * float(value) for value in values)) or 1.0
    return tuple(float(value) / length for value in values)


def quaternion_dot(left, right) -> float:
    return sum(float(left[index]) * float(right[index]) for index in range(4))


def track_times_valid(times: list[int]) -> bool:
    # Native Type-4 keys store the omitted quaternion W sign in bit 0. Time is
    # the remaining even value in the normalized 0..65534 range.
    times = [int(time) & 0xFFFE for time in times]
    if len(times) == 1:
        return 0 <= times[0] <= 65535
    if len(times) == 2:
        first, second = times
        return second > first or (first >= 55000 and second <= 30000)
    if len(set(times)) < 2:
        return False
    drops = [
        index
        for index, (first, second) in enumerate(zip(times, times[1:]))
        if second <= first
    ]
    if not drops:
        return all(second > first for first, second in zip(times, times[1:]))
    return (
        len(drops) == 1
        and drops[0] == len(times) - 2
        and all(second > first for first, second in zip(times[:-2], times[1:-1]))
    )


def animation_seconds(time: int, duration: float) -> float:
    timestamp = int(time) & 0xFFFE
    return max(0.0, min(float(duration), timestamp / 65534.0 * float(duration)))


def ordered_keys(keys: list[tuple[float, object]]) -> list[tuple[float, object]]:
    result = []
    for time, value in sorted(keys, key=lambda item: float(item[0])):
        if result and float(time) <= result[-1][0] + 1e-8:
            continue
        result.append((float(time), value))
    return result


def quaternion_xyz_valid(x: int, y: int, z: int) -> bool:
    return sum((float(value) / 32767.0) ** 2 for value in (x, y, z)) <= 1.05


def decode_rotation_track(
    blob: bytes,
    offset: int,
    section_end: int,
    bone_count: int,
) -> dict | None:
    if offset < 0 or offset + 12 > section_end:
        return None
    bone = int(blob[offset])
    stored_count = int(blob[offset + 1]) | (int(blob[offset + 2]) << 8)
    flags = int(blob[offset + 3])
    records_end = offset + 4 + stored_count * 8
    container_end = offset + 4 + stored_count * 8
    if (
        bone >= bone_count
        or stored_count < 1
        or stored_count > 512
        or flags != 0
        or container_end > section_end
    ):
        return None

    records = []
    for index in range(stored_count):
        record_offset = offset + 4 + index * 8
        x, y, z, time = struct.unpack_from(">hhhH", blob, record_offset)
        if not quaternion_xyz_valid(x, y, z):
            return None
        records.append((int(time), int(x), int(y), int(z)))
    # Explicit tracks pack q0, time1+q1, ..., timeN+qN. The word after
    # the final quaternion is continuation metadata, not another timestamp.
    semantic_times = [0] + [record[0] for record in records[:-1]]
    if not track_times_valid(semantic_times):
        return None
    native_records = records
    return {
        "bone": bone,
        "rel": offset,
        "end": records_end,
        "container_end": container_end,
        "record_count": stored_count,
        "records": native_records,
        "native_record_count": stored_count,
        "native_records": native_records,
        "layout": "explicit_qxyz_time",
    }


def decode_continuation_rotation_track(
    blob: bytes,
    start: int,
    end: int,
    bone: int,
    bone_count: int,
) -> dict | None:
    if not (0 <= bone < bone_count) or start >= end or (end - start) % 8:
        return None
    record_count = (end - start) // 8
    if record_count < 1 or record_count > 512:
        return None
    records = []
    for offset in range(start, end, 8):
        time, x, y, z = struct.unpack_from(">Hhhh", blob, offset)
        if not quaternion_xyz_valid(x, y, z):
            return None
        records.append((int(time), int(x), int(y), int(z)))
    if not track_times_valid([record[0] for record in records]):
        return None
    native_records = records
    return {
        "bone": bone,
        "rel": start,
        "end": end,
        "record_count": record_count,
        "records": native_records,
        "native_record_count": record_count,
        "native_records": native_records,
        "layout": "continuation_time_qxyz",
    }


def decode_native_rotation_stream(
    blob: bytes,
    section_start: int,
    section_end: int,
    bone_count: int,
) -> list[dict] | None:
    """Decode the native sequential Wii Type-4 rotation stream.

    Each bone entry starts with ``bone, key_count``. A zero count is a compact
    two-byte entry. Non-empty tracks add a zero u16, an un-timed qxyz key at
    frame zero, then ``time, qxyz`` records for the remaining keys.
    """
    cursor = section_start
    previous_bone = -1
    tracks = []
    while cursor + 2 <= section_end:
        if not any(blob[cursor:section_end]):
            cursor = section_end
            break
        rel = cursor
        bone = int(blob[cursor])
        record_count = int(blob[cursor + 1])
        cursor += 2
        if bone <= previous_bone or bone >= bone_count:
            return None
        previous_bone = bone
        if record_count == 0:
            continue
        if cursor + 8 > section_end or struct.unpack_from(">H", blob, cursor)[0] != 0:
            return None
        cursor += 2
        raw_x, raw_y, raw_z = struct.unpack_from(">hhh", blob, cursor)
        cursor += 6
        if not quaternion_xyz_valid(raw_x, raw_y, raw_z):
            return None
        records = [(0, int(raw_x), int(raw_y), int(raw_z))]
        for _index in range(1, record_count):
            if cursor + 8 > section_end:
                return None
            time, raw_x, raw_y, raw_z = struct.unpack_from(">Hhhh", blob, cursor)
            cursor += 8
            if not quaternion_xyz_valid(raw_x, raw_y, raw_z):
                return None
            records.append((int(time), int(raw_x), int(raw_y), int(raw_z)))
        if not track_times_valid([record[0] for record in records]):
            return None
        tracks.append(
            {
                "bone": bone,
                "rel": rel,
                "end": cursor,
                "container_end": cursor,
                "record_count": record_count,
                "records": records,
                "native_record_count": record_count,
                "native_records": records,
                "layout": "first_qxyz_then_time_qxyz",
            }
        )
    return tracks if cursor == section_end else None


def find_terminal_bone_table(
    blob: bytes,
    section_start: int,
    section_end: int,
    bone_count: int,
) -> int | None:
    search_start = max(section_start, section_end - 0x400)
    for offset in range(search_start, section_end - 8, 2):
        bone_ids = []
        cursor = offset
        while cursor + 2 <= section_end:
            bone = int(blob[cursor])
            if blob[cursor + 1] != 0 or bone >= bone_count or (bone_ids and bone == 0):
                break
            bone_ids.append(bone)
            cursor += 2
        if len(bone_ids) >= 4 and all(
            bone_ids[index] + 1 == bone_ids[index + 1]
            for index in range(len(bone_ids) - 1)
        ):
            return offset
    return None


def parse_rotation_tracks(blob: bytes, bone_count: int) -> list[dict]:
    section_start = be32(blob, 0x3C)
    section_end = be32(blob, 0x38)
    if not (0x40 <= section_start < section_end <= len(blob)):
        return []

    native_tracks = decode_native_rotation_stream(
        blob, section_start, section_end, bone_count
    )
    if native_tracks is not None:
        return native_tracks

    table_start = find_terminal_bone_table(blob, section_start, section_end, bone_count)
    scan_end = table_start if table_start is not None else section_end

    candidates = []
    for offset in range(section_start, max(section_start, scan_end - 3), 4):
        track = decode_rotation_track(blob, offset, scan_end, bone_count)
        if track is not None:
            candidates.append(track)

    explicit = []
    last_end = section_start
    for track in candidates:
        bone = int(track["bone"])
        if int(track["rel"]) < last_end:
            continue
        if explicit and bone < int(explicit[-1]["bone"]):
            continue
        explicit.append(track)
        last_end = int(track["container_end"])

    tracks = []
    for index, track in enumerate(explicit):
        tracks.append(track)
        if index + 1 >= len(explicit):
            continue
        next_track = explicit[index + 1]
        if int(next_track["bone"]) - int(track["bone"]) >= 2:
            continuation = decode_continuation_rotation_track(
                blob,
                int(track["container_end"]),
                int(next_track["rel"]),
                int(track["bone"]) + 1,
                bone_count,
            )
            if continuation is not None:
                tracks.append(continuation)

    if explicit:
        gap_start = int(explicit[-1]["container_end"])
        gap_end = table_start if table_start is not None else scan_end
        if gap_end > gap_start:
            continuation = decode_continuation_rotation_track(
                blob,
                gap_start,
                gap_end,
                int(explicit[-1]["bone"]) + 1,
                bone_count,
            )
            if continuation is not None:
                tracks.append(continuation)

    unique = []
    seen_bones = set()
    for track in sorted(tracks, key=lambda item: (int(item["rel"]), int(item["bone"]))):
        bone = int(track["bone"])
        if bone in seen_bones:
            continue
        seen_bones.add(bone)
        unique.append(track)
    return unique


def parse_translation_tracks(blob: bytes, bone_count: int) -> list[dict]:
    section_start = be32(blob, 0x38)
    section_end = be32(blob, 0x24)
    if not (0x40 <= section_start < section_end <= len(blob)):
        return []

    tracks = []
    previous_bone = -1
    cursor = section_start
    while cursor + 16 <= section_end:
        selected = None
        for scale_count in (3, 1):
            header_offset = cursor + scale_count * 4
            if header_offset + 4 > section_end:
                continue
            bone, key_count, last_key, flags = struct.unpack_from(">BBBB", blob, header_offset)
            records_end = header_offset + 4 + key_count * 8
            if (
                not (previous_bone < bone < bone_count)
                or key_count <= 0
                or last_key != key_count - 1
                or flags != 0
                or records_end > section_end
            ):
                continue
            raw_scales = struct.unpack_from(f">{scale_count}f", blob, cursor)
            if not all(math.isfinite(value) and abs(value) < 1.0e7 for value in raw_scales):
                continue
            times = [
                struct.unpack_from(">H", blob, header_offset + 4 + index * 8)[0]
                for index in range(key_count)
            ]
            if not track_times_valid(times):
                continue
            scales = raw_scales if scale_count == 3 else raw_scales * 3
            selected = {
                "bone": int(bone),
                "scales": tuple(float(value) for value in scales),
                "records_pos": header_offset + 4,
                "key_count": int(key_count),
                "last_key": int(last_key),
                "flags": int(flags),
                "rel": cursor,
                "end": records_end,
                "layout": "scale3_header_time_xyz",
            }
            break
        if selected is None:
            break
        tracks.append(selected)
        previous_bone = int(selected["bone"])
        cursor = (int(selected["end"]) + 15) & ~15
    return tracks


def quaternion_keys(
    records: list[tuple[int, int, int, int]],
    duration: float,
    rest_quaternion,
    layout: str,
) -> list[tuple[float, tuple[float, float, float, float]]]:
    result = []
    previous = None
    rest_quaternion = normalize_quaternion(rest_quaternion)
    for index, record in enumerate(records):
        stored_time, raw_x, raw_y, raw_z = record
        if layout == "explicit_qxyz_time":
            time = 0 if index == 0 else int(records[index - 1][0])
            sign_is_negative = None if index == 0 else bool(int(time) & 1)
        else:
            time = int(stored_time)
            sign_is_negative = (
                None
                if layout == "first_qxyz_then_time_qxyz" and index == 0
                else bool(time & 1)
            )
        x, y, z = (float(value) / 32767.0 for value in (raw_x, raw_y, raw_z))
        w = math.sqrt(max(0.0, 1.0 - x * x - y * y - z * z))
        reference = previous if previous is not None else rest_quaternion
        if sign_is_negative is None:
            if layout == "first_qxyz_then_time_qxyz":
                # The sequential Wii stream has no W-sign bit for its first
                # key. Native data and the matching writer therefore use the
                # positive square root; guessing from the rest pose can select
                # a different rotation when a clip is reversed.
                quaternion = normalize_quaternion((x, y, z, w))
            else:
                positive = normalize_quaternion((x, y, z, w))
                negative = normalize_quaternion((x, y, z, -w))
                quaternion = max(
                    (positive, negative),
                    key=lambda candidate: abs(quaternion_dot(candidate, reference)),
                )
        else:
            quaternion = normalize_quaternion((x, y, z, -w if sign_is_negative else w))
        # A complete quaternion sign flip preserves the represented rotation
        # and keeps interpolation on the shortest equivalent path.
        if quaternion_dot(quaternion, reference) < 0.0:
            quaternion = tuple(-value for value in quaternion)
        previous = quaternion
        result.append((animation_seconds(time, duration), quaternion))
    return ordered_keys(result)


def clean_entry_name(name: str) -> str:
    return name.split("/", 1)[-1]


def safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)[:120]


def decode_bdg_animations(
    animation_path: Path,
    skeleton: dict[int, dict],
    export_scale: float,
    output_dir: Path,
) -> tuple[list[dict], list[dict], dict]:
    parser = PipeworksParser(str(animation_path))
    entries = parser.parse()
    data = parser.file_data or b""
    bone_count = max(skeleton) + 1 if skeleton else 0
    animations = []
    resource_locations = []
    skipped = []
    rotation_track_count = 0
    translation_track_count = 0
    for entry in entries:
        if entry["file_type"] != 4 or entry["is_resource"]:
            continue
        entry_name = str(entry["name"])
        if "SKELETON" in entry_name.upper():
            continue
        offset = int(entry["offset"])
        size = int(entry["size"])
        blob = data[offset : offset + size]
        if len(blob) < 0x44 or be32(blob, 0x20) != 2:
            skipped.append({"name": entry_name, "reason": "not_wii_type4_animation"})
            continue
        declared_size = be32(blob, 0x24)
        native_bone_count = be32(blob, 0x2C)
        duration = bef(blob, 0x1C)
        if (
            declared_size > len(blob)
            or not (0 < native_bone_count <= bone_count)
            or not math.isfinite(duration)
            or not (0.0 < duration <= 120.0)
        ):
            skipped.append({"name": entry_name, "reason": "incompatible_type4_header"})
            continue

        rotation_tracks = parse_rotation_tracks(blob, bone_count)
        translation_tracks = parse_translation_tracks(blob, bone_count)
        tracks_by_bone: dict[int, dict] = {}
        for track in translation_tracks:
            bone = int(track["bone"])
            keys = []
            scales = track["scales"]
            for index in range(int(track["key_count"])):
                record_offset = int(track["records_pos"]) + index * 8
                time, x, y, z = struct.unpack_from(">Hhhh", blob, record_offset)
                value = tuple(
                    float(raw) / 32767.0 * float(scales[axis]) * float(export_scale)
                    for axis, raw in enumerate((x, y, z))
                )
                keys.append((animation_seconds(time, duration), value))
            keys = ordered_keys(keys)
            if keys:
                tracks_by_bone.setdefault(bone, {"bone": bone})["translation_keys"] = keys
                translation_track_count += 1
        for track in rotation_tracks:
            bone = int(track["bone"])
            keys = quaternion_keys(
                track["records"],
                duration,
                skeleton[bone]["q"],
                str(track["layout"]),
            )
            if keys:
                tracks_by_bone.setdefault(bone, {"bone": bone})["rotation_keys"] = keys
                tracks_by_bone[bone]["rotation_layout"] = track["layout"]
                tracks_by_bone[bone]["rotation_record_count"] = track["record_count"]
                rotation_track_count += 1

        if not tracks_by_bone:
            skipped.append({"name": entry_name, "reason": "no_valid_tracks"})
            continue
        name = clean_entry_name(entry_name)
        resource_id = int(entry["file_num"])
        filename = safe_filename(name) + ".bin"
        payload = bytes(blob)
        animations.append(
            {
                "resource_id": resource_id,
                "name": name,
                "duration": float(duration),
                "tracks": [tracks_by_bone[index] for index in sorted(tracks_by_bone)],
                "source_entry": entry_name,
            }
        )
        resource_locations.append(
            {
                "resource_id": resource_id,
                "name": name,
                "descriptor_name": entry_name,
                "safe_filename": filename,
                "absolute_offset": hex(offset),
                "size": size,
                "duration_seconds": float(duration),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "import_rule": "exact_raw_same_size_only",
                "native_rotation_tracks": [
                    {
                        "bone_id": int(track["bone"]),
                        "bone_name": str(skeleton[int(track["bone"])]["name"]),
                        "layout": str(track["layout"]),
                        "track_rel": hex(int(track["rel"])),
                        "record_count": int(track["native_record_count"]),
                    }
                    for track in rotation_tracks
                ],
            }
        )
    report = {
        "decoder": "bdg_animations.py",
        "source": animation_path.name,
        "clips": len(animations),
        "rotation_tracks": rotation_track_count,
        "translation_tracks": translation_track_count,
        "skipped_entries": skipped,
        "root_pvm_required": False,
        "sampling": "native big-endian Wii Type-4 tracks with encoded W-sign reconstruction exported to FBX at 60 fps",
    }
    return animations, resource_locations, report
