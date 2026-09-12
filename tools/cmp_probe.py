from __future__ import annotations

import argparse
import math
import struct
import sys
from pathlib import Path

from PIL import Image

TOOL_DIR = Path(__file__).resolve().parent
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from cmg_probe import bundle_strings, clean_name, global_matrices, write_fbx
from parser_core import PipeworksParser

def le32(data: bytes, off: int) -> int:
    return struct.unpack_from("<I", data, off)[0]


def sle32(data: bytes, off: int) -> int:
    return struct.unpack_from("<i", data, off)[0]


def normalize_quat(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    n = math.sqrt(sum(v * v for v in q))
    if n < 1e-8:
        return (0.0, 0.0, 0.0, 1.0)
    return tuple(v / n for v in q)


def read_cmp_vertex(data: bytes, off: int, scale: float) -> tuple[float, float, float]:
    x, y, z = struct.unpack_from("<fff", data, off)
    return (x * scale, y * scale, z * scale)


def parse_cmp_materials(data: bytes, entries: list[dict]) -> list[dict]:
    materials: list[dict] = []
    for entry in entries:
        if entry["file_type"] != 6 or entry["is_resource"]:
            continue
        blob = data[entry["offset"] : entry["offset"] + entry["size"]]
        if len(blob) < 0x28:
            continue
        material_word = le32(blob, 0)
        material_id = material_word & 0xFFFF
        material_kind = material_word >> 16
        name = clean_name(entry["name"].split("/", 1)[-1])
        ambient_rgba = tuple(float(v) for v in struct.unpack_from("<4f", blob, 0x08))
        diffuse_rgba = tuple(float(v) for v in struct.unpack_from("<4f", blob, 0x18))
        specular_rgba = tuple(float(v) for v in struct.unpack_from("<4f", blob, 0x28))
        ambient = ambient_rgba[:3]
        diffuse = diffuse_rgba[:3]
        specular = specular_rgba[:3]
        if not all(math.isfinite(v) and abs(v) <= 4.0 for v in ambient):
            ambient = (0.75, 0.75, 0.75)
        if not all(math.isfinite(v) and abs(v) <= 4.0 for v in diffuse):
            diffuse = (0.75, 0.78, 0.72)
        if not all(math.isfinite(v) and abs(v) <= 4.0 for v in specular):
            specular = (0.15, 0.15, 0.15)
        texture_hints = struct.unpack_from("<HH", blob, 0x38) if len(blob) >= 0x3C else (0, 0)
        shiny = max(specular) > 0.45 or material_kind == 5
        materials.append(
            {
                "id": material_id,
                "name": name,
                "color": diffuse,
                "ambient": ambient,
                "ambient_factor": ambient_rgba[3],
                "specular": specular,
                "diffuse_factor": diffuse_rgba[3],
                "specular_factor": specular_rgba[3],
                "opacity": diffuse_rgba[3],
                "shininess": 80.0 if shiny else 18.0,
                "kind": material_kind,
                "flags": le32(blob, 0x04) if len(blob) >= 8 else 0,
                "texture_hints": texture_hints,
            }
        )
    return materials


def cmp_entry_id(parser: PipeworksParser, entry: dict) -> int:
    data = parser.file_data or b""
    off = parser.metadata_offset + int(entry["file_num"]) * 0x10 + 4
    return le32(data, off) if off + 4 <= len(data) else -1


def unswizzle_psmt8(data: bytes, width: int, height: int) -> bytes:
    out = bytearray(width * height)
    for y in range(height):
        for x in range(width):
            block_location = (y & ~0x0F) * width + (x & ~0x0F) * 2
            swap_selector = (((y + 2) >> 2) & 1) * 4
            pos_y = (((y & ~3) >> 1) + (y & 1)) & 7
            column_location = pos_y * width * 2 + ((x + swap_selector) & 7) * 4
            byte_num = ((y >> 1) & 1) + ((x >> 2) & 2)
            out[y * width + x] = data[block_location + column_location + byte_num]
    return bytes(out)


def decode_ps2_indexed8(texture: bytes, palette: bytes, width: int, height: int) -> Image.Image:
    indices = unswizzle_psmt8(texture, width, height)
    colors = []
    for index in range(256):
        clut_index = (index & 0xE7) | ((index & 0x08) << 1) | ((index & 0x10) >> 1)
        r, g, b, a = palette[clut_index * 4 : clut_index * 4 + 4]
        colors.append((r, g, b, min(255, a * 2)))
    image = Image.new("RGBA", (width, height))
    image.putdata([colors[index] for index in indices])
    return image


def parse_cmp_textures(parser: PipeworksParser, entries: list[dict], texture_dir: Path) -> dict[int, dict]:
    data = parser.file_data or b""
    resources = {
        (entry["file_type"], entry["file_num"]): entry
        for entry in entries
        if entry["is_resource"]
    }
    palettes: dict[int, bytes] = {}
    for entry in entries:
        if entry["file_type"] != 13 or entry["is_resource"]:
            continue
        resource = resources.get((13, entry["file_num"]))
        if resource and resource["size"] >= 1024:
            palettes[cmp_entry_id(parser, entry)] = data[resource["offset"] : resource["offset"] + 1024]

    textures: dict[int, dict] = {}
    for entry in entries:
        if entry["file_type"] != 9 or entry["is_resource"]:
            continue
        resource = resources.get((9, entry["file_num"]))
        if not resource:
            continue
        blob = data[entry["offset"] : entry["offset"] + entry["size"]]
        if len(blob) < 0x24 or le32(blob, 0x04) != 0x40000013:
            continue
        dimensions = le32(blob, 0x08)
        width = dimensions & 0xFFFF
        height = dimensions >> 16
        if width <= 0 or height <= 0 or width % 16 or height % 16:
            continue
        palette = palettes.get(le32(blob, 0x20) & 0xFFFF)
        if palette is None or resource["size"] < width * height:
            continue
        raw = data[resource["offset"] : resource["offset"] + width * height]
        name = clean_name(entry["name"].split("/", 1)[-1])
        png_name = f"{name}.png"
        texture_dir.mkdir(parents=True, exist_ok=True)
        out_path = texture_dir / png_name
        # Blender's FBX importer always treats a diffuse PNG's alpha channel as
        # surface opacity. CMP Type-6 stores opacity separately, so link an RGB
        # diffuse image and preserve the material's native opacity on the FBX.
        decode_ps2_indexed8(raw, palette, width, height).convert("RGB").save(out_path)
        textures[cmp_entry_id(parser, entry)] = {
            "name": name,
            "width": width,
            "height": height,
            "file": out_path,
            "relative": f"textures/{png_name}",
        }
    return textures


def attach_cmp_material_textures(materials: list[dict], textures: dict[int, dict]) -> None:
    for material in materials:
        for texture_id in material.get("texture_hints") or ():
            texture = textures.get(int(texture_id))
            if texture is not None:
                material["texture"] = texture
                break


def parse_cmp_material_ranges(main: bytes, packets: list[dict], materials: list[dict]) -> list[dict]:
    if len(main) < 0x10 or not packets:
        return []
    material_by_id = {int(m["id"]): m for m in materials}
    first = le32(main, 0)
    record_count = first & 0xFFFF
    table_start = 0x10 + (first >> 16) * 4
    ranges: list[dict] = []
    for pair_index in range(record_count):
        off = table_start + pair_index * 0x60
        if off + 0x60 > len(main):
            break
        draw_off = off
        material_off = off + 0x30
        draw_count = le32(main, draw_off) + 2
        material_word = le32(main, material_off + 0x08)
        material_id = material_word & 0xFFFF
        material = material_by_id.get(material_id)
        if draw_count > 0 and material is not None:
            ranges.append(
                {
                    "index": pair_index,
                    "draw_offset": draw_off,
                    "material_offset": material_off,
                    "draw_count": draw_count,
                    "material_id": material_id,
                    "material": material,
                }
            )
    return ranges


def packet_material_for(index: int, packet_count: int, materials: list[dict], ranges: list[dict] | None = None) -> dict | None:
    if ranges:
        material_range = next((item for item in ranges if item["index"] == index), None)
        if material_range is not None:
            return material_range["material"]
    if not materials:
        return None
    mesh_materials = [m for m in materials if "biped" not in m["name"].lower()]
    if not mesh_materials:
        mesh_materials = materials
    if packet_count == 1:
        return mesh_materials[0]
    if len(mesh_materials) == 1:
        return mesh_materials[0]
    if index < len(mesh_materials):
        return mesh_materials[index]
    return mesh_materials[-1]


def read_cmp_vertex_control(data: bytes, off: int) -> bytes:
    if off + 16 <= len(data):
        return bytes(data[off + 12 : off + 16])
    return b"\x00\x00\x00\x00"


def parse_cmp_skin_palette(main: bytes, strings: list[str], bones: list[dict]) -> dict[int, int]:
    if len(main) < 0x10 or not bones:
        return {}

    table_count = le32(main, 0) >> 16
    if table_count <= 0 or table_count > 512 or 0x10 + table_count * 4 > len(main):
        return {}

    bone_by_name = {bone["name"]: int(bone["idx"]) for bone in bones}
    palette: dict[int, int] = {}
    name_offsets = [0x04, 0x08, 0x0C]
    name_offsets.extend(0x10 + i * 4 for i in range(table_count))
    for raw_id, off in enumerate(name_offsets):
        string_idx = le32(main, off)
        if not (0 <= string_idx < len(strings)):
            continue
        bone_idx = bone_by_name.get(strings[string_idx])
        if bone_idx is not None:
            palette[raw_id] = bone_idx
    return palette


def decode_cmp_skin_control(control: bytes, skin_palette: dict[int, int]) -> list[tuple[int, float]] | None:
    if len(control) != 4 or not skin_palette:
        return None
    blend, raw_a, raw_b = struct.unpack("<HBB", control)
    # Several stem CMPs contain small authored overshoots above the nominal
    # 4096 fixed-point maximum. The game accepts them as full first-bone weight.
    if blend > 0x1100 or raw_a not in skin_palette or raw_b not in skin_palette:
        return None

    bone_a = skin_palette[raw_a]
    bone_b = skin_palette[raw_b]

    weight_a = min(blend, 4096) / 4096.0
    weight_b = 1.0 - weight_a
    combined: dict[int, float] = {}
    if weight_a > 0.0:
        combined[bone_a] = combined.get(bone_a, 0.0) + weight_a
    if weight_b > 0.0:
        combined[bone_b] = combined.get(bone_b, 0.0) + weight_b
    return sorted(combined.items())


def cmp_packet_side_stream(packet: dict) -> int:
    return packet["rel"] + packet["count"] * 16


def cmp_packet_uv_stream(packet: dict) -> int:
    side_end = cmp_packet_side_stream(packet) + packet["count"] * 4
    return (side_end + 15) & ~15


def read_cmp_uv(data: bytes, uv_stream: int, index: int) -> tuple[float, float]:
    pos = uv_stream + index * 4
    if pos + 4 <= len(data):
        u_raw, v_raw = struct.unpack_from("<hh", data, pos)
        return (u_raw / 4096.0, 1.0 - v_raw / 4096.0)
    return (0.0, 0.0)


def read_cmp_normal(record: bytes) -> tuple[float, float, float]:
    if len(record) < 3:
        return (0.0, 0.0, 1.0)
    x, y, z = struct.unpack("<bbb", record[:3])
    length = math.sqrt(x * x + y * y + z * z)
    if length <= 1e-8:
        return (0.0, 0.0, 1.0)
    return (x / length, y / length, z / length)


def read_cmp_marker(data: bytes, side_stream: int, index: int) -> int:
    pos = side_stream + index * 4 + 3
    if 0 <= pos < len(data):
        return data[pos]
    return 0x7F


def read_cmp_side_record(data: bytes, side_stream: int, index: int) -> bytes:
    pos = side_stream + index * 4
    if 0 <= pos <= len(data) - 4:
        return bytes(data[pos : pos + 4])
    return b"\x00\x00\x00\x00"


def cmp_side_record_draws(record: bytes, all_draw_stream: bool = False) -> bool:
    if len(record) < 4 or record[3] == 0:
        return False
    return True


def cmp_packet_uses_vertex_draw_flags(vertex_controls: list[bytes]) -> bool:
    flags = set(vertex_controls)
    return bool(flags) and flags.issubset({b"\x00\x00\x00\x00", b"\x00\x00\x80\x3f"})


def cmp_vertex_control_draws(control: bytes) -> bool:
    return control == b"\x00\x00\x80\x3f"


def build_adc_strip_faces(
    verts: list[tuple[float, float, float]],
    markers: list[int],
    logical_indices: list[int] | None = None,
) -> tuple[list[tuple[int, int, int]], dict[int, tuple[float, float]], list[int]]:
    faces: list[tuple[int, int, int]] = []
    kept_source_indices: list[int] = []
    if logical_indices is None:
        logical_indices = list(range(len(verts)))
    for i in range(len(verts) - 2):
        if not markers[i + 2]:
            continue
        tri = (i, i + 1, i + 2) if i % 2 == 0 else (i + 1, i, i + 2)
        logical_tri = (logical_indices[tri[0]], logical_indices[tri[1]], logical_indices[tri[2]])
        if logical_tri[0] == logical_tri[1] or logical_tri[1] == logical_tri[2] or logical_tri[0] == logical_tri[2]:
            continue
        faces.append(tri)
        kept_source_indices.extend(tri)
    return faces, {}, kept_source_indices


def cmp_stream_logical_indices(
    resource: bytes,
    packet: dict,
    side_stream: int,
    uv_stream: int,
) -> list[int]:
    logical_by_record: dict[bytes, int] = {}
    out: list[int] = []
    for i in range(packet["count"]):
        vertex_offset = packet["rel"] + i * 16
        record = bytes(resource[vertex_offset : vertex_offset + 12])
        logical = logical_by_record.setdefault(record, len(logical_by_record))
        out.append(logical)
    return out


def find_cmp_packets(main: bytes, resource: bytes) -> list[dict]:
    packets: list[dict] = []
    for off in range(0, max(0, len(main) - 12), 4):
        count = le32(main, off)
        stride = le32(main, off + 4)
        fmt = le32(main, off + 8)
        if stride != 0x2C or fmt != 0x116 or not (0 < count < 100000):
            continue
        rel = le32(main, off + 0x10) if off + 0x14 <= len(main) else 0
        region_size = le32(main, off + 0x14) if off + 0x18 <= len(main) else 0
        if rel >= len(resource) or rel + count * 16 > len(resource):
            continue
        packet = {"rel": rel, "count": count, "desc": off}
        if 0 < region_size <= len(resource) - rel:
            packet["region_size"] = region_size
        packets.append(packet)

    by_rel: dict[int, dict] = {}
    for packet in packets:
        rel = packet["rel"]
        current = by_rel.get(rel)
        if current is None or packet["count"] > current["count"]:
            by_rel[rel] = packet
    packets = sorted(by_rel.values(), key=lambda packet: packet["rel"])

    compact_packets: list[dict] = []
    for off in range(0, max(0, len(main) - 0x18 + 1), 4):
        raw_count = le32(main, off)
        stride = le32(main, off + 4)
        fmt = le32(main, off + 8)
        rel = le32(main, off + 0x10)
        region_size = le32(main, off + 0x14)
        if stride not in (0x10, 0x14, 0x18, 0x2C) or fmt not in (0x2, 0x102, 0x112, 0x116):
            continue
        if not (0 < raw_count < 100000):
            continue
        if not (0 <= rel < len(resource) and 0 < region_size <= len(resource) - rel):
            continue
        count = raw_count
        side = rel + count * 16
        uv = (side + count * 4 + 15) & ~15
        end = uv + count * 4
        if end > rel + region_size:
            continue
        compact_packets.append(
            {
                "rel": rel,
                "count": count,
                "desc": off,
                "stored_count_delta": 0,
                "compact_fmt": fmt,
                "region_size": region_size,
            }
        )
    by_rel = {}
    for packet in [*packets, *compact_packets]:
        current = by_rel.get(packet["rel"])
        if current is None or packet["count"] > current["count"]:
            by_rel[packet["rel"]] = packet
    packets = sorted(by_rel.values(), key=lambda packet: packet["rel"])
    if packets:
        return packets

    if len(main) >= 0x1f8:
        count0 = le32(main, 0x198)
        count1 = le32(main, 0x1DC)
        rel1 = le32(main, 0x1EC)
        for rel, count, desc in ((0, count0, 0x198), (rel1, count1, 0x1DC)):
            if count and rel < len(resource) and rel + count * 16 <= len(resource):
                packets.append({"rel": rel, "count": count, "desc": desc})
    return packets


def parse_cmp_pose_records(data: bytes) -> dict[int, dict]:
    if len(data) < 0x38:
        return {}
    count = le32(data, 0x2C)
    if count <= 0 or count > 512 or 0x38 + count * 4 > len(data):
        return {}
    records: dict[int, dict] = {}
    for i in range(count):
        rel = le32(data, 0x38 + i * 4)
        if rel + 0x24 > len(data):
            continue
        idx = sle32(data, rel)
        if idx < 0 or idx > 512:
            continue
        translation_rel = le32(data, rel + 0x1C)
        rotation_rel = le32(data, rel + 0x20)
        translation_pos = rel + translation_rel
        rotation_pos = rel + rotation_rel
        if translation_pos + 12 > len(data) or rotation_pos + 16 > len(data):
            continue
        t = struct.unpack_from("<3f", data, translation_pos)
        qx, qy, qz, qw = normalize_quat(struct.unpack_from("<4f", data, rotation_pos))
        q = (
            (qx, qy, qz, qw)
            if (translation_rel, rotation_rel) != (0x24, 0x3C)
            else (-qx, -qy, -qz, qw)
        )
        records[idx] = {
            "idx": idx,
            "record_pos": rel,
            "translation_pos": translation_pos,
            "rotation_pos": rotation_pos,
            "translation": t,
            "rotation": q,
        }
    return records


def parse_cmp_pose_resource(data: bytes) -> dict[int, tuple[tuple[float, float, float], tuple[float, float, float, float]]]:
    return {
        idx: (record["translation"], record["rotation"])
        for idx, record in parse_cmp_pose_records(data).items()
    }


def parse_cmp_type3_pose_records(data: bytes) -> dict[int, dict]:
    if len(data) < 0x40:
        return {}
    count = le32(data, 0x20)
    pose_start = le32(data, 0x3C)
    if count <= 0 or count > 512 or pose_start < 0x40:
        return {}
    records: dict[int, dict] = {}
    for idx in range(count):
        record_pos = pose_start + idx * 0x50
        translation_pos = record_pos + 0x34
        if translation_pos + 12 > len(data):
            return {}
        records[idx] = {
            "idx": idx,
            "record_pos": record_pos,
            "inverse_bind_pos": record_pos + 0x04,
            "translation_pos": translation_pos,
            "translation": struct.unpack_from("<3f", data, translation_pos),
        }
    return records


def parse_cmp_skeleton(parser: PipeworksParser, entries: list[dict], scale: float) -> tuple[list[dict], dict[int, list[list[float]]]]:
    data = parser.file_data or b""
    hierarchy_entry = next((e for e in entries if e["file_type"] == 3 and "SKELETON" in e["name"].upper()), None)
    pose_entry = next((e for e in entries if e["file_type"] == 4 and "SKELETON" in e["name"].upper()), None)
    if not hierarchy_entry or not pose_entry:
        return [], {}

    hierarchy = data[hierarchy_entry["offset"] : hierarchy_entry["offset"] + hierarchy_entry["size"]]
    pose_data = data[pose_entry["offset"] : pose_entry["offset"] + pose_entry["size"]]
    if len(hierarchy) < 0x44:
        return [], {}

    bone_count = le32(hierarchy, 0x20)
    if bone_count <= 0 or bone_count > 512:
        return [], {}

    strings = bundle_strings(parser)
    poses = parse_cmp_pose_resource(pose_data)
    bones_by_idx: dict[int, dict] = {}
    seen_offsets: set[int] = set()

    def walk(rel: int) -> None:
        if rel in seen_offsets or rel + 0x10 > len(hierarchy):
            return
        seen_offsets.add(rel)
        idx = sle32(hierarchy, rel)
        parent = sle32(hierarchy, rel + 4)
        child_count = sle32(hierarchy, rel + 8)
        name_index = sle32(hierarchy, rel + 12)
        if idx < 0 or idx >= bone_count or child_count < 0 or child_count > 128:
            return
        name = strings[name_index] if 0 <= name_index < len(strings) else f"Bone_{idx:03d}"
        t, q = poses.get(idx, ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)))
        bones_by_idx[idx] = {"idx": idx, "parent": parent, "name": name, "t": t, "q": q, "display_size": 0.3}
        child_table = rel + 0x10
        for i in range(child_count):
            pos = child_table + i * 4
            if pos + 4 <= len(hierarchy):
                walk(le32(hierarchy, pos))

    walk(0x40)
    bones = [bones_by_idx[idx] for idx in sorted(bones_by_idx)]
    return bones, global_matrices(bones, scale)


def cmp_animation_seconds(time: int, duration: float) -> float:
    # Type-4 key times use bit 0 for the omitted quaternion W sign. The actual
    # normalized timestamp is therefore always even and spans 0..65534.
    timestamp = int(time) & 0xFFFE
    return max(0.0, min(float(duration), float(timestamp) / 65534.0 * float(duration)))


def ordered_cmp_animation_keys(keys: list[tuple[float, object]]) -> list[tuple[float, object]]:
    ordered = sorted(keys, key=lambda item: float(item[0]))
    result = []
    for time, value in ordered:
        if result and float(time) <= result[-1][0] + 1e-8:
            continue
        result.append((float(time), value))
    return result


def cmp_quaternion_keys(
    records: list[tuple[int, int, int, int]],
    duration: float,
    rest_quaternion: tuple[float, float, float, float],
    layout: str = "explicit_time_qxyz",
) -> list[tuple[float, tuple[float, float, float, float]]]:
    result = []
    previous = None
    rest = normalize_quat(rest_quaternion)
    for time, raw_x, raw_y, raw_z in records:
        semantic_time = time
        sign_time = time
        sign_is_negative = bool(sign_time & 1)
        reference = previous if previous is not None else rest
        x, y, z = (-float(value) / 32767.0 for value in (raw_x, raw_y, raw_z))
        w = math.sqrt(max(0.0, 1.0 - x * x - y * y - z * z))
        quaternion = normalize_quat((x, y, z, -w if sign_is_negative else w))
        # Only flip the complete quaternion, which preserves the represented
        # rotation, to keep FBX interpolation smooth.
        if sum(quaternion[axis] * reference[axis] for axis in range(4)) < 0.0:
            quaternion = tuple(-value for value in quaternion)
        previous = quaternion
        result.append((cmp_animation_seconds(semantic_time, duration), quaternion))
    return ordered_cmp_animation_keys(result)


def decode_cmp_type4_animations(
    data: bytes,
    entries: list[dict],
    bones: list[dict],
    scale: float,
) -> tuple[list[dict], dict]:
    # Imported lazily because cmp_fbx_import also imports this module.
    from cmp_fbx_import import (
        parse_cmp_animation_rotation_tracks,
        parse_cmp_animation_translation_tracks,
    )

    bone_by_index = {int(bone["idx"]): bone for bone in bones}
    animations = []
    skipped_bodyless = []
    translation_track_count = 0
    rotation_track_count = 0
    for entry in entries:
        if entry["file_type"] != 4 or entry["is_resource"]:
            continue
        base = int(entry["offset"])
        size = int(entry["size"])
        blob = data[base : base + size]
        if len(blob) < 0x44 or le32(blob, 0x20) != 2:
            continue
        if le32(blob, 0x3C) > len(blob):
            skipped_bodyless.append(str(entry["name"]))
            continue
        duration = struct.unpack_from("<f", blob, 0x1C)[0]
        if not math.isfinite(duration) or duration <= 0.0 or duration > 120.0:
            continue

        tracks_by_bone: dict[int, dict] = {}
        for track in parse_cmp_animation_translation_tracks(blob) or []:
            bone = int(track["bone"])
            if bone not in bone_by_index:
                continue
            scales = tuple(float(value) for value in track["scales"])
            keys = []
            for key_index in range(int(track["key_count"])):
                record_pos = int(track["records_pos"]) + key_index * 8
                time, x, y, z = struct.unpack_from("<Hhhh", blob, record_pos)
                value = tuple(
                    float(raw) / 32767.0 * scales[axis] * scale
                    for axis, raw in enumerate((x, y, z))
                )
                keys.append((cmp_animation_seconds(time, duration), value))
            ordered = ordered_cmp_animation_keys(keys)
            if ordered:
                tracks_by_bone.setdefault(bone, {"bone": bone})["translation_keys"] = ordered
                translation_track_count += 1

        rotation_tracks = parse_cmp_animation_rotation_tracks(blob)
        compact_zero_layout = any(
            int(track.get("encoded_record_count", -1)) == 0
            for track in rotation_tracks
        )
        for track in rotation_tracks:
            bone = int(track["bone"])
            if bone not in bone_by_index:
                continue
            if track.get("use_bind_pose"):
                keys = [(0.0, tuple(bone_by_index[bone]["q"]))]
            else:
                records = []
                for record_index in range(int(track["record_count"])):
                    record_pos = int(track["records_pos"]) + record_index * 8
                    if track["layout"] == "explicit_qxyz_time":
                        x, y, z, time = struct.unpack_from("<hhhH", blob, record_pos)
                    else:
                        time, x, y, z = struct.unpack_from("<Hhhh", blob, record_pos)
                    records.append((int(time), int(x), int(y), int(z)))
                keys = cmp_quaternion_keys(
                    records,
                    duration,
                    tuple(bone_by_index[bone]["q"]),
                    layout=str(track["layout"]),
                )
            if keys:
                animation_track = tracks_by_bone.setdefault(bone, {"bone": bone})
                animation_track["rotation_keys"] = keys
                if int(bone_by_index[bone].get("parent", -1)) < 0:
                    bind = normalize_quat(tuple(bone_by_index[bone]["q"]))
                    first = normalize_quat(tuple(keys[0][1]))
                    dot = abs(sum(bind[axis] * first[axis] for axis in range(4)))
                    angle = math.degrees(2.0 * math.acos(max(-1.0, min(1.0, dot))))
                    if not compact_zero_layout or angle <= 90.0:
                        animation_track["_preserve_native_root_orientation"] = True
                rotation_track_count += 1

        if tracks_by_bone:
            animations.append(
                {
                    "name": clean_name(str(entry["name"])),
                    "duration": float(duration),
                    "tracks": [tracks_by_bone[index] for index in sorted(tracks_by_bone)],
                    "source_entry": str(entry["name"]),
                }
            )
    return animations, {
        "clips": len(animations),
        "translation_tracks": translation_track_count,
        "rotation_tracks": rotation_track_count,
        "bodyless_stubs_skipped": skipped_bodyless,
        "sampling": "native Type-4 keys decoded and quaternion-sampled to FBX curves at 60 fps",
    }


def export_cmp(cmp: Path, fbx: Path, scale: float) -> dict:
    parser = PipeworksParser(str(cmp))
    entries = parser.parse()
    data = parser.file_data or b""
    main_entry = next((e for e in entries if e["file_type"] == 17 and not e["is_resource"]), None)
    res_entry = next((e for e in entries if e["file_type"] == 17 and e["is_resource"]), None)
    if not main_entry or not res_entry:
        raise SystemExit("No CMP type-17 mesh/resource pair found")

    main = data[main_entry["offset"] : main_entry["offset"] + main_entry["size"]]
    resource = data[res_entry["offset"] : res_entry["offset"] + res_entry["size"]]
    asset = clean_name(cmp.stem)
    materials = parse_cmp_materials(data, entries)
    textures = parse_cmp_textures(parser, entries, fbx.parent / "textures")
    attach_cmp_material_textures(materials, textures)
    strings = bundle_strings(parser)
    submeshes = []
    total_faces = 0
    total_vertices = 0
    packet_stats = []
    packets = find_cmp_packets(main, resource)
    if not packets:
        raise SystemExit("No CMP mesh packets found")
    material_ranges = parse_cmp_material_ranges(main, packets, materials)
    bones, globals_ = parse_cmp_skeleton(parser, entries, scale)
    animations, animation_report = decode_cmp_type4_animations(data, entries, bones, scale)
    skin_palette = parse_cmp_skin_palette(main, strings, bones)

    for i, packet in enumerate(packets):
        verts: list[tuple[float, float, float]] = []
        vertex_uvs: list[tuple[float, float]] = []
        vertex_normals: list[tuple[float, float, float]] = []
        markers: list[int] = []
        vertex_controls: list[bytes] = []
        side_stream = cmp_packet_side_stream(packet)
        uv_stream = cmp_packet_uv_stream(packet)
        side_records = [read_cmp_side_record(resource, side_stream, j) for j in range(packet["count"])]
        for j in range(packet["count"]):
            vertex_controls.append(read_cmp_vertex_control(resource, packet["rel"] + j * 16))
        use_vertex_draw_flags = cmp_packet_uses_vertex_draw_flags(vertex_controls)
        decoded_weights = [decode_cmp_skin_control(control, skin_palette) for control in vertex_controls]
        stream_skin = (
            not use_vertex_draw_flags
            and "compact_fmt" not in packet
            and bool(decoded_weights)
            and all(weights is not None for weights in decoded_weights)
        )
        compact_skin_bone = None
        compact_bone_off = int(packet["desc"]) + 0x18
        if "compact_fmt" in packet and compact_bone_off + 4 <= len(main):
            compact_skin_bone = skin_palette.get(le32(main, compact_bone_off))
        exact_skin = stream_skin or compact_skin_bone is not None
        if stream_skin:
            vertex_weights = decoded_weights
        elif compact_skin_bone is not None:
            vertex_weights = [[(compact_skin_bone, 1.0)] for _ in range(packet["count"])]
        else:
            vertex_weights = []
        logical_indices = cmp_stream_logical_indices(resource, packet, side_stream, uv_stream)
        for j in range(packet["count"]):
            vertex_offset = packet["rel"] + j * 16
            vert = read_cmp_vertex(resource, vertex_offset, scale)
            verts.append(vert)
            vertex_uvs.append(read_cmp_uv(resource, uv_stream, j))
            vertex_normals.append(read_cmp_normal(side_records[j]))
            if use_vertex_draw_flags:
                markers.append(1 if cmp_vertex_control_draws(vertex_controls[j]) else 0)
            else:
                markers.append(1 if cmp_side_record_draws(side_records[j]) else 0)
        faces, _, kept = build_adc_strip_faces(
            verts,
            markers,
            None if use_vertex_draw_flags else logical_indices,
        )
        material = packet_material_for(i, len(packets), materials, material_ranges)
        source_index_count = len(verts)
        unused_vertices = 0
        corner_uvs: dict[int, tuple[float, float]] = {}
        for corner, source_index in enumerate(kept):
            corner_uvs[corner] = vertex_uvs[source_index]
        submeshes.append(
            {
                "name": f"{asset}_packet{i}",
                "vertices": verts,
                "faces": faces,
                "uvs": corner_uvs,
                "normals": vertex_normals if not use_vertex_draw_flags else [],
                "cmp_packet": packet,
                "preserve_triangle_uvs": True,
                "material_name": material["name"] if material else f"{asset}_Material",
                "material_color": material["color"] if material else (0.75, 0.78, 0.72),
                "material_ambient": material.get("ambient") if material else None,
                "material_ambient_factor": material.get("ambient_factor") if material else None,
                "material_specular": material.get("specular") if material else None,
                "material_diffuse_factor": material.get("diffuse_factor") if material else None,
                "material_specular_factor": material.get("specular_factor") if material else None,
                "material_opacity": material.get("opacity") if material else None,
                "material_shininess": material.get("shininess") if material else None,
                "material_kind": material.get("kind") if material else None,
                "material_flags": material.get("flags") if material else None,
                "material_texture_hints": material.get("texture_hints") if material else None,
                "material_texture": material.get("texture") if material else None,
                "vertex_weights": vertex_weights,
            }
        )
        total_faces += len(faces)
        total_vertices += len(verts)
        packet_stats.append(
            {
                "name": f"{asset}_packet{i}",
                "raw_vertices": len(verts),
                "exported_vertices": len(verts),
                "no_draw": sum(1 for marker in markers if not marker),
                "triangles": len(faces),
                "rel": packet["rel"],
                "uv_stream": uv_stream,
                "topology": "triangle_strip",
                "draw_flag_source": "vertex_w" if use_vertex_draw_flags else "side_record_w",
                "source_index_count": source_index_count,
                "fbx_corner_count": len(kept),
                "unused_vertices": unused_vertices,
                "material": material["name"] if material else f"{asset}_Material",
                "skin_mode": "cmp_vertex_weights" if exact_skin else "unweighted",
                "weighted_vertices": len(vertex_weights),
                "skin_palette_entries": len(skin_palette),
            }
        )

    fbx.parent.mkdir(parents=True, exist_ok=True)
    native_source_actions = {}
    bone_names = {int(bone["idx"]): str(bone["name"]) for bone in bones}
    for animation in animations:
        source_tracks = {}
        for track in animation.get("tracks") or []:
            bone_name = bone_names.get(int(track["bone"]))
            if bone_name is None:
                continue
            source_track = {}
            if track.get("translation_keys"):
                source_track["translation_keys"] = track["translation_keys"]
            if track.get("rotation_keys"):
                source_track["rotation_euler_keys"] = [(0.0, (0.0, 0.0, 0.0))]
            if source_track:
                source_tracks[bone_name] = source_track
        native_source_actions[str(animation["name"])] = {"tracks": source_tracks}

    write_fbx(
        fbx,
        asset,
        submeshes,
        bones,
        globals_,
        skin_mode="weights",
        animations=animations,
        self_contained_animation_preview=True,
        animation_scale=scale,
    )
    action_baseline = None
    if animations:
        from animation_baseline import build_action_baseline

        action_baseline = fbx.with_suffix(".animation_action_baseline_v1.json.gz")
        expected_names = [str(animation["name"]) for animation in animations]
        build_action_baseline(
            fbx,
            expected_names,
            action_baseline,
            source_actions=native_source_actions,
        )
    return {
        "source": str(cmp),
        "fbx": str(fbx),
        "packets": packets,
        "packet_stats": packet_stats,
        "vertices": total_vertices,
        "triangles": total_faces,
        "bones": len(bones),
        "animations": animation_report,
        "animation_action_baseline": str(action_baseline) if action_baseline else None,
        "skin_palette_entries": len(skin_palette),
        "textures": textures,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Export a PS2 Pipeworks CMP mesh to FBX")
    ap.add_argument("cmp")
    ap.add_argument("--fbx", required=True)
    ap.add_argument("--scale", type=float, default=10.0)
    args = ap.parse_args()
    report = export_cmp(Path(args.cmp), Path(args.fbx), args.scale)
    print(
        f"CMP export: packets={len(report['packets'])} "
        f"verts={report['vertices']} tris={report['triangles']} "
        f"bones={report['bones']} animations={report['animations']['clips']} fbx={report['fbx']}"
    )
    for stat in report["packet_stats"]:
        print(
            f"  {stat['name']}: rel=0x{stat['rel']:x} raw={stat['raw_vertices']} "
            f"exported={stat['exported_vertices']} no_draw={stat['no_draw']} "
            f"tris={stat['triangles']} uv=0x{stat['uv_stream']:x} "
            f"topology={stat['topology']} draw_flags={stat['draw_flag_source']} "
            f"source_index_count={stat['source_index_count']} "
            f"fbx_corner_count={stat['fbx_corner_count']} unused_vertices={stat['unused_vertices']} "
            f"material={stat['material']} skin={stat['skin_mode']} "
            f"weighted_vertices={stat['weighted_vertices']} "
            f"skin_palette_entries={stat['skin_palette_entries']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
