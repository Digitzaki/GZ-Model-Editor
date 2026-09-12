"""GameCube Pipeworks CMG mesh exporter."""
from __future__ import annotations

import argparse
import collections
import hashlib
import io
import math
import re
import struct
import sys
from pathlib import Path

TOOL_DIR = Path(__file__).resolve().parent
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from parser_core import PipeworksParser
from decode_cmpr import decode_cmpr


VALID_PRIMS = {0x80, 0x90, 0x98, 0xA0}
FBX_TICKS_PER_SECOND = 46186158000


def be32(data: bytes, off: int) -> int:
    return struct.unpack_from(">I", data, off)[0]


def be16(data: bytes, off: int) -> int:
    return struct.unpack_from(">H", data, off)[0]


def sbe32(data: bytes, off: int) -> int:
    return struct.unpack_from(">i", data, off)[0]


def fbe(data: bytes, off: int) -> float:
    return struct.unpack_from(">f", data, off)[0]


def normalize_quat(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-8:
        return (0.0, 0.0, 0.0, 1.0)
    return (x / n, y / n, z / n, w / n)


def clean_name(name: str) -> str:
    name = name.replace("\x13", "").replace("\x00", "").strip()
    name = name.split("/", 1)[-1]
    return re.sub(r"[^A-Za-z0-9_. -]+", "_", name) or "CMG"


def bundle_strings(parser: PipeworksParser) -> list[str]:
    data = parser.file_data
    off = parser.string_offset
    if not data or off <= 0 or off + 4 > len(data):
        return []
    count = struct.unpack_from("<I", data, off)[0]
    if count <= 0 or count > 20000 or off + 4 + count * 4 > len(data):
        return []
    out = []
    for i in range(count):
        rel = struct.unpack_from("<I", data, off + 4 + i * 4)[0]
        pos = off + rel
        end = data.find(b"\0", pos)
        if end < 0:
            end = pos
        out.append(clean_name(data[pos:end].decode("latin1", "replace")))
    return out


def find_type17(entries: list[dict]) -> tuple[dict, dict]:
    main = next((e for e in entries if e["file_type"] == 17 and not e["is_resource"]), None)
    if not main:
        raise ValueError("No CMG type-17 mesh entry found")
    res = next(
        (
            e
            for e in entries
            if e["file_type"] == 17
            and e["is_resource"]
            and e["file_num"] == main["file_num"]
        ),
        None,
    )
    if not res:
        raise ValueError("No matching CMG type-17 mesh resource found")
    return main, res


def find_mesh_pairs(entries: list[dict], data: bytes) -> list[tuple[dict, dict, list[dict]]]:
    pairs = []
    for main in entries:
        if main["is_resource"]:
            continue
        res = next(
            (
                e
                for e in entries
                if e["is_resource"]
                and e["file_type"] == main["file_type"]
                and e["file_num"] == main["file_num"]
            ),
            None,
        )
        if not res:
            continue
        main_blob = data[main["offset"] : main["offset"] + main["size"]]
        descs = find_mesh_descriptors(main_blob, res["size"])
        if descs:
            pairs.append((main, res, descs))
    if not pairs:
        main, res = find_type17(entries)
        main_blob = data[main["offset"] : main["offset"] + main["size"]]
        pairs.append((main, res, find_mesh_descriptors(main_blob, res["size"])))
    return pairs


def find_mesh_descriptors(main: bytes, resource_size: int) -> list[dict]:
    descs = []

    def add_desc(off: int, rel_dl: int, dl_size: int) -> None:
        attr_count = be32(main, off + 0x00)
        vertex_count = be32(main, off + 0x10)
        fmt = be32(main, off + 0x18)
        rel_vertex = be32(main, off + 0x20)
        first_section = be32(main, off + 0x2C)
        if attr_count not in (1, 2, 3, 4):
            return
        if not (0 < vertex_count < 100000):
            return
        if fmt not in (0x102, 0x112, 0x116):
            return
        if not (0 <= rel_vertex < resource_size):
            return
        if not (0 <= rel_dl < resource_size and 0 < dl_size <= resource_size - rel_dl):
            return
        # Descriptor variants store display-list offsets in different fields.
        section_offsets = [be32(main, off + i * 4) for i in range(0, min(0x68, len(main) - off) // 4)]
        # The first section offset can mark the end of the position table.
        position_count = vertex_count
        if first_section not in (0, 0xFFFFFFFF) and first_section % 12 == 0:
            candidate_count = first_section // 12
            if 0 < candidate_count <= vertex_count:
                position_count = candidate_count
        key = (rel_dl, dl_size, rel_vertex, vertex_count, fmt)
        if any(d["key"] == key for d in descs):
            return
        descs.append(
            {
                "key": key,
                "main_offset": off,
                "attr_count": attr_count,
                "vertex_count": vertex_count,
                "position_count": position_count,
                "fmt": fmt,
                "rel_vertex": rel_vertex,
                "rel_dl": rel_dl,
                "dl_size": dl_size,
                "section_offsets": section_offsets,
            }
        )

    # Tail descriptors may be shorter than the common 0x68-byte form.
    for off in range(0, len(main) - 0x4C + 1, 4):
        # Large-body descriptor form.
        old_matches = False
        if off >= 8:
            old_rel_dl = be32(main, off - 8)
            old_dl_size = be32(main, off - 4)
            if old_rel_dl + old_dl_size == be32(main, off + 0x20):
                add_desc(off, old_rel_dl, old_dl_size)
                old_matches = True
        # Compact descriptor form.
        if not old_matches and off + 0x68 <= len(main):
            add_desc(off, be32(main, off + 0x60), be32(main, off + 0x64))
    descs.sort(key=lambda d: d["main_offset"])
    for desc in descs:
        desc.pop("key", None)
    return descs


def position_stride(desc: dict) -> int:
    return 24 if desc.get("fmt") == 0x116 else 12


def parse_positions(resource: bytes, rel: int, count: int, stride: int = 12) -> list[tuple[float, float, float]]:
    out = []
    for i in range(count):
        pos = rel + i * stride
        if pos + 12 > len(resource):
            break
        out.append((fbe(resource, pos), fbe(resource, pos + 4), fbe(resource, pos + 8)))
    return out


def parse_normals(resource: bytes, desc: dict) -> list[tuple[float, float, float]]:
    if position_stride(desc) < 24:
        return []
    out = []
    rel = desc["rel_vertex"]
    for i in range(desc["vertex_count"]):
        pos = rel + i * 24 + 12
        if pos + 12 > len(resource):
            break
        nx, ny, nz = fbe(resource, pos), fbe(resource, pos + 4), fbe(resource, pos + 8)
        length = math.sqrt(nx * nx + ny * ny + nz * nz)
        if not math.isfinite(length) or length <= 1e-8:
            out.append((0.0, 0.0, 1.0))
        else:
            out.append((nx / length, ny / length, nz / length))
    return out


def bbox_for_positions(positions: list[tuple[float, float, float]]) -> tuple[float, float, float, float, float, float] | None:
    if not positions:
        return None
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    zs = [p[2] for p in positions]
    return (min(xs), max(xs), min(ys), max(ys), min(zs), max(zs))


def parse_cmg_materials(entries: list[dict], data: bytes) -> dict[int, dict]:
    materials = {}
    for entry in entries:
        if entry["file_type"] != 6 or entry["is_resource"]:
            continue
        blob = data[entry["offset"] : entry["offset"] + entry["size"]]
        if len(blob) < 2:
            continue
        material_id = struct.unpack_from(">H", blob, 0)[0]
        name = clean_name(entry["name"].split("/", 1)[-1])
        ambient = (0.75, 0.75, 0.75)
        diffuse = (0.75, 0.78, 0.72)
        specular = (0.15, 0.15, 0.15)
        if len(blob) >= 0x38:
            ambient = tuple(float(fbe(blob, 0x08 + i * 4)) for i in range(3))
            diffuse = tuple(float(fbe(blob, 0x18 + i * 4)) for i in range(3))
            specular = tuple(float(fbe(blob, 0x28 + i * 4)) for i in range(3))
        material_kind = struct.unpack_from(">H", blob, 2)[0] if len(blob) >= 4 else 0
        shiny = max(specular) > 0.45 or material_kind == 5
        materials[material_id] = {
            "id": material_id,
            "name": name,
            "color": diffuse,
            "ambient": ambient,
            "specular": specular,
            "shininess": 80.0 if shiny else 18.0,
            "texture_hint": (be32(blob, 0x38) >> 16) if len(blob) >= 0x3C else 0,
        }
    return materials


def norm_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def parse_cmg_textures(entries: list[dict], data: bytes, texture_dir: Path) -> dict[str, dict]:
    texture_dir.mkdir(parents=True, exist_ok=True)
    resources = {
        (e["file_type"], e["file_num"]): e
        for e in entries
        if e["is_resource"]
    }
    textures: dict[str, dict] = {}
    for entry in entries:
        if entry["file_type"] != 9 or entry["is_resource"]:
            continue
        res = resources.get((9, entry["file_num"]))
        if not res:
            continue
        blob = data[entry["offset"] : entry["offset"] + entry["size"]]
        if len(blob) < 0x10:
            continue
        fmt = be32(blob, 0x04)
        dims = be32(blob, 0x08)
        width = dims >> 16
        height = dims & 0xFFFF
        if fmt != 0x0E or width <= 0 or height <= 0 or width > 2048 or height > 2048:
            continue
        raw = data[res["offset"] : res["offset"] + res["size"]]
        top_mip_size = width * height // 2
        if len(raw) < top_mip_size:
            continue
        name = clean_name(entry["name"].split("/", 1)[-1])
        png_name = f"{name}.png"
        out_path = texture_dir / png_name
        decode_cmpr(raw[:top_mip_size], width, height).save(out_path)
        textures[norm_key(name)] = {
            "name": name,
            "width": width,
            "height": height,
            "file": out_path,
            "relative": f"textures/{png_name}",
        }
    return textures


def attach_cmg_material_textures(materials: dict[int, dict], textures: dict[str, dict]) -> None:
    if not textures:
        return
    eye_textures = [tex for key, tex in textures.items() if "eye" in key]
    named_eye_material_exists = any("eye" in norm_key(material["name"]) for material in materials.values())
    for material in materials.values():
        mkey = norm_key(material["name"])
        chosen = None
        for tkey, texture in textures.items():
            if tkey and (tkey in mkey or mkey in tkey):
                chosen = texture
                break
        if chosen is None and "eye" in mkey and eye_textures:
            chosen = eye_textures[0]
        if chosen is None and material["name"].startswith("Material _") and not named_eye_material_exists and len(eye_textures) == 1:
            chosen = eye_textures[0]
        if chosen is not None:
            material["texture"] = chosen


def parse_cmg_material_ranges(main: bytes, materials: dict[int, dict]) -> list[dict]:
    if len(main) < 4 or not materials:
        return []
    first = be32(main, 0)
    record_count = first >> 16
    index_count = first & 0xFFFF
    table_start = 0x1C + index_count * 4
    ranges = []
    for index in range(record_count):
        off = table_start + index * 0x64
        if off + 0x64 > len(main):
            break
        material_id = struct.unpack_from(">H", main, off + 0x24)[0]
        material = materials.get(material_id)
        if material:
            bbox_key = tuple(fbe(main, off + rel) for rel in range(0, 0x20, 4))
            ranges.append({"index": index, "offset": off, "material_id": material_id, "material": material, "bbox_key": bbox_key})
    return ranges


def cmg_descriptor_bbox_key(positions: list[tuple[float, float, float]]) -> tuple[float, ...] | None:
    bbox = bbox_for_positions(positions)
    if not bbox:
        return None
    min_x, max_x, min_y, max_y, min_z, max_z = bbox
    return (
        min_y,
        min_z,
        max_x,
        max_y,
        max_z,
        (min_x + max_x) * 0.5,
        (min_y + max_y) * 0.5,
        (min_z + max_z) * 0.5,
    )


def cmg_material_for_positions(positions: list[tuple[float, float, float]], ranges: list[dict]) -> dict | None:
    key = cmg_descriptor_bbox_key(positions)
    if not key or not ranges:
        return None
    best = None
    for material_range in ranges:
        score = sum(abs(a - b) for a, b in zip(key, material_range["bbox_key"]))
        if best is None or score < best[0]:
            best = (score, material_range)
    if best and best[0] < 0.05:
        return best[1]["material"]
    return None


def find_compact_material_descriptors(resource: bytes, descs: list[dict], ranges: list[dict]) -> list[dict]:
    matched_range_offsets = set()
    for desc in descs:
        positions = parse_positions(resource, desc["rel_vertex"], desc["vertex_count"], position_stride(desc))
        key = cmg_descriptor_bbox_key(positions)
        if not key:
            continue
        for material_range in ranges:
            score = sum(abs(a - b) for a, b in zip(key, material_range["bbox_key"]))
            if score < 0.05:
                matched_range_offsets.add(material_range["offset"])

    compact_descs: list[dict] = []
    first_normal_dl = min((d["rel_dl"] for d in descs), default=len(resource))
    for material_range in ranges:
        if material_range["offset"] in matched_range_offsets:
            continue
        best = None
        for rel_vertex in range(0, min(len(resource) - 48, 0x4000), 4):
            for vertex_count in range(4, 33):
                if rel_vertex + vertex_count * 12 > len(resource):
                    break
                positions = parse_positions(resource, rel_vertex, vertex_count, 12)
                key = cmg_descriptor_bbox_key(positions)
                if not key:
                    continue
                score = sum(abs(a - b) for a, b in zip(key, material_range["bbox_key"]))
                if best is None or score < best[0] - 1e-5 or (abs(score - best[0]) <= 1e-5 and vertex_count > best[2]):
                    best = (score, rel_vertex, vertex_count)
        if not best or best[0] >= 0.05:
            continue
        _, rel_vertex, vertex_count = best
        if rel_vertex > 0 and rel_vertex <= first_normal_dl:
            desc = {
                "main_offset": material_range["offset"],
                "attr_count": 1,
                "vertex_count": vertex_count,
                "position_count": vertex_count,
                "fmt": 0x102,
                "rel_vertex": rel_vertex,
                "rel_dl": 0,
                "dl_size": rel_vertex,
                "section_offsets": [0] * 0x20,
                "compact_material": material_range["material"],
                "compact_position_only": True,
            }
            faces, _, _ = decode_display_list(resource, desc)
            if faces:
                compact_descs.append(desc)
    return compact_descs


def tri_faces(op: int, verts: list[int]) -> list[tuple[int, int, int]]:
    prim = op & 0xF8
    faces = []
    if prim == 0x80:
        for i in range(0, len(verts) - 3, 4):
            faces.append((verts[i], verts[i + 1], verts[i + 2]))
            faces.append((verts[i], verts[i + 2], verts[i + 3]))
    elif prim == 0x90:
        for i in range(0, len(verts) - 2, 3):
            faces.append((verts[i], verts[i + 1], verts[i + 2]))
    elif prim == 0x98:
        for i in range(len(verts) - 2):
            a, b, c = verts[i], verts[i + 1], verts[i + 2]
            if a != b and b != c and a != c:
                faces.append((a, b, c) if i % 2 == 0 else (b, a, c))
    elif prim == 0xA0 and len(verts) >= 3:
        root = verts[0]
        for i in range(1, len(verts) - 1):
            a, b, c = root, verts[i], verts[i + 1]
            if a != b and b != c and a != c:
                faces.append((a, b, c))
    return faces


def add_triangles_from_primitive(
    op: int,
    verts: list[int],
    src_uvs: list[tuple[float, float] | None],
    faces: list[tuple[int, int, int]],
    corner_uvs: list[tuple[float, float] | None],
) -> None:
    prim = op & 0xF8

    def emit(indices: tuple[int, int, int]) -> None:
        a, b, c = (verts[i] for i in indices)
        if a == b or b == c or a == c:
            return
        faces.append((a, b, c))
        corner_uvs.extend(src_uvs[i] for i in indices)

    if prim == 0x80:
        for i in range(0, len(verts) - 3, 4):
            emit((i, i + 1, i + 2))
            emit((i, i + 2, i + 3))
    elif prim == 0x90:
        for i in range(0, len(verts) - 2, 3):
            emit((i, i + 1, i + 2))
    elif prim == 0x98:
        for i in range(len(verts) - 2):
            emit((i, i + 1, i + 2) if i % 2 == 0 else (i + 1, i, i + 2))
    elif prim == 0xA0 and len(verts) >= 3:
        for i in range(1, len(verts) - 1):
            emit((0, i, i + 1))


def uv_table_offset(desc: dict) -> int | None:
    sections = desc.get("section_offsets", [])
    fmt = desc.get("fmt")
    # Indexed UV streams use the descriptor section offset, not stride math.
    if fmt in (0x102, 0x112, 0x116) and len(sections) > 12 and sections[12] not in (0, 0xFFFFFFFF):
        return desc["rel_vertex"] + sections[12]
    if len(sections) > 15 and sections[15] not in (0, 0xFFFFFFFF):
        return desc["rel_vertex"] + sections[15]
    if len(sections) > 11 and sections[11] not in (0, 0xFFFFFFFF):
        return desc["rel_vertex"] + sections[11]
    return None


def read_uv_table(resource: bytes, desc: dict) -> list[tuple[float, float]]:
    off = uv_table_offset(desc)
    if off is None:
        return []
    limit = len(resource)
    out = []
    pos = off
    while pos + 8 <= limit and len(out) < 10000:
        try:
            u, v = struct.unpack_from(">2f", resource, pos)
        except Exception:
            break
        if not (math.isfinite(u) and math.isfinite(v)):
            break
        if abs(u) > 32.0 or abs(v) > 32.0:
            break
        out.append((u, 1.0 - v))
        pos += 8
    return out


def read_record(data: bytes, pos: int, mode: str) -> tuple[int, tuple[float, float] | None, int | None, int]:
    if mode == "idx8_1":
        return data[pos], None, None, pos + 1
    if mode == "idx8_2":
        return data[pos], None, data[pos + 1], pos + 2
    if mode == "idx16_2":
        return struct.unpack_from(">H", data, pos)[0], None, struct.unpack_from(">H", data, pos + 2)[0], pos + 4
    if mode == "idx8_3":
        return data[pos], None, data[pos + 2], pos + 3
    if mode == "idx16_3":
        return struct.unpack_from(">H", data, pos)[0], None, struct.unpack_from(">H", data, pos + 4)[0], pos + 6
    if mode == "idx16_idx16_idx8":
        return struct.unpack_from(">H", data, pos)[0], None, data[pos + 4], pos + 5
    if mode == "pos_norm3_uvidx":
        return data[pos], None, data[pos + 13], pos + 14
    if mode == "pos16_norm3_uvidx16":
        return struct.unpack_from(">H", data, pos)[0], None, struct.unpack_from(">H", data, pos + 14)[0], pos + 16
    if mode == "pos_normidx_uv2":
        return data[pos], (fbe(data, pos + 2), 1.0 - fbe(data, pos + 6)), None, pos + 10
    if mode == "pos_norm3_uv2":
        return data[pos], (fbe(data, pos + 13), 1.0 - fbe(data, pos + 17)), None, pos + 21
    if mode == "pos16_norm3_uv2":
        return struct.unpack_from(">H", data, pos)[0], (fbe(data, pos + 14), 1.0 - fbe(data, pos + 18)), None, pos + 22
    raise ValueError(mode)


def descriptor_record_modes(desc: dict) -> list[str]:
    """Return display-list record layout(s) implied by the CMG descriptor.

    The CMG descriptor stores the attribute stream shape separately from the GX
    primitive opcode. The important fields seen in Godzilla2K are:
      section[16] low word: normal source, 1 = inline normal, 2/3 = indexed
      section[18] high word: UV source/width, 1 = inline UV, 2 = u8, 3 = u16
    """
    fmt = desc.get("fmt")
    if desc.get("compact_position_only"):
        return ["idx8_1"]
    attr_count = desc.get("attr_count")
    sections = desc.get("section_offsets", [])
    attr_flags = sections[16] if len(sections) > 16 else 0
    uv_flags = sections[18] if len(sections) > 18 else 0
    normal_kind = attr_flags & 0xFFFF
    uv_kind = (uv_flags >> 16) & 0xFFFF

    if attr_count == 2:
        if fmt == 0x102:
            return ["idx8_2"]
        return ["idx8_2", "idx16_2"]

    if fmt == 0x116:
        if normal_kind == 2 and uv_kind == 2:
            return ["idx8_3"]
        if uv_kind == 2:
            return ["idx16_idx16_idx8"]
        return ["idx16_3"]

    if fmt == 0x112:
        if normal_kind == 1 and uv_kind == 1:
            return ["pos_norm3_uv2", "pos16_norm3_uv2"]
        if normal_kind == 1:
            return ["pos_norm3_uvidx", "pos16_norm3_uvidx16"]
        if uv_kind == 1:
            return ["pos_normidx_uv2", "pos16_norm3_uv2"]
        return ["idx8_3", "idx16_3"]

    return ["idx8_3", "idx16_3", "pos_norm3_uvidx", "pos_normidx_uv2", "pos_norm3_uv2"]


def decode_display_list(
    resource: bytes,
    desc: dict,
) -> tuple[list[tuple[int, int, int]], dict[int, tuple[float, float]], str]:
    start = desc["rel_dl"]
    end = start + desc["dl_size"]
    modes = descriptor_record_modes(desc)

    best = ([], {}, "", -1)
    uv_table = read_uv_table(resource, desc)
    for mode in modes:
        pos = start
        faces: list[tuple[int, int, int]] = []
        triangle_corner_uvs: list[tuple[float, float] | None] = []
        ok = True
        while pos < end:
            op = resource[pos]
            if op == 0:
                pos += 1
                continue
            if (op & 0xF8) not in VALID_PRIMS or pos + 3 > end:
                break
            count = struct.unpack_from(">H", resource, pos + 1)[0]
            pos += 3
            verts = []
            src_uvs: list[tuple[float, float] | None] = []
            for _ in range(count):
                try:
                    vi, uv, uv_idx, pos = read_record(resource, pos, mode)
                except Exception:
                    ok = False
                    break
                if pos > end:
                    ok = False
                    break
                if vi >= desc["vertex_count"]:
                    ok = False
                    break
                verts.append(vi)
                if uv is not None:
                    src_uvs.append(uv)
                elif uv_idx is not None and 0 <= uv_idx < len(uv_table):
                    src_uvs.append(uv_table[uv_idx])
                else:
                    src_uvs.append(None)
            if not ok:
                break
            add_triangles_from_primitive(op, verts, src_uvs, faces, triangle_corner_uvs)
        score = len(faces)
        if ok and score > best[3]:
            best = (faces, {i: uv for i, uv in enumerate(triangle_corner_uvs) if uv is not None}, mode, score)
    return best[0], best[1], best[2]


def fallback_uv(index: int) -> tuple[float, float]:
    return ((index % 32) / 31.0, 1.0 - ((index // 32) % 32) / 31.0)


def unwrap_triangle_uvs(tri: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Move triangle UV corners by whole tiles to avoid false wrap seams.

    The game samples with repeat/wrap behavior, but Blender's UV editor draws a
    straight line through the 0..1 box. Keep each triangle in the closest
    repeated tile so body islands do not turn into a web of seam lines.
    """
    if len(tri) != 3:
        return tri
    raw_edges = [math.dist(tri[a], tri[b]) for a, b in ((0, 1), (1, 2), (2, 0))]
    if max(raw_edges) < 0.45:
        return tri
    shifts = (-1.0, 0.0, 1.0)
    # Only relative tile shifts matter for a triangle.
    best_score = float("inf")
    best_tri = tri
    for du1 in (-1.0, 0.0, 1.0):
        for dv1 in (-1.0, 0.0, 1.0):
            for du2 in (-1.0, 0.0, 1.0):
                for dv2 in (-1.0, 0.0, 1.0):
                    candidate = [
                        tri[0],
                        (tri[1][0] + du1, tri[1][1] + dv1),
                        (tri[2][0] + du2, tri[2][1] + dv2),
                    ]
                    edges = [math.dist(candidate[a], candidate[b]) for a, b in ((0, 1), (1, 2), (2, 0))]
                    # Prefer compact triangles when unwrapping seam crossings.
                    shifted = abs(du1) + abs(dv1) + abs(du2) + abs(dv2)
                    score = max(edges) * 10.0 + sum(edges) + shifted * 0.001
                    if score < best_score:
                        best_tri = candidate
                        best_score = score
    best_edges = [math.dist(best_tri[a], best_tri[b]) for a, b in ((0, 1), (1, 2), (2, 0))]
    if max(best_edges) + 0.01 >= max(raw_edges):
        return tri
    return best_tri


def squash_long_triangle_uvs(tri: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if len(tri) != 3:
        return tri
    edges = [math.dist(tri[a], tri[b]) for a, b in ((0, 1), (1, 2), (2, 0))]
    longest = max(edges)
    if longest <= 0.18:
        return tri
    (x1, y1), (x2, y2), (x3, y3) = tri
    area = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)) * 0.5
    if area / (longest * longest) >= 0.015:
        return tri
    center = (
        max(0.0, min(1.0, sum(u for u, _v in tri) / 3.0)),
        max(0.0, min(1.0, sum(v for _u, v in tri) / 3.0)),
    )
    return [center, center, center]


def remove_origin_marker_faces(
    vertices: list[tuple[float, float, float]],
    faces: list[tuple[int, int, int]],
) -> tuple[list[tuple[int, int, int]], int, list[int]]:
    """Drop CMG no-draw/normal-marker spikes from Blender export.

    Some tiny CMG chunks keep normal-like unit vectors in the same local blob as
    positions. Their display list can reference those marker entries, producing
    long triangles back near origin in Blender. Preserve the vertices for stable
    same-topology import, but do not emit faces that touch marker entries.
    """
    if not faces or not vertices:
        return faces, 0, list(range(len(faces)))
    lens = []
    for a, b, c in faces:
        for u, v in ((a, b), (b, c), (c, a)):
            if 0 <= u < len(vertices) and 0 <= v < len(vertices):
                lens.append(math.dist(vertices[u], vertices[v]))
    if not lens or max(lens) < 20.0:
        return faces, 0, list(range(len(faces)))
    marker = {
        i
        for i, p in enumerate(vertices)
        if math.sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]) < 3.0
    }
    if not marker:
        return faces, 0, list(range(len(faces)))
    kept = []
    kept_indices = []
    for fi, face in enumerate(faces):
        if any(i in marker for i in face):
            continue
        kept.append(face)
        kept_indices.append(fi)
    return kept, len(faces) - len(kept), kept_indices


def parse_cmg_skeleton(parser: PipeworksParser, entries: list[dict], scale: float) -> tuple[list[dict], dict[int, list[list[float]]]]:
    data = parser.file_data or b""
    strings = bundle_strings(parser)
    skel = next((e for e in entries if e["file_type"] == 3 and "SKELETON" in e["name"].upper()), None)
    pose = next((e for e in entries if e["file_type"] == 4 and "SKELETON" in e["name"].upper()), None)
    if not skel:
        return [], {}
    blob = data[skel["offset"] : skel["offset"] + skel["size"]]
    bone_count = be32(blob, 0x20) if len(blob) >= 0x24 else 0
    bones: dict[int, dict] = {}
    pose_by_index: dict[int, tuple[tuple[float, float, float], tuple[float, float, float, float]]] = {}

    if pose:
        pblob = data[pose["offset"] : pose["offset"] + pose["size"]]
        pcount = be32(pblob, 0x2C) if len(pblob) >= 0x30 else bone_count
        offsets = []
        for i in range(max(0, min(pcount, 512))):
            pos = 0x38 + i * 4
            if pos + 4 <= len(pblob):
                rel = be32(pblob, pos)
                if 0 <= rel + 36 <= len(pblob):
                    offsets.append(rel)
        for rel in offsets:
            try:
                idx = sbe32(pblob, rel)
                t = struct.unpack_from(">3f", pblob, rel + 0x18)
                qx, qy, qz, qw = normalize_quat(struct.unpack_from(">4f", pblob, rel + 0x30))
                q = (-qx, -qy, -qz, qw)
                pose_by_index[idx] = (t, q)
            except Exception:
                continue

    def walk(rel: int) -> None:
        if rel < 0 or rel + 16 > len(blob):
            return
        idx = sbe32(blob, rel + 0)
        parent = sbe32(blob, rel + 4)
        child_count = sbe32(blob, rel + 8)
        name_idx = sbe32(blob, rel + 12)
        if idx in bones or idx < 0 or idx > 512 or child_count < 0 or child_count > 64:
            return
        t, q = pose_by_index.get(idx, ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)))
        name = strings[name_idx] if 0 <= name_idx < len(strings) else f"Bone_{idx:02d}"
        children = []
        for i in range(child_count):
            coff = rel + 16 + i * 4
            if coff + 4 <= len(blob):
                children.append(be32(blob, coff))
        bones[idx] = {"idx": idx, "parent": parent, "name": name, "name_idx": name_idx, "q": q, "t": t, "children_rel": children, "display_size": 0.35}
        for child_rel in children:
            walk(child_rel)

    walk(0x40)
    ordered = [bones[i] for i in sorted(bones)]
    return ordered, global_matrices(ordered, scale)


def qmat(q: tuple[float, float, float, float]) -> list[list[float]]:
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return [[1 - yy - zz, xy - wz, xz + wy], [xy + wz, 1 - xx - zz, yz - wx], [xz - wy, yz + wx, 1 - xx - yy]]


def matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def local_matrix(bone: dict, scale: float) -> list[list[float]]:
    r = qmat(bone["q"])
    x, y, z = bone["t"]
    return [[r[0][0], r[0][1], r[0][2], x * scale], [r[1][0], r[1][1], r[1][2], y * scale], [r[2][0], r[2][1], r[2][2], z * scale], [0, 0, 0, 1]]


def global_matrices(bones: list[dict], scale: float) -> dict[int, list[list[float]]]:
    by_idx = {b["idx"]: b for b in bones}
    out: dict[int, list[list[float]]] = {}

    def comp(idx: int) -> list[list[float]]:
        if idx in out:
            return out[idx]
        b = by_idx[idx]
        m = local_matrix(b, scale)
        if b["parent"] in by_idx:
            m = matmul(comp(b["parent"]), m)
        out[idx] = m
        return m

    for b in bones:
        comp(b["idx"])
    return out


def matrix_fbx(m: list[list[float]]) -> list[float]:
    return [m[0][0], m[1][0], m[2][0], 0, m[0][1], m[1][1], m[2][1], 0, m[0][2], m[1][2], m[2][2], 0, m[0][3], m[1][3], m[2][3], 1]


def quat_to_euler(q: tuple[float, float, float, float]) -> tuple[float, float, float]:
    x, y, z, w = q
    sinr = 2 * (w * x + y * z)
    cosr = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr, cosr)
    sinp = 2 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
    siny = 2 * (w * z + x * y)
    cosy = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny, cosy)
    return (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))


def normalized_quaternion(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    length = math.sqrt(sum(float(value) * float(value) for value in q)) or 1.0
    return tuple(float(value) / length for value in q)


def quaternion_dot(a, b) -> float:
    return sum(float(a[index]) * float(b[index]) for index in range(4))


def quaternion_multiply(a, b):
    ax, ay, az, aw = normalized_quaternion(a)
    bx, by, bz, bw = normalized_quaternion(b)
    return normalized_quaternion(
        (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        )
    )


def quaternion_inverse(value):
    x, y, z, w = normalized_quaternion(value)
    return (-x, -y, -z, w)


def stabilize_root_quaternions(values, bind_quaternion):
    if not values:
        return values
    first_inverse = quaternion_inverse(values[0])
    bind = normalized_quaternion(bind_quaternion)
    return [
        quaternion_multiply(bind, quaternion_multiply(first_inverse, value))
        for value in values
    ]


def quaternion_slerp(a, b, amount: float):
    left = normalized_quaternion(a)
    right = normalized_quaternion(b)
    dot = quaternion_dot(left, right)
    if dot < 0.0:
        right = tuple(-value for value in right)
        dot = -dot
    dot = max(-1.0, min(1.0, dot))
    if dot > 0.9995:
        return normalized_quaternion(
            tuple(left[index] + amount * (right[index] - left[index]) for index in range(4))
        )
    theta = math.acos(dot)
    sine = math.sin(theta)
    left_scale = math.sin((1.0 - amount) * theta) / sine
    right_scale = math.sin(amount * theta) / sine
    return tuple(left[index] * left_scale + right[index] * right_scale for index in range(4))


def unwrap_euler_keys(values: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
    if not values:
        return []
    result = [list(values[0])]
    for value in values[1:]:
        current = list(value)
        previous = result[-1]
        for axis in range(3):
            while current[axis] - previous[axis] > 180.0:
                current[axis] -= 360.0
            while current[axis] - previous[axis] < -180.0:
                current[axis] += 360.0
        result.append(current)
    return [tuple(value) for value in result]


def dense_animation_times(duration: float, fps: float = 60.0) -> list[float]:
    duration = max(0.0, float(duration))
    if duration <= 1e-8:
        return [0.0]
    count = max(1, int(math.ceil(duration * fps)))
    times = [min(duration, index / fps) for index in range(count + 1)]
    if times[-1] < duration - 1e-8:
        times.append(duration)
    else:
        times[-1] = duration
    return times


def sample_animation_keys(keys: list[tuple[float, object]], times: list[float], quaternion=False):
    if not keys:
        return []
    ordered = sorted(keys, key=lambda item: float(item[0]))
    clean = []
    for time, value in ordered:
        if clean and float(time) <= clean[-1][0] + 1e-8:
            continue
        clean.append((float(time), value))
    if len(clean) == 1:
        return [clean[0][1] for _time in times]
    result = []
    key_index = 0
    for time in times:
        while key_index + 1 < len(clean) and clean[key_index + 1][0] < time - 1e-8:
            key_index += 1
        if time <= clean[0][0]:
            result.append(clean[0][1])
            continue
        if key_index + 1 >= len(clean):
            result.append(clean[-1][1])
            continue
        left_time, left_value = clean[key_index]
        right_time, right_value = clean[key_index + 1]
        amount = (time - left_time) / max(1e-12, right_time - left_time)
        amount = max(0.0, min(1.0, amount))
        if quaternion:
            result.append(quaternion_slerp(left_value, right_value, amount))
        else:
            result.append(
                tuple(
                    float(left_value[axis])
                    + amount * (float(right_value[axis]) - float(left_value[axis]))
                    for axis in range(3)
                )
            )
    return result


def cmg_animation_track_times_valid(times: list[int]) -> bool:
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
        if second + 16 < first
    ]
    if not drops:
        return all(second > first for first, second in zip(times, times[1:]))
    return (
        len(drops) == 1
        and drops[0] == len(times) - 2
        and times[-2] >= 55000
        and times[-1] <= 30000
        and all(second > first for first, second in zip(times[:-2], times[1:-1]))
    )


def parse_cmg_animation_translation_tracks(blob: bytes) -> list[dict] | None:
    if len(blob) < 0x44 or be32(blob, 0x20) != 2:
        return None
    bone_count = be32(blob, 0x2C)
    section_start = be32(blob, 0x38)
    section_end = be32(blob, 0x3C)
    if (
        bone_count <= 0
        or bone_count > 512
        or section_start < 0x44
        or section_end <= section_start
        or section_end > len(blob)
    ):
        return None

    def walk_tracks(pos: int, previous_bone: int, allow_interstitial: bool) -> list[dict] | None:
        if previous_bone == bone_count - 1:
            return []
        candidates = []
        for scale_count in (3, 1):
            header_pos = pos + scale_count * 4
            if header_pos + 4 > section_end:
                continue
            bone, key_count, last_key, flags = struct.unpack_from(">BBBB", blob, header_pos)
            records_end = header_pos + 4 + key_count * 8
            if previous_bone < bone < bone_count and key_count > 0 and records_end <= section_end:
                candidates.append(
                    (scale_count, int(bone), int(key_count), int(last_key), int(flags), records_end)
                )

        for scale_count, bone, key_count, last_key, flags, records_end in candidates:
            remainder = walk_tracks(records_end, bone, allow_interstitial)
            if remainder is None:
                continue
            raw_scales = struct.unpack_from(f">{scale_count}f", blob, pos)
            scales = raw_scales if scale_count == 3 else (raw_scales[0],) * 3
            return [
                {
                    "bone": bone,
                    "scale_pos": pos,
                    "scale_count": scale_count,
                    "scales": tuple(float(value) for value in scales),
                    "records_pos": pos + scale_count * 4 + 4,
                    "key_count": key_count,
                    "last_key": last_key,
                    "flags": flags,
                },
                *remainder,
            ]

        if allow_interstitial and previous_bone >= 0:
            for resume_pos in range(pos + 2, section_end - 7, 2):
                for scale_count in (3, 1):
                    header_pos = resume_pos + scale_count * 4
                    if header_pos + 4 > section_end:
                        continue
                    bone, key_count, _last_key, _flags = struct.unpack_from(">BBBB", blob, header_pos)
                    records_end = header_pos + 4 + key_count * 8
                    if bone == previous_bone + 1 and key_count > 0 and records_end <= section_end:
                        resumed = walk_tracks(resume_pos, previous_bone, False)
                        if resumed is not None:
                            return resumed
        return None

    tracks = walk_tracks(section_start, -1, True)
    return tracks if tracks else None


def decode_cmg_animation_rotation_track(
    blob: bytes,
    rel: int,
    section_end: int,
    bone_count: int,
) -> dict | None:
    if rel < 0 or rel + 4 > section_end:
        return None
    bone, record_count, zero = struct.unpack_from(">BBH", blob, rel)
    records_pos = rel + 4
    records_end = records_pos + 6 + (record_count - 1) * 8
    if (
        zero != 0
        or bone >= bone_count
        or record_count <= 0
        or records_end > section_end
    ):
        return None
    records = []
    qx, qy, qz = struct.unpack_from(">hhh", blob, records_pos)
    records.append((0, int(qx), int(qy), int(qz)))
    for index in range(1, record_count):
        record_pos = records_pos + 6 + (index - 1) * 8
        time, qx, qy, qz = struct.unpack_from(">Hhhh", blob, record_pos)
        if sum((value / 32767.0) ** 2 for value in (qx, qy, qz)) > 1.05:
            return None
        records.append((int(time), int(qx), int(qy), int(qz)))
    if sum((value / 32767.0) ** 2 for value in records[0][1:]) > 1.05:
        return None
    times = [record[0] for record in records]
    if not cmg_animation_track_times_valid(times):
        return None
    return {
        "bone": int(bone),
        "rel": rel,
        "end": records_end,
        "records_pos": records_pos,
        "record_count": int(record_count),
        "records": records,
        "layout": "explicit_qxyz_then_time_qxyz",
    }


def decode_cmg_animation_rotation_continuation(
    blob: bytes,
    start: int,
    end: int,
    bone: int,
    bone_count: int,
) -> dict | None:
    if start < 0 or start >= end or end > len(blob) or (end - start) % 8 or not (0 <= bone < bone_count):
        return None
    record_count = (end - start) // 8
    if record_count <= 0 or record_count > 255:
        return None
    times = []
    for record_pos in range(start, end, 8):
        time, qx, qy, qz = struct.unpack_from(">Hhhh", blob, record_pos)
        if sum((value / 32767.0) ** 2 for value in (qx, qy, qz)) > 1.05:
            return None
        times.append(int(time))
    if not cmg_animation_track_times_valid(times):
        return None
    return {
        "bone": int(bone),
        "rel": start,
        "end": end,
        "records_pos": start,
        "record_count": record_count,
        "layout": "continuation_time_qxyz",
    }


def best_cmg_rotation_continuation(
    blob: bytes,
    gap_start: int,
    gap_end: int,
    bone: int,
    bone_count: int,
) -> dict | None:
    best = None
    for lead in (0, 2, 4, 6):
        for remainder in (0, 2, 4, 6):
            track = decode_cmg_animation_rotation_continuation(
                blob, gap_start + lead, gap_end - remainder, bone, bone_count
            )
            if track is not None and (
                best is None or int(track["record_count"]) > int(best["record_count"])
            ):
                best = track
    return best


def cmg_terminal_bone_table_start(
    blob: bytes,
    search_start: int,
    section_end: int,
    bone_count: int,
) -> int | None:
    for rel in range(max(search_start, section_end - 0x400), section_end - 7, 2):
        ids = []
        pos = rel
        while pos + 2 <= section_end:
            bone, zero = blob[pos], blob[pos + 1]
            if zero != 0 or bone >= bone_count or (ids and bone == 0):
                break
            ids.append(int(bone))
            pos += 2
        if len(ids) >= 4 and all(ids[index] + 1 == ids[index + 1] for index in range(len(ids) - 1)):
            return rel
    return None


def parse_cmg_animation_rotation_tracks(blob: bytes) -> list[dict]:
    if len(blob) < 0x44 or be32(blob, 0x20) != 2:
        return []
    bone_count = be32(blob, 0x2C)
    section_start = be32(blob, 0x3C)
    declared_size = be32(blob, 0x24)
    if bone_count <= 0 or bone_count > 512 or section_start < 0x40 or section_start >= len(blob):
        return []
    boundaries = sorted(
        {
            value
            for value in (be32(blob, 0x40), declared_size, len(blob))
            if section_start < value <= len(blob)
        }
    )
    best_tracks: list[dict] = []
    for section_end in boundaries:
        candidates = []
        for rel in range(section_start, section_end - 3, 2):
            track = decode_cmg_animation_rotation_track(blob, rel, section_end, bone_count)
            if track is not None:
                candidates.append(track)

        by_bone: dict[int, list[dict]] = collections.defaultdict(list)
        for track in candidates:
            by_bone[int(track["bone"])].append(track)
        chain_cache: dict[tuple[int, int], list[dict]] = {}

        def explicit_chain(track: dict) -> list[dict]:
            key = (int(track["rel"]), int(track["bone"]))
            if key in chain_cache:
                return chain_cache[key]
            choices = [[track]]
            next_bone = int(track["bone"]) + 1
            for next_track in by_bone.get(next_bone, []):
                if int(next_track["rel"]) == int(track["end"]):
                    choices.append([track, *explicit_chain(next_track)])
            skipped_bone = next_bone
            for next_track in by_bone.get(int(track["bone"]) + 2, []):
                if int(next_track["rel"]) < int(track["end"]):
                    continue
                continuation = best_cmg_rotation_continuation(
                    blob,
                    int(track["end"]),
                    int(next_track["rel"]),
                    skipped_bone,
                    bone_count,
                )
                if continuation is not None:
                    choices.append([track, continuation, *explicit_chain(next_track)])
            result = max(
                choices,
                key=lambda chain: (
                    len(chain),
                    int(chain[-1]["bone"]),
                    sum(int(item["record_count"]) for item in chain),
                ),
            )
            chain_cache[key] = result
            return result

        starts = [track for track in by_bone.get(0, []) if int(track["rel"]) == section_start]
        expanded = max(
            (explicit_chain(track) for track in starts),
            key=lambda chain: (
                len(chain),
                int(chain[-1]["bone"]),
                sum(int(item["record_count"]) for item in chain),
            ),
            default=[],
        )

        if expanded:
            final_track = expanded[-1]
            table_start = cmg_terminal_bone_table_start(
                blob, int(final_track["end"]), section_end, bone_count
            )
            terminal_end = table_start if table_start is not None else section_end
            if int(final_track["bone"]) + 1 < bone_count and terminal_end > int(final_track["end"]):
                continuation = best_cmg_rotation_continuation(
                    blob,
                    int(final_track["end"]),
                    terminal_end,
                    int(final_track["bone"]) + 1,
                    bone_count,
                )
                if continuation is not None:
                    expanded.append(continuation)

        if (len(expanded), sum(int(item["record_count"]) for item in expanded)) > (
            len(best_tracks),
            sum(int(item["record_count"]) for item in best_tracks),
        ):
            best_tracks = expanded

    unique = []
    seen_bones = set()
    for track in best_tracks:
        bone = int(track["bone"])
        if bone in seen_bones:
            continue
        seen_bones.add(bone)
        unique.append(track)
    return unique


def cmg_animation_seconds(time: int, duration: float) -> float:
    timestamp = int(time) & 0xFFFE
    return max(0.0, min(float(duration), float(timestamp) / 65534.0 * float(duration)))


def ordered_cmg_animation_keys(keys: list[tuple[float, object]]) -> list[tuple[float, object]]:
    ordered = sorted(keys, key=lambda item: float(item[0]))
    result = []
    for time, value in ordered:
        if result and float(time) <= result[-1][0] + 1e-8:
            continue
        result.append((float(time), value))
    return result


def cmg_quaternion_keys(
    records: list[tuple[int, int, int, int]],
    duration: float,
    rest_quaternion: tuple[float, float, float, float],
) -> list[tuple[float, tuple[float, float, float, float]]]:
    result = []
    previous = None
    rest = normalize_quat(rest_quaternion)
    for time, raw_x, raw_y, raw_z in records:
        x, y, z = (-float(value) / 32767.0 for value in (raw_x, raw_y, raw_z))
        w = math.sqrt(max(0.0, 1.0 - x * x - y * y - z * z))
        if int(time) & 1:
            w = -w
        quaternion = normalize_quat((x, y, z, w))
        reference = previous if previous is not None else rest
        if quaternion_dot(quaternion, reference) < 0.0:
            quaternion = tuple(-value for value in quaternion)
        previous = quaternion
        result.append((cmg_animation_seconds(time, duration), quaternion))
    return ordered_cmg_animation_keys(result)


def decode_cmg_type4_animations(
    data: bytes,
    entries: list[dict],
    bones: list[dict],
    scale: float,
) -> tuple[list[dict], dict]:
    bone_by_index = {int(bone["idx"]): bone for bone in bones}
    animations = []
    skipped_bodyless = []
    malformed_translation = []
    translation_track_count = 0
    rotation_track_count = 0
    for entry in entries:
        if entry["file_type"] != 4 or entry["is_resource"]:
            continue
        base = int(entry["offset"])
        size = int(entry["size"])
        blob = data[base : base + size]
        if len(blob) < 0x44 or be32(blob, 0x20) != 2:
            continue
        if be32(blob, 0x3C) > len(blob):
            skipped_bodyless.append(str(entry["name"]))
            continue
        duration = fbe(blob, 0x1C)
        if not math.isfinite(duration) or duration <= 0.0 or duration > 120.0:
            continue

        tracks_by_bone: dict[int, dict] = {}
        translation_tracks = parse_cmg_animation_translation_tracks(blob)
        if translation_tracks is None:
            malformed_translation.append(str(entry["name"]))
            translation_tracks = []
        for track in translation_tracks:
            bone = int(track["bone"])
            if bone not in bone_by_index:
                continue
            scales = tuple(float(value) for value in track["scales"])
            keys = []
            for key_index in range(int(track["key_count"])):
                record_pos = int(track["records_pos"]) + key_index * 8
                time, x, y, z = struct.unpack_from(">Hhhh", blob, record_pos)
                value = tuple(
                    float(raw) / 32767.0 * scales[axis] * scale
                    for axis, raw in enumerate((x, y, z))
                )
                keys.append((cmg_animation_seconds(time, duration), value))
            ordered = ordered_cmg_animation_keys(keys)
            if ordered:
                tracks_by_bone.setdefault(bone, {"bone": bone})["translation_keys"] = ordered
                translation_track_count += 1

        for track in parse_cmg_animation_rotation_tracks(blob):
            bone = int(track["bone"])
            if bone not in bone_by_index:
                continue
            records = list(track.get("records") or [])
            if not records:
                for record_index in range(int(track["record_count"])):
                    record_pos = int(track["records_pos"]) + record_index * 8
                    time, x, y, z = struct.unpack_from(">Hhhh", blob, record_pos)
                    records.append((int(time), int(x), int(y), int(z)))
            keys = cmg_quaternion_keys(records, duration, tuple(bone_by_index[bone]["q"]))
            if keys:
                tracks_by_bone.setdefault(bone, {"bone": bone})["rotation_keys"] = keys
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
        "malformed_translation_clips": malformed_translation,
        "sampling": "native big-endian Type-4 keys decoded and quaternion-sampled to FBX curves at 60 fps",
    }


class Prop:
    def __init__(self, code: str, value):
        self.code = code
        self.value = value


class Arr:
    def __init__(self, code: str, values):
        self.code = code
        self.values = list(values)


class Node:
    def __init__(self, name: str, props=None, children=None):
        self.name = name
        self.props = props or []
        self.children = children or []


def PInt(v): return Prop("I", int(v))
def PLong(v): return Prop("L", int(v))
def PDouble(v): return Prop("D", float(v))
def PBool(v): return Prop("C", bool(v))
def PStr(v): return Prop("S", str(v))
def PRaw(v): return Prop("R", bytes(v))
def ADouble(v): return Arr("d", v)
def AFloat(v): return Arr("f", v)
def AInt(v): return Arr("i", v)
def ALong(v): return Arr("l", v)


def PObjectName(name: str, object_class: str) -> Prop:
    return PStr(f"{name}\x00\x01{object_class}")


def pack_prop(p):
    if isinstance(p, Prop):
        c = p.code.encode()
        if p.code == "I":
            return c + struct.pack("<i", p.value)
        if p.code == "L":
            return c + struct.pack("<q", p.value)
        if p.code == "D":
            return c + struct.pack("<d", p.value)
        if p.code == "C":
            return c + (b"\x01" if p.value else b"\x00")
        if p.code == "S":
            b = str(p.value).encode("utf-8")
            return c + struct.pack("<I", len(b)) + b
        if p.code == "R":
            return c + struct.pack("<I", len(p.value)) + p.value
    if isinstance(p, Arr):
        vals = p.values
        c = p.code.encode()
        if p.code == "d":
            raw = struct.pack("<%sd" % len(vals), *map(float, vals)) if vals else b""
        elif p.code == "f":
            raw = struct.pack("<%sf" % len(vals), *map(float, vals)) if vals else b""
        elif p.code == "i":
            raw = struct.pack("<%si" % len(vals), *map(int, vals)) if vals else b""
        elif p.code == "l":
            raw = struct.pack("<%sq" % len(vals), *map(int, vals)) if vals else b""
        else:
            raise ValueError(p.code)
        return c + struct.pack("<III", len(vals), 0, len(raw)) + raw
    raise TypeError(type(p))


NULL_RECORD = b"\0" * 13


def write_node(buf: io.BytesIO, node: Node) -> None:
    start = buf.tell()
    props = b"".join(pack_prop(p) for p in node.props)
    name = node.name.encode("ascii")
    buf.write(b"\0" * 12)
    buf.write(bytes([len(name)]))
    buf.write(name)
    buf.write(props)
    for child in node.children:
        write_node(buf, child)
    if node.children:
        buf.write(NULL_RECORD)
    end = buf.tell()
    cur = end
    buf.seek(start)
    buf.write(struct.pack("<III", end, len(node.props), len(props)))
    buf.write(bytes([len(name)]))
    buf.write(name)
    buf.seek(cur)


def p_node(name, ptype, label, flags, *values):
    props = [PStr(name), PStr(ptype), PStr(label), PStr(flags)]
    for value in values:
        if isinstance(value, bool):
            props.append(PBool(value))
        elif isinstance(value, int):
            props.append(PInt(value))
        elif isinstance(value, float):
            props.append(PDouble(value))
        else:
            props.append(PStr(value))
    return Node("P", props)


def flat3(vals):
    out = []
    for a, b, c in vals:
        out += [a, b, c]
    return out


def flat2(vals):
    out = []
    for a, b in vals:
        out += [a, b]
    return out


def identity():
    return [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]


def mat_pos(m: list[list[float]]) -> tuple[float, float, float]:
    return (float(m[0][3]), float(m[1][3]), float(m[2][3]))


def dist_to_segment_sq(p, a, b) -> float:
    ax, ay, az = a
    bx, by, bz = b
    px, py, pz = p
    vx, vy, vz = bx - ax, by - ay, bz - az
    wx, wy, wz = px - ax, py - ay, pz - az
    d = vx * vx + vy * vy + vz * vz
    if d <= 1e-8:
        dx, dy, dz = px - ax, py - ay, pz - az
        return dx * dx + dy * dy + dz * dz
    t = max(0.0, min(1.0, (wx * vx + wy * vy + wz * vz) / d))
    qx, qy, qz = ax + vx * t, ay + vy * t, az + vz * t
    dx, dy, dz = px - qx, py - qy, pz - qz
    return dx * dx + dy * dy + dz * dz


def skeleton_segments(bones: list[dict], globals_: dict[int, list[list[float]]]) -> list[tuple[int, tuple[float, float, float], tuple[float, float, float]]]:
    if not bones:
        return []
    children: dict[int, list[int]] = {}
    by_idx = {b["idx"]: b for b in bones}
    for b in bones:
        children.setdefault(b["parent"], []).append(b["idx"])
    positions = {idx: mat_pos(globals_[idx]) for idx in by_idx if idx in globals_}
    segments = []
    for b in bones:
        idx = b["idx"]
        if idx not in positions:
            continue
        child_ids = [c for c in children.get(idx, []) if c in positions]
        if child_ids:
            for child in child_ids:
                segments.append((idx, positions[idx], positions[child]))
        elif b["parent"] in positions:
            segments.append((idx, positions[b["parent"]], positions[idx]))
        else:
            segments.append((idx, positions[idx], positions[idx]))
    return segments


def submesh_center(submesh: dict) -> tuple[float, float, float] | None:
    verts = submesh.get("vertices") or []
    if not verts:
        return None
    return (
        sum(p[0] for p in verts) / len(verts),
        sum(p[1] for p in verts) / len(verts),
        sum(p[2] for p in verts) / len(verts),
    )


def dominant_bone_for_submesh(submesh: dict, bones: list[dict], globals_: dict[int, list[list[float]]]) -> int | None:
    center = submesh_center(submesh)
    segments = skeleton_segments(bones, globals_)
    if center is None or not segments:
        return None
    return min(segments, key=lambda seg: dist_to_segment_sq(center, seg[1], seg[2]))[0]


def build_weighted_skin_clusters(submesh: dict, bone_count: int) -> dict[int, tuple[list[int], list[float]]]:
    vertex_weights = submesh.get("vertex_weights") or []
    if not vertex_weights:
        return {}
    cluster_indices: dict[int, list[int]] = {}
    cluster_weights: dict[int, list[float]] = {}
    for vi, weights in enumerate(vertex_weights):
        total = sum(float(w) for _bone, w in weights)
        if total <= 1e-8:
            continue
        for bone_idx, weight in weights:
            bone_idx = int(bone_idx)
            if 0 <= bone_idx < bone_count and weight > 1e-8:
                cluster_indices.setdefault(bone_idx, []).append(vi)
                cluster_weights.setdefault(bone_idx, []).append(float(weight) / total)
    return {idx: (cluster_indices[idx], cluster_weights[idx]) for idx in cluster_indices}


def build_packet_rigid_skin_clusters(submesh: dict, bones: list[dict], globals_: dict[int, list[list[float]]]) -> dict[int, tuple[list[int], list[float]]]:
    bone_idx = dominant_bone_for_submesh(submesh, bones, globals_)
    if bone_idx is None:
        return {}
    count = len(submesh.get("vertices") or [])
    if count <= 0:
        return {}
    submesh["skin_mode"] = "packet_rigid"
    submesh["skin_bone"] = bone_idx
    return {bone_idx: (list(range(count)), [1.0] * count)}


def build_auto_skin_clusters(submesh: dict, bones: list[dict], globals_: dict[int, list[list[float]]]) -> dict[int, tuple[list[int], list[float]]]:
    segments = skeleton_segments(bones, globals_)
    if not segments:
        return {}
    jaw_bone = next((bone for bone in bones if "jaw" in str(bone["name"]).lower()), None)
    jaw_index = int(jaw_bone["idx"]) if jaw_bone is not None else None
    parent_point = None
    jaw_point = None
    if (
        jaw_bone is not None
        and int(jaw_bone["parent"]) in globals_
        and jaw_index in globals_
    ):
        parent_index = int(jaw_bone["parent"])
        parent_point = tuple(globals_[parent_index][axis][3] for axis in range(3))
        jaw_point = tuple(globals_[jaw_index][axis][3] for axis in range(3))
        segments = [
            segment
            for segment in segments
            if not (
                int(segment[0]) == parent_index
                and segment[1] == parent_point
                and segment[2] == jaw_point
            )
        ]
        segments.insert(0, (jaw_index, parent_point, jaw_point))
    submesh["skin_mode"] = "auto_nearest_with_jaw_segment"
    assignments = [
        min(segments, key=lambda seg: dist_to_segment_sq(point, seg[1], seg[2]))[0]
        for point in submesh["vertices"]
    ]
    cluster_indices: dict[int, list[int]] = {}
    cluster_weights: dict[int, list[float]] = {}
    for vi, bone_idx in enumerate(assignments):
        cluster_indices.setdefault(bone_idx, []).append(vi)
        cluster_weights.setdefault(bone_idx, []).append(1.0)
    if jaw_index is not None and len(cluster_indices.get(jaw_index, [])) > len(submesh["vertices"]) / 2:
        count = len(submesh["vertices"])
        cluster_indices = {jaw_index: list(range(count))}
        cluster_weights = {jaw_index: [1.0] * count}
        submesh["skin_mode"] = "jaw_rigid_submesh"
    return {idx: (cluster_indices[idx], cluster_weights[idx]) for idx in cluster_indices}


def cmg_mesh_bone_palette(main: bytes, bones: list[dict]) -> list[int]:
    """Resolve the Type-17 mesh's palette of skeleton-name IDs to bone indices."""
    if len(main) < 4:
        return []
    palette_count = be16(main, 2)
    if palette_count <= 0 or palette_count > 512 or 4 + palette_count * 4 > len(main):
        return []
    bone_by_name_id = {int(bone.get("name_idx", -1)): int(bone["idx"]) for bone in bones}
    palette = []
    for index in range(palette_count):
        name_id = be32(main, 4 + index * 4)
        if name_id not in bone_by_name_id:
            return []
        palette.append(bone_by_name_id[name_id])
    return palette


def cmg_descriptor_skin_ranges(main: bytes, desc: dict, next_desc_off: int, palette_count: int) -> list[tuple[int, int, int]]:
    """Read contiguous vertex counts and their one- or two-joint palette channels."""
    start = int(desc["main_offset"]) + 0x4C
    end = max(start, min(next_desc_off, len(main)))
    vertex_count = int(desc.get("vertex_count", 0))
    remaining = vertex_count
    ranges: list[tuple[int, int, int]] = []
    for off in range(start, end - 3, 4):
        word = be32(main, off)
        count = word >> 16
        palette_a = (word >> 8) & 0xFF
        palette_b = word & 0xFF
        if count <= 0 or count > remaining:
            return []
        if not (0 <= palette_a < palette_count and 0 <= palette_b < palette_count):
            return []
        ranges.append((count, palette_a, palette_b))
        remaining -= count
        if remaining == 0:
            break
    if vertex_count <= 0 or remaining != 0:
        return []
    return ranges


def cmg_vertex_weights_from_ranges(
    resource: bytes,
    desc: dict,
    ranges: list[tuple[int, int, int]],
    palette: list[int],
) -> list[list[tuple[int, float]]]:
    """Decode rigid ranges and the two BE floats stored for each blended vertex."""
    vertex_count = int(desc.get("vertex_count", 0))
    if not ranges or sum(count for count, _a, _b in ranges) != vertex_count:
        return []
    weight_pos = int(desc["rel_vertex"]) + vertex_count * position_stride(desc)
    sections = desc.get("section_offsets") or []
    weight_end = (
        int(desc["rel_vertex"]) + int(sections[12])
        if len(sections) > 12 and sections[12] not in (0, 0xFFFFFFFF)
        else len(resource)
    )
    weights: list[list[tuple[int, float]]] = []
    for count, palette_a, palette_b in ranges:
        bone_a = palette[palette_a]
        bone_b = palette[palette_b]
        if bone_a == bone_b:
            weights.extend([[(bone_a, 1.0)] for _ in range(count)])
            continue
        for _ in range(count):
            if weight_pos + 8 > min(weight_end, len(resource)):
                return []
            weight_a, weight_b = struct.unpack_from(">2f", resource, weight_pos)
            weight_pos += 8
            total = weight_a + weight_b
            if (
                not math.isfinite(weight_a)
                or not math.isfinite(weight_b)
                or weight_a < -1e-5
                or weight_b < -1e-5
                or total <= 1e-8
            ):
                return []
            weights.append([(bone_a, max(0.0, weight_a) / total), (bone_b, max(0.0, weight_b) / total)])
    if weight_pos != weight_end or len(weights) != vertex_count:
        return []
    return weights


def make_mesh_nodes(asset: str, submesh: dict, mesh_id: int, geom_id: int) -> tuple[Node, Node]:
    name = submesh["name"]
    verts = submesh["vertices"]
    faces = submesh["faces"]
    vertex_normals = submesh.get("normals") or []
    poly = []
    normals = []
    uvs = []
    for a, b, c in faces:
        poly += [a, b, ~c]
        tri_uvs = [
            submesh["uvs"].get(len(uvs) + j, fallback_uv(vi))
            for j, vi in enumerate((a, b, c))
        ]
        if not submesh.get("preserve_triangle_uvs"):
            tri_uvs = squash_long_triangle_uvs(tri_uvs)
        for vi, uv in zip((a, b, c), tri_uvs):
            if 0 <= vi < len(vertex_normals):
                normals.append(vertex_normals[vi])
            else:
                normals.append((0.0, 0.0, 1.0))
            uvs.append(uv)
    geom = Node(
        "Geometry",
        [PLong(geom_id), PObjectName(f"{name}_Geometry", "Geometry"), PStr("Mesh")],
        [
            Node("Vertices", [ADouble(flat3(verts))]),
            Node("PolygonVertexIndex", [AInt(poly)]),
            Node("GeometryVersion", [PInt(124)]),
            Node("LayerElementNormal", [PInt(0)], [Node("Version", [PInt(101)]), Node("Name", [PStr("")]), Node("MappingInformationType", [PStr("ByPolygonVertex")]), Node("ReferenceInformationType", [PStr("Direct")]), Node("Normals", [ADouble(flat3(normals))])]),
            Node("LayerElementUV", [PInt(0)], [Node("Version", [PInt(101)]), Node("Name", [PStr("UVChannel_1")]), Node("MappingInformationType", [PStr("ByPolygonVertex")]), Node("ReferenceInformationType", [PStr("Direct")]), Node("UV", [ADouble(flat2(uvs))])]),
            Node("Layer", [PInt(0)], [Node("Version", [PInt(100)]), Node("LayerElement", children=[Node("Type", [PStr("LayerElementNormal")]), Node("TypedIndex", [PInt(0)])]), Node("LayerElement", children=[Node("Type", [PStr("LayerElementUV")]), Node("TypedIndex", [PInt(0)])])]),
        ],
    )
    model = Node(
        "Model",
        [PLong(mesh_id), PObjectName(name, "Model"), PStr("Mesh")],
        [
            Node("Version", [PInt(232)]),
            Node("Properties70", children=[p_node("Lcl Translation", "Lcl Translation", "", "A", 0.0, 0.0, 0.0), p_node("Lcl Rotation", "Lcl Rotation", "", "A", 0.0, 0.0, 0.0), p_node("Lcl Scaling", "Lcl Scaling", "", "A", 1.0, 1.0, 1.0)]),
            Node("Shading", [PBool(True)]),
            Node("Culling", [PStr("CullingOff")]),
        ],
    )
    return geom, model


def write_fbx(
    path: Path,
    asset: str,
    submeshes: list[dict],
    bones: list[dict],
    globals_: dict[int, list[list[float]]],
    skin_mode: str = "none",
    animations: list[dict] | None = None,
    self_contained_animation_preview: bool = False,
    animation_scale: float = 10.0,
) -> None:
    animations = list(animations or [])
    if self_contained_animation_preview and animations and bones:
        completed_animations = []
        for animation in animations:
            duration = max(0.0, float(animation.get("duration", 0.0)))
            tracks_by_bone = {
                int(track["bone"]): dict(track)
                for track in animation.get("tracks") or []
            }
            completed_tracks = []
            for bone in bones:
                bone_index = int(bone["idx"])
                track = tracks_by_bone.get(bone_index, {"bone": bone_index})
                if not track.get("translation_keys"):
                    rest_translation = tuple(
                        float(value) * animation_scale for value in bone["t"]
                    )
                    track["translation_keys"] = [
                        (0.0, rest_translation),
                        (max(duration, 1.0 / 60.0), rest_translation),
                    ]
                    track["_preview_constant_translation"] = True
                if not track.get("rotation_keys"):
                    rest_rotation = tuple(float(value) for value in bone["q"])
                    track["rotation_keys"] = [
                        (0.0, rest_rotation),
                        (max(duration, 1.0 / 60.0), rest_rotation),
                    ]
                    track["_preview_constant_rotation"] = True
                completed_tracks.append(track)
            completed = dict(animation)
            completed["tracks"] = completed_tracks
            completed_animations.append(completed)
        animations = completed_animations
    if animations and bones:
        rest_tracks = []
        for bone in bones:
            rest_tracks.append(
                {
                    "bone": int(bone["idx"]),
                    "translation_keys": [
                        (0.0, tuple(float(value) * animation_scale for value in bone["t"]))
                    ],
                    "rotation_keys": [
                        (0.0, tuple(float(value) for value in bone["q"]))
                    ],
                }
            )
        animations.insert(
            0,
            {
                "name": "000_REST_POSE",
                "duration": 1.0 / 60.0,
                "tracks": rest_tracks,
                "synthetic_rest": True,
            },
        )
    base = (int(hashlib.sha1(asset.encode("utf-8")).hexdigest()[:8], 16) % 1000000000) + 3100000000
    group_id = base + 1
    default_mat_id = base + 10
    bone_model_base = base + 1000
    bone_attr_base = base + 2000
    skin_base = base + 100000
    cluster_base = base + 200000
    texture_base = base + 300000
    video_base = base + 400000
    animation_base = base + 1000000
    objects_children = []
    connections = []
    definitions_count = 1
    deformer_count = 0
    material_defs: list[dict] = []
    material_key_to_id: dict[tuple, int] = {}

    def material_id_for(sub: dict) -> int:
        name = str(sub.get("material_name") or f"{asset}_Material")
        color_src = sub.get("material_color") or (0.75, 0.78, 0.72)
        ambient_src = sub.get("material_ambient") or color_src
        specular_src = sub.get("material_specular") or (0.15, 0.15, 0.15)
        color = tuple(float(v) for v in color_src[:3])
        ambient = tuple(float(v) for v in ambient_src[:3])
        specular = tuple(float(v) for v in specular_src[:3])
        shininess = float(sub.get("material_shininess") or 18.0)
        ambient_factor = float(sub.get("material_ambient_factor") if sub.get("material_ambient_factor") is not None else 1.0)
        diffuse_factor = float(sub.get("material_diffuse_factor") if sub.get("material_diffuse_factor") is not None else 1.0)
        specular_factor = float(sub.get("material_specular_factor") if sub.get("material_specular_factor") is not None else 1.0)
        opacity = float(sub.get("material_opacity") if sub.get("material_opacity") is not None else 1.0)
        texture = sub.get("material_texture")
        texture_key = str(texture.get("relative")) if isinstance(texture, dict) else ""
        kind = sub.get("material_kind")
        flags = sub.get("material_flags")
        texture_hints_src = sub.get("material_texture_hints") or ()
        texture_hints = tuple(int(v) for v in texture_hints_src)
        key = (name, color, ambient, specular, shininess, ambient_factor, diffuse_factor, specular_factor, opacity, texture_key, kind, flags, texture_hints)
        if key not in material_key_to_id:
            mat_id = default_mat_id + len(material_defs)
            material_key_to_id[key] = mat_id
            material_defs.append({"id": mat_id, "name": name, "color": color, "ambient": ambient, "specular": specular, "shininess": shininess, "ambient_factor": ambient_factor, "diffuse_factor": diffuse_factor, "specular_factor": specular_factor, "opacity": opacity, "texture": texture, "kind": kind, "flags": flags, "texture_hints": texture_hints})
        return material_key_to_id[key]

    for sub in submeshes:
        material_id_for(sub)

    materials = []
    for mat in material_defs:
        properties = [
            p_node("AmbientColor", "Color", "", "A", *mat["ambient"]),
            p_node("AmbientFactor", "double", "Number", "A", mat["ambient_factor"]),
            p_node("DiffuseColor", "Color", "", "A", *mat["color"]),
            p_node("DiffuseFactor", "double", "Number", "A", mat["diffuse_factor"]),
            p_node("SpecularColor", "Color", "", "A", *mat["specular"]),
            p_node("Shininess", "double", "Number", "", mat["shininess"]),
            p_node("SpecularFactor", "double", "Number", "A", mat["specular_factor"]),
            p_node("Opacity", "double", "Number", "A", mat["opacity"]),
            p_node("TransparencyFactor", "double", "Number", "A", 1.0 - mat["opacity"]),
        ]
        if mat.get("kind") is not None:
            properties.append(p_node("PipeworksType6Kind", "int", "Integer", "U", int(mat["kind"])))
        if mat.get("flags") is not None:
            properties.append(p_node("PipeworksType6Flags", "int", "Integer", "U", int(mat["flags"])))
        for hint_index, hint in enumerate(mat.get("texture_hints") or ()):
            properties.append(p_node(f"PipeworksTextureHint{hint_index}", "int", "Integer", "U", int(hint)))
        materials.append(
            Node(
                "Material",
                [PLong(mat["id"]), PObjectName(mat["name"], "Material"), PStr("")],
                [Node("Version", [PInt(102)]), Node("ShadingModel", [PStr("phong")]), Node("Properties70", children=properties)],
            )
        )
    texture_nodes = []
    video_nodes = []
    texture_connections = []
    for ti, mat in enumerate(material_defs):
        texture = mat.get("texture")
        if not isinstance(texture, dict):
            continue
        tid = texture_base + ti
        vid = video_base + ti
        label = clean_name(f"{mat['name']}_{texture['name']}")
        abs_file = str(Path(texture["file"]).resolve())
        rel_file = str(texture["relative"]).replace("\\", "/")
        texture_nodes.append(
            Node(
                "Texture",
                [PLong(tid), PObjectName(label, "Texture"), PStr("")],
                [
                    Node("Type", [PStr("TextureVideoClip")]),
                    Node("Version", [PInt(202)]),
                    Node("TextureName", [PStr(f"Texture::{label}")]),
                    Node(
                        "Properties70",
                        children=[
                            p_node("WrapModeU", "enum", "", "", 0),
                            p_node("WrapModeV", "enum", "", "", 0),
                            p_node("UseMaterial", "bool", "", "", 1),
                            p_node("UseMipMap", "bool", "", "", 1),
                        ],
                    ),
                    Node("Media", [PStr(f"Video::{label}")]),
                    Node("FileName", [PStr(abs_file)]),
                    Node("RelativeFilename", [PStr(rel_file)]),
                    Node("ModelUVTranslation", [PDouble(0.0), PDouble(0.0)]),
                    Node("ModelUVScaling", [PDouble(1.0), PDouble(1.0)]),
                    Node("Texture_Alpha_Source", [PStr("None")]),
                    Node("Cropping", [PInt(0), PInt(0), PInt(0), PInt(0)]),
                ],
            )
        )
        video_nodes.append(
            Node(
                "Video",
                [PLong(vid), PObjectName(label, "Video"), PStr("Clip")],
                [
                    Node("Type", [PStr("Clip")]),
                    Node("Properties70", children=[p_node("Path", "KString", "XRefUrl", "", rel_file)]),
                    Node("UseMipMap", [PInt(0)]),
                    Node("FileName", [PStr(abs_file)]),
                    Node("RelativeFilename", [PStr(rel_file)]),
                ],
            )
        )
        texture_connections += [
            Node("C", [PStr("OO"), PLong(vid), PLong(tid)]),
            Node("C", [PStr("OP"), PLong(tid), PLong(mat["id"]), PStr("DiffuseColor")]),
        ]
    group = Node(
        "Model",
        [PLong(group_id), PObjectName(f"{asset}_Armature", "Model"), PStr("Null")],
        [
            Node("Version", [PInt(232)]),
            Node("Properties70", children=[p_node("Lcl Translation", "Lcl Translation", "", "A", 0.0, 0.0, 0.0), p_node("Lcl Rotation", "Lcl Rotation", "", "A", 0.0, 0.0, 0.0), p_node("Lcl Scaling", "Lcl Scaling", "", "A", 1.0, 1.0, 1.0)]),
            Node("Shading", [PBool(True)]),
            Node("Culling", [PStr("CullingOff")]),
        ],
    )
    objects_children += materials + texture_nodes + video_nodes + [group]
    connections.append(Node("C", [PStr("OO"), PLong(group_id), PLong(0)]))
    connections += texture_connections
    definitions_count += len(materials) + len(texture_nodes) + len(video_nodes) + 1
    for i, sub in enumerate(submeshes):
        geom_id = base + 100 + i
        mesh_id = base + 500 + i
        geom, model = make_mesh_nodes(asset, sub, mesh_id, geom_id)
        objects_children += [geom, model]
        mat_id = material_id_for(sub)
        connections += [Node("C", [PStr("OO"), PLong(mesh_id), PLong(group_id)]), Node("C", [PStr("OO"), PLong(geom_id), PLong(mesh_id)]), Node("C", [PStr("OO"), PLong(mat_id), PLong(mesh_id)])]
        definitions_count += 2
        if skin_mode != "none" and bones and sub.get("vertices"):
            clusters = build_weighted_skin_clusters(sub, len(bones))
            if clusters:
                sub["skin_mode"] = "native_palette_weights"
            elif skin_mode == "packet_rigid":
                clusters = build_packet_rigid_skin_clusters(sub, bones, globals_)
            elif skin_mode == "auto":
                clusters = build_auto_skin_clusters(sub, bones, globals_)
            clusters = dict(clusters or {})
            # FBX importers only create mesh vertex groups for serialized skin
            # clusters. Custom/root-bound models therefore exposed only the one
            # or two bones that already had weights, despite carrying the full
            # armature. Empty clusters make every deform bone immediately
            # available for manual weight painting without changing weights.
            for bone in bones:
                # Blender 5.x discards a completely empty cluster. A vertex-0
                # sentinel with exactly zero weight preserves the vertex-group
                # name; the bridge importer ignores weights <= 1e-8.
                clusters.setdefault(int(bone["idx"]), ([0], [0.0]))
            skin_id = skin_base + i
            objects_children.append(Node("Deformer", [PLong(skin_id), PObjectName(f"{sub['name']}_Skin", "Deformer"), PStr("Skin")], [Node("Version", [PInt(101)]), Node("Link_DeformAcuracy", [PDouble(50.0)])]))
            connections.append(Node("C", [PStr("OO"), PLong(skin_id), PLong(geom_id)]))
            definitions_count += 1
            deformer_count += 1
            for bone_idx, (indices, weights) in clusters.items():
                cluster_id = cluster_base + i * 10000 + bone_idx
                bone_name = next((b["name"] for b in bones if b["idx"] == bone_idx), f"Bone_{bone_idx}")
                objects_children.append(Node("Deformer", [PLong(cluster_id), PObjectName(f"{sub['name']}_{bone_name}_Cluster", "SubDeformer"), PStr("Cluster")], [Node("Version", [PInt(100)]), Node("UserData", [PStr(""), PStr("")]), Node("Indexes", [AInt(indices)]), Node("Weights", [ADouble(weights)]), Node("Transform", [ADouble(identity())]), Node("TransformLink", [ADouble(matrix_fbx(globals_[bone_idx]))])]))
                connections += [Node("C", [PStr("OO"), PLong(cluster_id), PLong(skin_id)]), Node("C", [PStr("OO"), PLong(bone_model_base + bone_idx), PLong(cluster_id)])]
                definitions_count += 1
                deformer_count += 1
    for b in bones:
        idx = b["idx"]
        tx, ty, tz = b["t"]
        rx, ry, rz = quat_to_euler(b["q"])
        bone_name = b["name"]
        display_size = float(b.get("display_size", 1.5))
        objects_children.append(Node("NodeAttribute", [PLong(bone_attr_base + idx), PObjectName(bone_name, "NodeAttribute"), PStr("LimbNode")], [Node("TypeFlags", [PStr("Skeleton")]), Node("Properties70", children=[p_node("Size", "double", "Number", "", display_size)])]))
        objects_children.append(Node("Model", [PLong(bone_model_base + idx), PObjectName(bone_name, "Model"), PStr("LimbNode")], [Node("Version", [PInt(232)]), Node("Properties70", children=[p_node("Lcl Translation", "Lcl Translation", "", "A", float(tx) * 10.0, float(ty) * 10.0, float(tz) * 10.0), p_node("Lcl Rotation", "Lcl Rotation", "", "A", float(rx), float(ry), float(rz)), p_node("Lcl Scaling", "Lcl Scaling", "", "A", 1.0, 1.0, 1.0), p_node("Size", "double", "Number", "", display_size)]), Node("Shading", [PBool(True)])]))
        parent_id = bone_model_base + b["parent"] if b["parent"] >= 0 else group_id
        connections += [Node("C", [PStr("OO"), PLong(bone_attr_base + idx), PLong(bone_model_base + idx)]), Node("C", [PStr("OO"), PLong(bone_model_base + idx), PLong(parent_id)])]
        definitions_count += 2
    pose_children = [Node("Type", [PStr("BindPose")]), Node("Version", [PInt(100)]), Node("NbPoseNodes", [PInt(len(bones))])]
    for b in bones:
        pose_children.append(Node("PoseNode", children=[Node("Node", [PLong(bone_model_base + b["idx"])]), Node("Matrix", [ADouble(matrix_fbx(globals_[b["idx"]]))])]))
    if bones:
        objects_children.append(Node("Pose", [PLong(base + 900), PObjectName(f"{asset}_BindPose", "Pose"), PStr("BindPose")], pose_children))
        definitions_count += 1
    animation_objects = []
    animation_connections = []
    animation_stack_count = 0
    animation_layer_count = 0
    animation_curve_node_count = 0
    animation_curve_count = 0

    def animation_time_property(name: str, value: int) -> Node:
        return Node("P", [PStr(name), PStr("KTime"), PStr("Time"), PStr(""), PLong(value)])

    def animation_curve(curve_id: int, name: str, times: list[float], values: list[float]) -> Node:
        nonlocal animation_curve_count
        animation_curve_count += 1
        if len(times) == 1:
            times = [times[0], times[0] + 1.0 / 60.0]
            values = [values[0], values[0]]
        return Node(
            "AnimationCurve",
            [PLong(curve_id), PObjectName(name, "AnimCurve"), PStr("")],
            [
                Node("Default", [PDouble(0.0)]),
                Node("KeyVer", [PInt(4008)]),
                Node("KeyTime", [ALong([int(round(time * FBX_TICKS_PER_SECOND)) for time in times])]),
                Node("KeyValueFloat", [AFloat([float(value) for value in values])]),
                Node("KeyAttrFlags", [AInt([4])]),
                Node("KeyAttrDataFloat", [AFloat([0.0, 0.0, 0.0, 0.0])]),
                Node("KeyAttrRefCount", [AInt([len(values)])]),
            ],
        )

    for animation_index, animation in enumerate(animations):
        action_base = animation_base + animation_index * 1000000
        stack_id = action_base + 1
        layer_id = action_base + 2
        duration = max(0.0, float(animation.get("duration", 0.0)))
        stop = int(round(duration * FBX_TICKS_PER_SECOND))
        name = str(animation.get("name") or f"Animation_{animation_index:03d}")
        animation_objects.extend(
            [
                Node(
                    "AnimationStack",
                    [PLong(stack_id), PObjectName(name, "AnimStack"), PStr("")],
                    [
                        Node(
                            "Properties70",
                            children=[
                                animation_time_property("LocalStart", 0),
                                animation_time_property("LocalStop", stop),
                                animation_time_property("ReferenceStart", 0),
                                animation_time_property("ReferenceStop", stop),
                            ],
                        )
                    ],
                ),
                Node("AnimationLayer", [PLong(layer_id), PObjectName("Layer", "AnimLayer"), PStr("")]),
            ]
        )
        animation_stack_count += 1
        animation_layer_count += 1
        animation_connections.append(Node("C", [PStr("OO"), PLong(layer_id), PLong(stack_id)]))
        sample_times = dense_animation_times(duration)
        for track in animation.get("tracks") or []:
            bone = int(track["bone"])
            if bone < 0 or bone >= len(bones):
                continue
            bone_name = str(bones[bone]["name"])
            track_base = action_base + 1000 + bone * 100
            channels = []
            translation_keys = track.get("translation_keys") or []
            if translation_keys:
                if track.get("_preview_constant_translation"):
                    channel_times = [0.0, max(duration, 1.0 / 60.0)]
                    translations = [translation_keys[0][1], translation_keys[0][1]]
                else:
                    channel_times = sample_times
                    translations = sample_animation_keys(translation_keys, channel_times)
                channels.append(("T", "Lcl Translation", channel_times, translations, track_base))
            rotation_keys = track.get("rotation_keys") or []
            if rotation_keys:
                if track.get("_preview_constant_rotation"):
                    channel_times = [0.0, max(duration, 1.0 / 60.0)]
                    quaternions = [rotation_keys[0][1], rotation_keys[0][1]]
                else:
                    channel_times = sample_times
                    quaternions = sample_animation_keys(rotation_keys, channel_times, quaternion=True)
                if (
                    self_contained_animation_preview
                    and int(bones[bone].get("parent", -1)) < 0
                    and not track.get("_preserve_native_root_orientation")
                ):
                    quaternions = stabilize_root_quaternions(quaternions, bones[bone]["q"])
                rotations = unwrap_euler_keys([quat_to_euler(value) for value in quaternions])
                channels.append(("R", "Lcl Rotation", channel_times, rotations, track_base + 10))
            for channel_name, property_name, channel_times, values, curve_node_id in channels:
                curve_node = Node(
                    "AnimationCurveNode",
                    [
                        PLong(curve_node_id),
                        PObjectName(f"{name}_{bone_name}_{channel_name}", "AnimCurveNode"),
                        PStr(""),
                    ],
                    [
                        Node(
                            "Properties70",
                            children=[
                                p_node("d|X", "Number", "", "A", 0.0),
                                p_node("d|Y", "Number", "", "A", 0.0),
                                p_node("d|Z", "Number", "", "A", 0.0),
                            ],
                        )
                    ],
                )
                animation_objects.append(curve_node)
                animation_curve_node_count += 1
                animation_connections.extend(
                    [
                        Node("C", [PStr("OO"), PLong(curve_node_id), PLong(layer_id)]),
                        Node(
                            "C",
                            [
                                PStr("OP"),
                                PLong(curve_node_id),
                                PLong(bone_model_base + bone),
                                PStr(property_name),
                            ],
                        ),
                    ]
                )
                for axis, axis_index in (("X", 0), ("Y", 1), ("Z", 2)):
                    curve_id = curve_node_id + axis_index + 1
                    animation_objects.append(
                        animation_curve(
                            curve_id,
                            f"{name}_{bone_name}_{channel_name}_{axis}",
                            channel_times,
                            [value[axis_index] for value in values],
                        )
                    )
                    animation_connections.append(
                        Node(
                            "C",
                            [PStr("OP"), PLong(curve_id), PLong(curve_node_id), PStr(f"d|{axis}")],
                        )
                    )
    objects_children += animation_objects
    connections += animation_connections
    definitions_count += len(animation_objects)
    objects = Node("Objects", children=objects_children)
    con = Node("Connections", children=connections)

    def objtype(name, count):
        return Node("ObjectType", [PStr(name)], [Node("Count", [PInt(count)])])

    definitions = Node("Definitions", children=[Node("Version", [PInt(100)]), Node("Count", [PInt(definitions_count)]), objtype("Geometry", len(submeshes)), objtype("Model", len(submeshes) + len(bones) + 1), objtype("Material", len(materials)), objtype("Texture", len(texture_nodes)), objtype("Video", len(video_nodes)), objtype("NodeAttribute", len(bones)), objtype("Deformer", deformer_count), objtype("Pose", 1 if bones else 0), objtype("AnimationStack", animation_stack_count), objtype("AnimationLayer", animation_layer_count), objtype("AnimationCurveNode", animation_curve_node_count), objtype("AnimationCurve", animation_curve_count)])
    global_settings = Node("GlobalSettings", children=[Node("Version", [PInt(1000)]), Node("Properties70", children=[p_node("UpAxis", "int", "Integer", "", 2), p_node("UpAxisSign", "int", "Integer", "", 1), p_node("FrontAxis", "int", "Integer", "", 1), p_node("FrontAxisSign", "int", "Integer", "", -1), p_node("CoordAxis", "int", "Integer", "", 0), p_node("CoordAxisSign", "int", "Integer", "", 1), p_node("UnitScaleFactor", "double", "Number", "", 1.0)])])
    header = Node("FBXHeaderExtension", children=[Node("FBXHeaderVersion", [PInt(1003)]), Node("FBXVersion", [PInt(7400)]), Node("Creator", [PStr("GZ Blender Bridge")])])
    takes_children = [Node("Current", [PStr("000_REST_POSE" if animations else "")])]
    for animation in animations:
        name = str(animation.get("name") or "Animation")
        stop = int(round(max(0.0, float(animation.get("duration", 0.0))) * FBX_TICKS_PER_SECOND))
        takes_children.append(
            Node(
                "Take",
                [PStr(name)],
                [
                    Node("FileName", [PStr(f"{name}.tak")]),
                    Node("LocalTime", [PLong(0), PLong(stop)]),
                    Node("ReferenceTime", [PLong(0), PLong(stop)]),
                ],
            )
        )
    buf = io.BytesIO()
    buf.write(b"Kaydara FBX Binary  \x00\x1a\x00")
    buf.write(struct.pack("<I", 7400))
    for node in [header, Node("FileId", [PRaw(b"\0" * 16)]), global_settings, definitions, objects, con, Node("Takes", children=takes_children)]:
        write_node(buf, node)
    buf.write(NULL_RECORD)
    buf.write(b"\0" * 160)
    path.write_bytes(buf.getvalue())


def export_cmg(cmg: Path, fbx: Path, scale: float) -> dict:
    parser = PipeworksParser(str(cmg))
    entries = parser.parse()
    data = parser.file_data or b""
    asset = clean_name(Path(cmg).stem)
    mesh_pairs = find_mesh_pairs(entries, data)
    materials = parse_cmg_materials(entries, data)
    fbx.parent.mkdir(parents=True, exist_ok=True)
    textures = parse_cmg_textures(entries, data, fbx.parent / "textures")
    attach_cmg_material_textures(materials, textures)
    bones, globals_ = parse_cmg_skeleton(parser, entries, scale)
    animations, animation_report = decode_cmg_type4_animations(data, entries, bones, scale)
    submeshes = []
    total_faces = 0
    total_verts = 0
    resource_names = []
    material_names: set[str] = set()
    unassigned_material_submeshes = 0
    textured_material_names: set[str] = set()
    detected_skin_ranges = 0
    detected_skin_range_vertices = 0
    native_weighted_vertices = 0
    native_blended_vertices = 0
    native_palette_entries = 0
    for pair_index, (main_entry, res_entry, descs) in enumerate(mesh_pairs):
        main = data[main_entry["offset"] : main_entry["offset"] + main_entry["size"]]
        resource = data[res_entry["offset"] : res_entry["offset"] + res_entry["size"]]
        bone_palette = cmg_mesh_bone_palette(main, bones) if bones else []
        native_palette_entries += len(bone_palette)
        material_ranges = parse_cmg_material_ranges(main, materials)
        descs = find_compact_material_descriptors(resource, descs, material_ranges) + descs
        material_by_desc_id = {
            id(desc): material_ranges[index]["material"]
            for index, desc in enumerate(sorted([d for d in descs if not d.get("compact_position_only")], key=lambda d: d["main_offset"]))
            if index < len(material_ranges)
        }
        desc_next_offsets = {}
        ordered_real_descs = sorted([d for d in descs if not d.get("compact_position_only")], key=lambda d: d["main_offset"])
        for desc_index, real_desc in enumerate(ordered_real_descs):
            desc_next_offsets[id(real_desc)] = (
                ordered_real_descs[desc_index + 1]["main_offset"]
                if desc_index + 1 < len(ordered_real_descs)
                else len(main)
            )
        resource_names.append(res_entry["name"])
        pair_name = clean_name(main_entry["name"])
        for i, desc in enumerate(descs):
            positions = parse_positions(resource, desc["rel_vertex"], desc["vertex_count"], position_stride(desc))
            normals = parse_normals(resource, desc)
            faces, uv_inline, mode = decode_display_list(resource, desc)
            faces, removed_spikes, kept_face_indices = remove_origin_marker_faces(positions, faces)
            if removed_spikes:
                filtered_uvs = {}
                out_corner = 0
                for old_face_index in kept_face_indices:
                    for j in range(3):
                        old_corner = old_face_index * 3 + j
                        if old_corner in uv_inline:
                            filtered_uvs[out_corner] = uv_inline[old_corner]
                        out_corner += 1
                uv_inline = filtered_uvs
            verts = [(x * scale, y * scale, z * scale) for x, y, z in positions]
            material = desc.get("compact_material") or cmg_material_for_positions(positions, material_ranges)
            if material is None:
                material = material_by_desc_id.get(id(desc))
            if material is not None:
                material_names.add(material["name"])
                if material.get("texture"):
                    textured_material_names.add(material["name"])
            else:
                unassigned_material_submeshes += 1
            skin_ranges = []
            vertex_weights = []
            if bone_palette and not desc.get("compact_position_only"):
                next_off = desc_next_offsets.get(id(desc), len(main))
                skin_ranges = cmg_descriptor_skin_ranges(main, desc, next_off, len(bone_palette))
                vertex_weights = cmg_vertex_weights_from_ranges(resource, desc, skin_ranges, bone_palette)
                if vertex_weights:
                    native_weighted_vertices += len(vertex_weights)
                    native_blended_vertices += sum(len(weights) > 1 for weights in vertex_weights)
                detected_skin_ranges += len(skin_ranges)
                detected_skin_range_vertices += sum(r[0] for r in skin_ranges)
            submesh = {
                "name": f"{asset}_{pair_index:02d}_{pair_name}_submesh_{i:02d}_{desc['main_offset']:04x}",
                "vertices": verts,
                "normals": normals,
                "faces": faces,
                "uvs": uv_inline,
                "mode": mode,
                "removed_origin_marker_faces": removed_spikes,
                "vertex_weights": vertex_weights,
                "skin_range_count": len(skin_ranges) if vertex_weights else 0,
                "skin_range_vertices": len(vertex_weights),
                "detected_skin_range_count": len(skin_ranges),
                "detected_skin_range_vertices": sum(r[0] for r in skin_ranges),
                "material_name": material["name"] if material else f"{asset}_{pair_index:02d}_{pair_name}_submesh_{i:02d}_Material",
                "material_color": material["color"] if material else (0.75, 0.78, 0.72),
                "material_ambient": material.get("ambient") if material else (0.75, 0.75, 0.75),
                "material_specular": material.get("specular") if material else (0.15, 0.15, 0.15),
                "material_shininess": material.get("shininess") if material else 18.0,
                "material_texture": material.get("texture") if material else None,
                "desc": desc,
            }
            submeshes.append(submesh)
            total_faces += len(faces)
            total_verts += len(verts)
    write_fbx(fbx, asset, submeshes, bones, globals_, skin_mode="auto", animations=animations)
    action_baseline = None
    if animations:
        from animation_baseline import build_action_baseline

        action_baseline = fbx.with_suffix(".animation_action_baseline_v1.json.gz")
        expected_names = [str(animation["name"]) for animation in animations]
        build_action_baseline(fbx, expected_names, action_baseline)
    bone_names = {b["idx"]: b["name"] for b in bones}
    skin_stats = [
        {
            "submesh": sub["name"],
            "mode": sub.get("skin_mode", "none"),
            "bone": bone_names.get(sub.get("skin_bone"), sub.get("skin_bone")),
            "ranges": sub.get("skin_range_count", 0),
            "range_vertices": sub.get("skin_range_vertices", 0),
        }
        for sub in submeshes
        if sub.get("skin_mode") is not None or sub.get("skin_bone") is not None or sub.get("vertex_weights")
    ]
    return {
        "resources": ", ".join(resource_names),
        "submeshes": len(submeshes),
        "vertices": total_verts,
        "triangles": total_faces,
        "bones": len(bones),
        "animations": animation_report,
        "animation_action_baseline": str(action_baseline) if action_baseline else None,
        "skin_mode": "native_palette_weights_with_geometric_fallback" if bones else "none",
        "detected_skin_stream": "native_palette_ranges_and_float_pairs",
        "native_palette_entries": native_palette_entries,
        "native_weighted_vertices": native_weighted_vertices,
        "native_blended_vertices": native_blended_vertices,
        "detected_skin_ranges": detected_skin_ranges,
        "detected_skin_range_vertices": detected_skin_range_vertices,
        "material_count": len(material_names),
        "materials": ", ".join(sorted(material_names)),
        "texture_count": len(textures),
        "textures": ", ".join(sorted(t["name"] for t in textures.values())),
        "textured_materials": ", ".join(sorted(textured_material_names)),
        "unassigned_material_submeshes": unassigned_material_submeshes,
        "skinned_submeshes": len(skin_stats),
        "skin_stats": skin_stats,
        "fbx": str(fbx),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Export a GameCube Pipeworks CMG mesh to FBX")
    ap.add_argument("cmg")
    ap.add_argument("--fbx", required=True)
    ap.add_argument("--scale", type=float, default=10.0)
    args = ap.parse_args(argv)
    report = export_cmg(Path(args.cmg), Path(args.fbx), args.scale)
    for key, value in report.items():
        if key == "skin_stats":
            for item in value[:80]:
                print(
                    f"skin: {item['submesh']} -> {item['bone']} ({item['mode']}; "
                    f"ranges={item['ranges']} verts={item['range_vertices']})"
                )
            if len(value) > 80:
                print(f"skin: ... {len(value) - 80} more")
            continue
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
