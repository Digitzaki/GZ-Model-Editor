#!/usr/bin/env python3
"""GameCube Pipeworks CMG mesh exporter."""
from __future__ import annotations

import argparse
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


VALID_PRIMS = {0x80, 0x90, 0x98, 0xA0}


def be32(data: bytes, off: int) -> int:
    return struct.unpack_from(">I", data, off)[0]


def sbe32(data: bytes, off: int) -> int:
    return struct.unpack_from(">i", data, off)[0]


def fbe(data: bytes, off: int) -> float:
    return struct.unpack_from(">f", data, off)[0]


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
    descs.sort(key=lambda d: (d["rel_dl"], d["rel_vertex"], d["main_offset"]))
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


def decode_display_list(resource: bytes, desc: dict) -> tuple[list[tuple[int, int, int]], dict[int, tuple[float, float]], str]:
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
                # Type 4 stores translation per bone; rotation is left neutral.
                t = struct.unpack_from(">3f", pblob, rel + 24)
                pose_by_index[idx] = (t, (0.0, 0.0, 0.0, 1.0))
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
        bones[idx] = {"idx": idx, "parent": parent, "name": name, "q": q, "t": t, "children_rel": children}
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
def AInt(v): return Arr("i", v)


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
        elif p.code == "i":
            raw = struct.pack("<%si" % len(vals), *map(int, vals)) if vals else b""
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


def make_mesh_nodes(asset: str, submesh: dict, mesh_id: int, geom_id: int) -> tuple[Node, Node]:
    name = submesh["name"]
    verts = submesh["vertices"]
    faces = submesh["faces"]
    poly = []
    normals = []
    uvs = []
    for a, b, c in faces:
        poly += [a, b, ~c]
        tri_uvs = [
            submesh["uvs"].get(len(uvs) + j, fallback_uv(vi))
            for j, vi in enumerate((a, b, c))
        ]
        tri_uvs = squash_long_triangle_uvs(tri_uvs)
        for vi, uv in zip((a, b, c), tri_uvs):
            normals.append((0.0, 0.0, 1.0))
            uvs.append(uv)
    geom = Node(
        "Geometry",
        [PLong(geom_id), PStr(f"Geometry::{name}_Geometry"), PStr("Mesh")],
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
        [PLong(mesh_id), PStr(f"Model::{name}"), PStr("Mesh")],
        [
            Node("Version", [PInt(232)]),
            Node("Properties70", children=[p_node("Lcl Translation", "Lcl Translation", "", "A", 0.0, 0.0, 0.0), p_node("Lcl Rotation", "Lcl Rotation", "", "A", 0.0, 0.0, 0.0), p_node("Lcl Scaling", "Lcl Scaling", "", "A", 1.0, 1.0, 1.0)]),
            Node("Shading", [PBool(True)]),
            Node("Culling", [PStr("CullingOff")]),
        ],
    )
    return geom, model


def write_fbx(path: Path, asset: str, submeshes: list[dict], bones: list[dict], globals_: dict[int, list[list[float]]]) -> None:
    base = (int(hashlib.sha1(asset.encode("utf-8")).hexdigest()[:8], 16) % 1000000000) + 3100000000
    mat_id = base + 1
    group_id = base + 2
    bone_model_base = base + 1000
    bone_attr_base = base + 2000
    objects_children = []
    connections = []
    definitions_count = 1
    material = Node("Material", [PLong(mat_id), PStr(f"Material::{asset}_Material"), PStr("")], [Node("Version", [PInt(102)]), Node("ShadingModel", [PStr("phong")]), Node("Properties70", children=[p_node("DiffuseColor", "Color", "", "A", 0.75, 0.78, 0.72)])])
    group = Node(
        "Model",
        [PLong(group_id), PStr(f"Model::{asset}"), PStr("Null")],
        [
            Node("Version", [PInt(232)]),
            Node("Properties70", children=[p_node("Lcl Translation", "Lcl Translation", "", "A", 0.0, 0.0, 0.0), p_node("Lcl Rotation", "Lcl Rotation", "", "A", 0.0, 0.0, 0.0), p_node("Lcl Scaling", "Lcl Scaling", "", "A", 1.0, 1.0, 1.0)]),
            Node("Shading", [PBool(True)]),
            Node("Culling", [PStr("CullingOff")]),
        ],
    )
    objects_children += [material, group]
    connections.append(Node("C", [PStr("OO"), PLong(group_id), PLong(0)]))
    definitions_count += 2
    for i, sub in enumerate(submeshes):
        geom_id = base + 100 + i
        mesh_id = base + 500 + i
        geom, model = make_mesh_nodes(asset, sub, mesh_id, geom_id)
        objects_children += [geom, model]
        connections += [Node("C", [PStr("OO"), PLong(mesh_id), PLong(group_id)]), Node("C", [PStr("OO"), PLong(geom_id), PLong(mesh_id)]), Node("C", [PStr("OO"), PLong(mat_id), PLong(mesh_id)])]
        definitions_count += 2
    for b in bones:
        idx = b["idx"]
        tx, ty, tz = b["t"]
        rx, ry, rz = quat_to_euler(b["q"])
        bone_name = b["name"]
        objects_children.append(Node("NodeAttribute", [PLong(bone_attr_base + idx), PStr(f"NodeAttribute::{bone_name}"), PStr("LimbNode")], [Node("TypeFlags", [PStr("Skeleton")]), Node("Properties70", children=[p_node("Size", "double", "Number", "", 1.5)])]))
        objects_children.append(Node("Model", [PLong(bone_model_base + idx), PStr(f"Model::{bone_name}"), PStr("LimbNode")], [Node("Version", [PInt(232)]), Node("Properties70", children=[p_node("Lcl Translation", "Lcl Translation", "", "A", float(tx) * 10.0, float(ty) * 10.0, float(tz) * 10.0), p_node("Lcl Rotation", "Lcl Rotation", "", "A", float(rx), float(ry), float(rz)), p_node("Lcl Scaling", "Lcl Scaling", "", "A", 1.0, 1.0, 1.0), p_node("Size", "double", "Number", "", 1.5)]), Node("Shading", [PBool(True)])]))
        connections += [Node("C", [PStr("OO"), PLong(bone_attr_base + idx), PLong(bone_model_base + idx)]), Node("C", [PStr("OO"), PLong(bone_model_base + idx), PLong(bone_model_base + b["parent"] if b["parent"] >= 0 else 0)])]
        definitions_count += 2
    pose_children = [Node("Type", [PStr("BindPose")]), Node("Version", [PInt(100)]), Node("NbPoseNodes", [PInt(len(bones))])]
    for b in bones:
        pose_children.append(Node("PoseNode", children=[Node("Node", [PLong(bone_model_base + b["idx"])]), Node("Matrix", [ADouble(matrix_fbx(globals_[b["idx"]]))])]))
    if bones:
        objects_children.append(Node("Pose", [PLong(base + 50), PStr(f"Pose::{asset}_BindPose"), PStr("BindPose")], pose_children))
        definitions_count += 1
    objects = Node("Objects", children=objects_children)
    con = Node("Connections", children=connections)

    def objtype(name, count):
        return Node("ObjectType", [PStr(name)], [Node("Count", [PInt(count)])])

    definitions = Node("Definitions", children=[Node("Version", [PInt(100)]), Node("Count", [PInt(definitions_count)]), objtype("Geometry", len(submeshes)), objtype("Model", len(submeshes) + len(bones) + 1), objtype("Material", 1), objtype("NodeAttribute", len(bones)), objtype("Pose", 1 if bones else 0)])
    global_settings = Node("GlobalSettings", children=[Node("Version", [PInt(1000)]), Node("Properties70", children=[p_node("UpAxis", "int", "Integer", "", 2), p_node("UpAxisSign", "int", "Integer", "", 1), p_node("FrontAxis", "int", "Integer", "", 1), p_node("FrontAxisSign", "int", "Integer", "", -1), p_node("CoordAxis", "int", "Integer", "", 0), p_node("CoordAxisSign", "int", "Integer", "", 1), p_node("UnitScaleFactor", "double", "Number", "", 1.0)])])
    header = Node("FBXHeaderExtension", children=[Node("FBXHeaderVersion", [PInt(1003)]), Node("FBXVersion", [PInt(7400)]), Node("Creator", [PStr("CMG Blender Bridge restored")])])
    buf = io.BytesIO()
    buf.write(b"Kaydara FBX Binary  \x00\x1a\x00")
    buf.write(struct.pack("<I", 7400))
    for node in [header, Node("FileId", [PRaw(b"\0" * 16)]), global_settings, definitions, objects, con, Node("Takes", children=[Node("Current", [PStr("")])])]:
        write_node(buf, node)
    buf.write(NULL_RECORD)
    buf.write(b"\0" * 160)
    path.write_bytes(buf.getvalue())


def export_cmg(cmg: Path, fbx: Path, scale: float) -> dict:
    parser = PipeworksParser(str(cmg))
    entries = parser.parse()
    data = parser.file_data or b""
    main_entry, res_entry = find_type17(entries)
    main = data[main_entry["offset"] : main_entry["offset"] + main_entry["size"]]
    resource = data[res_entry["offset"] : res_entry["offset"] + res_entry["size"]]
    asset = clean_name(Path(cmg).stem)
    descs = find_mesh_descriptors(main, len(resource))
    submeshes = []
    total_faces = 0
    total_verts = 0
    for i, desc in enumerate(descs):
        positions = parse_positions(resource, desc["rel_vertex"], desc["vertex_count"], position_stride(desc))
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
        submeshes.append(
            {
                "name": f"{asset}_submesh_{i:02d}_{desc['main_offset']:04x}",
                "vertices": verts,
                "faces": faces,
                "uvs": uv_inline,
                "mode": mode,
                "removed_origin_marker_faces": removed_spikes,
                "desc": desc,
            }
        )
        total_faces += len(faces)
        total_verts += len(verts)
    bones: list[dict] = []
    globals_: dict[int, list[list[float]]] = {}
    fbx.parent.mkdir(parents=True, exist_ok=True)
    write_fbx(fbx, asset, submeshes, bones, globals_)
    return {
        "resource": res_entry["name"],
        "submeshes": len(submeshes),
        "vertices": total_verts,
        "triangles": total_faces,
        "bones": len(bones),
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
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
