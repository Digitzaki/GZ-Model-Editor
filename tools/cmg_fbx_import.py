"""Patch GameCube CMG mesh positions from an edited FBX."""
from __future__ import annotations

import argparse
import collections
import math
import re
import shutil
import struct
import sys
from pathlib import Path

TOOL_DIR = Path(__file__).resolve().parent
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from cmg_probe import (
    cmg_descriptor_bbox_key,
    cmg_descriptor_skin_ranges,
    cmg_mesh_bone_palette,
    cmg_vertex_weights_from_ranges,
    decode_display_list,
    find_mesh_descriptors,
    find_type17,
    global_matrices,
    parse_cmg_animation_translation_tracks,
    parse_cmg_skeleton,
    parse_normals,
    parse_positions,
    position_stride,
    read_uv_table,
)
from cmp_fbx_import import (
    cmp_inverse_bind_values,
    cmp_bone_name_candidates,
    fbx_bone_models,
    find_cmp_bone_model,
    geometry_normals_by_cp,
    geometry_weights_by_cp,
)
from fbx_to_bdg_import import object_nodes, p_values, parse_fbx
from parser_core import PipeworksParser


def clean_name(value: str) -> str:
    if "::" in value:
        value = value.split("::", 1)[1]
    return value.replace("\x00\x01", "")


def geometry_payloads(fbx: Path) -> dict[int, dict]:
    roots, _version = parse_fbx(fbx)
    out: dict[int, dict] = {}
    for geom in object_nodes(roots, "Geometry"):
        if len(geom.props) < 2:
            continue
        name = clean_name(str(geom.props[1]))
        match = re.search(r"_submesh_(\d+)_[0-9a-fA-F]+", name)
        if not match:
            continue
        verts_node = geom.child("Vertices")
        pvi_node = geom.child("PolygonVertexIndex")
        if not verts_node or not verts_node.props or not pvi_node or not pvi_node.props:
            continue
        raw = verts_node.props[0]
        verts = []
        for i in range(0, len(raw) - 2, 3):
            verts.append((float(raw[i]), float(raw[i + 1]), float(raw[i + 2])))
        faces = []
        cp_seq = []
        face = []
        face_loop_indices = []
        loop_indices = []
        for loop_index, value in enumerate(pvi_node.props[0]):
            cp = int(value if value >= 0 else ~value)
            cp_seq.append(cp)
            face.append(cp)
            loop_indices.append(loop_index)
            if value < 0:
                if len(face) == 3:
                    faces.append(tuple(face))
                    face_loop_indices.append(tuple(loop_indices))
                face = []
                loop_indices = []
        uvs: list[list[tuple[float, float]]] = [[] for _ in verts]
        corner_uvs: list[tuple[float, float] | None] = [None] * len(cp_seq)
        uv_layer = geom.child("LayerElementUV")
        if uv_layer and uv_layer.child("UV"):
            uv_values = uv_layer.child("UV").props[0]
            uv_pairs = [tuple(map(float, uv_values[i : i + 2])) for i in range(0, len(uv_values), 2)]
            uv_index = uv_layer.child("UVIndex")
            if uv_index:
                for loop_index, idx in enumerate(uv_index.props[0]):
                    idx = int(idx)
                    if loop_index < len(corner_uvs) and 0 <= idx < len(uv_pairs):
                        corner_uvs[loop_index] = uv_pairs[idx]
            else:
                for loop_index, uv in enumerate(uv_pairs[: len(corner_uvs)]):
                    corner_uvs[loop_index] = uv
            for pv_i, cp in enumerate(cp_seq):
                if 0 <= cp < len(uvs) and pv_i < len(corner_uvs) and corner_uvs[pv_i] is not None:
                    uvs[cp].append(corner_uvs[pv_i])
        uv_by_cp = []
        for samples in uvs:
            if not samples:
                uv_by_cp.append(None)
            else:
                uv_by_cp.append(max(set(samples), key=samples.count))
        face_uvs = [
            tuple(corner_uvs[loop_index] for loop_index in indices)
            for indices in face_loop_indices
        ]
        normals = geometry_normals_by_cp(geom, len(verts))
        weights, weight_report = geometry_weights_by_cp(roots, geom, len(verts))
        submesh_index = int(match.group(1))
        if submesh_index in out:
            base = len(out[submesh_index]["vertices"])
            out[submesh_index]["vertices"].extend(verts)
            out[submesh_index]["faces"].extend(tuple(i + base for i in tri) for tri in faces)
            out[submesh_index]["uvs"].extend(uv_by_cp)
            out[submesh_index]["face_uvs"].extend(face_uvs)
            out[submesh_index]["normals"].extend(normals)
            out[submesh_index]["weights"].extend(weights)
            out[submesh_index]["weight_reports"].append(weight_report)
        else:
            out[submesh_index] = {
                "name": name,
                "vertices": verts,
                "faces": faces,
                "uvs": uv_by_cp,
                "face_uvs": face_uvs,
                "normals": normals,
                "weights": weights,
                "weight_reports": [weight_report],
            }
    return out


def align16(value: int) -> int:
    return (value + 15) & ~15


def align32(value: int) -> int:
    return (value + 31) & ~31


def patch_bundle_resource_resize(
    data: bytearray,
    parser: PipeworksParser,
    entries: list[dict],
    resource_entry: dict,
    old_size: int,
    new_size: int,
) -> None:
    delta = new_size - old_size
    if delta == 0:
        return
    endian = ">" if parser.is_big_endian else "<"
    insert_pos = resource_entry["offset"] + old_size
    struct.pack_into(f"{endian}I", data, resource_entry["toc_entry_offset"] + 14, new_size)
    for header_off in (0x34, 0x64, 0x68, 0x70):
        if header_off + 4 > len(data):
            continue
        value = struct.unpack_from(f"{endian}I", data, header_off)[0]
        if value >= insert_pos:
            struct.pack_into(f"{endian}I", data, header_off, value + delta)
    seen_toc_entries: set[int] = set()
    for entry in entries:
        if entry is resource_entry:
            entry["size"] = new_size
        toc = entry["toc_entry_offset"]
        if toc in seen_toc_entries:
            continue
        seen_toc_entries.add(toc)
        main_rel = struct.unpack_from(f"{endian}I", data, toc + 2)[0]
        main_size = struct.unpack_from(f"{endian}I", data, toc + 6)[0]
        if main_size:
            main_abs = parser.main_data_offset + main_rel
            if main_abs >= insert_pos:
                struct.pack_into(f"{endian}I", data, toc + 2, main_rel + delta)
        res_rel = struct.unpack_from(f"{endian}I", data, toc + 10)[0]
        res_size = struct.unpack_from(f"{endian}I", data, toc + 14)[0]
        if res_size:
            res_abs = parser.resource_data_offset + res_rel
            if res_abs >= insert_pos:
                struct.pack_into(f"{endian}I", data, toc + 10, res_rel + delta)
    for entry in entries:
        if entry is resource_entry:
            continue
        if entry["offset"] >= insert_pos:
            entry["offset"] += delta
    # The game uses this as the allocation/load boundary for the complete
    # resource-data section. The parser can tolerate a stale value, but newly
    # appended mesh streams will then sit outside the region loaded in-game.
    if 0x74 + 4 <= len(data):
        struct.pack_into(
            f"{endian}I",
            data,
            0x74,
            len(data) - int(parser.resource_data_offset),
        )


def patch_bundle_main_insert(
    data: bytearray,
    parser: PipeworksParser,
    entries: list[dict],
    main_entry: dict,
    insert_rel: int,
    payload: bytes,
) -> None:
    if not payload:
        return
    delta = len(payload)
    endian = ">" if parser.is_big_endian else "<"
    insert_pos = main_entry["offset"] + insert_rel
    old_size = main_entry["size"]
    data[insert_pos:insert_pos] = payload
    struct.pack_into(f"{endian}I", data, main_entry["toc_entry_offset"] + 6, old_size + delta)
    main_entry["size"] = old_size + delta
    for header_off in (0x34, 0x64, 0x68, 0x70):
        if header_off + 4 > len(data):
            continue
        value = struct.unpack_from(f"{endian}I", data, header_off)[0]
        if value >= insert_pos:
            struct.pack_into(f"{endian}I", data, header_off, value + delta)
            if header_off == 0x70:
                parser.resource_data_offset += delta
            elif header_off == 0x68:
                parser.main_data_offset += delta
    seen_toc_entries: set[int] = set()
    for entry in entries:
        toc = entry["toc_entry_offset"]
        if toc in seen_toc_entries:
            continue
        seen_toc_entries.add(toc)
        main_rel = struct.unpack_from(f"{endian}I", data, toc + 2)[0]
        main_size = struct.unpack_from(f"{endian}I", data, toc + 6)[0]
        if main_size:
            main_abs = parser.main_data_offset + main_rel
            if entry is not main_entry and main_abs >= insert_pos:
                struct.pack_into(f"{endian}I", data, toc + 2, main_rel + delta)
        # Resource data moves as a whole when the main block grows, so the
        # resource base header is enough. Per-resource relative offsets stay put.
    for entry in entries:
        if entry is main_entry:
            continue
        if entry["offset"] >= insert_pos:
            entry["offset"] += delta


def patch_type2_loader_refs(data: bytearray, entries: list[dict], ref_map: dict[int, int]) -> int:
    if not ref_map:
        return 0
    patched = 0
    for entry in entries:
        if entry.get("file_type") != 2 or entry.get("is_resource"):
            continue
        start = entry["offset"]
        end = start + entry["size"]
        if start < 0 or end > len(data):
            continue
        for off in range(start, end - 3, 4):
            value = struct.unpack_from(">I", data, off)[0]
            replacement = ref_map.get(value)
            if replacement is not None:
                struct.pack_into(">I", data, off, replacement)
                patched += 1
    return patched


def cmg_first_primitive_opcode(resource: bytes, desc: dict) -> int:
    rel = int(desc.get("rel_dl", 0))
    end = rel + int(desc.get("dl_size", 0))
    while rel + 3 <= end and rel < len(resource):
        opcode = resource[rel]
        count = struct.unpack_from(">H", resource, rel + 1)[0]
        if opcode and count:
            return opcode
        rel += 1
    return 0x90


def cmg_triangle_list_opcode(resource: bytes, desc: dict) -> int:
    return 0x90 | (cmg_first_primitive_opcode(resource, desc) & 0x07)


def build_idx8_2_triangle_display_list(faces: list[tuple[int, int, int]], primitive_opcode: int) -> bytes:
    out = bytearray()
    max_verts_per_batch = 80
    for start in range(0, len(faces), max_verts_per_batch // 3):
        batch = faces[start : start + max_verts_per_batch // 3]
        refs = [idx for tri in batch for idx in tri]
        out.append(primitive_opcode & 0xFF)
        out.extend(struct.pack(">H", len(refs)))
        for vi in refs:
            out.append(vi & 0xFF)
            out.append(vi & 0xFF)
    return bytes(out)


def build_idx16_idx16_idx8_triangle_display_list(
    faces: list[tuple[int, int, int]],
    face_uv_indices: list[tuple[int, int, int]],
    primitive_opcode: int,
) -> bytes:
    if len(faces) != len(face_uv_indices):
        raise ValueError("CMG face and UV-index counts differ")
    out = bytearray()
    # Native CMG streams split GX primitives into small batches (this model's
    # largest triangle-list command has 261 records). Keep generated commands
    # in the same range instead of feeding the game one enormous FIFO command.
    max_faces_per_batch = 85
    for start in range(0, len(faces), max_faces_per_batch):
        face_batch = faces[start : start + max_faces_per_batch]
        uv_batch = face_uv_indices[start : start + max_faces_per_batch]
        out.append(primitive_opcode & 0xFF)
        out.extend(struct.pack(">H", len(face_batch) * 3))
        for face, uv_indices in zip(face_batch, uv_batch):
            for vertex_index, uv_index in zip(face, uv_indices):
                if not (0 <= int(vertex_index) <= 0xFFFF):
                    raise ValueError(f"CMG vertex index exceeds u16: {vertex_index}")
                if not (0 <= int(uv_index) <= 0xFF):
                    raise ValueError(f"CMG UV index exceeds u8: {uv_index}")
                out.extend(
                    struct.pack(">HHB", int(vertex_index), int(vertex_index), int(uv_index))
                )
    return bytes(out)


def extend_last_idx16_idx16_idx8_triangle_command(
    native_prefix: bytes,
    faces: list[tuple[int, int, int]],
    face_uv_indices: list[tuple[int, int, int]],
) -> bytes:
    if len(faces) != len(face_uv_indices):
        raise ValueError("CMG face and UV-index counts differ")
    pos = 0
    last_command = None
    while pos < len(native_prefix):
        opcode = native_prefix[pos]
        if opcode == 0:
            pos += 1
            continue
        if (opcode & 0xF8) not in (0x80, 0x90, 0x98, 0xA0) or pos + 3 > len(native_prefix):
            raise ValueError("native CMG display list has an invalid command")
        count = struct.unpack_from(">H", native_prefix, pos + 1)[0]
        command_end = pos + 3 + count * 5
        if command_end > len(native_prefix):
            raise ValueError("native CMG display-list command is truncated")
        last_command = (pos, opcode, count)
        pos = command_end
    if last_command is None or (last_command[1] & 0xF8) != 0x90:
        raise ValueError("native CMG display list does not end in a triangle command")

    added_count = len(faces) * 3
    combined_count = last_command[2] + added_count
    if combined_count > 0xFFFF:
        raise ValueError("extended CMG triangle command exceeds the u16 count limit")
    out = bytearray(native_prefix)
    struct.pack_into(">H", out, last_command[0] + 1, combined_count)
    for face, uv_indices in zip(faces, face_uv_indices):
        for vertex_index, uv_index in zip(face, uv_indices):
            if not (0 <= int(vertex_index) <= 0xFFFF):
                raise ValueError(f"CMG vertex index exceeds u16: {vertex_index}")
            if not (0 <= int(uv_index) <= 0xFF):
                raise ValueError(f"CMG UV index exceeds u8: {uv_index}")
            out.extend(struct.pack(">HHB", int(vertex_index), int(vertex_index), int(uv_index)))
    return bytes(out)


def indexed_face_uvs(
    geom: dict,
    face_indices: list[int] | None = None,
    initial_uv_table: list[tuple[float, float]] | None = None,
) -> tuple[list[tuple[float, float]], list[tuple[int, int, int]]]:
    uv_table = list(initial_uv_table or [])
    uv_lookup: dict[tuple[float, float], int] = {
        (round(float(u), 5), round(float(v), 5)): index
        for index, (u, v) in enumerate(uv_table)
    }
    face_uv_indices: list[tuple[int, int, int]] = []
    face_uvs = list(geom.get("face_uvs") or [])
    cp_uvs = list(geom.get("uvs") or [])
    faces = list(geom.get("faces") or [])
    if face_indices is None:
        face_indices = list(range(len(faces)))
    for face_index in face_indices:
        face = faces[face_index]
        source_uvs = face_uvs[face_index] if face_index < len(face_uvs) else (None, None, None)
        indices = []
        for corner, vertex_index in enumerate(face):
            uv = source_uvs[corner] if corner < len(source_uvs) else None
            if uv is None and 0 <= int(vertex_index) < len(cp_uvs):
                uv = cp_uvs[int(vertex_index)]
            if uv is None:
                uv = (0.0, 0.0)
            stored = (float(uv[0]), 1.0 - float(uv[1]))
            key = (round(stored[0], 5), round(stored[1], 5))
            uv_index = uv_lookup.get(key)
            if uv_index is None:
                uv_index = len(uv_table)
                uv_lookup[key] = uv_index
                uv_table.append(stored)
            indices.append(uv_index)
        face_uv_indices.append(tuple(indices))
    return uv_table, face_uv_indices


def native_display_list_prefix(
    resource: bytes,
    desc: dict,
    record_size: int,
) -> tuple[bytes, int]:
    start = int(desc["rel_dl"])
    end = start + int(desc["dl_size"])
    pos = start
    last_real_end = start
    record_count = 0
    while pos < end:
        opcode = resource[pos]
        if opcode == 0:
            pos += 1
            continue
        if (opcode & 0xF8) not in (0x80, 0x90, 0x98, 0xA0) or pos + 3 > end:
            raise ValueError(f"invalid native GX opcode {hex(opcode)}")
        count = struct.unpack_from(">H", resource, pos + 1)[0]
        command_end = pos + 3 + count * record_size
        if count < 3 or command_end > end:
            raise ValueError("native GX command exceeds its display-list range")
        record_count += count
        last_real_end = command_end
        pos = command_end
    return bytes(resource[start:last_real_end]), record_count


def encode_cmg_vertex_weights(
    raw_weights: list[dict[str, float]],
    bones: list[dict],
    palette: list[int],
) -> tuple[list[tuple[int, int, int]], bytes, dict]:
    bone_name_to_index: dict[str, int] = {}
    for bone in bones:
        for candidate in cmp_bone_name_candidates(str(bone["name"])):
            bone_name_to_index[candidate] = int(bone["idx"])
    palette_by_bone = {int(bone_index): palette_index for palette_index, bone_index in enumerate(palette)}
    encoded_vertices: list[tuple[int, int, float, float]] = []
    unknown_bones: set[str] = set()
    vertices_over_limit = 0
    for vertex_weights in raw_weights:
        combined: dict[int, float] = {}
        for bone_name, weight in vertex_weights.items():
            bone_index = next(
                (
                    bone_name_to_index[candidate]
                    for candidate in cmp_bone_name_candidates(bone_name)
                    if candidate in bone_name_to_index
                ),
                None,
            )
            if bone_index is None or bone_index not in palette_by_bone:
                unknown_bones.add(str(bone_name))
                continue
            if float(weight) > 1.0e-8:
                combined[bone_index] = combined.get(bone_index, 0.0) + float(weight)
        if len(combined) > 2:
            vertices_over_limit += 1
        strongest = sorted(combined.items(), key=lambda item: item[1], reverse=True)[:2]
        total = sum(weight for _bone, weight in strongest)
        if total <= 1.0e-8:
            raise ValueError("CMG added topology contains an unweighted vertex")
        normalized = [(bone, weight / total) for bone, weight in strongest]
        if len(normalized) == 1:
            palette_index = palette_by_bone[normalized[0][0]]
            encoded_vertices.append((palette_index, palette_index, 1.0, 0.0))
            continue
        by_palette = sorted(
            (palette_by_bone[bone], weight) for bone, weight in normalized
        )
        encoded_vertices.append(
            (by_palette[0][0], by_palette[1][0], by_palette[0][1], by_palette[1][1])
        )

    ranges: list[tuple[int, int, int]] = []
    weight_bytes = bytearray()
    start = 0
    while start < len(encoded_vertices):
        palette_a, palette_b, _weight_a, _weight_b = encoded_vertices[start]
        end = start + 1
        while end < len(encoded_vertices) and encoded_vertices[end][:2] == (palette_a, palette_b):
            end += 1
        ranges.append((end - start, palette_a, palette_b))
        if palette_a != palette_b:
            for _a, _b, weight_a, weight_b in encoded_vertices[start:end]:
                weight_bytes.extend(struct.pack(">2f", float(weight_a), float(weight_b)))
        start = end
    return ranges, bytes(weight_bytes), {
        "vertices": len(encoded_vertices),
        "ranges": len(ranges),
        "blended_vertices": sum(a != b for a, b, _wa, _wb in encoded_vertices),
        "unknown_bones": sorted(unknown_bones),
        "vertices_reduced_to_two_weights": vertices_over_limit,
    }


def grow_skinned_cmg_submesh(
    data: bytearray,
    parser: PipeworksParser,
    entries: list[dict],
    main_entry: dict,
    res_entry: dict,
    desc: dict,
    range_end_offset: int,
    geom: dict,
    bones: list[dict],
    palette: list[int],
    scale: float,
    material_range_offset: int | None,
) -> dict:
    if desc.get("attr_count") != 3 or desc.get("fmt") != 0x116:
        return {
            "status": "skipped_skinned_grow_unsupported_descriptor",
            "attr_count": desc.get("attr_count"),
            "fmt": hex(desc.get("fmt", 0)),
        }
    vertices = list(geom.get("vertices") or [])
    faces = list(geom.get("faces") or [])
    if not vertices or not faces:
        return {"status": "skipped_skinned_grow_empty_geometry"}
    if len(vertices) > 0xFFFF:
        return {"status": "skipped_skinned_grow_u16_limit", "vertices": len(vertices)}

    try:
        ranges, weight_bytes, weight_report = encode_cmg_vertex_weights(
            list(geom.get("weights") or []), bones, palette
        )
    except ValueError as exc:
        return {"status": f"skipped_skinned_grow_weights: {exc}"}
    # A large-body descriptor has a four-word draw header immediately before
    # it. The following descriptor's header is therefore the true end of this
    # descriptor's packed skin ranges, not the following descriptor itself.
    range_capacity = max(
        0,
        (int(range_end_offset) - (int(desc["main_offset"]) + 0x4C)) // 4,
    )
    if len(ranges) > range_capacity:
        return {
            "status": "skipped_skinned_grow_range_capacity",
            "ranges": len(ranges),
            "range_capacity": range_capacity,
        }

    resource = bytes(data[res_entry["offset"] : res_entry["offset"] + res_entry["size"]])
    old_vertex_count = int(desc["vertex_count"])
    added_face_indices = [
        index
        for index, face in enumerate(faces)
        if any(int(vertex_index) >= old_vertex_count for vertex_index in face)
    ]
    base_faces = [
        tuple(int(vertex_index) for vertex_index in face)
        for face in faces
        if all(int(vertex_index) < old_vertex_count for vertex_index in face)
    ]
    native_faces, _native_uvs, native_mode = decode_display_list(resource, desc)
    face_key = lambda face: tuple(sorted(int(vertex_index) for vertex_index in face))
    if collections.Counter(map(face_key, base_faces)) != collections.Counter(
        map(face_key, native_faces)
    ):
        return {
            "status": "skipped_skinned_grow_base_topology_changed",
            "native_faces": len(native_faces),
            "fbx_base_faces": len(base_faces),
        }
    if not added_face_indices:
        return {"status": "skipped_skinned_grow_no_added_faces"}

    sections = list(desc.get("section_offsets") or [])
    if len(sections) <= 15 or sections[15] < sections[12]:
        return {"status": "skipped_skinned_grow_invalid_uv_sections"}
    source_uv_count = (int(sections[15]) - int(sections[12])) // 8
    source_uvs = read_uv_table(resource, desc)[:source_uv_count]
    initial_uv_table = [(float(u), 1.0 - float(v)) for u, v in source_uvs]
    uv_table, added_face_uv_indices = indexed_face_uvs(
        geom,
        added_face_indices,
        initial_uv_table,
    )
    if len(uv_table) > 0x100:
        return {
            "status": "skipped_skinned_grow_u8_uv_limit",
            "unique_uvs": len(uv_table),
        }

    primitive_opcode = cmg_triangle_list_opcode(resource, desc)
    try:
        native_prefix, native_record_count = native_display_list_prefix(
            resource,
            desc,
            5,
        )
        added_faces = [faces[index] for index in added_face_indices]
        display_list = extend_last_idx16_idx16_idx8_triangle_command(
            native_prefix,
            added_faces,
            added_face_uv_indices,
        )
    except ValueError as exc:
        return {"status": f"skipped_skinned_grow_display_list: {exc}"}

    source_normals = parse_normals(resource, desc)
    normals = list(geom.get("normals") or [])
    while len(normals) < len(vertices):
        normals.append(None)
    for index, normal in enumerate(normals):
        if normal is None and index < len(source_normals):
            normals[index] = source_normals[index]
        if normals[index] is None:
            normals[index] = (0.0, 0.0, 1.0)

    old_size = int(res_entry["size"])
    new_rel_dl = align32(old_size)
    new_rel_vertex = align32(new_rel_dl + len(display_list))
    stored_dl_size = new_rel_vertex - new_rel_dl
    vertex_bytes = len(vertices) * 24
    new_rel_uv = new_rel_vertex + vertex_bytes + len(weight_bytes)
    uv_bytes = len(uv_table) * 8
    new_size = align32(new_rel_uv + uv_bytes)
    data[res_entry["offset"] + old_size : res_entry["offset"] + old_size] = b"\x00" * (new_size - old_size)
    patch_bundle_resource_resize(data, parser, entries, res_entry, old_size, new_size)
    res_entry["size"] = new_size

    abs_dl = int(res_entry["offset"]) + new_rel_dl
    data[abs_dl : abs_dl + len(display_list)] = display_list
    abs_vertex = int(res_entry["offset"]) + new_rel_vertex
    for index, ((x, y, z), normal) in enumerate(zip(vertices, normals)):
        nx, ny, nz = normal
        struct.pack_into(
            ">6f",
            data,
            abs_vertex + index * 24,
            float(x) / scale,
            float(y) / scale,
            float(z) / scale,
            float(nx),
            float(ny),
            float(nz),
        )
    weight_start = abs_vertex + vertex_bytes
    data[weight_start : weight_start + len(weight_bytes)] = weight_bytes
    abs_uv = int(res_entry["offset"]) + new_rel_uv
    for index, (u, v) in enumerate(uv_table):
        struct.pack_into(">2f", data, abs_uv + index * 8, float(u), float(v))

    desc_abs = int(main_entry["offset"]) + int(desc["main_offset"])
    struct.pack_into(
        ">I",
        data,
        desc_abs - 0x10,
        native_record_count + len(added_face_indices) * 3,
    )
    struct.pack_into(">I", data, desc_abs - 8, new_rel_dl)
    struct.pack_into(">I", data, desc_abs - 4, stored_dl_size)
    struct.pack_into(">I", data, desc_abs + 0x10, len(vertices))
    struct.pack_into(">I", data, desc_abs + 0x20, new_rel_vertex)
    normal_section = len(vertices) * 12 + len(weight_bytes)
    uv_section = vertex_bytes + len(weight_bytes)
    struct.pack_into(">I", data, desc_abs + 0x2C, normal_section)
    for offset in (0x30, 0x34, 0x38):
        struct.pack_into(">I", data, desc_abs + offset, uv_section)
    struct.pack_into(">I", data, desc_abs + 0x3C, uv_section + uv_bytes)
    range_start = desc_abs + 0x4C
    data[range_start : range_start + range_capacity * 4] = b"\x00" * (range_capacity * 4)
    for range_index, (count, palette_a, palette_b) in enumerate(ranges):
        struct.pack_into(
            ">I",
            data,
            range_start + range_index * 4,
            (int(count) << 16) | (int(palette_a) << 8) | int(palette_b),
        )

    if material_range_offset is not None:
        bbox_key = cmg_descriptor_bbox_key(
            [(float(x) / scale, float(y) / scale, float(z) / scale) for x, y, z in vertices]
        )
        if bbox_key:
            range_abs = int(main_entry["offset"]) + int(material_range_offset)
            for index, value in enumerate(bbox_key):
                struct.pack_into(">f", data, range_abs + index * 4, float(value))
    return {
        "status": "patched_grown_skinned_0x116_extend_last_draw",
        "old_vertices": int(desc["vertex_count"]),
        "new_vertices": len(vertices),
        "added_vertices": len(vertices) - int(desc["vertex_count"]),
        "faces": len(faces),
        "native_faces_preserved": len(native_faces),
        "added_faces": len(added_face_indices),
        "native_display_mode": native_mode,
        "native_display_bytes_preserved": len(native_prefix),
        "unique_uvs": len(uv_table),
        "skin": weight_report,
        "skin_range_capacity": range_capacity,
        "new_rel_dl": hex(new_rel_dl),
        "new_dl_size": hex(stored_dl_size),
        "new_rel_vertex": hex(new_rel_vertex),
        "new_rel_uv": hex(new_rel_uv),
        "resource_size_before": old_size,
        "resource_size_after": new_size,
        "primitive_opcode": hex(primitive_opcode),
    }


def cmg_display_list_ref_count(display_list: bytes) -> int:
    pos = 0
    refs = 0
    while pos + 3 <= len(display_list):
        opcode = display_list[pos]
        count = struct.unpack_from(">H", display_list, pos + 1)[0]
        if not opcode or not count:
            pos += 1
            continue
        refs += count
        pos += 3 + count * 2
    return refs


def grow_simple_cmg_submesh(
    data: bytearray,
    parser: PipeworksParser,
    entries: list[dict],
    res_entry: dict,
    desc: dict,
    geom: dict,
    scale: float,
    material_range_offset: int | None = None,
) -> dict:
    if desc.get("attr_count") != 2 or desc.get("fmt") != 0x102:
        return {"status": "skipped_grow_unsupported_descriptor", "attr_count": desc.get("attr_count"), "fmt": hex(desc.get("fmt", 0))}
    vertices = list(geom["vertices"])
    faces = list(geom.get("faces") or [])
    if len(vertices) > 255:
        return {"status": "skipped_grow_idx8_limit", "vertices": len(vertices)}
    if not faces:
        return {"status": "skipped_grow_no_faces"}

    resource = bytes(data[res_entry["offset"] : res_entry["offset"] + res_entry["size"]])
    old_uvs = read_uv_table(resource, desc)
    uv_by_cp = list(geom.get("uvs") or [])
    while len(uv_by_cp) < len(vertices):
        idx = len(uv_by_cp)
        uv_by_cp.append(old_uvs[idx] if idx < len(old_uvs) else (0.0, 0.0))
    for i, uv in enumerate(uv_by_cp):
        if uv is None:
            uv_by_cp[i] = old_uvs[i] if i < len(old_uvs) else (0.0, 0.0)

    new_rel_dl = align16(res_entry["size"])
    primitive_opcode = cmg_triangle_list_opcode(resource, desc)
    dl = build_idx8_2_triangle_display_list(faces, primitive_opcode)
    new_rel_vertex = align16(new_rel_dl + len(dl))
    stored_dl_size = new_rel_vertex - new_rel_dl
    new_uv_offset = len(vertices) * 12
    new_rel_uv = new_rel_vertex + new_uv_offset
    new_size = new_rel_uv + len(vertices) * 8
    old_size = res_entry["size"]
    if new_size > old_size:
        data[res_entry["offset"] + old_size : res_entry["offset"] + old_size] = b"\x00" * (new_size - old_size)
        patch_bundle_resource_resize(data, parser, entries, res_entry, old_size, new_size)
        res_entry["size"] = new_size
    abs_dl = res_entry["offset"] + new_rel_dl
    abs_vertex = res_entry["offset"] + new_rel_vertex
    data[abs_dl : abs_dl + len(dl)] = dl
    for vi, (x, y, z) in enumerate(vertices):
        struct.pack_into(">3f", data, abs_vertex + vi * 12, float(x) / scale, float(y) / scale, float(z) / scale)
    abs_uv = res_entry["offset"] + new_rel_uv
    for vi, uv in enumerate(uv_by_cp):
        u, v = uv if uv is not None else (0.0, 0.0)
        struct.pack_into(">2f", data, abs_uv + vi * 8, float(u), 1.0 - float(v))

    main_abs = entries[0]["offset"] + 0
    main_entry = next(e for e in entries if e["file_type"] == 17 and not e["is_resource"])
    desc_abs = main_entry["offset"] + desc["main_offset"]
    struct.pack_into(">I", data, desc_abs - 0x10, cmg_display_list_ref_count(dl))
    struct.pack_into(">I", data, desc_abs - 8, new_rel_dl)
    struct.pack_into(">I", data, desc_abs - 4, stored_dl_size)
    struct.pack_into(">I", data, desc_abs + 0x10, len(vertices))
    struct.pack_into(">I", data, desc_abs + 0x20, new_rel_vertex)
    for off in (0x2C, 0x30, 0x34, 0x38):
        struct.pack_into(">I", data, desc_abs + off, new_uv_offset)
    struct.pack_into(">I", data, desc_abs + 0x3C, len(vertices) * 16)
    if material_range_offset is not None:
        main_positions = [(float(x) / scale, float(y) / scale, float(z) / scale) for x, y, z in vertices]
        bbox_key = cmg_descriptor_bbox_key(main_positions)
        if bbox_key:
            main_entry = next(e for e in entries if e["file_type"] == 17 and not e["is_resource"])
            range_abs = main_entry["offset"] + material_range_offset
            for bi, value in enumerate(bbox_key):
                struct.pack_into(">f", data, range_abs + bi * 4, float(value))
    return {
        "status": "patched_grown_simple_idx8_2",
        "old_vertices": desc["vertex_count"],
        "new_vertices": len(vertices),
        "faces": len(faces),
        "new_rel_dl": hex(new_rel_dl),
        "new_dl_size": hex(stored_dl_size),
        "new_rel_vertex": hex(new_rel_vertex),
        "new_rel_uv": hex(new_rel_uv),
        "primitive_opcode": hex(primitive_opcode),
        "material_range_offset": hex(material_range_offset) if material_range_offset is not None else None,
    }


def clone_added_cmg_submesh(
    data: bytearray,
    parser: PipeworksParser,
    entries: list[dict],
    main_entry: dict,
    res_entry: dict,
    desc: dict,
    geom: dict,
    scale: float,
    table_start: int,
    record_count: int,
    material_range_offset: int | None = None,
    source_range: bytes | None = None,
    source_desc: bytes | None = None,
) -> dict:
    if desc.get("attr_count") != 2 or desc.get("fmt") != 0x102:
        return {"status": "skipped_clone_unsupported_descriptor", "attr_count": desc.get("attr_count"), "fmt": hex(desc.get("fmt", 0))}
    source_vertex_count = int(desc["vertex_count"])
    added_faces = [tuple(face) for face in geom.get("faces", []) if any(int(i) >= source_vertex_count for i in face)]
    if not added_faces:
        return {"status": "skipped_clone_no_added_faces", "old_vertices": source_vertex_count, "fbx_vertices": len(geom.get("vertices", []))}

    remap: dict[int, int] = {}
    vertices = []
    uvs = []
    for face in added_faces:
        for cp in face:
            cp = int(cp)
            if cp in remap:
                continue
            remap[cp] = len(vertices)
            vertices.append(geom["vertices"][cp])
            uv = None
            if cp < len(geom.get("uvs", [])):
                uv = geom["uvs"][cp]
            uvs.append(uv)
    if len(vertices) > 255:
        return {"status": "skipped_clone_idx8_limit", "vertices": len(vertices)}
    faces = [tuple(remap[int(i)] for i in face) for face in added_faces]

    resource = bytes(data[res_entry["offset"] : res_entry["offset"] + res_entry["size"]])
    old_uvs = read_uv_table(resource, desc)
    for i, uv in enumerate(uvs):
        if uv is None:
            source_cp = next((src for src, dst in remap.items() if dst == i), i)
            uvs[i] = old_uvs[source_cp] if source_cp < len(old_uvs) else (0.0, 0.0)

    new_rel_dl = align16(res_entry["size"])
    primitive_opcode = cmg_triangle_list_opcode(resource, desc)
    dl = build_idx8_2_triangle_display_list(faces, primitive_opcode)
    new_rel_vertex = align16(new_rel_dl + len(dl))
    stored_dl_size = new_rel_vertex - new_rel_dl
    new_uv_offset = len(vertices) * 12
    new_rel_uv = new_rel_vertex + new_uv_offset
    new_size = new_rel_uv + len(vertices) * 8
    old_size = res_entry["size"]
    if new_size > old_size:
        data[res_entry["offset"] + old_size : res_entry["offset"] + old_size] = b"\x00" * (new_size - old_size)
        patch_bundle_resource_resize(data, parser, entries, res_entry, old_size, new_size)
    abs_dl = res_entry["offset"] + new_rel_dl
    abs_vertex = res_entry["offset"] + new_rel_vertex
    data[abs_dl : abs_dl + len(dl)] = dl
    for vi, (x, y, z) in enumerate(vertices):
        struct.pack_into(">3f", data, abs_vertex + vi * 12, float(x) / scale, float(y) / scale, float(z) / scale)
    abs_uv = res_entry["offset"] + new_rel_uv
    for vi, uv in enumerate(uvs):
        u, v = uv if uv is not None else (0.0, 0.0)
        struct.pack_into(">2f", data, abs_uv + vi * 8, float(u), 1.0 - float(v))

    if material_range_offset is None:
        return {"status": "skipped_clone_missing_material_range"}
    main_blob = bytes(data[main_entry["offset"] : main_entry["offset"] + main_entry["size"]])
    if source_range is None:
        source_range = bytes(main_blob[material_range_offset : material_range_offset + 0x64])
    if source_desc is None:
        source_desc_start = desc["main_offset"] - 0x10
        source_desc = bytes(main_blob[source_desc_start : source_desc_start + 0x68])
    if len(source_range) != 0x64 or len(source_desc) != 0x68:
        return {"status": "skipped_clone_short_source_record"}

    first = struct.unpack_from(">I", main_blob, 0)[0]
    index_count = first & 0xFFFF
    material_insert = table_start + record_count * 0x64
    prelude_insert = max(0, material_insert - 0x0C)
    patch_bundle_main_insert(data, parser, entries, main_entry, prelude_insert, b"\x00" * 0x0C + source_range[:0x58])
    struct.pack_into(">I", data, main_entry["offset"], ((record_count + 1) << 16) | index_count)

    new_desc_start = main_entry["size"]
    patch_bundle_main_insert(data, parser, entries, main_entry, new_desc_start, bytes(source_desc))
    desc_abs = main_entry["offset"] + new_desc_start + 0x10
    range_abs = main_entry["offset"] + material_insert

    struct.pack_into(">I", data, desc_abs - 0x10, cmg_display_list_ref_count(dl))
    struct.pack_into(">I", data, desc_abs - 8, new_rel_dl)
    struct.pack_into(">I", data, desc_abs - 4, stored_dl_size)
    struct.pack_into(">I", data, desc_abs + 0x10, len(vertices))
    struct.pack_into(">I", data, desc_abs + 0x20, new_rel_vertex)
    for off in (0x2C, 0x30, 0x34, 0x38):
        struct.pack_into(">I", data, desc_abs + off, new_uv_offset)
    struct.pack_into(">I", data, desc_abs + 0x3C, len(vertices) * 16)
    bbox_key = cmg_descriptor_bbox_key([(float(x) / scale, float(y) / scale, float(z) / scale) for x, y, z in vertices])
    if bbox_key:
        for bi, value in enumerate(bbox_key):
            struct.pack_into(">f", data, range_abs + bi * 4, float(value))

    return {
        "status": "patched_added_as_cloned_submesh",
        "source_vertices": source_vertex_count,
        "added_vertices": len(vertices),
        "added_faces": len(faces),
        "new_submesh_index": record_count,
        "new_desc_main_offset": hex(new_desc_start + 0x10),
        "new_material_range_offset": hex(material_insert),
        "new_rel_dl": hex(new_rel_dl),
        "new_dl_size": hex(stored_dl_size),
        "new_rel_vertex": hex(new_rel_vertex),
        "new_rel_uv": hex(new_rel_uv),
        "primitive_opcode": hex(primitive_opcode),
    }


def collapse_marker_vertices_for_game(resource: bytes, desc: dict, edited_vertices: list[tuple[float, float, float]], scale: float) -> tuple[list[tuple[float, float, float]], int]:
    """Keep CMG marker/normal slots from stretching into visible triangles.

    The cleaned FBX export hides faces that touch near-origin marker vertices,
    but the game still uses the original display list. Before writeback, move
    those marker slots onto nearby real vertices in the edited mesh so any
    marker-referenced triangles collapse locally instead of stretching to origin.
    """
    original = parse_positions(resource, desc["rel_vertex"], desc["vertex_count"], position_stride(desc))
    faces, _uvs, _mode = decode_display_list(resource, desc)
    marker = {
        i
        for i, p in enumerate(original)
        if (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]) ** 0.5 < 3.0
    }
    if not marker:
        return edited_vertices, 0
    # Only activate this for chunks whose display list actually stretches from
    # real geometry to marker vertices. Body chunks may have valid low-magnitude
    # coordinates in other games; do not blanket-edit those.
    has_stretch = False
    for a, b, c in faces:
        ids = (a, b, c)
        if any(i in marker for i in ids) and any(i not in marker for i in ids):
            has_stretch = True
            break
    if not has_stretch:
        return edited_vertices, 0
    edited = list(edited_vertices)
    changed = 0
    for idx in marker:
        neighbors = []
        for face in faces:
            if idx in face:
                neighbors.extend(i for i in face if i not in marker and 0 <= i < len(edited))
        if not neighbors:
            continue
        pts = [edited[i] for i in neighbors]
        edited[idx] = (
            sum(p[0] for p in pts) / len(pts),
            sum(p[1] for p in pts) / len(pts),
            sum(p[2] for p in pts) / len(pts),
        )
        changed += 1
    return edited, changed


def parse_cmg_type3_pose_records(blob: bytes) -> dict[int, dict]:
    if len(blob) < 0x40:
        return {}
    count = struct.unpack_from(">I", blob, 0x20)[0]
    pose_start = struct.unpack_from(">I", blob, 0x3C)[0]
    if count <= 0 or count > 512 or pose_start < 0x40:
        return {}
    records = {}
    for index in range(count):
        record_pos = pose_start + index * 0x50
        translation_pos = record_pos + 0x34
        if translation_pos + 12 > len(blob):
            return {}
        records[index] = {
            "idx": index,
            "inverse_bind_pos": record_pos + 0x04,
            "translation_pos": translation_pos,
            "translation": struct.unpack_from(">3f", blob, translation_pos),
        }
    return records


def parse_cmg_type4_rest_records(blob: bytes) -> dict[int, dict]:
    if len(blob) < 0x38:
        return {}
    count = struct.unpack_from(">I", blob, 0x2C)[0]
    if count <= 0 or count > 512 or 0x38 + count * 4 > len(blob):
        return {}
    records = {}
    for table_index in range(count):
        record_pos = struct.unpack_from(">I", blob, 0x38 + table_index * 4)[0]
        if record_pos + 0x24 > len(blob):
            continue
        index = struct.unpack_from(">i", blob, record_pos)[0]
        translation_pos = record_pos + 0x18
        if index < 0 or index > 512 or translation_pos + 12 > len(blob):
            continue
        records[index] = {
            "idx": index,
            "translation_pos": translation_pos,
            "translation": struct.unpack_from(">3f", blob, translation_pos),
        }
    return records


def patch_cmg_skeleton_from_fbx(
    data: bytearray,
    entries: list[dict],
    roots,
    bones: list[dict],
    scale: float,
) -> tuple[dict, dict[int, tuple[float, float, float]]]:
    hierarchy_entry = next(
        (
            entry
            for entry in entries
            if entry["file_type"] == 3
            and not entry["is_resource"]
            and "SKELETON" in str(entry["name"]).upper()
            and "CAMERA" not in str(entry["name"]).upper()
        ),
        None,
    )
    pose_entry = next(
        (
            entry
            for entry in entries
            if entry["file_type"] == 4
            and not entry["is_resource"]
            and "SKELETON" in str(entry["name"]).upper()
        ),
        None,
    )
    if not hierarchy_entry or not pose_entry or not bones:
        return {"status": "skipped_missing_cmg_skeleton"}, {}

    hierarchy_start = int(hierarchy_entry["offset"])
    hierarchy_blob = bytes(
        data[hierarchy_start : hierarchy_start + int(hierarchy_entry["size"])]
    )
    hierarchy_records = parse_cmg_type3_pose_records(hierarchy_blob)
    pose_start = int(pose_entry["offset"])
    pose_blob = bytes(data[pose_start : pose_start + int(pose_entry["size"])])
    pose_records = parse_cmg_type4_rest_records(pose_blob)
    models = fbx_bone_models(roots)
    matched = {}
    missing = []
    for bone in bones:
        model = find_cmp_bone_model(models, str(bone["name"]))
        if model is None:
            missing.append(str(bone["name"]))
        else:
            matched[int(bone["idx"])] = model

    locked_indices = {int(bone["idx"]) for bone in bones if int(bone["idx"]) not in matched}
    safe_scale = scale if abs(scale) > 1.0e-8 else 1.0
    raw_local_translations = {}
    raw_bones = [dict(bone) for bone in bones]
    raw_by_index = {int(bone["idx"]): bone for bone in raw_bones}
    for bone in bones:
        index = int(bone["idx"])
        if index in locked_indices:
            continue
        translation = p_values(matched[index].child("Properties70"), "Lcl Translation")
        if not translation or len(translation) < 3:
            continue
        new_translation = tuple(float(value) / safe_scale for value in translation[:3])
        raw_local_translations[index] = new_translation
        raw_by_index[index]["t"] = new_translation

    old_globals = global_matrices(bones, 1.0)
    raw_globals = global_matrices(raw_bones, 1.0)
    effective_local_translations = dict(raw_local_translations)
    ignored_child_compensations = []

    def global_delta(index: int) -> tuple[float, float, float]:
        return tuple(
            float(raw_globals[index][axis][3] - old_globals[index][axis][3])
            for axis in range(3)
        )

    def delta_length(delta: tuple[float, float, float]) -> float:
        return math.sqrt(sum(value * value for value in delta))

    # Blender keeps disconnected Edit-Mode children in world space by writing
    # an equal and opposite local translation. Ignore that compensation so a
    # moved CMG parent carries its native child hierarchy like Pose Mode.
    for bone in bones:
        index = int(bone["idx"])
        parent = int(bone["parent"])
        new_translation = raw_local_translations.get(index)
        if new_translation is None or parent == index or parent not in raw_globals:
            continue
        old_translation = tuple(float(value) for value in bone["t"])
        if not any(abs(new_translation[axis] - old_translation[axis]) > 1.0e-3 for axis in range(3)):
            continue
        parent_delta = global_delta(parent)
        child_delta = global_delta(index)
        if delta_length(parent_delta) <= 1.0e-3 or delta_length(child_delta) > 1.0e-3:
            continue
        effective_local_translations[index] = old_translation
        ignored_child_compensations.append(
            {
                "idx": index,
                "name": str(bone["name"]),
                "parent_index": parent,
                "discarded_local_translation": new_translation,
                "preserved_local_translation": old_translation,
            }
        )

    changed_bones = []
    new_bones = [dict(bone) for bone in bones]
    new_by_index = {int(bone["idx"]): bone for bone in new_bones}
    rotations_seen = 0
    unchanged = 0
    for bone in bones:
        index = int(bone["idx"])
        if index in locked_indices:
            continue
        new_translation = effective_local_translations.get(index)
        if new_translation is None:
            continue
        rotation = p_values(matched[index].child("Properties70"), "Lcl Rotation")
        if rotation and any(abs(float(value)) > 1.0e-6 for value in rotation[:3]):
            rotations_seen += 1
        old_translation = tuple(float(value) for value in bone["t"])
        if not any(abs(new_translation[axis] - old_translation[axis]) > 1.0e-3 for axis in range(3)):
            unchanged += 1
            continue
        hierarchy_record = hierarchy_records.get(index)
        pose_record = pose_records.get(index)
        if hierarchy_record is None or pose_record is None:
            continue
        if (
            any(abs(float(hierarchy_record["translation"][axis]) - old_translation[axis]) > 0.05 for axis in range(3))
            or any(abs(float(pose_record["translation"][axis]) - old_translation[axis]) > 0.05 for axis in range(3))
        ):
            continue
        struct.pack_into(
            ">3f", data, hierarchy_start + int(hierarchy_record["translation_pos"]), *new_translation
        )
        struct.pack_into(
            ">3f", data, pose_start + int(pose_record["translation_pos"]), *new_translation
        )
        new_by_index[index]["t"] = new_translation
        changed_bones.append(
            {
                "idx": index,
                "name": str(bone["name"]),
                "old_local_translation": old_translation,
                "new_local_translation": new_translation,
            }
        )

    new_globals = global_matrices(new_bones, 1.0)
    global_deltas = {}
    inverse_bind_indices = []
    for index, old_matrix in old_globals.items():
        new_matrix = new_globals.get(index)
        if new_matrix is None:
            continue
        delta = tuple(float(new_matrix[axis][3] - old_matrix[axis][3]) for axis in range(3))
        if not any(abs(value) > 1.0e-5 for value in delta):
            continue
        global_deltas[int(index)] = delta
        hierarchy_record = hierarchy_records.get(int(index))
        if hierarchy_record is None:
            continue
        inverse_bind_pos = hierarchy_start + int(hierarchy_record["inverse_bind_pos"])
        stored = struct.unpack_from(">12f", data, inverse_bind_pos)
        old_inverse = cmp_inverse_bind_values(old_matrix)
        new_inverse = cmp_inverse_bind_values(new_matrix)
        updated = tuple(
            native + (new_value - old_value)
            for native, old_value, new_value in zip(stored, old_inverse, new_inverse)
        )
        struct.pack_into(">12f", data, inverse_bind_pos, *updated)
        inverse_bind_indices.append(int(index))

    return {
        "status": "patched" if changed_bones else "skipped_no_changed_bone_positions",
        "position_bones_patched": len(changed_bones),
        "changed_bones": changed_bones,
        "bones_with_global_delta": len(global_deltas),
        "inverse_bind_bones_patched": len(inverse_bind_indices),
        "inverse_bind_bone_indices": inverse_bind_indices,
        "deleted_or_missing_bones_preserved": missing[:20],
        "deleted_or_missing_count": len(missing),
        "unchanged_bones_seen": unchanged,
        "rotation_values_seen_but_preserved": rotations_seen,
        "hierarchy_compensations_ignored": len(ignored_child_compensations),
        "hierarchy_compensation_bones": ignored_child_compensations,
        "fbx_export_scale": safe_scale,
        "writeback_scope": "big-endian Type-3 inverse binds and Type-3/Type-4 rest translations",
    }, global_deltas


def patch_cmg_animation_translation_tracks(
    data: bytearray, entries: list[dict], skeleton_patch: dict
) -> dict:
    local_deltas = {
        int(bone["idx"]): tuple(
            float(bone["new_local_translation"][axis])
            - float(bone["old_local_translation"][axis])
            for axis in range(3)
        )
        for bone in skeleton_patch.get("changed_bones") or []
    }
    local_deltas = {
        bone: delta
        for bone, delta in local_deltas.items()
        if any(abs(value) > 1.0e-6 for value in delta)
    }
    if not local_deltas:
        return {"status": "skipped_no_changed_bone_positions"}

    clips_patched = 0
    tracks_patched = 0
    keys_patched = 0
    malformed_clips = []
    for entry in entries:
        if entry["file_type"] != 4 or entry["is_resource"]:
            continue
        base = int(entry["offset"])
        size = int(entry["size"])
        if base < 0 or size < 0 or base + size > len(data):
            continue
        blob = bytes(data[base : base + size])
        if len(blob) < 0x24 or struct.unpack_from(">I", blob, 0x20)[0] != 2:
            continue
        tracks = parse_cmg_animation_translation_tracks(blob)
        if tracks is None:
            malformed_clips.append(str(entry["name"]))
            continue
        clip_tracks = 0
        clip_keys = 0
        for track in tracks:
            delta = local_deltas.get(int(track["bone"]))
            if delta is None:
                continue
            key_count = int(track["key_count"])
            records_pos = base + int(track["records_pos"])
            old_scales = tuple(float(value) for value in track["scales"])
            new_scales = list(old_scales)
            axis_values = [[], [], []]
            raw_records = []
            for key_index in range(key_count):
                record_pos = records_pos + key_index * 8
                time, x, y, z = struct.unpack_from(">Hhhh", data, record_pos)
                raw_records.append((record_pos, time, (x, y, z)))
                for axis, raw in enumerate((x, y, z)):
                    axis_values[axis].append(float(raw) / 32767.0 * old_scales[axis] + delta[axis])
            for axis in range(3):
                if abs(delta[axis]) > 1.0e-8:
                    new_scales[axis] = max(
                        abs(old_scales[axis]),
                        max(abs(value) for value in axis_values[axis]),
                        1.0e-12,
                    )
            scale_pos = base + int(track["scale_pos"])
            if int(track.get("scale_count", 3)) == 1:
                shared_scale = max(new_scales)
                new_scales = [shared_scale] * 3
                struct.pack_into(">f", data, scale_pos, shared_scale)
            else:
                struct.pack_into(">3f", data, scale_pos, *new_scales)
            for key_index, (record_pos, _time, _raw) in enumerate(raw_records):
                for axis in range(3):
                    if abs(delta[axis]) <= 1.0e-8:
                        continue
                    quantized = int(round(axis_values[axis][key_index] / new_scales[axis] * 32767.0))
                    struct.pack_into(">h", data, record_pos + 2 + axis * 2, max(-32767, min(32767, quantized)))
            clip_tracks += 1
            clip_keys += key_count
        if clip_tracks:
            clips_patched += 1
            tracks_patched += clip_tracks
            keys_patched += clip_keys
    return {
        "status": "patched" if clips_patched else "skipped_no_matching_animation_tracks",
        "clips_patched": clips_patched,
        "tracks_patched": tracks_patched,
        "keys_patched": keys_patched,
        "bones_retargeted": sorted(local_deltas),
        "malformed_clips_skipped": malformed_clips,
    }


def bake_cmg_vertices_for_skeleton_move(
    geoms: dict[int, dict],
    descs: list[dict],
    main: bytes,
    resource: bytes,
    bone_palette: list[int],
    global_deltas: dict[int, tuple[float, float, float]],
    scale: float,
) -> dict:
    if not global_deltas:
        return {"status": "skipped_no_global_bone_delta", "vertices_moved": 0}
    ordered = sorted(descs, key=lambda desc: int(desc["main_offset"]))
    next_offsets = {
        id(desc): (
            int(ordered[index + 1]["main_offset"])
            if index + 1 < len(ordered)
            else len(main)
        )
        for index, desc in enumerate(ordered)
    }
    vertices_moved = 0
    max_weighted_delta = 0.0
    submeshes = []
    for index, desc in enumerate(descs):
        geom = geoms.get(index)
        if not geom or not bone_palette:
            continue
        ranges = cmg_descriptor_skin_ranges(
            main, desc, next_offsets.get(id(desc), len(main)), len(bone_palette)
        )
        weights = cmg_vertex_weights_from_ranges(resource, desc, ranges, bone_palette)
        if not weights:
            continue
        moved_here = 0
        vertices = list(geom["vertices"])
        for vertex_index, influences in enumerate(weights[: len(vertices)]):
            dx = dy = dz = 0.0
            for bone, weight in influences:
                delta = global_deltas.get(int(bone))
                if delta is None:
                    continue
                dx += delta[0] * float(weight) * scale
                dy += delta[1] * float(weight) * scale
                dz += delta[2] * float(weight) * scale
            magnitude = math.sqrt(dx * dx + dy * dy + dz * dz)
            if magnitude <= 1.0e-5:
                continue
            x, y, z = vertices[vertex_index]
            vertices[vertex_index] = (x + dx, y + dy, z + dz)
            moved_here += 1
            vertices_moved += 1
            max_weighted_delta = max(max_weighted_delta, magnitude / scale)
        if moved_here:
            geom["vertices"] = vertices
            submeshes.append({"submesh": index, "vertices_moved": moved_here})
    return {
        "status": "patched" if vertices_moved else "skipped_no_weighted_vertices",
        "vertices_moved": vertices_moved,
        "bones_with_global_delta": len(global_deltas),
        "max_weighted_delta": max_weighted_delta,
        "submeshes": submeshes,
    }


def patch_cmg_positions(
    original: Path,
    fbx: Path,
    out: Path,
    scale: float,
    keep_mesh_in_place: bool = False,
) -> dict:
    parser = PipeworksParser(str(original))
    entries = parser.parse()
    data = bytearray(parser.file_data or b"")
    main_entry, res_entry = find_type17(entries)
    main = bytes(data[main_entry["offset"] : main_entry["offset"] + main_entry["size"]])
    resource = bytes(data[res_entry["offset"] : res_entry["offset"] + res_entry["size"]])
    descs = find_mesh_descriptors(main, res_entry["size"])
    bones, _globals = parse_cmg_skeleton(parser, entries, scale)
    bone_palette = cmg_mesh_bone_palette(main, bones)
    roots, _fbx_version = parse_fbx(fbx)
    skeleton_patch, skeleton_global_deltas = patch_cmg_skeleton_from_fbx(
        data, entries, roots, bones, scale
    )
    animation_translation_patch = patch_cmg_animation_translation_tracks(
        data, entries, skeleton_patch
    )
    first = struct.unpack_from(">I", main, 0)[0] if len(main) >= 4 else 0
    record_count = first >> 16
    index_count = first & 0xFFFF
    table_start = 0x1C + index_count * 4
    material_range_by_desc_id: dict[int, int] = {}
    if record_count > 0 and table_start + record_count * 0x64 <= len(main):
        for order_index, ordered_desc in enumerate(sorted(descs, key=lambda d: d["main_offset"])):
            if order_index < record_count:
                material_range_by_desc_id[id(ordered_desc)] = table_start + order_index * 0x64
    geoms = geometry_payloads(fbx)
    mesh_skeleton_bake = (
        {"status": "skipped_keep_mesh_in_place", "vertices_moved": 0}
        if keep_mesh_in_place
        else bake_cmg_vertices_for_skeleton_move(
            geoms,
            descs,
            main,
            resource,
            bone_palette,
            skeleton_global_deltas,
            scale,
        )
    )
    patched = []
    skipped = []
    clone_jobs = []
    for i, desc in enumerate(descs):
        geom = geoms.get(i)
        if not geom:
            skipped.append({"submesh": i, "status": "missing_fbx_geometry"})
            continue
        if len(geom["vertices"]) > desc["vertex_count"]:
            if desc.get("attr_count") == 3 and desc.get("fmt") == 0x116:
                range_end_offset = (
                    int(descs[i + 1]["main_offset"]) - 0x10
                    if i + 1 < len(descs)
                    else int(main_entry["size"])
                )
                item = {"submesh": i}
                item.update(
                    grow_skinned_cmg_submesh(
                        data,
                        parser,
                        entries,
                        main_entry,
                        res_entry,
                        desc,
                        range_end_offset,
                        geom,
                        bones,
                        bone_palette,
                        scale,
                        material_range_by_desc_id.get(id(desc)),
                    )
                )
                if item.get("status", "").startswith("patched_"):
                    patched.append(item)
                else:
                    skipped.append(item)
                continue
            source_range = None
            material_range_offset = material_range_by_desc_id.get(id(desc))
            if material_range_offset is not None:
                source_range = bytes(main[material_range_offset : material_range_offset + 0x64])
            source_desc_start = desc["main_offset"] - 0x10
            source_desc = bytes(main[source_desc_start : source_desc_start + 0x68])
            base_vertices, collapsed_markers = collapse_marker_vertices_for_game(resource, desc, geom["vertices"][: desc["vertex_count"]], scale)
            abs_vertex = res_entry["offset"] + desc["rel_vertex"]
            stride = position_stride(desc)
            for vi, (x, y, z) in enumerate(base_vertices):
                struct.pack_into(">3f", data, abs_vertex + vi * stride, x / scale, y / scale, z / scale)
            item = {
                "submesh": i,
                "vertices": desc["vertex_count"],
                "added_geometry": "queued_cloned_submesh",
                "fbx_vertices": len(geom["vertices"]),
            }
            if collapsed_markers:
                item["collapsed_marker_vertices"] = collapsed_markers
            patched.append(item)
            clone_jobs.append((i, desc, geom, material_range_offset, source_range, source_desc))
            continue
        if len(geom["vertices"]) != desc["vertex_count"]:
            skipped.append(
                {
                    "submesh": i,
                    "status": f"skipped_vertex_count_{len(geom['vertices'])}_expected_{desc['vertex_count']}",
                }
            )
            continue
        vertices, collapsed_markers = collapse_marker_vertices_for_game(resource, desc, geom["vertices"], scale)
        abs_vertex = res_entry["offset"] + desc["rel_vertex"]
        stride = position_stride(desc)
        for vi, (x, y, z) in enumerate(vertices):
            struct.pack_into(">3f", data, abs_vertex + vi * stride, x / scale, y / scale, z / scale)
        item = {"submesh": i, "vertices": len(geom["vertices"])}
        if collapsed_markers:
            item["collapsed_marker_vertices"] = collapsed_markers
        patched.append(item)
    next_record_count = record_count
    successful_clones = 0
    for i, desc, geom, material_range_offset, source_range, source_desc in clone_jobs:
        item = {"submesh": i}
        item.update(
            clone_added_cmg_submesh(
                data,
                parser,
                entries,
                main_entry,
                res_entry,
                desc,
                geom,
                scale,
                table_start,
                next_record_count,
                material_range_offset,
                source_range,
                source_desc,
            )
        )
        if item.get("status", "").startswith("patched_"):
            patched.append(item)
            successful_clones += 1
            next_record_count += 1
        else:
            skipped.append(item)
    if successful_clones:
        descriptor_ref_map = {}
        descriptor_shift = successful_clones * 0x64
        for desc in descs:
            for ref in (int(desc["main_offset"]) - 0x10, int(desc["main_offset"])):
                descriptor_ref_map[ref] = ref + descriptor_shift
        patched_refs = patch_type2_loader_refs(data, entries, descriptor_ref_map)
        if patched_refs:
            patched.append({"status": "patched_type2_loader_refs", "references": patched_refs, "shift": hex(descriptor_shift)})
    from cmg_animation_import import import_cmg_actions

    rebuilt_data, action_animation_patch = import_cmg_actions(
        bytes(data), entries, bones, fbx, scale, skeleton_patch=skeleton_patch
    )
    data = bytearray(rebuilt_data)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    return {
        "output": str(out),
        "patched": patched,
        "skipped": skipped,
        "skeleton_patch": skeleton_patch,
        "mesh_skeleton_bake": mesh_skeleton_bake,
        "animation_translation_patch": animation_translation_patch,
        "action_animation_patch": action_animation_patch,
        "note": "CMG writeback patches mesh positions, existing rest-bone translations, inverse binds, and supported animations.",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Import same-topology FBX edits into a GameCube CMG copy")
    ap.add_argument("fbx")
    ap.add_argument("original")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=float, default=10.0)
    ap.add_argument(
        "--keep-mesh-in-place",
        action="store_true",
        help="Import rest-bone positions without automatically moving weighted vertices",
    )
    args = ap.parse_args(argv)
    original = Path(args.original)
    out = Path(args.out)
    if original.resolve() != out.resolve():
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, out)
    report = patch_cmg_positions(
        out,
        Path(args.fbx),
        out,
        args.scale,
        keep_mesh_in_place=args.keep_mesh_in_place,
    )
    for key, value in report.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
