"""Inject geometry from one Shapes.BDG into another Shapes.BDG template.

The template keeps ownership of its skeleton, materials, textures, and other
resources. Donor geometry is rebuilt into the template's existing Type-17 draw
streams and is rigidly bound to template bone 0 for a safe first import.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
import tempfile
import zipfile

from bdg_to_fbx_extract_all import (
    choose_meshes,
    find_skeleton,
    find_strtab,
    parse_vertex_by_layout,
)
from cmp_custom_model import extract_custom_triangles
from cmp_fbx_import import cmp_bone_name_candidates
from fbx_to_bdg_import import find_first, parse_fbx
from parser_core import PipeworksParser
from topology_bdg_writer import (
    _align,
    _build_triangle_commands,
    _find_mesh_descriptors,
    _mesh_resource_entry,
    _p32,
    _pack_index_record,
    _patch_native_vertex_record,
    _strict_validate_descriptor_streams,
    _u32,
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def _skeleton(data: bytes):
    _off, _count, strings = find_strtab(data)
    return find_skeleton(data, strings)


def _type6_material_flags(data: bytes, entries: list[dict]) -> list[dict]:
    materials = []
    for entry in entries:
        if entry.get("file_type") != 6 or entry.get("is_resource"):
            continue
        offset = int(entry["offset"])
        if offset + 8 > len(data):
            continue
        materials.append({
            "name": entry.get("name"),
            "offset": offset,
            "flags": int(data[offset + 7]),
            "two_sided": bool(data[offset + 7] & 0x01),
        })
    return materials


def _decoded_mesh(path: Path):
    data = path.read_bytes()
    _skel_base, _skel_root, bones = _skeleton(data)
    submeshes, skipped = choose_meshes(data, len(bones))
    if skipped:
        # Descriptor-owned streams are authoritative; scanner-only candidates
        # are expected and are not additional draw sections.
        skipped = list(skipped)
    return data, bones, submeshes, skipped


def _find_mesh_summary_records(
    data: bytes,
    main_entry: dict,
    descs: list[dict],
    endian: str,
) -> list[int]:
    """Locate the Type-17 draw summaries paired with the stream descriptors.

    Each summary is 0x70 bytes. Its first two words mirror the corresponding
    descriptor's GX record count minus two and vertex count. The game consults
    these summaries before the later stream descriptors, so both copies must be
    updated when custom geometry changes a stream's size.
    """
    if not descs:
        raise ValueError("Template Type-17 mesh has no stream descriptors")
    main_start = int(main_entry["offset"])
    main_size = int(main_entry["size"])
    main = data[main_start:main_start + main_size]
    sequence_size = (len(descs) - 1) * 0x70 + 8
    candidates = []
    for start in range(0, max(0, len(main) - sequence_size + 1), 4):
        matches = True
        for stream_index, desc in enumerate(descs):
            summary = start + stream_index * 0x70
            expected_records = max(0, int(desc["record_count"]) - 2)
            if (
                _u32(main, summary, endian) != expected_records
                or _u32(main, summary + 4, endian) != int(desc["v_count"])
            ):
                matches = False
                break
        if matches:
            candidates.append(start)
    if len(candidates) != 1:
        raise ValueError(
            "Could not uniquely locate the Type-17 draw-summary sequence "
            f"({len(candidates)} candidates for {len(descs)} streams)"
        )
    first = main_start + candidates[0]
    return [first + stream_index * 0x70 for stream_index in range(len(descs))]


def _stream_bounds(groups: list[dict], position_scale: float) -> dict:
    positions = [
        tuple(float(value) / float(position_scale) for value in position)
        for group in groups
        for position in group["vertices"]
    ]
    if not positions:
        raise ValueError("Cannot calculate bounds for an empty custom stream")
    minimum = tuple(min(position[axis] for position in positions) for axis in range(3))
    maximum = tuple(max(position[axis] for position in positions) for axis in range(3))
    center = tuple((minimum[axis] + maximum[axis]) * 0.5 for axis in range(3))
    radius_squared = sum(
        ((maximum[axis] - minimum[axis]) * 0.5) ** 2 for axis in range(3)
    )
    return {
        "min": minimum,
        "max": maximum,
        "center": center,
        "radius_squared": radius_squared,
    }


def _pf32(data: bytearray, offset: int, value: float, endian: str) -> None:
    struct.pack_into(f"{endian}f", data, offset, float(value))


def _partition_contiguous(groups: list[dict], bucket_count: int) -> list[list[dict]]:
    if bucket_count <= 0:
        raise ValueError("Template has no mesh streams")
    if not groups:
        raise ValueError("Donor has no mesh streams")
    if bucket_count == 1:
        return [groups]

    weights = [max(1, len(g.get("faces") or [])) for g in groups]
    total = sum(weights)
    buckets: list[list[dict]] = []
    start = 0
    consumed = 0
    for bucket_i in range(bucket_count):
        remaining_buckets = bucket_count - bucket_i
        remaining_groups = len(groups) - start
        if remaining_groups <= 0:
            buckets.append([])
            continue
        if remaining_buckets == 1:
            end = len(groups)
        else:
            target = total * (bucket_i + 1) / bucket_count
            end = start + 1
            running = consumed + weights[start]
            max_end = len(groups) - (remaining_buckets - 1)
            while end < max_end:
                next_running = running + weights[end]
                if abs(next_running - target) > abs(running - target):
                    break
                running = next_running
                end += 1
        bucket = groups[start:end]
        buckets.append(bucket)
        consumed += sum(weights[start:end])
        start = end
    return buckets


def _corner_key(corner: dict) -> tuple:
    return (
        tuple(map(float, corner["position"])),
        tuple(map(float, corner["normal"])),
        tuple(map(float, corner["uv"])),
        tuple(sorted((str(name), float(weight)) for name, weight in corner["weights"].items() if float(weight) > 1e-8)),
    )


def _convert_fbx_axes_to_bdg(roots, mesh_groups: list[list[dict]]) -> dict:
    settings = find_first(roots, "GlobalSettings")
    properties = settings.child("Properties70") if settings else None
    values = {}
    if properties:
        for prop in properties.children_named("P"):
            if prop.props:
                values[str(prop.props[0])] = prop.props[-1]
    up_axis = int(values.get("UpAxis", 2))
    up_sign = int(values.get("UpAxisSign", 1))
    front_axis = int(values.get("FrontAxis", 1))
    front_sign = int(values.get("FrontAxisSign", -1))
    coord_axis = int(values.get("CoordAxis", 0))
    coord_sign = int(values.get("CoordAxisSign", 1))
    axes = (coord_axis, front_axis, up_axis)
    if sorted(axes) != [0, 1, 2] or any(sign not in (-1, 1) for sign in (coord_sign, front_sign, up_sign)):
        raise ValueError(
            "Replacement FBX has unsupported global axis metadata: "
            f"coord={coord_axis}/{coord_sign}, front={front_axis}/{front_sign}, up={up_axis}/{up_sign}"
        )

    def convert(vector):
        return (
            float(vector[coord_axis]) * coord_sign,
            -float(vector[front_axis]) * front_sign,
            float(vector[up_axis]) * up_sign,
        )

    converted = (coord_axis, coord_sign, front_axis, front_sign, up_axis, up_sign) != (0, 1, 1, -1, 2, 1)
    if converted:
        for group in mesh_groups:
            for triangle in group:
                for corner in triangle["corners"]:
                    corner["position"] = convert(corner["position"])
                    corner["normal"] = convert(corner["normal"])
    return {
        "source_coord_axis": coord_axis,
        "source_coord_sign": coord_sign,
        "source_front_axis": front_axis,
        "source_front_sign": front_sign,
        "source_up_axis": up_axis,
        "source_up_sign": up_sign,
        "converted_to_bdg_z_up": converted,
    }


def _partition_fbx_triangles(
    mesh_groups: list[list[dict]],
    template_sms: list[dict],
    template_materials: list[dict],
) -> tuple[list[list[dict]], dict]:
    triangles = [triangle for group in mesh_groups for triangle in group]
    if len(triangles) < len(template_sms):
        raise ValueError(
            f"Replacement FBX has {len(triangles)} triangles but the template requires "
            f"{len(template_sms)} non-empty draw streams"
        )

    material_count = max(1, len(template_materials))
    stream_materials = [
        min(material_count - 1, stream_index * material_count // len(template_sms))
        for stream_index in range(len(template_sms))
    ]
    streams_by_material = collections.defaultdict(list)
    for stream_index, material_index in enumerate(stream_materials):
        streams_by_material[material_index].append(stream_index)

    source_materials = []
    for triangle in triangles:
        key = (int(triangle.get("material_index", 0)), str(triangle.get("material_name", "")))
        if key not in source_materials:
            source_materials.append(key)

    def normalized(value: str) -> str:
        return "".join(ch for ch in str(value).lower() if ch.isalnum())

    template_names = [normalized(material.get("name", "")) for material in template_materials]

    def target_material(triangle: dict) -> int:
        source_index = int(triangle.get("material_index", 0))
        source_name = normalized(triangle.get("material_name", ""))
        for index, template_name in enumerate(template_names):
            if source_name and template_name and (source_name in template_name or template_name in source_name):
                return index
        special_tokens = ("shard", "crystal", "spike")
        if any(token in source_name for token in special_tokens):
            for index, template_name in enumerate(template_names):
                if any(token in template_name for token in special_tokens):
                    return index
        if len(source_materials) == material_count and 0 <= source_index < material_count:
            return source_index
        return 0

    triangles_by_material = collections.defaultdict(list)
    for triangle in triangles:
        triangles_by_material[target_material(triangle)].append(triangle)

    buckets: list[list[dict]] = [[] for _sm in template_sms]
    unique_keys: list[set[tuple]] = [set() for _sm in template_sms]
    capacities = [256 if int(sm.get("index_width", 6)) in (3, 4) else 65536 for sm in template_sms]
    targets = [max(1, len(sm.get("faces") or [])) for sm in template_sms]

    for material_index, material_triangles in triangles_by_material.items():
        candidate_streams = streams_by_material.get(material_index) or streams_by_material[0]
        target_total = sum(targets[index] for index in candidate_streams)
        scaled_targets = {
            index: max(1.0, len(material_triangles) * targets[index] / target_total)
            for index in candidate_streams
        }
        for triangle in material_triangles:
            triangle_keys = {_corner_key(corner) for corner in triangle["corners"]}
            candidates = [
                stream_index
                for stream_index in candidate_streams
                if len(unique_keys[stream_index] | triangle_keys) <= capacities[stream_index]
            ]
            if not candidates:
                raise ValueError(
                    f"Replacement FBX material {material_index} cannot fit its template native streams"
                )
            stream_index = min(
                candidates,
                key=lambda index: (
                    len(buckets[index]) / scaled_targets[index],
                    len(buckets[index]),
                    index,
                ),
            )
            buckets[stream_index].append(triangle)
            unique_keys[stream_index].update(triangle_keys)

    hidden_fillers = []
    filler_source = triangles[0]
    for stream_index, bucket in enumerate(buckets):
        if bucket:
            continue
        # Native draw descriptors are retained. Keep an unused material stream
        # structurally valid with a zero-area triangle instead of assigning a
        # visible body polygon to a crystal/shard shader.
        corner = dict(filler_source["corners"][0])
        position = tuple(map(float, corner["position"]))
        epsilon = 1.0e-4
        corner_b = dict(corner)
        corner_c = dict(corner)
        corner_b["position"] = (position[0] + epsilon, position[1], position[2])
        corner_c["position"] = (position[0], position[1] + epsilon, position[2])
        uv = tuple(map(float, corner.get("uv", (0.0, 0.0))))
        corner_b["uv"] = (uv[0] + 0.01, uv[1])
        corner_c["uv"] = (uv[0], uv[1] + 0.01)
        filler = {
            "mesh": filler_source.get("mesh", ""),
            "material_index": stream_materials[stream_index],
            "material_name": "__hidden_degenerate_filler__",
            "corners": [corner, corner_b, corner_c],
        }
        bucket.append(filler)
        hidden_fillers.append(stream_index)

    return buckets, {
        "template_materials": [material.get("name") for material in template_materials],
        "stream_material_indices": stream_materials,
        "source_materials": [name for _index, name in source_materials],
        "real_triangles_by_material": {
            str(index): len(rows) for index, rows in sorted(triangles_by_material.items())
        },
        "hidden_degenerate_filler_streams": hidden_fillers,
    }


def _group_from_fbx_triangles(triangles: list[dict], group_index: int) -> dict:
    vertices = []
    normals = []
    uvs = []
    weights = []
    faces = []
    local_by_corner: dict[tuple, int] = {}
    for triangle in triangles:
        face = []
        for corner in triangle["corners"]:
            key = _corner_key(corner)
            local_index = local_by_corner.get(key)
            if local_index is None:
                local_index = len(vertices)
                local_by_corner[key] = local_index
                vertices.append(tuple(map(float, corner["position"])))
                normals.append(tuple(map(float, corner["normal"])))
                uvs.append(tuple(map(float, corner["uv"])))
                weights.append(dict(corner["weights"]))
            face.append(local_index)
        faces.append(tuple(face))
    group = {
        "source": {"custom_group": group_index},
        "vertices": vertices,
        "normals": normals,
        "uvs": uvs,
        "weights": weights,
        "faces": faces,
    }
    group["tangents"], group["bitangents"] = _compute_tangent_basis(group)
    return group


def _normalize_vector(value, fallback=(0.0, 0.0, 1.0)):
    length = math.sqrt(sum(float(component) ** 2 for component in value))
    if length <= 1.0e-12:
        return tuple(map(float, fallback))
    return tuple(float(component) / length for component in value)


def _cross_vector(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _fallback_tangent(normal):
    axis = (0.0, 0.0, 1.0) if abs(normal[2]) < 0.9 else (0.0, 1.0, 0.0)
    return _normalize_vector(_cross_vector(axis, normal), (1.0, 0.0, 0.0))


def _compute_tangent_basis(group: dict) -> tuple[list[tuple], list[tuple]]:
    """Build the native tangent frame from positions and stored (V-flipped) UVs."""
    vertices = list(group.get("vertices") or [])
    normals = [_normalize_vector(value) for value in group.get("normals") or []]
    uvs = [(float(value[0]), 1.0 - float(value[1])) for value in group.get("uvs") or []]
    tangent_sums = [[0.0, 0.0, 0.0] for _ in vertices]
    bitangent_sums = [[0.0, 0.0, 0.0] for _ in vertices]

    for face in group.get("faces") or []:
        i0, i1, i2 = map(int, face)
        p0, p1, p2 = (vertices[index] for index in (i0, i1, i2))
        uv0, uv1, uv2 = (uvs[index] for index in (i0, i1, i2))
        edge1 = tuple(float(p1[axis]) - float(p0[axis]) for axis in range(3))
        edge2 = tuple(float(p2[axis]) - float(p0[axis]) for axis in range(3))
        du1, dv1 = uv1[0] - uv0[0], uv1[1] - uv0[1]
        du2, dv2 = uv2[0] - uv0[0], uv2[1] - uv0[1]
        determinant = du1 * dv2 - du2 * dv1
        if abs(determinant) <= 1.0e-12:
            continue
        reciprocal = 1.0 / determinant
        tangent = tuple((edge1[axis] * dv2 - edge2[axis] * dv1) * reciprocal for axis in range(3))
        bitangent = tuple((edge2[axis] * du1 - edge1[axis] * du2) * reciprocal for axis in range(3))
        for index in (i0, i1, i2):
            for axis in range(3):
                tangent_sums[index][axis] += tangent[axis]
                bitangent_sums[index][axis] += bitangent[axis]

    tangents = []
    bitangents = []
    for index, normal in enumerate(normals):
        tangent_sum = tangent_sums[index]
        projection = sum(normal[axis] * tangent_sum[axis] for axis in range(3))
        tangent = _normalize_vector(
            tuple(tangent_sum[axis] - normal[axis] * projection for axis in range(3)),
            _fallback_tangent(normal),
        )
        cross = _cross_vector(normal, tangent)
        handedness = -1.0 if sum(cross[axis] * bitangent_sums[index][axis] for axis in range(3)) < 0.0 else 1.0
        bitangent = tuple(component * handedness for component in cross)
        tangents.append(tangent)
        bitangents.append(bitangent)
    return tangents, bitangents


def _bone_name_map(bones: dict) -> dict[str, int]:
    result: dict[str, int] = {}
    for fallback_index, bone in bones.items():
        index = int(bone.get("idx", fallback_index))
        name = str(bone.get("name", ""))
        for candidate in cmp_bone_name_candidates(name):
            result.setdefault(candidate, index)
            result.setdefault(candidate.lower(), index)
    return result


def _resolve_fbx_weights(
    source_weights: dict[str, float],
    bone_name_to_index: dict[str, int],
    max_influences: int,
    root_bone: int,
    bind_unweighted_root: bool,
    stats: collections.Counter,
) -> list[tuple[int, float]]:
    combined: dict[int, float] = collections.defaultdict(float)
    unknown = []
    for name, weight in source_weights.items():
        weight = float(weight)
        if weight <= 1e-8:
            continue
        candidates = cmp_bone_name_candidates(str(name))
        bone_index = next(
            (bone_name_to_index[candidate] for candidate in candidates if candidate in bone_name_to_index),
            None,
        )
        if bone_index is None:
            bone_index = next(
                (bone_name_to_index[candidate.lower()] for candidate in candidates if candidate.lower() in bone_name_to_index),
                None,
            )
        if bone_index is None:
            unknown.append(str(name))
            continue
        combined[int(bone_index)] += weight
    if unknown:
        stats["unknown_weight_groups"] += len(set(unknown))
    ordered = sorted(combined.items(), key=lambda item: item[1], reverse=True)
    if len(ordered) > max_influences:
        stats["vertices_reduced_to_layout_weight_limit"] += 1
        ordered = ordered[:max_influences]
    total = sum(weight for _bone, weight in ordered)
    if total > 1e-8:
        stats["vertices_with_fbx_weights"] += 1
        return [(bone, weight / total) for bone, weight in ordered]
    if not bind_unweighted_root:
        raise ValueError(
            "Replacement FBX contains unweighted vertices; enable temporary root binding or weight-paint every vertex"
        )
    stats["vertices_bound_to_template_root"] += 1
    return [(int(root_bone), 1.0)]


def _decode_group(data: bytes, bone_count: int, sm: dict) -> dict:
    vertices = []
    normals = []
    uvs = []
    tangents = []
    bitangents = []
    stride = int(sm["v_stride"])
    layout = str(sm["layout"])
    for i in range(int(sm["v_count"])):
        record_offset = int(sm["v_start"]) + i * stride
        pos, uv, normal, _weights = parse_vertex_by_layout(
            data,
            record_offset,
            bone_count,
            layout,
        )
        vertices.append(tuple(map(float, pos)))
        normals.append(tuple(map(float, normal)))
        uvs.append(tuple(map(float, uv)))
        if layout == "skin64":
            tangents.append(struct.unpack_from(">3f", data, record_offset + 40))
            bitangents.append(struct.unpack_from(">3f", data, record_offset + 52))
        elif layout == "blend76":
            tangents.append(struct.unpack_from(">3f", data, record_offset + 52))
            bitangents.append(struct.unpack_from(">3f", data, record_offset + 64))
    group = {
        "source": sm,
        "vertices": vertices,
        "normals": normals,
        "uvs": uvs,
        "faces": [tuple(map(int, face)) for face in sm.get("faces") or []],
    }
    if len(tangents) == len(vertices):
        group["tangents"] = tangents
        group["bitangents"] = bitangents
    return group


def _template_primitive_profile(
    template_data: bytes, template_sm: dict
) -> tuple[int, int, dict[str, int]]:
    """Read the native GX command family without scanning inside index records."""
    pos = int(template_sm["dl_start"])
    end = int(template_sm["dl_end"])
    index_width = int(template_sm.get("index_width", 6))
    opcodes = []
    while pos + 3 <= end:
        opcode = int(template_data[pos])
        if opcode == 0:
            pos += 1
            continue
        if opcode not in (0x80, 0x90, 0x98, 0xA0):
            break
        record_count = int.from_bytes(template_data[pos + 1:pos + 3], "big")
        command_size = 3 + record_count * index_width
        if record_count < 3 or pos + command_size > end:
            break
        opcodes.append(opcode)
        pos += command_size
    if not opcodes:
        raise ValueError("Template stream has no readable GX primitive commands")
    counts = collections.Counter(opcodes)
    first_positions = {opcode: opcodes.index(opcode) for opcode in counts}
    dominant = max(counts, key=lambda opcode: (counts[opcode], -first_positions[opcode]))
    return (
        dominant,
        opcodes[0],
        {hex(opcode): int(count) for opcode, count in sorted(counts.items())},
    )


def _build_template_triangle_commands(
    faces: list[tuple[int, int, int]],
    index_width: int,
    primitive_opcode: int,
) -> tuple[bytes, int]:
    """Encode exact triangles using the primitive family expected by the template."""
    if primitive_opcode == 0x90:
        command_count = (len(faces) + 340) // 341
        return _build_triangle_commands(faces, index_width), command_count
    if primitive_opcode not in (0x98, 0xA0):
        raise ValueError(
            f"Template GX primitive {hex(primitive_opcode)} cannot safely encode isolated triangles"
        )
    out = bytearray()
    for face in faces:
        out.append(primitive_opcode)
        out.extend(struct.pack(">H", 3))
        for vertex_index in face:
            out.extend(_pack_index_record(vertex_index, index_width))
    return bytes(out), len(faces)


def _build_stream(
    template_data: bytes,
    template_sm: dict,
    donor_groups: list[dict],
    root_bone: int,
    bone_name_to_index: dict[str, int] | None = None,
    bind_unweighted_root: bool = True,
    position_scale: float = 1.0,
    weight_stats: collections.Counter | None = None,
) -> tuple[bytes, bytes, dict]:
    layout = str(template_sm["layout"])
    stride = int(template_sm["v_stride"])
    index_width = int(template_sm.get("index_width", 6))
    template_record = template_data[
        int(template_sm["v_start"]):int(template_sm["v_start"]) + stride
    ]
    if len(template_record) != stride:
        raise ValueError("Template vertex record is truncated")

    records = bytearray()
    faces: list[tuple[int, int, int]] = []
    vertex_count = 0
    weight_stats = weight_stats if weight_stats is not None else collections.Counter()
    max_influences = 2 if layout in ("skin64", "skin48", "skin40") else 4
    for group in donor_groups:
        base = vertex_count
        group_weights = list(group.get("weights") or [])
        group_tangents = list(group.get("tangents") or [])
        group_bitangents = list(group.get("bitangents") or [])
        if len(group_tangents) != len(group["vertices"]) or len(group_bitangents) != len(group["vertices"]):
            group_tangents, group_bitangents = _compute_tangent_basis(group)
        for local_index, (pos, normal, uv) in enumerate(zip(group["vertices"], group["normals"], group["uvs"])):
            record = bytearray(template_record)
            stored_uv = (float(uv[0]), 1.0 - float(uv[1]))
            native_position = tuple(float(value) / position_scale for value in pos)
            if group_weights and local_index < len(group_weights):
                if bone_name_to_index is None:
                    raise ValueError("FBX weight mapping requires the template skeleton")
                native_weights = _resolve_fbx_weights(
                    group_weights[local_index],
                    bone_name_to_index,
                    max_influences,
                    root_bone,
                    bind_unweighted_root,
                    weight_stats,
                )
            else:
                native_weights = [(int(root_bone), 1.0)]
            records.extend(
                _patch_native_vertex_record(
                    record,
                    layout,
                    native_position,
                    normal,
                    stored_uv,
                    native_weights,
                    group_tangents[local_index],
                    group_bitangents[local_index],
                )
            )
        for face in group["faces"]:
            faces.append((base + face[0], base + face[1], base + face[2]))
        vertex_count += len(group["vertices"])

    if not faces or vertex_count <= 0:
        raise ValueError("Every template stream must receive donor geometry")
    if index_width in (3, 4) and vertex_count > 256:
        raise ValueError(
            f"Template stream uses compact 8-bit indices but needs {vertex_count} donor vertices"
        )
    if vertex_count > 65535:
        raise ValueError(f"Template stream exceeds GX 16-bit vertex range: {vertex_count}")

    primitive_opcode, template_first_opcode, template_primitive_commands = _template_primitive_profile(
        template_data, template_sm
    )
    display_list, output_command_count = _build_template_triangle_commands(
        faces, index_width, primitive_opcode
    )
    return display_list, bytes(records), {
        "layout": layout,
        "index_width": index_width,
        "vertex_count": vertex_count,
        "triangle_count": len(faces),
        "record_count": len(faces) * 3,
        "template_first_primitive_opcode": hex(template_first_opcode),
        "template_primitive_opcode": hex(primitive_opcode),
        "template_primitive_commands": template_primitive_commands,
        "output_primitive_opcode": hex(primitive_opcode),
        "output_command_count": output_command_count,
        "donor_groups": [int(g["source"].get("custom_group", -1)) for g in donor_groups],
    }


def inject_bdg_custom_model(
    donor_path: Path,
    template_path: Path,
    out_path: Path,
    scale: float = 10.0,
    bind_unweighted_root: bool = False,
) -> dict:
    donor_path = donor_path.resolve()
    template_path = template_path.resolve()
    out_path = out_path.resolve()
    template_data, template_bones, template_sms, template_skipped = _decoded_mesh(template_path)
    parser, entries, mesh_entry = _mesh_resource_entry(template_path)
    template_materials = _type6_material_flags(template_data, entries)

    is_fbx = donor_path.suffix.lower() == ".fbx"
    source_parse_report = {}
    donor_materials = []
    donor_skipped = []
    if is_fbx:
        if abs(float(scale)) <= 1e-8:
            raise ValueError("FBX scale cannot be zero")
        roots, fbx_version = parse_fbx(donor_path)
        mesh_groups, source_parse_report = extract_custom_triangles(roots)
        source_parse_report["axis_conversion"] = _convert_fbx_axes_to_bdg(roots, mesh_groups)
        triangle_buckets, material_partition_report = _partition_fbx_triangles(
            mesh_groups, template_sms, template_materials
        )
        decoded_groups = [
            _group_from_fbx_triangles(triangles, group_index)
            for group_index, triangles in enumerate(triangle_buckets)
        ]
        assignments = [[group] for group in decoded_groups]
        source_parse_report["material_partition"] = material_partition_report
        source_parse_report["fbx_version"] = fbx_version
        source_mode = "FBX geometry into template Type-17 streams"
        position_scale = float(scale)
        donor_needs_two_sided = False
    else:
        donor_data, donor_bones, donor_sms, donor_skipped = _decoded_mesh(donor_path)
        if len(donor_sms) < len(template_sms):
            raise ValueError(
                f"Donor has {len(donor_sms)} streams but template requires {len(template_sms)} non-empty streams"
            )
        decoded_groups = []
        for group_i, sm in enumerate(donor_sms):
            sm = dict(sm)
            sm["custom_group"] = group_i
            decoded_groups.append(_decode_group(donor_data, len(donor_bones), sm))
        assignments = _partition_contiguous(decoded_groups, len(template_sms))
        donor_parser = PipeworksParser(str(donor_path))
        donor_entries = donor_parser.parse()
        donor_materials = _type6_material_flags(donor_data, donor_entries)
        donor_needs_two_sided = any(material["two_sided"] for material in donor_materials)
        source_mode = "BDG donor geometry into template Type-17 streams"
        position_scale = 1.0

    main_entries = [e for e in entries if e.get("file_type") == 17 and not e.get("is_resource")]
    if len(main_entries) != 1:
        raise ValueError("Could not uniquely identify the template Type-17 descriptor resource")
    main_entry = main_entries[0]
    endian = ">" if parser.is_big_endian else "<"
    descs = _find_mesh_descriptors(
        template_data, main_entry, mesh_entry, template_sms, endian
    )
    summary_records = _find_mesh_summary_records(
        template_data, main_entry, descs, endian
    )

    root_bone = 0
    bone_name_to_index = _bone_name_map(template_bones)
    weight_stats = collections.Counter()
    streams = []
    stream_bounds = []
    for sm, groups in zip(template_sms, assignments):
        streams.append(
            _build_stream(
                template_data,
                sm,
                groups,
                root_bone,
                bone_name_to_index=bone_name_to_index if is_fbx else None,
                bind_unweighted_root=bind_unweighted_root if is_fbx else True,
                position_scale=position_scale,
                weight_stats=weight_stats,
            )
        )
        stream_bounds.append(_stream_bounds(groups, position_scale))

    resource = bytearray()
    new_descs = []
    stream_reports = []
    for stream_i, ((dl_raw, vertex_raw, stream_report), old_desc) in enumerate(zip(streams, descs)):
        rel_dl = len(resource)
        resource.extend(dl_raw)
        rel_v = _align(len(resource), 0x20)
        resource.extend(b"\x00" * (rel_v - len(resource)))
        resource.extend(vertex_raw)
        if stream_i + 1 < len(streams):
            resource.extend(b"\x00" * (_align(len(resource), 0x20) - len(resource)))

        nd = dict(old_desc)
        nd.update({
            "rel_dl": rel_dl,
            "dl_size": rel_v - rel_dl,
            "record_count": int(stream_report["record_count"]),
            "v_count": int(stream_report["vertex_count"]),
            "rel_v": rel_v,
            "v_size": len(vertex_raw),
        })
        new_descs.append(nd)
        stream_report.update({
            "stream": stream_i,
            "display_list_offset": hex(rel_dl),
            "display_list_size": rel_v - rel_dl,
            "vertex_offset": hex(rel_v),
            "vertex_bytes": len(vertex_raw),
        })
        stream_reports.append(stream_report)

    old_mesh_start = int(mesh_entry["offset"])
    old_mesh_size = int(mesh_entry["size"])
    old_mesh_end = old_mesh_start + old_mesh_size
    resource_payload_size = len(resource)
    # GZBuildr's slot-preserving rebuild keeps the physical resource slot when a
    # replacement becomes smaller, but writes the replacement's logical size to
    # the TOC. Keeping the old capacity as the live Type-17 size makes the game
    # continue parsing zeroed former mesh bytes as part of the replacement.
    resource_capacity_padding = max(0, old_mesh_size - len(resource))
    if resource_capacity_padding:
        resource.extend(b"\x00" * resource_capacity_padding)
        resource_alignment_padding = 0
    else:
        resource_alignment_padding = (old_mesh_size - len(resource)) % 0x20
        resource.extend(b"\x00" * resource_alignment_padding)
    resource_slot_size = len(resource)
    new_mesh_size = resource_payload_size
    delta = resource_slot_size - old_mesh_size
    out = bytearray(template_data)
    out[old_mesh_start:old_mesh_end] = resource

    mesh_toc = int(mesh_entry["toc_entry_offset"])
    _p32(out, mesh_toc + 14, new_mesh_size, endian)
    # Walk every raw TOC row. The slim parser omits resource entries whose size
    # is zero, but their resource-offset fields can still be section cursors and
    # must move with a grown resource block.
    for entry_index in range(int(parser.file_count)):
        toc = 0x78 + entry_index * 0x12
        if toc == mesh_toc:
            continue
        old_relative = _u32(template_data, toc + 10, endian)
        old_absolute = int(parser.resource_data_offset) + old_relative
        if old_relative and old_absolute > old_mesh_start:
            _p32(out, toc + 10, old_relative + delta, endian)

    resource_section_size = len(out) - int(parser.resource_data_offset)
    _p32(out, 0x74, resource_section_size, endian)

    for desc, nd in zip(descs, new_descs):
        base = int(desc["base"])
        _p32(out, base + 0, int(nd["record_count"]), endian)
        _p32(out, base + 8, int(nd["rel_dl"]), endian)
        _p32(out, base + 12, int(nd["dl_size"]), endian)
        _p32(out, base + 32, int(nd["v_count"]), endian)
        _p32(out, base + 48, int(nd["rel_v"]), endian)
        _p32(out, base + 52, int(nd["v_size"]), endian)

    for summary, nd, bounds in zip(summary_records, new_descs, stream_bounds):
        _p32(out, summary, max(0, int(nd["record_count"]) - 2), endian)
        _p32(out, summary + 4, int(nd["v_count"]), endian)
        for axis, value in enumerate(bounds["min"]):
            _pf32(out, summary + 0x14 + axis * 4, value, endian)
        for axis, value in enumerate(bounds["max"]):
            _pf32(out, summary + 0x24 + axis * 4, value, endian)
        for axis, value in enumerate(bounds["center"]):
            _pf32(out, summary + 0x34 + axis * 4, value, endian)
        _pf32(out, summary + 0x40, bounds["radius_squared"], endian)

    patched_materials = []
    if donor_needs_two_sided:
        # Thin cloth/wing/web geometry is physically single-sided in BDG. Its
        # Type-6 material sets bit 0 at +7 (0x3D versus the usual 0x3C) to
        # disable back-face culling. Custom injection can merge several donor
        # ranges into one template range, so enable that state on every retained
        # template material instead of fabricating reverse triangles.
        for material in _type6_material_flags(template_data, entries):
            old_flags = int(out[material["offset"] + 7])
            new_flags = old_flags | 0x01
            out[material["offset"] + 7] = new_flags
            patched_materials.append({
                "name": material["name"],
                "offset": hex(material["offset"] + 7),
                "flags_before": hex(old_flags),
                "flags_after": hex(new_flags),
            })

    candidate = bytes(out)
    if _u32(candidate, 0x74, endian) != len(candidate) - int(parser.resource_data_offset):
        raise ValueError("BDG header 0x74 does not match the rebuilt resource-section size")
    fake_mesh = dict(mesh_entry)
    fake_mesh["size"] = new_mesh_size
    strict_ok, strict_message = _strict_validate_descriptor_streams(
        candidate, main_entry, fake_mesh, template_sms, new_descs, endian
    )
    if not strict_ok:
        raise ValueError(f"Rebuilt Type-17 streams failed validation: {strict_message}")
    for stream_index, (summary, desc) in enumerate(zip(summary_records, new_descs)):
        if (
            _u32(candidate, summary, endian) != max(0, int(desc["record_count"]) - 2)
            or _u32(candidate, summary + 4, endian) != int(desc["v_count"])
        ):
            raise ValueError(f"Rebuilt Type-17 summary {stream_index} does not match its descriptor")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".BDG") as temp_file:
        temp_path = Path(temp_file.name)
        temp_file.write(candidate)
    try:
        check_data, check_bones, check_sms, _check_skipped = _decoded_mesh(temp_path)
        actual_faces = sum(len(sm.get("faces") or []) for sm in check_sms)
        expected_faces = sum(len(group["faces"]) for group in decoded_groups)
        actual_vertices = sum(int(sm["v_count"]) for sm in check_sms)
        expected_vertices = sum(len(group["vertices"]) for group in decoded_groups)
        if len(check_sms) != len(template_sms):
            raise ValueError(
                f"Reload decoded {len(check_sms)} streams; expected {len(template_sms)}"
            )
        if actual_faces != expected_faces or actual_vertices != expected_vertices:
            raise ValueError(
                "Reload count mismatch: "
                f"faces {actual_faces}/{expected_faces}, vertices {actual_vertices}/{expected_vertices}"
            )
        for stream_index, (check_sm, stream_report) in enumerate(zip(check_sms, stream_reports)):
            output_opcode, first_opcode, command_counts = _template_primitive_profile(
                check_data, check_sm
            )
            expected_opcode = int(str(stream_report["output_primitive_opcode"]), 16)
            expected_commands = int(stream_report["output_command_count"])
            expected_counts = {hex(expected_opcode): expected_commands}
            if (
                output_opcode != expected_opcode
                or first_opcode != expected_opcode
                or command_counts != expected_counts
            ):
                raise ValueError(
                    f"Reload GX primitive mismatch in stream {stream_index}: "
                    f"{command_counts}, expected {expected_counts}"
                )
        if len(check_bones) != len(template_bones):
            raise ValueError("Template skeleton changed during custom injection")
    finally:
        temp_path.unlink(missing_ok=True)

    out_path.write_bytes(candidate)
    zip_path = out_path.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(out_path, arcname=out_path.name)

    report = {
        "status": "ok",
        "mode": source_mode,
        "replacement": str(donor_path),
        "template": str(template_path),
        "output": str(out_path),
        "zip": str(zip_path),
        "source_kind": "fbx" if is_fbx else "bdg",
        "source_submeshes": int(source_parse_report.get("mesh_objects", 0)) if is_fbx else len(donor_sms),
        "template_submeshes": len(template_sms),
        "source_vertices": sum(len(group["vertices"]) for group in decoded_groups),
        "source_triangles": sum(len(group["faces"]) for group in decoded_groups),
        "output_vertices": actual_vertices,
        "output_triangles": actual_faces,
        "root_bone": root_bone,
        "bind_unweighted_root": bool(bind_unweighted_root) if is_fbx else True,
        "fbx_scale": position_scale if is_fbx else None,
        "fbx_parse": source_parse_report if is_fbx else None,
        "weight_write": dict(weight_stats),
        "resource_size_before": old_mesh_size,
        "resource_payload_size": resource_payload_size,
        "resource_size_after": new_mesh_size,
        "resource_slot_size_after": resource_slot_size,
        "resource_delta": delta,
        "resource_capacity_padding": resource_capacity_padding,
        "resource_alignment_padding": resource_alignment_padding,
        "resource_section_size_0x74": resource_section_size,
        "draw_summaries": [
            {
                "stream": stream_index,
                "offset": hex(summary),
                "record_count_minus_two": max(0, int(desc["record_count"]) - 2),
                "vertex_count": int(desc["v_count"]),
                "bounds": {
                    key: list(value) if isinstance(value, tuple) else value
                    for key, value in bounds.items()
                },
            }
            for stream_index, (summary, desc, bounds) in enumerate(
                zip(summary_records, new_descs, stream_bounds)
            )
        ],
        "strict_validation": strict_message,
        "donor_two_sided_materials": [
            material["name"] for material in donor_materials if material["two_sided"]
        ],
        "template_two_sided_patches": patched_materials,
        "streams": stream_reports,
        "sha256": _sha256(candidate),
        "notes": [
            "Template skeleton, materials, textures, and non-mesh resources are preserved.",
            (
                "FBX vertex groups matching template bone names are written using each native stream's influence limit."
                if is_fbx
                else "BDG donor geometry is rigidly bound to template bone 0."
            ),
            (
                "Unweighted FBX vertices are rigidly bound to template bone 0."
                if is_fbx and bind_unweighted_root
                else "Unweighted FBX vertices are rejected."
                if is_fbx
                else "Donor UV coordinates and normals are copied without FBX conversion."
            ),
            "FBX UV coordinates and normals are preserved; FBX positions are converted back from the bridge's export scale."
            if is_fbx
            else "Donor UV coordinates and normals are copied without FBX conversion.",
            "Each output stream uses the template stream's dominant native GX primitive family.",
            "Thin two-sided donor materials enable the template Type-6 two-sided flag; no reverse faces are generated.",
        ],
    }
    report_path = out_path.with_name(out_path.stem + "_custom_report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inject replacement FBX or donor Shapes.BDG geometry into a template Shapes.BDG"
    )
    parser.add_argument("donor", help="Replacement FBX or donor *_Shapes.BDG containing replacement geometry")
    parser.add_argument("template", help="Template *_Shapes.BDG providing skeleton/material resources")
    parser.add_argument("--out", required=True, help="Output *_Shapes.BDG")
    parser.add_argument("--scale", type=float, default=10.0, help="FBX export scale")
    parser.add_argument(
        "--bind-unweighted-root",
        action="store_true",
        help="Rigidly bind unweighted FBX vertices to template bone 0",
    )
    args = parser.parse_args()
    report = inject_bdg_custom_model(
        Path(args.donor),
        Path(args.template),
        Path(args.out),
        scale=args.scale,
        bind_unweighted_root=args.bind_unweighted_root,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
