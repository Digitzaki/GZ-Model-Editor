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

from cmp_probe import (
    build_adc_strip_faces,
    bundle_strings,
    cmp_side_record_draws,
    cmp_packet_uses_vertex_draw_flags,
    cmp_stream_logical_indices,
    cmp_vertex_control_draws,
    decode_cmp_skin_control,
    find_cmp_packets,
    parse_cmp_material_ranges,
    parse_cmp_materials,
    parse_cmp_pose_records,
    parse_cmp_skeleton,
    parse_cmp_skin_palette,
    parse_cmp_type3_pose_records,
    read_cmp_uv,
    read_cmp_vertex_control,
)
from cmp_probe import cmp_packet_side_stream, cmp_packet_uv_stream
from cmg_probe import global_matrices
from fbx_to_bdg_import import clean_fbx_object_name, find_first, object_nodes, parse_fbx, p_values
from parser_core import PipeworksParser


def geometry_name(node) -> str:
    value = str(node.props[1] if len(node.props) > 1 else "")
    if "\x00\x01" in value:
        return value.split("\x00\x01", 1)[0]
    return value.split("::", 1)[-1]


def geometry_packet_sort_key(node) -> tuple[int, str]:
    name = geometry_name(node)
    lower = name.lower()
    marker = lower.rfind("_packet")
    if marker >= 0:
        pos = marker + len("_packet")
        end = pos
        while end < len(lower) and lower[end].isdigit():
            end += 1
        if end > pos:
            return (int(lower[pos:end]), lower)
    return (10**9, lower)


def geometry_vertices(node) -> list[tuple[float, float, float]]:
    flat = node.child("Vertices").props[0]
    return [tuple(map(float, flat[i : i + 3])) for i in range(0, len(flat), 3)]


def geometry_faces(node) -> list[tuple[int, int, int]]:
    out: list[tuple[int, int, int]] = []
    pvi_node = node.child("PolygonVertexIndex")
    if not pvi_node:
        return out
    face: list[int] = []
    for value in pvi_node.props[0]:
        if value < 0:
            face.append(~value)
            if len(face) == 3:
                out.append(tuple(face))
            face = []
        else:
            face.append(value)
    return out


def face_key(face: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(sorted(face))


def geometry_uvs_by_cp(node, vertex_count: int) -> list[tuple[float, float] | None]:
    out: list[list[tuple[float, float]]] = [[] for _ in range(vertex_count)]
    pvi_node = node.child("PolygonVertexIndex")
    uv_layer = node.child("LayerElementUV")
    if not pvi_node or not uv_layer or not uv_layer.child("UV"):
        return [None] * vertex_count
    cp_seq = [i if i >= 0 else ~i for i in pvi_node.props[0]]
    values = uv_layer.child("UV").props[0]
    pairs = [tuple(map(float, values[i : i + 2])) for i in range(0, len(values), 2)]
    uv_index = uv_layer.child("UVIndex")
    if uv_index:
        indices = uv_index.props[0]
        pairs = [pairs[i] for i in indices if 0 <= i < len(pairs)]
    for pv_i, cp_i in enumerate(cp_seq):
        if 0 <= cp_i < vertex_count and pv_i < len(pairs):
            out[cp_i].append(pairs[pv_i])
    averaged = []
    for samples in out:
        if not samples:
            averaged.append(None)
            continue
        counts = collections.Counter((round(u, 6), round(v, 6)) for u, v in samples)
        if len(counts) > 1:
            averaged.append(None)
            continue
        (u, v), _count = counts.most_common(1)[0]
        averaged.append((float(u), float(v)))
    return averaged


def normalize_vector3(values: tuple[float, float, float]) -> tuple[float, float, float] | None:
    length = sum(value * value for value in values) ** 0.5
    if length <= 1e-12:
        return None
    return tuple(value / length for value in values)


def geometry_normals_by_cp(node, vertex_count: int) -> list[tuple[float, float, float] | None]:
    out: list[list[tuple[float, float, float]]] = [[] for _ in range(vertex_count)]
    pvi_node = node.child("PolygonVertexIndex")
    normal_layer = node.child("LayerElementNormal")
    if not pvi_node or not normal_layer or not normal_layer.child("Normals"):
        return [None] * vertex_count

    values = normal_layer.child("Normals").props[0]
    direct = [
        normalize_vector3(tuple(map(float, values[i : i + 3])))
        for i in range(0, len(values), 3)
    ]
    mapping_node = normal_layer.child("MappingInformationType")
    mapping = str(mapping_node.props[0]) if mapping_node else "ByPolygonVertex"
    index_node = normal_layer.child("NormalsIndex") or normal_layer.child("NormalIndex")
    indices = [int(value) for value in index_node.props[0]] if index_node else list(range(len(direct)))

    if mapping in ("ByVertice", "ByVertex", "ByControlPoint"):
        for cp_index in range(min(vertex_count, len(indices))):
            normal_index = indices[cp_index]
            if 0 <= normal_index < len(direct) and direct[normal_index] is not None:
                out[cp_index].append(direct[normal_index])
    else:
        cp_sequence = [value if value >= 0 else ~value for value in pvi_node.props[0]]
        for loop_index, cp_index in enumerate(cp_sequence):
            normal_index = indices[loop_index] if loop_index < len(indices) else loop_index
            if 0 <= cp_index < vertex_count and 0 <= normal_index < len(direct) and direct[normal_index] is not None:
                out[cp_index].append(direct[normal_index])

    result: list[tuple[float, float, float] | None] = []
    for samples in out:
        if not samples:
            result.append(None)
            continue
        result.append(
            normalize_vector3(
                tuple(sum(sample[axis] for sample in samples) for axis in range(3))
            )
        )
    return result


def strip_triangle(record_index: int) -> tuple[int, int, int]:
    first = record_index - 2
    return (first, first + 1, record_index) if first % 2 == 0 else (first + 1, first, record_index)


def oriented_face_key(face: tuple[int, int, int]) -> tuple[int, int, int]:
    a, b, c = face
    return min((a, b, c), (b, c, a), (c, a, b))


def added_vertex_write_order(
    old_count: int,
    new_count: int,
    faces: list[tuple[int, int, int]],
) -> tuple[list[int], list[dict]]:
    order = list(range(new_count))
    adjacency: dict[int, set[int]] = collections.defaultdict(set)
    new_faces = [face for face in faces if all(old_count <= index < new_count for index in face)]
    for face in new_faces:
        for index in face:
            adjacency[index].update(other for other in face if other != index)

    components = []
    unseen = set(adjacency)
    while unseen:
        pending = [unseen.pop()]
        component = set(pending)
        while pending:
            current = pending.pop()
            for neighbor in adjacency[current]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    component.add(neighbor)
                    pending.append(neighbor)
        components.append(sorted(component))

    reversed_components = []
    for component in components:
        start, end = component[0], component[-1]
        if component != list(range(start, end + 1)):
            continue
        desired_faces = [face for face in new_faces if all(index in component for index in face)]
        desired_keys = {face_key(face) for face in desired_faces}
        desired_oriented = {oriented_face_key(face) for face in desired_faces}

        def score(candidate: list[int]) -> tuple[int, int]:
            candidate_order = dict(zip(component, candidate))
            topology = orientation = 0
            for record_index in range(start + 2, end + 1):
                record_face = strip_triangle(record_index)
                mapped_face = tuple(candidate_order[index] for index in record_face)
                if face_key(mapped_face) in desired_keys:
                    topology += 1
                    if oriented_face_key(mapped_face) in desired_oriented:
                        orientation += 1
            return topology, orientation

        forward = component
        backward = list(reversed(component))
        forward_score = score(forward)
        backward_score = score(backward)
        if backward_score[0] == len(desired_keys) and backward_score[1] > forward_score[1]:
            for record_index, source_index in zip(component, backward):
                order[record_index] = source_index
            reversed_components.append(
                {
                    "start": start,
                    "end": end,
                    "faces": len(desired_keys),
                    "forward_oriented": forward_score[1],
                    "reversed_oriented": backward_score[1],
                }
            )
    return order, reversed_components


def geometry_weights_by_cp(roots, geometry, vertex_count: int) -> tuple[list[dict[str, float]], dict]:
    weights: list[dict[str, float]] = [collections.defaultdict(float) for _ in range(vertex_count)]
    objects = object_nodes(roots)
    object_by_id = {
        int(node.props[0]): node
        for node in objects
        if node.props and isinstance(node.props[0], int)
    }
    connections = find_first(roots, "Connections")
    relations: list[tuple[int, int]] = []
    if connections:
        for connection in connections.children_named("C"):
            if len(connection.props) >= 3 and str(connection.props[0]) == "OO":
                relations.append((int(connection.props[1]), int(connection.props[2])))

    geometry_id = int(geometry.props[0])
    skin_ids = {
        child
        for child, parent in relations
        if parent == geometry_id
        and child in object_by_id
        and object_by_id[child].name == "Deformer"
        and len(object_by_id[child].props) >= 3
        and str(object_by_id[child].props[2]) == "Skin"
    }
    cluster_ids = {
        child
        for child, parent in relations
        if parent in skin_ids
        and child in object_by_id
        and object_by_id[child].name == "Deformer"
        and len(object_by_id[child].props) >= 3
        and str(object_by_id[child].props[2]) == "Cluster"
    }
    invalid_indices = 0
    clusters_without_bones = 0
    for cluster_id in cluster_ids:
        cluster = object_by_id[cluster_id]
        bone_models = [
            object_by_id[child]
            for child, parent in relations
            if parent == cluster_id
            and child in object_by_id
            and object_by_id[child].name == "Model"
        ]
        if bone_models:
            bone_name = clean_fbx_object_name(bone_models[0].props[1])
        else:
            cluster_name = clean_fbx_object_name(cluster.props[1] if len(cluster.props) > 1 else "")
            suffix = "_Cluster"
            bone_name = cluster_name[: -len(suffix)] if cluster_name.endswith(suffix) else cluster_name
            geometry_prefix = geometry_name(geometry) + "_"
            if bone_name.startswith(geometry_prefix):
                bone_name = bone_name[len(geometry_prefix) :]
            clusters_without_bones += 1
        indices = cluster.child("Indexes")
        values = cluster.child("Weights")
        if not indices or not values:
            continue
        for cp_index, weight in zip(indices.props[0], values.props[0]):
            cp_index = int(cp_index)
            if 0 <= cp_index < vertex_count:
                if float(weight) > 1e-8:
                    weights[cp_index][bone_name] += float(weight)
            else:
                invalid_indices += 1
    return [dict(item) for item in weights], {
        "skins": len(skin_ids),
        "clusters": len(cluster_ids),
        "clusters_without_bones": clusters_without_bones,
        "invalid_indices": invalid_indices,
    }


def cmp_bone_name_candidates(name: str) -> tuple[str, ...]:
    clean = clean_fbx_object_name(str(name)).strip()
    candidates = [clean]
    blender_suffix = re.fullmatch(r"(.*?)(?:_?Model)(?:\.\d+)?", clean, re.IGNORECASE)
    if blender_suffix and blender_suffix.group(1).strip():
        candidates.append(blender_suffix.group(1).strip())
    expanded: list[str] = []
    for candidate in candidates:
        expanded.extend((candidate, candidate.replace(" ", "_"), candidate.replace("_", " ")))
    return tuple(dict.fromkeys(item for item in expanded if item))


def cmp_bone_name_map(bones: list[dict]) -> dict[str, int]:
    result: dict[str, int] = {}
    for bone in bones:
        name = str(bone["name"])
        index = int(bone["idx"])
        for candidate in cmp_bone_name_candidates(name):
            result.setdefault(candidate, index)
            result.setdefault(candidate.lower(), index)
    return result


def cmp_weight_bone_index(name: str, bone_name_to_index: dict[str, int]) -> int | None:
    candidates = cmp_bone_name_candidates(str(name))
    bone_index = next(
        (
            bone_name_to_index[candidate]
            for candidate in candidates
            if candidate in bone_name_to_index
        ),
        None,
    )
    if bone_index is not None:
        return bone_index
    return next(
        (
            bone_name_to_index[candidate.lower()]
            for candidate in candidates
            if candidate.lower() in bone_name_to_index
        ),
        None,
    )


def fbx_bone_models(roots) -> dict[str, object]:
    result: dict[str, object] = {}
    for model in object_nodes(roots, "Model"):
        if len(model.props) < 3 or str(model.props[2]) not in ("LimbNode", "Null"):
            continue
        name = clean_fbx_object_name(str(model.props[1]))
        for candidate in cmp_bone_name_candidates(name):
            result.setdefault(candidate, model)
            result.setdefault(candidate.lower(), model)
    return result


def find_cmp_bone_model(models: dict[str, object], name: str):
    for candidate in cmp_bone_name_candidates(name):
        model = models.get(candidate) or models.get(candidate.lower())
        if model is not None:
            return model
    return None


def cmp_inverse_bind_values(global_matrix: list[list[float]]) -> tuple[float, ...]:
    translation = [float(global_matrix[axis][3]) for axis in range(3)]
    inverse_translation = [
        -sum(float(global_matrix[row][column]) * translation[row] for row in range(3))
        for column in range(3)
    ]
    # CMP stores the inverse rotation column-major, which is the global
    # rotation's rows followed by the inverse translation.
    return tuple(
        [float(global_matrix[row][column]) for row in range(3) for column in range(3)]
        + inverse_translation
    )


def parse_cmp_animation_translation_tracks(blob: bytes) -> list[dict] | None:
    if len(blob) < 0x44 or struct.unpack_from("<I", blob, 0x20)[0] != 2:
        return None
    bone_count = struct.unpack_from("<I", blob, 0x2C)[0]
    section_start = struct.unpack_from("<I", blob, 0x38)[0]
    section_end = struct.unpack_from("<I", blob, 0x3C)[0]
    if (
        bone_count <= 0
        or bone_count > 512
        or section_start < 0x44
        or section_end <= section_start
        or section_end > len(blob)
    ):
        return None

    def walk_tracks(
        pos: int,
        previous_bone: int,
        allow_interstitial: bool,
    ) -> list[dict] | None:
        if previous_bone == bone_count - 1:
            return []

        candidates = []
        # Most tracks use an XYZ scale vector. A small number of authored STEM
        # clips use one shared XYZ scale for a high-key-count track.
        for scale_count in (3, 1):
            header_pos = pos + scale_count * 4
            if header_pos + 4 > section_end:
                continue
            bone, key_count, last_key, flags = struct.unpack_from("<BBBB", blob, header_pos)
            records_end = header_pos + 4 + key_count * 8
            if (
                previous_bone < bone < bone_count
                and key_count > 0
                and records_end <= section_end
            ):
                candidates.append(
                    (scale_count, int(bone), int(key_count), int(last_key), int(flags), records_end)
                )

        for scale_count, bone, key_count, last_key, flags, records_end in candidates:
            remainder = walk_tracks(records_end, bone, allow_interstitial)
            if remainder is None:
                continue
            raw_scales = struct.unpack_from(f"<{scale_count}f", blob, pos)
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
            # Gigan FW's left jab has a root-motion block between the bone 1
            # and bone 2 translation tracks. It has no bone-track header, but
            # the normal ordered track stream resumes afterward on a 2-byte
            # boundary. Only accept a resync that validates through the final
            # bone, avoiding a match on arbitrary payload bytes.
            for resume_pos in range(pos + 2, section_end - 7, 2):
                for scale_count in (3, 1):
                    header_pos = resume_pos + scale_count * 4
                    if header_pos + 4 > section_end:
                        continue
                    bone, key_count, _last_key, _flags = struct.unpack_from(
                        "<BBBB", blob, header_pos
                    )
                    records_end = header_pos + 4 + key_count * 8
                    if (
                        bone == previous_bone + 1
                        and key_count > 0
                        and records_end <= section_end
                    ):
                        resumed = walk_tracks(resume_pos, previous_bone, False)
                        if resumed is not None:
                            return resumed
        return None

    # Tracks may omit bones, contain one validated interstitial motion block,
    # or leave a short terminal metadata block. Reaching the final bone proves
    # the track walk.
    tracks = walk_tracks(section_start, -1, True)
    return tracks if tracks else None


def cmp_animation_track_times_valid(times: list[int]) -> bool:
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


def decode_cmp_animation_rotation_track(
    blob: bytes,
    rel: int,
    section_end: int,
    bone_count: int,
    layout: str,
) -> dict | None:
    header_size = 2
    if rel < 0 or rel + header_size > section_end:
        return None
    bone, encoded_record_count = struct.unpack_from("<BB", blob, rel)
    record_count = int(encoded_record_count)
    if record_count == 0:
        # Some PS2 clips encode a one-key channel as count zero. Unmodified
        # all-zero records mean hold the bind pose; animation-lock patching can
        # replace that record with an explicit bind quaternion.
        if rel + header_size + 8 > section_end:
            return None
        record_count = 1
    if (
        bone >= bone_count
        or record_count <= 0
        or record_count > 255
        or rel + header_size + record_count * 8 > section_end
    ):
        return None

    records_pos = rel + header_size
    times = []
    for index in range(record_count):
        record_pos = records_pos + index * 8
        if layout == "explicit_qxyz_time":
            qx, qy, qz, time = struct.unpack_from("<hhhH", blob, record_pos)
        else:
            time, qx, qy, qz = struct.unpack_from("<Hhhh", blob, record_pos)
        norm_squared = sum((value / 32767.0) ** 2 for value in (qx, qy, qz))
        if norm_squared > 1.05:
            return None
        times.append(int(time))
    if not cmp_animation_track_times_valid(times):
        return None
    return {
        "bone": int(bone),
        "rel": rel,
        "end": records_pos + record_count * 8,
        "records_pos": records_pos,
        "record_count": int(record_count),
        "encoded_record_count": int(encoded_record_count),
        "use_bind_pose": bool(
            encoded_record_count == 0
            and not any(blob[records_pos : records_pos + 8])
        ),
        "layout": layout,
    }


def decode_cmp_animation_rotation_continuation(
    blob: bytes,
    start: int,
    end: int,
    bone: int,
    bone_count: int,
) -> dict | None:
    if (
        start < 0
        or start >= end
        or end > len(blob)
        or (end - start) % 8
        or bone < 0
        or bone >= bone_count
    ):
        return None
    if not any(blob[start:end]):
        return None
    record_count = (end - start) // 8
    if record_count <= 0 or record_count > 255:
        return None
    times = []
    for record_pos in range(start, end, 8):
        time, qx, qy, qz = struct.unpack_from("<Hhhh", blob, record_pos)
        norm_squared = sum((value / 32767.0) ** 2 for value in (qx, qy, qz))
        if norm_squared > 1.05:
            return None
        times.append(int(time))
    if not cmp_animation_track_times_valid(times):
        return None
    return {
        "bone": int(bone),
        "rel": start,
        "end": end,
        "records_pos": start,
        "record_count": record_count,
        "layout": "continuation_time_qxyz",
    }


def best_cmp_rotation_continuation(
    blob: bytes,
    gap_start: int,
    gap_end: int,
    bone: int,
    bone_count: int,
) -> dict | None:
    best = None
    for lead in (0, 2, 4, 6):
        for remainder in (0, 2, 4, 6):
            track = decode_cmp_animation_rotation_continuation(
                blob,
                gap_start + lead,
                gap_end - remainder,
                bone,
                bone_count,
            )
            if track is not None and (
                best is None or int(track["record_count"]) > int(best["record_count"])
            ):
                best = track
    return best


def cmp_terminal_bone_table_start(
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


def parse_cmp_animation_rotation_tracks(blob: bytes) -> list[dict]:
    if len(blob) < 0x44 or struct.unpack_from("<I", blob, 0x20)[0] != 2:
        return []
    bone_count = struct.unpack_from("<I", blob, 0x2C)[0]
    section_start = struct.unpack_from("<I", blob, 0x3C)[0]
    declared_size = struct.unpack_from("<I", blob, 0x24)[0]
    if bone_count <= 0 or bone_count > 512 or section_start < 0x40 or section_start >= len(blob):
        return []

    boundaries = sorted(
        {
            value
            for value in (
                struct.unpack_from("<I", blob, 0x38)[0],
                struct.unpack_from("<I", blob, 0x40)[0],
                declared_size,
                len(blob),
            )
            if section_start < value <= len(blob)
        }
    )
    best_tracks: list[dict] = []
    for section_end in boundaries:
        standard_candidates = []
        time_first_candidates = []
        for rel in range(section_start, section_end - 1, 2):
            standard = decode_cmp_animation_rotation_track(
                blob, rel, section_end, bone_count, "explicit_qxyz_time"
            )
            if standard is not None:
                standard_candidates.append(standard)
            time_first = decode_cmp_animation_rotation_track(
                blob, rel, section_end, bone_count, "explicit_time_qxyz"
            )
            if time_first is not None:
                time_first_candidates.append(time_first)

        standard_tracks = []
        last_end = section_start
        for track in standard_candidates:
            if track["rel"] < last_end:
                continue
            if standard_tracks and track["bone"] < standard_tracks[-1]["bone"]:
                continue
            standard_tracks.append(track)
            last_end = track["end"]

        by_start: dict[int, list[dict]] = collections.defaultdict(list)
        for track in time_first_candidates:
            by_start[int(track["rel"])].append(track)

        chain_cache: dict[tuple[int, int], list[dict]] = {}

        def time_first_chain(track: dict) -> list[dict]:
            key = (int(track["rel"]), int(track["bone"]))
            if key in chain_cache:
                return chain_cache[key]
            choices = [
                candidate
                for candidate in by_start.get(int(track["end"]), [])
                if int(candidate["bone"]) > int(track["bone"])
            ]
            tail = max(
                (time_first_chain(candidate) for candidate in choices),
                key=lambda chain: (len(chain), sum(int(item["record_count"]) for item in chain)),
                default=[],
            )
            chain_cache[key] = [track, *tail]
            return chain_cache[key]

        time_first_tracks = max(
            (time_first_chain(track) for track in time_first_candidates),
            key=lambda chain: (len(chain), sum(int(item["record_count"]) for item in chain)),
            default=[],
        )
        expanded_standard_tracks = []
        for index, track in enumerate(standard_tracks):
            expanded_standard_tracks.append(track)
            if index + 1 >= len(standard_tracks):
                continue
            next_track = standard_tracks[index + 1]
            if int(next_track["bone"]) - int(track["bone"]) != 2:
                continue
            continuation = best_cmp_rotation_continuation(
                blob,
                int(track["end"]),
                int(next_track["rel"]),
                int(track["bone"]) + 1,
                bone_count,
            )
            if continuation is not None:
                expanded_standard_tracks.append(continuation)

        if standard_tracks:
            final_track = standard_tracks[-1]
            table_start = cmp_terminal_bone_table_start(
                blob, int(final_track["end"]), section_end, bone_count
            )
            terminal_end = table_start if table_start is not None else section_end
            if int(final_track["bone"]) + 1 < bone_count and terminal_end > int(final_track["end"]):
                continuation = best_cmp_rotation_continuation(
                    blob,
                    int(final_track["end"]),
                    terminal_end,
                    int(final_track["bone"]) + 1,
                    bone_count,
                )
                if continuation is not None:
                    expanded_standard_tracks.append(continuation)

        candidate_tracks = max(
            (expanded_standard_tracks, time_first_tracks),
            key=lambda tracks: (len(tracks), sum(int(item["record_count"]) for item in tracks)),
        )
        if (
            len(candidate_tracks),
            sum(int(item["record_count"]) for item in candidate_tracks),
        ) > (
            len(best_tracks),
            sum(int(item["record_count"]) for item in best_tracks),
        ):
            best_tracks = candidate_tracks

    unique = []
    seen_bones = set()
    for track in best_tracks:
        bone = int(track["bone"])
        if bone in seen_bones:
            continue
        seen_bones.add(bone)
        unique.append(track)
    return unique


def patch_cmp_animation_translation_tracks(
    data: bytearray,
    entries: list[dict],
    skeleton_patch: dict,
) -> dict:
    changed_bones = skeleton_patch.get("changed_bones") or []
    local_deltas = {
        int(bone["idx"]): tuple(
            float(bone["new_local_translation"][axis])
            - float(bone["old_local_translation"][axis])
            for axis in range(3)
        )
        for bone in changed_bones
        if "idx" in bone
        and len(bone.get("old_local_translation") or ()) == 3
        and len(bone.get("new_local_translation") or ()) == 3
    }
    local_deltas = {
        bone: delta
        for bone, delta in local_deltas.items()
        if any(abs(value) > 1e-6 for value in delta)
    }
    if not local_deltas:
        return {"status": "skipped_no_changed_bone_positions"}

    clips_patched = 0
    tracks_patched = 0
    keys_patched = 0
    malformed_clips = []
    bodyless_clips = []
    clip_reports = []
    for entry in entries:
        if entry["file_type"] != 4 or entry["is_resource"]:
            continue
        base = int(entry["offset"])
        size = int(entry["size"])
        if base < 0 or size < 0 or base + size > len(data):
            continue
        blob = bytes(data[base : base + size])
        if len(blob) < 0x24 or struct.unpack_from("<I", blob, 0x20)[0] != 2:
            continue
        if len(blob) >= 0x40 and struct.unpack_from("<I", blob, 0x3C)[0] > len(blob):
            # Some CMPs contain tiny animation reference stubs whose headers
            # describe an external/shared body that is not stored in this
            # entry. There are no local translation keys to rewrite.
            bodyless_clips.append(str(entry["name"]))
            continue
        tracks = parse_cmp_animation_translation_tracks(blob)
        if tracks is None:
            malformed_clips.append(str(entry["name"]))
            continue

        clip_tracks = 0
        clip_keys = 0
        for track in tracks:
            bone = int(track["bone"])
            delta = local_deltas.get(bone)
            if delta is None:
                continue
            key_count = int(track["key_count"])
            records_pos = base + int(track["records_pos"])
            old_scales = tuple(float(value) for value in track["scales"])
            new_scales = list(old_scales)
            axis_values: list[list[float]] = [[], [], []]
            raw_records = []
            for key_index in range(key_count):
                record_pos = records_pos + key_index * 8
                time, x, y, z = struct.unpack_from("<Hhhh", data, record_pos)
                raw_records.append((record_pos, time, (x, y, z)))
                for axis, raw in enumerate((x, y, z)):
                    value = float(raw) / 32767.0 * old_scales[axis]
                    axis_values[axis].append(value + delta[axis])

            for axis in range(3):
                if abs(delta[axis]) <= 1e-8:
                    continue
                required_scale = max(abs(value) for value in axis_values[axis])
                old_capacity = abs(old_scales[axis])
                new_scales[axis] = max(old_capacity, required_scale, 1e-12)
            if int(track.get("scale_count", 3)) == 1:
                shared_scale = max(new_scales)
                new_scales = [shared_scale, shared_scale, shared_scale]
                struct.pack_into("<f", data, base + int(track["scale_pos"]), shared_scale)
            else:
                struct.pack_into("<3f", data, base + int(track["scale_pos"]), *new_scales)

            for key_index, (record_pos, _time, raw_values) in enumerate(raw_records):
                for axis in range(3):
                    if abs(delta[axis]) <= 1e-8:
                        continue
                    scale_value = new_scales[axis]
                    quantized = int(round(axis_values[axis][key_index] / scale_value * 32767.0))
                    quantized = max(-32767, min(32767, quantized))
                    struct.pack_into("<h", data, record_pos + 2 + axis * 2, quantized)
            clip_tracks += 1
            clip_keys += key_count

        if clip_tracks:
            clips_patched += 1
            tracks_patched += clip_tracks
            keys_patched += clip_keys
            clip_reports.append(
                {
                    "clip": str(entry["name"]),
                    "tracks_patched": clip_tracks,
                    "keys_patched": clip_keys,
                }
            )

    return {
        "status": "patched" if clips_patched else "skipped_no_matching_animation_tracks",
        "clips_patched": clips_patched,
        "tracks_patched": tracks_patched,
        "keys_patched": keys_patched,
        "bones_retargeted": sorted(local_deltas),
        "malformed_clips_skipped": malformed_clips,
        "bodyless_animation_stubs_skipped": bodyless_clips,
        "clips": clip_reports[:20],
        "writeback_scope": "compressed local translation keys only; key times and quaternion rotation streams are preserved",
    }


def patch_cmp_deleted_bone_animation_locks(
    data: bytearray,
    entries: list[dict],
    skeleton_patch: dict,
    bones: list[dict],
) -> dict:
    locked = {int(index) for index in skeleton_patch.get("deleted_or_missing_bone_indices") or []}
    rest = {int(bone["idx"]): bone for bone in bones if int(bone["idx"]) in locked}
    if not rest:
        return {"status": "skipped_no_deleted_bone_locks"}

    clips_patched = 0
    translation_tracks = 0
    translation_records = 0
    rotation_tracks = 0
    rotation_records = 0
    bodyless_clips = []
    clip_reports = []
    for entry in entries:
        if entry["file_type"] != 4 or entry["is_resource"]:
            continue
        base = int(entry["offset"])
        size = int(entry["size"])
        if base < 0 or size < 0 or base + size > len(data):
            continue
        blob = bytes(data[base : base + size])
        if len(blob) < 0x24 or struct.unpack_from("<I", blob, 0x20)[0] != 2:
            continue
        if len(blob) >= 0x40 and struct.unpack_from("<I", blob, 0x3C)[0] > len(blob):
            bodyless_clips.append(str(entry["name"]))
            continue

        clip_translation_tracks = 0
        clip_rotation_tracks = 0
        tracks = parse_cmp_animation_translation_tracks(blob) or []
        for track in tracks:
            bone = int(track["bone"])
            if bone not in rest:
                continue
            target = tuple(float(value) for value in rest[bone].get("t", (0.0, 0.0, 0.0)))
            old_scales = tuple(float(value) for value in track["scales"])
            new_scales = [max(abs(old_scales[axis]), abs(target[axis]), 1e-12) for axis in range(3)]
            if int(track.get("scale_count", 3)) == 1:
                shared_scale = max(new_scales)
                new_scales = [shared_scale] * 3
                struct.pack_into("<f", data, base + int(track["scale_pos"]), shared_scale)
            else:
                struct.pack_into("<3f", data, base + int(track["scale_pos"]), *new_scales)
            quantized = [
                max(-32767, min(32767, int(round(target[axis] / new_scales[axis] * 32767.0))))
                for axis in range(3)
            ]
            for key_index in range(int(track["key_count"])):
                record_pos = base + int(track["records_pos"]) + key_index * 8
                struct.pack_into("<hhh", data, record_pos + 2, *quantized)
                translation_records += 1
            translation_tracks += 1
            clip_translation_tracks += 1

        for track in parse_cmp_animation_rotation_tracks(blob):
            bone = int(track["bone"])
            if bone not in rest:
                continue
            quaternion = tuple(float(value) for value in rest[bone].get("q", (0.0, 0.0, 0.0, 1.0)))
            length = math.sqrt(sum(value * value for value in quaternion)) or 1.0
            normalized = tuple(value / length for value in quaternion)
            for record_index in range(int(track["record_count"])):
                record_pos = base + int(track["records_pos"]) + record_index * 8
                if track["layout"] == "explicit_qxyz_time":
                    q_offset = 0
                    time = struct.unpack_from("<H", data, record_pos + 6)[0]
                else:
                    q_offset = 2
                    time = struct.unpack_from("<H", data, record_pos)[0]
                w_negative = bool(time & 1)
                serialized = normalized
                if (serialized[3] < 0.0) != w_negative:
                    serialized = tuple(-value for value in serialized)
                qxyz = tuple(
                    max(-32767, min(32767, int(round(-serialized[axis] * 32767.0))))
                    for axis in range(3)
                )
                struct.pack_into("<hhh", data, record_pos + q_offset, *qxyz)
                rotation_records += 1
            rotation_tracks += 1
            clip_rotation_tracks += 1

        if clip_translation_tracks or clip_rotation_tracks:
            clips_patched += 1
            clip_reports.append(
                {
                    "clip": str(entry["name"]),
                    "translation_tracks": clip_translation_tracks,
                    "rotation_tracks": clip_rotation_tracks,
                }
            )

    return {
        "status": "patched_deleted_bone_locks" if clips_patched else "skipped_no_matching_deleted_bone_tracks",
        "locked_bones": sorted(locked),
        "clips_patched": clips_patched,
        "translation_tracks_patched": translation_tracks,
        "translation_records_patched": translation_records,
        "rotation_tracks_patched": rotation_tracks,
        "rotation_records_patched": rotation_records,
        "bodyless_animation_stubs_skipped": bodyless_clips,
        "clips": clip_reports[:20],
        "lock_scope": "exact deleted FBX bone nodes only; children remain animated unless separately deleted",
    }


def patch_cmp_skeleton_from_fbx(
    data: bytearray,
    parser: PipeworksParser,
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
            and "SKELETON" in entry["name"].upper()
            and "CAMERA" not in entry["name"].upper()
        ),
        None,
    )
    pose_entry = next(
        (
            entry
            for entry in entries
            if entry["file_type"] == 4
            and not entry["is_resource"]
            and "SKELETON" in entry["name"].upper()
        ),
        None,
    )
    if not hierarchy_entry or not pose_entry or not bones:
        return {"status": "skipped_missing_cmp_skeleton"}, {}

    hierarchy_start = int(hierarchy_entry["offset"])
    hierarchy_data = bytes(
        data[hierarchy_start : hierarchy_start + int(hierarchy_entry["size"])]
    )
    hierarchy_records = parse_cmp_type3_pose_records(hierarchy_data)
    pose_start = int(pose_entry["offset"])
    pose_data = bytes(data[pose_start : pose_start + int(pose_entry["size"])])
    pose_records = parse_cmp_pose_records(pose_data)
    models = fbx_bone_models(roots)
    matched: dict[int, object] = {}
    missing: list[str] = []
    for bone in bones:
        model = find_cmp_bone_model(models, str(bone["name"]))
        if model is None:
            missing.append(str(bone["name"]))
        else:
            matched[int(bone["idx"])] = model

    locked_indices = {int(bone["idx"]) for bone in bones if int(bone["idx"]) not in matched}

    safe_scale = scale if abs(scale) > 1e-8 else 1.0
    raw_local_translations = {}
    raw_bones = [dict(bone) for bone in bones]
    raw_by_idx = {int(bone["idx"]): bone for bone in raw_bones}
    for bone in bones:
        idx = int(bone["idx"])
        if idx in locked_indices:
            continue
        props = matched[idx].child("Properties70")
        translation = p_values(props, "Lcl Translation")
        if not translation or len(translation) < 3:
            continue
        new_t = tuple(float(value) / safe_scale for value in translation[:3])
        raw_local_translations[idx] = new_t
        raw_by_idx[idx]["t"] = new_t

    old_globals = global_matrices(bones, 1.0)
    raw_globals = global_matrices(raw_bones, 1.0)
    effective_local_translations = dict(raw_local_translations)
    ignored_child_compensations = []

    def global_translation_delta(index: int) -> tuple[float, float, float]:
        old_matrix = old_globals[index]
        new_matrix = raw_globals[index]
        return tuple(float(new_matrix[axis][3] - old_matrix[axis][3]) for axis in range(3))

    def delta_length(delta: tuple[float, float, float]) -> float:
        return math.sqrt(sum(value * value for value in delta))

    # Blender keeps disconnected child bones at their old world position when
    # only a parent is translated in Edit Mode. That appears in FBX as an equal
    # and opposite child-local edit. Treat that exact zero-world-motion pattern
    # like Pose Mode parenting so the child subtree follows the moved parent.
    for bone in bones:
        idx = int(bone["idx"])
        parent = int(bone["parent"])
        new_t = raw_local_translations.get(idx)
        if new_t is None or parent == idx or parent not in raw_globals:
            continue
        old_t = tuple(float(value) for value in bone["t"])
        if not any(abs(new_t[axis] - old_t[axis]) > 1e-3 for axis in range(3)):
            continue
        parent_delta = global_translation_delta(parent)
        child_delta = global_translation_delta(idx)
        if delta_length(parent_delta) <= 1e-3 or delta_length(child_delta) > 1e-3:
            continue
        effective_local_translations[idx] = old_t
        ignored_child_compensations.append(
            {
                "idx": idx,
                "name": str(bone["name"]),
                "parent_index": parent,
                "parent_name": str(next(item["name"] for item in bones if int(item["idx"]) == parent)),
                "discarded_local_translation": new_t,
                "preserved_local_translation": old_t,
            }
        )

    changed_bones = []
    new_bones = [dict(bone) for bone in bones]
    new_by_idx = {int(bone["idx"]): bone for bone in new_bones}
    rotations_seen = 0
    unchanged = 0
    for bone in bones:
        idx = int(bone["idx"])
        if idx in locked_indices:
            continue
        model = matched[idx]
        props = model.child("Properties70")
        new_t = effective_local_translations.get(idx)
        if new_t is None:
            continue
        rotation = p_values(props, "Lcl Rotation")
        if rotation and any(abs(float(value)) > 1e-6 for value in rotation[:3]):
            rotations_seen += 1
        old_t = tuple(float(value) for value in bone["t"])
        if not any(abs(new_t[axis] - old_t[axis]) > 1e-3 for axis in range(3)):
            unchanged += 1
            continue
        record = pose_records.get(idx)
        hierarchy_record = hierarchy_records.get(idx)
        if record is None or hierarchy_record is None:
            continue
        current = tuple(float(value) for value in record["translation"])
        hierarchy_current = tuple(float(value) for value in hierarchy_record["translation"])
        if (
            any(abs(current[axis] - old_t[axis]) > 0.05 for axis in range(3))
            or any(abs(hierarchy_current[axis] - old_t[axis]) > 0.05 for axis in range(3))
        ):
            continue
        translation_pos = pose_start + int(record["translation_pos"])
        hierarchy_translation_pos = hierarchy_start + int(hierarchy_record["translation_pos"])
        struct.pack_into("<3f", data, translation_pos, *new_t)
        struct.pack_into("<3f", data, hierarchy_translation_pos, *new_t)
        new_by_idx[idx]["t"] = new_t
        changed_bones.append(
            {
                "idx": idx,
                "name": str(bone["name"]),
                "old_local_translation": old_t,
                "new_local_translation": new_t,
                "type3_pose_offset": hex(int(hierarchy_record["translation_pos"])),
                "type4_pose_offset": hex(int(record["translation_pos"])),
            }
        )

    new_globals = global_matrices(new_bones, 1.0)
    global_deltas: dict[int, tuple[float, float, float]] = {}
    inverse_bind_bones_patched = []
    for idx, old_matrix in old_globals.items():
        new_matrix = new_globals.get(idx)
        if new_matrix is None:
            continue
        delta = tuple(float(new_matrix[axis][3] - old_matrix[axis][3]) for axis in range(3))
        if any(abs(value) > 1e-5 for value in delta):
            global_deltas[int(idx)] = delta
            hierarchy_record = hierarchy_records.get(int(idx))
            if hierarchy_record is None:
                continue
            inverse_bind_pos = hierarchy_start + int(hierarchy_record["inverse_bind_pos"])
            old_values = cmp_inverse_bind_values(old_matrix)
            new_values = cmp_inverse_bind_values(new_matrix)
            stored_values = struct.unpack_from("<12f", data, inverse_bind_pos)
            # Some bones, notably Titanosaurus's Jaw, intentionally carry a
            # bind offset that does not equal the inverse of the exported rest
            # transform. Preserve that native offset and apply only the matrix
            # delta caused by the edited bone position.
            updated_values = tuple(
                stored + (new - old)
                for stored, old, new in zip(stored_values, old_values, new_values)
            )
            struct.pack_into("<12f", data, inverse_bind_pos, *updated_values)
            inverse_bind_bones_patched.append(int(idx))

    return {
        "status": "patched" if changed_bones else "skipped_no_changed_bone_positions",
        "position_bones_patched": len(changed_bones),
        "changed_bones": changed_bones,
        "bones_with_global_delta": len(global_deltas),
        "inverse_bind_bones_patched": len(inverse_bind_bones_patched),
        "inverse_bind_bone_indices": inverse_bind_bones_patched,
        "inverse_bind_policy": "preserve native bind offsets and apply edited global-transform deltas",
        "deleted_or_missing_bones_preserved": missing[:20],
        "deleted_or_missing_count": len(missing),
        "deleted_or_missing_bone_indices": sorted(locked_indices),
        "unchanged_bones_seen": unchanged,
        "rotation_values_seen_but_preserved": rotations_seen,
        "hierarchy_compensations_ignored": len(ignored_child_compensations),
        "hierarchy_compensation_bones": ignored_child_compensations,
        "hierarchy_move_policy": "parent translations carry child subtrees when Blender leaves disconnected children at zero world displacement",
        "fbx_export_scale": safe_scale,
        "lock_rule": "delete a bone node from the edited FBX to freeze only that CMP bone; children require their own deletion",
        "writeback_scope": "Type-3 inverse bind matrices plus Type-3 and Type-4 rest translations; hierarchy and rest rotations are preserved",
    }, global_deltas


def bake_cmp_vertex_for_skeleton_move(
    target: bytearray,
    position_offset: int,
    weights: list[tuple[int, float]] | None,
    global_deltas: dict[int, tuple[float, float, float]],
) -> float:
    if not weights or not global_deltas:
        return 0.0
    dx = dy = dz = 0.0
    for bone, weight in weights:
        delta = global_deltas.get(int(bone))
        if delta is None:
            continue
        dx += delta[0] * float(weight)
        dy += delta[1] * float(weight)
        dz += delta[2] * float(weight)
    magnitude = math.sqrt(dx * dx + dy * dy + dz * dz)
    if magnitude <= 1e-5:
        return 0.0
    x, y, z = struct.unpack_from("<3f", target, position_offset)
    struct.pack_into("<3f", target, position_offset, x + dx, y + dy, z + dz)
    return magnitude


def bake_existing_cmp_packets_for_skeleton_move(
    data: bytearray,
    main: bytes,
    resource: bytes | bytearray,
    resource_offset: int,
    packets: list[dict],
    skin_palette: dict[int, int],
    skeleton_global_deltas: dict[int, tuple[float, float, float]],
) -> dict:
    vertices_moved = 0
    max_weighted_delta = 0.0
    packet_reports = []
    for packet_index, packet in enumerate(packets):
        count = int(packet["count"])
        controls = [
            read_cmp_vertex_control(resource, int(packet["rel"]) + j * 16)
            for j in range(count)
        ]
        use_vertex_draw_flags = cmp_packet_uses_vertex_draw_flags(controls)
        decoded_weights = [decode_cmp_skin_control(control, skin_palette) for control in controls]
        stream_skin = (
            not use_vertex_draw_flags
            and "compact_fmt" not in packet
            and bool(decoded_weights)
            and all(weights is not None for weights in decoded_weights)
        )
        compact_skin_bone = None
        compact_bone_off = int(packet["desc"]) + 0x18
        if "compact_fmt" in packet and compact_bone_off + 4 <= len(main):
            compact_skin_bone = skin_palette.get(struct.unpack_from("<I", main, compact_bone_off)[0])

        packet_vertices_moved = 0
        for vertex_index in range(count):
            weights = None
            if stream_skin:
                weights = decoded_weights[vertex_index]
            elif compact_skin_bone is not None:
                weights = [(compact_skin_bone, 1.0)]
            position_offset = resource_offset + int(packet["rel"]) + vertex_index * 16
            delta = bake_cmp_vertex_for_skeleton_move(
                data, position_offset, weights, skeleton_global_deltas
            )
            if delta > 0.0:
                packet_vertices_moved += 1
                vertices_moved += 1
                max_weighted_delta = max(max_weighted_delta, delta)

        packet_reports.append(
            {
                "packet": packet_index,
                "vertices": count,
                "vertices_moved": packet_vertices_moved,
                "skin_mode": (
                    "two_bone_stream"
                    if stream_skin
                    else ("compact" if compact_skin_bone is not None else "preserved_unrecognized")
                ),
            }
        )

    return {
        "status": "patched" if vertices_moved else "skipped_no_weighted_global_bone_delta",
        "vertices_moved": vertices_moved,
        "bones_with_global_delta": len(skeleton_global_deltas),
        "max_weighted_delta": max_weighted_delta,
        "packets": packet_reports,
    }


def remove_full_original_fallback_weight(
    fbx_weights: dict[str, float],
    original_weights: list[tuple[int, float]],
    bone_name_to_index: dict[str, int],
    palette_by_bone: dict[int, list[int]],
) -> tuple[dict[str, float], bool]:
    if len(original_weights) != 1 or original_weights[0][1] < 0.999999:
        return fbx_weights, False

    fallback_bone = int(original_weights[0][0])
    resolved = [
        (name, cmp_weight_bone_index(name, bone_name_to_index), float(weight))
        for name, weight in fbx_weights.items()
        if float(weight) > 1e-8
    ]
    fallback_total = sum(weight for _name, bone, weight in resolved if bone == fallback_bone)
    has_painted_bone = any(
        bone is not None and bone in palette_by_bone and bone != fallback_bone
        for _name, bone, _weight in resolved
    )
    if fallback_total < 0.999999 or not has_painted_bone:
        return fbx_weights, False

    return {
        name: weight
        for name, weight in fbx_weights.items()
        if cmp_weight_bone_index(name, bone_name_to_index) != fallback_bone
    }, True


def normalized_cmp_weights(
    fbx_weights: dict[str, float],
    bone_name_to_index: dict[str, int],
    palette_by_bone: dict[int, list[int]],
) -> tuple[list[tuple[int, float]], list[str], bool]:
    combined: dict[int, float] = collections.defaultdict(float)
    unknown: list[str] = []
    for name, weight in fbx_weights.items():
        bone_index = cmp_weight_bone_index(name, bone_name_to_index)
        if bone_index is None or bone_index not in palette_by_bone:
            unknown.append(str(name))
            continue
        if float(weight) > 1e-8:
            combined[bone_index] += float(weight)
    ordered = sorted(combined.items(), key=lambda item: item[1], reverse=True)
    reduced = len(ordered) > 2
    ordered = ordered[:2]
    total = sum(weight for _bone, weight in ordered)
    if total <= 1e-8:
        return [], sorted(set(unknown)), reduced
    return [(bone, weight / total) for bone, weight in ordered], sorted(set(unknown)), reduced


def weight_maps_match(a: list[tuple[int, float]], b: list[tuple[int, float]]) -> bool:
    left = dict(a)
    right = dict(b)
    tolerance = 0.51 / 4096.0
    return all(abs(left.get(bone, 0.0) - right.get(bone, 0.0)) <= tolerance for bone in set(left) | set(right))


def encode_cmp_skin_control(
    weights: list[tuple[int, float]],
    old_control: bytes,
    skin_palette: dict[int, int],
    palette_by_bone: dict[int, list[int]],
) -> bytes | None:
    if not weights:
        return None
    old_raw_a = old_raw_b = None
    if len(old_control) == 4:
        _old_blend, old_raw_a, old_raw_b = struct.unpack("<HBB", old_control)

    target = dict(weights)
    if len(target) == 1:
        bone = next(iter(target))
        preferred = [raw for raw in (old_raw_a, old_raw_b) if raw is not None and skin_palette.get(raw) == bone]
        raw = preferred[0] if preferred else palette_by_bone[bone][0]
        return struct.pack("<HBB", 4096, raw, raw)

    bones = [bone for bone, _weight in weights]
    old_bones = (
        skin_palette.get(old_raw_a) if old_raw_a is not None else None,
        skin_palette.get(old_raw_b) if old_raw_b is not None else None,
    )
    if old_bones[0] in target and old_bones[1] in target and old_bones[0] != old_bones[1]:
        bone_a, bone_b = old_bones
        raw_a, raw_b = int(old_raw_a), int(old_raw_b)
    else:
        bone_a, bone_b = bones
        raw_a = palette_by_bone[bone_a][0]
        raw_b = palette_by_bone[bone_b][0]
    blend = max(0, min(4096, int(round(target[bone_a] * 4096.0))))
    return struct.pack("<HBB", blend, raw_a, raw_b)


def clamp_s16(value: float) -> int:
    return max(-32768, min(32767, int(round(value))))


def pack_cmp_uv(value: float) -> int:
    return max(-32768, min(32767, int(round(value * 4096.0))))


def pack_cmp_normal(normal: tuple[float, float, float]) -> bytes:
    normalized = normalize_vector3(normal) or (0.0, 0.0, 1.0)
    values = [max(-127, min(127, int(round(value * 127.0)))) for value in normalized]
    return struct.pack("<bbb", *values)


def nearest_source_index(
    vertex: tuple[float, float, float],
    uv: tuple[float, float] | None,
    source_vertices: list[tuple[float, float, float]],
    source_uvs: list[tuple[float, float] | None],
) -> int:
    best_i = 0
    best_score = float("inf")
    for i, src in enumerate(source_vertices):
        if uv is not None and source_uvs[i] is not None:
            du = uv[0] - source_uvs[i][0]
            dv = uv[1] - source_uvs[i][1]
            score = du * du + dv * dv
        else:
            dx = vertex[0] - src[0]
            dy = vertex[1] - src[1]
            dz = vertex[2] - src[2]
            score = dx * dx + dy * dy + dz * dz
        if score < best_score:
            best_score = score
            best_i = i
    return best_i


def packet_count_main_offset(packet: dict) -> int | None:
    return packet.get("desc")


def packet_stored_count(packet: dict, vertex_count: int) -> int:
    return max(0, vertex_count + int(packet.get("stored_count_delta", 0)))


def packet_draw_count_offsets(index: int, packet_count: int, material_ranges: list[dict]) -> tuple[int, ...]:
    if len(material_ranges) == packet_count and index < len(material_ranges):
        return (material_ranges[index]["draw_offset"],)
    return ()


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

    # The bundle header stores the total byte length of the resource section at
    # 0x74. The game uses this outer limit even when a resource's TOC size grows.
    if len(data) < 0x78:
        raise ValueError("Pipeworks bundle header is missing the resource-section size")
    resource_section_size = struct.unpack_from(f"{endian}I", data, 0x74)[0]
    resized_resource_section = resource_section_size + delta
    if resized_resource_section < 0:
        raise ValueError("CMP resource resize would make the bundle resource section negative")
    struct.pack_into(f"{endian}I", data, 0x74, resized_resource_section)
    actual_resource_section = len(data) - parser.resource_data_offset
    if resized_resource_section != actual_resource_section:
        raise ValueError(
            "CMP resource resize produced an inconsistent bundle resource-section size: "
            f"header={resized_resource_section}, actual={actual_resource_section}"
        )

    for header_off in (0x34, 0x64, 0x68, 0x70):
        value = struct.unpack_from(f"{endian}I", data, header_off)[0]
        if value >= insert_pos:
            struct.pack_into(f"{endian}I", data, header_off, value + delta)

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
            if main_abs >= insert_pos:
                struct.pack_into(f"{endian}I", data, toc + 2, main_rel + delta)

        res_rel = struct.unpack_from(f"{endian}I", data, toc + 10)[0]
        res_size = struct.unpack_from(f"{endian}I", data, toc + 14)[0]
        if res_size:
            res_abs = parser.resource_data_offset + res_rel
            if res_abs >= insert_pos:
                struct.pack_into(f"{endian}I", data, toc + 10, res_rel + delta)


def patch_cmp_positions(
    original: Path,
    fbx: Path,
    out: Path,
    scale: float,
    keep_mesh_in_place: bool = False,
) -> dict:
    if original.resolve() != out.resolve():
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, out)

    parser = PipeworksParser(str(out))
    entries = parser.parse()
    data = bytearray(parser.file_data or b"")
    main_entry = next((e for e in entries if e["file_type"] == 17 and not e["is_resource"]), None)
    res_entry = next((e for e in entries if e["file_type"] == 17 and e["is_resource"]), None)
    if not main_entry or not res_entry:
        raise SystemExit("No CMP type-17 mesh/resource pair found")
    main = bytes(data[main_entry["offset"] : main_entry["offset"] + main_entry["size"]])
    resource = data[res_entry["offset"] : res_entry["offset"] + res_entry["size"]]
    original_resource_size = len(resource)
    packets = find_cmp_packets(main, bytes(resource))
    materials = parse_cmp_materials(bytes(data), entries)
    material_ranges = parse_cmp_material_ranges(main, packets, materials)
    strings = bundle_strings(parser)
    bones, _globals = parse_cmp_skeleton(parser, entries, 1.0)
    skin_palette = parse_cmp_skin_palette(main, strings, bones)
    palette_by_bone: dict[int, list[int]] = collections.defaultdict(list)
    for raw_id, bone_index in skin_palette.items():
        palette_by_bone[int(bone_index)].append(int(raw_id))
    bone_name_to_index = cmp_bone_name_map(bones)

    roots, _version = parse_fbx(fbx)
    geometries = sorted(
        [g for g in object_nodes(roots, "Geometry") if "_packet" in geometry_name(g).lower()],
        key=geometry_packet_sort_key,
    )
    if len(geometries) != len(packets):
        mesh_mismatch = {
            "status": "skipped_packet_count_changed",
            "fbx_packets": len(geometries),
            "cmp_packets": len(packets),
        }
    else:
        mesh_mismatch = next(
            (
                {
                    "status": "skipped_vertex_count_changed",
                    "packet": i,
                    "fbx_vertices": len(geometry_vertices(geom)),
                    "cmp_vertices": int(packet["count"]),
                }
                for i, (packet, geom) in enumerate(zip(packets, geometries))
                if len(geometry_vertices(geom)) < int(packet["count"])
            ),
            None,
        )

    skeleton_patch, skeleton_global_deltas = patch_cmp_skeleton_from_fbx(
        data, parser, entries, roots, bones, scale
    )
    mesh_bake_deltas = {} if keep_mesh_in_place else skeleton_global_deltas
    animation_translation_patch = patch_cmp_animation_translation_tracks(
        data, entries, skeleton_patch
    )
    animation_lock_patch = patch_cmp_deleted_bone_animation_locks(
        data, entries, skeleton_patch, bones
    )
    if mesh_mismatch is not None:
        mesh_skeleton_bake = bake_existing_cmp_packets_for_skeleton_move(
            data,
            main,
            resource,
            int(res_entry["offset"]),
            packets,
            skin_palette,
            mesh_bake_deltas,
        )
        if keep_mesh_in_place:
            mesh_skeleton_bake.update(
                {
                    "status": "skipped_keep_mesh_in_place",
                    "bones_with_global_delta": len(skeleton_global_deltas),
                }
            )
        from cmp_animation_import import import_cmp_actions

        rebuilt_data, action_animation_patch = import_cmp_actions(
            bytes(data), entries, bones, fbx, scale, skeleton_patch=skeleton_patch
        )
        data = bytearray(rebuilt_data)
        out.write_bytes(data)
        return {
            "status": (
                "patched_skeleton_only_mesh_mismatch"
                if skeleton_patch.get("status") == "patched"
                else mesh_mismatch["status"]
            ),
            "mesh_import": mesh_mismatch,
            "skeleton_patch": skeleton_patch,
            "animation_translation_patch": animation_translation_patch,
            "animation_lock_patch": animation_lock_patch,
            "action_animation_patch": action_animation_patch,
            "mesh_skeleton_bake": mesh_skeleton_bake,
        }

    patched_packets = []
    totals = collections.Counter()
    unknown_bones: set[str] = set()
    max_skeleton_bake_delta = 0.0
    for i, (packet, geom) in enumerate(zip(packets, geometries)):
        vertices = geometry_vertices(geom)
        faces = geometry_faces(geom)
        uvs = geometry_uvs_by_cp(geom, len(vertices))
        normals = geometry_normals_by_cp(geom, len(vertices))
        fbx_weights, cluster_report = geometry_weights_by_cp(roots, geom, len(vertices))
        old_count = packet["count"]

        old_side = cmp_packet_side_stream(packet)
        old_uv = cmp_packet_uv_stream(packet)
        old_vertex_controls = [
            read_cmp_vertex_control(resource, packet["rel"] + j * 16)
            for j in range(old_count)
        ]
        old_side_records = [bytes(resource[old_side + j * 4 : old_side + j * 4 + 4]) for j in range(old_count)]
        use_vertex_draw_flags = cmp_packet_uses_vertex_draw_flags(old_vertex_controls)
        decoded_old_weights = [decode_cmp_skin_control(control, skin_palette) for control in old_vertex_controls]
        stream_skin = (
            not use_vertex_draw_flags
            and "compact_fmt" not in packet
            and bool(decoded_old_weights)
            and all(weights is not None for weights in decoded_old_weights)
        )
        compact_skin_bone = None
        compact_bone_off = int(packet["desc"]) + 0x18
        if "compact_fmt" in packet and compact_bone_off + 4 <= len(main):
            compact_skin_bone = skin_palette.get(struct.unpack_from("<I", main, compact_bone_off)[0])
        packet_stats = collections.Counter()
        packet_unknown_bones: set[str] = set()
        face_keys = {face_key(face) for face in faces}
        current_markers = [
            1
            if (
                cmp_vertex_control_draws(old_vertex_controls[j])
                if use_vertex_draw_flags
                else cmp_side_record_draws(old_side_records[j])
            )
            else 0
            for j in range(old_count)
        ]
        current_logical_indices = (
            None
            if use_vertex_draw_flags
            else cmp_stream_logical_indices(resource, packet, old_side, old_uv)
        )
        current_faces, _unused_uvs, _kept = build_adc_strip_faces(
            [(0.0, 0.0, 0.0)] * old_count,
            current_markers,
            current_logical_indices,
        )
        current_face_keys = {face_key(face) for face in current_faces}
        packed_position_records = [
            struct.pack("<fff", float(x) / scale, float(y) / scale, float(z) / scale)
            for x, y, z in vertices[:old_count]
        ]
        position_changed = []
        for record_index, vertex in enumerate(vertices[:old_count]):
            source_vertex = struct.unpack_from("<fff", resource, packet["rel"] + record_index * 16)
            delta_sq = sum((vertex[axis] - source_vertex[axis] * scale) ** 2 for axis in range(3))
            position_changed.append(delta_sq > 1e-8)
        topology_markers_to_enable: set[int] = set()
        restart_markers_to_disable: set[int] = set()
        for record_index in range(2, old_count):
            tri = strip_triangle(record_index)
            key = face_key(tri)
            if key in face_keys and key not in current_face_keys:
                topology_markers_to_enable.add(record_index)
            if current_logical_indices is None or not current_markers[record_index] or key in face_keys:
                continue
            originally_degenerate = len({current_logical_indices[index] for index in tri}) < 3
            edited_nondegenerate = len({packed_position_records[index] for index in tri}) == 3
            if originally_degenerate and edited_nondegenerate:
                restart_markers_to_disable.add(record_index)

        if len(vertices) == old_count:
            for j, (x, y, z) in enumerate(vertices):
                off = res_entry["offset"] + packet["rel"] + j * 16
                struct.pack_into("<fff", data, off, float(x) / scale, float(y) / scale, float(z) / scale)
                if not use_vertex_draw_flags and position_changed[j] and normals[j] is not None:
                    normal_off = res_entry["offset"] + old_side + j * 4
                    packed_normal = pack_cmp_normal(normals[j])
                    if bytes(data[normal_off : normal_off + 3]) != packed_normal:
                        data[normal_off : normal_off + 3] = packed_normal
                        packet_stats["normals_patched_moved_vertices"] += 1
                if uvs[j] is not None:
                    u, v = uvs[j]
                    packed_uv = struct.pack("<hh", pack_cmp_uv(u), pack_cmp_uv(1.0 - v))
                    uv_off = res_entry["offset"] + old_uv + j * 4
                    if bytes(data[uv_off : uv_off + 4]) != packed_uv:
                        data[uv_off : uv_off + 4] = packed_uv
                        packet_stats["uvs_patched"] += 1
                else:
                    packet_stats["uvs_preserved_ambiguous_or_missing"] += 1

                if stream_skin and fbx_weights[j]:
                    source_fbx_weights, removed_fallback = remove_full_original_fallback_weight(
                        fbx_weights[j],
                        decoded_old_weights[j] or [],
                        bone_name_to_index,
                        palette_by_bone,
                    )
                    if removed_fallback:
                        packet_stats["weights_removed_full_original_fallback"] += 1
                    target_weights, unknown, reduced = normalized_cmp_weights(
                        source_fbx_weights, bone_name_to_index, palette_by_bone
                    )
                    packet_unknown_bones.update(unknown)
                    if unknown:
                        packet_stats["weights_preserved_unknown_bone"] += 1
                        target_weights = []
                    if reduced:
                        packet_stats["weights_preserved_more_than_two"] += 1
                        target_weights = []
                    old_weights = decoded_old_weights[j] or []
                    if target_weights and not weight_maps_match(target_weights, old_weights):
                        encoded = encode_cmp_skin_control(
                            target_weights,
                            old_vertex_controls[j],
                            skin_palette,
                            palette_by_bone,
                        )
                        if encoded is not None:
                            data[off + 12 : off + 16] = encoded
                            packet_stats["weights_patched"] += 1
                    elif target_weights:
                        packet_stats["weights_unchanged"] += 1
                elif stream_skin:
                    packet_stats["weights_preserved_missing_fbx"] += 1

                if j in topology_markers_to_enable:
                    if use_vertex_draw_flags:
                        data[off + 12 : off + 16] = b"\x00\x00\x80\x3f"
                    else:
                        marker_off = res_entry["offset"] + old_side + j * 4 + 3
                        data[marker_off] = 0x7F
                    packet_stats["topology_markers_patched"] += 1
                elif j in restart_markers_to_disable:
                    if use_vertex_draw_flags:
                        data[off + 12 : off + 16] = b"\x00\x00\x00\x00"
                    else:
                        marker_off = res_entry["offset"] + old_side + j * 4 + 3
                        data[marker_off] = 0
                    packet_stats["opened_restart_markers_disabled"] += 1

                bake_weights = None
                if stream_skin:
                    bake_weights = decode_cmp_skin_control(bytes(data[off + 12 : off + 16]), skin_palette)
                elif compact_skin_bone is not None:
                    bake_weights = [(compact_skin_bone, 1.0)]
                bake_delta = bake_cmp_vertex_for_skeleton_move(
                    data, off, bake_weights, mesh_bake_deltas
                )
                if bake_delta > 0.0:
                    packet_stats["skeleton_bake_vertices"] += 1
                    max_skeleton_bake_delta = max(max_skeleton_bake_delta, bake_delta)

            if not stream_skin and any(fbx_weights):
                packet_stats[
                    "weights_preserved_compact_binding"
                    if "compact_fmt" in packet
                    else "weights_preserved_draw_or_unrecognized_control"
                ] += sum(1 for weights in fbx_weights if weights)
            totals.update(packet_stats)
            unknown_bones.update(packet_unknown_bones)
            patched_packets.append(
                {
                    "packet": i,
                    "vertices": len(vertices),
                    "rel": packet["rel"],
                    "mode": "positions_uvs_weights",
                    "skin_mode": "two_bone_stream" if stream_skin else ("compact" if "compact_fmt" in packet else "preserved"),
                    "cluster_report": cluster_report,
                    **dict(packet_stats),
                    "unknown_bones": sorted(packet_unknown_bones),
                }
            )
            continue

        next_rel = packets[i + 1]["rel"] if i + 1 < len(packets) else len(resource)
        new_count = len(vertices)
        new_side = packet["rel"] + new_count * 16
        new_uv = (new_side + new_count * 4 + 15) & ~15
        new_end = new_uv + new_count * 4
        count_off = packet_count_main_offset(packet)
        can_grow_resource = i == len(packets) - 1 and new_end > len(resource)
        if count_off is None or (new_end > next_rel and not can_grow_resource):
            return {
                "status": "skipped_cmp_packet_grow_no_room",
                "packet": i,
                "fbx_vertices": new_count,
                "cmp_vertices": old_count,
                "new_end": new_end,
                "next_packet": next_rel,
            }

        source_vertices = [
            tuple(struct.unpack_from("<fff", resource, packet["rel"] + j * 16))
            for j in range(old_count)
        ]
        source_vertices_scaled = [(x * scale, y * scale, z * scale) for x, y, z in source_vertices]
        source_uvs = [
            read_cmp_uv(resource, old_uv, j)
            for j in range(old_count)
        ]
        write_order, reversed_components = added_vertex_write_order(old_count, new_count, faces)
        if reversed_components:
            packet_stats["winding_components_reversed"] += len(reversed_components)
            packet_stats["winding_vertices_reordered"] += sum(
                component["end"] - component["start"] + 1
                for component in reversed_components
            )
        out_resource = bytearray(resource)
        if new_end > len(out_resource):
            out_resource.extend(b"\x00" * (new_end - len(out_resource)))
        out_resource[packet["rel"] : next_rel] = b"\x00" * (next_rel - packet["rel"])
        for j in range(new_count):
            source_j = write_order[j]
            x, y, z = vertices[source_j]
            uv = uvs[source_j]
            dst_vertex = packet["rel"] + j * 16
            struct.pack_into("<fff", out_resource, dst_vertex, float(x) / scale, float(y) / scale, float(z) / scale)
            if j < old_count:
                out_resource[dst_vertex + 12 : dst_vertex + 16] = resource[packet["rel"] + j * 16 + 12 : packet["rel"] + j * 16 + 16]
                out_resource[new_side + j * 4 : new_side + j * 4 + 4] = resource[old_side + j * 4 : old_side + j * 4 + 4]
                if not use_vertex_draw_flags and position_changed[j] and normals[source_j] is not None:
                    out_resource[new_side + j * 4 : new_side + j * 4 + 3] = pack_cmp_normal(normals[source_j])
                    packet_stats["normals_patched_moved_vertices"] += 1
                if use_vertex_draw_flags:
                    if j in topology_markers_to_enable:
                        out_resource[dst_vertex + 12 : dst_vertex + 16] = b"\x00\x00\x80\x3f"
                        packet_stats["topology_markers_patched"] += 1
                    elif j in restart_markers_to_disable:
                        out_resource[dst_vertex + 12 : dst_vertex + 16] = b"\x00\x00\x00\x00"
                        packet_stats["opened_restart_markers_disabled"] += 1
            else:
                src_i = nearest_source_index((x, y, z), uv, source_vertices_scaled, source_uvs)
                if use_vertex_draw_flags:
                    tri = tuple(write_order[index] for index in strip_triangle(j)) if j >= 2 else ()
                    control = b"\x00\x00\x80\x3f" if tri and face_key(tri) in face_keys else b"\x00\x00\x00\x00"
                    out_resource[dst_vertex + 12 : dst_vertex + 16] = control
                else:
                    out_resource[dst_vertex + 12 : dst_vertex + 16] = resource[packet["rel"] + src_i * 16 + 12 : packet["rel"] + src_i * 16 + 16]
                if not use_vertex_draw_flags and normals[source_j] is not None:
                    out_resource[new_side + j * 4 : new_side + j * 4 + 3] = pack_cmp_normal(normals[source_j])
                    packet_stats["normals_written_grown_stream"] += 1
                else:
                    out_resource[new_side + j * 4 : new_side + j * 4 + 3] = resource[old_side + src_i * 4 : old_side + src_i * 4 + 3]
                    packet_stats["normals_inherited_grown_stream"] += 1
            if stream_skin and fbx_weights[source_j]:
                source_control = bytes(out_resource[dst_vertex + 12 : dst_vertex + 16])
                source_weights = decode_cmp_skin_control(source_control, skin_palette) or []
                source_fbx_weights, removed_fallback = remove_full_original_fallback_weight(
                    fbx_weights[source_j],
                    source_weights,
                    bone_name_to_index,
                    palette_by_bone,
                )
                if removed_fallback:
                    packet_stats["weights_removed_full_original_fallback"] += 1
                target_weights, unknown, reduced = normalized_cmp_weights(
                    source_fbx_weights, bone_name_to_index, palette_by_bone
                )
                packet_unknown_bones.update(unknown)
                if unknown:
                    packet_stats["weights_preserved_unknown_bone"] += 1
                    target_weights = []
                if reduced:
                    packet_stats["weights_preserved_more_than_two"] += 1
                    target_weights = []
                if target_weights and not weight_maps_match(target_weights, source_weights):
                    encoded = encode_cmp_skin_control(
                        target_weights,
                        source_control,
                        skin_palette,
                        palette_by_bone,
                    )
                    if encoded is not None:
                        out_resource[dst_vertex + 12 : dst_vertex + 16] = encoded
                        packet_stats["weights_patched"] += 1
                elif target_weights:
                    packet_stats["weights_unchanged"] += 1
            elif stream_skin:
                packet_stats["weights_preserved_missing_fbx"] += 1
            if j < old_count:
                marker = resource[old_side + j * 4 + 3]
                if not use_vertex_draw_flags:
                    if j in topology_markers_to_enable:
                        marker = 0x7F
                        packet_stats["topology_markers_patched"] += 1
                    elif j in restart_markers_to_disable:
                        marker = 0
                        packet_stats["opened_restart_markers_disabled"] += 1
            elif use_vertex_draw_flags:
                marker = resource[old_side + nearest_source_index((x, y, z), uv, source_vertices_scaled, source_uvs) * 4 + 3]
            elif j < 2:
                marker = 0
            else:
                tri = tuple(write_order[index] for index in strip_triangle(j))
                marker = 0x7F if face_key(tri) in face_keys else 0
            out_resource[new_side + j * 4 + 3] = marker
            u, v = uv if uv is not None else source_uvs[nearest_source_index((x, y, z), None, source_vertices_scaled, source_uvs)]
            struct.pack_into("<hh", out_resource, new_uv + j * 4, pack_cmp_uv(u), pack_cmp_uv(1.0 - v))
            if uv is not None:
                packet_stats["uvs_written_grown_stream"] += 1
            else:
                packet_stats["uvs_inherited_grown_stream"] += 1
            bake_weights = None
            if stream_skin:
                bake_weights = decode_cmp_skin_control(
                    bytes(out_resource[dst_vertex + 12 : dst_vertex + 16]), skin_palette
                )
            elif compact_skin_bone is not None:
                bake_weights = [(compact_skin_bone, 1.0)]
            bake_delta = bake_cmp_vertex_for_skeleton_move(
                out_resource, dst_vertex, bake_weights, mesh_bake_deltas
            )
            if bake_delta > 0.0:
                packet_stats["skeleton_bake_vertices"] += 1
                max_skeleton_bake_delta = max(max_skeleton_bake_delta, bake_delta)

        data[res_entry["offset"] : res_entry["offset"] + original_resource_size] = out_resource
        patch_bundle_resource_resize(data, parser, entries, res_entry, original_resource_size, len(out_resource))
        resource = out_resource
        struct.pack_into("<I", data, main_entry["offset"] + count_off, packet_stored_count(packet, new_count))
        old_region_size = packet.get("region_size")
        if old_region_size is not None and new_end > packet["rel"] + old_region_size:
            new_region_size = len(out_resource) - packet["rel"]
            struct.pack_into("<I", data, main_entry["offset"] + count_off + 0x14, new_region_size)
            packet_stats["packet_region_size_patched"] += 1
            packet_stats["packet_region_size_delta"] += new_region_size - old_region_size
        for draw_count_off in packet_draw_count_offsets(i, len(packets), material_ranges):
            struct.pack_into("<I", data, main_entry["offset"] + draw_count_off, max(0, new_count - 2))
        if not stream_skin and any(fbx_weights):
            packet_stats[
                "weights_preserved_compact_binding"
                if "compact_fmt" in packet
                else "weights_preserved_draw_or_unrecognized_control"
            ] += sum(1 for weights in fbx_weights if weights)
        totals.update(packet_stats)
        unknown_bones.update(packet_unknown_bones)
        patched_packets.append(
            {
                "packet": i,
                "vertices": new_count,
                "old_vertices": old_count,
                "added_vertices": new_count - old_count,
                "rel": packet["rel"],
                "mode": "grown_resource" if len(out_resource) > original_resource_size else "grown_in_padding",
                "resource_delta": len(out_resource) - original_resource_size,
                "skin_mode": "two_bone_stream" if stream_skin else ("compact" if "compact_fmt" in packet else "preserved"),
                "cluster_report": cluster_report,
                "reversed_winding_components": reversed_components,
                **dict(packet_stats),
                "unknown_bones": sorted(packet_unknown_bones),
            }
        )

    from cmp_animation_import import import_cmp_actions

    rebuilt_data, action_animation_patch = import_cmp_actions(
        bytes(data), entries, bones, fbx, scale, skeleton_patch=skeleton_patch
    )
    data = bytearray(rebuilt_data)
    out.write_bytes(data)
    skeleton_bake_vertices = int(totals.get("skeleton_bake_vertices", 0))
    return {
        "status": "patched_positions_uvs_weights",
        "packets": patched_packets,
        "totals": dict(totals),
        "unknown_bones": sorted(unknown_bones),
        "weight_limit": "two influences per vertex for native two-bone stream packets",
        "skeleton_patch": skeleton_patch,
        "animation_translation_patch": animation_translation_patch,
        "animation_lock_patch": animation_lock_patch,
        "action_animation_patch": action_animation_patch,
        "mesh_skeleton_bake": {
            "status": (
                "skipped_keep_mesh_in_place"
                if keep_mesh_in_place
                else ("patched" if skeleton_bake_vertices else "skipped_no_weighted_global_bone_delta")
            ),
            "vertices_moved": skeleton_bake_vertices,
            "bones_with_global_delta": len(skeleton_global_deltas),
            "max_weighted_delta": max_skeleton_bake_delta,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Import FBX position, UV, and supported two-bone weight edits into a PS2 Pipeworks CMP")
    ap.add_argument("fbx")
    ap.add_argument("original")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=float, default=10.0)
    ap.add_argument(
        "--keep-mesh-in-place",
        action="store_true",
        help="Import rest-bone positions without automatically moving weighted vertices",
    )
    args = ap.parse_args()
    report = patch_cmp_positions(
        Path(args.original),
        Path(args.fbx),
        Path(args.out),
        args.scale,
        keep_mesh_in_place=args.keep_mesh_in_place,
    )
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
