from __future__ import annotations

import argparse
import collections
import math
import struct
import sys
from pathlib import Path

TOOL_DIR = Path(__file__).resolve().parent
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from cmp_fbx_import import (
    cmp_bone_name_candidates,
    cmp_bone_name_map,
    encode_cmp_skin_control,
    geometry_name,
    geometry_weights_by_cp,
    normalized_cmp_weights,
    pack_cmp_normal,
    pack_cmp_uv,
    patch_bundle_resource_resize,
)
from cmp_probe import (
    build_adc_strip_faces,
    bundle_strings,
    cmp_packet_side_stream,
    cmp_packet_uses_vertex_draw_flags,
    decode_cmp_skin_control,
    find_cmp_packets,
    parse_cmp_material_ranges,
    parse_cmp_materials,
    parse_cmp_skeleton,
    parse_cmp_skin_palette,
    read_cmp_side_record,
    read_cmp_vertex_control,
)
from fbx_to_bdg_import import clean_fbx_object_name, find_first, object_nodes, p_values, parse_fbx
from parser_core import PipeworksParser


def align(value: int, boundary: int) -> int:
    return (value + boundary - 1) & ~(boundary - 1)


def vector_normal(a: tuple[float, float, float], b: tuple[float, float, float], c: tuple[float, float, float]) -> tuple[float, float, float]:
    ab = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    ac = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
    cross = (
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    )
    length = math.sqrt(sum(value * value for value in cross))
    if length <= 1e-12:
        return (0.0, 0.0, 1.0)
    return tuple(value / length for value in cross)


def fbx_relations(roots) -> tuple[dict[int, object], list[tuple[int, int]]]:
    nodes = object_nodes(roots)
    by_id = {
        int(node.props[0]): node
        for node in nodes
        if node.props and isinstance(node.props[0], int)
    }
    relations: list[tuple[int, int]] = []
    connections = find_first(roots, "Connections")
    if connections:
        for connection in connections.children_named("C"):
            if len(connection.props) >= 3 and str(connection.props[0]) == "OO":
                relations.append((int(connection.props[1]), int(connection.props[2])))
    return by_id, relations


def rotate_vector(
    vector: tuple[float, float, float],
    rotation: tuple[float, float, float],
    order: str,
) -> tuple[float, float, float]:
    result = vector
    angles = {axis: math.radians(rotation[index]) for index, axis in enumerate("XYZ")}
    for axis in order:
        angle = angles[axis]
        sine, cosine = math.sin(angle), math.cos(angle)
        x, y, z = result
        if axis == "X":
            result = (x, y * cosine - z * sine, y * sine + z * cosine)
        elif axis == "Y":
            result = (x * cosine + z * sine, y, -x * sine + z * cosine)
        else:
            result = (x * cosine - y * sine, x * sine + y * cosine, z)
    return result


def geometry_transform(geometry, by_id: dict[int, object], relations: list[tuple[int, int]]) -> dict:
    geometry_id = int(geometry.props[0])
    model_ids = [parent for child, parent in relations if child == geometry_id]
    models = [by_id[model_id] for model_id in model_ids if model_id in by_id and getattr(by_id[model_id], "name", "") == "Model"]
    if len(models) > 1:
        raise ValueError(f"FBX mesh {geometry_name(geometry)!r} is connected to more than one model object")
    result = {
        "name": geometry_name(geometry),
        "translation": (0.0, 0.0, 0.0),
        "rotation": (0.0, 0.0, 0.0),
        "scaling": (1.0, 1.0, 1.0),
        "order": "XYZ",
        "applied": False,
    }
    rotation_orders = ("XYZ", "XZY", "YZX", "YXZ", "ZXY", "ZYX")
    for model in models:
        props = model.child("Properties70")
        translation = p_values(props, "Lcl Translation") if props else None
        rotation = p_values(props, "Lcl Rotation") if props else None
        scaling = p_values(props, "Lcl Scaling") if props else None
        rotation_order = p_values(props, "RotationOrder") if props else None
        translation = tuple(map(float, translation[:3])) if translation else (0.0, 0.0, 0.0)
        rotation = tuple(map(float, rotation[:3])) if rotation else (0.0, 0.0, 0.0)
        scaling = tuple(map(float, scaling[:3])) if scaling else (1.0, 1.0, 1.0)
        order_index = int(rotation_order[0]) if rotation_order else 0
        if not (0 <= order_index < len(rotation_orders)):
            raise ValueError(
                f"Mesh object {clean_fbx_object_name(model.props[1])!r} uses unsupported FBX rotation order {order_index}"
            )
        if any(abs(value) <= 1e-12 for value in scaling):
            raise ValueError(f"Mesh object {clean_fbx_object_name(model.props[1])!r} has a zero scale axis")
        result = {
            "name": clean_fbx_object_name(model.props[1] if len(model.props) > 1 else geometry_name(geometry)),
            "translation": translation,
            "rotation": rotation,
            "scaling": scaling,
            "order": rotation_orders[order_index],
            "applied": (
            any(abs(value) > 1e-6 for value in translation)
            or any(abs(value) > 1e-6 for value in rotation)
            or any(abs(value - 1.0) > 1e-6 for value in scaling)
            ),
        }
    return result


def transform_position(position: tuple[float, float, float], transform: dict) -> tuple[float, float, float]:
    scaled = tuple(position[index] * transform["scaling"][index] for index in range(3))
    rotated = rotate_vector(scaled, transform["rotation"], transform["order"])
    return tuple(rotated[index] + transform["translation"][index] for index in range(3))


def transform_normal(normal: tuple[float, float, float], transform: dict) -> tuple[float, float, float]:
    inverse_scaled = tuple(normal[index] / transform["scaling"][index] for index in range(3))
    rotated = rotate_vector(inverse_scaled, transform["rotation"], transform["order"])
    length = math.sqrt(sum(value * value for value in rotated))
    if length <= 1e-12:
        return (0.0, 0.0, 1.0)
    return tuple(value / length for value in rotated)


def polygon_corners(geometry) -> list[list[tuple[int, int]]]:
    pvi_node = geometry.child("PolygonVertexIndex")
    if not pvi_node:
        raise ValueError(f"FBX mesh {geometry_name(geometry)!r} has no polygon index data")
    polygons: list[list[tuple[int, int]]] = []
    polygon: list[tuple[int, int]] = []
    for polygon_vertex, raw in enumerate(pvi_node.props[0]):
        raw = int(raw)
        polygon.append((~raw if raw < 0 else raw, polygon_vertex))
        if raw < 0:
            polygons.append(polygon)
            polygon = []
    if polygon:
        raise ValueError(f"FBX mesh {geometry_name(geometry)!r} has an unterminated polygon")
    return polygons


def layer_samples(
    geometry,
    layer_name: str,
    value_name: str,
    width: int,
    index_names: tuple[str, ...],
    control_point_count: int,
    polygon_count: int,
    polygon_vertex_count: int,
) -> tuple[object | None, str]:
    layers = geometry.children_named(layer_name)
    if not layers:
        return None, "missing"
    layer = layers[0]
    values_node = layer.child(value_name)
    if not values_node:
        return None, "missing"
    flat = values_node.props[0]
    direct = [tuple(map(float, flat[i : i + width])) for i in range(0, len(flat), width)]
    mapping_node = layer.child("MappingInformationType")
    reference_node = layer.child("ReferenceInformationType")
    mapping = str(mapping_node.props[0]) if mapping_node else "ByPolygonVertex"
    reference = str(reference_node.props[0]) if reference_node else "Direct"
    index_node = next((layer.child(name) for name in index_names if layer.child(name)), None)
    indices = [int(value) for value in index_node.props[0]] if index_node else []

    expected = {
        "ByPolygonVertex": polygon_vertex_count,
        "ByVertice": control_point_count,
        "ByVertex": control_point_count,
        "ByControlPoint": control_point_count,
        "ByPolygon": polygon_count,
        "AllSame": 1,
    }.get(mapping)
    if expected is None:
        raise ValueError(f"Unsupported {layer_name} mapping {mapping!r} in {geometry_name(geometry)!r}")

    def sample(control_point: int, polygon_index: int, polygon_vertex: int):
        source_index = {
            "ByPolygonVertex": polygon_vertex,
            "ByVertice": control_point,
            "ByVertex": control_point,
            "ByControlPoint": control_point,
            "ByPolygon": polygon_index,
            "AllSame": 0,
        }[mapping]
        if reference == "IndexToDirect":
            if not (0 <= source_index < len(indices)):
                return None
            source_index = indices[source_index]
        elif reference != "Direct":
            raise ValueError(f"Unsupported {layer_name} reference {reference!r} in {geometry_name(geometry)!r}")
        if not (0 <= source_index < len(direct)):
            return None
        return direct[source_index]

    return sample, f"{mapping}/{reference}"


def geometry_material_slots(geometry, by_id: dict[int, object], relations: list[tuple[int, int]]) -> list[str]:
    geometry_id = int(geometry.props[0])
    model_ids = [parent for child, parent in relations if child == geometry_id]
    material_ids = [
        child
        for child, parent in relations
        if parent in model_ids
        and child in by_id
        and getattr(by_id[child], "name", "") == "Material"
    ]
    return [
        clean_fbx_object_name(by_id[material_id].props[1])
        for material_id in material_ids
        if len(by_id[material_id].props) > 1
    ]


def polygon_material_indices(geometry, polygon_count: int, polygon_vertex_count: int) -> list[int]:
    layers = geometry.children_named("LayerElementMaterial")
    if not layers:
        return [0] * polygon_count
    layer = layers[0]
    values_node = layer.child("Materials")
    values = [int(value) for value in values_node.props[0]] if values_node and values_node.props else []
    mapping_node = layer.child("MappingInformationType")
    mapping = str(mapping_node.props[0]) if mapping_node else "AllSame"
    if not values:
        return [0] * polygon_count
    if mapping == "AllSame":
        return [values[0]] * polygon_count
    if mapping == "ByPolygon":
        return [values[index] if index < len(values) else values[-1] for index in range(polygon_count)]
    if mapping == "ByPolygonVertex":
        # A polygon cannot use more than one native draw material. Use its first
        # corner, matching Blender's material-index behavior.
        result = []
        cursor = 0
        polygons = polygon_corners(geometry)
        for polygon in polygons:
            result.append(values[cursor] if cursor < len(values) else values[-1])
            cursor += len(polygon)
        return result
    raise ValueError(
        f"Unsupported LayerElementMaterial mapping {mapping!r} in {geometry_name(geometry)!r}"
    )


def extract_custom_triangles(roots) -> tuple[list[list[dict]], dict]:
    geometries = [
        geometry
        for geometry in object_nodes(roots, "Geometry")
        if len(geometry.props) < 3 or str(geometry.props[2]).lower() == "mesh"
    ]
    if not geometries:
        raise ValueError("No FBX mesh geometry was found")

    by_id, relations = fbx_relations(roots)
    mesh_groups: list[list[dict]] = []
    report = collections.Counter()
    report["mesh_objects"] = len(geometries)
    for geometry in geometries:
        transform = geometry_transform(geometry, by_id, relations)
        report["transformed_mesh_objects"] += int(transform["applied"])
        vertices_node = geometry.child("Vertices")
        if not vertices_node:
            raise ValueError(f"FBX mesh {geometry_name(geometry)!r} has no vertices")
        flat = vertices_node.props[0]
        vertices = [
            transform_position(tuple(map(float, flat[i : i + 3])), transform)
            for i in range(0, len(flat), 3)
        ]
        polygons = polygon_corners(geometry)
        polygon_vertex_count = sum(len(polygon) for polygon in polygons)
        material_slots = geometry_material_slots(geometry, by_id, relations)
        material_indices = polygon_material_indices(geometry, len(polygons), polygon_vertex_count)
        report["fbx_material_slots"] = max(report["fbx_material_slots"], len(material_slots))
        uv_sample, _uv_mode = layer_samples(
            geometry,
            "LayerElementUV",
            "UV",
            2,
            ("UVIndex", "TextureUVIndex"),
            len(vertices),
            len(polygons),
            polygon_vertex_count,
        )
        normal_sample, _normal_mode = layer_samples(
            geometry,
            "LayerElementNormal",
            "Normals",
            3,
            ("NormalsIndex", "NormalIndex"),
            len(vertices),
            len(polygons),
            polygon_vertex_count,
        )
        weights, weight_report = geometry_weights_by_cp(roots, geometry, len(vertices))
        report.update({f"fbx_{key}": value for key, value in weight_report.items()})
        triangles: list[dict] = []
        for polygon_index, polygon in enumerate(polygons):
            if len(polygon) < 3:
                raise ValueError(f"Mesh {geometry_name(geometry)!r} contains a polygon with fewer than three vertices")
            report["source_polygons"] += 1
            report["triangulated_polygons"] += max(0, len(polygon) - 3)
            for fan_index in range(1, len(polygon) - 1):
                source_corners = (polygon[0], polygon[fan_index], polygon[fan_index + 1])
                if len({control_point for control_point, _pv in source_corners}) != 3:
                    raise ValueError(
                        f"Mesh {geometry_name(geometry)!r} polygon {polygon_index} contains a degenerate triangle"
                    )
                positions = []
                corners = []
                for control_point, polygon_vertex in source_corners:
                    if not (0 <= control_point < len(vertices)):
                        raise ValueError(
                            f"Mesh {geometry_name(geometry)!r} polygon {polygon_index} references vertex {control_point}, "
                            f"but the mesh has {len(vertices)} vertices"
                        )
                    position = vertices[control_point]
                    positions.append(position)
                    uv = uv_sample(control_point, polygon_index, polygon_vertex) if uv_sample else None
                    normal = normal_sample(control_point, polygon_index, polygon_vertex) if normal_sample else None
                    if normal is not None:
                        normal = transform_normal(tuple(map(float, normal[:3])), transform)
                    if uv is None:
                        uv = (0.0, 0.0)
                        report["corners_default_uv"] += 1
                    if normal is None:
                        report["corners_generated_normal"] += 1
                    corners.append(
                        {
                            "position": position,
                            "uv": tuple(map(float, uv[:2])),
                            "normal": tuple(map(float, normal[:3])) if normal is not None else None,
                            "weights": weights[control_point],
                        }
                    )
                fallback_normal = vector_normal(*positions)
                for corner in corners:
                    if corner["normal"] is None:
                        corner["normal"] = fallback_normal
                material_index = material_indices[polygon_index] if polygon_index < len(material_indices) else 0
                material_name = (
                    material_slots[material_index]
                    if 0 <= material_index < len(material_slots)
                    else f"Material_{material_index}"
                )
                triangles.append({
                    "mesh": geometry_name(geometry),
                    "material_index": int(material_index),
                    "material_name": material_name,
                    "corners": corners,
                })
                report[f"material_{material_index}_triangles"] += 1
                report["triangles"] += 1
        if triangles:
            mesh_groups.append(triangles)
    if not mesh_groups:
        raise ValueError("The FBX contains no drawable triangles")
    return mesh_groups, dict(report)


def cyclic_triangle_matches(actual: tuple[int, int, int], expected: tuple[int, int, int]) -> bool:
    return actual in (
        expected,
        (expected[1], expected[2], expected[0]),
        (expected[2], expected[0], expected[1]),
    )


def stream_corner_matches(left: dict, right: dict) -> bool:
    for key in ("position", "uv", "normal"):
        if any(abs(float(a) - float(b)) > 1e-6 for a, b in zip(left[key], right[key])):
            return False
    left_weights = left["weights"]
    right_weights = right["weights"]
    if set(left_weights) != set(right_weights):
        return False
    return all(abs(float(left_weights[name]) - float(right_weights[name])) <= 1e-6 for name in left_weights)


def extract_cmp_stream_groups(roots) -> list[dict]:
    """Recover the original draw-record order when an FBX still exposes one vertex per CMP record."""
    geometries = [
        geometry
        for geometry in object_nodes(roots, "Geometry")
        if len(geometry.props) < 3 or str(geometry.props[2]).lower() == "mesh"
    ]
    by_id, relations = fbx_relations(roots)
    groups: list[dict] = []
    for geometry in geometries:
        transform = geometry_transform(geometry, by_id, relations)
        vertices_node = geometry.child("Vertices")
        if not vertices_node:
            return []
        flat = vertices_node.props[0]
        vertices = [
            transform_position(tuple(map(float, flat[i : i + 3])), transform)
            for i in range(0, len(flat), 3)
        ]
        polygons = polygon_corners(geometry)
        polygon_vertex_count = sum(len(polygon) for polygon in polygons)
        uv_sample, _uv_mode = layer_samples(
            geometry,
            "LayerElementUV",
            "UV",
            2,
            ("UVIndex", "TextureUVIndex"),
            len(vertices),
            len(polygons),
            polygon_vertex_count,
        )
        normal_sample, _normal_mode = layer_samples(
            geometry,
            "LayerElementNormal",
            "Normals",
            3,
            ("NormalsIndex", "NormalIndex"),
            len(vertices),
            len(polygons),
            polygon_vertex_count,
        )
        weights, _weight_report = geometry_weights_by_cp(roots, geometry, len(vertices))
        records: list[dict | None] = [None] * len(vertices)
        markers = [0] * len(vertices)
        for polygon_index, polygon in enumerate(polygons):
            if len(polygon) != 3:
                return []
            control_points = tuple(int(corner[0]) for corner in polygon)
            if len(set(control_points)) != 3:
                return []
            record_index = max(control_points)
            if record_index < 2 or set(control_points) != {record_index - 2, record_index - 1, record_index}:
                return []
            expected = (
                (record_index - 2, record_index - 1, record_index)
                if (record_index - 2) % 2 == 0
                else (record_index - 1, record_index - 2, record_index)
            )
            if not cyclic_triangle_matches(control_points, expected) or markers[record_index]:
                return []
            positions = [vertices[control_point] for control_point in control_points]
            fallback_normal = vector_normal(*positions)
            for control_point, polygon_vertex in polygon:
                if not (0 <= control_point < len(vertices)):
                    return []
                uv = uv_sample(control_point, polygon_index, polygon_vertex) if uv_sample else (0.0, 0.0)
                normal = normal_sample(control_point, polygon_index, polygon_vertex) if normal_sample else None
                if uv is None:
                    uv = (0.0, 0.0)
                if normal is None:
                    normal = fallback_normal
                else:
                    normal = transform_normal(tuple(map(float, normal[:3])), transform)
                corner = {
                    "position": vertices[control_point],
                    "uv": tuple(map(float, uv[:2])),
                    "normal": tuple(map(float, normal[:3])),
                    "weights": weights[control_point],
                }
                if records[control_point] is not None and not stream_corner_matches(records[control_point], corner):
                    return []
                records[control_point] = corner
            markers[record_index] = 0x7F
        if not records:
            return []
        # A bridge-exported CMP keeps one FBX control point per draw record, but
        # polygon-corner UV/normal layers cannot describe records that never
        # participate in a drawn triangle. Those are valid no-draw records, not
        # evidence that the native stream layout was lost.
        for index, record in enumerate(records):
            if record is None:
                records[index] = {
                    "position": vertices[index],
                    "uv": (0.0, 0.0),
                    "normal": (0.0, 0.0, 1.0),
                    "weights": weights[index],
                }
        groups.append(
            {
                "mesh": geometry_name(geometry),
                "records": [
                    {**record, "marker": markers[index]}
                    for index, record in enumerate(records)
                    if record is not None
                ],
                "triangles": len(polygons),
            }
        )
    return groups


def bone_depths(bones: list[dict]) -> dict[int, int]:
    by_index = {int(bone["idx"]): bone for bone in bones}
    depths: dict[int, int] = {}

    def depth(index: int, active: set[int] | None = None) -> int:
        if index in depths:
            return depths[index]
        active = set() if active is None else active
        if index in active:
            return 10**6
        active.add(index)
        bone = by_index[index]
        parent = int(bone.get("parent", -1))
        value = 0 if parent < 0 or parent not in by_index else depth(parent, active) + 1
        depths[index] = value
        return value

    for index in by_index:
        depth(index)
    return depths


def root_skin_bone(bones: list[dict], skin_palette: dict[int, int]) -> int:
    depths = bone_depths(bones)
    candidates = sorted(set(skin_palette.values()), key=lambda index: (depths.get(index, 10**6), index))
    if not candidates:
        raise ValueError("The template CMP has no skin-palette bone that can receive unweighted vertices")
    return int(candidates[0])


def distribute_triangles(mesh_groups: list[list[dict]], packet_count: int) -> list[list[dict]]:
    if len(mesh_groups) == packet_count:
        return mesh_groups
    triangles = [triangle for group in mesh_groups for triangle in group]
    if len(triangles) < packet_count:
        raise ValueError(
            f"The replacement has {len(triangles)} triangles, but the template requires {packet_count} non-empty packets"
        )
    buckets: list[list[dict]] = []
    start = 0
    for packet_index in range(packet_count):
        remaining_triangles = len(triangles) - start
        remaining_packets = packet_count - packet_index
        count = (remaining_triangles + remaining_packets - 1) // remaining_packets
        buckets.append(triangles[start : start + count])
        start += count
    return buckets


def distribute_stream_groups(stream_groups: list[dict], packet_count: int) -> list[list[dict]] | None:
    if not stream_groups or len(stream_groups) < packet_count:
        return None
    if packet_count == 1:
        return [stream_groups]
    if len(stream_groups) == packet_count:
        return [[group] for group in stream_groups]

    buckets: list[list[dict]] = []
    start = 0
    for packet_index in range(packet_count):
        remaining_groups = len(stream_groups) - start
        remaining_packets = packet_count - packet_index
        count = (remaining_groups + remaining_packets - 1) // remaining_packets
        buckets.append(stream_groups[start : start + count])
        start += count
    return buckets


def resolve_corner_weights(
    corner: dict,
    bone_name_to_index: dict[str, int],
    palette_by_bone: dict[int, list[int]],
    root_bone: int,
    bind_unweighted_root: bool,
    location: str,
) -> tuple[list[tuple[int, float]], bool]:
    source_weights = dict(corner["weights"])
    if not source_weights:
        if not bind_unweighted_root:
            raise ValueError(f"{location} is unweighted; enable root binding or weight-paint the mesh")
        return [(root_bone, 1.0)], True
    resolved_sources = []
    for name, weight in source_weights.items():
        candidates = cmp_bone_name_candidates(str(name))
        bone_index = next(
            (
                bone_name_to_index[candidate]
                for candidate in candidates
                if candidate in bone_name_to_index
            ),
            None,
        )
        if bone_index is None:
            bone_index = next(
                (
                    bone_name_to_index[candidate.lower()]
                    for candidate in candidates
                    if candidate.lower() in bone_name_to_index
                ),
                None,
            )
        resolved_sources.append((str(name), float(weight), bone_index))

    has_painted_weight = any(
        bone_index is not None and bone_index != root_bone and weight > 1e-8
        for _name, weight, bone_index in resolved_sources
    )
    if has_painted_weight:
        # A root-bound custom-model starter carries root/pelvis=1.0 on every
        # vertex. Once the user paints a real bone, that placeholder must stop
        # competing with the authored weight. Keep deliberately fractional
        # root weights, but remove the untouched full-weight fallback.
        for name, weight, bone_index in resolved_sources:
            if bone_index == root_bone and weight >= 1.0 - 1e-6:
                source_weights.pop(name, None)

    weights, unknown, reduced = normalized_cmp_weights(source_weights, bone_name_to_index, palette_by_bone)
    if unknown:
        raise ValueError(f"{location} uses unknown or unavailable bones: {', '.join(unknown)}")
    if reduced:
        raise ValueError(f"{location} uses more than the CMP limit of two bone influences")
    if not weights:
        if not bind_unweighted_root:
            raise ValueError(f"{location} has no usable CMP bone weights")
        return [(root_bone, 1.0)], True
    return weights, False


def build_packet(
    triangles: list[dict],
    scale: float,
    skin_palette: dict[int, int],
    palette_by_bone: dict[int, list[int]],
    bone_name_to_index: dict[str, int],
    root_bone: int,
    bind_unweighted_root: bool,
) -> tuple[bytes, dict]:
    position_records = bytearray()
    side_records = bytearray()
    uv_records = bytearray()
    root_bound = 0
    for triangle_index, triangle in enumerate(triangles):
        base_record = triangle_index * 3
        corners = list(triangle["corners"])
        if base_record & 1:
            corners[0], corners[1] = corners[1], corners[0]
        for corner_index, corner in enumerate(corners):
            location = f"mesh {triangle['mesh']!r}, triangle {triangle_index}, corner {corner_index}"
            weights, used_root = resolve_corner_weights(
                corner,
                bone_name_to_index,
                palette_by_bone,
                root_bone,
                bind_unweighted_root,
                location,
            )
            root_bound += int(used_root)
            control = encode_cmp_skin_control(weights, b"\x00\x00\x00\x00", skin_palette, palette_by_bone)
            if control is None:
                raise ValueError(f"Could not encode weights for {location}")
            x, y, z = corner["position"]
            position_records.extend(struct.pack("<fff", x / scale, y / scale, z / scale))
            position_records.extend(control)
            marker = 0x7F if corner_index == 2 else 0
            side_records.extend(pack_cmp_normal(corner["normal"]))
            side_records.append(marker)
            u, v = corner["uv"]
            uv_records.extend(struct.pack("<hh", pack_cmp_uv(u), pack_cmp_uv(1.0 - v)))

    packet = bytearray(position_records)
    packet.extend(side_records)
    packet.extend(b"\x00" * (align(len(packet), 16) - len(packet)))
    packet.extend(uv_records)
    packet.extend(b"\x00" * (align(len(packet), 64) - len(packet)))
    return bytes(packet), {
        "triangles": len(triangles),
        "records": len(triangles) * 3,
        "bytes": len(packet),
        "root_bound_records": root_bound,
        "encoding": "isolated_triangles",
    }


def build_stream_packet(
    groups: list[dict],
    scale: float,
    skin_palette: dict[int, int],
    palette_by_bone: dict[int, list[int]],
    bone_name_to_index: dict[str, int],
    root_bone: int,
    bind_unweighted_root: bool,
) -> tuple[bytes, dict]:
    records: list[dict] = []
    padding_records = 0
    for group in groups:
        group_records = group["records"]
        if records and len(records) & 1:
            records.append({**group_records[0], "marker": 0})
            padding_records += 1
        records.extend(group_records)

    position_records = bytearray()
    side_records = bytearray()
    uv_records = bytearray()
    root_bound = 0
    for record_index, corner in enumerate(records):
        location = f"CMP stream record {record_index}"
        weights, used_root = resolve_corner_weights(
            corner,
            bone_name_to_index,
            palette_by_bone,
            root_bone,
            bind_unweighted_root,
            location,
        )
        root_bound += int(used_root)
        control = encode_cmp_skin_control(weights, b"\x00\x00\x00\x00", skin_palette, palette_by_bone)
        if control is None:
            raise ValueError(f"Could not encode weights for {location}")
        x, y, z = corner["position"]
        position_records.extend(struct.pack("<fff", x / scale, y / scale, z / scale))
        position_records.extend(control)
        side_records.extend(pack_cmp_normal(corner["normal"]))
        side_records.append(int(corner["marker"]) & 0xFF)
        u, v = corner["uv"]
        uv_records.extend(struct.pack("<hh", pack_cmp_uv(u), pack_cmp_uv(1.0 - v)))

    packet = bytearray(position_records)
    packet.extend(side_records)
    packet.extend(b"\x00" * (align(len(packet), 16) - len(packet)))
    packet.extend(uv_records)
    packet.extend(b"\x00" * (align(len(packet), 64) - len(packet)))
    return bytes(packet), {
        "triangles": sum(int(group["triangles"]) for group in groups),
        "records": len(records),
        "bytes": len(packet),
        "root_bound_records": root_bound,
        "encoding": "preserved_cmp_stream",
        "source_mesh_groups": len(groups),
        "restart_padding_records": padding_records,
    }


def legacy_skin_controls_recognized(controls: list[bytes], skin_palette: dict[int, int]) -> bool:
    """Recognize the older weight word with its low-byte control flags intact."""
    if not controls or not skin_palette:
        return False
    for control in controls:
        blend, raw_bone_a, raw_bone_b = struct.unpack("<HBB", control)
        if blend > 4096 + 0xFF:
            return False
        if raw_bone_a not in skin_palette or raw_bone_b not in skin_palette:
            return False
    return True


def validate_template_packets(main: bytes, resource: bytes, packets: list[dict], skin_palette: dict[int, int]) -> None:
    unsupported: list[str] = []
    for index, packet in enumerate(packets):
        controls = [read_cmp_vertex_control(resource, packet["rel"] + record * 16) for record in range(packet["count"])]
        decoded = [decode_cmp_skin_control(control, skin_palette) for control in controls]
        if "compact_fmt" in packet:
            unsupported.append(f"packet{index}=compact format {packet['compact_fmt']:#x}")
        elif cmp_packet_uses_vertex_draw_flags(controls):
            unsupported.append(f"packet{index}=draw flags stored in vertex controls")
        elif (not decoded or any(weights is None for weights in decoded)) and not legacy_skin_controls_recognized(
            controls, skin_palette
        ):
            unsupported.append(f"packet{index}=unrecognized skin controls")
    if unsupported:
        raise ValueError(
            "Custom Model currently requires template packets with decoded per-vertex two-bone skin controls; "
            + "; ".join(unsupported)
        )


def rebuild_custom_model(
    fbx: Path,
    template: Path,
    out: Path,
    scale: float,
    bind_unweighted_root: bool,
) -> dict:
    if not fbx.exists():
        raise FileNotFoundError(f"Missing replacement FBX: {fbx}")
    if not template.exists():
        raise FileNotFoundError(f"Missing template CMP: {template}")
    if template.resolve() == out.resolve():
        raise ValueError("Custom Model output cannot overwrite the template CMP")
    if scale == 0:
        raise ValueError("Scale cannot be zero")

    parser = PipeworksParser(str(template))
    entries = parser.parse()
    source_data = bytes(parser.file_data or b"")
    main_entry = next((entry for entry in entries if entry["file_type"] == 17 and not entry["is_resource"]), None)
    resource_entry = next((entry for entry in entries if entry["file_type"] == 17 and entry["is_resource"]), None)
    if not main_entry or not resource_entry:
        raise ValueError("The template has no CMP Type-17 mesh/resource pair")
    main = source_data[main_entry["offset"] : main_entry["offset"] + main_entry["size"]]
    old_resource = source_data[resource_entry["offset"] : resource_entry["offset"] + resource_entry["size"]]
    packets = find_cmp_packets(main, old_resource)
    if not packets:
        raise ValueError("The template CMP has no decoded mesh packets")
    materials = parse_cmp_materials(source_data, entries)
    material_ranges = parse_cmp_material_ranges(main, packets, materials)
    if len(material_ranges) != len(packets):
        raise ValueError(
            f"The template exposes {len(packets)} packets but {len(material_ranges)} material draw records"
        )
    bones, _globals = parse_cmp_skeleton(parser, entries, 1.0)
    skin_palette = parse_cmp_skin_palette(main, bundle_strings(parser), bones)
    if not bones or not skin_palette:
        raise ValueError("The template CMP has no decoded skeleton skin palette")
    validate_template_packets(main, old_resource, packets, skin_palette)

    roots, fbx_version = parse_fbx(fbx)
    mesh_groups, fbx_report = extract_custom_triangles(roots)
    source_stream_groups = extract_cmp_stream_groups(roots)
    packet_stream_groups = distribute_stream_groups(source_stream_groups, len(packets))
    packet_triangles = None if packet_stream_groups is not None else distribute_triangles(mesh_groups, len(packets))
    palette_by_bone: dict[int, list[int]] = collections.defaultdict(list)
    for raw_id, bone_index in skin_palette.items():
        palette_by_bone[int(bone_index)].append(int(raw_id))
    bone_name_to_index = cmp_bone_name_map(bones)
    root_bone = root_skin_bone(bones, skin_palette)
    root_name = next(str(bone["name"]) for bone in bones if int(bone["idx"]) == root_bone)

    packet_bytes: list[bytes] = []
    packet_reports: list[dict] = []
    for index in range(len(packets)):
        if packet_stream_groups is not None:
            built, report = build_stream_packet(
                packet_stream_groups[index],
                scale,
                skin_palette,
                palette_by_bone,
                bone_name_to_index,
                root_bone,
                bind_unweighted_root,
            )
        else:
            built, report = build_packet(
                packet_triangles[index],
                scale,
                skin_palette,
                palette_by_bone,
                bone_name_to_index,
                root_bone,
                bind_unweighted_root,
            )
        report["packet"] = index
        packet_bytes.append(built)
        packet_reports.append(report)

    used_resource = b"".join(packet_bytes)
    new_resource = used_resource
    if len(new_resource) < len(old_resource):
        new_resource += b"\x00" * (len(old_resource) - len(new_resource))
    data = bytearray(source_data)
    old_size = len(old_resource)
    data[resource_entry["offset"] : resource_entry["offset"] + old_size] = new_resource
    patch_bundle_resource_resize(data, parser, entries, resource_entry, old_size, len(new_resource))

    relative_offset = 0
    for index, (packet, built, report) in enumerate(zip(packets, packet_bytes, packet_reports)):
        descriptor = main_entry["offset"] + int(packet["desc"])
        struct.pack_into("<I", data, descriptor, int(report["records"]))
        struct.pack_into("<I", data, descriptor + 0x10, relative_offset)
        struct.pack_into("<I", data, descriptor + 0x14, len(built))
        draw_offset = main_entry["offset"] + int(material_ranges[index]["draw_offset"])
        struct.pack_into("<I", data, draw_offset, max(0, int(report["records"]) - 2))
        report["relative_offset"] = relative_offset
        relative_offset += len(built)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)

    verify_parser = PipeworksParser(str(out))
    verify_entries = verify_parser.parse()
    verify_data = bytes(verify_parser.file_data or b"")
    verify_main_entry = next(entry for entry in verify_entries if entry["file_type"] == 17 and not entry["is_resource"])
    verify_resource_entry = next(entry for entry in verify_entries if entry["file_type"] == 17 and entry["is_resource"])
    verify_main = verify_data[
        verify_main_entry["offset"] : verify_main_entry["offset"] + verify_main_entry["size"]
    ]
    verify_resource = verify_data[
        verify_resource_entry["offset"] : verify_resource_entry["offset"] + verify_resource_entry["size"]
    ]
    verify_packets = find_cmp_packets(verify_main, verify_resource)
    if len(verify_packets) != len(packets):
        raise ValueError(
            f"Generated CMP verification found {len(verify_packets)} packets instead of {len(packets)}"
        )
    for index, (packet, expected) in enumerate(zip(verify_packets, packet_reports)):
        if packet["count"] != expected["records"]:
            raise ValueError(
                f"Generated packet{index} has {packet['count']} records instead of {expected['records']}"
            )
        side = cmp_packet_side_stream(packet)
        markers = [read_cmp_side_record(verify_resource, side, record)[3] for record in range(packet["count"])]
        faces, _uvs, _kept = build_adc_strip_faces(
            [(0.0, 0.0, 0.0)] * packet["count"],
            markers,
        )
        if len(faces) != expected["triangles"]:
            raise ValueError(
                f"Generated packet{index} decodes to {len(faces)} triangles instead of {expected['triangles']}"
            )

    return {
        "status": "custom_model_built",
        "template": str(template),
        "fbx": str(fbx),
        "output": str(out),
        "fbx_version": fbx_version,
        "packet_count": len(packet_reports),
        "triangles": sum(report["triangles"] for report in packet_reports),
        "records": sum(report["records"] for report in packet_reports),
        "old_resource_size": old_size,
        "new_resource_size": len(new_resource),
        "used_resource_size": len(used_resource),
        "resource_delta": len(new_resource) - old_size,
        "bind_unweighted_root": bind_unweighted_root,
        "root_bone": root_name,
        "fbx_report": fbx_report,
        "source_cmp_streams_preserved": packet_stream_groups is not None,
        "packets": packet_reports,
        "verification": "packet and triangle counts reparsed successfully",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build entirely new FBX geometry into a template PS2 CMP")
    parser.add_argument("fbx", type=Path, help="Replacement Blender binary FBX")
    parser.add_argument("template", type=Path, help="Template CMP providing skeleton, materials, and metadata")
    parser.add_argument("--out", required=True, type=Path, help="Output CMP copy")
    parser.add_argument("--scale", type=float, default=10.0)
    parser.add_argument(
        "--bind-unweighted-root",
        action="store_true",
        help="Rigidly bind unweighted FBX vertices to the shallowest template skin bone",
    )
    args = parser.parse_args()
    report = rebuild_custom_model(
        args.fbx.resolve(),
        args.template.resolve(),
        args.out.resolve(),
        args.scale,
        args.bind_unweighted_root,
    )
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
