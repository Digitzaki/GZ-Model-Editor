from __future__ import annotations

import argparse
import json
import math
import re
import struct
import sys
from pathlib import Path

TOOL_DIR = Path(__file__).resolve().parent
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from bdg_animation_import import replace_bundle_main_entries
from bdg_to_fbx_extract_all import find_skeleton
from fbx_to_bdg_import import (
    clean_fbx_object_name,
    euler_xyz_degrees_to_quat,
    find_first,
    object_nodes,
    p_values,
    parse_fbx,
)
from parser_core import PipeworksParser


def be32(data: bytes, offset: int) -> int:
    return struct.unpack_from(">I", data, offset)[0]


def align(value: int, boundary: int = 0x10) -> int:
    return (value + boundary - 1) // boundary * boundary


def normalized_quaternion(value) -> tuple[float, float, float, float]:
    length = math.sqrt(sum(float(component) ** 2 for component in value)) or 1.0
    return tuple(float(component) / length for component in value)


def normalized_name(value: str) -> str:
    value = clean_fbx_object_name(str(value)).strip()
    value = re.sub(r"(?:_?Model)(?:\.\d+)?$", "", value, flags=re.IGNORECASE).strip()
    return re.sub(r"[ _]+", " ", value).strip().lower()


def model_name(model) -> str:
    return re.sub(
        r"(?:_?Model)(?:\.\d+)?$",
        "",
        clean_fbx_object_name(str(model.props[1])).strip(),
        flags=re.IGNORECASE,
    ).strip()


def native_index_for_name(name: str, native_names: dict[str, int]) -> int | None:
    clean = normalized_name(name)
    if clean in native_names:
        return native_names[clean]
    without_index = re.sub(r"^\d{3}", "", clean).strip()
    return native_names.get(without_index)


def is_generated_leaf_end(name: str, native_names: dict[str, int]) -> bool:
    match = re.fullmatch(r"(.+)_end(?:\.\d+)?", name, re.IGNORECASE)
    return bool(match and native_index_for_name(match.group(1), native_names) is not None)


def fbx_object_relations(roots) -> list[tuple[int, int]]:
    connections = find_first(roots, "Connections")
    relations = []
    if connections:
        relations = [
            (int(connection.props[1]), int(connection.props[2]))
            for connection in connections.children_named("C")
            if len(connection.props) >= 3 and str(connection.props[0]) == "OO"
        ]
    return relations


def detect_new_leaf_bones(roots, bones: list[dict], scale: float) -> list[dict]:
    limb_models = [
        model
        for model in object_nodes(roots, "Model")
        if len(model.props) >= 3 and str(model.props[2]) == "LimbNode"
    ]
    native_names = {normalized_name(str(bone["name"])): int(bone["idx"]) for bone in bones}
    relations = fbx_object_relations(roots)
    model_ids = {id(model): int(model.props[0]) for model in limb_models}
    native_model_indices = {
        model_ids[id(model)]: native_index_for_name(model_name(model), native_names)
        for model in limb_models
    }
    extra_models = [
        model
        for model in limb_models
        if native_model_indices[model_ids[id(model)]] is None
        and not is_generated_leaf_end(model_name(model), native_names)
    ]
    if not extra_models:
        return []

    present_native = {index for index in native_model_indices.values() if index is not None}
    missing_native = [str(bone["name"]) for bone in bones if int(bone["idx"]) not in present_native]
    if missing_native:
        raise ValueError(
            "adding and deleting bones in the same BDG import is not supported; "
            f"missing native bones: {', '.join(missing_native[:8])}"
        )

    next_index = max(int(bone["idx"]) for bone in bones) + 1
    if next_index + len(extra_models) > 256:
        raise ValueError("BDG Type 4 animation bone IDs are limited to 0..255")

    pending = {model_ids[id(model)]: model for model in extra_models}
    index_by_model = {
        model_id: index
        for model_id, index in native_model_indices.items()
        if index is not None
    }
    result = []
    safe_scale = scale if abs(scale) > 1.0e-8 else 1.0
    while pending:
        progressed = False
        for model_id, model in list(pending.items()):
            parent_ids = [parent for child, parent in relations if child == model_id]
            parent_model_id = next((parent for parent in parent_ids if parent in index_by_model), None)
            if parent_model_id is None:
                continue
            props = model.child("Properties70")
            translation = p_values(props, "Lcl Translation") or (0.0, 0.0, 0.0)
            rotation = p_values(props, "Lcl Rotation") or (0.0, 0.0, 0.0)
            pre_rotation = p_values(props, "PreRotation") or (0.0, 0.0, 0.0)
            post_rotation = p_values(props, "PostRotation") or (0.0, 0.0, 0.0)
            if any(abs(float(value)) > 1.0e-5 for value in (*pre_rotation[:3], *post_rotation[:3])):
                raise ValueError(
                    f"new bone {model_name(model)!r} uses FBX pre/post rotation, which is not safe to import"
                )
            index = next_index + len(result)
            result.append(
                {
                    "idx": index,
                    "parent": index_by_model[parent_model_id],
                    "name": model_name(model),
                    "t": tuple(float(value) / safe_scale for value in translation[:3]),
                    "q": normalized_quaternion(euler_xyz_degrees_to_quat(*rotation[:3])),
                }
            )
            index_by_model[model_id] = index
            del pending[model_id]
            progressed = True
        if not progressed:
            names = ", ".join(model_name(model) for model in pending.values())
            raise ValueError(f"new bones must descend from the native BDG skeleton: {names}")

    parent_ids = {int(bone["parent"]) for bone in result}
    non_leaf = [str(bone["name"]) for bone in result if int(bone["idx"]) in parent_ids]
    if non_leaf:
        raise ValueError(
            "the initial BDG add-bone path supports new leaf bones only; nested new bones: "
            + ", ".join(non_leaf)
        )
    return result


def append_bundle_strings(
    data: bytes,
    string_offset: int,
    metadata_offset: int,
    names: list[str],
) -> tuple[bytes, list[int]]:
    if not names:
        return data, []
    count = struct.unpack_from("<I", data, string_offset)[0]
    offsets = [struct.unpack_from("<I", data, string_offset + 4 + index * 4)[0] for index in range(count)]
    if not offsets:
        raise ValueError("BDG string table is empty")
    first_relative = min(offsets)
    used_end = first_relative
    for relative in offsets:
        absolute = string_offset + relative
        end = data.find(b"\0", absolute, metadata_offset)
        if end < 0:
            raise ValueError("BDG string table contains an unterminated string")
        used_end = max(used_end, end + 1 - string_offset)

    encoded = []
    for name in names:
        try:
            encoded.append(name.encode("ascii") + b"\0")
        except UnicodeEncodeError as exc:
            raise ValueError(f"BDG bone names must be ASCII: {name!r}") from exc

    table_growth = 4 * len(names)
    old_payload = bytes(data[string_offset + first_relative : string_offset + used_end])
    new_strings_start = used_end + table_growth
    required_end = new_strings_start + sum(len(value) for value in encoded)
    old_capacity = metadata_offset - string_offset
    new_capacity = align(max(old_capacity, required_end), 0x20)

    table = bytearray(new_capacity)
    struct.pack_into("<I", table, 0, count + len(names))
    for index, relative in enumerate(offsets):
        struct.pack_into("<I", table, 4 + index * 4, relative + table_growth)
    table[first_relative + table_growth : used_end + table_growth] = old_payload

    new_ids = []
    cursor = new_strings_start
    for add_index, value in enumerate(encoded):
        new_ids.append(count + add_index)
        struct.pack_into("<I", table, 4 + (count + add_index) * 4, cursor)
        table[cursor : cursor + len(value)] = value
        cursor += len(value)

    output = bytearray(data[:string_offset])
    output.extend(table)
    output.extend(data[metadata_offset:])
    delta = new_capacity - old_capacity
    if delta:
        for header_offset in (0x64, 0x68, 0x70):
            struct.pack_into(">I", output, header_offset, be32(data, header_offset) + delta)
    return bytes(output), new_ids


def parse_type3_nodes(blob: bytes) -> tuple[dict[int, dict], dict[int, int]]:
    count = be32(blob, 0x20)
    root = be32(blob, 0x1C)
    nodes: dict[int, dict] = {}
    index_by_offset: dict[int, int] = {}

    def walk(relative: int) -> None:
        if relative in index_by_offset:
            return
        if relative < root or relative + 0x30 > len(blob):
            raise ValueError(f"invalid BDG Type 3 hierarchy node offset {relative:#x}")
        index, parent, child_count, name_index = struct.unpack_from(">4i", blob, relative)
        if index < 0 or index >= count or child_count < 0 or child_count > 128:
            raise ValueError(f"invalid BDG Type 3 hierarchy node at {relative:#x}")
        children = [be32(blob, relative + 0x30 + child * 4) for child in range(child_count)]
        index_by_offset[relative] = index
        nodes[index] = {
            "idx": index,
            "parent": parent,
            "name_index": name_index,
            "children": children,
            "record": bytes(blob[relative : relative + 0x30]),
        }
        for child in children:
            walk(child)

    walk(root)
    if len(nodes) != count:
        raise ValueError(f"BDG Type 3 hierarchy exposes {len(nodes)} of {count} bones")
    for node in nodes.values():
        node["children"] = [index_by_offset[relative] for relative in node["children"]]
    return nodes, index_by_offset


def quaternion_matrix(q) -> list[list[float]]:
    x, y, z, w = normalized_quaternion(q)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return [
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy), 0.0],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx), 0.0],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy), 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def matrix_multiply(left, right) -> list[list[float]]:
    return [
        [sum(left[row][axis] * right[axis][column] for axis in range(4)) for column in range(4)]
        for row in range(4)
    ]


def global_matrices(bones: list[dict]) -> dict[int, list[list[float]]]:
    by_index = {int(bone["idx"]): bone for bone in bones}
    result: dict[int, list[list[float]]] = {}

    def matrix_for(index: int) -> list[list[float]]:
        if index in result:
            return result[index]
        bone = by_index[index]
        local = quaternion_matrix(bone["q"])
        local[0][3], local[1][3], local[2][3] = (float(value) for value in bone["t"])
        parent = int(bone["parent"])
        result[index] = matrix_multiply(matrix_for(parent), local) if parent in by_index else local
        return result[index]

    for bone_index in by_index:
        matrix_for(bone_index)
    return result


def inverse_bind_record(global_matrix: list[list[float]]) -> bytes:
    rotation = [[global_matrix[row][column] for column in range(3)] for row in range(3)]
    translation = [global_matrix[row][3] for row in range(3)]
    inverse = [[rotation[column][row] for column in range(3)] for row in range(3)]
    inverse_translation = [
        -sum(inverse[row][column] * translation[column] for column in range(3))
        for row in range(3)
    ]
    matrix = [
        [inverse[0][0], inverse[0][1], inverse[0][2], inverse_translation[0]],
        [inverse[1][0], inverse[1][1], inverse[1][2], inverse_translation[1]],
        [inverse[2][0], inverse[2][1], inverse[2][2], inverse_translation[2]],
        [0.0, 0.0, 0.0, 0.0],
    ]
    column_major = [matrix[row][column] for column in range(4) for row in range(4)]
    return struct.pack(">16f", *column_major)


def rebuild_type3_skeleton(
    blob: bytes,
    old_bones: list[dict],
    new_bones: list[dict],
    name_indices: list[int],
) -> bytes:
    old_count = len(old_bones)
    nodes, index_by_offset = parse_type3_nodes(blob)
    root_start = be32(blob, 0x1C)
    table1_start = be32(blob, 0x2C)
    table2_start = be32(blob, 0x30)
    third_field = be32(blob, 0x38)
    old_table2_end = table2_start + old_count * 4
    third_start = third_field if third_field else old_table2_end
    pose_start = be32(blob, 0x3C)
    if not (root_start < table1_start <= table2_start <= third_start <= pose_start <= len(blob)):
        raise ValueError("unsupported BDG Type 3 skeleton section layout")

    for bone, name_index in zip(new_bones, name_indices):
        index = int(bone["idx"])
        parent = int(bone["parent"])
        if parent not in nodes:
            raise ValueError(f"new bone {bone['name']!r} has invalid parent {parent}")
        qx, qy, qz, qw = normalized_quaternion(bone["q"])
        tx, ty, tz = (float(value) for value in bone["t"])
        record = struct.pack(">4i4f3fI", index, parent, 0, int(name_index), qx, qy, qz, qw, tx, ty, tz, 0)
        nodes[index] = {
            "idx": index,
            "parent": parent,
            "name_index": int(name_index),
            "children": [],
            "record": record,
        }
        nodes[parent]["children"].append(index)

    order = []

    def flatten(index: int) -> None:
        order.append(index)
        for child in nodes[index]["children"]:
            flatten(child)

    root_index = index_by_offset[root_start]
    flatten(root_index)
    if len(order) != len(nodes):
        raise ValueError("rebuilt BDG Type 3 hierarchy is disconnected")

    offsets = {}
    cursor = root_start
    for index in order:
        offsets[index] = cursor
        cursor = align(cursor + 0x30 + 4 * len(nodes[index]["children"]))
    new_table1_start = cursor
    new_table2_start = new_table1_start + len(nodes) * 4
    new_third_start = new_table2_start + len(nodes) * 4
    third_bytes = blob[third_start:pose_start]
    new_pose_start = new_third_start + len(third_bytes)

    output = bytearray(blob[:root_start])
    for index in order:
        node = nodes[index]
        record = bytearray(node["record"])
        struct.pack_into(">4i", record, 0, index, int(node["parent"]), len(node["children"]), int(node["name_index"]))
        output.extend(record)
        output.extend(struct.pack(f">{len(node['children'])}I", *(offsets[child] for child in node["children"])))
        output.extend(b"\0" * (align(len(output)) - len(output)))

    output.extend(struct.pack(f">{len(nodes)}I", *(offsets[index] for index in range(len(nodes)))))
    old_table2 = [be32(blob, table2_start + index * 4) for index in range(old_count)]
    old_table2_indices = [index_by_offset[relative] for relative in old_table2]
    table2_indices = old_table2_indices + [int(bone["idx"]) for bone in new_bones]
    output.extend(struct.pack(f">{len(nodes)}I", *(offsets[index] for index in table2_indices)))
    output.extend(third_bytes)

    old_pose_end = pose_start + old_count * 0x40
    old_records = [blob[pose_start + index * 0x40 : pose_start + (index + 1) * 0x40] for index in range(old_count)]
    if any(len(record) != 0x40 for record in old_records):
        raise ValueError("truncated BDG Type 3 inverse-bind records")
    output.extend(b"".join(old_records))
    all_bones = [dict(bone) for bone in old_bones] + [dict(bone) for bone in new_bones]
    globals_by_index = global_matrices(all_bones)
    for bone in new_bones:
        output.extend(inverse_bind_record(globals_by_index[int(bone["idx"])]))
    output.extend(blob[old_pose_end:])

    struct.pack_into(">I", output, 0x20, len(nodes))
    struct.pack_into(">I", output, 0x28, len(output))
    struct.pack_into(">I", output, 0x2C, new_table1_start)
    struct.pack_into(">I", output, 0x30, new_table2_start)
    struct.pack_into(">I", output, 0x38, new_third_start if third_field else 0)
    struct.pack_into(">I", output, 0x3C, new_pose_start)
    return bytes(output)


def rebuild_type4_skeleton(blob: bytes, old_bones: list[dict], new_bones: list[dict]) -> bytes:
    old_count = len(old_bones)
    if be32(blob, 0x20) != 1 or be32(blob, 0x2C) != old_count:
        raise ValueError("unsupported BDG Type 4 rest-pose layout")
    table_start = be32(blob, 0x30)
    first_record = be32(blob, 0x34)
    old_offsets = [be32(blob, table_start + index * 4) for index in range(old_count)]
    if old_offsets != [first_record + index * 0x24 for index in range(old_count)]:
        raise ValueError("unsupported non-sequential BDG Type 4 rest-pose records")

    old_records = [bytes(blob[offset : offset + 0x24]) for offset in old_offsets]
    old_payloads = []
    for offset in old_offsets:
        translation = offset + be32(blob, offset + 0x1C)
        rotation = offset + be32(blob, offset + 0x20)
        if rotation != translation + 0x20 or rotation + 0x30 > len(blob):
            raise ValueError("unsupported BDG Type 4 rest-pose payload")
        old_payloads.append(bytes(blob[translation : rotation + 0x30]))

    new_count = old_count + len(new_bones)
    new_first_record = table_start + new_count * 4
    new_payload_start = new_first_record + new_count * 0x24
    new_offsets = [new_first_record + index * 0x24 for index in range(new_count)]
    payload_offsets = [new_payload_start + index * 0x50 for index in range(new_count)]
    output = bytearray(blob[:table_start])
    output.extend(struct.pack(f">{new_count}I", *new_offsets))

    all_bones = [dict(bone) for bone in old_bones] + [dict(bone) for bone in new_bones]
    globals_by_index = global_matrices(all_bones)
    for index in range(new_count):
        if index < old_count:
            record = bytearray(old_records[index])
        else:
            bone = all_bones[index]
            record = bytearray(old_records[int(bone["parent"])])
            struct.pack_into(">i", record, 0x00, index)
            global_matrix = globals_by_index[index]
            distance = math.sqrt(sum(global_matrix[axis][3] ** 2 for axis in range(3)))
            struct.pack_into(">f", record, 0x04, distance)
        struct.pack_into(">I", record, 0x1C, payload_offsets[index] - new_offsets[index])
        struct.pack_into(">I", record, 0x20, payload_offsets[index] + 0x20 - new_offsets[index])
        output.extend(record)

    output.extend(b"".join(old_payloads))
    for bone in new_bones:
        payload = bytearray(0x50)
        struct.pack_into(">3f", payload, 0x00, *(float(value) for value in bone["t"]))
        struct.pack_into(">4f", payload, 0x20, *normalized_quaternion(bone["q"]))
        output.extend(payload)

    struct.pack_into(">I", output, 0x24, len(output))
    struct.pack_into(">I", output, 0x2C, new_count)
    struct.pack_into(">I", output, 0x34, new_first_record)
    return bytes(output)


def find_skeleton_entries(entries: list[dict]) -> tuple[dict, dict]:
    hierarchy = next(
        (
            entry
            for entry in entries
            if entry["file_type"] == 3
            and not entry["is_resource"]
            and "SKELETON" in str(entry["name"]).upper()
            and "CAMERA" not in str(entry["name"]).upper()
            and "INTRO_CAM" not in str(entry["name"]).upper()
        ),
        None,
    )
    pose = next(
        (
            entry
            for entry in entries
            if entry["file_type"] == 4
            and not entry["is_resource"]
            and "SKELETON" in str(entry["name"]).upper()
            and "CAMERA" not in str(entry["name"]).upper()
            and "INTRO_CAM" not in str(entry["name"]).upper()
        ),
        None,
    )
    if hierarchy is None or pose is None:
        raise ValueError("BDG does not contain a character Type 3/Type 4 skeleton pair")
    return hierarchy, pose


def read_bundle_strings(source: bytes, string_offset: int) -> list[str]:
    count = struct.unpack_from("<I", source, string_offset)[0]
    offsets = [struct.unpack_from("<I", source, string_offset + 4 + index * 4)[0] for index in range(count)]
    strings = []
    for relative in offsets:
        start = string_offset + relative
        end = source.find(b"\0", start)
        if start < string_offset or end < 0:
            raise ValueError("invalid BDG string table")
        strings.append(source[start:end].decode("latin1"))
    return strings


def bundle_bones(source: bytes) -> list[dict]:
    class MemoryParser(PipeworksParser):
        def parse(self):
            self.file_data = source
            self.is_big_endian = struct.unpack_from("<H", source, 0x2C)[0] == 0
            self.string_offset = self.read_long(0x34)

    parser = MemoryParser("<memory>")
    parser.parse()
    strings = read_bundle_strings(source, parser.string_offset)
    _base, _root, skeleton = find_skeleton(source, strings)
    return [dict(skeleton[index]) for index in range(len(skeleton))]


def rebuild_bundle(source_path: Path, new_bones: list[dict]) -> tuple[bytes, dict]:
    parser = PipeworksParser(str(source_path))
    entries = parser.parse()
    if not parser.is_big_endian:
        raise ValueError("bdg_add_bones.py supports big-endian Wii/GameCube BDG files only")
    source = parser.file_data or b""
    old_bones = bundle_bones(source)
    patched_strings, name_indices = append_bundle_strings(
        source,
        parser.string_offset,
        parser.metadata_offset,
        [str(bone["name"]) for bone in new_bones],
    )
    hierarchy_entry, pose_entry = find_skeleton_entries(entries)
    hierarchy_blob = source[
        int(hierarchy_entry["offset"]) : int(hierarchy_entry["offset"]) + int(hierarchy_entry["size"])
    ]
    pose_blob = source[int(pose_entry["offset"]) : int(pose_entry["offset"]) + int(pose_entry["size"])]
    replacements = {
        int(hierarchy_entry["file_num"]): rebuild_type3_skeleton(
            hierarchy_blob, old_bones, new_bones, name_indices
        ),
        int(pose_entry["file_num"]): rebuild_type4_skeleton(pose_blob, old_bones, new_bones),
    }
    output = replace_bundle_main_entries(patched_strings, replacements, alignment=0x10)
    return output, {
        "file": source_path.name,
        "old_bone_count": len(old_bones),
        "new_bone_count": len(old_bones) + len(new_bones),
        "type3_file_num": int(hierarchy_entry["file_num"]),
        "type4_file_num": int(pose_entry["file_num"]),
    }


def import_new_bones(
    shapes_path: Path,
    fbx_path: Path,
    character_path: Path | None,
    scale: float,
) -> tuple[bytes, bytes | None, dict]:
    shapes_source = shapes_path.read_bytes()
    old_bones = bundle_bones(shapes_source)
    roots, _version = parse_fbx(fbx_path)
    new_bones = detect_new_leaf_bones(roots, old_bones, scale)
    if not new_bones:
        return shapes_source, character_path.read_bytes() if character_path else None, {
            "status": "preserved_no_new_leaf_bones",
            "old_bone_count": len(old_bones),
            "new_bone_count": len(old_bones),
            "animation_policy": "byte-for-byte",
            "bones": [],
        }

    rebuilt_shapes, shapes_report = rebuild_bundle(shapes_path, new_bones)
    rebuilt_character = None
    character_report = None
    if character_path is not None:
        character_bones = bundle_bones(character_path.read_bytes())
        if len(character_bones) != len(old_bones):
            raise ValueError(
                f"Shapes/Character skeleton count mismatch: {len(old_bones)} != {len(character_bones)}"
            )
        for shape_bone, character_bone in zip(old_bones, character_bones):
            if (
                int(shape_bone["idx"]) != int(character_bone["idx"])
                or int(shape_bone["parent"]) != int(character_bone["parent"])
                or normalized_name(str(shape_bone["name"])) != normalized_name(str(character_bone["name"]))
            ):
                raise ValueError(
                    f"Shapes/Character skeleton mismatch at bone {shape_bone['idx']}: "
                    f"{shape_bone['name']!r} != {character_bone['name']!r}"
                )
        rebuilt_character, character_report = rebuild_bundle(character_path, new_bones)

    return rebuilt_shapes, rebuilt_character, {
        "status": "added_weightable_leaf_bones",
        "old_bone_count": len(old_bones),
        "new_bone_count": len(old_bones) + len(new_bones),
        "animation_policy": "all Type 4 animation clips preserved byte-for-byte; new leaf bones inherit parent motion",
        "bundles": [item for item in (shapes_report, character_report) if item is not None],
        "bones": [
            {
                "index": int(bone["idx"]),
                "name": str(bone["name"]),
                "parent_index": int(bone["parent"]),
                "local_translation": [round(float(value), 7) for value in bone["t"]],
                "local_rotation": [round(float(value), 7) for value in bone["q"]],
            }
            for bone in new_bones
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Append weightable leaf bones from a Blender FBX to Wii/GameCube Shapes and Character BDGs."
    )
    parser.add_argument("shapes")
    parser.add_argument("fbx")
    parser.add_argument("output_shapes")
    parser.add_argument("--character")
    parser.add_argument("--output-character")
    parser.add_argument("--report")
    parser.add_argument("--scale", type=float, default=10.0)
    args = parser.parse_args()

    character = Path(args.character) if args.character else None
    output_character = Path(args.output_character) if args.output_character else character
    if character is not None and output_character is None:
        raise ValueError("--output-character is required when --character is supplied")
    shapes, character_bytes, report = import_new_bones(
        Path(args.shapes), Path(args.fbx), character, float(args.scale)
    )
    output_shapes = Path(args.output_shapes)
    output_shapes.parent.mkdir(parents=True, exist_ok=True)
    output_shapes.write_bytes(shapes)
    if character_bytes is not None and output_character is not None:
        output_character.parent.mkdir(parents=True, exist_ok=True)
        output_character.write_bytes(character_bytes)
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
