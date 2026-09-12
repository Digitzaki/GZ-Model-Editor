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
from cmg_probe import global_matrices
from cmp_fbx_import import (
    cmp_bone_name_candidates,
    cmp_inverse_bind_values,
    find_cmp_bone_model,
    fbx_bone_models,
)
from cmp_probe import parse_cmp_skeleton
from fbx_to_bdg_import import (
    clean_fbx_object_name,
    euler_xyz_degrees_to_quat,
    find_first,
    object_nodes,
    p_values,
    parse_fbx,
)
from parser_core import PipeworksParser


def le32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<I", data, offset)[0]


def sle32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<i", data, offset)[0]


def normalized_quaternion(value) -> tuple[float, float, float, float]:
    length = math.sqrt(sum(float(component) ** 2 for component in value)) or 1.0
    return tuple(float(component) / length for component in value)


def model_name(model) -> str:
    raw = clean_fbx_object_name(str(model.props[1]))
    match = re.fullmatch(r"(.*?)(?:_?Model)(?:\.\d+)?", raw, re.IGNORECASE)
    return match.group(1).strip() if match and match.group(1).strip() else raw


def fbx_object_relations(roots) -> tuple[dict[int, object], list[tuple[int, int]]]:
    objects = {
        int(node.props[0]): node
        for node in object_nodes(roots)
        if node.props and isinstance(node.props[0], int)
    }
    connections = find_first(roots, "Connections")
    relations = []
    if connections:
        relations = [
            (int(connection.props[1]), int(connection.props[2]))
            for connection in connections.children_named("C")
            if len(connection.props) >= 3 and str(connection.props[0]) == "OO"
        ]
    return objects, relations


def native_index_for_name(name: str, native_names: dict[str, int]) -> int | None:
    for candidate in cmp_bone_name_candidates(name):
        if candidate in native_names:
            return native_names[candidate]
        if candidate.lower() in native_names:
            return native_names[candidate.lower()]
    return None


def is_generated_leaf_end(name: str, native_names: dict[str, int]) -> bool:
    match = re.fullmatch(r"(.+)_end(?:\.\d+)?", name, re.IGNORECASE)
    return bool(match and native_index_for_name(match.group(1), native_names) is not None)


def detect_new_leaf_bones(roots, bones: list[dict], scale: float) -> list[dict]:
    limb_models = [
        model
        for model in object_nodes(roots, "Model")
        if len(model.props) >= 3 and str(model.props[2]) == "LimbNode"
    ]
    native_names: dict[str, int] = {}
    for bone in bones:
        for candidate in cmp_bone_name_candidates(str(bone["name"])):
            native_names.setdefault(candidate, int(bone["idx"]))
            native_names.setdefault(candidate.lower(), int(bone["idx"]))

    objects, relations = fbx_object_relations(roots)
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

    models_by_name = fbx_bone_models(roots)
    missing_native = [
        str(bone["name"])
        for bone in bones
        if find_cmp_bone_model(models_by_name, str(bone["name"])) is None
    ]
    if missing_native:
        raise ValueError(
            "adding and deleting bones in the same CMP import is not supported; "
            f"missing native bones: {', '.join(missing_native[:8])}"
        )

    next_index = max(int(bone["idx"]) for bone in bones) + 1
    if next_index + len(extra_models) > 256:
        raise ValueError("CMP Type 4 animation bone IDs are limited to 0..255")

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
                    f"new bone {model_name(model)!r} uses FBX pre/post rotation, which is not yet safe to import"
                )
            index = next_index + len(result)
            bone = {
                "idx": index,
                "parent": index_by_model[parent_model_id],
                "name": model_name(model),
                "t": tuple(float(value) / safe_scale for value in translation[:3]),
                "q": normalized_quaternion(euler_xyz_degrees_to_quat(*rotation[:3])),
                "display_size": 0.3,
            }
            result.append(bone)
            index_by_model[model_id] = index
            del pending[model_id]
            progressed = True
        if not progressed:
            names = ", ".join(model_name(model) for model in pending.values())
            raise ValueError(f"new bones must descend from the native skeleton: {names}")

    parent_ids = {int(bone["parent"]) for bone in result}
    non_leaf = [str(bone["name"]) for bone in result if int(bone["idx"]) in parent_ids]
    if non_leaf:
        raise ValueError(
            "this first add-bone path only supports new leaf bones; nested new bones: "
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
    count = le32(data, string_offset)
    offsets = [le32(data, string_offset + 4 + index * 4) for index in range(count)]
    if not offsets:
        raise ValueError("CMP string table is empty")
    first_relative = min(offsets)
    used_end = first_relative
    for relative in offsets:
        absolute = string_offset + relative
        end = data.find(b"\0", absolute, metadata_offset)
        if end < 0:
            raise ValueError("CMP string table contains an unterminated string")
        used_end = max(used_end, end + 1 - string_offset)

    encoded = []
    for name in names:
        try:
            encoded.append(name.encode("ascii") + b"\0")
        except UnicodeEncodeError as exc:
            raise ValueError(f"CMP bone names must be ASCII: {name!r}") from exc

    table_growth = 4 * len(names)
    new_strings_start = used_end + table_growth
    required_end = new_strings_start + sum(len(value) for value in encoded)
    capacity = metadata_offset - string_offset
    if required_end > capacity:
        raise ValueError(
            "the fixed CMP string-table padding is too small for these names; "
            f"need {required_end - capacity} more bytes"
        )

    out = bytearray(data)
    old_payload = bytes(data[string_offset + first_relative : string_offset + used_end])
    out[string_offset + first_relative + table_growth : string_offset + used_end + table_growth] = old_payload
    struct.pack_into("<I", out, string_offset, count + len(names))
    for index, relative in enumerate(offsets):
        struct.pack_into("<I", out, string_offset + 4 + index * 4, relative + table_growth)

    new_ids = []
    cursor = new_strings_start
    for add_index, value in enumerate(encoded):
        new_ids.append(count + add_index)
        struct.pack_into("<I", out, string_offset + 4 + (count + add_index) * 4, cursor)
        out[string_offset + cursor : string_offset + cursor + len(value)] = value
        cursor += len(value)
    return bytes(out), new_ids


def parse_type3_nodes(blob: bytes) -> tuple[dict[int, dict], dict[int, int]]:
    count = le32(blob, 0x20)
    root = le32(blob, 0x1C)
    nodes: dict[int, dict] = {}
    index_by_offset: dict[int, int] = {}

    def walk(relative: int) -> None:
        if relative in index_by_offset:
            return
        if relative < root or relative + 0x10 > len(blob):
            raise ValueError(f"invalid Type 3 hierarchy node offset {relative:#x}")
        index, parent, child_count, name_index = struct.unpack_from("<4i", blob, relative)
        if index < 0 or index >= count or child_count < 0 or child_count > 128:
            raise ValueError(f"invalid Type 3 hierarchy node at {relative:#x}")
        children = [le32(blob, relative + 0x10 + child * 4) for child in range(child_count)]
        index_by_offset[relative] = index
        nodes[index] = {
            "idx": index,
            "parent": parent,
            "name_index": name_index,
            "children": children,
            "old_offset": relative,
        }
        for child in children:
            walk(child)

    walk(root)
    if len(nodes) != count:
        raise ValueError(f"Type 3 hierarchy exposes {len(nodes)} of {count} bones")
    for node in nodes.values():
        node["children"] = [index_by_offset[relative] for relative in node["children"]]
    return nodes, index_by_offset


def rebuild_type3_skeleton(
    blob: bytes,
    old_bones: list[dict],
    new_bones: list[dict],
    name_indices: list[int],
) -> bytes:
    old_count = len(old_bones)
    nodes, index_by_offset = parse_type3_nodes(blob)
    table1_start = le32(blob, 0x2C)
    table2_start = le32(blob, 0x30)
    third_start = le32(blob, 0x38)
    pose_start = le32(blob, 0x3C)
    root_start = le32(blob, 0x1C)
    if not (root_start < table1_start <= table2_start <= third_start <= pose_start <= len(blob)):
        raise ValueError("unsupported Type 3 skeleton section layout")

    for bone, name_index in zip(new_bones, name_indices):
        index = int(bone["idx"])
        parent = int(bone["parent"])
        if parent not in nodes:
            raise ValueError(f"new bone {bone['name']!r} has invalid parent {parent}")
        nodes[index] = {
            "idx": index,
            "parent": parent,
            "name_index": int(name_index),
            "children": [],
            "old_offset": None,
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
        raise ValueError("rebuilt Type 3 hierarchy is disconnected")

    offsets = {}
    cursor = root_start
    for index in order:
        offsets[index] = cursor
        cursor += 0x10 + 4 * len(nodes[index]["children"])
    new_table1_start = cursor
    new_table2_start = new_table1_start + len(nodes) * 4
    new_third_start = new_table2_start + len(nodes) * 4
    third_bytes = blob[third_start:pose_start]
    new_pose_start = new_third_start + len(third_bytes)

    output = bytearray(blob[:root_start])
    for index in order:
        node = nodes[index]
        output.extend(
            struct.pack(
                "<4i",
                index,
                int(node["parent"]),
                len(node["children"]),
                int(node["name_index"]),
            )
        )
        output.extend(struct.pack(f"<{len(node['children'])}I", *(offsets[child] for child in node["children"])))

    output.extend(struct.pack(f"<{len(nodes)}I", *(offsets[index] for index in range(len(nodes)))))
    old_table2 = [le32(blob, table2_start + index * 4) for index in range(old_count)]
    old_table2_indices = [index_by_offset[relative] for relative in old_table2]
    table2_indices = old_table2_indices + [int(bone["idx"]) for bone in new_bones]
    output.extend(struct.pack(f"<{len(nodes)}I", *(offsets[index] for index in table2_indices)))
    output.extend(third_bytes)

    old_records = [blob[pose_start + index * 0x50 : pose_start + (index + 1) * 0x50] for index in range(old_count)]
    if any(len(record) != 0x50 for record in old_records):
        raise ValueError("truncated Type 3 pose records")
    output.extend(b"".join(old_records))

    all_bones = [dict(bone) for bone in old_bones] + [dict(bone) for bone in new_bones]
    globals_by_index = global_matrices(all_bones, 1.0)
    for bone in new_bones:
        parent_record = old_records[int(bone["parent"])]
        record = bytearray(parent_record)
        struct.pack_into("<12f", record, 0x04, *cmp_inverse_bind_values(globals_by_index[int(bone["idx"])]))
        struct.pack_into("<3f", record, 0x34, *bone["t"])
        qx, qy, qz, qw = normalized_quaternion(bone["q"])
        struct.pack_into("<4f", record, 0x40, -qx, -qy, -qz, qw)
        output.extend(record)

    tail_start = pose_start + old_count * 0x50
    output.extend(blob[tail_start:])
    struct.pack_into("<I", output, 0x20, len(nodes))
    struct.pack_into("<I", output, 0x28, len(output))
    struct.pack_into("<I", output, 0x2C, new_table1_start)
    struct.pack_into("<I", output, 0x30, new_table2_start)
    struct.pack_into("<I", output, 0x38, new_third_start)
    struct.pack_into("<I", output, 0x3C, new_pose_start)
    return bytes(output)


def rebuild_type4_skeleton(blob: bytes, old_bones: list[dict], new_bones: list[dict]) -> bytes:
    old_count = len(old_bones)
    if le32(blob, 0x20) != 1 or le32(blob, 0x2C) != old_count:
        raise ValueError("unsupported Type 4 skeleton pose layout")
    table_start = le32(blob, 0x30)
    old_offsets = [le32(blob, table_start + index * 4) for index in range(old_count)]
    old_records = [blob[offset : offset + 0x60] for offset in old_offsets]
    if any(len(record) != 0x60 for record in old_records):
        raise ValueError("truncated Type 4 skeleton pose records")

    new_count = old_count + len(new_bones)
    first_record = table_start + new_count * 4
    output = bytearray(blob[:table_start])
    new_offsets = [first_record + index * 0x60 for index in range(new_count)]
    output.extend(struct.pack(f"<{new_count}I", *new_offsets))
    output.extend(b"".join(old_records))
    for bone in new_bones:
        record = bytearray(old_records[int(bone["parent"])])
        struct.pack_into("<i", record, 0x00, int(bone["idx"]))
        struct.pack_into("<3f", record, 0x24, *bone["t"])
        qx, qy, qz, qw = normalized_quaternion(bone["q"])
        struct.pack_into("<4f", record, 0x3C, -qx, -qy, -qz, qw)
        output.extend(record)

    old_records_end = max(old_offsets) + 0x60
    output.extend(blob[old_records_end:])
    struct.pack_into("<I", output, 0x24, len(output))
    struct.pack_into("<I", output, 0x2C, new_count)
    struct.pack_into("<I", output, 0x34, first_record)
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
        ),
        None,
    )
    if hierarchy is None or pose is None:
        raise ValueError("CMP does not contain a character Type 3/Type 4 skeleton pair")
    return hierarchy, pose


def assign_type17_skin_palette_slots(
    blob: bytes,
    new_bones: list[dict],
    name_indices: list[int],
) -> tuple[bytes, list[dict]]:
    if len(blob) < 0x10:
        raise ValueError("CMP Type 17 mesh entry is too small for a skin palette")
    table_count = le32(blob, 0) >> 16
    slot_offsets = [0x04, 0x08, 0x0C]
    slot_offsets.extend(0x10 + index * 4 for index in range(table_count))
    if not slot_offsets or slot_offsets[-1] + 4 > len(blob):
        raise ValueError("CMP Type 17 skin palette extends outside the mesh entry")

    reserved = []
    for raw_id, offset in reversed(list(enumerate(slot_offsets))):
        if le32(blob, offset) != 0:
            break
        reserved.append((raw_id, offset))
    reserved.reverse()
    if len(new_bones) > len(reserved):
        raise ValueError(
            "CMP Type 17 has only "
            f"{len(reserved)} unused skin-palette slot(s), but {len(new_bones)} new bones were added"
        )

    output = bytearray(blob)
    assignments = []
    for bone, name_index, (raw_id, offset) in zip(new_bones, name_indices, reserved):
        if raw_id > 0xFF:
            raise ValueError("CMP vertex skin palette IDs are limited to 0..255")
        struct.pack_into("<I", output, offset, int(name_index))
        assignments.append(
            {
                "bone_index": int(bone["idx"]),
                "bone_name": str(bone["name"]),
                "palette_id": int(raw_id),
                "palette_offset": int(offset),
            }
        )
    return bytes(output), assignments


def import_new_bones(cmp_path: Path, fbx_path: Path, scale: float) -> tuple[bytes, dict]:
    parser = PipeworksParser(str(cmp_path))
    entries = parser.parse()
    if parser.is_big_endian:
        raise ValueError("cmp_add_bones.py currently supports little-endian PS2 CMP files only")
    source = parser.file_data or b""
    old_bones, _globals = parse_cmp_skeleton(parser, entries, scale)
    if not old_bones:
        raise ValueError("CMP skeleton could not be decoded")
    roots, _version = parse_fbx(fbx_path)
    new_bones = detect_new_leaf_bones(roots, old_bones, scale)
    if not new_bones:
        return source, {
            "status": "preserved_no_new_leaf_bones",
            "old_bone_count": len(old_bones),
            "new_bone_count": len(old_bones),
            "animation_policy": "byte-for-byte",
            "mesh_entries_changed": 0,
            "bones": [],
        }

    patched_strings, name_indices = append_bundle_strings(
        source,
        parser.string_offset,
        parser.metadata_offset,
        [str(bone["name"]) for bone in new_bones],
    )
    hierarchy_entry, pose_entry = find_skeleton_entries(entries)
    mesh_entry = next(
        (
            entry
            for entry in entries
            if entry["file_type"] == 17 and not entry["is_resource"]
        ),
        None,
    )
    if mesh_entry is None:
        raise ValueError("CMP does not contain a Type 17 mesh entry for skin-palette assignment")
    hierarchy_blob = source[
        int(hierarchy_entry["offset"]) : int(hierarchy_entry["offset"]) + int(hierarchy_entry["size"])
    ]
    pose_blob = source[int(pose_entry["offset"]) : int(pose_entry["offset"]) + int(pose_entry["size"])]
    mesh_blob = source[int(mesh_entry["offset"]) : int(mesh_entry["offset"]) + int(mesh_entry["size"])]
    rebuilt_mesh, palette_assignments = assign_type17_skin_palette_slots(
        mesh_blob, new_bones, name_indices
    )
    replacements = {
        int(hierarchy_entry["file_num"]): rebuild_type3_skeleton(
            hierarchy_blob, old_bones, new_bones, name_indices
        ),
        int(pose_entry["file_num"]): rebuild_type4_skeleton(pose_blob, old_bones, new_bones),
        int(mesh_entry["file_num"]): rebuilt_mesh,
    }

    animation_files = []
    for entry in entries:
        if entry["file_type"] != 4 or entry["is_resource"] or int(entry["size"]) < 0x44:
            continue
        blob = source[int(entry["offset"]) : int(entry["offset"]) + int(entry["size"])]
        if le32(blob, 0x20) != 2:
            continue
        animation_files.append(int(entry["file_num"]))

    output = replace_bundle_main_entries(patched_strings, replacements, alignment=0x20)
    report = {
        "status": "added_weightable_leaf_bones",
        "old_bone_count": len(old_bones),
        "new_bone_count": len(old_bones) + len(new_bones),
        "animation_count_preserved": len(animation_files),
        "animation_policy": "byte-for-byte; new leaf bones inherit their parent from the rest skeleton",
        "mesh_entries_changed": 1,
        "skin_palette_assignments": palette_assignments,
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
    return output, report


def main() -> int:
    argument_parser = argparse.ArgumentParser(
        description="Append weightable leaf bones from a Blender FBX to a PS2 CMP."
    )
    argument_parser.add_argument("cmp")
    argument_parser.add_argument("fbx")
    argument_parser.add_argument("output")
    argument_parser.add_argument("--scale", type=float, default=10.0)
    args = argument_parser.parse_args()

    output, report = import_new_bones(Path(args.cmp), Path(args.fbx), float(args.scale))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(output)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
