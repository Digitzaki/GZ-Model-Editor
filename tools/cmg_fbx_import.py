"""Patch same-topology GameCube CMG mesh positions from an edited FBX."""
from __future__ import annotations

import argparse
import re
import shutil
import struct
import sys
from pathlib import Path

TOOL_DIR = Path(__file__).resolve().parent
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from cmg_probe import decode_display_list, find_mesh_descriptors, find_type17, parse_positions, position_stride
from fbx_to_bdg_import import object_nodes, parse_fbx
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
        if not verts_node or not verts_node.props:
            continue
        raw = verts_node.props[0]
        verts = []
        for i in range(0, len(raw) - 2, 3):
            verts.append((float(raw[i]), float(raw[i + 1]), float(raw[i + 2])))
        out[int(match.group(1))] = {"name": name, "vertices": verts}
    return out


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


def patch_cmg_positions(original: Path, fbx: Path, out: Path, scale: float) -> dict:
    parser = PipeworksParser(str(original))
    entries = parser.parse()
    data = bytearray(parser.file_data or b"")
    main_entry, res_entry = find_type17(entries)
    main = bytes(data[main_entry["offset"] : main_entry["offset"] + main_entry["size"]])
    resource = bytes(data[res_entry["offset"] : res_entry["offset"] + res_entry["size"]])
    descs = find_mesh_descriptors(main, res_entry["size"])
    geoms = geometry_payloads(fbx)
    patched = []
    skipped = []
    for i, desc in enumerate(descs):
        geom = geoms.get(i)
        if not geom:
            skipped.append({"submesh": i, "status": "missing_fbx_geometry"})
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
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    return {
        "output": str(out),
        "patched": patched,
        "skipped": skipped,
        "note": "Same-topology CMG writeback patched vertex positions only; topology and unknown bytes are preserved.",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Import same-topology FBX edits into a GameCube CMG copy")
    ap.add_argument("fbx")
    ap.add_argument("original")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=float, default=10.0)
    args = ap.parse_args(argv)
    original = Path(args.original)
    out = Path(args.out)
    if original.resolve() != out.resolve():
        shutil.copy2(original, out)
    report = patch_cmg_positions(out, Path(args.fbx), out, args.scale)
    for key, value in report.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
