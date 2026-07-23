"""BDG topology writer for preserving edited mesh topology.

The writer keeps original display-list data stable for removals and appends
new vertices/triangles to one decoded BDG mesh stream for simple duplicates or
extrudes.
"""
from __future__ import annotations

from pathlib import Path
import struct
from typing import Any

from parser_core import PipeworksParser
import bdg_to_fbx_extract_all as bdg

STRIDES = {'skin64':64, 'blend76':76, 'blend52':52, 'blend60':60, 'skin48':48, 'skin40':40}
CMD_TRIS = 0x90


def _mesh_resource_entry(path: Path):
    parser = PipeworksParser(str(path))
    entries = parser.parse()
    meshes = [e for e in entries if e.get('file_type') == 17 and e.get('is_resource')]
    if not meshes:
        raise ValueError('No type-17 mesh resource found in this Shapes.BDG')
    if len(meshes) > 1:
        raise ValueError('Multiple type-17 mesh resources found; refusing topology save until owner mapping is implemented')
    return parser, entries, meshes[0]


def _template_src(src):
    """Return (v_start, local_index, layout) for a real BDG vertex source."""
    if not src:
        return None
    try:
        if int(src[0]) >= 0 and str(src[2]) in STRIDES:
            return (int(src[0]), int(src[1]), str(src[2]))
    except Exception:
        pass
    # Virtual vertices from extrude/duplicate carry their source template here,
    # but they still need a *new* vertex record to save correctly.  Do not map
    # them back to the old template or the game will draw stretched garbage.
    return None


def _write_tri_display_list(indices: list[int], index_width: int) -> bytes:
    if not indices:
        return b''
    max_idx = max(indices)
    if index_width in (3, 4) and max_idx > 255:
        # Cannot widen an in-place compact display list safely.
        raise ValueError('Edited mesh needs 16-bit indices but original display list is compact 8-bit')
    if max_idx > 65535:
        raise ValueError('Too many vertices for GX display-list indices')
    out = bytearray()
    max_count = 1023
    max_count -= max_count % 3
    for start in range(0, len(indices), max_count):
        chunk = indices[start:start + max_count]
        chunk = chunk[:len(chunk) - (len(chunk) % 3)]
        if not chunk:
            continue
        out.append(CMD_TRIS)
        out.extend(struct.pack('>H', len(chunk)))
        for idx in chunk:
            if index_width == 3:
                out.extend(bytes((idx & 0xff, idx & 0xff, idx & 0xff)))
            elif index_width == 4:
                out.extend(bytes((idx & 0xff, idx & 0xff, idx & 0xff, idx & 0xff)))
            elif index_width == 8:
                out.extend(struct.pack('>4H', idx, idx, idx, idx))
            else:
                out.extend(struct.pack('>3H', idx, idx, idx))
    return bytes(out)


def _tri_key(tri):
    return tuple(sorted(int(v) for v in tri))


def _record_from_bytes(buf: bytes, index_width: int):
    if index_width == 6:
        a, b, c = struct.unpack('>3H', buf[:6])
        return {'raw': bytes(buf[:6]), 'a': a, 'b': b, 'c': c}
    if index_width == 3:
        return {'raw': bytes(buf[:3]), 'a': buf[0], 'b': buf[1], 'c': buf[2]}
    if index_width == 4:
        return {'raw': bytes(buf[:4]), 'a': buf[0], 'b': buf[1], 'c': buf[2]}
    if index_width == 8:
        a, b, c, d = struct.unpack('>4H', buf[:8])
        return {'raw': bytes(buf[:8]), 'a': a, 'b': b, 'c': c, 'd': d}
    raise ValueError('unsupported display-list index width')


def _degenerate_record_like(rec: dict, index_width: int) -> bytes:
    idx = int(rec.get('a', 0))
    if index_width == 6:
        return struct.pack('>3H', idx, idx, idx)
    if index_width == 3:
        b = idx & 0xff
        return bytes((b, b, b))
    if index_width == 4:
        b = idx & 0xff
        # Preserve the fourth attribute byte when present; for observed compact
        # streams it is usually another duplicated index/color slot.
        d = int(rec.get('d', idx)) & 0xff
        return bytes((b, b, b, d))
    if index_width == 8:
        return struct.pack('>4H', idx, idx, idx, idx)
    raise ValueError('unsupported display-list index width')


def _parse_dl_commands_with_records(D: bytes, start: int, end: int, index_width: int):
    commands = []
    pos = start
    while pos + 3 <= end and pos + 3 <= len(D) and D[pos] in (0x80, 0x90, 0x98, 0xA0):
        op = D[pos]
        count = int.from_bytes(D[pos+1:pos+3], 'big')
        cmd_start = pos
        pos += 3
        byte_count = index_width * count
        if count < 3 or count > 4096 or pos + byte_count > end or pos + byte_count > len(D):
            break
        records = []
        for _ in range(count):
            raw = D[pos:pos+index_width]
            records.append(_record_from_bytes(raw, index_width))
            pos += index_width
        faces = []
        if op == 0x80:  # quads
            for i in range(0, len(records) - 3, 4):
                faces.append((records[i], records[i+1], records[i+2]))
                faces.append((records[i], records[i+2], records[i+3]))
        elif op == 0x90:  # triangles
            for i in range(0, len(records) - 2, 3):
                faces.append((records[i], records[i+1], records[i+2]))
        elif op == 0x98:  # strip
            for i in range(len(records) - 2):
                a, b, c = records[i], records[i+1], records[i+2]
                if i % 2 == 0:
                    faces.append((a, b, c))
                else:
                    faces.append((b, a, c))
        elif op == 0xA0:  # fan
            root = records[0]
            for i in range(1, len(records) - 1):
                faces.append((root, records[i], records[i+1]))
        commands.append({'start': cmd_start, 'end': pos, 'op': op, 'count': count, 'records': records, 'faces': faces})
    return commands


def _face_local_tuple(face):
    return tuple(int(r.get('a', 0)) for r in face)



def _emit_dl_command(op: int, records: list[dict], index_width: int) -> bytes:
    """Build a GX primitive command from existing parsed attribute records."""
    if not records:
        return b''
    if len(records) > 65535:
        raise ValueError('Display-list command is too large')
    out = bytearray((op & 0xff,))
    out.extend(len(records).to_bytes(2, 'big'))
    for r in records:
        raw = r.get('raw')
        if raw is None or len(raw) != index_width:
            raise ValueError('Cannot re-emit malformed display-list record')
        out.extend(raw)
    return bytes(out)


def _pack_compact_command_same_space(cmd: dict, keep_flags: list[bool], index_width: int) -> bytes | None:
    """Try to represent a mixed strip/fan/quad command within its original byte size.

    The first safe writer refused any compact primitive where the kept triangles
    could not fit as a single raw triangle list.  That was too conservative:
    Blender-style face deletion often removes one connected patch from a long GX
    strip/fan, and the remainder can be represented as two or three smaller
    strips/fans in exactly the same display-list island.

    Returns same-or-smaller replacement bytes, or None when the command truly
    cannot be represented without growing/moving the BDG stream.
    """
    op = int(cmd.get('op', 0))
    records = list(cmd.get('records') or [])
    count = int(cmd.get('count', len(records)))
    capacity = 3 + count * index_width
    if not records or not keep_flags:
        return None

    out = bytearray()

    if op == 0x98:  # triangle strip: contiguous kept face runs remain strips
        i = 0
        while i < len(keep_flags):
            if not keep_flags[i]:
                i += 1
                continue
            j = i
            while j < len(keep_flags) and keep_flags[j]:
                j += 1
            # faces i..j-1 need records i..j+1
            run_records = records[i:j+2]
            if len(run_records) >= 3:
                out.extend(_emit_dl_command(0x98, run_records, index_width))
            i = j
        if len(out) <= capacity:
            return bytes(out)

    if op == 0xA0:  # triangle fan: contiguous kept face runs become smaller fans
        root = records[0]
        i = 0
        while i < len(keep_flags):
            if not keep_flags[i]:
                i += 1
                continue
            j = i
            while j < len(keep_flags) and keep_flags[j]:
                j += 1
            # fan face i uses root, records[i+1], records[i+2]
            run_records = [root] + records[i+1:j+2]
            if len(run_records) >= 3:
                out.extend(_emit_dl_command(0xA0, run_records, index_width))
            i = j
        if len(out) <= capacity:
            return bytes(out)

    if op == 0x80:  # quads: patch each independent quad in-place when possible
        # A GX quad emits two triangles: (a,b,c) and (a,c,d).  If only one half
        # is deleted, duplicate b or d so just that half degenerates while the
        # other half still draws normally.
        body = bytearray()
        qi = 0
        fi = 0
        while qi + 3 < len(records):
            a, b, c, d = records[qi], records[qi+1], records[qi+2], records[qi+3]
            keep0 = bool(keep_flags[fi]) if fi < len(keep_flags) else False
            keep1 = bool(keep_flags[fi+1]) if fi + 1 < len(keep_flags) else False
            if keep0 and keep1:
                quad = [a, b, c, d]
            elif keep0 and not keep1:
                # second triangle (a,c,d) degenerates when d == a
                quad = [a, b, c, a]
            elif (not keep0) and keep1:
                # first triangle (a,b,c) degenerates when b == a
                quad = [a, a, c, d]
            else:
                # both gone: fully degenerate quad
                quad = [a, a, a, a]
            for r in quad:
                body.extend(r['raw'])
            qi += 4
            fi += 2
        repl = bytearray((0x80,)) + count.to_bytes(2, 'big') + body
        if len(repl) <= capacity:
            return bytes(repl[:capacity])

    # Last resort: independent triangles. This can still fit for large deletions.
    tri_records = []
    for face, keep in zip(cmd.get('faces') or [], keep_flags):
        if keep:
            tri_records.extend(face)
    if tri_records:
        repl = _emit_dl_command(0x90, tri_records, index_width)
        if len(repl) <= capacity:
            return repl
    return None


def _patch_display_list_in_place(D: bytes, sm: dict, keep_keys: set[tuple[int, int, int]], index_width: int, known_original_keys: set[tuple[int, int, int]] | None = None) -> bytes:
    """Return a same-length display-list byte string with removed faces degenerated.

    This preserves every display-list island's original byte footprint. For normal
    triangle-list streams it simply degenerates deleted triangles in-place. For
    strips/fans/quads it leaves untouched commands alone, removes whole commands
    when every generated face was deleted, and converts mixed commands to a
    same-size triangle list only when the kept faces fit into the existing vertex
    record count.
    """
    dl_start = int(sm['dl_start'])
    dl_end = int(sm.get('dl_end', sm['v_start']))
    original = bytearray(D[dl_start:dl_end])
    commands = _parse_dl_commands_with_records(D, dl_start, dl_end, index_width)
    if not commands:
        raise ValueError('Could not parse original display-list commands for safe in-place patching')
    patched = bytearray(original)

    for cmd in commands:
        faces = cmd['faces']
        if not faces:
            continue
        face_keys = [_tri_key(_face_local_tuple(f)) for f in faces]
        # IMPORTANT: the raw GX strip/fan parser can expose helper/degenerate
        # triangles that the editor never surfaced as editable faces.  Earlier
        # builds treated every raw GX-generated triangle absent from the current
        # editor mesh as deleted, which removed unrelated body parts when only an
        # arm/hand patch was selected.  Only delete faces that were part of the
        # editor's original editable face set and are now missing.  Unknown raw
        # GX triangles are preserved byte-for-byte.
        if known_original_keys is None:
            keep_flags = [k in keep_keys for k in face_keys]
        else:
            keep_flags = [(k not in known_original_keys) or (k in keep_keys) for k in face_keys]
        if all(keep_flags):
            continue
        rel = cmd['start'] - dl_start
        count = int(cmd['count'])
        records = cmd['records']
        deg = _degenerate_record_like(records[0], index_width)

        if not any(keep_flags):
            # Same command byte count, same original GX primitive opcode,
            # but every vertex record points to the same source vertex so the
            # primitive becomes degenerate. Do NOT change strips/fans/quads to
            # TRIANGLES here: their vertex count may not be a valid triangle
            # multiple, and changing the primitive type was unsafe in-game.
            patched[rel] = cmd['op']
            patched[rel+1:rel+3] = count.to_bytes(2, 'big')
            body = deg * count
            patched[rel+3:rel+3+len(body)] = body
            continue

        if cmd['op'] == CMD_TRIS:
            # Existing triangles have one independent record triplet per face.
            body = bytearray()
            fi = 0
            for i in range(0, count, 3):
                if fi < len(keep_flags) and keep_flags[fi]:
                    for r in records[i:i+3]:
                        body.extend(r['raw'])
                else:
                    body.extend(deg * min(3, count - i))
                fi += 1
            patched[rel+3:rel+3+len(body)] = body[:index_width * count]
            continue

        packed = _pack_compact_command_same_space(cmd, keep_flags, index_width)
        if packed is not None and len(packed) <= (cmd['end'] - cmd['start']):
            # Clear this whole original command footprint, then write one or more
            # smaller safe GX primitive commands into it.  Unused command space is filled with GX NOP (0x00). 0xff is not safe
            # for Dolphin/GX FIFO and can cause unknown-opcode crashes.
            cmd_len = cmd['end'] - cmd['start']
            patched[rel:rel+cmd_len] = b'\x00' * cmd_len
            patched[rel:rel+len(packed)] = packed
            continue

        # Fallback for compact GX primitives that cannot represent the exact
        # Blender face-mask inside the original byte budget: delete the whole
        # original primitive island by degenerating every record in-place while
        # keeping the original opcode and vertex count. This preserves byte count
        # and avoids introducing malformed TRIANGLE commands or invalid opcodes.
        patched[rel] = cmd['op']
        patched[rel+1:rel+3] = count.to_bytes(2, 'big')
        body = deg * count
        patched[rel+3:rel+3+len(body)] = body[:index_width * count]
        continue
    return bytes(patched)


def _get_submeshes(path: Path):
    D = Path(path).read_bytes()
    _, _, strings = bdg.find_strtab(D)
    _, _, skeleton = bdg.find_skeleton(D, strings)
    bone_count = max(skeleton) + 1
    submeshes, _ = bdg.choose_meshes(D, bone_count)
    return D, submeshes


def save_topology_payload_to_bdg(shape_path: str | Path, payload: dict[str, Any]):
    """Persist safe topology removals without moving any BDG mesh data.

    The current BDG reverse map is safe for deletion because the original vertex
    streams can remain in-place and the display list can simply stop referencing
    removed faces.  Added virtual vertices are refused for now because saving
    them requires growing/relocating vertex streams and updating internal game
    offsets that are not fully mapped yet.
    """
    path = Path(shape_path)
    D, submeshes = _get_submeshes(path)
    parser, entries, mesh_entry = _mesh_resource_entry(path)

    vertices = payload.get('vertices') or []
    triangles = [tuple(t) for t in (payload.get('triangles') or [])]
    tri_groups = list(payload.get('tri_groups') or [])
    vertex_src = list(payload.get('vertex_src') or [])
    preserve_original_display_lists = bool(payload.get('preserve_original_display_lists'))
    preserve_original_face_keys_raw = payload.get('preserve_original_face_keys') or {}
    preserve_delete_keys_by_group: dict[int, set[tuple[int, int, int]]] = {}
    if isinstance(preserve_original_face_keys_raw, dict):
        for group_key, keys in preserve_original_face_keys_raw.items():
            try:
                group_i = int(group_key)
            except Exception:
                continue
            preserve_delete_keys_by_group[group_i] = {
                _tri_key(key) for key in (keys or []) if len(key) == 3
            }

    if len(tri_groups) != len(triangles):
        tri_groups = [tri_groups[i] if i < len(tri_groups) else 0 for i in range(len(triangles))]

    # Refuse virtual/new vertices. Mapping these back to their template source
    # was the cause of the long spike/corrupt Gigan result.
    for i, src in enumerate(vertex_src):
        if _template_src(src) is None:
            # It is okay if the vertex is totally orphaned and no triangle uses it,
            # but any referenced virtual vertex requires a real new BDG record.
            if any(i in tri for tri in triangles):
                raise ValueError(
                    'This edit adds new vertices. Persistent add/extrude/duplicate saving is disabled in this safe build; '
                    'delete/removal saves are supported, but added topology needs the BDG grow/offset map first.'
                )

    for i, tri in enumerate(triangles):
        if len(tri) != 3 or not all(isinstance(v, int) and 0 <= v < len(vertices) for v in tri):
            raise ValueError(f'Cannot save malformed triangle #{i}: {tri!r}')

    resource_start = int(mesh_entry['offset'])
    resource_end = resource_start + int(mesh_entry['size'])
    out = bytearray(D)

    for gi, sm in enumerate(submeshes):
        dl_start = int(sm['dl_start'])
        v_start = int(sm['v_start'])
        if not (resource_start <= dl_start < v_start <= resource_end):
            raise ValueError('Decoded mesh stream is outside the type-17 mesh resource')
        dl_capacity = v_start - dl_start
        layout = sm['layout']
        source_v_start = int(sm['v_start'])

        indices: list[int] = []
        for tri, g in zip(triangles, tri_groups):
            if int(g) != gi:
                continue
            local = []
            for vid in tri:
                src = _template_src(vertex_src[vid] if vid < len(vertex_src) else None)
                if src is None:
                    raise ValueError('Triangle references a new/virtual vertex; refusing unsafe save')
                sv_start, src_idx, src_layout = src
                if sv_start != source_v_start or src_layout != layout:
                    raise ValueError('Triangle crossed BDG submesh streams; refusing unsafe save')
                if src_idx < 0 or src_idx >= int(sm['v_count']):
                    raise ValueError('Triangle references a vertex outside its original BDG stream')
                local.append(src_idx)
            # Skip degenerate faces; they are harmless in the editor but waste DL space.
            if len(set(local)) == 3:
                indices.extend(local)

        # If this submesh's face set is unchanged, leave its original compact
        # display-list bytes alone. Rewriting an unchanged strip/fan/quad island
        # as raw triangles can be larger than the original, which caused false
        # save failures on Gigan's compact detail island.
        original_keys = {_tri_key(face) for face in sm.get('faces', [])}
        current_keys = set()
        for tri, g in zip(triangles, tri_groups):
            if int(g) != gi:
                continue
            local = []
            for vid in tri:
                src = _template_src(vertex_src[vid] if vid < len(vertex_src) else None)
                if src is None:
                    raise ValueError('Triangle references a new/virtual vertex; refusing unsafe save')
                sv_start, src_idx, src_layout = src
                if sv_start != source_v_start or src_layout != layout:
                    raise ValueError('Triangle crossed BDG submesh streams; refusing unsafe save')
                if src_idx < 0 or src_idx >= int(sm['v_count']):
                    raise ValueError('Triangle references a vertex outside its original BDG stream')
                local.append(src_idx)
            if len(set(local)) == 3:
                current_keys.add(_tri_key(tuple(local)))

        if current_keys == original_keys:
            continue

        # First try a compact, same-footprint patch of the original primitive
        # stream. Triangle-list streams degenerate deleted faces in-place; compact
        # strips/fans/quads are only converted when the kept faces fit inside the
        # existing command record count.
        patched = _patch_display_list_in_place(D, sm, current_keys, int(sm.get('index_width', 6)), original_keys)
        if len(patched) > dl_capacity:
            raise ValueError(
                f'Patched display list for submesh {gi} is {len(patched)} bytes but only {dl_capacity} bytes fit in-place. '
                'Refusing save to avoid corrupting the game display list.'
            )
        # Patch only the original decoded display-list bytes.  Leave the small
        # alignment/padding gap before the vertex stream exactly as it was; some
        # files use that area in ways the extractor should not reinterpret.
        out[dl_start:dl_start + len(patched)] = patched

    path.write_bytes(bytes(out))



# ---------------------------------------------------------------------------
# Experimental V19: aligned grow writer for added faces.
# ---------------------------------------------------------------------------
# This stays based on the stable delete/remove writer above.  The important
# difference from V18 is that added GX commands are inserted at the true end of
# the valid display-list command stream, not after the descriptor-sized region.
# Descriptor-sized regions contain alignment bytes; if those bytes sit before a
# newly appended primitive, Dolphin/the game can execute them as invalid GX
# opcodes.  V19 rebuilds only the touched stream's DL gap, keeps following
# streams aligned, and validates every descriptor display list for invalid
# opcodes before committing.


def _u32(data: bytes | bytearray, off: int, endian: str = '>') -> int:
    return struct.unpack_from(endian + 'I', data, off)[0]


def _p32(data: bytearray, off: int, value: int, endian: str = '>') -> None:
    struct.pack_into(endian + 'I', data, off, int(value) & 0xffffffff)


def _align(n: int, a: int = 0x20) -> int:
    return (int(n) + a - 1) // a * a


def _find_mesh_descriptors(D: bytes, main_entry: dict, mesh_entry: dict, submeshes: list[dict], endian: str) -> list[dict]:
    """Find type-17 main-file stream descriptors for each resource submesh.

    Observed descriptor layout, relative to descriptor base:
      +0  GX display-list vertex-record count
      +4  vertex attribute/index format marker (0x10 for 16-bit, 0x8 for compact)
      +8  display-list offset inside the type-17 resource
      +12 display-list byte size, including alignment padding up to vertex stream
      +16 display-list attribute count (observed 3)
      +32 vertex count
      +36 vertex stride
      +40 vertex-layout flags
      +48 vertex-stream offset inside the type-17 resource
      +52 vertex-stream byte size
    """
    main_start = int(main_entry['offset'])
    main_size = int(main_entry['size'])
    B = D[main_start:main_start + main_size]
    res_start = int(mesh_entry['offset'])
    descs = []
    used = set()
    for sm in submeshes:
        rel_dl = int(sm['dl_start']) - res_start
        dl_size = int(sm.get('dl_end', sm['v_start'])) - int(sm['dl_start'])
        rel_v = int(sm['v_start']) - res_start
        v_count = int(sm['v_count'])
        stride = int(sm['v_stride'])
        v_size = v_count * stride
        found = None
        for j in range(8, max(8, main_size - 56), 4):
            base = j - 8
            if base in used:
                continue
            try:
                if _u32(B, base + 8, endian) != rel_dl:
                    continue
                if _u32(B, base + 12, endian) != dl_size:
                    continue
                if _u32(B, base + 32, endian) != v_count:
                    continue
                if _u32(B, base + 36, endian) != stride:
                    continue
                if _u32(B, base + 48, endian) != rel_v:
                    continue
                if _u32(B, base + 52, endian) != v_size:
                    continue
            except Exception:
                continue
            found = {
                'base': main_start + base,
                'base_rel': base,
                'rel_dl': rel_dl,
                'dl_size': dl_size,
                'record_count': _u32(B, base + 0, endian),
                'format_marker': _u32(B, base + 4, endian),
                'attr_count': _u32(B, base + 16, endian),
                'v_count': v_count,
                'stride': stride,
                'layout_flags': _u32(B, base + 40, endian),
                'rel_v': rel_v,
                'v_size': v_size,
            }
            used.add(base)
            break
        if found is None:
            raise ValueError('Could not find a type-17 mesh descriptor for one decoded stream; refusing add/extrude save.')
        descs.append(found)
    return descs


def _virtual_template(src):
    try:
        if len(src) >= 4 and str(src[2]) == 'virtual':
            tmpl = src[3]
            if len(tmpl) >= 3:
                return (int(tmpl[0]), int(tmpl[1]), str(tmpl[2]))
    except Exception:
        return None
    return None


def _virtual_display_record(src):
    try:
        if len(src) >= 5 and str(src[2]) == 'virtual':
            rec = src[4]
        elif len(src) >= 4 and str(src[2]) == 'virtual':
            tmpl = src[3]
            rec = tmpl[3] if len(tmpl) >= 4 else None
        else:
            rec = None
        if isinstance(rec, dict):
            raw_hex = rec.get('raw_hex')
            raw = bytes.fromhex(raw_hex) if raw_hex else None
            out = dict(rec)
            if raw is not None:
                out['raw'] = raw
            return out
    except Exception:
        return None
    return None


def _pack_index_record(idx: int, index_width: int) -> bytes:
    idx = int(idx)
    if index_width == 6:
        if idx > 65535:
            raise ValueError('New face index exceeds 16-bit GX range')
        return struct.pack('>3H', idx, idx, idx)
    if index_width == 3:
        if idx > 255:
            raise ValueError('New face index exceeds compact 8-bit GX range')
        b = idx & 0xff
        return bytes((b, b, b))
    if index_width == 4:
        if idx > 255:
            raise ValueError('New face index exceeds compact 8-bit GX range')
        b = idx & 0xff
        return bytes((b, b, b, b))
    if index_width == 8:
        if idx > 65535:
            raise ValueError('New face index exceeds 16-bit GX range')
        return struct.pack('>4H', idx, idx, idx, idx)
    raise ValueError(f'Unsupported GX index-record width {index_width}')


def _pack_index_record_like(idx: int, index_width: int, template: dict | None = None) -> bytes:
    """Pack an appended GX record for a newly appended native vertex.

    The editor stores position/normal/UV together in each appended BDG vertex
    record.  If a model's GX display-list vertex uses separate attr indices,
    every geometry-bearing slot must point at the new vertex; otherwise the
    game combines the new position with the template vertex's old normal/UV,
    which renders as smeared or scrambled extrusion textures.
    """
    if not template or template.get('raw') is None:
        return _pack_index_record(idx, index_width)
    idx = int(idx)
    raw = bytearray(template.get('raw') or b'')
    if len(raw) != index_width:
        return _pack_index_record(idx, index_width)
    if index_width in (3, 4):
        if idx > 255:
            raise ValueError('New face index exceeds compact 8-bit GX range')
        raw[0] = idx & 0xff
        raw[1] = idx & 0xff
        raw[2] = idx & 0xff
        if index_width == 4:
            raw[3] = idx & 0xff
        return bytes(raw)
    if index_width == 6:
        if idx > 65535:
            raise ValueError('New face index exceeds 16-bit GX range')
        struct.pack_into('>3H', raw, 0, idx, idx, idx)
        return bytes(raw)
    if index_width == 8:
        if idx > 65535:
            raise ValueError('New face index exceeds 16-bit GX range')
        struct.pack_into('>4H', raw, 0, idx, idx, idx, idx)
        return bytes(raw)
    return _pack_index_record(idx, index_width)


def _build_triangle_commands(local_tris: list[tuple[int, int, int]], index_width: int) -> bytes:
    out = bytearray()
    if not local_tris:
        return b''
    max_records = 1023
    max_tris = max_records // 3
    for start in range(0, len(local_tris), max_tris):
        chunk = local_tris[start:start + max_tris]
        out.append(CMD_TRIS)
        out.extend(struct.pack('>H', len(chunk) * 3))
        for tri in chunk:
            for idx in tri:
                out.extend(_pack_index_record(idx, index_width))
    return bytes(out)


def _build_triangle_record_commands(local_tris: list[tuple[dict, dict, dict]], index_width: int) -> bytes:
    out = bytearray()
    if not local_tris:
        return b''
    max_records = 1023
    max_tris = max_records // 3
    for start in range(0, len(local_tris), max_tris):
        chunk = local_tris[start:start + max_tris]
        out.append(CMD_TRIS)
        out.extend(struct.pack('>H', len(chunk) * 3))
        for tri in chunk:
            for rec in tri:
                if rec.get('new_vertex'):
                    out.extend(_pack_index_record(int(rec['idx']), index_width))
                else:
                    out.extend(_pack_index_record_like(int(rec['idx']), index_width, rec.get('template')))
    return bytes(out)


def _template_records_by_primary_index(D: bytes, sm: dict, index_width: int) -> dict[int, dict]:
    out = {}
    for cmd in _parse_dl_commands_with_records(D, int(sm['dl_start']), int(sm.get('dl_end', sm['v_start'])), index_width):
        for rec in cmd.get('records') or []:
            try:
                out.setdefault(int(rec.get('a', 0)), rec)
            except Exception:
                pass
    return out


def _patch_native_vertex_record(record: bytearray, layout: str, pos, normal=None, uv=None, weights=None) -> bytes:
    struct.pack_into('>3f', record, 0, float(pos[0]), float(pos[1]), float(pos[2]))
    if normal is not None:
        try:
            if layout == 'skin64':
                struct.pack_into('>3f', record, 28, float(normal[0]), float(normal[1]), float(normal[2]))
            elif layout == 'blend76':
                struct.pack_into('>3f', record, 40, float(normal[0]), float(normal[1]), float(normal[2]))
            elif layout in ('blend52', 'blend60'):
                struct.pack_into('>3f', record, 32, float(normal[0]), float(normal[1]), float(normal[2]))
            elif layout in ('skin48', 'skin40'):
                struct.pack_into('>3f', record, 20, float(normal[0]), float(normal[1]), float(normal[2]))
        except Exception:
            pass
    if uv is not None:
        try:
            if layout == 'skin64':
                struct.pack_into('>2f', record, 20, float(uv[0]), float(uv[1]))
            elif layout == 'blend76':
                struct.pack_into('>2f', record, 32, float(uv[0]), float(uv[1]))
            elif layout in ('blend52', 'blend60'):
                struct.pack_into('>2f', record, 44, float(uv[0]), float(uv[1]))
            elif layout in ('skin48', 'skin40'):
                struct.pack_into('>2f', record, 32, float(uv[0]), float(uv[1]))
        except Exception:
            pass
    if weights:
        try:
            pairs = [(int(b), float(w)) for b, w in weights if float(w) > 1e-7]
            pairs.sort(key=lambda x: x[1], reverse=True)
            if layout in ('skin64', 'skin48', 'skin40'):
                pairs = pairs[:2]
                if len(pairs) == 1:
                    b0, w0 = pairs[0]
                    b1 = b0
                elif pairs:
                    b0, w0 = pairs[0]
                    b1, _w1 = pairs[1]
                else:
                    b0 = b1 = 0
                    w0 = 1.0
                struct.pack_into('>f', record, 12, float(w0))
                struct.pack_into('>2H', record, 16, int(b0), int(b1))
            elif layout in ('blend76', 'blend52', 'blend60'):
                pairs = pairs[:4]
                while pairs and len(pairs) < 4:
                    pairs.append((pairs[-1][0], 0.0))
                if pairs:
                    total = sum(w for _b, w in pairs) or 1.0
                    pairs = [(b, w / total) for b, w in pairs]
                    struct.pack_into('>3f', record, 12, float(pairs[0][1]), float(pairs[1][1]), float(pairs[2][1]))
                    struct.pack_into('>4H', record, 24, int(pairs[0][0]), int(pairs[1][0]), int(pairs[2][0]), int(pairs[3][0]))
        except Exception:
            pass
    return bytes(record)


def _valid_command_end(D: bytes | bytearray, start: int, end: int, index_width: int) -> tuple[int, int, str]:
    """Return (last_real_command_end, record_count, error_text).

    NOPs are allowed only after at least one real command.  Any other byte in the
    descriptor display-list region is treated as unsafe because Dolphin will see
    it if the descriptor byte size includes it.
    """
    pos = int(start)
    last = int(start)
    records = 0
    real = 0
    while pos < end:
        b = D[pos]
        if b == 0x00 and real > 0:
            pos += 1
            continue
        if b not in (0x80, 0x90, 0x98, 0xA0):
            return last, records, f'invalid GX opcode 0x{b:02x} at 0x{pos:x}'
        if pos + 3 > end:
            return last, records, f'truncated GX command at 0x{pos:x}'
        count = int.from_bytes(D[pos+1:pos+3], 'big')
        cmd_end = pos + 3 + count * index_width
        if count < 3 or count > 4096 or cmd_end > end:
            return last, records, f'invalid GX count {count} at 0x{pos:x}'
        real += 1
        records += count
        last = cmd_end
        pos = cmd_end
    return last, records, ''


def _strict_validate_descriptor_streams(D: bytes, main_entry: dict, mesh_entry: dict, submeshes: list[dict], descs: list[dict], endian: str) -> tuple[bool, str]:
    res_start = int(mesh_entry['offset'])
    for i, (sm, desc) in enumerate(zip(submeshes, descs)):
        iw = int(sm.get('index_width', 6))
        start = res_start + int(desc['rel_dl'])
        end = start + int(desc['dl_size'])
        if not (res_start <= start < end <= res_start + int(mesh_entry['size']) + 0x1000000):
            return False, f'submesh {i} descriptor range is outside resource bounds'
        _last, count, err = _valid_command_end(D, start, end, iw)
        if err:
            return False, f'submesh {i}: {err}'
        expected = int(desc.get('record_count', 0))
        # Record count is the game's most likely draw count.  Keep it exact so a
        # valid descriptor does not draw beyond the real command stream.
        if expected and count != expected:
            return False, f'submesh {i}: descriptor record_count={expected}, parsed={count}'
    return True, 'strict GX stream validation passed'


def _write_candidate_debug(path: Path, text: str, candidate: bytes | None = None) -> None:
    try:
        path.with_name(path.stem + '_add_save_debug.txt').write_text(text, encoding='utf-8')
        if candidate is not None:
            path.with_name(path.stem + '_ADD_CANDIDATE_DO_NOT_USE.BDG').write_bytes(candidate)
    except Exception:
        pass


def _validate_candidate(candidate_path: Path, original_submesh_count: int) -> tuple[bool, str]:
    try:
        _D, submeshes = _get_submeshes(candidate_path)
        if len(submeshes) != original_submesh_count:
            return False, f'candidate decoded {len(submeshes)} submeshes; expected {original_submesh_count}'
        return True, 'candidate reload validation passed'
    except Exception as exc:
        return False, f'candidate reload validation error: {exc}'


def _validate_non_mesh_resources_preserved(
    original: bytes,
    candidate: bytes,
    entries: list[dict],
    mesh_entry: dict,
    total_delta: int,
) -> tuple[bool, str]:
    """Ensure geometry growth did not rewrite shader/decal/texture resources."""
    mesh_off = int(mesh_entry['offset'])
    mesh_size = int(mesh_entry['size'])
    mesh_end = mesh_off + mesh_size
    for e in entries:
        if not e.get('is_resource') or e is mesh_entry:
            continue
        old_off = int(e['offset'])
        size = int(e['size'])
        if size <= 0:
            continue
        new_off = old_off + int(total_delta) if old_off > mesh_off else old_off
        old_payload = original[old_off:old_off + size]
        new_payload = candidate[new_off:new_off + size]
        if old_payload != new_payload:
            return False, f"resource payload changed unexpectedly: {e.get('name', '<unnamed>')}"
    return True, 'non-mesh resource payloads preserved'


def _save_added_topology_grow_v19(shape_path: str | Path, payload: dict[str, Any]):
    path = Path(shape_path)
    original = path.read_bytes()
    D, submeshes = _get_submeshes(path)
    parser, entries, mesh_entry = _mesh_resource_entry(path)
    main_entries = [e for e in entries if e.get('file_type') == 17 and not e.get('is_resource')]
    if len(main_entries) != 1:
        raise ValueError('Could not uniquely identify the type-17 main mesh descriptor file')
    main_entry = main_entries[0]
    endian = '>' if parser.is_big_endian else '<'

    vertices = list(payload.get('vertices') or [])
    normals = list(payload.get('normals') or [])
    uvs = list(payload.get('uvs') or [])
    weights = list(payload.get('weights') or [])
    triangles = [tuple(t) for t in (payload.get('triangles') or [])]
    tri_groups = list(payload.get('tri_groups') or [])
    vertex_src = list(payload.get('vertex_src') or [])
    preserve_original_display_lists = bool(payload.get('preserve_original_display_lists'))
    preserve_original_face_keys_raw = payload.get('preserve_original_face_keys') or {}
    preserve_delete_keys_by_group: dict[int, set[tuple[int, int, int]]] = {}
    if isinstance(preserve_original_face_keys_raw, dict):
        for group_key, keys in preserve_original_face_keys_raw.items():
            try:
                group_i = int(group_key)
            except Exception:
                continue
            preserve_delete_keys_by_group[group_i] = {
                _tri_key(key) for key in (keys or []) if len(key) == 3
            }
    if len(tri_groups) != len(triangles):
        tri_groups = [tri_groups[i] if i < len(tri_groups) else 0 for i in range(len(triangles))]

    virtual_refs = set()
    for tri in triangles:
        for vid in tri:
            if isinstance(vid, int) and 0 <= vid < len(vertex_src) and _virtual_template(vertex_src[vid]) is not None:
                virtual_refs.add(vid)
    if not virtual_refs:
        return _safe_delete_entry_point(path, payload)

    descs = _find_mesh_descriptors(D, main_entry, mesh_entry, submeshes, endian)

    owner_groups = set()
    for vid in virtual_refs:
        tmpl = _virtual_template(vertex_src[vid])
        if tmpl is None:
            raise ValueError('New vertex does not have a stable original BDG template record')
        t_v_start, _t_idx, t_layout = tmpl
        matched = None
        for gi, sm in enumerate(submeshes):
            if int(sm['v_start']) == t_v_start and str(sm['layout']) == t_layout:
                matched = gi
                break
        if matched is None:
            decoded = [(i, int(sm['v_start']), str(sm['layout'])) for i, sm in enumerate(submeshes)]
            raise ValueError(
                f'New vertex template does not match a decoded BDG submesh stream: '
                f'vid={vid} template={(t_v_start, _t_idx, t_layout)} decoded={decoded}'
            )
        owner_groups.add(matched)
    groups_with_new_faces: dict[int, int] = {}
    for tri, g in zip(triangles, tri_groups):
        if any(v in virtual_refs for v in tri):
            groups_with_new_faces[int(g)] = groups_with_new_faces.get(int(g), 0) + 1
    if groups_with_new_faces:
        # New topology is appended to the stream that owns the new faces.  Some
        # duplicated vertices can carry templates from another nearby stream;
        # those templates are only copy sources, not new submesh ownership.
        gi = max(groups_with_new_faces, key=lambda g: (groups_with_new_faces[g], g))
    elif len(owner_groups) == 1:
        gi = next(iter(owner_groups))
    else:
        touched = owner_groups | set(groups_with_new_faces)
        raise ValueError(f'Add/extrude save currently supports one BDG submesh at a time; touched={sorted(touched)}')
    sm = submeshes[gi]
    desc = descs[gi]
    layout = str(sm['layout'])
    stride = int(sm['v_stride'])
    index_width = int(sm.get('index_width', 6))
    res_start = int(mesh_entry['offset'])
    old_dl_start = int(sm['dl_start'])
    old_v_start = int(sm['v_start'])
    orig_v_count = int(sm['v_count'])
    old_v_size = orig_v_count * stride
    old_v_end = old_v_start + old_v_size
    if index_width in (3, 4) and orig_v_count + len(virtual_refs) > 256:
        raise ValueError(
            'Add/extrude save cannot append this many vertices to the selected compact 8-bit BDG submesh. '
            'Try duplicating fewer faces at once or duplicate from a larger/main mesh stream.'
        )
    template_records = _template_records_by_primary_index(D, sm, index_width)

    def _template_for_target_stream(vid: int):
        tmpl = _virtual_template(vertex_src[vid])
        if tmpl is not None:
            t_v_start, t_idx, t_layout = tmpl
            if t_v_start == old_v_start and t_layout == layout and 0 <= t_idx < orig_v_count:
                return tmpl
        fallback = None
        best = float('inf')
        try:
            pos = vertices[vid]
        except Exception:
            pos = None
        for src in vertex_src:
            real = _template_src(src)
            if real is None:
                real = _virtual_template(src)
            if real is None:
                continue
            sv_start, src_idx, src_layout = real
            if sv_start != old_v_start or src_layout != layout or not (0 <= src_idx < orig_v_count):
                continue
            if pos is None:
                return real
            try:
                off = old_v_start + int(src_idx) * stride
                sx, sy, sz = struct.unpack_from('>3f', D, off)
                dx = float(pos[0]) - sx
                dy = float(pos[1]) - sy
                dz = float(pos[2]) - sz
                dist = dx * dx + dy * dy + dz * dz
            except Exception:
                dist = 0.0
            if dist < best:
                best = dist
                fallback = real
        return fallback

    # Current original-face set per group, for the proven-safe delete patch.
    current_original_keys_by_group = []
    for sj, sj_sm in enumerate(submeshes):
        keys = set()
        if preserve_original_display_lists:
            current_original_keys_by_group.append({_tri_key(face) for face in sj_sm.get('faces', [])})
            continue
        for tri, g in zip(triangles, tri_groups):
            if int(g) != sj:
                continue
            local = []
            ok = True
            for vid in tri:
                src = _template_src(vertex_src[vid] if 0 <= vid < len(vertex_src) else None)
                if src is None:
                    ok = False
                    break
                sv_start, src_idx, src_layout = src
                if sv_start != int(sj_sm['v_start']) or src_layout != str(sj_sm['layout']):
                    ok = False
                    break
                local.append(src_idx)
            if ok and len(set(local)) == 3:
                keys.add(_tri_key(tuple(local)))
        current_original_keys_by_group.append(keys)

    virtual_local: dict[int, int] = {}
    new_records = bytearray()
    for vid in sorted(virtual_refs):
        tmpl = _template_for_target_stream(vid)
        if tmpl is None:
            raise ValueError('New vertex missing template source')
        t_v_start, t_idx, t_layout = tmpl
        virtual_local[vid] = orig_v_count + len(virtual_local)
        rec_off = old_v_start + t_idx * stride
        rec = bytearray(D[rec_off:rec_off + stride])
        if len(rec) != stride:
            raise ValueError('Could not copy native source vertex record')
        pos = vertices[vid] if vid < len(vertices) else (0, 0, 0)
        nrm = normals[vid] if vid < len(normals) else None
        uv = uvs[vid] if vid < len(uvs) else None
        wts = weights[vid] if vid < len(weights) else None
        new_records.extend(_patch_native_vertex_record(rec, layout, pos, nrm, uv, wts))

    appended_local_tris: list[tuple[int, int, int]] = []
    appended_record_tris: list[tuple[dict, dict, dict]] = []
    for tri, g in zip(triangles, tri_groups):
        if not any(v in virtual_refs for v in tri):
            continue
        local = []
        record_tri = []
        for vid in tri:
            if vid in virtual_local:
                local_idx = virtual_local[vid]
                local.append(local_idx)
                record_tri.append({'idx': local_idx, 'new_vertex': True})
            else:
                src = _template_src(vertex_src[vid] if 0 <= vid < len(vertex_src) else None)
                if src is None:
                    raise ValueError('New face references a vertex that is neither original nor virtual')
                sv_start, src_idx, src_layout = src
                if sv_start != old_v_start or src_layout != layout:
                    raise ValueError('New face crosses BDG submesh streams; refusing save')
                local.append(src_idx)
                record_tri.append({'idx': src_idx, 'template': template_records.get(src_idx)})
        if len(set(local)) == 3:
            appended_local_tris.append(tuple(local))
            appended_record_tris.append(tuple(record_tri))
    if not appended_local_tris:
        raise ValueError('Add/extrude save found new vertices but no non-degenerate new faces')
    new_dl_raw = _build_triangle_record_commands(appended_record_tris, index_width)

    out = bytearray(original)
    # Keep delete behavior byte-for-byte safe first.
    for sj, sj_sm in enumerate(submeshes):
        original_keys = {_tri_key(face) for face in sj_sm.get('faces', [])}
        current_keys = current_original_keys_by_group[sj]
        if current_keys != original_keys:
            patched = _patch_display_list_in_place(D, sj_sm, current_keys, int(sj_sm.get('index_width', 6)), original_keys)
            out[int(sj_sm['dl_start']):int(sj_sm['dl_start']) + len(patched)] = patched

    # Replace the touched display-list region up to the old vertex stream.  This
    # removes any old padding that would otherwise sit before the appended command.
    old_region_len = old_v_start - old_dl_start
    last_cmd_end, parsed_records, parse_err = _valid_command_end(D, old_dl_start, old_v_start, index_width)
    if parse_err:
        raise ValueError(f'Original touched display list is not clean enough to grow safely: {parse_err}')
    command_prefix = bytes(out[old_dl_start:last_cmd_end])
    new_v_start = _align(last_cmd_end + len(new_dl_raw), 0x20)
    new_region = command_prefix + new_dl_raw + (b'\x00' * (new_v_start - (last_cmd_end + len(new_dl_raw))))
    _patched_last, touched_record_count, patched_count_err = _valid_command_end(new_region, 0, len(new_region), index_width)
    if patched_count_err:
        raise ValueError(f'Rebuilt touched display list failed local validation: {patched_count_err}')
    dl_delta = len(new_region) - old_region_len
    out[old_dl_start:old_v_start] = new_region

    shifted_old_v_end = old_v_end + dl_delta
    # Preserve 0x20 alignment of every following mesh stream/resource.  Original
    # later stream offsets are 0x20-aligned, so the total inserted byte count
    # must also be a multiple of 0x20.  Aligning only the touched vertex-stream
    # end is not enough because these streams often have a small original gap
    # before the next display-list island.
    vertex_pad_len = (- (dl_delta + len(new_records))) % 0x20
    vertex_insert = bytes(new_records) + (b'\x00' * vertex_pad_len)
    out[shifted_old_v_end:shifted_old_v_end] = vertex_insert
    vertex_delta = len(vertex_insert)
    total_delta = dl_delta + vertex_delta

    # Patch TOC resource size and later resource offsets.
    mesh_toc = int(mesh_entry['toc_entry_offset'])
    _p32(out, mesh_toc + 14, int(mesh_entry['size']) + total_delta, endian)
    for e in entries:
        if not e.get('is_resource') or e is mesh_entry:
            continue
        if int(e['offset']) > int(mesh_entry['offset']):
            toc = int(e['toc_entry_offset'])
            old_rel = _u32(original, toc + 10, endian)
            _p32(out, toc + 10, old_rel + total_delta, endian)

    # Patch descriptors.  Offsets shift when they live after the replaced DL gap
    # and again when they live after the appended vertex records.
    new_record_count = len(appended_local_tris) * 3
    new_descs_for_validation = []
    for sj, sj_desc in enumerate(descs):
        base = int(sj_desc['base'])
        rel_dl_j = int(sj_desc['rel_dl'])
        rel_v_j = int(sj_desc['rel_v'])
        abs_dl_j = res_start + rel_dl_j
        abs_v_j = res_start + rel_v_j
        dl_shift = 0
        v_shift = 0
        if abs_dl_j >= old_v_start:
            dl_shift += dl_delta
        if abs_dl_j >= old_v_end:
            dl_shift += vertex_delta
        if abs_v_j >= old_v_start:
            v_shift += dl_delta
        if abs_v_j >= old_v_end:
            v_shift += vertex_delta
        nd = dict(sj_desc)
        if sj == gi:
            new_rel_v = new_v_start - res_start
            new_dl_size = new_rel_v - int(sj_desc['rel_dl'])
            _p32(out, base + 0, touched_record_count, endian)
            _p32(out, base + 12, new_dl_size, endian)
            _p32(out, base + 32, int(sj_desc['v_count']) + len(virtual_local), endian)
            _p32(out, base + 48, new_rel_v, endian)
            _p32(out, base + 52, int(sj_desc['v_size']) + len(new_records), endian)
            nd['record_count'] = touched_record_count
            nd['dl_size'] = new_dl_size
            nd['v_count'] = int(sj_desc['v_count']) + len(virtual_local)
            nd['rel_v'] = new_rel_v
            nd['v_size'] = int(sj_desc['v_size']) + len(new_records)
        else:
            if dl_shift:
                _p32(out, base + 8, rel_dl_j + dl_shift, endian)
                nd['rel_dl'] = rel_dl_j + dl_shift
            if v_shift:
                _p32(out, base + 48, rel_v_j + v_shift, endian)
                nd['rel_v'] = rel_v_j + v_shift
        new_descs_for_validation.append(nd)

    cand = bytes(out)

    ok_resources, resource_msg = _validate_non_mesh_resources_preserved(
        original, cand, entries, mesh_entry, total_delta
    )
    if not ok_resources:
        text = '\n'.join([
            'BDG add/extrude grow validation failed; original file was left unchanged.',
            'writer_version=V19_aligned_descriptor_stream',
            f'touched_submesh={gi}',
            f'new_vertices={len(virtual_local)}',
            f'new_faces={len(appended_local_tris)}',
            f'dl_delta={dl_delta}',
            f'vertex_delta={vertex_delta}',
            f'total_delta={total_delta}',
            f'resource_validation={resource_msg}',
            f'original_file_size={len(original)} candidate_file_size={len(cand)}',
        ])
        _write_candidate_debug(path, text, cand)
        raise ValueError('Add/extrude save was refused: non-mesh resource bytes changed unexpectedly. Original file was left unchanged.')

    # Strictly validate the game-facing descriptor streams, not just the editor's
    # permissive mesh finder.  This is the guard that catches 0xff/0x54/0xc1
    # opcode situations before they are written.
    fake_mesh = dict(mesh_entry)
    fake_mesh['size'] = int(mesh_entry['size']) + total_delta
    ok_strict, strict_msg = _strict_validate_descriptor_streams(cand, main_entry, fake_mesh, submeshes, new_descs_for_validation, endian)

    tmp = path.with_name(path.stem + '_V19_VALIDATE_TMP.BDG')
    try:
        tmp.write_bytes(cand)
        ok_reload, reload_msg = _validate_candidate(tmp, len(submeshes))
    finally:
        try:
            tmp.unlink()
        except Exception:
            pass

    if not ok_strict:
        text = '\n'.join([
            'BDG add/extrude grow validation failed; original file was left unchanged.',
            'writer_version=V19_aligned_descriptor_stream',
            f'touched_submesh={gi}',
            f'new_vertices={len(virtual_local)}',
            f'new_faces={len(appended_local_tris)}',
            f'new_dl_bytes={len(new_dl_raw)}',
            f'dl_delta={dl_delta}',
            f'vertex_delta={vertex_delta}',
            f'total_delta={total_delta}',
            f'strict_validation={strict_msg}',
            f'editor_reload_validation={reload_msg}',
            f'original_file_size={len(original)} candidate_file_size={len(cand)}',
        ])
        _write_candidate_debug(path, text, cand)
        raise ValueError('Add/extrude save was refused: rebuilt BDG failed strict stream validation. Diagnostic files were written next to the BDG.')

    path.write_bytes(cand)

    if not ok_reload:
        text = '\n'.join([
            'BDG add/extrude grow strict validation passed and file was written.',
            'writer_version=V19_aligned_descriptor_stream',
            'note=editor reload validation is advisory; strict GX stream validation is authoritative for save',
            f'touched_submesh={gi}',
            f'new_vertices={len(virtual_local)}',
            f'new_faces={len(appended_local_tris)}',
            f'new_dl_bytes={len(new_dl_raw)}',
            f'dl_delta={dl_delta}',
            f'vertex_delta={vertex_delta}',
            f'total_delta={total_delta}',
            f'strict_validation={strict_msg}',
            f'editor_reload_validation={reload_msg}',
            f'original_file_size={len(original)} candidate_file_size={len(cand)}',
        ])
        _write_candidate_debug(path, text, None)


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------
_safe_delete_entry_point = save_topology_payload_to_bdg

def _virtual_groups_in_payload(payload: dict[str, Any]) -> set[int]:
    triangles = [tuple(t) for t in (payload.get('triangles') or [])]
    tri_groups = list(payload.get('tri_groups') or [])
    vertex_src = list(payload.get('vertex_src') or [])
    if len(tri_groups) != len(triangles):
        tri_groups = [tri_groups[i] if i < len(tri_groups) else 0 for i in range(len(triangles))]
    groups: set[int] = set()
    for tri, group in zip(triangles, tri_groups):
        has_virtual = False
        for vid in tri:
            if not isinstance(vid, int) or not (0 <= vid < len(vertex_src)):
                continue
            src = vertex_src[vid]
            if len(src) >= 3 and str(src[2]) == 'virtual':
                has_virtual = True
                break
        if has_virtual:
            groups.add(int(group))
    return groups

def _payload_for_virtual_group(payload: dict[str, Any], keep_group: int) -> dict[str, Any]:
    triangles = [tuple(t) for t in (payload.get('triangles') or [])]
    tri_groups = list(payload.get('tri_groups') or [])
    vertex_src = list(payload.get('vertex_src') or [])
    if len(tri_groups) != len(triangles):
        tri_groups = [tri_groups[i] if i < len(tri_groups) else 0 for i in range(len(triangles))]
    keep_triangles = []
    keep_groups = []
    for tri, group in zip(triangles, tri_groups):
        has_virtual = any(
            isinstance(vid, int)
            and 0 <= vid < len(vertex_src)
            and len(vertex_src[vid]) >= 3
            and str(vertex_src[vid][2]) == 'virtual'
            for vid in tri
        )
        if has_virtual and int(group) != int(keep_group):
            continue
        keep_triangles.append(tri)
        keep_groups.append(group)
    out = dict(payload)
    out['triangles'] = keep_triangles
    out['tri_groups'] = keep_groups
    return out

def _retarget_payload_stream_offsets(payload: dict[str, Any], original_streams, current_submeshes) -> dict[str, Any]:
    current_by_group = {
        i: (int(sm['v_start']), str(sm['layout']))
        for i, sm in enumerate(current_submeshes)
    }
    stream_to_group = {
        (int(v_start), str(layout)): i
        for i, (v_start, layout) in enumerate(original_streams)
    }

    def remap_template(tmpl):
        try:
            v_start, idx, layout = int(tmpl[0]), int(tmpl[1]), str(tmpl[2])
        except Exception:
            return tmpl
        group = stream_to_group.get((v_start, layout))
        if group is None or group not in current_by_group:
            return tmpl
        new_v_start, new_layout = current_by_group[group]
        if len(tmpl) >= 4:
            return (new_v_start, idx, new_layout, tmpl[3])
        return (new_v_start, idx, new_layout)

    remapped = []
    for src in list(payload.get('vertex_src') or []):
        try:
            if len(src) >= 3 and str(src[2]) == 'virtual':
                if len(src) >= 5:
                    remapped.append((src[0], src[1], src[2], remap_template(src[3]), src[4]))
                elif len(src) >= 4:
                    remapped.append((src[0], src[1], src[2], remap_template(src[3])))
                else:
                    remapped.append(src)
                continue
            tmpl = remap_template(src)
            remapped.append(tmpl if tmpl is not src else src)
        except Exception:
            remapped.append(src)
    out = dict(payload)
    out['vertex_src'] = remapped
    return out

def save_topology_payload_to_bdg(shape_path: str | Path, payload: dict[str, Any]):
    triangles = [tuple(t) for t in (payload.get('triangles') or [])]
    vertex_src = list(payload.get('vertex_src') or [])
    referenced = {v for tri in triangles for v in tri if isinstance(v, int)}
    has_virtual = False
    for vid in referenced:
        try:
            src = vertex_src[vid]
            if len(src) >= 3 and str(src[2]) == 'virtual':
                has_virtual = True
                break
        except Exception:
            raise ValueError(
                'This edit references a vertex without a stable original BDG source. Refusing save to avoid corrupting the model.'
            )
    if has_virtual:
        groups = _virtual_groups_in_payload(payload)
        if len(groups) > 1:
            D, submeshes = _get_submeshes(shape_path)
            original_streams = [(int(sm['v_start']), str(sm['layout'])) for sm in submeshes]
            order = sorted(
                groups,
                key=lambda gi: int(submeshes[gi]['v_start']) if 0 <= gi < len(submeshes) else -1,
                reverse=True,
            )
            for group in order:
                _D_current, current_submeshes = _get_submeshes(shape_path)
                group_payload = _payload_for_virtual_group(payload, group)
                group_payload = _retarget_payload_stream_offsets(group_payload, original_streams, current_submeshes)
                _save_added_topology_grow_v19(shape_path, group_payload)
            return None
        return _save_added_topology_grow_v19(shape_path, payload)
    return _safe_delete_entry_point(shape_path, payload)
