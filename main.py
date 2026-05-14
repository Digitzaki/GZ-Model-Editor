#!/usr/bin/env python3
"""PyQt6 viewer for Unleashed Shapes.BDG models."""
from __future__ import annotations

import os
import re
import sys
import json
import math
import struct
import shutil
import subprocess
from pathlib import Path

CONFIG_PATH = Path.home() / '.gzme_config.json'


def load_config() -> dict:
    try:
        return json.loads(CONFIG_PATH.read_text())
    except Exception:
        return {}


def save_config(data: dict) -> None:
    try:
        CONFIG_PATH.write_text(json.dumps(data, indent=2))
    except Exception:
        pass

from PyQt6.QtCore import Qt, QFileSystemWatcher, QTimer, QPoint, QSize
from PyQt6.QtGui import (
    QAction, QActionGroup, QColor, QImage, QKeySequence, QPainter, QPen,
    QPixmap, QShortcut, QSurfaceFormat,
)
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QFileDialog, QHBoxLayout,
    QLabel, QMainWindow, QMenu, QMenuBar, QMessageBox, QPushButton,
    QStatusBar, QToolButton, QVBoxLayout, QWidget,
)
from PyQt6.QtOpenGLWidgets import QOpenGLWidget

from OpenGL import GL
from PIL import Image

TOOL_DIR = Path(__file__).resolve().parent
UTILS_DIR = TOOL_DIR / 'utils'
for p in (TOOL_DIR, UTILS_DIR):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

import bdg_to_fbx_extract_all as bdg
from fbx_to_bdg_import import encode_cmpr_png, encode_rgb565_png, encode_i8_png
from parser_core import PipeworksParser
from themes import THEMES, get_theme
from edit_utils import EditModeMixin, CameraMixin

TEXTURE_SUFFIXES = {
    'M': 'Critical Mass Texture',
    'B': 'Bump Map Texture',
    'C': 'Monster Texture',
    'S': 'Decal/Shader Texture',
}

SIZE_TO_FORMAT = {
    0x2aac0: ('CMPR', 512, 512),
    0xaaaa0: ('RGB5A3', 512, 512),
    0x2aaa0: ('RGB5A3', 256, 256),
    0x15560: ('I8', 256, 256),
    0x15600: ('I8', 256, 256),
}


def load_mesh_from_bdg(shape_path: Path):
    D = shape_path.read_bytes()
    _, _, strings = bdg.find_strtab(D)
    _, _, skeleton = bdg.find_skeleton(D, strings)
    bone_count = max(skeleton) + 1

    # BFS from root so children are resolved after their parents.
    bone_globals: list[list[list[float]]] = [None] * bone_count
    bone_parents: list[int] = [-1] * bone_count
    children: dict[int, list[int]] = {}
    root_idx = -1
    for idx, rec in skeleton.items():
        bone_parents[idx] = rec['parent']
        if rec['parent'] < 0:
            root_idx = idx
        else:
            children.setdefault(rec['parent'], []).append(idx)
    if root_idx < 0:
        root_idx = 0
    queue = [root_idx]
    visited = set()
    while queue:
        idx = queue.pop(0)
        if idx in visited:
            continue
        visited.add(idx)
        rec = skeleton.get(idx)
        if not rec:
            continue
        local = bdg.col_local(rec)
        parent = rec['parent']
        if parent < 0 or bone_globals[parent] is None:
            bone_globals[idx] = local
        else:
            bone_globals[idx] = bdg.mm(bone_globals[parent], local)
        for c in children.get(idx, ()):
            queue.append(c)
    bone_positions: list[tuple[float, float, float]] = []
    for i, m in enumerate(bone_globals):
        if m is None:
            bone_positions.append((0.0, 0.0, 0.0))
            bone_parents[i] = -1
        else:
            bone_positions.append((m[0][3], m[1][3], m[2][3]))

    submeshes, _ = bdg.choose_meshes(D, bone_count)

    verts, norms, uvs, tris = [], [], [], []
    src_to_render: dict[tuple[int, int], list[int]] = {}
    vertex_src: list[tuple[int, int, str]] = []
    tri_groups: list[int] = []

    def _read_pos_uv_nrm(off: int, layout: str):
        if layout == 'skin64':
            x, y, z = struct.unpack('>3f', D[off:off + 12])
            u, v = struct.unpack('>2f', D[off + 20:off + 28])
            nx, ny, nz = struct.unpack('>3f', D[off + 28:off + 40])
        elif layout == 'blend76':
            x, y, z = struct.unpack('>3f', D[off:off + 12])
            u, v = struct.unpack('>2f', D[off + 32:off + 40])
            nx, ny, nz = struct.unpack('>3f', D[off + 40:off + 52])
        elif layout in ('blend52', 'blend60'):
            x, y, z = struct.unpack('>3f', D[off:off + 12])
            nx, ny, nz = struct.unpack('>3f', D[off + 32:off + 44])
            u, v = struct.unpack('>2f', D[off + 44:off + 52])
        elif layout == 'skin48':
            x, y, z = struct.unpack('>3f', D[off:off + 12])
            nx, ny, nz = struct.unpack('>3f', D[off + 20:off + 32])
            u, v = struct.unpack('>2f', D[off + 32:off + 40])
        else:  # skin40
            x, y, z = struct.unpack('>3f', D[off:off + 12])
            nx, ny, nz = struct.unpack('>3f', D[off + 20:off + 32])
            u, v = struct.unpack('>2f', D[off + 32:off + 40])
        return (x, y, z), (u, 1.0 - v), (nx, ny, nz)

    for grp_idx, sm in enumerate(submeshes):
        layout = sm['layout']
        for face in sm['faces']:
            ids = []
            for idx in face:
                src_key = (sm['v_start'], idx)
                off = sm['v_start'] + idx * sm['v_stride']
                pos, uv, nrm = _read_pos_uv_nrm(off, layout)
                vid = len(verts)
                ids.append(vid)
                verts.append(tuple(map(float, pos)))
                norms.append(tuple(map(float, nrm)))
                uvs.append((float(uv[0]), 1.0 - float(uv[1])))
                vertex_src.append((sm['v_start'], idx, layout))
                src_to_render.setdefault(src_key, []).append(vid)
            tris.append(tuple(ids))
            tri_groups.append(grp_idx)
    bone_record_offsets = [0] * bone_count
    bone_local_translations = [(0.0, 0.0, 0.0)] * bone_count
    bone_local_quats = [(0.0, 0.0, 0.0, 1.0)] * bone_count
    for i, rec in skeleton.items():
        bone_record_offsets[i] = rec['off']
        bone_local_translations[i] = tuple(rec['t'])
        bone_local_quats[i] = tuple(rec['q'])
    return (verts, norms, uvs, tris, bone_count, bone_positions, bone_parents,
            vertex_src, src_to_render,
            bone_record_offsets, bone_local_translations, bone_local_quats,
            tri_groups)


def find_texture_pairs(shape_path: Path):
    parser = PipeworksParser(str(shape_path))
    files = parser.parse()
    by_num: dict[int, dict] = {}
    for entry in files:
        if entry['file_type'] != 9:
            continue
        slot = by_num.setdefault(entry['file_num'], {})
        slot['resource' if entry['is_resource'] else 'header'] = entry

    pairs = {}
    for slot in by_num.values():
        head = slot.get('header')
        res = slot.get('resource')
        if not head or not res:
            continue
        base = head['name'].rsplit('/', 1)[-1]
        m = re.search(r'_([MBCS])$', base)
        if not m:
            continue
        suffix = m.group(1)
        info = SIZE_TO_FORMAT.get(res['size'])
        if not info:
            continue
        fmt, w, h = info
        pairs[suffix] = {
            'parser': parser,
            'header': head,
            'resource': res,
            'w': w,
            'h': h,
            'fmt': fmt,
        }
    return pairs


def _decode_top_mip(raw: bytes, w: int, h: int, fmt: str) -> Image.Image | None:
    if fmt == 'CMPR':
        return bdg.decode_cmpr(raw[:w * h // 2], w, h)
    if fmt == 'RGB565':
        return bdg.decode_rgb565(raw[:w * h * 2], w, h)
    if fmt == 'RGB5A3':
        from wii_tex_decode import decode_rgb5a3
        return decode_rgb5a3(raw[:w * h * 2], w, h)
    if fmt == 'I8':
        return bdg.decode_i8(raw[:w * h], w, h)
    return None


def decode_texture_image(shape_path: Path, suffix: str) -> tuple[Image.Image | None, str]:
    pairs = find_texture_pairs(shape_path)
    pair = pairs.get(suffix)
    if not pair:
        return None, f'_{suffix} not present'
    raw = pair['parser'].read_bytes(pair['resource']['offset'], pair['resource']['size'])
    img = _decode_top_mip(raw, pair['w'], pair['h'], pair['fmt'])
    if img is None:
        return None, f'unsupported format {pair["fmt"]} for _{suffix}'
    if suffix == 'B':
        lum = img.convert('L')
        img = Image.merge('RGBA', (lum, lum, lum, Image.new('L', lum.size, 255)))
    return img, f'_{suffix} {pair["fmt"]} {pair["w"]}x{pair["h"]}'


def _build_mip_chain(img: Image.Image, fmt: str, original_size: int) -> bytes:
    out = bytearray()
    w, h = img.width, img.height
    cur = img
    while True:
        if fmt == 'CMPR':
            payload = _encode_cmpr_inline(cur)
        elif fmt == 'RGB565':
            payload = _encode_rgb565_inline(cur)
        elif fmt == 'RGB5A3':
            payload = _encode_rgb5a3_inline(cur)
        elif fmt == 'I8':
            payload = _encode_i8_inline(cur)
        else:
            raise ValueError(f'unsupported format {fmt}')
        if len(out) + len(payload) > original_size:
            break
        out.extend(payload)
        if w <= 1 and h <= 1:
            break
        nw = max(1, w // 2)
        nh = max(1, h // 2)
        if nw == w and nh == h:
            break
        cur = cur.resize((nw, nh), Image.LANCZOS)
        w, h = nw, nh
    if len(out) < original_size:
        out.extend(b'\x00' * (original_size - len(out)))
    return bytes(out[:original_size])


def _encode_rgb565_inline(img: Image.Image) -> bytes:
    img = img.convert('RGB')
    w, h = img.size
    px = img.load()
    out = bytearray()
    for ty in range(0, h, 4):
        for tx in range(0, w, 4):
            for y in range(4):
                for x in range(4):
                    xx, yy = tx + x, ty + y
                    r, g, b = px[xx, yy] if (xx < w and yy < h) else (0, 0, 0)
                    v = ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)
                    out += struct.pack('>H', v)
    return bytes(out)


def _encode_rgb5a3_inline(img: Image.Image) -> bytes:
    img = img.convert('RGBA')
    w, h = img.size
    px = img.load()
    out = bytearray()
    for ty in range(0, h, 4):
        for tx in range(0, w, 4):
            for y in range(4):
                for x in range(4):
                    xx, yy = tx + x, ty + y
                    r, g, b, a = px[xx, yy] if (xx < w and yy < h) else (0, 0, 0, 0)
                    if a >= 0xE0:
                        v = 0x8000 | ((r >> 3) << 10) | ((g >> 3) << 5) | (b >> 3)
                    else:
                        v = ((a >> 5) << 12) | ((r >> 4) << 8) | ((g >> 4) << 4) | (b >> 4)
                    out += struct.pack('>H', v)
    return bytes(out)


def _encode_i8_inline(img: Image.Image) -> bytes:
    img = img.convert('L')
    w, h = img.size
    px = img.load()
    out = bytearray()
    for ty in range(0, h, 4):
        for tx in range(0, w, 8):
            for y in range(4):
                for x in range(8):
                    xx, yy = tx + x, ty + y
                    out.append(px[xx, yy] if (xx < w and yy < h) else 0)
    return bytes(out)


def _encode_cmpr_inline(img: Image.Image) -> bytes:
    from fbx_to_bdg_import import dxt1_block_encode
    img = img.convert('RGBA')
    w, h = img.size
    px = img.load()
    out = bytearray()
    for y in range(0, h, 8):
        for x in range(0, w, 8):
            for by, bx in [(0, 0), (0, 4), (4, 0), (4, 4)]:
                block = []
                for py in range(4):
                    for pxn in range(4):
                        sx, sy = x + bx + pxn, y + by + py
                        if sx < w and sy < h:
                            block.append(px[sx, sy])
                        else:
                            block.append((0, 0, 0, 255))
                out += dxt1_block_encode(block)
    return bytes(out)


def _multiply(a: Image.Image, b: Image.Image) -> Image.Image:
    from PIL import ImageChops
    ar, ag, ab, aa = a.convert('RGBA').split()
    br, bg, bb, _ = b.convert('RGBA').split()
    rgb = ImageChops.multiply(Image.merge('RGB', (ar, ag, ab)),
                              Image.merge('RGB', (br, bg, bb)))
    r, g, bl = rgb.split()
    return Image.merge('RGBA', (r, g, bl, aa))


def replace_texture(shape_path: Path, suffix: str, png_path: Path) -> str:
    pairs = find_texture_pairs(shape_path)
    pair = pairs.get(suffix)
    if not pair:
        raise ValueError(f'_{suffix} not present in this BDG')
    fmt = pair['fmt']
    img = Image.open(png_path)
    target_w, target_h = pair['w'], pair['h']
    if (img.width, img.height) != (target_w, target_h):
        img = img.resize((target_w, target_h), Image.LANCZOS)
    payload = _build_mip_chain(img, fmt, pair['resource']['size'])
    pair['parser'].replace_file_bytes(pair['resource'], payload)
    return f'_{suffix} replaced ({fmt} {target_w}x{target_h}, {len(payload)} bytes)'


class MeshViewer(EditModeMixin, CameraMixin, QOpenGLWidget):
    DEFAULT_YAW = 210.0
    DEFAULT_PITCH = 12.0
    DEFAULT_ZOOM = 1.1
    DEFAULT_PAN_X_FRAC = 0.0
    DEFAULT_MODEL_PAN_X_FRAC = 0.0
    DEFAULT_MODEL_PAN_Z_FRAC = 0.0
    SPIN_SPEEDS = (0.0, 0.45, 1.6)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.vertices = []
        self.normals = []
        self.uvs = []
        self.triangles = []
        self.center = (0.0, 0.0, 0.0)
        self.radius = 1.0
        self.yaw = self.DEFAULT_YAW
        self.pitch = self.DEFAULT_PITCH
        self.zoom = self.DEFAULT_ZOOM
        self.pan_x = 0.0
        self.pan_y = 0.0
        # Upright orientation (+90 deg X). Render-only, never written to disk.
        self.model_rot = [
            1.0,  0.0, 0.0,
            0.0,  0.0, 1.0,
            0.0, -1.0, 0.0,
        ]

        self._target_yaw = self.yaw
        self._target_pitch = self.pitch
        self._target_zoom = self.zoom
        self._target_pan_x = 0.0
        self._target_pan_y = 0.0
        self._default_pan_x = 0.0
        self._frame_action_next = 'frame'
        self.model_pan_x = 0.0
        self.model_pan_y = 0.0
        self.model_pan_z = 0.0
        self._yaw_velocity = 0.0
        self._pitch_velocity = 0.0
        self._spin_stage = 0

        self.bone_positions: list[tuple[float, float, float]] = []
        self.bone_parents: list[int] = []
        self.wire_mode = 1
        self.show_wireframe = True
        self.show_bones = False
        self.show_uv_overlay = False
        self._uv_tex_id = 0
        self.clear_color = (0.10, 0.11, 0.13)
        self.light_index = 0

        self.edit_mode = False
        self.edit_target = 'vertex'
        self.selected_verts: set[int] = set()
        self.selected_bones: set[int] = set()
        self.vertex_src: list[tuple[int, int, str]] = []
        self.src_to_render: dict[tuple[int, int], list[int]] = {}
        self.bone_record_offsets: list[int] = []
        self.bone_locals: list[tuple[float, float, float]] = []
        self.bone_quats: list[tuple[float, float, float, float]] = []
        self.bone_locked: set[int] = set()
        self.bone_locals_baseline: list[tuple[float, float, float]] = []
        self._bone_grab_origin_locals: dict[int, tuple[float, float, float]] = {}
        self._marquee_active = False
        self._marquee_start: QPoint | None = None
        self._marquee_end: QPoint | None = None
        self._grab_active = False
        self._grab_start: QPoint | None = None
        self._grab_origin_positions: dict[int, tuple[float, float, float]] = {}
        self._rotate_active = False
        self._rotate_pivot: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._rotate_origin_positions: dict[int, tuple[float, float, float]] = {}
        self._rotate_origin_bone_locals: dict[int, tuple[float, float, float]] = {}
        self._hover_vert: int | None = None
        self._hover_bone: int | None = None
        self._rotate_axis_mode: str = 'view'
        self._gizmo_drag_active = False
        self._hover_ring: str | None = None
        self._hover_arrow: str | None = None
        self._gizmo_mode: str | None = None
        self._gizmo_translate_active = False
        self._gizmo_translate_axis: str | None = None
        self._grab_axis_mode: str | None = None
        self.overlay_mode = 0
        self.show_grid = True
        self._view_gizmo_mode: str | None = None
        self._view_gizmo_drag_active = False
        self._view_gizmo_translate_active = False
        self._view_gizmo_axis: str | None = None
        self._view_drag_start: QPoint | None = None
        self._view_drag_origin_pan = (0.0, 0.0, 0.0)
        self._view_hover_ring: str | None = None
        self._view_hover_arrow: str | None = None
        self._cached_view_matrix: list[float] | None = None
        self._cached_proj_matrix: list[float] | None = None
        self._cached_viewport: tuple[int, int, int, int] | None = None

        self._last_pos: QPoint | None = None
        self._tex_id = 0
        self._pending_tex_image: Image.Image | None = None
        self._clear_texture = False
        self._mesh_list = 0
        self._mesh_dirty = False
        self.setMinimumSize(640, 480)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)

        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(16)
        self._anim_timer.timeout.connect(self._tick)
        self._anim_timer.start()

    def set_mesh(self, vertices, normals, uvs, triangles):
        if hasattr(self, '_cancel_grab'):
            try:
                self._cancel_grab()
            except Exception:
                pass
        for attr in ('_grab_origin_positions', '_bone_grab_origin_locals',
                     '_rotate_origin_positions', '_rotate_origin_bone_locals',
                     '_rotate_origin_bone_quats', '_rotate_origin_parent_world'):
            d = getattr(self, attr, None)
            if isinstance(d, dict):
                d.clear()
        self.selected_verts.clear()
        self.selected_bones.clear()
        self.bone_locked.clear()
        self._hover_vert = None
        self._hover_bone = None
        self._hover_ring = None
        self._hover_arrow = None
        self._gizmo_drag_active = False
        self._gizmo_translate_active = False
        self._gizmo_translate_axis = None
        self._gizmo_mode = None
        self._view_gizmo_mode = None
        self._view_gizmo_drag_active = False
        self._view_gizmo_translate_active = False
        self._marquee_active = False
        self._marquee_start = None
        self._marquee_end = None
        self.vertices = vertices
        self.normals = normals
        self.uvs = uvs
        self.triangles = triangles
        if vertices:
            xs = [v[0] for v in vertices]
            ys = [v[1] for v in vertices]
            zs = [v[2] for v in vertices]
            cx = (min(xs) + max(xs)) * 0.5
            cy = (min(ys) + max(ys)) * 0.5
            cz = (min(zs) + max(zs)) * 0.5
            self.center = (cx, cy, cz)
            self.radius = max(
                math.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2)
                for x, y, z in vertices
            ) or 1.0
            self._default_pan_x = self.radius * self.DEFAULT_PAN_X_FRAC
            self._target_pan_x = self._default_pan_x
            self._target_pan_y = 0.0
            self._target_zoom = self.DEFAULT_ZOOM
            self.model_pan_x = self.radius * self.DEFAULT_MODEL_PAN_X_FRAC
            self.model_pan_y = 0.0
            self.model_pan_z = self.radius * self.DEFAULT_MODEL_PAN_Z_FRAC
            self.model_rot = [
                1.0,  0.0, 0.0,
                0.0,  0.0, 1.0,
                0.0, -1.0, 0.0,
            ]
            self._target_yaw = self.DEFAULT_YAW
            self._target_pitch = self.DEFAULT_PITCH
            self._frame_action_next = 'frame'
        self._mesh_dirty = True
        self.update()

    def set_skeleton(self, positions, parents):
        self.bone_positions = list(positions)
        self.bone_parents = list(parents)
        self.update()

    def set_wireframe(self, on: bool):
        self.show_wireframe = bool(on)
        self.wire_mode = 1 if on else 0
        self.update()

    def cycle_wire_mode(self):
        self.wire_mode = (self.wire_mode + 1) % 3
        self.show_wireframe = self.wire_mode != 0
        self.update()

    def set_show_bones(self, on: bool):
        self.show_bones = bool(on)
        self.update()

    def set_show_uv_overlay(self, on: bool):
        self.show_uv_overlay = bool(on)
        self.update()

    def _ensure_uv_overlay_texture(self):
        if self._uv_tex_id:
            return self._uv_tex_id
        size = 256
        cells = 8
        cell = size // cells
        pixels = bytearray(size * size * 4)
        for y in range(size):
            for x in range(size):
                cx = x // cell
                cy = y // cell
                light = ((cx + cy) & 1) == 0
                base = 220 if light else 150
                r = base
                g = base
                b = base
                u_t = x / (size - 1)
                v_t = y / (size - 1)
                r = int(r * 0.6 + 255 * 0.4 * u_t)
                g = int(g * 0.6 + 255 * 0.4 * v_t)
                b = int(b * 0.7)
                on_grid = (x % cell == 0) or (y % cell == 0)
                if on_grid:
                    r = g = b = 20
                idx = (y * size + x) * 4
                pixels[idx] = r
                pixels[idx + 1] = g
                pixels[idx + 2] = b
                pixels[idx + 3] = 255
        tex_id = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex_id)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, size, size, 0,
            GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, bytes(pixels),
        )
        self._uv_tex_id = int(tex_id)
        return self._uv_tex_id

    LIGHT_PRESETS = [
        ('Neutral fill',   (1.00, 1.00, 1.00), False),
        ('Red spotlight',     (1.00, 0.20, 0.20), True),
        ('Green spotlight',   (0.25, 1.00, 0.30), True),
        ('Blue spotlight',    (0.30, 0.45, 1.00), True),
        ('Purple spotlight',  (0.85, 0.30, 1.00), True),
        ('Amber spotlight',   (1.00, 0.70, 0.25), True),
        ('Cyan spotlight',    (0.30, 1.00, 1.00), True),
    ]

    def set_light_index(self, idx: int):
        self.light_index = idx % len(self.LIGHT_PRESETS)
        self.update()
        return self.light_index

    def cycle_light(self):
        return self.set_light_index(self.light_index + 1)

    def _apply_lighting_state(self):
        name, (r, g, b), is_spot = self.LIGHT_PRESETS[self.light_index]
        if not is_spot:
            GL.glLightfv(GL.GL_LIGHT0, GL.GL_POSITION, (0.5, 1.0, 0.8, 0.0))
            GL.glLightfv(GL.GL_LIGHT0, GL.GL_DIFFUSE, (r, g, b, 1.0))
            GL.glLightfv(GL.GL_LIGHT0, GL.GL_AMBIENT, (0.25, 0.25, 0.28, 1.0))
            GL.glLightf(GL.GL_LIGHT0, GL.GL_SPOT_CUTOFF, 180.0)
            GL.glLightf(GL.GL_LIGHT0, GL.GL_SPOT_EXPONENT, 0.0)
            GL.glLightf(GL.GL_LIGHT0, GL.GL_CONSTANT_ATTENUATION, 1.0)
            GL.glLightf(GL.GL_LIGHT0, GL.GL_LINEAR_ATTENUATION, 0.0)
            GL.glLightf(GL.GL_LIGHT0, GL.GL_QUADRATIC_ATTENUATION, 0.0)
            return
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_POSITION, (0.0, 0.0, 0.0, 1.0))
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_SPOT_DIRECTION, (0.0, 0.0, -1.0))
        GL.glLightf(GL.GL_LIGHT0, GL.GL_SPOT_CUTOFF, 35.0)
        GL.glLightf(GL.GL_LIGHT0, GL.GL_SPOT_EXPONENT, 6.0)
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_DIFFUSE, (r * 1.4, g * 1.4, b * 1.4, 1.0))
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_SPECULAR, (r, g, b, 1.0))
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_AMBIENT,
                     (0.20 * r + 0.10, 0.20 * g + 0.10, 0.20 * b + 0.10, 1.0))
        GL.glLightf(GL.GL_LIGHT0, GL.GL_CONSTANT_ATTENUATION, 1.0)
        GL.glLightf(GL.GL_LIGHT0, GL.GL_LINEAR_ATTENUATION, 0.0)
        GL.glLightf(GL.GL_LIGHT0, GL.GL_QUADRATIC_ATTENUATION, 0.0)

    def set_clear_color(self, rgb):
        self.clear_color = tuple(rgb)
        if self.isValid():
            self.makeCurrent()
            GL.glClearColor(*self.clear_color, 1.0)
            self.doneCurrent()
        self.update()

    def set_texture_image(self, img: Image.Image | None):
        if img is None:
            self._clear_texture = True
            self._pending_tex_image = None
        else:
            self._pending_tex_image = img.convert('RGBA')
            self._clear_texture = False
        self.update()

    def initializeGL(self):
        GL.glClearColor(*self.clear_color, 1.0)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_LIGHTING)
        GL.glEnable(GL.GL_LIGHT0)
        GL.glEnable(GL.GL_NORMALIZE)
        GL.glEnable(GL.GL_COLOR_MATERIAL)
        GL.glColorMaterial(GL.GL_FRONT_AND_BACK, GL.GL_AMBIENT_AND_DIFFUSE)
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_POSITION, (0.5, 1.0, 0.8, 0.0))
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_DIFFUSE, (1.0, 1.0, 1.0, 1.0))
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_AMBIENT, (0.25, 0.25, 0.28, 1.0))
        GL.glShadeModel(GL.GL_SMOOTH)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glEnable(GL.GL_ALPHA_TEST)
        GL.glAlphaFunc(GL.GL_GREATER, 0.02)

    def _upload_pending_texture(self):
        if self._clear_texture:
            if self._tex_id:
                GL.glDeleteTextures([self._tex_id])
                self._tex_id = 0
            self._clear_texture = False
        if self._pending_tex_image is None:
            return
        img = self._pending_tex_image
        self._pending_tex_image = None
        if self._tex_id:
            GL.glDeleteTextures([self._tex_id])
        self._tex_id = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._tex_id)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, img.width, img.height, 0,
            GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, img.tobytes(),
        )

    def resizeGL(self, w, h):
        GL.glViewport(0, 0, max(1, w), max(1, h))

    def _rebuild_mesh_list(self):
        if self._mesh_list:
            GL.glDeleteLists(self._mesh_list, 1)
            self._mesh_list = 0
        if not self.triangles:
            return
        self._mesh_list = GL.glGenLists(1)
        GL.glNewList(self._mesh_list, GL.GL_COMPILE)
        GL.glBegin(GL.GL_TRIANGLES)
        has_uvs = bool(self.uvs)
        for a, b, c in self.triangles:
            for idx in (a, b, c):
                nx, ny, nz = self.normals[idx]
                GL.glNormal3f(nx, ny, nz)
                if has_uvs:
                    u, v = self.uvs[idx]
                    GL.glTexCoord2f(u, v)
                vx, vy, vz = self.vertices[idx]
                GL.glVertex3f(vx, vy, vz)
        GL.glEnd()
        GL.glEndList()

    def _draw_grid_and_axes(self):
        GL.glDisable(GL.GL_LIGHTING)
        GL.glDisable(GL.GL_TEXTURE_2D)
        extent = max(self.radius * 4.0, 4.0)
        step = max(self.radius * 0.25, 0.25)
        order = 10 ** math.floor(math.log10(step)) if step > 0 else 1.0
        step = order * (1 if step / order < 2 else (2 if step / order < 5 else 5))
        n = int(extent / step) + 1
        gy = -self.radius * 1.1

        GL.glLineWidth(1.0)
        GL.glBegin(GL.GL_LINES)
        GL.glColor3f(0.28, 0.30, 0.34)
        skip_center = self.overlay_mode == 0
        for i in range(-n, n + 1):
            if i == 0 and skip_center:
                continue
            x = i * step
            GL.glVertex3f(x, gy, -extent); GL.glVertex3f(x, gy, extent)
            GL.glVertex3f(-extent, gy, x); GL.glVertex3f(extent, gy, x)
        GL.glEnd()

        if self.overlay_mode == 0:
            GL.glLineWidth(1.6)
            GL.glBegin(GL.GL_LINES)
            GL.glColor3f(0.85, 0.25, 0.30)
            GL.glVertex3f(-extent, gy, 0.0); GL.glVertex3f(extent, gy, 0.0)
            GL.glColor3f(0.25, 0.50, 0.90)
            GL.glVertex3f(0.0, gy, -extent); GL.glVertex3f(0.0, gy, extent)
            GL.glColor3f(0.40, 0.80, 0.30)
            GL.glVertex3f(0.0, gy, 0.0); GL.glVertex3f(0.0, gy + extent, 0.0)
            GL.glEnd()
        GL.glLineWidth(1.0)

    def paintGL(self):
        self._upload_pending_texture()
        if self._mesh_dirty:
            self._rebuild_mesh_list()
            self._mesh_dirty = False
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        w = max(1, self.width())
        h = max(1, self.height())
        aspect = w / h

        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        fov = math.radians(45.0)
        near = max(self.radius * 0.01, 0.01)
        far = self.radius * 50.0 + 10.0
        f = 1.0 / math.tan(fov / 2.0)
        proj = [
            f / aspect, 0, 0, 0,
            0, f, 0, 0,
            0, 0, (far + near) / (near - far), -1,
            0, 0, (2 * far * near) / (near - far), 0,
        ]
        GL.glLoadMatrixf(proj)

        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()
        self._apply_lighting_state()
        dist = self.radius * 2.5 * self.zoom
        GL.glTranslatef(self.pan_x, self.pan_y, -dist)
        GL.glRotatef(self.pitch, 1.0, 0.0, 0.0)
        GL.glRotatef(self.yaw, 0.0, 1.0, 0.0)
        if self.show_grid and self.overlay_mode != 2:
            cx_w, cy_w, cz_w = self.center
            GL.glPushMatrix()
            GL.glTranslatef(-cx_w, -cy_w, -cz_w)
            self._draw_grid_and_axes()
            GL.glPopMatrix()
            GL.glEnable(GL.GL_LIGHTING)
        try:
            self._cached_world_mv = list(GL.glGetFloatv(GL.GL_MODELVIEW_MATRIX).flatten())
        except Exception:
            self._cached_world_mv = None
        GL.glTranslatef(self.model_pan_x, self.model_pan_y, self.model_pan_z)
        m = self.model_rot
        gl_mat = [
            m[0], m[3], m[6], 0.0,
            m[1], m[4], m[7], 0.0,
            m[2], m[5], m[8], 0.0,
            0.0,  0.0,  0.0,  1.0,
        ]
        GL.glMultMatrixf(gl_mat)
        cx, cy, cz = self.center
        GL.glTranslatef(-cx, -cy, -cz)

        try:
            self._cached_view_matrix = list(GL.glGetFloatv(GL.GL_MODELVIEW_MATRIX).flatten())
            self._cached_proj_matrix = list(GL.glGetFloatv(GL.GL_PROJECTION_MATRIX).flatten())
        except Exception:
            self._cached_view_matrix = None
            self._cached_proj_matrix = None
        self._cached_viewport = (0, 0, w, h)

        if not self.triangles:
            return

        if self.show_uv_overlay and self.uvs:
            tex = self._ensure_uv_overlay_texture()
            GL.glEnable(GL.GL_TEXTURE_2D)
            GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
            GL.glColor3f(1.0, 1.0, 1.0)
            textured = True
        else:
            textured = self._tex_id != 0 and self.uvs
            if textured:
                GL.glEnable(GL.GL_TEXTURE_2D)
                GL.glBindTexture(GL.GL_TEXTURE_2D, self._tex_id)
                GL.glColor3f(1.0, 1.0, 1.0)
            else:
                GL.glColor3f(0.62, 0.64, 0.68)

        if self._mesh_list:
            GL.glCallList(self._mesh_list)

        if textured:
            GL.glDisable(GL.GL_TEXTURE_2D)

        if self.wire_mode != 0 and self._mesh_list:
            GL.glDisable(GL.GL_LIGHTING)
            GL.glDisable(GL.GL_TEXTURE_2D)
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
            GL.glLineWidth(1.0)
            GL.glEnable(GL.GL_POLYGON_OFFSET_LINE)
            GL.glPolygonOffset(-1.0, -1.0)
            if self.wire_mode == 2:
                GL.glColor4f(0.05, 0.95, 0.55, 0.9)
            else:
                GL.glColor4f(0.02, 0.03, 0.04, 0.95)
            GL.glCallList(self._mesh_list)
            GL.glDisable(GL.GL_POLYGON_OFFSET_LINE)
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
            GL.glEnable(GL.GL_LIGHTING)

        bones_visible = self.show_bones and self.bone_positions and not (
            self.edit_mode and self.edit_target == 'vertex'
        )
        if bones_visible:
            GL.glDisable(GL.GL_LIGHTING)
            GL.glDisable(GL.GL_DEPTH_TEST)
            GL.glDisable(GL.GL_TEXTURE_2D)
            GL.glLineWidth(2.0)
            GL.glColor3f(1.0, 0.85, 0.2)
            GL.glBegin(GL.GL_LINES)
            for i, parent in enumerate(self.bone_parents):
                if parent is None or parent < 0 or parent >= len(self.bone_positions):
                    continue
                px, py, pz = self.bone_positions[parent]
                cx_, cy_, cz_ = self.bone_positions[i]
                GL.glVertex3f(px, py, pz)
                GL.glVertex3f(cx_, cy_, cz_)
            GL.glEnd()
            GL.glPointSize(5.0)
            GL.glColor3f(1.0, 0.4, 0.2)
            GL.glBegin(GL.GL_POINTS)
            for x_, y_, z_ in self.bone_positions:
                GL.glVertex3f(x_, y_, z_)
            GL.glEnd()
            GL.glEnable(GL.GL_DEPTH_TEST)
            GL.glEnable(GL.GL_LIGHTING)

        if self.edit_mode:
            self._depth_buffer = self._read_depth_snapshot(w, h)
            GL.glDisable(GL.GL_LIGHTING)
            GL.glDisable(GL.GL_TEXTURE_2D)
            if self.edit_target == 'vertex':
                GL.glDisable(GL.GL_DEPTH_TEST)
                visible = self._visible_source_keys()
                GL.glPointSize(3.0)
                GL.glColor3f(0.55, 0.85, 1.0)
                GL.glBegin(GL.GL_POINTS)
                seen = set()
                for vid, src in enumerate(self.vertex_src):
                    key = (src[0], src[1])
                    if key in seen or vid in self.selected_verts:
                        continue
                    seen.add(key)
                    if visible and key not in visible:
                        continue
                    vx, vy, vz = self.vertices[vid]
                    GL.glVertex3f(vx, vy, vz)
                GL.glEnd()
                if self.selected_verts:
                    GL.glPointSize(7.0)
                    GL.glColor3f(1.0, 0.55, 0.15)
                    GL.glBegin(GL.GL_POINTS)
                    for vid in self.selected_verts:
                        if 0 <= vid < len(self.vertices):
                            vx, vy, vz = self.vertices[vid]
                            GL.glVertex3f(vx, vy, vz)
                    GL.glEnd()
                if self._hover_vert is not None and 0 <= self._hover_vert < len(self.vertices):
                    GL.glPointSize(11.0)
                    GL.glColor3f(1.0, 1.0, 0.6)
                    GL.glBegin(GL.GL_POINTS)
                    vx, vy, vz = self.vertices[self._hover_vert]
                    GL.glVertex3f(vx, vy, vz)
                    GL.glEnd()
            else:  # bone target
                GL.glDisable(GL.GL_DEPTH_TEST)
                GL.glPointSize(6.0)
                GL.glColor3f(0.95, 0.85, 0.25)
                GL.glBegin(GL.GL_POINTS)
                for i, p in enumerate(self.bone_positions):
                    if i in self.selected_bones or i in self.bone_locked:
                        continue
                    GL.glVertex3f(p[0], p[1], p[2])
                GL.glEnd()
                if self.bone_locked:
                    GL.glPointSize(8.0)
                    GL.glColor3f(0.45, 0.75, 1.0)
                    GL.glBegin(GL.GL_POINTS)
                    for bid in self.bone_locked:
                        if bid in self.selected_bones:
                            continue
                        if 0 <= bid < len(self.bone_positions):
                            p = self.bone_positions[bid]
                            GL.glVertex3f(p[0], p[1], p[2])
                    GL.glEnd()
                if self.selected_bones:
                    GL.glPointSize(10.0)
                    GL.glColor3f(1.0, 0.4, 0.15)
                    GL.glBegin(GL.GL_POINTS)
                    for bid in self.selected_bones:
                        if 0 <= bid < len(self.bone_positions):
                            p = self.bone_positions[bid]
                            GL.glVertex3f(p[0], p[1], p[2])
                    GL.glEnd()
                if self._hover_bone is not None and 0 <= self._hover_bone < len(self.bone_positions):
                    GL.glPointSize(13.0)
                    GL.glColor3f(1.0, 1.0, 0.6)
                    GL.glBegin(GL.GL_POINTS)
                    p = self.bone_positions[self._hover_bone]
                    GL.glVertex3f(p[0], p[1], p[2])
                    GL.glEnd()
                GL.glLineWidth(1.5)
                GL.glColor3f(0.7, 0.6, 0.2)
                GL.glBegin(GL.GL_LINES)
                for i, parent in enumerate(self.bone_parents):
                    if parent is None or parent < 0 or parent >= len(self.bone_positions):
                        continue
                    px, py, pz = self.bone_positions[parent]
                    cx_, cy_, cz_ = self.bone_positions[i]
                    GL.glVertex3f(px, py, pz)
                    GL.glVertex3f(cx_, cy_, cz_)
                GL.glEnd()
            self._draw_rotate_gizmo()
            self._draw_translate_gizmo()
            GL.glEnable(GL.GL_DEPTH_TEST)
            GL.glEnable(GL.GL_LIGHTING)
            self._draw_marquee_overlay(w, h)
        else:
            if self._view_gizmo_mode in ('rotate', 'translate'):
                GL.glDisable(GL.GL_LIGHTING)
                GL.glDisable(GL.GL_DEPTH_TEST)
                GL.glDisable(GL.GL_TEXTURE_2D)
                if getattr(self, '_cached_world_mv', None):
                    GL.glPushMatrix()
                    GL.glLoadMatrixf(self._cached_world_mv)
                    GL.glTranslatef(self.model_pan_x, self.model_pan_y, self.model_pan_z)
                    if self._view_gizmo_mode == 'rotate':
                        self._draw_view_rotate_gizmo()
                    else:
                        self._draw_view_translate_gizmo()
                    GL.glPopMatrix()
                GL.glEnable(GL.GL_DEPTH_TEST)
                GL.glEnable(GL.GL_LIGHTING)

    def _draw_rotate_gizmo(self):
        info = self.gizmo_pivot_and_radius() if hasattr(self, 'gizmo_pivot_and_radius') else None
        if info is None:
            return
        (px, py, pz), radius = info
        GL.glDisable(GL.GL_LIGHTING)
        GL.glDisable(GL.GL_TEXTURE_2D)
        GL.glLineWidth(2.5)
        axis_colors = (
            ('x', (1.0, 0.30, 0.30)),
            ('y', (0.30, 1.0, 0.40)),
            ('z', (0.30, 0.55, 1.0)),
        )
        active = self._rotate_axis_mode if self._rotate_active else None
        hovered = self._hover_ring
        steps = 64
        for axis_name, (r, g, b) in axis_colors:
            if active == axis_name or hovered == axis_name:
                GL.glColor3f(min(1.0, r + 0.2), min(1.0, g + 0.2), min(1.0, b + 0.2))
                GL.glLineWidth(4.0)
            else:
                dim = 0.55 if active in ('x', 'y', 'z') else 0.85
                GL.glColor3f(r * dim, g * dim, b * dim)
                GL.glLineWidth(2.0)
            GL.glBegin(GL.GL_LINE_LOOP)
            for i in range(steps):
                t = (i / steps) * math.tau
                ct = math.cos(t) * radius
                st = math.sin(t) * radius
                if axis_name == 'x':
                    GL.glVertex3f(px, py + ct, pz + st)
                elif axis_name == 'y':
                    GL.glVertex3f(px + ct, py, pz + st)
                else:
                    GL.glVertex3f(px + ct, py + st, pz)
            GL.glEnd()
        GL.glLineWidth(1.0)

    def _draw_translate_gizmo(self):
        info = self.gizmo_translate_info() if hasattr(self, 'gizmo_translate_info') else None
        if info is None:
            return
        (px, py, pz), length = info
        GL.glDisable(GL.GL_LIGHTING)
        GL.glDisable(GL.GL_TEXTURE_2D)
        axis_colors = (
            ('x', (1.0, 0.30, 0.30), (length, 0.0, 0.0)),
            ('y', (0.30, 1.0, 0.40), (0.0, length, 0.0)),
            ('z', (0.30, 0.55, 1.0), (0.0, 0.0, length)),
        )
        head = length * 0.18
        active = self._gizmo_translate_axis if self._gizmo_translate_active else None
        hovered = self._hover_arrow
        for axis_name, (r, g, b), (dx, dy, dz) in axis_colors:
            if active == axis_name or hovered == axis_name:
                GL.glColor3f(min(1.0, r + 0.2), min(1.0, g + 0.2), min(1.0, b + 0.2))
                GL.glLineWidth(4.0)
            else:
                dim = 0.55 if active in ('x', 'y', 'z') else 0.95
                GL.glColor3f(r * dim, g * dim, b * dim)
                GL.glLineWidth(3.0)
            GL.glBegin(GL.GL_LINES)
            GL.glVertex3f(px, py, pz)
            GL.glVertex3f(px + dx, py + dy, pz + dz)
            GL.glEnd()
            GL.glPointSize(10.0 if (active == axis_name or hovered == axis_name) else 8.0)
            GL.glBegin(GL.GL_POINTS)
            GL.glVertex3f(px + dx, py + dy, pz + dz)
            GL.glEnd()
        GL.glLineWidth(1.0)
        GL.glPointSize(1.0)

    def _read_depth_snapshot(self, w: int, h: int):
        if w <= 0 or h <= 0:
            return None
        pw, ph = w, h
        try:
            raw = GL.glReadPixels(0, 0, pw, ph, GL.GL_DEPTH_COMPONENT, GL.GL_FLOAT)
        except Exception:
            return None
        if raw is None:
            return None
        try:
            buf = raw.reshape(-1)
        except AttributeError:
            import struct as _struct
            count = pw * ph
            buf = _struct.unpack(f'{count}f', bytes(raw)[:count * 4])
        return buf, pw, ph

    def _draw_marquee_overlay(self, w: int, h: int):
        if not self._marquee_active or not self._marquee_start or not self._marquee_end:
            return
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glPushMatrix()
        GL.glLoadIdentity()
        GL.glOrtho(0, w, h, 0, -1, 1)
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glPushMatrix()
        GL.glLoadIdentity()
        GL.glDisable(GL.GL_LIGHTING)
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glDisable(GL.GL_TEXTURE_2D)
        GL.glColor4f(1.0, 0.55, 0.15, 1.0)
        GL.glLineWidth(1.5)
        x0 = float(self._marquee_start.x())
        y0 = float(self._marquee_start.y())
        x1 = float(self._marquee_end.x())
        y1 = float(self._marquee_end.y())
        GL.glBegin(GL.GL_LINE_LOOP)
        GL.glVertex2f(x0, y0)
        GL.glVertex2f(x1, y0)
        GL.glVertex2f(x1, y1)
        GL.glVertex2f(x0, y1)
        GL.glEnd()
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_LIGHTING)
        GL.glPopMatrix()
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glPopMatrix()
        GL.glMatrixMode(GL.GL_MODELVIEW)

    def mousePressEvent(self, e):
        p = e.position().toPoint()
        self._last_pos = p
        self.setFocus()
        if not self.edit_mode and e.button() == Qt.MouseButton.LeftButton:
            if self._view_gizmo_mode == 'rotate':
                ring = self._pick_view_ring(float(p.x()), float(p.y()))
                if ring is not None:
                    self._begin_view_rotate(ring, p)
                    self.update()
                    return
                self._view_gizmo_mode = None
                self.update()
            elif self._view_gizmo_mode == 'translate':
                arrow = self._pick_view_arrow(float(p.x()), float(p.y()))
                if arrow is not None:
                    self._begin_view_translate(arrow, p)
                    self.update()
                    return
                self._view_gizmo_mode = None
                self.update()
        if self.edit_mode and e.button() == Qt.MouseButton.LeftButton:
            if self._grab_active or self._rotate_active:
                # Click while grabbing/rotating = confirm placement.
                self._confirm_grab()
                self._gizmo_drag_active = False
                self._gizmo_translate_active = False
                self._gizmo_translate_axis = None
                self.update()
                return
            if self._gizmo_mode == 'rotate':
                ring = self.pick_gizmo_ring(float(p.x()), float(p.y()))
                if ring is not None and self.begin_gizmo_drag(ring, p):
                    self.update()
                    return
                # Click off the gizmo dismisses it without rotating.
                self._gizmo_mode = None
                self.update()
            elif self._gizmo_mode == 'translate':
                arrow = self.pick_gizmo_arrow(float(p.x()), float(p.y()))
                if arrow is not None and self.begin_translate_drag(arrow, p):
                    self.update()
                    return
                self._gizmo_mode = None
                self.update()
            self._marquee_active = True
            self._marquee_start = p
            self._marquee_end = p
            self.update()
            return
        if e.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton) and self._spin_stage:
            self._spin_stage = 0
            self._yaw_velocity = 0.0

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton and self._view_gizmo_drag_active:
            self._view_gizmo_drag_active = False
            self._view_drag_start = None
            self._view_gizmo_axis = None
            self.update()
            return
        if e.button() == Qt.MouseButton.LeftButton and self._view_gizmo_translate_active:
            self._view_gizmo_translate_active = False
            self._view_drag_start = None
            self._view_gizmo_axis = None
            self.update()
            return
        if self.edit_mode and e.button() == Qt.MouseButton.LeftButton and self._gizmo_drag_active:
            self._gizmo_drag_active = False
            self._confirm_grab()
            self._gizmo_mode = 'rotate'
            self.update()
            return
        if self.edit_mode and e.button() == Qt.MouseButton.LeftButton and self._gizmo_translate_active:
            self._gizmo_translate_active = False
            self._gizmo_translate_axis = None
            self._confirm_grab()
            self._gizmo_mode = 'translate'
            self.update()
            return
        if self.edit_mode and e.button() == Qt.MouseButton.LeftButton and self._marquee_active:
            end = e.position().toPoint()
            self._marquee_end = end
            sx0 = self._marquee_start.x() if self._marquee_start else end.x()
            sy0 = self._marquee_start.y() if self._marquee_start else end.y()
            drag_pix = abs(end.x() - sx0) + abs(end.y() - sy0)
            ctrl = bool(e.modifiers() & Qt.KeyboardModifier.ControlModifier)
            shift = bool(e.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            click_select = getattr(self, '_click_select', None)
            if drag_pix <= 4 and click_select is not None:
                click_select(float(end.x()), float(end.y()), ctrl, shift)
            else:
                additive = ctrl or shift
                self._select_in_marquee(additive)
            self._marquee_active = False
            self._marquee_start = None
            self._marquee_end = None
            self.update()
        if not (e.buttons() & (Qt.MouseButton.LeftButton | Qt.MouseButton.MiddleButton | Qt.MouseButton.RightButton)):
            self._last_pos = None

    def mouseMoveEvent(self, e):
        p = e.position().toPoint()
        if self._view_gizmo_translate_active and self._view_drag_start is not None:
            self._apply_view_translate(p)
            return
        if self._view_gizmo_drag_active and self._view_drag_start is not None:
            self._apply_view_rotate(p)
            return
        if not self.edit_mode and self._view_gizmo_mode is not None:
            new_ring = None
            new_arrow = None
            if self._view_gizmo_mode == 'rotate':
                new_ring = self._pick_view_ring(float(p.x()), float(p.y()))
            else:
                new_arrow = self._pick_view_arrow(float(p.x()), float(p.y()))
            if new_ring != self._view_hover_ring:
                self._view_hover_ring = new_ring
                self.update()
            if new_arrow != self._view_hover_arrow:
                self._view_hover_arrow = new_arrow
                self.update()
        if self.edit_mode and self._gizmo_drag_active:
            self._apply_rotate(p)
            self._last_pos = p
            return
        if self.edit_mode and self._gizmo_translate_active:
            self._apply_grab(p)
            self._last_pos = p
            return
        if self.edit_mode and not self._grab_active and not self._rotate_active and not self._marquee_active:
            new_ring = None
            new_arrow = None
            if self._gizmo_mode == 'rotate':
                new_ring = self.pick_gizmo_ring(float(p.x()), float(p.y()))
            elif self._gizmo_mode == 'translate':
                new_arrow = self.pick_gizmo_arrow(float(p.x()), float(p.y()))
            if new_ring != self._hover_ring:
                self._hover_ring = new_ring
                self.update()
            if new_arrow != self._hover_arrow:
                self._hover_arrow = new_arrow
                self.update()
            updater = getattr(self, 'update_hover', None)
            if updater is not None and updater(float(p.x()), float(p.y())):
                self.update()
        if self.edit_mode and self._grab_active:
            self._apply_grab(p)
            self._last_pos = p
            return
        if self.edit_mode and self._rotate_active:
            self._apply_rotate(p)
            self._last_pos = p
            return
        if self.edit_mode and self._marquee_active:
            self._marquee_end = p
            self._last_pos = p
            self.update()
            return
        if self._last_pos is None:
            return
        dx = p.x() - self._last_pos.x()
        dy = p.y() - self._last_pos.y()
        self._last_pos = p
        buttons = e.buttons()
        if buttons & Qt.MouseButton.LeftButton:
            self._target_yaw = (self._target_yaw + dx * 0.5) % 360
            self._target_pitch = max(-89.0, min(89.0, self._target_pitch + dy * 0.5))
            self._arm_frame_next()
        elif buttons & Qt.MouseButton.RightButton:
            yaw_rad = math.radians(self.yaw)
            pitch_rad = math.radians(self.pitch)
            cy_, sy_ = math.cos(yaw_rad), math.sin(yaw_rad)
            cp_, sp_ = math.cos(pitch_rad), math.sin(pitch_rad)
            right_axis = (cy_, 0.0, sy_)
            up_axis = (sy_ * sp_, cp_, -cy_ * sp_)
            self._apply_model_rotation(up_axis, dx * 0.5)
            self._apply_model_rotation(right_axis, dy * 0.5)
        elif buttons & Qt.MouseButton.MiddleButton:
            scale = self.radius * self.zoom * 0.0025
            self._target_pan_x += dx * scale
            self._target_pan_y -= dy * scale
            self._arm_frame_next()

    def wheelEvent(self, e):
        delta = e.angleDelta().y() / 120.0
        self._target_zoom = max(0.05, min(20.0, self._target_zoom * (0.9 ** delta)))
        self._arm_frame_next()

    def _draw_view_rotate_gizmo(self):
        radius = self._view_gizmo_radius()
        steps = 64
        axis_colors = (
            ('x', (1.0, 0.30, 0.30)),
            ('y', (0.30, 1.0, 0.40)),
            ('z', (0.30, 0.55, 1.0)),
        )
        active = self._view_gizmo_axis if self._view_gizmo_drag_active else None
        hovered = self._view_hover_ring
        for axis_name, (r, g, b) in axis_colors:
            if active == axis_name or hovered == axis_name:
                GL.glColor3f(min(1.0, r + 0.2), min(1.0, g + 0.2), min(1.0, b + 0.2))
                GL.glLineWidth(4.0)
            else:
                dim = 0.55 if active in ('x', 'y', 'z') else 0.95
                GL.glColor3f(r * dim, g * dim, b * dim)
                GL.glLineWidth(2.5)
            GL.glBegin(GL.GL_LINE_LOOP)
            for i in range(steps):
                t = (i / steps) * math.tau
                ct = math.cos(t) * radius
                st = math.sin(t) * radius
                if axis_name == 'x':
                    GL.glVertex3f(0.0, ct, st)
                elif axis_name == 'y':
                    GL.glVertex3f(ct, 0.0, st)
                else:
                    GL.glVertex3f(ct, st, 0.0)
            GL.glEnd()
        GL.glLineWidth(1.0)

    def _draw_view_translate_gizmo(self):
        length = self._view_gizmo_arrow_length()
        axis_colors = (
            ('x', (1.0, 0.30, 0.30), (length, 0.0, 0.0)),
            ('y', (0.30, 1.0, 0.40), (0.0, length, 0.0)),
            ('z', (0.30, 0.55, 1.0), (0.0, 0.0, length)),
        )
        active = self._view_gizmo_axis if self._view_gizmo_translate_active else None
        hovered = self._view_hover_arrow
        for axis_name, (r, g, b), (dx, dy, dz) in axis_colors:
            if active == axis_name or hovered == axis_name:
                GL.glColor3f(min(1.0, r + 0.2), min(1.0, g + 0.2), min(1.0, b + 0.2))
                GL.glLineWidth(4.0)
            else:
                dim = 0.55 if active in ('x', 'y', 'z') else 0.95
                GL.glColor3f(r * dim, g * dim, b * dim)
                GL.glLineWidth(3.0)
            GL.glBegin(GL.GL_LINES)
            GL.glVertex3f(0.0, 0.0, 0.0)
            GL.glVertex3f(dx, dy, dz)
            GL.glEnd()
            GL.glPointSize(11.0 if (active == axis_name or hovered == axis_name) else 9.0)
            GL.glBegin(GL.GL_POINTS)
            GL.glVertex3f(dx, dy, dz)
            GL.glEnd()
        GL.glLineWidth(1.0)
        GL.glPointSize(1.0)

    def _mesh_world_center(self) -> tuple[float, float, float]:
        return (self.model_pan_x, self.model_pan_y, self.model_pan_z)

    def _project_world(self, p):
        mv = getattr(self, '_cached_world_mv', None)
        pr = self._cached_proj_matrix
        vp = self._cached_viewport
        if not (mv and pr and vp):
            return None
        x, y, z = p
        ex = mv[0] * x + mv[4] * y + mv[8] * z + mv[12]
        ey = mv[1] * x + mv[5] * y + mv[9] * z + mv[13]
        ez = mv[2] * x + mv[6] * y + mv[10] * z + mv[14]
        ew = mv[3] * x + mv[7] * y + mv[11] * z + mv[15]
        cx = pr[0] * ex + pr[4] * ey + pr[8] * ez + pr[12] * ew
        cy = pr[1] * ex + pr[5] * ey + pr[9] * ez + pr[13] * ew
        cw = pr[3] * ex + pr[7] * ey + pr[11] * ez + pr[15] * ew
        if abs(cw) < 1e-9:
            return None
        ndc_x = cx / cw
        ndc_y = cy / cw
        vx, vy, vw, vh = vp
        sx = vx + (ndc_x * 0.5 + 0.5) * vw
        sy = vy + (1.0 - (ndc_y * 0.5 + 0.5)) * vh
        return sx, sy

    def _view_gizmo_radius(self) -> float:
        return max(self.radius * 0.55, 0.1)

    def _view_gizmo_arrow_length(self) -> float:
        return max(self.radius * 0.6, 0.1)

    def _pick_view_ring(self, sx: float, sy: float, tol_pixels: float = 10.0) -> str | None:
        if not getattr(self, '_cached_world_mv', None):
            return None
        px, py, pz = self._mesh_world_center()
        radius = self._view_gizmo_radius()
        steps = 64
        best = None
        best_d2 = tol_pixels * tol_pixels
        for axis in ('x', 'y', 'z'):
            for i in range(steps):
                t = (i / steps) * math.tau
                ct = math.cos(t) * radius
                st = math.sin(t) * radius
                if axis == 'x':
                    p3 = (px, py + ct, pz + st)
                elif axis == 'y':
                    p3 = (px + ct, py, pz + st)
                else:
                    p3 = (px + ct, py + st, pz)
                proj = self._project_world(p3)
                if proj is None:
                    continue
                spx, spy = proj
                d2 = (spx - sx) ** 2 + (spy - sy) ** 2
                if d2 < best_d2:
                    best_d2 = d2
                    best = axis
        return best

    def _pick_view_arrow(self, sx: float, sy: float, tol_pixels: float = 10.0) -> str | None:
        px, py, pz = self._mesh_world_center()
        length = self._view_gizmo_arrow_length()
        steps = 24
        best = None
        best_d2 = tol_pixels * tol_pixels
        for axis in ('x', 'y', 'z'):
            for i in range(steps + 1):
                t = i / steps
                if axis == 'x':
                    p3 = (px + length * t, py, pz)
                elif axis == 'y':
                    p3 = (px, py + length * t, pz)
                else:
                    p3 = (px, py, pz + length * t)
                proj = self._project_world(p3)
                if proj is None:
                    continue
                spx, spy = proj
                d2 = (spx - sx) ** 2 + (spy - sy) ** 2
                if d2 < best_d2:
                    best_d2 = d2
                    best = axis
        return best

    def _begin_view_translate(self, axis: str, anchor):
        self._view_gizmo_translate_active = True
        self._view_gizmo_axis = axis
        self._view_drag_start = anchor
        self._view_drag_origin_pan = (self.model_pan_x, self.model_pan_y, self.model_pan_z)
        self._arm_frame_next()

    def _apply_view_translate(self, current):
        if self._view_drag_start is None:
            return
        dx_pix = current.x() - self._view_drag_start.x()
        dy_pix = current.y() - self._view_drag_start.y()
        h = max(1, self.height())
        dist = self.radius * 2.5 * self.zoom
        fov = math.radians(45.0)
        world_per_pix = (2.0 * dist * math.tan(fov / 2.0)) / h
        dxw = dx_pix * world_per_pix
        dyw = -dy_pix * world_per_pix
        yaw_rad = math.radians(self.yaw)
        pitch_rad = math.radians(self.pitch)
        cy_, sy_ = math.cos(yaw_rad), math.sin(yaw_rad)
        cp_, sp_ = math.cos(pitch_rad), math.sin(pitch_rad)
        right = (cy_, 0.0, sy_)
        up = (sy_ * sp_, cp_, -cy_ * sp_)
        wx = right[0] * dxw + up[0] * dyw
        wy = right[1] * dxw + up[1] * dyw
        wz = right[2] * dxw + up[2] * dyw
        ax = self._view_gizmo_axis
        if ax == 'x':
            wy = wz = 0.0
        elif ax == 'y':
            wx = wz = 0.0
        elif ax == 'z':
            wx = wy = 0.0
        ox, oy, oz = self._view_drag_origin_pan
        self.model_pan_x = ox + wx
        self.model_pan_y = oy + wy
        self.model_pan_z = oz + wz
        self.update()

    def _begin_view_rotate(self, axis: str, anchor):
        self._view_gizmo_drag_active = True
        self._view_gizmo_axis = axis
        self._view_drag_start = anchor

    def _apply_view_rotate(self, current):
        if self._view_drag_start is None:
            return
        dx = current.x() - self._view_drag_start.x()
        dy = current.y() - self._view_drag_start.y()
        angle = (dx + dy) * 0.5
        ax = self._view_gizmo_axis
        if ax == 'x':
            world_axis = (1.0, 0.0, 0.0)
        elif ax == 'y':
            world_axis = (0.0, 1.0, 0.0)
        else:
            world_axis = (0.0, 0.0, 1.0)
        self._apply_model_rotation(world_axis, angle)
        self._view_drag_start = current

    def keyPressEvent(self, e):
        key = e.key()
        if self.edit_mode:
            mods = e.modifiers()
            if mods & Qt.KeyboardModifier.ControlModifier:
                if key == Qt.Key.Key_Z and not (mods & Qt.KeyboardModifier.ShiftModifier):
                    self.undo_edit()
                    return
                if key == Qt.Key.Key_Y or (
                    key == Qt.Key.Key_Z and (mods & Qt.KeyboardModifier.ShiftModifier)
                ):
                    self.redo_edit()
                    return
            if key == Qt.Key.Key_Escape:
                if self._grab_active or self._rotate_active:
                    self._cancel_grab()
                    self._gizmo_drag_active = False
                    self._gizmo_translate_active = False
                    self._gizmo_translate_axis = None
                    self.update()
                    return
                if self._gizmo_mode is not None:
                    self._gizmo_mode = None
                    self.update()
                    return
                self.selected_verts.clear()
                self.selected_bones.clear()
                self.update()
                return
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                if self._grab_active or self._rotate_active:
                    self._confirm_grab()
                    self.update()
                    return
            if key == Qt.Key.Key_G:
                has_sel = bool(self.selected_bones if self.edit_target == 'bone' else self.selected_verts)
                if has_sel and not self._grab_active and not self._rotate_active:
                    self._gizmo_mode = 'translate'
                    self.update()
                    return
            if key == Qt.Key.Key_R and (e.modifiers() & Qt.KeyboardModifier.ShiftModifier) \
                    and self.edit_target == 'bone':
                if self.selected_bones:
                    for bid in list(self.selected_bones):
                        self.reset_bone(bid)
                return
            if key == Qt.Key.Key_R:
                has_sel = bool(self.selected_bones if self.edit_target == 'bone' else self.selected_verts)
                if has_sel and not self._grab_active and not self._rotate_active:
                    self._gizmo_mode = 'rotate'
                    self.update()
                    return
            if key in (Qt.Key.Key_X, Qt.Key.Key_Y, Qt.Key.Key_Z):
                axis = {Qt.Key.Key_X: 'x', Qt.Key.Key_Y: 'y', Qt.Key.Key_Z: 'z'}[key]
                if self._rotate_active:
                    self.set_rotate_axis_mode(axis)
                    return
                if self._grab_active:
                    self._grab_axis_mode = None if self._grab_axis_mode == axis else axis
                    from PyQt6.QtGui import QCursor
                    cur = self.mapFromGlobal(QCursor.pos())
                    self._apply_grab(cur)
                    self.update()
                    return
                self._rotate_axis_mode = axis
                self.update()
                return
            if key == Qt.Key.Key_A:
                self.select_all_in_target()
                self.update()
                return
            if key == Qt.Key.Key_V:
                self.set_edit_target('bone' if self.edit_target == 'vertex' else 'vertex')
                return
            if key == Qt.Key.Key_L and self.edit_target == 'vertex':
                seed = self._hover_vert
                if seed is None and not self.selected_verts:
                    return
                if not (e.modifiers() & Qt.KeyboardModifier.ShiftModifier):
                    if seed is not None:
                        self.selected_verts.clear()
                self.select_linked(seed)
                return
            if key == Qt.Key.Key_L and self.edit_target == 'bone':
                if self.selected_bones:
                    self.push_undo()
                    any_unlocked = any(b not in self.bone_locked for b in self.selected_bones)
                    for bid in list(self.selected_bones):
                        if any_unlocked:
                            self.bone_locked.add(bid)
                        else:
                            self.bone_locked.discard(bid)
                    self.update()
                return
        if key == Qt.Key.Key_E:
            super().keyPressEvent(e)
            return
        if self.edit_mode:
            return
        if key == Qt.Key.Key_R:
            self._view_gizmo_mode = None if self._view_gizmo_mode == 'rotate' else 'rotate'
            self._view_hover_ring = None
            self._view_hover_arrow = None
            self.update()
        elif key == Qt.Key.Key_F:
            self.toggle_frame_or_reset()
        elif key == Qt.Key.Key_O:
            self.cycle_spin()
        elif key == Qt.Key.Key_G:
            self._view_gizmo_mode = None if self._view_gizmo_mode == 'translate' else 'translate'
            self._view_hover_ring = None
            self._view_hover_arrow = None
            self.update()
        elif key == Qt.Key.Key_Escape and self._view_gizmo_mode is not None:
            self._view_gizmo_mode = None
            self._view_gizmo_drag_active = False
            self._view_gizmo_translate_active = False
            self._view_drag_start = None
            self.update()
        else:
            super().keyPressEvent(e)


_UV_GROUP_COLORS = [
    (231,  76,  60),  # red
    ( 46, 204, 113),  # green
    ( 52, 152, 219),  # blue
    (241, 196,  15),  # amber
    ( 26, 188, 156),  # teal
    (230, 126,  34),  # orange
    (149, 165, 166),  # gray
    (255, 105, 180),  # pink
    (127, 255,   0),  # chartreuse
    (  0, 191, 255),  # deep sky
]


class UVMapView(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(QSize(420, 420))
        self._tex_pixmap: QPixmap | None = None
        self._uvs: list[tuple[float, float]] = []
        self._tris: list[tuple[int, int, int]] = []
        self._groups: list[int] = []
        self._uv_bounds = (0.0, 0.0, 1.0, 1.0)
        self._show_texture = True
        self._show_fill = False
        self._zoom = 1.0
        self._pan = QPoint(0, 0)
        self._dragging = False
        self._last_drag: QPoint | None = None
        self.setMouseTracking(True)

    def set_data(self, tex_image, uvs, tris, groups):
        if tex_image is not None:
            qimg = QImage(
                tex_image.tobytes('raw', 'RGBA'),
                tex_image.width, tex_image.height,
                tex_image.width * 4,
                QImage.Format.Format_RGBA8888,
            ).copy()
            self._tex_pixmap = QPixmap.fromImage(qimg)
        else:
            self._tex_pixmap = None
        self._uvs = list(uvs)
        self._tris = list(tris)
        self._groups = list(groups) if groups else [0] * len(tris)
        self._uv_bounds = (0.0, 0.0, 1.0, 1.0)
        self.reset_view()

    def set_show_texture(self, on: bool):
        self._show_texture = on
        self.update()

    def set_show_fill(self, on: bool):
        self._show_fill = on
        self.update()

    def reset_view(self):
        self._zoom = 1.0
        self._pan = QPoint(0, 0)
        self.update()

    def wheelEvent(self, e):
        delta = e.angleDelta().y()
        if delta == 0:
            return
        factor = 1.15 if delta > 0 else 1 / 1.15
        self._zoom = max(0.1, min(40.0, self._zoom * factor))
        self.update()

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._last_drag = e.position().toPoint()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            self._last_drag = None

    def mouseMoveEvent(self, e):
        if self._dragging and self._last_drag is not None:
            cur = e.position().toPoint()
            delta = cur - self._last_drag
            self._pan += delta
            self._last_drag = cur
            self.update()

    def _texture_rect(self):
        umin, vmin, umax, vmax = self._uv_bounds
        span = max(umax - umin, vmax - vmin, 1e-6)
        avail = max(64, min(self.width(), self.height()) - 24)
        side = (avail / span) * self._zoom
        cx = self.width() / 2 + self._pan.x()
        cy = self.height() / 2 + self._pan.y()
        bw = (umax - umin) * side
        bh = (vmax - vmin) * side
        ox = cx - bw / 2 - umin * side
        oy = cy - bh / 2 - vmin * side
        return ox, oy, side

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.fillRect(self.rect(), QColor(24, 26, 30))
        ox, oy, side = self._texture_rect()
        ox_i, oy_i, side_i = int(ox), int(oy), max(1, int(side))
        if self._show_texture and self._tex_pixmap is not None:
            p.drawPixmap(ox_i, oy_i, side_i, side_i, self._tex_pixmap)
        else:
            p.fillRect(ox_i, oy_i, side_i, side_i, QColor(40, 42, 48))
            grid = QPen(QColor(60, 62, 70))
            grid.setWidth(1)
            p.setPen(grid)
            for i in range(1, 8):
                fx = ox_i + side_i * i // 8
                fy = oy_i + side_i * i // 8
                p.drawLine(fx, oy_i, fx, oy_i + side_i)
                p.drawLine(ox_i, fy, ox_i + side_i, fy)
        p.setPen(QPen(QColor(100, 105, 115), 1))
        p.drawRect(ox_i, oy_i, side_i, side_i)

        if not self._uvs or not self._tris:
            p.end()
            return

        def to_screen(uv):
            u, v = uv
            return (ox + u * side, oy + v * side)

        eps = 1e-4
        for ti, (a, b, c) in enumerate(self._tris):
            if a >= len(self._uvs) or b >= len(self._uvs) or c >= len(self._uvs):
                continue
            ua, va = self._uvs[a]
            ub, vb = self._uvs[b]
            uc, vc = self._uvs[c]
            if (min(ua, ub, uc) < -eps or max(ua, ub, uc) > 1 + eps or
                    min(va, vb, vc) < -eps or max(va, vb, vc) > 1 + eps):
                continue
            grp = self._groups[ti] if ti < len(self._groups) else 0
            r, g, bl = _UV_GROUP_COLORS[grp % len(_UV_GROUP_COLORS)]
            pa = to_screen(self._uvs[a])
            pb = to_screen(self._uvs[b])
            pc = to_screen(self._uvs[c])
            if self._show_fill:
                from PyQt6.QtGui import QPolygonF
                from PyQt6.QtCore import QPointF
                poly = QPolygonF([
                    QPointF(*pa), QPointF(*pb), QPointF(*pc),
                ])
                p.setBrush(QColor(r, g, bl, 55))
                p.setPen(Qt.PenStyle.NoPen)
                p.drawPolygon(poly)
            pen = QPen(QColor(r, g, bl, 220))
            pen.setWidthF(1.0)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawLine(int(pa[0]), int(pa[1]), int(pb[0]), int(pb[1]))
            p.drawLine(int(pb[0]), int(pb[1]), int(pc[0]), int(pc[1]))
            p.drawLine(int(pc[0]), int(pc[1]), int(pa[0]), int(pa[1]))
        p.end()


class UVMapDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('UV Map')
        self.resize(640, 720)
        self.view = UVMapView(self)
        self.show_tex = QCheckBox('Show texture', self)
        self.show_tex.setChecked(True)
        self.show_tex.toggled.connect(self.view.set_show_texture)
        self.show_fill = QCheckBox('Fill islands', self)
        self.show_fill.setChecked(False)
        self.show_fill.toggled.connect(self.view.set_show_fill)
        reset = QPushButton('Reset view', self)
        reset.clicked.connect(self.view.reset_view)
        self.info = QLabel('', self)
        controls = QHBoxLayout()
        controls.addWidget(self.show_tex)
        controls.addWidget(self.show_fill)
        controls.addWidget(reset)
        controls.addStretch(1)
        controls.addWidget(self.info)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addLayout(controls)
        layout.addWidget(self.view, 1)

    def populate(self, tex_image, uvs, tris, groups):
        self.view.set_data(tex_image, uvs, tris, groups)
        n_groups = len(set(groups)) if groups else 0
        self.info.setText(
            f'{len(tris):,} tris  |  {len(uvs):,} UVs  |  {n_groups} groups'
        )


class MainWindow(QMainWindow):
    def __init__(self, root: Path):
        super().__init__()
        self.root = root
        self.shape_path: Path | None = None
        self.active_texture_suffix: str | None = None
        self._preview_mode = False
        self._tri_groups: list[int] = []
        self._uv_dialog: 'UVMapDialog | None' = None
        self.crit_mass_action = None

        self.setWindowTitle('[GZME] GodZilla Model Editor')
        self.resize(1024, 720)

        self.viewer = MeshViewer()
        self._config = load_config()
        self._theme_key = self._config.get('theme', 'dark')
        if self._theme_key not in THEMES:
            self._theme_key = 'dark'

        bar_widget = self._build_top_bar()
        self.controls_overlay = self._build_controls_panel(self.viewer)
        self.controls_overlay.move(12, 12)
        self.controls_overlay.show()
        self._controls_visible = True

        self.texture_preview = QLabel(self.viewer)
        self.texture_preview.setFixedSize(180, 180)
        self.texture_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.texture_preview.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.texture_preview.hide()

        self.viewer.installEventFilter(self)

        self._overlay_shortcut = QShortcut(QKeySequence('Z'), self)
        self._overlay_shortcut.activated.connect(self._cycle_overlay_mode)

        self.path_label = QLabel('No Shapes.BDG loaded  |  Digitzaki & itsaiden66')

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(bar_widget)
        layout.addWidget(self.path_label)
        layout.addWidget(self.viewer, 1)

        wrap = QWidget()
        wrap.setLayout(layout)
        self.setCentralWidget(wrap)
        self.setStatusBar(QStatusBar())
        self.geom_label = QLabel('No model loaded')
        self.statusBar().addPermanentWidget(self.geom_label)

        self.watcher = QFileSystemWatcher(self)
        self.watcher.fileChanged.connect(self._on_file_changed)
        self._reload_timer = QTimer(self)
        self._reload_timer.setSingleShot(True)
        self._reload_timer.setInterval(400)
        self._reload_timer.timeout.connect(self.reload_model)

        self.apply_theme(self._theme_key)

        saved_folder = self._config.get('folder')
        if saved_folder and Path(saved_folder).is_dir():
            self.root = Path(saved_folder)
        self._scan_folder(self.root)
        saved_shape = self._config.get('shape')
        target: Path | None = None
        if saved_shape and Path(saved_shape).is_file():
            target = Path(saved_shape)
        elif self.model_combo.count() > 0:
            target = Path(self.model_combo.itemData(0))
        if target:
            self.load_shape(target)

    def _build_controls_panel(self, parent: QWidget | None = None) -> QWidget:
        label = QLabel('', parent)
        label.setTextFormat(Qt.TextFormat.RichText)
        label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        return label

    def _refresh_controls_panel(self):
        theme = get_theme(self._theme_key)
        text = theme['text_color']
        muted = theme['muted_color']
        if self.viewer.edit_mode:
            target = 'bones' if self.viewer.edit_target == 'bone' else 'verts'
            l_row = ('L', 'Stiffen bones (no in-game bend)') if target == 'bones' else ('L', 'Select linked (fill island)')
            rows = [
                ('E', 'Exit edit mode'),
                ('V', f'Toggle verts / bones ({target})'),
                ('', ''),
                ('LMB drag', 'Box-select'),
                ('Shift+drag', 'Add to selection'),
                ('A', 'Select all'),
                ('', ''),
                ('G', 'Grab / move selection'),
                ('G then X/Y/Z', 'Lock grab to world axis'),
                ('R', 'Rotate selection'),
                ('R then X/Y/Z', 'Lock rotate to world axis'),
                ('Enter', 'Confirm transform'),
                ('Esc', 'Cancel / clear selection'),
                ('Ctrl+Z', 'Undo'),
                ('Ctrl+Y', 'Redo'),
                ('', ''),
                l_row,
                ('Shift+R', 'Reset bones to original'),
                ('', ''),
                ('H', 'Toggle this help'),
                ('LMB', 'Rotate camera'),
                ('MMB', 'Pan'),
                ('RMB', 'Rotate model'),
            ]
        else:
            rows = [
                ('E', 'Edit mode'),
                ('R', 'Rotate mesh (axis rings)'),
                ('G', 'Move mesh (axis arrows)'),
                ('F', 'Frame -> Reset (alternates)'),
                ('O', 'Cycle auto-orbit'),
                ('H', 'Toggle this help'),
                ('Z', 'Cycle overlays (axes/grid/off)'),
                ('', ''),
                ('T', 'Cycle texture (M/B/C/S/off)'),
                ('W', 'Cycle wireframe (black/colored/off)'),
                ('M', 'Toggle Critical Mass'),
                ('N', 'Game preview composite'),
                ('', ''),
                ('LMB', 'Rotate camera'),
                ('MMB', 'Pan'),
                ('RMB', 'Rotate model'),
            ]
        html = '<table style="font-family:Consolas,monospace;font-size:11px;">'
        for key, desc in rows:
            if not key and not desc:
                html += '<tr><td colspan="2" style="height:6px;"></td></tr>'
            else:
                html += (
                    f'<tr><td style="padding:1px 10px 1px 0;color:{text};'
                    f'font-weight:bold;">{key}</td>'
                    f'<td style="padding:1px 0;color:{muted};">{desc}</td></tr>'
                )
        html += '</table>'
        self.controls_overlay.setText(html)
        self.controls_overlay.setStyleSheet(theme['overlay_qss'])
        self.controls_overlay.adjustSize()

    def apply_theme(self, key: str):
        self._theme_key = key
        theme = get_theme(key)
        text = theme['text_color']
        muted = theme['muted_color']
        r, g, b = theme['viewer_clear']
        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
        if lum < 0.45:
            bg, hover_bg, sel_bg = '#23272e', '#343b45', '#3f4753'
        else:
            bg, hover_bg, sel_bg = '#ffffff', '#e6ecf5', '#cfdcef'
        chrome_overrides = {
            'win98':  ('#ffffff', '#000080', '#000080'),
            'hotdog': ('#ffff00', '#ff0000', '#ff0000'),
            'amber':  ('#1a0e00', '#3a2200', '#5a3300'),
            'matrix': ('#001a08', '#003315', '#005c25'),
        }
        if key in chrome_overrides:
            bg, hover_bg, sel_bg = chrome_overrides[key]
        combo_qss = (
            f'QComboBox{{background:{bg};color:{text};border:1px solid {muted};'
            f'padding:3px 8px;border-radius:3px;}}'
            f'QComboBox:hover{{background:{hover_bg};}}'
            f'QComboBox::drop-down{{border:0;width:18px;}}'
            f'QComboBox QAbstractItemView{{background:{bg};color:{text};'
            f'selection-background-color:{sel_bg};selection-color:{text};'
            f'border:1px solid {muted};outline:0;}}'
            f'QComboBox QAbstractItemView::item{{padding:4px 8px;min-height:18px;}}'
        )
        self.setStyleSheet(theme['app_qss'] + combo_qss)
        if hasattr(self, 'model_combo'):
            self.model_combo.view().setStyleSheet(combo_qss)
        self.path_label.setStyleSheet(theme['path_label_qss'])
        for sep in getattr(self, '_bar_separators', []):
            sep.setStyleSheet(f'color:{theme["muted_color"]};padding:0 4px;')
        self._refresh_controls_panel()
        self.texture_preview.setStyleSheet(theme['preview_qss'])
        self.viewer.set_clear_color(theme['viewer_clear'])
        self._reposition_overlays()
        cfg = load_config()
        cfg['theme'] = key
        save_config(cfg)

    def _cycle_overlay_mode(self):
        v = self.viewer
        v.overlay_mode = (getattr(v, 'overlay_mode', 0) + 1) % 3
        v.update()

    def _build_top_bar(self) -> QWidget:
        folder_btn = QPushButton('Open Folder')
        folder_btn.setToolTip('Pick a folder; all *_Shapes.BDG files inside (and subfolders) appear in the dropdown.')
        folder_btn.clicked.connect(self.pick_folder)

        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(200)
        self.model_combo.setToolTip('Loaded models in the selected folder')
        self.model_combo.currentIndexChanged.connect(self._on_model_combo_changed)

        model_btn = QToolButton()
        model_btn.setText('Model')
        model_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        model_menu = QMenu(model_btn)
        a_export = model_menu.addAction('Run Export (BDG to FBX)')
        a_import = model_menu.addAction('Run Import (FBX to BDG)')
        model_menu.addSeparator()
        a_reload = model_menu.addAction('Reload Model')
        a_export.triggered.connect(self.run_export)
        a_import.triggered.connect(self.run_import)
        a_reload.triggered.connect(self.reload_model)
        model_menu.addSeparator()
        self.edit_mode_action = model_menu.addAction('Edit Mode (E)')
        self.edit_mode_action.setCheckable(True)
        self.edit_mode_action.toggled.connect(self._on_edit_mode_toggled)
        a_save_geom = model_menu.addAction('Save Geometry to BDG')
        a_save_geom.triggered.connect(self.save_geometry)
        model_btn.setMenu(model_menu)

        tex_btn = QToolButton()
        tex_btn.setText('Texture')
        tex_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        tex_menu = QMenu(tex_btn)
        for suffix, label in TEXTURE_SUFFIXES.items():
            show = tex_menu.addAction(f'Show _{suffix} - {label}')
            show.triggered.connect(lambda _c=False, s=suffix: self.show_texture(s))
        tex_menu.addSeparator()
        preview_act = tex_menu.addAction('Load Game Preview')
        preview_act.triggered.connect(self.show_game_preview)
        self.crit_mass_action = tex_menu.addAction('Enable Critical Mass')
        self.crit_mass_action.setCheckable(True)
        self.crit_mass_action.toggled.connect(self._on_crit_mass_toggled)
        tex_menu.addSeparator()
        for suffix, label in TEXTURE_SUFFIXES.items():
            rep = tex_menu.addAction(f'Replace _{suffix} from PNG...')
            rep.triggered.connect(lambda _c=False, s=suffix: self.replace_texture(s))
        tex_menu.addSeparator()
        clear = tex_menu.addAction('Clear Texture Overlay')
        clear.triggered.connect(self.clear_texture)
        tex_btn.setMenu(tex_menu)

        opt_btn = QToolButton()
        opt_btn.setText('Options')
        opt_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        opt_menu = QMenu(opt_btn)
        self.wireframe_action = opt_menu.addAction('Show Wireframe')
        self.wireframe_action.setCheckable(True)
        self.wireframe_action.setChecked(True)
        self.wireframe_action.toggled.connect(self.viewer.set_wireframe)
        self.bones_action = opt_menu.addAction('Show Bones')
        self.bones_action.setCheckable(True)
        self.bones_action.toggled.connect(self.viewer.set_show_bones)
        self.uv_overlay_action = opt_menu.addAction('Show UV Map on Model')
        self.uv_overlay_action.setCheckable(True)
        self.uv_overlay_action.toggled.connect(self.viewer.set_show_uv_overlay)
        opt_menu.addSeparator()
        uv_action = opt_menu.addAction('Generate UV Map...')
        uv_action.triggered.connect(self.show_uv_map)
        opt_menu.addSeparator()

        opt_menu.addSeparator()
        theme_menu = opt_menu.addMenu('Theme')
        self._theme_group = QActionGroup(self)
        self._theme_group.setExclusive(True)
        for key, theme in THEMES.items():
            act = theme_menu.addAction(theme['name'])
            act.setCheckable(True)
            act.setData(key)
            if key == self._theme_key:
                act.setChecked(True)
            act.triggered.connect(lambda _c=False, k=key: self.apply_theme(k))
            self._theme_group.addAction(act)
        opt_btn.setMenu(opt_menu)

        bar = QHBoxLayout()
        bar.setContentsMargins(8, 8, 8, 0)
        bar.setSpacing(6)
        bar.addWidget(folder_btn)
        bar.addWidget(self.model_combo)
        self._bar_separators = []
        for _ in range(3):
            sep = QLabel('|')
            self._bar_separators.append(sep)
        bar.addWidget(self._bar_separators[0])
        bar.addWidget(model_btn)
        bar.addWidget(self._bar_separators[1])
        bar.addWidget(tex_btn)
        bar.addWidget(self._bar_separators[2])
        bar.addWidget(opt_btn)
        bar.addStretch(1)
        self.edit_indicator = QLabel('Editor Mode Enabled')
        self.edit_indicator.setStyleSheet(
            'color: #ff8a3d; font-weight: 600; padding: 2px 10px;'
            ' border: 1px solid #ff8a3d; border-radius: 4px;'
        )
        self.edit_indicator.setVisible(False)
        bar.addWidget(self.edit_indicator)

        wrap = QWidget()
        wrap.setLayout(bar)
        return wrap

    def _reposition_overlays(self):
        self.controls_overlay.move(12, 12)
        self.controls_overlay.adjustSize()
        y = 12 + self.controls_overlay.height() + 10
        self.texture_preview.move(12, y)

    def _show_texture_preview(self, img: Image.Image | None):
        if img is None:
            self.texture_preview.hide()
            return
        thumb = img.convert('RGBA').copy()
        thumb.thumbnail((160, 160), Image.LANCZOS)
        self._preview_bytes = thumb.tobytes('raw', 'RGBA')
        qimg = QImage(self._preview_bytes, thumb.width, thumb.height,
                      thumb.width * 4, QImage.Format.Format_RGBA8888)
        self.texture_preview.setPixmap(QPixmap.fromImage(qimg.copy()))
        self.texture_preview.show()
        self._reposition_overlays()

    def eventFilter(self, obj, event):
        from PyQt6.QtCore import QEvent
        if obj is self.viewer:
            if event.type() == QEvent.Type.KeyPress:
                if self._handle_viewer_key(event):
                    return True
                if self.viewer.edit_mode and event.key() in (Qt.Key.Key_V, Qt.Key.Key_B):
                    QTimer.singleShot(0, self._refresh_controls_panel)
            if event.type() == QEvent.Type.Resize:
                self._reposition_overlays()
        return super().eventFilter(obj, event)

    def _handle_viewer_key(self, event) -> bool:
        if self.viewer.edit_mode:
            return False
        key = event.key()
        if key == Qt.Key.Key_H:
            self._controls_visible = not self._controls_visible
            self.controls_overlay.setVisible(self._controls_visible)
            return True
        if key == Qt.Key.Key_T:
            self.cycle_texture()
            return True
        if key == Qt.Key.Key_W:
            self.viewer.cycle_wire_mode()
            mode = self.viewer.wire_mode
            label = {0: 'off', 1: 'black wire', 2: 'colored wire'}.get(mode, '')
            if hasattr(self, 'wireframe_action'):
                self.wireframe_action.setChecked(mode != 0)
            self.statusBar().showMessage(f'Wireframe: {label}', 3000)
            return True
        if key == Qt.Key.Key_N:
            self.toggle_game_preview()
            return True
        if key == Qt.Key.Key_M:
            self.toggle_critical_mass()
            return True
        return False

    _TEXTURE_CYCLE = ('M', 'B', 'C', 'S')

    def cycle_texture(self):
        if not self.shape_path:
            return
        cur = self.active_texture_suffix
        steps: list[str | None] = list(self._TEXTURE_CYCLE) + [None]
        if self._preview_mode or cur is None or cur not in self._TEXTURE_CYCLE:
            start = -1 if self._preview_mode or cur is None else len(steps) - 1
        else:
            start = self._TEXTURE_CYCLE.index(cur)
        for offset in range(1, len(steps) + 1):
            nxt = steps[(start + offset) % len(steps)]
            if nxt is None:
                self.clear_texture()
                self.statusBar().showMessage('Textures off', 3000)
                return
            if self._try_show_texture(nxt):
                return

    def _try_show_texture(self, suffix: str) -> bool:
        if not self.shape_path:
            return False
        try:
            img, info = decode_texture_image(self.shape_path, suffix)
        except Exception:
            return False
        if img is None:
            return False
        self.active_texture_suffix = suffix
        self._preview_mode = False
        self.viewer.set_texture_image(img)
        self._show_texture_preview(img)
        self.statusBar().showMessage(f'Showing {info}', 6000)
        return True

    def toggle_game_preview(self):
        if not self.shape_path:
            return
        if self._preview_mode:
            self.clear_texture()
            self.statusBar().showMessage('Game preview off', 3000)
        else:
            self.show_game_preview()

    def _set_light(self, idx: int):
        new_idx = self.viewer.set_light_index(idx)
        if hasattr(self, '_light_actions') and 0 <= new_idx < len(self._light_actions):
            self._light_actions[new_idx].setChecked(True)
        name = MeshViewer.LIGHT_PRESETS[new_idx][0]
        self.statusBar().showMessage(f'Lighting: {name}', 3000)

    def _cycle_light(self):
        self._set_light((self.viewer.light_index + 1) % len(MeshViewer.LIGHT_PRESETS))

    def toggle_critical_mass(self):
        if not self.crit_mass_action:
            return
        self.crit_mass_action.setChecked(not self.crit_mass_action.isChecked())

    def pick_folder(self):
        start = str(self.root)
        path = QFileDialog.getExistingDirectory(
            self, 'Choose folder containing *_Shapes.BDG models', start,
        )
        if not path:
            return
        self.root = Path(path)
        cfg = load_config()
        cfg['folder'] = str(self.root)
        cfg.pop('shape', None)
        save_config(cfg)
        self._scan_folder(self.root)
        if self.model_combo.count() > 0:
            first = Path(self.model_combo.itemData(0))
            self.load_shape(first)
        else:
            self.statusBar().showMessage('No *_Shapes.BDG files found in that folder', 6000)

    def _scan_folder(self, folder: Path):
        found: list[Path] = []
        try:
            for p in folder.rglob('*'):
                if p.is_file() and p.name.lower().endswith('_shapes.bdg'):
                    found.append(p)
        except Exception:
            pass
        found.sort(key=lambda p: p.name.lower())
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        for p in found:
            display = re.sub(r'(?i)_shapes\.bdg$', '', p.name)
            self.model_combo.addItem(display, str(p))
        self.model_combo.blockSignals(False)

    def _on_model_combo_changed(self, idx: int):
        if idx < 0:
            return
        path = self.model_combo.itemData(idx)
        if path:
            cfg = load_config()
            cfg['shape'] = path
            save_config(cfg)
            self.load_shape(Path(path))

    def load_shape(self, path: Path):
        if self.shape_path and self.watcher.files():
            self.watcher.removePaths(self.watcher.files())
        self.shape_path = path
        self.watcher.addPath(str(path))
        self._update_path_label()
        if hasattr(self, 'model_combo'):
            for i in range(self.model_combo.count()):
                if self.model_combo.itemData(i) == str(path):
                    self.model_combo.blockSignals(True)
                    self.model_combo.setCurrentIndex(i)
                    self.model_combo.blockSignals(False)
                    break
        if not self.reload_model():
            return

    def _update_path_label(self):
        name = self.shape_path.name if self.shape_path else 'No Shapes.BDG loaded'
        self.path_label.setText(f'Model: {name}  |  Digitzaki & itsaiden66')

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_E and not (e.modifiers() & Qt.KeyboardModifier.ControlModifier):
            self.edit_mode_action.toggle()
            return
        super().keyPressEvent(e)

    def _on_edit_mode_toggled(self, on: bool):
        self.viewer.set_edit_mode(on)
        if hasattr(self, 'edit_indicator'):
            self.edit_indicator.setVisible(on)
        self._refresh_controls_panel()
        self._reposition_overlays()
        if on:
            self.statusBar().showMessage(
                'Edit mode ON — V verts / B bones, drag-select, click +Shift add / +Ctrl toggle, L select linked, G grab, R rotate, A all, Enter confirm, Esc cancel',
                0,
            )
            self.viewer.show_bones = True
        else:
            self.viewer.show_bones = False
            if hasattr(self, 'bones_action'):
                self.bones_action.setChecked(False)
            self.viewer.update()
            self.statusBar().showMessage('Edit mode OFF', 4000)

    def save_geometry(self):
        if not self.shape_path or not self.shape_path.exists():
            QMessageBox.information(self, 'Save Geometry', 'Open a Shapes.BDG first.')
            return
        vert_edits = self.viewer.collect_position_writes()
        bone_edits = self.viewer.collect_bone_writes()
        stiff_remap = self._build_stiff_bone_remap()
        if not vert_edits and not bone_edits and not stiff_remap:
            self.statusBar().showMessage('Nothing to save.', 4000)
            return
        try:
            backup = self.shape_path.with_suffix(self.shape_path.suffix + '.bak')
            if not backup.exists():
                shutil.copy2(self.shape_path, backup)
            data = bytearray(self.shape_path.read_bytes())
            for v_start, src_idx, layout, pos in vert_edits:
                stride = {'skin64': 64, 'blend76': 76, 'blend52': 52, 'blend60': 60, 'skin48': 48, 'skin40': 40}.get(layout, 40)
                off = v_start + src_idx * stride
                struct.pack_into('>3f', data, off, float(pos[0]), float(pos[1]), float(pos[2]))
            for rec_off, t in bone_edits:
                if rec_off <= 0 or rec_off + 44 > len(data):
                    continue
                struct.pack_into('>3f', data, rec_off + 32, float(t[0]), float(t[1]), float(t[2]))
            stiff_count = self._apply_stiff_bone_remap(data, stiff_remap)
            self.shape_path.write_bytes(bytes(data))
        except Exception as exc:
            QMessageBox.critical(self, 'Save failed', str(exc))
            return
        self.statusBar().showMessage(
            f'Wrote {len(vert_edits)} verts, {len(bone_edits)} bones, '
            f'{stiff_count} skin slots reweighted to {self.shape_path.name}', 6000,
        )

    def _build_stiff_bone_remap(self) -> dict[int, int]:
        locked = getattr(self.viewer, 'bone_locked', set())
        if not locked:
            return {}
        parents = self.viewer.bone_parents
        remap: dict[int, int] = {}
        for bid in locked:
            target = bid
            seen = {bid}
            while target in locked:
                if target < 0 or target >= len(parents):
                    break
                p = parents[target]
                if p is None or p < 0 or p in seen:
                    break
                seen.add(p)
                target = p
            if target != bid and target >= 0:
                remap[bid] = target
        return remap

    def _apply_stiff_bone_remap(self, data: bytearray, remap: dict[int, int]) -> int:
        if not remap:
            return 0
        changed = 0
        seen_recs: set[tuple[int, int, str]] = set()
        for src in self.viewer.vertex_src:
            v_start, src_idx, layout = src[0], src[1], src[2]
            key = (v_start, src_idx, layout)
            if key in seen_recs:
                continue
            seen_recs.add(key)
            stride = 64 if layout == 'skin64' else (76 if layout == 'blend76' else 40)
            off = v_start + src_idx * stride
            if layout in ('skin64', 'skin40', 'skin48'):
                if off + 20 > len(data):
                    continue
                b0, b1 = struct.unpack_from('>2H', data, off + 16)
                nb0 = remap.get(b0, b0)
                nb1 = remap.get(b1, b1)
                if (nb0, nb1) != (b0, b1):
                    struct.pack_into('>2H', data, off + 16, nb0, nb1)
                    changed += int(nb0 != b0) + int(nb1 != b1)
            else:
                if off + 32 > len(data):
                    continue
                slots = struct.unpack_from('>4H', data, off + 24)
                new_slots = tuple(remap.get(b, b) for b in slots)
                if new_slots != slots:
                    struct.pack_into('>4H', data, off + 24, *new_slots)
                    changed += sum(1 for a, b in zip(slots, new_slots) if a != b)
        return changed

    def reload_model(self):
        if not self.shape_path or not self.shape_path.exists():
            return False
        try:
            (verts, norms, uvs, tris, bones, bpos, bparents,
             vertex_src, src_to_render,
             bone_offsets, bone_locals, bone_quats,
             tri_groups) = load_mesh_from_bdg(self.shape_path)
        except Exception as exc:
            self.statusBar().showMessage(f'Load failed: {exc}', 8000)
            return False
        self.viewer.set_mesh(verts, norms, uvs, tris)
        self.viewer.set_edit_metadata(vertex_src, src_to_render)
        self.viewer.set_skeleton(bpos, bparents)
        self.viewer.set_bone_edit_metadata(bone_offsets, bone_locals, bone_quats)
        self._tri_groups = tri_groups
        self.geom_label.setText(
            f'Triangles: {len(tris):,}  |  Vertices: {len(verts):,}  |  '
            f'Bones: {bones}  |  UVs: {len(uvs):,}'
        )
        self.statusBar().showMessage(
            f'Loaded {len(tris)} triangles, {len(verts)} verts, {bones} bones', 6000,
        )
        if self._preview_mode:
            self.show_game_preview()
        elif self.active_texture_suffix:
            self.show_texture(self.active_texture_suffix)
        if str(self.shape_path) not in self.watcher.files():
            self.watcher.addPath(str(self.shape_path))
        return True

    def show_uv_map(self):
        if not self.shape_path:
            self.statusBar().showMessage('Load a model first.', 4000)
            return
        if not self.viewer.uvs or not self.viewer.triangles:
            self.statusBar().showMessage('No UVs available for this model.', 4000)
            return
        tex_img = None
        for suffix in ('C', 'M', 'B', 'S'):
            try:
                img, _ = decode_texture_image(self.shape_path, suffix)
            except Exception:
                img = None
            if img is not None:
                tex_img = img.convert('RGBA')
                break
        if self._uv_dialog is None:
            self._uv_dialog = UVMapDialog(self)
        self._uv_dialog.populate(
            tex_img, self.viewer.uvs, self.viewer.triangles, self._tri_groups,
        )
        self._uv_dialog.show()
        self._uv_dialog.raise_()
        self._uv_dialog.activateWindow()

    def show_texture(self, suffix: str):
        if not self.shape_path:
            return
        try:
            img, info = decode_texture_image(self.shape_path, suffix)
        except Exception as exc:
            self.statusBar().showMessage(f'Texture decode failed: {exc}', 8000)
            return
        if img is None:
            self.statusBar().showMessage(info, 6000)
            return
        self.active_texture_suffix = suffix
        self._preview_mode = False
        self.viewer.set_texture_image(img)
        self._show_texture_preview(img)
        self.statusBar().showMessage(f'Showing {info}', 6000)

    def show_game_preview(self):
        if not self.shape_path:
            return
        composite, info = self._build_game_preview()
        if composite is None:
            self.statusBar().showMessage(info, 6000)
            return
        self.active_texture_suffix = None
        self._preview_mode = True
        self.viewer.set_texture_image(composite)
        self._show_texture_preview(None)
        self.statusBar().showMessage(f'Game preview: {info}', 6000)

    def _build_game_preview(self) -> tuple[Image.Image | None, str]:
        from PIL import ImageChops
        layers = {}
        for s in ('C', 'B', 'S', 'M'):
            try:
                img, _ = decode_texture_image(self.shape_path, s)
            except Exception as exc:
                return None, f'_{s} decode failed: {exc}'
            if img is not None:
                layers[s] = img
        base = layers.get('C') or layers.get('B')
        if base is None:
            return None, 'No monster (_C) or bump (_B) texture found'
        size = base.size
        out = base.convert('RGBA').copy()

        if 'B' in layers and 'C' in layers:
            bump_lum = layers['B'].resize(size, Image.LANCZOS).convert('L')
            bump_rgb = Image.merge('RGB', (bump_lum, bump_lum, bump_lum))
            shaded = ImageChops.multiply(out.convert('RGB'),
                                         bump_rgb.point(lambda v: 96 + (v * 159) // 255))
            shaded_rgba = shaded.convert('RGBA')
            shaded_rgba.putalpha(out.split()[3])
            out = Image.blend(out, shaded_rgba, 0.45)

        if 'S' in layers:
            shade = layers['S'].resize(size, Image.LANCZOS).convert('L')
            shade_rgb = Image.merge('RGB', (shade, shade, shade))
            mult = ImageChops.multiply(out.convert('RGB'),
                                       shade_rgb.point(lambda v: 160 + (v * 95) // 255))
            mult_rgba = mult.convert('RGBA')
            mult_rgba.putalpha(out.split()[3])
            out = Image.blend(out, mult_rgba, 0.35)

        used = ['C' if 'C' in layers else 'B']
        if 'B' in layers and 'C' in layers: used.append('B')
        if 'S' in layers: used.append('S')
        if self.crit_mass_action and self.crit_mass_action.isChecked() and 'M' in layers:
            mass = layers['M'].resize(size, Image.LANCZOS).convert('L')
            glow = Image.merge('RGBA', (
                mass.point(lambda v: min(255, int(v * 1.6))),
                mass.point(lambda v: int(v * 0.5)),
                mass.point(lambda v: int(v * 0.1)),
                mass.point(lambda v: int(v * 0.9)),
            ))
            out = Image.alpha_composite(out, glow)
            used.append('M')
        return out, f'{"+".join(used)} {size[0]}x{size[1]}'

    def _on_crit_mass_toggled(self, _checked: bool):
        if self._preview_mode:
            self.show_game_preview()

    def clear_texture(self):
        self.active_texture_suffix = None
        self._preview_mode = False
        if self.crit_mass_action:
            self.crit_mass_action.setChecked(False)
        self.viewer.set_texture_image(None)
        self._show_texture_preview(None)

    def replace_texture(self, suffix: str):
        if not self.shape_path:
            QMessageBox.information(self, 'Texture', 'Open a Shapes.BDG first.')
            return
        stem = re.sub(r'(?i)_shapes\.bdg$', '', self.shape_path.name)
        suggested = str(self.shape_path.parent / f'{stem}_{suffix}.png')
        png_path, _ = QFileDialog.getOpenFileName(
            self,
            f'Choose PNG for _{suffix} ({TEXTURE_SUFFIXES[suffix]})',
            suggested,
            'PNG image (*.png);;All files (*)',
        )
        if not png_path:
            return
        try:
            msg = replace_texture(self.shape_path, suffix, Path(png_path))
        except Exception as exc:
            QMessageBox.critical(self, 'Texture replace failed', str(exc))
            return
        self.statusBar().showMessage(msg, 8000)
        self.show_texture(suffix)

    def _on_file_changed(self, _path: str):
        self._reload_timer.start()

    def run_export(self):
        self._run_tool('bdg_to_fbx_extract_all.py', 'Export')

    def run_import(self):
        self._run_tool('fbx_to_bdg_import_all.py', 'Import')

    def _run_tool(self, script_name: str, label: str):
        script = UTILS_DIR / script_name
        if not script.exists():
            script = TOOL_DIR / script_name
        if not script.exists():
            QMessageBox.critical(self, label, f'Could not find {script_name}')
            return
        self.statusBar().showMessage(f'{label} running...', 0)
        QApplication.processEvents()
        try:
            proc = subprocess.run(
                [sys.executable, str(script), str(self.root), '--all', '--force'],
                cwd=str(self.root), capture_output=True, text=True,
            )
        except Exception as exc:
            QMessageBox.critical(self, label, f'Failed to launch: {exc}')
            self.statusBar().clearMessage()
            return
        if proc.returncode != 0:
            QMessageBox.critical(
                self, f'{label} failed',
                (proc.stderr or proc.stdout or 'Unknown error')[-2000:]
            )
            self.statusBar().showMessage(f'{label} failed', 8000)
            return
        self.statusBar().showMessage(f'{label} finished', 6000)
        self.reload_model()


def main():
    fmt = QSurfaceFormat()
    fmt.setDepthBufferSize(24)
    fmt.setVersion(2, 1)
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication(sys.argv)
    root = Path(os.environ.get('BDG_ROOT', TOOL_DIR)).resolve()
    win = MainWindow(root)
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
