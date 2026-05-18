"""Vertex / bone editing mixin for MeshViewer.

Owns: selection state mutation, grab/rotate transforms, marquee selection,
projection/screen-delta helpers, and write collection. The host class is
expected to provide rendering attributes and an `update()` method.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import sys
from pathlib import Path

_UTILS_DIR = Path(__file__).resolve().parent.parent / 'utils'
if _UTILS_DIR.exists() and str(_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_UTILS_DIR))

import bdg_to_fbx_extract_all as bdg

if TYPE_CHECKING:
    from PyQt6.QtCore import QPoint


class EditModeMixin:
    def set_edit_metadata(self, vertex_src, src_to_render):
        """Provide per-vertex BDG offsets so edit-mode can write changes back."""
        self.vertex_src = vertex_src
        self.src_to_render = src_to_render
        self.selected_verts.clear()
        self._cancel_grab()
        self._undo_stack = []
        self._redo_stack = []
        self.update()

    def _edit_snapshot(self):
        return {
            'vertices': [tuple(v) for v in self.vertices],
            'uvs': [tuple(u) for u in getattr(self, 'uvs', [])],
            'bone_locals': [tuple(t) for t in getattr(self, 'bone_locals', [])],
            'bone_locked': set(getattr(self, 'bone_locked', set())),
            'selected_verts': set(self.selected_verts),
            'selected_bones': set(self.selected_bones),
        }

    def _apply_snapshot(self, snap):
        if not snap:
            return
        self.vertices = [tuple(v) for v in snap['vertices']]
        if 'uvs' in snap and snap['uvs']:
            self.uvs = [tuple(u) for u in snap['uvs']]
        if hasattr(self, 'bone_locals'):
            self.bone_locals = [tuple(t) for t in snap['bone_locals']]
            self._recompute_bone_world_positions()
        self.bone_locked = set(snap['bone_locked'])
        if 'selected_verts' in snap:
            self.selected_verts = set(snap['selected_verts'])
        if 'selected_bones' in snap:
            self.selected_bones = set(snap['selected_bones'])
        self._mesh_dirty = True
        self.update()

    def push_undo(self):
        """Record current state. Caller pushes BEFORE mutating."""
        if not hasattr(self, '_undo_stack'):
            self._undo_stack = []
            self._redo_stack = []
        self._undo_stack.append(self._edit_snapshot())
        if len(self._undo_stack) > 100:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def undo_edit(self) -> bool:
        if not getattr(self, '_undo_stack', None):
            return False
        self._redo_stack.append(self._edit_snapshot())
        self._apply_snapshot(self._undo_stack.pop())
        return True

    def redo_edit(self) -> bool:
        if not getattr(self, '_redo_stack', None):
            return False
        self._undo_stack.append(self._edit_snapshot())
        self._apply_snapshot(self._redo_stack.pop())
        return True

    def set_bone_edit_metadata(self, offsets, locals_, quats):
        self.bone_record_offsets = list(offsets)
        self.bone_locals = list(locals_)
        self.bone_quats = list(quats)
        self.bone_locals_baseline = [tuple(t) for t in locals_]
        self.bone_locked = set()
        self.selected_bones.clear()

    def toggle_bone_lock(self, bid: int) -> bool:
        self.push_undo()
        if bid in self.bone_locked:
            self.bone_locked.discard(bid)
            locked = False
        else:
            self.bone_locked.add(bid)
            locked = True
        self.update()
        return locked

    def reset_bone(self, bid: int):
        if 0 <= bid < len(self.bone_locals_baseline):
            self.push_undo()
            self.bone_locals[bid] = self.bone_locals_baseline[bid]
            self._recompute_bone_world_positions()
            self.update()

    def _filter_unlocked(self, ids):
        return [b for b in ids if b not in self.bone_locked]

    def set_edit_target(self, target: str):
        if target not in ('vertex', 'bone') or target == self.edit_target:
            return
        self._cancel_grab()
        self.edit_target = target
        self.selected_verts.clear()
        self.selected_bones.clear()
        self.update()

    def set_edit_mode(self, on: bool):
        if on == self.edit_mode:
            return
        self.edit_mode = on
        if not on:
            self._cancel_grab()
            self.selected_verts.clear()
            self._marquee_active = False
            self._hover_vert = None
            self._hover_bone = None
        self.update()

    def has_unsaved_edits(self) -> bool:
        # The viewer doesn't track a baseline, so callers determine "dirty"
        # state via their own snapshot. This stub keeps the API simple.
        return False

    def _compute_parent_world_matrix(self, bid: int) -> list[list[float]]:
        """3x3 world rotation of bone `bid`'s parent (identity if root)."""
        parent = self.bone_parents[bid] if 0 <= bid < len(self.bone_parents) else -1
        if parent is None or parent < 0:
            return [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]
        # Walk to root, accumulating rotation matrices.
        chain = []
        p = parent
        while p is not None and p >= 0:
            chain.append(p)
            p = self.bone_parents[p]
        chain.reverse()
        M = [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]
        for idx in chain:
            R = bdg.qmat(self.bone_quats[idx]) if idx < len(self.bone_quats) else [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            # M = M * R
            M = [[sum(M[i][k] * R[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
        return M

    def _recompute_bone_world_positions(self):
        if not self.bone_locals or not self.bone_parents:
            return
        n = len(self.bone_locals)
        globals_: list[list[list[float]] | None] = [None] * n
        children: dict[int, list[int]] = {}
        root_idx = -1
        for i, parent in enumerate(self.bone_parents):
            if parent < 0:
                root_idx = i
            else:
                children.setdefault(parent, []).append(i)
        if root_idx < 0:
            root_idx = 0
        queue = [root_idx]
        visited = set()
        while queue:
            i = queue.pop(0)
            if i in visited:
                continue
            visited.add(i)
            tx, ty, tz = self.bone_locals[i]
            R = bdg.qmat(self.bone_quats[i]) if i < len(self.bone_quats) else [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            local = [
                [R[0][0], R[0][1], R[0][2], tx],
                [R[1][0], R[1][1], R[1][2], ty],
                [R[2][0], R[2][1], R[2][2], tz],
                [0, 0, 0, 1],
            ]
            parent = self.bone_parents[i]
            if parent < 0 or globals_[parent] is None:
                globals_[i] = local
            else:
                globals_[i] = bdg.mm(globals_[parent], local)
            for c in children.get(i, ()):
                queue.append(c)
        new_positions = []
        for m in globals_:
            if m is None:
                new_positions.append((0.0, 0.0, 0.0))
            else:
                new_positions.append((m[0][3], m[1][3], m[2][3]))
        self.bone_positions = new_positions

    def _cancel_grab(self):
        if self._grab_active:
            if self.edit_target == 'bone':
                for bid, t in self._bone_grab_origin_locals.items():
                    self.bone_locals[bid] = t
                self._recompute_bone_world_positions()
            else:
                for vid, pos in self._grab_origin_positions.items():
                    self.vertices[vid] = pos
                self._mesh_dirty = True
        if self._rotate_active:
            for vid, pos in self._rotate_origin_positions.items():
                self.vertices[vid] = pos
            for bid, t in self._rotate_origin_bone_locals.items():
                self.bone_locals[bid] = t
            for bid, q in getattr(self, '_rotate_origin_bone_quats', {}).items():
                self.bone_quats[bid] = q
            if self._rotate_origin_bone_locals:
                self._recompute_bone_world_positions()
            self._mesh_dirty = True
        if getattr(self, '_scale_active', False):
            for vid, pos in getattr(self, '_scale_origin_positions', {}).items():
                self.vertices[vid] = pos
            for bid, t in getattr(self, '_scale_origin_bone_locals', {}).items():
                self.bone_locals[bid] = t
            if getattr(self, '_scale_origin_bone_locals', {}):
                self._recompute_bone_world_positions()
            self._mesh_dirty = True
        self._grab_axis_mode = None
        self._grab_active = False
        self._rotate_active = False
        self._scale_active = False
        self._scale_axis_mode = None
        self._grab_origin_positions.clear()
        self._bone_grab_origin_locals.clear()
        self._rotate_origin_positions.clear()
        self._rotate_origin_bone_locals.clear()
        if hasattr(self, '_scale_origin_positions'):
            self._scale_origin_positions.clear()
        if hasattr(self, '_scale_origin_bone_locals'):
            self._scale_origin_bone_locals.clear()
        if hasattr(self, '_rotate_origin_bone_world'):
            self._rotate_origin_bone_world.clear()
        if hasattr(self, '_rotate_origin_bone_quats'):
            self._rotate_origin_bone_quats.clear()
        if hasattr(self, '_rotate_origin_parent_world'):
            self._rotate_origin_parent_world.clear()
        if hasattr(self, '_rotate_root_bones'):
            self._rotate_root_bones = []
        self._grab_start = None

    def _confirm_grab(self):
        self._grab_axis_mode = None
        self._grab_active = False
        self._rotate_active = False
        self._scale_active = False
        self._scale_axis_mode = None
        self._grab_origin_positions.clear()
        self._bone_grab_origin_locals.clear()
        self._rotate_origin_positions.clear()
        self._rotate_origin_bone_locals.clear()
        if hasattr(self, '_scale_origin_positions'):
            self._scale_origin_positions.clear()
        if hasattr(self, '_scale_origin_bone_locals'):
            self._scale_origin_bone_locals.clear()
        if hasattr(self, '_rotate_origin_bone_world'):
            self._rotate_origin_bone_world.clear()
        if hasattr(self, '_rotate_origin_bone_quats'):
            self._rotate_origin_bone_quats.clear()
        if hasattr(self, '_rotate_origin_parent_world'):
            self._rotate_origin_parent_world.clear()
        if hasattr(self, '_rotate_root_bones'):
            self._rotate_root_bones = []
        self._grab_start = None

    def _project_vertex(self, v):
        """Project a world-space vertex to window pixel coords."""
        if not (self._cached_view_matrix and self._cached_proj_matrix and self._cached_viewport):
            return None
        mv = self._cached_view_matrix
        pr = self._cached_proj_matrix
        x, y, z = v
        ex = mv[0] * x + mv[4] * y + mv[8] * z + mv[12]
        ey = mv[1] * x + mv[5] * y + mv[9] * z + mv[13]
        ez = mv[2] * x + mv[6] * y + mv[10] * z + mv[14]
        ew = mv[3] * x + mv[7] * y + mv[11] * z + mv[15]
        cx = pr[0] * ex + pr[4] * ey + pr[8] * ez + pr[12] * ew
        cy = pr[1] * ex + pr[5] * ey + pr[9] * ez + pr[13] * ew
        cz = pr[2] * ex + pr[6] * ey + pr[10] * ez + pr[14] * ew
        cw = pr[3] * ex + pr[7] * ey + pr[11] * ez + pr[15] * ew
        if abs(cw) < 1e-9:
            return None
        ndc_x = cx / cw
        ndc_y = cy / cw
        ndc_z = cz / cw
        if ndc_z < -1.0 or ndc_z > 1.0:
            return None
        vx, vy, vw, vh = self._cached_viewport
        sx = vx + (ndc_x * 0.5 + 0.5) * vw
        sy = vy + (1.0 - (ndc_y * 0.5 + 0.5)) * vh
        return sx, sy, ndc_z

    def _screen_delta_to_world(self, dx_pix: float, dy_pix: float) -> tuple[float, float, float]:
        """Convert a pixel-space drag at the model's depth into world-space delta."""
        if not self._cached_proj_matrix:
            return (0.0, 0.0, 0.0)
        h = max(1, self.height())
        dist = self.radius * 2.5 * self.zoom
        fov = math.radians(45.0)
        world_per_pix = (2.0 * dist * math.tan(fov / 2.0)) / h
        dx_world = dx_pix * world_per_pix
        dy_world = -dy_pix * world_per_pix
        yaw_rad = math.radians(self.yaw)
        pitch_rad = math.radians(self.pitch)
        cy_, sy_ = math.cos(yaw_rad), math.sin(yaw_rad)
        cp_, sp_ = math.cos(pitch_rad), math.sin(pitch_rad)
        right = (cy_, 0.0, sy_)
        up = (sy_ * sp_, cp_, -cy_ * sp_)
        m = self.model_rot

        def inv(v):
            return (
                m[0] * v[0] + m[3] * v[1] + m[6] * v[2],
                m[1] * v[0] + m[4] * v[1] + m[7] * v[2],
                m[2] * v[0] + m[5] * v[1] + m[8] * v[2],
            )

        rw = inv(right)
        uw = inv(up)
        return (
            rw[0] * dx_world + uw[0] * dy_world,
            rw[1] * dx_world + uw[1] * dy_world,
            rw[2] * dx_world + uw[2] * dy_world,
        )

    def update_hover(self, sx: float, sy: float) -> bool:
        """Refresh `_hover_vert` / `_hover_bone` from a screen-space cursor.
        Returns True if the hover target changed (caller should redraw)."""
        if not self.edit_mode:
            changed = (self._hover_vert is not None) or (self._hover_bone is not None)
            self._hover_vert = None
            self._hover_bone = None
            return changed
        kind, ident = self._pick_at(sx, sy)
        new_v = ident if kind == 'vertex' else None
        new_b = ident if kind == 'bone' else None
        changed = (new_v != self._hover_vert) or (new_b != self._hover_bone)
        self._hover_vert = new_v
        self._hover_bone = new_b
        return changed

    def _pick_at(self, sx: float, sy: float, max_pixels: float = 14.0):
        """Return (kind, id) under the cursor or (None, None).

        kind is 'vertex' or 'bone' depending on edit_target. The closest hit
        within max_pixels (in screen space) wins; depth is used as a tiebreaker.
        """
        if not self.edit_mode:
            return None, None
        best = None
        best_score = (max_pixels * max_pixels, 2.0)
        if self.edit_target == 'bone':
            for i, pos in enumerate(self.bone_positions):
                proj = self._project_vertex(pos)
                if proj is None:
                    continue
                px, py, pz = proj
                d2 = (px - sx) ** 2 + (py - sy) ** 2
                if d2 < best_score[0] or (d2 == best_score[0] and pz < best_score[1]):
                    best = ('bone', i)
                    best_score = (d2, pz)
            return (best or (None, None))
        visible = self._visible_source_keys()
        seen = set()
        for vid, src in enumerate(self.vertex_src):
            key = (src[0], src[1])
            if key in seen:
                continue
            seen.add(key)
            if visible and key not in visible:
                continue
            proj = self._project_vertex(self.vertices[vid])
            if proj is None:
                continue
            px, py, pz = proj
            d2 = (px - sx) ** 2 + (py - sy) ** 2
            if d2 < best_score[0] or (d2 == best_score[0] and pz < best_score[1]):
                best = ('vertex', vid)
                best_score = (d2, pz)
        return (best or (None, None))

    def _click_select(self, sx: float, sy: float, ctrl: bool, shift: bool) -> bool:
        """Single-click selection. Returns True if something was hit."""
        self.push_undo()
        kind, ident = self._pick_at(sx, sy)
        if kind is None:
            # No vertex hit -- try edge pick before giving up
            if self.edit_target == 'vertex':
                edge_ids = self._pick_edge_at(sx, sy)
                if edge_ids:
                    target = self.selected_verts
                    if ctrl:
                        if edge_ids.issubset(target):
                            target.difference_update(edge_ids)
                        else:
                            target.update(edge_ids)
                    elif shift:
                        target.update(edge_ids)
                    else:
                        target.clear()
                        target.update(edge_ids)
                    self._edge_cycle_pool = sorted(edge_ids)
                    self._edge_cycle_index = 0
                    self.update()
                    return True
            if not (ctrl or shift):
                if self.edit_target == 'bone':
                    self.selected_bones.clear()
                else:
                    self.selected_verts.clear()
                self._edge_cycle_pool = []
                self.update()
            return False
        if kind == 'bone':
            target = self.selected_bones
            ids = {ident}
        else:
            key = (self.vertex_src[ident][0], self.vertex_src[ident][1])
            ids = set(self.src_to_render.get(key, [ident]))
            target = self.selected_verts
        if ctrl:
            if ids.issubset(target):
                target.difference_update(ids)
            else:
                target.update(ids)
        elif shift:
            target.update(ids)
        else:
            target.clear()
            target.update(ids)
        self._edge_cycle_pool = []
        self.update()
        return True

    def _pick_edge_at(self, sx: float, sy: float, max_pixels: float = 6.0) -> set[int]:
        """Pick the closest mesh edge within max_pixels and return both endpoint render-vertex sets."""
        if not self.triangles or not self.vertex_src:
            return set()
        visible = self._visible_source_keys()
        best_edge: tuple[int, int] | None = None
        best_dist = max_pixels
        seen_edges: set[tuple[int, int]] = set()
        for tri in self.triangles:
            a, b, c = tri
            for v0, v1 in ((a, b), (b, c), (c, a)):
                edge_key = (min(v0, v1), max(v0, v1))
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)
                if visible:
                    src0 = self.vertex_src[v0] if v0 < len(self.vertex_src) else None
                    src1 = self.vertex_src[v1] if v1 < len(self.vertex_src) else None
                    if src0 is None or src1 is None:
                        continue
                    if (src0[0], src0[1]) not in visible or (src1[0], src1[1]) not in visible:
                        continue
                p0 = self._project_vertex(self.vertices[v0]) if v0 < len(self.vertices) else None
                p1 = self._project_vertex(self.vertices[v1]) if v1 < len(self.vertices) else None
                if p0 is None or p1 is None:
                    continue
                d = self._point_to_segment_dist_3d(sx, sy, p0[0], p0[1], p1[0], p1[1])
                if d < best_dist:
                    best_dist = d
                    best_edge = (v0, v1)
        if best_edge is None:
            return set()
        ids: set[int] = set()
        for vid in best_edge:
            src = self.vertex_src[vid]
            key = (src[0], src[1])
            ids.update(self.src_to_render.get(key, [vid]))
        return ids

    def _point_to_segment_dist_3d(self, px, py, ax, ay, bx, by) -> float:
        dx, dy = bx - ax, by - ay
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq < 1e-6:
            return math.sqrt((px - ax) ** 2 + (py - ay) ** 2)
        t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg_len_sq))
        proj_x = ax + t * dx
        proj_y = ay + t * dy
        return math.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)

    def cycle_edge_selection(self) -> bool:
        """Tab-cycle through edge endpoints. Returns True if cycled."""
        pool = getattr(self, '_edge_cycle_pool', [])
        if not pool:
            return False
        idx = getattr(self, '_edge_cycle_index', 0) % len(pool)
        vid = pool[idx]
        src = self.vertex_src[vid]
        key = (src[0], src[1])
        self.selected_verts = set(self.src_to_render.get(key, [vid]))
        self._edge_cycle_index = (idx + 1) % len(pool)
        self.update()
        return True

    def _select_in_marquee(self, additive: bool):
        if not self._marquee_start or not self._marquee_end:
            return
        self.push_undo()
        x0 = min(self._marquee_start.x(), self._marquee_end.x())
        x1 = max(self._marquee_start.x(), self._marquee_end.x())
        y0 = min(self._marquee_start.y(), self._marquee_end.y())
        y1 = max(self._marquee_start.y(), self._marquee_end.y())
        if self.edit_target == 'bone':
            if not additive:
                self.selected_bones.clear()
            for i, pos in enumerate(self.bone_positions):
                proj = self._project_vertex(pos)
                if proj is None:
                    continue
                sx, sy, _ = proj
                if x0 <= sx <= x1 and y0 <= sy <= y1:
                    self.selected_bones.add(i)
            return
        if not additive:
            self.selected_verts.clear()
        visible = self._visible_source_keys()
        seen_src = set()
        for vid, src in enumerate(self.vertex_src):
            key = (src[0], src[1])
            if key in seen_src:
                continue
            seen_src.add(key)
            if visible and key not in visible:
                continue
            proj = self._project_vertex(self.vertices[vid])
            if proj is None:
                continue
            sx, sy, _ = proj
            if x0 <= sx <= x1 and y0 <= sy <= y1:
                for copy_id in self.src_to_render.get(key, [vid]):
                    self.selected_verts.add(copy_id)

    def selection_pivot(self) -> tuple[float, float, float] | None:
        """Centroid of the current selection (vertex or bone target)."""
        if self.edit_target == 'bone':
            ids = [b for b in self.selected_bones if 0 <= b < len(self.bone_positions)]
            if not ids:
                return None
            xs = [self.bone_positions[b][0] for b in ids]
            ys = [self.bone_positions[b][1] for b in ids]
            zs = [self.bone_positions[b][2] for b in ids]
        else:
            ids = [v for v in self.selected_verts if 0 <= v < len(self.vertices)]
            if not ids:
                return None
            xs = [self.vertices[v][0] for v in ids]
            ys = [self.vertices[v][1] for v in ids]
            zs = [self.vertices[v][2] for v in ids]
        n = len(xs)
        return (sum(xs) / n, sum(ys) / n, sum(zs) / n)

    def gizmo_pivot_and_radius(self) -> tuple[tuple[float, float, float], float] | None:
        """Where the rotate gizmo should render, and its world-space radius."""
        if self._rotate_active:
            pivot = self._rotate_pivot
        else:
            if getattr(self, '_gizmo_mode', None) != 'rotate':
                return None
            pivot = self.selection_pivot()
            if pivot is None:
                return None
        radius = max(self.radius * 0.18, 0.05)
        return pivot, radius

    def gizmo_translate_info(self) -> tuple[tuple[float, float, float], float] | None:
        """Pivot + arrow length for the translate gizmo."""
        if self._grab_active and self._gizmo_translate_active:
            pivot = self._grab_pivot_world if hasattr(self, '_grab_pivot_world') and self._grab_pivot_world else self.selection_pivot()
        else:
            if getattr(self, '_gizmo_mode', None) != 'translate':
                return None
            pivot = self.selection_pivot()
        if pivot is None:
            return None
        length = max(self.radius * 0.22, 0.06)
        return pivot, length

    def pick_gizmo_arrow(self, sx: float, sy: float, tol_pixels: float = 8.0) -> str | None:
        """Return 'x'|'y'|'z' if cursor is over that translate arrow."""
        info = self.gizmo_translate_info()
        if info is None:
            return None
        (px, py, pz), length = info
        best = None
        best_d2 = tol_pixels * tol_pixels
        for axis in ('x', 'y', 'z'):
            steps = 20
            for i in range(steps + 1):
                t = i / steps
                if axis == 'x':
                    p3 = (px + length * t, py, pz)
                elif axis == 'y':
                    p3 = (px, py + length * t, pz)
                else:
                    p3 = (px, py, pz + length * t)
                proj = self._project_vertex(p3)
                if proj is None:
                    continue
                spx, spy, _ = proj
                d2 = (spx - sx) ** 2 + (spy - sy) ** 2
                if d2 < best_d2:
                    best_d2 = d2
                    best = axis
        return best

    def begin_translate_drag(self, axis: str, anchor: 'QPoint') -> bool:
        if axis not in ('x', 'y', 'z'):
            return False
        self._begin_grab(anchor)
        if not self._grab_active:
            return False
        self._gizmo_translate_active = True
        self._gizmo_translate_axis = axis
        return True

    def pick_gizmo_ring(self, sx: float, sy: float, tol_pixels: float = 8.0) -> str | None:
        """Return 'x'|'y'|'z' if the cursor is over that ring, else None."""
        info = self.gizmo_pivot_and_radius()
        if info is None:
            return None
        (px, py, pz), radius = info
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
                proj = self._project_vertex(p3)
                if proj is None:
                    continue
                spx, spy, _ = proj
                d2 = (spx - sx) ** 2 + (spy - sy) ** 2
                if d2 < best_d2:
                    best_d2 = d2
                    best = axis
        return best

    def begin_gizmo_drag(self, axis: str, anchor: QPoint) -> bool:
        """Start a rotate constrained to `axis`, anchored at `anchor`."""
        if axis not in ('x', 'y', 'z'):
            return False
        # Reuse _begin_rotate's setup; it captures origins and pivot.
        self._begin_rotate(anchor)
        if not self._rotate_active:
            return False
        self._rotate_axis_mode = axis
        self._gizmo_drag_active = True
        return True

    def _begin_grab(self, anchor: QPoint):
        if self.edit_target == 'bone':
            if not self.selected_bones:
                return
            self.push_undo()
            movable = [bid for bid in self.selected_bones
                       if 0 <= bid < len(self.bone_locals)
                       and bid not in self.bone_locked]
            if not movable:
                return
            self._grab_active = True
            self._grab_start = anchor
            self._bone_grab_origin_locals = {
                bid: self.bone_locals[bid] for bid in movable
            }
            return
        if not self.selected_verts:
            return
        self.push_undo()
        self._grab_active = True
        self._grab_start = anchor
        self._grab_origin_positions = {
            vid: self.vertices[vid] for vid in self.selected_verts
        }

    def _apply_grab(self, current: QPoint):
        if not self._grab_active or not self._grab_start:
            return
        dx = current.x() - self._grab_start.x()
        dy = current.y() - self._grab_start.y()
        wx, wy, wz = self._screen_delta_to_world(dx, dy)
        axis = None
        if getattr(self, '_gizmo_translate_active', False):
            axis = getattr(self, '_gizmo_translate_axis', None)
        elif getattr(self, '_grab_axis_mode', None):
            axis = self._grab_axis_mode
        # Project the world-space delta onto the chosen world axis so the
        # selection slides only along X / Y / Z.
        if axis == 'x':
            wy = 0.0; wz = 0.0
        elif axis == 'y':
            wx = 0.0; wz = 0.0
        elif axis == 'z':
            wx = 0.0; wy = 0.0
        if self.edit_target == 'bone':
            for bid, origin in self._bone_grab_origin_locals.items():
                self.bone_locals[bid] = (origin[0] + wx, origin[1] + wy, origin[2] + wz)
            self._recompute_bone_world_positions()
            self.update()
            return
        for vid, origin in self._grab_origin_positions.items():
            self.vertices[vid] = (origin[0] + wx, origin[1] + wy, origin[2] + wz)
        self._mesh_dirty = True
        self.update()

    def _begin_scale(self, anchor: 'QPoint'):
        """Start a scale operation from the selection pivot."""
        if self.edit_target == 'bone':
            if not self.selected_bones:
                return
            movable = [bid for bid in self.selected_bones
                       if 0 <= bid < len(self.bone_locals)
                       and bid not in self.bone_locked]
            if not movable:
                return
            self.push_undo()
            self._scale_origin_bone_locals = {
                bid: self.bone_locals[bid] for bid in movable
            }
            xs = [self.bone_positions[bid][0] for bid in movable if 0 <= bid < len(self.bone_positions)]
            ys = [self.bone_positions[bid][1] for bid in movable if 0 <= bid < len(self.bone_positions)]
            zs = [self.bone_positions[bid][2] for bid in movable if 0 <= bid < len(self.bone_positions)]
            if not xs:
                return
            self._scale_pivot = (sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs))
        else:
            if not self.selected_verts:
                return
            self.push_undo()
            self._scale_origin_positions = {
                vid: self.vertices[vid] for vid in self.selected_verts
            }
            xs = [self.vertices[v][0] for v in self.selected_verts]
            ys = [self.vertices[v][1] for v in self.selected_verts]
            zs = [self.vertices[v][2] for v in self.selected_verts]
            self._scale_pivot = (sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs))
        self._scale_active = True
        self._scale_axis_mode = None
        self._grab_start = anchor

    def _apply_scale(self, current: 'QPoint'):
        if not self._scale_active or not self._grab_start:
            return
        dx = current.x() - self._grab_start.x()
        # Scale factor: 1.0 at anchor, grows/shrinks with horizontal drag
        factor = 1.0 + dx * 0.005
        if factor < 0.01:
            factor = 0.01
        px, py, pz = self._scale_pivot
        axis = getattr(self, '_scale_axis_mode', None)

        def scale_point(origin):
            ox, oy, oz = origin
            sx = px + (ox - px) * (factor if axis in (None, 'x') else 1.0)
            sy = py + (oy - py) * (factor if axis in (None, 'y') else 1.0)
            sz = pz + (oz - pz) * (factor if axis in (None, 'z') else 1.0)
            return (sx, sy, sz)

        if self.edit_target == 'bone':
            for bid, origin in self._scale_origin_bone_locals.items():
                if 0 <= bid < len(self.bone_positions):
                    world_orig = self.bone_positions[bid]
                    target = scale_point(world_orig)
                    # Adjust local translation by the world delta
                    self.bone_locals[bid] = (
                        origin[0] + (target[0] - world_orig[0]),
                        origin[1] + (target[1] - world_orig[1]),
                        origin[2] + (target[2] - world_orig[2]),
                    )
            self._recompute_bone_world_positions()
        else:
            for vid, origin in self._scale_origin_positions.items():
                self.vertices[vid] = scale_point(origin)
            self._mesh_dirty = True
        self.update()

    def set_scale_axis_mode(self, mode: str):
        """Switch scale axis constraint. Resets positions so new axis applies from zero."""
        if mode not in (None, 'x', 'y', 'z'):
            return
        self._scale_axis_mode = mode
        if self._scale_active:
            for vid, pos in getattr(self, '_scale_origin_positions', {}).items():
                self.vertices[vid] = pos
            for bid, t in getattr(self, '_scale_origin_bone_locals', {}).items():
                self.bone_locals[bid] = t
            if getattr(self, '_scale_origin_bone_locals', {}):
                self._recompute_bone_world_positions()
            self._mesh_dirty = True
            from PyQt6.QtGui import QCursor
            cur = self.mapFromGlobal(QCursor.pos())
            self._grab_start = cur
            self.update()

    def _visible_source_keys(self) -> set[tuple[int, int]]:
        """Set of source-vertex keys whose projected pixel survives the depth
        buffer. The host samples the depth buffer once per frame and stashes
        it on `self._depth_buffer` -- we just compare each vertex's window-z
        to the rasterized z at its pixel.

        Falls back to a face-normal test when no depth buffer is available
        yet (e.g. before the first paint), which still beats picking nothing.
        """
        if not self.triangles or not self.vertex_src:
            return set()
        depth = getattr(self, '_depth_buffer', None)
        if depth is not None:
            return self._visible_keys_from_depth(depth)
        # Pre-paint fallback: face winding.
        cam = self._camera_view_axis()
        out: set[tuple[int, int]] = set()
        verts = self.vertices
        src = self.vertex_src
        n = len(verts)
        for tri in self.triangles:
            a, b, c = tri
            if a >= n or b >= n or c >= n:
                continue
            ax, ay, az = verts[a]
            bx, by, bz = verts[b]
            cx_, cy_, cz_ = verts[c]
            ex, ey, ez = bx - ax, by - ay, bz - az
            fx, fy, fz = cx_ - ax, cy_ - ay, cz_ - az
            nxv = ey * fz - ez * fy
            nyv = ez * fx - ex * fz
            nzv = ex * fy - ey * fx
            if nxv * cam[0] + nyv * cam[1] + nzv * cam[2] < 0.0:
                for vid in (a, b, c):
                    s = src[vid]
                    out.add((s[0], s[1]))
        return out

    def _project_vertex_window(self, v):
        """Like _project_vertex but returns window-space z (0..1) for depth
        comparison, plus integer pixel coords."""
        if not (self._cached_view_matrix and self._cached_proj_matrix and self._cached_viewport):
            return None
        mv = self._cached_view_matrix
        pr = self._cached_proj_matrix
        x, y, z = v
        ex = mv[0] * x + mv[4] * y + mv[8] * z + mv[12]
        ey = mv[1] * x + mv[5] * y + mv[9] * z + mv[13]
        ez = mv[2] * x + mv[6] * y + mv[10] * z + mv[14]
        ew = mv[3] * x + mv[7] * y + mv[11] * z + mv[15]
        cx = pr[0] * ex + pr[4] * ey + pr[8] * ez + pr[12] * ew
        cy = pr[1] * ex + pr[5] * ey + pr[9] * ez + pr[13] * ew
        cz = pr[2] * ex + pr[6] * ey + pr[10] * ez + pr[14] * ew
        cw = pr[3] * ex + pr[7] * ey + pr[11] * ez + pr[15] * ew
        if abs(cw) < 1e-9 or cw <= 0.0:
            return None
        ndc_x = cx / cw
        ndc_y = cy / cw
        ndc_z = cz / cw
        if ndc_z < -1.0 or ndc_z > 1.0:
            return None
        vx, vy, vw, vh = self._cached_viewport
        sx = vx + (ndc_x * 0.5 + 0.5) * vw
        sy_gl = vy + (ndc_y * 0.5 + 0.5) * vh  # GL y-up
        win_z = ndc_z * 0.5 + 0.5
        return sx, sy_gl, win_z

    def _visible_keys_from_depth(self, depth) -> set[tuple[int, int]]:
        buf, dw, dh = depth
        out: set[tuple[int, int]] = set()
        seen: set[tuple[int, int]] = set()
        for vid, src in enumerate(self.vertex_src):
            key = (src[0], src[1])
            if key in seen:
                continue
            seen.add(key)
            proj = self._project_vertex_window(self.vertices[vid])
            if proj is None:
                continue
            sx, sy_gl, win_z = proj
            ix = int(sx)
            iy = int(sy_gl)
            if ix < 0 or iy < 0 or ix >= dw or iy >= dh:
                continue
            d = buf[iy * dw + ix]
            # Depth-buffer precision degrades with distance (non-linear). Use a
            # tolerance that grows with depth so far verts aren't culled by a
            # hard line. Near 0 use a small constant; near 1 (far plane) use up
            # to ~0.01.
            tol = 0.001 + win_z * win_z * 0.004
            if win_z <= d + tol:
                out.add(key)
        return out

    def _camera_view_axis(self) -> tuple[float, float, float]:
        """World-space direction the camera is looking along."""
        yaw_rad = math.radians(self.yaw)
        pitch_rad = math.radians(self.pitch)
        cy_, sy_ = math.cos(yaw_rad), math.sin(yaw_rad)
        cp_, sp_ = math.cos(pitch_rad), math.sin(pitch_rad)
        forward_view = (-sy_ * cp_, sp_, cy_ * cp_)
        m = self.model_rot
        return (
            m[0] * forward_view[0] + m[3] * forward_view[1] + m[6] * forward_view[2],
            m[1] * forward_view[0] + m[4] * forward_view[1] + m[7] * forward_view[2],
            m[2] * forward_view[0] + m[5] * forward_view[1] + m[8] * forward_view[2],
        )

    def _begin_rotate(self, anchor: QPoint):
        if self.edit_target == 'bone':
            if not self.selected_bones:
                return
            movable = [bid for bid in self.selected_bones
                       if 0 <= bid < len(self.bone_locals)
                       and bid not in self.bone_locked]
            if not movable:
                return
            self.push_undo()
            # Only rotate the topmost selected ancestors. Descendants inherit
            # through the bone hierarchy, so rotating both would double up.
            sel_set = set(movable)
            roots = []
            for bid in movable:
                p = self.bone_parents[bid]
                ancestor_selected = False
                while p is not None and p >= 0:
                    if p in sel_set:
                        ancestor_selected = True
                        break
                    p = self.bone_parents[p]
                if not ancestor_selected:
                    roots.append(bid)
            self._rotate_origin_bone_locals = {
                bid: self.bone_locals[bid] for bid in movable
            }
            self._rotate_origin_bone_quats = {
                bid: tuple(self.bone_quats[bid]) for bid in roots
                if 0 <= bid < len(self.bone_quats)
            }
            self._rotate_origin_parent_world = {
                bid: self._compute_parent_world_matrix(bid) for bid in roots
            }
            self._rotate_origin_bone_world = {
                bid: self.bone_positions[bid] for bid in roots
                if 0 <= bid < len(self.bone_positions)
            }
            self._rotate_root_bones = roots
            xs, ys, zs = [], [], []
            for bid in movable:
                if 0 <= bid < len(self.bone_positions):
                    p = self.bone_positions[bid]
                    xs.append(p[0]); ys.append(p[1]); zs.append(p[2])
            if not xs:
                return
            self._rotate_pivot = (sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs))
        else:
            if not self.selected_verts:
                return
            self.push_undo()
            self._rotate_origin_positions = {
                vid: self.vertices[vid] for vid in self.selected_verts
            }
            xs = [self.vertices[v][0] for v in self.selected_verts]
            ys = [self.vertices[v][1] for v in self.selected_verts]
            zs = [self.vertices[v][2] for v in self.selected_verts]
            self._rotate_pivot = (sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs))
        self._rotate_active = True
        self._rotate_axis_mode = 'view'
        self._grab_start = anchor

    def _rotate_axis_world(self) -> tuple[float, float, float]:
        """Active rotation axis in world space, honoring the constraint mode."""
        mode = getattr(self, '_rotate_axis_mode', 'view')
        if mode == 'x':
            return (1.0, 0.0, 0.0)
        if mode == 'y':
            return (0.0, 1.0, 0.0)
        if mode == 'z':
            return (0.0, 0.0, 1.0)
        return self._camera_view_axis()

    def set_rotate_axis_mode(self, mode: str):
        """Switch the active rotation axis (called from X/Y/Z keys).
        If a rotation is in progress, snap origin positions back so the new
        axis is applied from zero."""
        if mode not in ('view', 'x', 'y', 'z'):
            return
        self._rotate_axis_mode = mode
        if self._rotate_active:
            # Reset positions to origin so the constraint takes effect from
            # the current pixel anchor.
            for vid, pos in self._rotate_origin_positions.items():
                self.vertices[vid] = pos
            for bid, t in self._rotate_origin_bone_locals.items():
                self.bone_locals[bid] = t
            for bid, q in getattr(self, '_rotate_origin_bone_quats', {}).items():
                self.bone_quats[bid] = q
            if self._rotate_origin_bone_locals:
                self._recompute_bone_world_positions()
            self._mesh_dirty = True
            from PyQt6.QtGui import QCursor
            cur = self.mapFromGlobal(QCursor.pos())
            self._grab_start = cur
            self.update()

    def _apply_rotate(self, current: QPoint):
        if not self._rotate_active or not self._grab_start:
            return
        dx = current.x() - self._grab_start.x()
        angle = math.radians(dx * 0.5)
        ax, ay, az = self._rotate_axis_world()
        n = math.sqrt(ax * ax + ay * ay + az * az) or 1.0
        ax, ay, az = ax / n, ay / n, az / n
        c = math.cos(angle); s = math.sin(angle); t = 1.0 - c
        R = (
            (t * ax * ax + c,    t * ax * ay - s * az, t * ax * az + s * ay),
            (t * ax * ay + s * az, t * ay * ay + c,    t * ay * az - s * ax),
            (t * ax * az - s * ay, t * ay * az + s * ax, t * az * az + c),
        )
        px, py, pz = self._rotate_pivot

        def rotate_point(p):
            x, y, z = p[0] - px, p[1] - py, p[2] - pz
            return (
                R[0][0] * x + R[0][1] * y + R[0][2] * z + px,
                R[1][0] * x + R[1][1] * y + R[1][2] * z + py,
                R[2][0] * x + R[2][1] * y + R[2][2] * z + pz,
            )

        if self.edit_target == 'bone':
            # Restore originals so each frame is applied to the captured baseline.
            for bid, t in self._rotate_origin_bone_locals.items():
                self.bone_locals[bid] = t
            for bid, q in getattr(self, '_rotate_origin_bone_quats', {}).items():
                self.bone_quats[bid] = q
            self._recompute_bone_world_positions()

            # Quaternion for the rotation: q = (sin(angle/2)*axis, cos(angle/2)).
            half = angle * 0.5
            sh = math.sin(half); ch = math.cos(half)
            world_quat = (ax * sh, ay * sh, az * sh, ch)

            def qmul(a, b):
                ax_, ay_, az_, aw_ = a
                bx_, by_, bz_, bw_ = b
                return (
                    aw_ * bx_ + ax_ * bw_ + ay_ * bz_ - az_ * by_,
                    aw_ * by_ - ax_ * bz_ + ay_ * bw_ + az_ * bx_,
                    aw_ * bz_ + ax_ * by_ - ay_ * bx_ + az_ * bw_,
                    aw_ * bw_ - ax_ * bx_ - ay_ * by_ - az_ * bz_,
                )

            origin_world = getattr(self, '_rotate_origin_bone_world', {})
            origin_quats = getattr(self, '_rotate_origin_bone_quats', {})
            parent_world_rot = getattr(self, '_rotate_origin_parent_world', {})
            for bid in getattr(self, '_rotate_root_bones', []):
                if bid not in origin_quats:
                    continue
                # 1) Compose the world-space rotation onto the bone's quaternion.
                # Parent world rotation P; bone local R0; new world rotation
                # of bone is W * (P * R0) where W is our delta.
                # Express the new local rotation as P^-1 * W * P * R0.
                P = parent_world_rot.get(bid)
                # Convert P to a quaternion.
                def mat_to_quat(M):
                    tr = M[0][0] + M[1][1] + M[2][2]
                    if tr > 0:
                        s = math.sqrt(tr + 1.0) * 2
                        qw = 0.25 * s
                        qx = (M[2][1] - M[1][2]) / s
                        qy = (M[0][2] - M[2][0]) / s
                        qz = (M[1][0] - M[0][1]) / s
                    elif M[0][0] > M[1][1] and M[0][0] > M[2][2]:
                        s = math.sqrt(1.0 + M[0][0] - M[1][1] - M[2][2]) * 2
                        qw = (M[2][1] - M[1][2]) / s
                        qx = 0.25 * s
                        qy = (M[0][1] + M[1][0]) / s
                        qz = (M[0][2] + M[2][0]) / s
                    elif M[1][1] > M[2][2]:
                        s = math.sqrt(1.0 + M[1][1] - M[0][0] - M[2][2]) * 2
                        qw = (M[0][2] - M[2][0]) / s
                        qx = (M[0][1] + M[1][0]) / s
                        qy = 0.25 * s
                        qz = (M[1][2] + M[2][1]) / s
                    else:
                        s = math.sqrt(1.0 + M[2][2] - M[0][0] - M[1][1]) * 2
                        qw = (M[1][0] - M[0][1]) / s
                        qx = (M[0][2] + M[2][0]) / s
                        qy = (M[1][2] + M[2][1]) / s
                        qz = 0.25 * s
                    return (qx, qy, qz, qw)

                pq = mat_to_quat(P) if P else (0.0, 0.0, 0.0, 1.0)
                pq_inv = (-pq[0], -pq[1], -pq[2], pq[3])
                local_delta = qmul(pq_inv, qmul(world_quat, pq))
                new_local_quat = qmul(local_delta, origin_quats[bid])
                # Normalize.
                qx, qy, qz, qw = new_local_quat
                nrm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw) or 1.0
                self.bone_quats[bid] = (qx / nrm, qy / nrm, qz / nrm, qw / nrm)

                # 2) Move the bone so it pivots around the gizmo center.
                origin_pos = origin_world[bid]
                target_world = rotate_point(origin_pos)
                self._recompute_bone_world_positions()
                current_world = self.bone_positions[bid]
                lx, ly, lz = self.bone_locals[bid]
                self.bone_locals[bid] = (
                    lx + (target_world[0] - current_world[0]),
                    ly + (target_world[1] - current_world[1]),
                    lz + (target_world[2] - current_world[2]),
                )
            self._recompute_bone_world_positions()
        else:
            for vid, origin in self._rotate_origin_positions.items():
                self.vertices[vid] = rotate_point(origin)
            self._mesh_dirty = True
        self.update()

    def collect_position_writes(self) -> list[tuple[int, int, str, tuple[float, float, float]]]:
        """For each unique source vertex, return (v_start, src_idx, layout, pos)."""
        out = []
        seen = set()
        for vid, src in enumerate(self.vertex_src):
            key = (src[0], src[1])
            if key in seen:
                continue
            seen.add(key)
            out.append((src[0], src[1], src[2], self.vertices[vid]))
        return out

    def collect_uv_writes(self) -> list[tuple[int, int, str, tuple[float, float]]]:
        """For each unique source vertex, return (v_start, src_idx, layout, uv)."""
        out = []
        seen = set()
        uvs = getattr(self, 'uvs', [])
        if not uvs:
            return out
        for vid, src in enumerate(self.vertex_src):
            key = (src[0], src[1])
            if key in seen:
                continue
            seen.add(key)
            out.append((src[0], src[1], src[2], self.uvs[vid]))
        return out

    def collect_bone_writes(self) -> list[tuple[int, tuple[float, float, float]]]:
        """Per-bone (record_offset, local_translation) pairs for write-back."""
        return [
            (self.bone_record_offsets[i], self.bone_locals[i])
            for i in range(len(self.bone_locals))
            if i < len(self.bone_record_offsets) and self.bone_record_offsets[i]
        ]

    def _build_vertex_adjacency(self) -> dict[int, set[int]]:
        """Adjacency map keyed by render-vertex id: ids that share a triangle.
        Cached because we rebuild only when the topology changes."""
        cached = getattr(self, '_adj_cache', None)
        cached_len = getattr(self, '_adj_cache_len', -1)
        if cached is not None and cached_len == len(self.triangles):
            return cached
        adj: dict[int, set[int]] = {}
        for tri in self.triangles:
            a, b, c = tri
            adj.setdefault(a, set()).update((b, c))
            adj.setdefault(b, set()).update((a, c))
            adj.setdefault(c, set()).update((a, b))
        # Also link every render-copy of a source vertex so islands don't get
        # cut along UV/seam splits.
        for ids in self.src_to_render.values():
            if len(ids) > 1:
                s = set(ids)
                for vid in ids:
                    adj.setdefault(vid, set()).update(s - {vid})
        self._adj_cache = adj
        self._adj_cache_len = len(self.triangles)
        return adj

    def select_linked(self, seed_id: int | None = None):
        """Blender L: flood-fill from the hovered vertex (or current selection)
        across shared triangles. With a seed it grows from there; without, it
        grows the existing selection."""
        adj = self._build_vertex_adjacency()
        if seed_id is None:
            frontier = set(self.selected_verts)
        else:
            frontier = {seed_id}
        if not frontier:
            return
        visible = self._visible_source_keys()
        out: set[int] = set()
        while frontier:
            v = frontier.pop()
            if v in out:
                continue
            out.add(v)
            for nb in adj.get(v, ()):
                if nb in out:
                    continue
                if visible:
                    src = self.vertex_src[nb] if nb < len(self.vertex_src) else None
                    if src is not None and (src[0], src[1]) not in visible:
                        continue
                frontier.add(nb)
        # Pull in every render-copy of each source vertex so the island feels
        # solid when later transformed.
        for vid in list(out):
            src = self.vertex_src[vid] if vid < len(self.vertex_src) else None
            if src is not None:
                out.update(self.src_to_render.get((src[0], src[1]), [vid]))
        self.selected_verts.update(out)
        self.update()

    def select_all_in_target(self):
        if self.edit_target == 'bone':
            self.selected_bones = set(range(len(self.bone_positions)))
        else:
            self.selected_verts.clear()
            seen = set()
            for vid, src in enumerate(self.vertex_src):
                key_src = (src[0], src[1])
                if key_src in seen:
                    continue
                seen.add(key_src)
                for copy_id in self.src_to_render.get(key_src, [vid]):
                    self.selected_verts.add(copy_id)

    # ------------------------------------------------------------------
    # Mirror tool
    # ------------------------------------------------------------------

    def mirror_select(self, axis: str, tolerance: float = 0.01) -> int:
        """Select the mirrored counterpart verts/bones across the given axis.
        Finds vertices on the opposite side of the model center and adds them
        to the selection. Returns the number of new verts added."""
        px, py, pz = self.center
        added = 0

        if self.edit_target == 'bone':
            if not self.selected_bones:
                return 0
            new_bones: set[int] = set()
            for bid in list(self.selected_bones):
                if bid >= len(self.bone_positions):
                    continue
                wp = self.bone_positions[bid]
                # Compute the mirrored position
                if axis == 'x':
                    target = (2.0 * px - wp[0], wp[1], wp[2])
                elif axis == 'y':
                    target = (wp[0], 2.0 * py - wp[1], wp[2])
                else:
                    target = (wp[0], wp[1], 2.0 * pz - wp[2])
                # Find the closest bone to the mirrored position
                best_bid = -1
                best_dist = float('inf')
                tol = self.radius * tolerance
                for i, pos in enumerate(self.bone_positions):
                    if i == bid:
                        continue
                    d = math.sqrt((pos[0]-target[0])**2 + (pos[1]-target[1])**2 + (pos[2]-target[2])**2)
                    if d < best_dist and d < tol:
                        best_dist = d
                        best_bid = i
                if best_bid >= 0:
                    new_bones.add(best_bid)
            added = len(new_bones - self.selected_bones)
            self.selected_bones.update(new_bones)
        else:
            if not self.selected_verts:
                return 0
            # Build a spatial lookup of all verts by source key
            tol = self.radius * tolerance
            new_verts: set[int] = set()
            # Collect mirrored targets from selected verts
            seen_src: set[tuple[int, int]] = set()
            for vid in list(self.selected_verts):
                src = self.vertex_src[vid]
                key = (src[0], src[1])
                if key in seen_src:
                    continue
                seen_src.add(key)
                ox, oy, oz = self.vertices[vid]
                if axis == 'x':
                    target = (2.0 * px - ox, oy, oz)
                elif axis == 'y':
                    target = (ox, 2.0 * py - oy, oz)
                else:
                    target = (ox, oy, 2.0 * pz - oz)
                # Find closest unselected source vert to the mirrored position
                best_vid = -1
                best_dist = float('inf')
                checked: set[tuple[int, int]] = set()
                for i, s in enumerate(self.vertex_src):
                    skey = (s[0], s[1])
                    if skey in checked or skey in seen_src:
                        continue
                    checked.add(skey)
                    vx, vy, vz = self.vertices[i]
                    d = math.sqrt((vx-target[0])**2 + (vy-target[1])**2 + (vz-target[2])**2)
                    if d < best_dist and d < tol:
                        best_dist = d
                        best_vid = i
                if best_vid >= 0:
                    src_match = self.vertex_src[best_vid]
                    match_key = (src_match[0], src_match[1])
                    for copy_id in self.src_to_render.get(match_key, [best_vid]):
                        new_verts.add(copy_id)
            added = len(new_verts - self.selected_verts)
            self.selected_verts.update(new_verts)
        self.update()
        return added

    # ------------------------------------------------------------------
    # Proportional editing helpers
    # ------------------------------------------------------------------

    def _compute_proportional_weights(self, radius: float) -> dict[int, float]:
        """Compute smooth falloff weights for unselected verts within radius
        of the selection centroid. Returns {vid: weight} where weight in (0,1]."""
        pivot = self.selection_pivot()
        if pivot is None:
            return {}
        px, py, pz = pivot
        weights: dict[int, float] = {}
        if self.edit_target == 'bone':
            for bid in range(len(self.bone_positions)):
                if bid in self.selected_bones or bid in self.bone_locked:
                    continue
                bx, by, bz = self.bone_positions[bid]
                d = math.sqrt((bx - px)**2 + (by - py)**2 + (bz - pz)**2)
                if d < radius:
                    t = d / radius
                    w = (1.0 - t * t) ** 2  # smooth falloff
                    if w > 0.001:
                        weights[bid] = w
        else:
            seen = set()
            for vid in range(len(self.vertices)):
                if vid in self.selected_verts:
                    continue
                src = self.vertex_src[vid]
                key = (src[0], src[1])
                if key in seen:
                    continue
                seen.add(key)
                vx, vy, vz = self.vertices[vid]
                d = math.sqrt((vx - px)**2 + (vy - py)**2 + (vz - pz)**2)
                if d < radius:
                    t = d / radius
                    w = (1.0 - t * t) ** 2
                    if w > 0.001:
                        for copy_id in self.src_to_render.get(key, [vid]):
                            weights[copy_id] = w
        return weights

    def _begin_proportional_grab(self, anchor: 'QPoint', radius: float):
        """Start a grab that also affects nearby unselected verts with falloff."""
        if self.edit_target == 'bone':
            if not self.selected_bones:
                return
            self.push_undo()
            movable = [bid for bid in self.selected_bones
                       if 0 <= bid < len(self.bone_locals)
                       and bid not in self.bone_locked]
            if not movable:
                return
            self._grab_active = True
            self._grab_start = anchor
            self._bone_grab_origin_locals = {
                bid: self.bone_locals[bid] for bid in movable
            }
            self._proportional_active = True
            self._proportional_radius = radius
            weights = self._compute_proportional_weights(radius)
            self._proportional_weights = weights
            self._proportional_bone_origins = {
                bid: self.bone_locals[bid] for bid in weights
            }
        else:
            if not self.selected_verts:
                return
            self.push_undo()
            self._grab_active = True
            self._grab_start = anchor
            self._grab_origin_positions = {
                vid: self.vertices[vid] for vid in self.selected_verts
            }
            self._proportional_active = True
            self._proportional_radius = radius
            weights = self._compute_proportional_weights(radius)
            self._proportional_weights = weights
            self._proportional_origins = {
                vid: self.vertices[vid] for vid in weights
            }

    def _begin_proportional_rotate(self, anchor: 'QPoint', radius: float):
        """Start a rotation that also affects nearby unselected verts."""
        self._begin_rotate(anchor)
        if not self._rotate_active:
            return
        self._proportional_active = True
        self._proportional_radius = radius
        weights = self._compute_proportional_weights(radius)
        self._proportional_weights = weights
        if self.edit_target == 'bone':
            self._proportional_bone_origins = {
                bid: self.bone_locals[bid] for bid in weights
            }
        else:
            self._proportional_origins = {
                vid: self.vertices[vid] for vid in weights
            }

    def _begin_proportional_scale(self, anchor: 'QPoint', radius: float):
        """Start a scale that also affects nearby unselected verts."""
        self._begin_scale(anchor)
        if not self._scale_active:
            return
        self._proportional_active = True
        self._proportional_radius = radius
        weights = self._compute_proportional_weights(radius)
        self._proportional_weights = weights
        if self.edit_target == 'bone':
            self._proportional_bone_origins = {
                bid: self.bone_locals[bid] for bid in weights
            }
        else:
            self._proportional_origins = {
                vid: self.vertices[vid] for vid in weights
            }

    def _apply_proportional_grab(self, current: 'QPoint'):
        """Apply grab with proportional falloff to nearby verts."""
        if not self._grab_active or not self._grab_start:
            return
        # First apply normal grab
        self._apply_grab(current)
        # Then apply proportional falloff to nearby verts
        if not getattr(self, '_proportional_active', False):
            return
        dx = current.x() - self._grab_start.x()
        dy = current.y() - self._grab_start.y()
        wx, wy, wz = self._screen_delta_to_world(dx, dy)
        axis = getattr(self, '_grab_axis_mode', None)
        if getattr(self, '_gizmo_translate_active', False):
            axis = getattr(self, '_gizmo_translate_axis', None)
        if axis == 'x':
            wy = 0.0; wz = 0.0
        elif axis == 'y':
            wx = 0.0; wz = 0.0
        elif axis == 'z':
            wx = 0.0; wy = 0.0
        weights = getattr(self, '_proportional_weights', {})
        if self.edit_target == 'bone':
            origins = getattr(self, '_proportional_bone_origins', {})
            for bid, w in weights.items():
                orig = origins.get(bid)
                if orig is None:
                    continue
                self.bone_locals[bid] = (
                    orig[0] + wx * w,
                    orig[1] + wy * w,
                    orig[2] + wz * w,
                )
            self._recompute_bone_world_positions()
        else:
            origins = getattr(self, '_proportional_origins', {})
            for vid, w in weights.items():
                orig = origins.get(vid)
                if orig is None:
                    continue
                self.vertices[vid] = (
                    orig[0] + wx * w,
                    orig[1] + wy * w,
                    orig[2] + wz * w,
                )
            self._mesh_dirty = True
        self.update()

    def _apply_proportional_rotate(self, current: 'QPoint'):
        """Apply rotation with proportional falloff."""
        if not self._rotate_active or not self._grab_start:
            return
        self._apply_rotate(current)
        if not getattr(self, '_proportional_active', False):
            return
        dx = current.x() - self._grab_start.x()
        angle = math.radians(dx * 0.5)
        ax, ay, az = self._rotate_axis_world()
        n = math.sqrt(ax*ax + ay*ay + az*az) or 1.0
        ax, ay, az = ax/n, ay/n, az/n
        px, py, pz = self._rotate_pivot
        weights = getattr(self, '_proportional_weights', {})
        if self.edit_target != 'bone':
            origins = getattr(self, '_proportional_origins', {})
            for vid, w in weights.items():
                orig = origins.get(vid)
                if orig is None:
                    continue
                a = angle * w
                c = math.cos(a); s = math.sin(a); t = 1.0 - c
                R = (
                    (t*ax*ax + c,      t*ax*ay - s*az, t*ax*az + s*ay),
                    (t*ax*ay + s*az,   t*ay*ay + c,    t*ay*az - s*ax),
                    (t*ax*az - s*ay,   t*ay*az + s*ax, t*az*az + c),
                )
                x, y, z = orig[0] - px, orig[1] - py, orig[2] - pz
                self.vertices[vid] = (
                    R[0][0]*x + R[0][1]*y + R[0][2]*z + px,
                    R[1][0]*x + R[1][1]*y + R[1][2]*z + py,
                    R[2][0]*x + R[2][1]*y + R[2][2]*z + pz,
                )
            self._mesh_dirty = True
            self.update()

    def _apply_proportional_scale(self, current: 'QPoint'):
        """Apply scale with proportional falloff."""
        if not self._scale_active or not self._grab_start:
            return
        self._apply_scale(current)
        if not getattr(self, '_proportional_active', False):
            return
        dx = current.x() - self._grab_start.x()
        factor = 1.0 + dx * 0.005
        if factor < 0.01:
            factor = 0.01
        px, py, pz = self._scale_pivot
        axis = getattr(self, '_scale_axis_mode', None)
        weights = getattr(self, '_proportional_weights', {})
        if self.edit_target != 'bone':
            origins = getattr(self, '_proportional_origins', {})
            for vid, w in weights.items():
                orig = origins.get(vid)
                if orig is None:
                    continue
                f = 1.0 + (factor - 1.0) * w
                ox, oy, oz = orig
                sx = px + (ox - px) * (f if axis in (None, 'x') else 1.0)
                sy = py + (oy - py) * (f if axis in (None, 'y') else 1.0)
                sz = pz + (oz - pz) * (f if axis in (None, 'z') else 1.0)
                self.vertices[vid] = (sx, sy, sz)
            self._mesh_dirty = True
            self.update()

    def _cancel_proportional(self):
        """Clean up proportional state."""
        self._proportional_active = False
        self._proportional_weights = {}
        self._proportional_origins = {}
        self._proportional_bone_origins = {}
        self._proportional_radius = 0.0
