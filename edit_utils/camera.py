"""Camera / view-control mixin for MeshViewer.

Owns: framing, view reset, model-rotation accumulation, auto-spin cycling,
and the per-tick smoothing toward target view parameters. Pure logic — no
GL state is touched here.
"""
from __future__ import annotations

import math


class CameraMixin:
    def frame_model(self):
        """Restore the camera to its default angle/zoom/pan so the mesh is back
        in view. Leaves the user's mesh rotation/translation alone -- a second
        F press triggers the full reset_view."""
        self._target_yaw = self.DEFAULT_YAW
        self._target_pitch = self.DEFAULT_PITCH
        self._target_zoom = self.DEFAULT_ZOOM
        self._target_pan_x = self._default_pan_x
        self._target_pan_y = 0.0
        self._frame_action_next = 'reset'
        self.update()

    def toggle_frame_or_reset(self):
        """F hotkey: alternates between framing the model and resetting the
        full view. Any subsequent camera input re-arms it to 'frame'."""
        action = getattr(self, '_frame_action_next', 'frame')
        if action == 'reset':
            self.reset_view()
            self._frame_action_next = 'frame'
        else:
            self.frame_model()

    def _arm_frame_next(self):
        """Called whenever the user moves the camera so the next F frames."""
        self._frame_action_next = 'frame'

    def reset_view(self):
        self._target_yaw = self.DEFAULT_YAW
        self._target_pitch = self.DEFAULT_PITCH
        self._target_zoom = self.DEFAULT_ZOOM
        self._target_pan_x = self._default_pan_x
        self._target_pan_y = 0.0
        self.model_rot = [
            1.0,  0.0, 0.0,
            0.0,  0.0, 1.0,
            0.0, -1.0, 0.0,
        ]
        self.model_pan_x = self.radius * self.DEFAULT_MODEL_PAN_X_FRAC
        self.model_pan_y = 0.0
        self.model_pan_z = self.radius * self.DEFAULT_MODEL_PAN_Z_FRAC
        self._yaw_velocity = 0.0
        self._pitch_velocity = 0.0
        self._spin_stage = 0
        self._frame_action_next = 'frame'
        self.update()

    def _apply_model_rotation(self, axis, angle_deg):
        """Pre-multiply self.model_rot by a rotation around `axis` (world space)."""
        ax, ay, az = axis
        length = math.sqrt(ax * ax + ay * ay + az * az)
        if length < 1e-9 or abs(angle_deg) < 1e-6:
            return
        ax, ay, az = ax / length, ay / length, az / length
        a = math.radians(angle_deg)
        c = math.cos(a)
        s = math.sin(a)
        t = 1.0 - c
        r = [
            t * ax * ax + c,        t * ax * ay - s * az,   t * ax * az + s * ay,
            t * ax * ay + s * az,   t * ay * ay + c,        t * ay * az - s * ax,
            t * ax * az - s * ay,   t * ay * az + s * ax,   t * az * az + c,
        ]
        m = self.model_rot
        nm = [0.0] * 9
        for i in range(3):
            for j in range(3):
                nm[i * 3 + j] = (
                    r[i * 3 + 0] * m[0 * 3 + j]
                    + r[i * 3 + 1] * m[1 * 3 + j]
                    + r[i * 3 + 2] * m[2 * 3 + j]
                )
        self.model_rot = nm
        self.update()

    def cycle_spin(self):
        self._spin_stage = (self._spin_stage + 1) % len(self.SPIN_SPEEDS)
        self._yaw_velocity = self.SPIN_SPEEDS[self._spin_stage]
        return self._spin_stage

    def _tick(self):
        # Auto-spin (yaw) updates the target so it integrates with smoothing.
        moved = False
        if self._spin_stage:
            self._target_yaw = (self._target_yaw + self._yaw_velocity) % 360.0
            moved = True

        # Critically-damped style smoothing toward targets.
        smooth = 0.18
        for attr_cur, attr_tgt in (
            ('yaw', '_target_yaw'),
            ('pitch', '_target_pitch'),
            ('zoom', '_target_zoom'),
            ('pan_x', '_target_pan_x'),
            ('pan_y', '_target_pan_y'),
        ):
            cur = getattr(self, attr_cur)
            tgt = getattr(self, attr_tgt)
            if attr_cur in ('yaw', 'pitch'):
                # Wrap-aware shortest-arc interpolation for angles.
                diff = (tgt - cur + 540.0) % 360.0 - 180.0
                if abs(diff) > 1e-3:
                    setattr(self, attr_cur, (cur + diff * smooth) % 360.0)
                    moved = True
            else:
                if abs(tgt - cur) > 1e-4:
                    setattr(self, attr_cur, cur + (tgt - cur) * smooth)
                    moved = True
                else:
                    setattr(self, attr_cur, tgt)
        if moved:
            self.update()
