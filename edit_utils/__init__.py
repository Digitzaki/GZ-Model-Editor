"""Mixins that provide edit-mode and camera behavior to MeshViewer.

The MeshViewer class composes these mixins so the rendering / GL code in
main.py stays focused on draw responsibilities. Mixins assume the viewer
already initializes the relevant state attributes (selection sets, grab
buffers, camera targets, model_rot, etc.).
"""
from .edit_mode import EditModeMixin
from .camera import CameraMixin

__all__ = ['EditModeMixin', 'CameraMixin']
