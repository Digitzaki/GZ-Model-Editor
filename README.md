# Godzilla Kaiju Tool

A model viewer and editor for Godzilla: Unleashed `.BDG` shape files. View, edit vertices, replace textures, and extract/import models to and from FBX.

---

## Installation

### Pre-built Release

Download the latest `.exe` from the [Releases](../../releases) tab. No additional setup required — just run it.

### Running from Source

**Requirements:**

- Python 3.10+
- pip

**Install dependencies:**

```bash
pip install -r requirements.txt
```

The required packages are:

- Pillow
- PyQt6
- PyOpenGL

**Launch:**

```bash
python main.py
```

---

## How to Use

### Opening Models

Use `File > Open` and select a **folder** containing `_Shapes.BDG` files. The dropdown will populate with each character found in that folder, letting you switch between them.

### View Mode

- **Orbit** — Left-click drag to rotate the camera around the model.
- **Zoom** — Scroll wheel to zoom in/out.
- **Pan** — Middle-click drag to pan the view.
- **Textures** — View and replace textures via the texture panel. Supported formats: PNG (CMPR, RGB565, RGB5A3, I8).
- **Themes** — Switch between visual themes from the View menu.

### Edit Mode

Toggle edit mode to manipulate the mesh directly:

- **Select vertices** — Left-click to select individual vertices; hold Shift to add to selection.
- **Move vertices** — Drag selected vertices to reposition them.
- **Export to FBX** — Extract the model and textures for use in Blender or other 3D software.
- **Import from FBX** — Bring edited meshes back into the BDG format.

### Backups

The tool automatically creates backups of your original `.BDG` files before making any changes, so you can always restore the unmodified data.

---

## Credits

- **ItsAiden66** — Model extract/import source code
