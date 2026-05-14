# Unleashed Kaiju Editor (Indev)

A model viewer and editor for Godzilla: Unleashed `.BDG` shape files. View, edit vertices, replace textures, and extract/import models to and from FBX.

---

## Installation

### Pre-built Release

Download the latest `.exe` from the [Releases](https://github.com/Digitzaki/Unleashed-Kaiju-Editor/releases/tag/Indev) tab. No additional setup required - just run it.

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

Camera controls:

| Input | Action |
|-------|--------|
| LMB drag | Orbit camera |
| RMB drag | Rotate model |
| MMB drag | Pan |
| Scroll | Zoom |

Hotkeys:

| Key | Action |
|-----|--------|
| E | Enter edit mode |
| R | Rotate mesh (axis rings) |
| G | Move mesh (axis arrows) |
| F | Frame model / Reset view (alternates) |
| O | Cycle auto-orbit speed |
| Z | Cycle overlays (axes + grid / grid only / off) |
| T | Cycle texture (M / B / C / S / off) |
| W | Cycle wireframe (black / colored / off) |
| M | Toggle Critical Mass texture |
| N | Toggle game preview composite |
| H | Toggle hotkey overlay |

### Edit Mode

Press **E** to enter edit mode. You can edit either vertices or bones.

| Key | Action |
|-----|--------|
| E | Exit edit mode |
| V | Toggle between vertex / bone editing |
| LMB drag | Box-select |
| Shift + LMB drag | Add to selection |
| A | Select all |
| L | Select linked (vertices) / Stiffen bone (bones) |
| G | Grab / move selection |
| G then X/Y/Z | Lock grab to world axis |
| R | Rotate selection |
| R then X/Y/Z | Lock rotation to world axis |
| Shift + R | Reset selected bones to original |
| Enter | Confirm transform |
| Esc | Cancel transform / clear selection |
| Ctrl+Z | Undo |
| Ctrl+Y | Redo |
| H | Toggle hotkey overlay |
| LMB | Orbit camera |
| MMB | Pan |
| RMB | Rotate model |

### Export / Import

- **Export (BDG to FBX)** — Extracts the model and textures for use in Blender or other 3D software.
- **Import (FBX to BDG)** — Brings edited meshes back into the BDG format.

### Backups

The tool automatically creates backups of your original `.BDG` files before making any changes, so you can always restore the unmodified data.

---

## Credits

- **ItsAiden66** — Model extract/import source code

