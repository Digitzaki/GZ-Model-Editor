# Godzilla Blender Converter

Godzilla Blender Converter is a small standalone bridge for moving Pipeworks Godzilla model files between the game formats and Blender-friendly FBX projects.

It is meant for quick import/export work without opening the full editor.

## What It Supports

- Export Wii/PS2-style `*_Shapes.BDG` files to FBX.
- Export GameCube `.CMG` files to FBX.
- Import edited FBX files back into copied BDG or CMG files.
- Extract textures into the exported project folder.
- Preserve the original file structure needed for safe writeback.
- Pack DAMM CMG files ending in `_0.cmg`, `_1.cmg`, or `_2.cmg` back into zip files on import. `_3.cmg` stays as a normal CMG file.

BDG export is currently shapes-only. Animation BDGs are not decoded into the exported FBX project.

## Basic Workflow

1. Open `GZ Blender Converter.exe` or run `python bridge_gui.py`.
2. Use the **Export** tab.
3. Pick an input model file:
   - `Character_Shapes.BDG` for Unleashed-style BDG models.
   - `.CMG` for DAMM/GameCube models.
4. Pick an output folder.
5. Click **Export to FBX**.
6. Open the exported FBX in Blender.
7. Edit the model.
8. Use the **Import** tab.
9. Pick the edited FBX.
10. Pick the original BDG, CMG, or CMG zip.
11. Pick an output folder.
12. Click **Import from FBX**.

The importer writes a new copied game file. It does not overwrite the original model.

## Exported Project Folder

Each export goes into its own character folder. A typical export contains:

- `Character.fbx` - the Blender model.
- `import_log.json` - BDG metadata required for importing BDG edits back.
- `textures/` - decoded texture images.

Do not delete `import_log.json` if you plan to import BDG edits back into the game format. It tells the importer where the original vertex streams, textures, scale, and file layout came from. CMG import reads this structure from the original CMG or CMG zip instead.

## Blender Notes

- Edit the exported FBX, then save or export it back as FBX.
- Keep the exported project folder intact.
- For same-topology BDG edits, move existing mesh data rather than deleting `import_log.json` or rebuilding the project from scratch.
- New parts can be attached when the importer has enough original layout data to preserve the file safely, but the original game format still controls what can be written back.

## Command Line

Export:

```powershell
python bridge.py export "C:\path\to\Godzilla_Shapes.BDG" --out "C:\path\to\export_folder"
```

```powershell
python bridge.py export "C:\path\to\Godzilla2K_0.cmg" --out "C:\path\to\export_folder"
```

Import:

```powershell
python bridge.py import "C:\path\to\edited.fbx" --original "C:\path\to\Godzilla_Shapes.BDG" --out "C:\path\to\reimport_folder"
```

```powershell
python bridge.py import "C:\path\to\edited.fbx" --original "C:\path\to\Godzilla2K_0.cmg" --out "C:\path\to\reimport_folder"
```

If the edited FBX is not inside the exported project folder, pass the project folder manually:

```powershell
python bridge.py import "C:\path\to\edited.fbx" --project "C:\path\to\exported_character_folder" --original "C:\path\to\Godzilla_Shapes.BDG" --out "C:\path\to\reimport_folder"
```

## Building The EXE

From inside the `BDG_Blender_Bridge` folder:

```powershell
pyinstaller --onefile --windowed --name "GZ Blender Converter" --icon ".\gz.ico" --add-data ".\gz.ico;." --add-data ".\tools;tools" --hidden-import PIL --hidden-import PIL.Image --hidden-import PIL.ImageFile .\bridge_gui.py
```

`--icon` sets the executable and taskbar icon. The `gz.ico` data entry lets the Tkinter window use the same icon while the program is running.

## Requirements

- Python 3.10+
- PyInstaller only if you want to build the standalone exe.
- Blender for editing the exported FBX files.

The GUI itself uses Tkinter, which is included with the standard Windows Python install.
