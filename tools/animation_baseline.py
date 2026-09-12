from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from fbx_animation_data import read_fbx_actions, read_fbx_bone_translations, write_action_baseline


def find_blender() -> Path | None:
    configured = os.environ.get("BLENDER_EXECUTABLE")
    candidates = [Path(configured)] if configured else []
    discovered = shutil.which("blender")
    if discovered:
        candidates.append(Path(discovered))
    program_files = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
    candidates.extend(sorted(program_files.glob("Blender Foundation/Blender */blender.exe"), reverse=True))
    return next((path for path in candidates if path.is_file()), None)


def build_action_baseline(
    fbx: Path,
    expected_names: list[str],
    output: Path,
    source_actions: dict[str, dict] | None = None,
) -> dict:
    raw_actions = read_fbx_actions(fbx, expected_names)
    baseline_sources = source_actions if source_actions is not None else raw_actions
    blender = find_blender()
    if blender is None:
        write_action_baseline(
            output,
            raw_actions,
            source_actions=baseline_sources,
            raw_actions=raw_actions,
        )
        print("Warning: Blender was not found; edited Action import comparison is not normalized.")
        return {"normalized": False, "bone_translations": read_fbx_bone_translations(fbx)}

    script = Path(__file__).with_name("blender_normalize_actions.py")
    with tempfile.TemporaryDirectory(prefix="gzbridge_action_baseline_") as folder:
        normalized_fbx = Path(folder) / fbx.name
        creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        subprocess.run(
            [
                str(blender),
                "--background",
                "--python",
                str(script),
                "--",
                str(fbx),
                str(normalized_fbx),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            creationflags=creation_flags,
        )
        normalized_actions = read_fbx_actions(normalized_fbx, expected_names)
        bone_translations = read_fbx_bone_translations(normalized_fbx)
    write_action_baseline(
        output,
        normalized_actions,
        source_actions=baseline_sources,
        raw_actions=raw_actions,
    )
    return {"normalized": True, "bone_translations": bone_translations}
