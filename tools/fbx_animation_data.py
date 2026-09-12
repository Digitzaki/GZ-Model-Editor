from __future__ import annotations

import bisect
import gzip
import json
import math
from pathlib import Path

from fbx_to_bdg_import import (
    clean_fbx_object_name,
    euler_xyz_degrees_to_quat,
    find_first,
    object_nodes,
    parse_fbx,
)


FBX_TICKS_PER_SECOND = 46186158000.0


def _object_label(node) -> str:
    if len(node.props) < 2:
        return ""
    return str(node.props[1]).split("\x00", 1)[0]


def _array(node, name: str) -> list:
    child = node.child(name)
    if child is None or not child.props:
        return []
    value = child.props[0]
    return list(value) if isinstance(value, list) else [value]


def _property_value(node, name: str, default=0):
    props = node.child("Properties70")
    if props is None:
        return default
    for child in props.children_named("P"):
        if child.props and str(child.props[0]) == name and child.props:
            return child.props[-1]
    return default


def read_fbx_bone_translations(path: Path) -> dict[str, list[float]]:
    roots, _version = parse_fbx(path)
    translations = {}
    for model in object_nodes(roots, "Model"):
        if len(model.props) < 3 or str(model.props[2]) not in ("LimbNode", "Null"):
            continue
        value = _property_value(model, "Lcl Translation", None)
        if not isinstance(value, (list, tuple)):
            props = model.child("Properties70")
            value = None
            if props is not None:
                for child in props.children_named("P"):
                    if child.props and str(child.props[0]) == "Lcl Translation" and len(child.props) >= 7:
                        value = child.props[-3:]
                        break
        if isinstance(value, (list, tuple)) and len(value) >= 3:
            translations[clean_fbx_object_name(model.props[1])] = [float(item) for item in value[:3]]
    return translations


def _curve_samples(curve) -> tuple[list[float], list[float]]:
    times = [float(value) / FBX_TICKS_PER_SECOND for value in _array(curve, "KeyTime")]
    values = [float(value) for value in _array(curve, "KeyValueFloat")]
    count = min(len(times), len(values))
    samples = sorted(zip(times[:count], values[:count]), key=lambda item: item[0])
    unique = []
    for time, value in samples:
        if unique and abs(time - unique[-1][0]) <= 1.0e-9:
            unique[-1] = (time, value)
        else:
            unique.append((time, value))
    return [item[0] for item in unique], [item[1] for item in unique]


def _sample_curve(times: list[float], values: list[float], time: float, default: float) -> float:
    if not times:
        return float(default)
    if time <= times[0]:
        return values[0]
    if time >= times[-1]:
        return values[-1]
    right = bisect.bisect_right(times, time)
    left = right - 1
    span = times[right] - times[left]
    if span <= 1.0e-12:
        return values[right]
    alpha = (time - times[left]) / span
    return values[left] + (values[right] - values[left]) * alpha


def _channel_keys(axis_curves: dict[str, object], defaults: tuple[float, float, float]):
    decoded = {}
    union = set()
    for axis in "XYZ":
        curve = axis_curves.get(axis)
        times, values = _curve_samples(curve) if curve is not None else ([], [])
        decoded[axis] = (times, values)
        union.update(times)
    keys = []
    for time in sorted(union):
        value = tuple(
            _sample_curve(*decoded[axis], time, defaults[index])
            for index, axis in enumerate("XYZ")
        )
        keys.append((time, value))
    return keys


def _match_action_name(label: str, expected_names: list[str]) -> str:
    clean = label.split("\x00", 1)[0]
    upper = clean.upper()
    if "REST_POSE" in upper:
        return "REST_POSE"
    parts = []
    for part in clean.split("|"):
        for suffix in ("_Layer", ".Layer"):
            if part.endswith(suffix):
                part = part[: -len(suffix)]
        parts.append(part.upper())
    exact = {part for part in parts if part}
    for name in expected_names:
        if str(name).upper() in exact:
            return str(name)
    for name in sorted(expected_names, key=len, reverse=True):
        if str(name).upper() in upper:
            return str(name)
    tail = clean.rsplit("|", 1)[-1]
    for suffix in ("_Layer", ".Layer"):
        if tail.endswith(suffix):
            tail = tail[: -len(suffix)]
    return tail


def read_fbx_actions(path: Path, expected_names: list[str] | None = None) -> dict[str, dict]:
    roots, version = parse_fbx(Path(path))
    expected_names = [str(name) for name in (expected_names or [])]
    objects = {
        int(node.props[0]): node
        for node in object_nodes(roots)
        if node.props and isinstance(node.props[0], int)
    }
    models = {
        object_id: node
        for object_id, node in objects.items()
        if node.name == "Model" and len(node.props) >= 3 and str(node.props[2]) == "LimbNode"
    }
    stacks = {
        object_id: node for object_id, node in objects.items() if node.name == "AnimationStack"
    }
    layers = {
        object_id: node for object_id, node in objects.items() if node.name == "AnimationLayer"
    }
    curve_nodes = {
        object_id: node for object_id, node in objects.items() if node.name == "AnimationCurveNode"
    }
    curves = {
        object_id: node for object_id, node in objects.items() if node.name == "AnimationCurve"
    }

    oo_by_parent: dict[int, list[int]] = {}
    op_by_source: dict[int, list[tuple[int, str]]] = {}
    op_by_destination: dict[int, list[tuple[int, str]]] = {}
    connections = find_first(roots, "Connections")
    for connection in connections.children if connections is not None else []:
        if connection.name != "C" or len(connection.props) < 3:
            continue
        kind = str(connection.props[0])
        source = int(connection.props[1])
        destination = int(connection.props[2])
        if kind == "OO":
            oo_by_parent.setdefault(destination, []).append(source)
        elif kind == "OP" and len(connection.props) >= 4:
            op_by_source.setdefault(source, []).append((destination, str(connection.props[3])))
            op_by_destination.setdefault(destination, []).append((source, str(connection.props[3])))

    actions = {}
    for stack_id, stack in stacks.items():
        name = _match_action_name(_object_label(stack), expected_names)
        if name == "REST_POSE":
            continue
        layer_ids = [child for child in oo_by_parent.get(stack_id, []) if child in layers]
        tracks_by_bone: dict[str, dict] = {}
        all_times = []
        for layer_id in layer_ids:
            for curve_node_id in oo_by_parent.get(layer_id, []):
                curve_node = curve_nodes.get(curve_node_id)
                if curve_node is None:
                    continue
                target = next(
                    (
                        (destination, prop)
                        for destination, prop in op_by_source.get(curve_node_id, [])
                        if destination in models and prop in ("Lcl Translation", "Lcl Rotation")
                    ),
                    None,
                )
                if target is None:
                    continue
                model_id, property_name = target
                bone_name = clean_fbx_object_name(_object_label(models[model_id]))
                axis_curves = {}
                for curve_id, axis_property in op_by_destination.get(curve_node_id, []):
                    if curve_id in curves and axis_property.startswith("d|"):
                        axis_curves[axis_property[-1].upper()] = curves[curve_id]
                defaults = tuple(
                    float(_property_value(curve_node, f"d|{axis}", 0.0)) for axis in "XYZ"
                )
                keys = _channel_keys(axis_curves, defaults)
                if not keys:
                    continue
                all_times.extend(time for time, _value in keys)
                track = tracks_by_bone.setdefault(bone_name, {"bone_name": bone_name})
                if property_name == "Lcl Translation":
                    track["translation_keys"] = keys
                else:
                    track["rotation_euler_keys"] = keys
                    track["rotation_keys"] = [
                        (time, euler_xyz_degrees_to_quat(*value)) for time, value in keys
                    ]

        if not tracks_by_bone:
            continue
        start = min(all_times) if all_times else 0.0
        stop = max(all_times) if all_times else start
        declared_start = float(_property_value(stack, "LocalStart", 0)) / FBX_TICKS_PER_SECOND
        declared_stop = float(_property_value(stack, "LocalStop", 0)) / FBX_TICKS_PER_SECOND
        if math.isfinite(declared_start) and math.isfinite(declared_stop) and declared_stop > declared_start:
            start = min(start, declared_start)
            stop = max(stop, declared_stop)
        for track in tracks_by_bone.values():
            for key_name in ("translation_keys", "rotation_euler_keys", "rotation_keys"):
                if key_name in track:
                    track[key_name] = [(time - start, value) for time, value in track[key_name]]
        actions[name] = {
            "name": name,
            "source_stack": _object_label(stack),
            "duration": max(0.0, stop - start),
            "tracks": tracks_by_bone,
            "fbx_version": version,
        }
    return actions


def write_action_baseline(
    path: Path,
    actions: dict[str, dict],
    source_actions: dict[str, dict] | None = None,
    raw_actions: dict[str, dict] | None = None,
) -> None:
    fps = 25.0
    def compact(action_map: dict[str, dict]) -> dict[str, dict]:
        compact_actions = {}
        for name, action in action_map.items():
            duration = float(action.get("duration", 0.0))
            frame_count = max(1, int(math.ceil(duration * fps)) + 1)
            times = [min(duration, frame / fps) for frame in range(frame_count)]
            compact_tracks = {}
            for bone_name, track in action.get("tracks", {}).items():
                compact_track = {}
                translation = track.get("translation_keys") or []
                rotation = track.get("rotation_euler_keys") or []
                if translation:
                    compact_track["t"] = [
                        [round(float(value), 5) for value in _sample_curve_keys(translation, time)]
                        for time in times
                    ]
                if rotation:
                    compact_track["r"] = [
                        [round(float(value), 5) for value in _sample_curve_keys(rotation, time)]
                        for time in times
                    ]
                if compact_track:
                    compact_tracks[bone_name] = compact_track
            compact_actions[name] = {"duration": duration, "fps": fps, "tracks": compact_tracks}
        return compact_actions

    compact_actions = compact(actions)
    serializable = {"version": 2, "actions": compact_actions}
    if source_actions is not None:
        source_channels = {}
        for name, action in source_actions.items():
            channels = {}
            for bone_name, track in action.get("tracks", {}).items():
                present = []
                if track.get("translation_keys"):
                    present.append("t")
                if track.get("rotation_euler_keys"):
                    present.append("r")
                if present:
                    channels[bone_name] = present
            source_channels[name] = channels
        serializable["version"] = 3
        serializable["source_channels"] = source_channels
    if raw_actions is not None:
        serializable["version"] = 4
        serializable["raw_actions"] = compact(raw_actions)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=9) as stream:
        json.dump(serializable, stream, separators=(",", ":"))


def read_action_baseline(path: Path) -> dict[str, dict]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    version = int(payload.get("version", 0))
    if version == 1:
        return payload.get("actions", {})
    if version not in (2, 3, 4):
        raise ValueError("unsupported FBX Action baseline version")
    def expand(action_map: dict[str, dict]) -> dict[str, dict]:
        expanded = {}
        for name, action in action_map.items():
            fps = float(action.get("fps", 25.0))
            duration = float(action.get("duration", 0.0))
            tracks = {}
            for bone_name, compact_track in action.get("tracks", {}).items():
                track = {"bone_name": bone_name}
                if compact_track.get("t"):
                    track["translation_keys"] = [
                        (min(duration, index / fps), value)
                        for index, value in enumerate(compact_track["t"])
                    ]
                if compact_track.get("r"):
                    track["rotation_euler_keys"] = [
                        (min(duration, index / fps), value)
                        for index, value in enumerate(compact_track["r"])
                    ]
                tracks[bone_name] = track
            expanded[name] = {"name": name, "duration": duration, "tracks": tracks}
        return expanded

    actions = expand(payload.get("actions", {}))
    raw = expand(payload.get("raw_actions", {})) if version == 4 else {}
    if version >= 3:
        for name, action in actions.items():
            action["source_channels"] = payload.get("source_channels", {}).get(name, {})
            if name in raw:
                action["raw_action"] = raw[name]
    return actions


def _sample_curve_keys(keys, time: float):
    times = [float(item[0]) for item in keys]
    values = [item[1] for item in keys]
    if not times:
        return (0.0, 0.0, 0.0)
    if time <= times[0]:
        return values[0]
    if time >= times[-1]:
        return values[-1]
    right = bisect.bisect_right(times, time)
    left = right - 1
    span = times[right] - times[left]
    alpha = 0.0 if span <= 1.0e-12 else (time - times[left]) / span
    return tuple(
        float(a) + (float(b) - float(a)) * alpha for a, b in zip(values[left], values[right])
    )
