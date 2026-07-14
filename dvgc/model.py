"""Read-only inspection of the user-supplied MuJoCo model."""
from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


def inspect_model(path: str | Path) -> dict[str, Any]:
    """Inspects the supplied XML without creating or rewriting another model."""
    path = Path(path)
    root = ET.fromstring(path.read_text(encoding="utf-8"))
    compiler = root.find("compiler")
    meshdir = compiler.attrib.get("meshdir", "") if compiler is not None else ""

    mesh_assets = []
    asset_root = root.find("asset")
    if asset_root is not None:
        for mesh in asset_root.findall("mesh"):
            filename = mesh.attrib.get("file", "")
            resolved = path.parent / meshdir / filename
            mesh_assets.append(
                {
                    "name": mesh.attrib.get("name"),
                    "file": filename,
                    "resolved_path": str(resolved),
                    "exists_in_current_copy": resolved.is_file(),
                }
            )

    step = next((g for g in root.iter("geom") if g.attrib.get("name") == "step"), None)
    actuators = []
    actuator_root = root.find("actuator")
    if actuator_root is not None:
        for actuator in list(actuator_root):
            actuators.append(
                {
                    "name": actuator.attrib.get("name"),
                    "type": actuator.tag,
                    "joint": actuator.attrib.get("joint"),
                    "ctrlrange": actuator.attrib.get("ctrlrange"),
                    "forcerange": actuator.attrib.get("forcerange"),
                    "kp": actuator.attrib.get("kp"),
                    "kv": actuator.attrib.get("kv"),
                }
            )

    joints = []
    for joint in root.iter("joint"):
        joints.append(
            {
                "name": joint.attrib.get("name"),
                "type": joint.attrib.get("type", "hinge"),
                "range": joint.attrib.get("range"),
                "axis": joint.attrib.get("axis"),
            }
        )

    result: dict[str, Any] = {
        "xml_path": str(path),
        "xml_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "model_name": root.attrib.get("model"),
        "meshdir": meshdir,
        "mesh_assets": mesh_assets,
        "joints": joints,
        "actuators": actuators,
    }
    if step is not None:
        pos = [float(x) for x in step.attrib["pos"].split()]
        size = [float(x) for x in step.attrib["size"].split()]
        result["step"] = {
            "center": pos,
            "half_size": size,
            "front_x": pos[0] - size[0],
            "back_x": pos[0] + size[0],
            "top_z": pos[2] + size[2],
        }
    return result


def save_model_report(model_path: str | Path, out: str | Path) -> None:
    """Writes a read-only structural/hash report for the original XML."""
    Path(out).write_text(json.dumps(inspect_model(model_path), indent=2), encoding="utf-8")
