"""Shared CoT and trajectory checks used before search and serialization."""

import json
from pathlib import Path
import numpy as np

TEMPLATES = Path(__file__).resolve().parents[1] / "prompts" / "templates"
with (TEMPLATES / "navsim_trajcond_format_release.json").open(encoding="utf-8") as stream:
    RELEASE = json.load(stream)


def explanation_body(text):
    if not isinstance(text, str):
        raise ValueError("Explanation must be a string")
    body = text
    while body.startswith("<thinking>") and body.endswith("</thinking>"):
        body = body[len("<thinking>"):-len("</thinking>")]
    if not body.strip() or "<thinking>" in body or "</thinking>" in body:
        raise ValueError("Empty explanation or malformed thinking tags")
    return body


def validate_cot(label):
    if label is None:
        raise ValueError("Missing CoT")
    objects = label.critical_objects
    if objects is None or set(objects) != set(RELEASE["critical_objects"]) or any(v not in {"yes", "no"} for v in objects.values()):
        raise ValueError("CoT must audit all twelve object classes")
    meta = label.meta_behaviour
    if meta is None or set(meta) != {"speed", "command"}:
        raise ValueError("CoT must include speed and command")
    for key in ("speed", "command"):
        if meta[key] not in RELEASE[f"{key}_labels"]["answer"]:
            raise ValueError(f"Invalid CoT {key}")
    explanation_body(label.explanation)


def select_points(points, times):
    """Select exact prescribed times without interpolation or renumbering."""
    selected = []
    for time in times:
        matches = [p for p in (points or []) if abs(p.t - time) <= 1e-6]
        if len(matches) != 1:
            raise ValueError(f"Missing or duplicate trajectory time {time}")
        point = matches[0]
        if not np.isfinite([point.x, point.y, point.heading]).all():
            raise ValueError("Non-finite trajectory")
        selected.append(point)
    return selected


def future_points(points):
    return select_points(points, np.arange(1, 9) * 0.5)
