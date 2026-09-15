"""Selection of real frames; skipped spatial bins never create synthetic states."""
from __future__ import annotations
import math


def observed_slice(x, phase, seen, *, spacing=0.05, x_min=2.5, x_max=3.8):
    if not math.isfinite(x) or not math.isfinite(spacing) or spacing <= 0:
        raise ValueError("finite coordinate and positive spacing required")
    if phase not in {"upstream", "downstream"} or not x_min <= x <= x_max:
        return None
    index = math.floor(x / spacing + 0.5)
    if (phase, index) in seen:
        return None
    return index, index * spacing
