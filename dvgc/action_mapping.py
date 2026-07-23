"""Single action-to-target convention used by training, evaluation and certification."""
from __future__ import annotations

from typing import Any


def knee_position_target(
    knee_position: Any,
    knee_action: Any,
    *,
    target_delta: float,
    knee_min: float,
    knee_max: float,
    xp: Any,
) -> tuple[Any, Any, Any]:
    """Returns clipped action, incremental target, and requested target change.

    Sign convention:
      * positive action -> smaller XML knee angle.  For the authoritative
        preload key (knee=2.5), this is the launch-extension direction seen
        in the successful reference trajectory;
      * negative action -> larger XML knee angle, toward the contracted key;
      * zero action -> hold the current knee position.

    The function changes only the controller target.  It never changes the XML
    joint range, actuator gains, force range, collision geometry, or keyframe.
    """
    action = xp.clip(knee_action, -1.0, 1.0)
    target = knee_position - action * float(target_delta)
    target = xp.clip(target, float(knee_min), float(knee_max))
    return action, target, target - knee_position
