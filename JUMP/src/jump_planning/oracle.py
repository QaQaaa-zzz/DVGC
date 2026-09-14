"""Complete narrow-obstacle task decisions from consecutive physics samples.

This module has no simulator or learning dependency. Contact flags are supplied
by the runtime's actual contact-pair/force checks; geometric clearance alone is
never treated as either a collision or a successful landing.
"""

from __future__ import annotations

from collections.abc import Mapping
import copy
import math


_EVENT_NAMES = (
    "qualification_time", "trigger_time", "liftoff_time",
    "liftoff_confirmed_time", "front_landing_time", "rear_landing_time",
    "passage_time", "recovery_start_time", "hold_start_time", "terminal_time",
)
_PHYSICAL_SCALARS = (
    "time", "x", "y", "z", "vx", "vy", "vz", "roll", "pitch", "yaw",
    "wx", "wy", "wz", "front_clearance", "rear_clearance", "robot_back_x",
)
_CONTACT_FLAGS = ("front_contact", "rear_contact", "body_contact", "obstacle_contact")


def _finite_number(value):
    try:
        return not isinstance(value, (str, bytes, bool)) and math.isfinite(value)
    except (TypeError, ValueError, OverflowError):
        return False


def _wrapped_angle(value):
    return math.atan2(math.sin(value), math.cos(value))


class TaskMonitor:
    """Latch one auditable decision for one complete task episode.

    ``qualified`` comes from the runtime's continuous grounded-approach check.
    It never disables collision, pose, corridor, or minimum-forward-speed rules.
    Calling ``finish`` declares that the runtime has executed its declared
    horizon; interrupted/error runs must retain an unknown/error receipt instead.
    """

    def __init__(self, config: Mapping):
        self.config = copy.deepcopy(config)
        self.timing = self.config["timing"]
        self.limits = self.config["limits"]
        self.scene = self.config["scene"]
        for key in ("sim_dt", "control_dt", "episode_seconds", "approach_timeout_seconds",
                    "settle_seconds", "airborne_seconds", "recovery_seconds", "hold_seconds"):
            value = self.timing[key]
            if not _finite_number(value) or value <= 0:
                raise ValueError(f"timing.{key} must be finite and positive")
        if self.timing["hold_seconds"] > self.timing["recovery_seconds"]:
            raise ValueError("hold_seconds cannot exceed recovery_seconds")
        self.obstacle_back = self.scene["front_x"] + self.scene["length"]
        self.landing_min = self.obstacle_back + self.scene["landing_near"]
        self.landing_max = self.obstacle_back + self.scene["landing_far"]
        if not all(_finite_number(v) for v in (self.obstacle_back, self.landing_min, self.landing_max)):
            raise ValueError("obstacle and landing bounds must be finite")
        if self.landing_min > self.landing_max:
            raise ValueError("landing region must have ordered bounds")
        self._status = "ongoing"
        self._reason = None
        self._detail = None
        self._events = dict.fromkeys(_EVENT_NAMES)
        self._landing_x = {"front": None, "rear": None}
        self._last_time = None
        self._first_time = None
        self._grounded_after_qualification = False
        self._air_start = None
        self._epsilon = max(1e-12, self.timing["sim_dt"] * 1e-7)

    def _result(self):
        return {
            "status": self._status,
            "done": self._status != "ongoing",
            "reason": self._reason,
            "detail": self._detail,
            **self._events,
            "front_landing_x": self._landing_x["front"],
            "rear_landing_x": self._landing_x["rear"],
        }

    def _end(self, status, reason, time, detail=None):
        self._status = status
        self._reason = reason
        self._detail = detail
        self._events["terminal_time"] = time
        return self._result()

    def _invalid_frame(self, frame):
        if not isinstance(frame, Mapping):
            return "frame is not a mapping"
        if not frame.get("finite", False):
            return "runtime marked physical state nonfinite"
        for key in _PHYSICAL_SCALARS:
            if not _finite_number(frame.get(key)):
                return f"missing or nonfinite {key}"
        for key in _CONTACT_FLAGS:
            if key not in frame or not isinstance(frame[key], bool):
                return f"missing or invalid contact flag {key}"
        for wheel in ("front", "rear"):
            if frame[f"{wheel}_contact"] and not _finite_number(frame.get(f"{wheel}_touch_x")):
                return f"missing or nonfinite {wheel} floor contact position"
        for key in ("qpos", "qvel", "action", "ctrl"):
            if key in frame:
                try:
                    if not all(_finite_number(v) for v in frame[key]):
                        return f"nonfinite {key}"
                except TypeError:
                    return f"invalid {key}"
        return None

    def _physical_failure(self, frame):
        if frame["obstacle_contact"]:
            return "obstacle_contact"
        if frame["body_contact"]:
            return "body_contact"
        if frame["vx"] < self.limits["min_forward_speed"]:
            return "forward_speed_violation"
        if abs(frame["y"]) > self.limits["corridor_half_width"]:
            return "corridor_violation"
        for key in ("roll", "pitch", "yaw"):
            if abs(_wrapped_angle(frame[key])) > self.limits[f"max_{key}"]:
                return f"{key}_limit"
        return None

    def _tracking_good(self, frame):
        return (
            abs(_wrapped_angle(frame["roll"])) <= self.limits["recovery_roll"]
            and abs(_wrapped_angle(frame["pitch"])) <= self.limits["recovery_pitch"]
            and abs(_wrapped_angle(frame["yaw"])) <= self.limits["recovery_yaw"]
            and abs(frame["wx"]) <= self.limits["recovery_rate"]
            and abs(frame["wy"]) <= self.limits["recovery_rate"]
            and abs(frame["wz"]) <= self.limits["recovery_rate"]
            and abs(frame["y"]) <= self.limits["recovery_lateral"]
            and abs(frame["vx"] - self.limits["nominal_speed"]) <= self.limits["recovery_speed_error"]
        )

    def update(self, frame: Mapping, triggered: bool, qualified: bool = True) -> dict:
        """Consume exactly one physical-step frame; return a detached decision."""
        if self._status != "ongoing":
            return self._result()
        invalid = self._invalid_frame(frame)
        if invalid:
            time = frame.get("time") if isinstance(frame, Mapping) else None
            time = time if _finite_number(time) else self._last_time
            return self._end("unknown", "unknown_numerical", time, invalid)
        time = float(frame["time"])
        if self._last_time is not None:
            elapsed = time - self._last_time
            if elapsed <= 0 or abs(elapsed - self.timing["sim_dt"]) > self._epsilon:
                return self._end("unknown", "invalid_physics_timeline", time,
                                 "expected consecutive sim_dt-spaced physics frames")
        if self._first_time is None:
            self._first_time = time
        if triggered and self._events["trigger_time"] is None:
            command_time = frame.get("trigger_command_time")
            if command_time is None:
                command_time = time
            if not _finite_number(command_time):
                return self._end("unknown", "unknown_numerical", time,
                                 "nonfinite trigger_command_time")
            earliest_command = self._last_time if self._last_time is not None else time - self.timing["sim_dt"]
            if command_time > time + self._epsilon or command_time < earliest_command - self._epsilon:
                return self._end("unknown", "invalid_trigger_time", time,
                                 "new trigger must occur within the current physics interval")
            self._events["trigger_time"] = float(command_time)
        self._last_time = time
        failure = self._physical_failure(frame)
        if failure:
            return self._end("failure", failure, time)

        if qualified and self._events["qualification_time"] is None:
            self._events["qualification_time"] = time
        if (self._events["qualification_time"] is not None
                and frame["front_contact"] and frame["rear_contact"]):
            self._grounded_after_qualification = True

        clear = (not frame["front_contact"] and not frame["rear_contact"]
                 and frame["front_clearance"] >= self.limits["airborne_clearance"]
                 and frame["rear_clearance"] >= self.limits["airborne_clearance"])
        if self._events["liftoff_time"] is None:
            if self._grounded_after_qualification and clear:
                if self._air_start is None:
                    self._air_start = time
                if time - self._air_start + self._epsilon >= self.timing["airborne_seconds"]:
                    self._events["liftoff_time"] = self._air_start
                    self._events["liftoff_confirmed_time"] = time
                    trigger_time = self._events["trigger_time"]
                    if trigger_time is None or trigger_time > self._air_start + self._epsilon:
                        return self._end("failure", "premature_liftoff", time)
            else:
                self._air_start = None

        if self._events["liftoff_time"] is not None:
            for wheel in ("front", "rear"):
                event = f"{wheel}_landing_time"
                if frame[f"{wheel}_contact"] and self._events[event] is None:
                    self._events[event] = time
                    touch_x = float(frame[f"{wheel}_touch_x"])
                    self._landing_x[wheel] = touch_x
                    if not self.landing_min <= touch_x <= self.landing_max:
                        return self._end("failure", f"{wheel}_landing_outside_region", time)
            if frame["robot_back_x"] > self.obstacle_back and self._events["passage_time"] is None:
                self._events["passage_time"] = time
            if (self._events["recovery_start_time"] is None
                    and all(self._events[key] is not None for key in
                            ("passage_time", "front_landing_time", "rear_landing_time"))):
                self._events["recovery_start_time"] = time

        recovery_start = self._events["recovery_start_time"]
        if recovery_start is not None:
            if self._tracking_good(frame):
                if self._events["hold_start_time"] is None:
                    self._events["hold_start_time"] = time
            else:
                self._events["hold_start_time"] = None
            if time - recovery_start + self._epsilon >= self.timing["recovery_seconds"]:
                hold_start = self._events["hold_start_time"]
                if hold_start is not None and time - hold_start + self._epsilon >= self.timing["hold_seconds"]:
                    return self._end("success", "complete_task", time)
                return self._end("failure", "recovery_not_recovered", time)
        return self._result()

    def finish(self) -> dict:
        """Close a fully executed episode that did not reach a terminal event."""
        if self._status != "ongoing":
            return self._result()
        if self._last_time is None:
            return self._end("unknown", "empty_episode", None,
                             "no physical frames were evaluated")
        reason = "approach_not_qualified" if self._events["qualification_time"] is None else "episode_timeout"
        return self._end("failure", reason, self._last_time)
