"""Immutable public contracts for the independent JIT runtime."""

from __future__ import annotations


SIM_DT = 0.005
CTRL_DT = 0.020
N_SUBSTEPS = 4

ACTION_ORDER = ("steer", "rear_wheel_drive", "hip", "knee")
ACTOR_FRAME_FIELDS = (
    "gravity_x",
    "gravity_y",
    "gravity_z",
    "angular_velocity_x",
    "angular_velocity_y",
    "angular_velocity_z",
    "acceleration_x",
    "acceleration_y",
    "acceleration_z",
    "steering_position",
    "hip_position",
    "knee_position",
    "steering_velocity",
    "hip_velocity",
    "knee_velocity",
    "front_wheel_velocity",
    "rear_wheel_velocity",
    "last_action_steer",
    "last_action_rear_wheel_drive",
    "last_action_hip",
    "last_action_knee",
    "estimated_forward_velocity",
    "obstacle_relative_x",
    "estimated_structure_clearance",
    "front_wheel_support",
    "rear_wheel_support",
    "history_valid",
)
ACTOR_FRAME_SIZE = len(ACTOR_FRAME_FIELDS)
ACTOR_OBSERVATION_SIZE = 3 * ACTOR_FRAME_SIZE
PRIVILEGED_OBSERVATION_SIZE = 114

EXPECTED_XML_SHA256 = "e2762bec49fdce61eff6ad01b6a67925934d8997b53929b0a67ace7f44109192"
EXPECTED_REFERENCE_SHA256 = "612fe758eb1042481b9c7642cc9b92d3e9c14b4a75c9deaf5340183c928bc41f"

REWARD_COMPONENT_KEYS = (
    "drive",
    "window",
    "liftoff",
    "stable_airborne",
    "ascent",
    "clearance",
    "apex_progress",
    "apex_success",
    "attitude",
    "rate",
    "smoothness",
    "action_magnitude",
    "illegal_contact",
    "physical_failure",
    "timeout",
)

END_ONGOING = 0
END_APEX_SUCCESS = 1
END_NONFINITE = 2
END_ROLL_LIMIT = 3
END_PITCH_LIMIT = 4
END_PROHIBITED_CONTACT = 5
END_ILLEGAL_WHEEL_CONTACT = 6
END_BACKWARD_EXIT = 7
END_PLATFORM_OVERRUN = 8
END_TIMEOUT = 9

END_REASONS = {
    END_ONGOING: "ongoing",
    END_APEX_SUCCESS: "apex_success",
    END_NONFINITE: "nonfinite",
    END_ROLL_LIMIT: "roll_limit",
    END_PITCH_LIMIT: "pitch_limit",
    END_PROHIBITED_CONTACT: "prohibited_contact",
    END_ILLEGAL_WHEEL_CONTACT: "illegal_wheel_contact",
    END_BACKWARD_EXIT: "backward_exit",
    END_PLATFORM_OVERRUN: "platform_overrun",
    END_TIMEOUT: "timeout",
}
