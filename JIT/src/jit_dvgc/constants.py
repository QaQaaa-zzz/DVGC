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
    "root_height",
    "history_valid",
)
ACTOR_FRAME_SIZE = len(ACTOR_FRAME_FIELDS)
ACTOR_TASK_FIELDS = ("jump_signal",)
ACTOR_OBSERVATION_SIZE = 3 * ACTOR_FRAME_SIZE + len(ACTOR_TASK_FIELDS)
PRIVILEGED_OBSERVATION_SIZE = 106

EXPECTED_XML_SHA256 = "0b56d3672773ef05a2b5982117fa53a7fdffcaf2b7f3f04a7a7941233d6e9c8a"
EXPECTED_REFERENCE_SHA256 = "612fe758eb1042481b9c7642cc9b92d3e9c14b4a75c9deaf5340183c928bc41f"

REWARD_COMPONENT_KEYS = (
    "roll",
    "pitch",
    "yaw",
    "speed",
    "survival",
    "height",
    "low_height",
    "action_smoothness",
    "action_magnitude",
    "roll_rate",
    "pitch_rate",
    "yaw_rate",
    "joint_energy",
    "apex_success",
    "illegal_contact",
    "physical_failure",
    "roll_pitch_failure",
    "jump_zone_missed",
    "stuck",
    "yaw_limit",
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
END_STUCK = 8
END_TIMEOUT = 9
END_YAW_LIMIT = 10
END_JUMP_ZONE_MISSED = 11
END_RECOVERY_SUCCESS = 12

END_REASONS = {
    END_ONGOING: "ongoing",
    END_APEX_SUCCESS: "apex_success",
    END_NONFINITE: "nonfinite",
    END_ROLL_LIMIT: "roll_limit",
    END_PITCH_LIMIT: "pitch_limit",
    END_PROHIBITED_CONTACT: "prohibited_contact",
    END_ILLEGAL_WHEEL_CONTACT: "illegal_wheel_contact",
    END_BACKWARD_EXIT: "backward_exit",
    END_STUCK: "stuck",
    END_TIMEOUT: "timeout",
    END_YAW_LIMIT: "yaw_limit",
    END_JUMP_ZONE_MISSED: "jump_zone_missed",
    END_RECOVERY_SUCCESS: "recovery_success",
}
