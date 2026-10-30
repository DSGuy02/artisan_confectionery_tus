"""
Motion package for the Artisan Confectionery Robotic Cell.
Contains Cartesian waypoints, coordinate frame transformations,
and trajectory planning profiles for the Universal Robots UR5e.
"""
from .waypoints import (
    HOME_POSE_JOINTS,
    STANDBY_POSE_JOINTS,
    INFEED_APPROACH_JOINTS,
    INFEED_PICK_JOINTS,
    INFEED_RETRACT_JOINTS,
    TRAY_SLOT_JOINTS,
    REJECT_CHUTE_JOINTS,
    SAFE_RETRACT_JOINTS,
    INFEED_PICK_TARGET,
    REJECT_CHUTE_TARGET,
    TRAY_SLOT_TARGETS,
    get_tray_slot_target,
    get_tray_slot_joints,
    get_slot_by_variety,
    compute_wrist_orientation,
    TRANSFORM_WORLD_TO_BASE,
    TRANSFORM_WORLD_TO_CAMERA,
    TRANSFORM_WORLD_TO_INFEED,
    TRANSFORM_WORLD_TO_TRAY,
    TRANSFORM_WORLD_TO_REJECT,
    forward_kinematics,
    verify_reachability
)
from .trajectory_planner import (
    generate_cartesian_trajectory,
    compute_adaptive_approach_pose
)

__all__ = [
    'HOME_POSE_JOINTS',
    'STANDBY_POSE_JOINTS',
    'INFEED_APPROACH_JOINTS',
    'INFEED_PICK_JOINTS',
    'INFEED_RETRACT_JOINTS',
    'TRAY_SLOT_JOINTS',
    'REJECT_CHUTE_JOINTS',
    'SAFE_RETRACT_JOINTS',
    'INFEED_PICK_TARGET',
    'REJECT_CHUTE_TARGET',
    'TRAY_SLOT_TARGETS',
    'get_tray_slot_target',
    'get_tray_slot_joints',
    'get_slot_by_variety',
    'compute_wrist_orientation',
    'TRANSFORM_WORLD_TO_BASE',
    'TRANSFORM_WORLD_TO_CAMERA',
    'TRANSFORM_WORLD_TO_INFEED',
    'TRANSFORM_WORLD_TO_TRAY',
    'TRANSFORM_WORLD_TO_REJECT',
    'forward_kinematics',
    'verify_reachability',
    'generate_cartesian_trajectory',
    'compute_adaptive_approach_pose'
]
