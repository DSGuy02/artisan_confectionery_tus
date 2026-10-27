"""
Motion package for the Artisan Confectionery Robotic Cell.
Contains Cartesian waypoints, coordinate frame transformations,
and trajectory planning profiles for the Universal Robots UR5e.
"""
from .waypoints import (
    HOME_POSE_JOINTS,
    INFEED_PICK_TARGET,
    REJECT_CHUTE_TARGET,
    TRAY_SLOT_TARGETS,
    get_tray_slot_target,
    TRANSFORM_WORLD_TO_BASE
)
from .trajectory_planner import (
    generate_cartesian_trajectory,
    compute_adaptive_approach_pose
)

__all__ = [
    'HOME_POSE_JOINTS',
    'INFEED_PICK_TARGET',
    'REJECT_CHUTE_TARGET',
    'TRAY_SLOT_TARGETS',
    'get_tray_slot_target',
    'TRANSFORM_WORLD_TO_BASE',
    'generate_cartesian_trajectory',
    'compute_adaptive_approach_pose'
]
