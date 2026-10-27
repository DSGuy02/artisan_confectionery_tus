"""
waypoints.py - Named Target Waypoints & Coordinate Frame Definitions
Technological University of the Shannon (TUS Athlone)
Integrated Robotics 4 - Continuous Assessment 3
"""

import numpy as np

# ---------------------------------------------------------------------------
# 1. ROBOT DEFAULT JOINT CONFIGURATIONS (UR5e 6-DOF)
# Joint order: [shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3]
# ---------------------------------------------------------------------------
HOME_POSE_JOINTS = [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]
STANDBY_POSE_JOINTS = [0.0, -1.2217, 1.7453, -2.0944, -1.5708, 0.0]

# ---------------------------------------------------------------------------
# 2. CARTESIAN TARGET COORDINATES (Expressed in Robot Base Frame {R0})
# All values in meters: [X, Y, Z, Roll, Pitch, Yaw]
# ---------------------------------------------------------------------------
# Infeed pick zone center (Where chocolates arrive from the conveyor)
INFEED_PICK_TARGET = {
    "approach": [0.0, 0.40, 0.18, 3.14159, 0.0, 0.0],
    "pick":     [0.0, 0.40, 0.05, 3.14159, 0.0, 0.0],
    "retract":  [0.0, 0.40, 0.22, 3.14159, 0.0, 0.0]
}

# 2x2 Blister Tray Presentation Base Target (Corner Slot 1)
# Grid spacing: delta_x = 0.045m, delta_y = 0.045m
TRAY_BASE_TARGET = {
    "X_corner": 0.35,
    "Y_corner": 0.20,
    "Z_place": 0.05,
    "Z_approach": 0.16,
    "delta_x": 0.045,
    "delta_y": 0.045
}

# Pre-computed Cartesian target for each cavity in the 2x2 blister tray
TRAY_SLOT_TARGETS = {
    1: {"name": "Slot 1 (Dark)",  "X": 0.350, "Y": 0.200, "Z_place": 0.050, "Z_approach": 0.160},
    2: {"name": "Slot 2 (Milk)",  "X": 0.395, "Y": 0.200, "Z_place": 0.050, "Z_approach": 0.160},
    3: {"name": "Slot 3 (Ruby)",  "X": 0.350, "Y": 0.245, "Z_place": 0.050, "Z_approach": 0.160},
    4: {"name": "Slot 4 (White)", "X": 0.395, "Y": 0.245, "Z_place": 0.050, "Z_approach": 0.160}
}

# Gravity Remelt Reject Chute (Scrap deposit for chipped or defective chocolates)
REJECT_CHUTE_TARGET = {
    "approach": [-0.35, 0.25, 0.20, 3.14159, 0.0, 0.0],
    "deposit":  [-0.35, 0.25, 0.10, 3.14159, 0.0, 0.0],
    "retract":  [-0.35, 0.25, 0.25, 3.14159, 0.0, 0.0]
}

# ---------------------------------------------------------------------------
# 3. COORDINATE TRANSFORMATION UTILITIES
# ---------------------------------------------------------------------------
TRANSFORM_WORLD_TO_BASE = np.array([
    [1.0, 0.0, 0.0, 0.0],   # Robot base mounted centrally at World origin
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
])

def get_tray_slot_target(slot_id):
    """
    Returns Cartesian targets [X, Y, Z_place, Z_approach] for blister cavity 1..4.
    """
    if slot_id not in TRAY_SLOT_TARGETS:
        raise ValueError(f"Invalid slot ID {slot_id}. Must be between 1 and 4.")
    return TRAY_SLOT_TARGETS[slot_id]


def get_slot_by_variety(chocolate_variety):
    """Maps chocolate variety string to corresponding tray cavity."""
    mapping = {
        "Dark": 1,
        "Milk": 2,
        "Ruby": 3,
        "White": 4
    }
    return mapping.get(chocolate_variety, 1)
