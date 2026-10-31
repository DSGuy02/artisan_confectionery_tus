"""
waypoints.py - Named Target Waypoints, Kinematics & Coordinate Frame Definitions
Technological University of the Shannon (TUS Athlone)
Integrated Robotics 4 - Continuous Assessment 3
Author: Oladeji Sanyaolu (A00316054) - Simulation, Kinematics & Motion Lead
"""

import numpy as np

# ---------------------------------------------------------------------------
# 1. COORDINATE FRAME TRANSFORMATIONS (Homogeneous Transformation Matrices)
# Frames:
#   {W}   = World origin (Table center, Z = 0.0)
#   {R0}  = Robot Base plate (mounted at [0.0, 0.0, 0.0])
#   {TCP} = Tool Center Point (equidistant between gripper finger pads)
#   {C}   = Overhead Inspection Camera ([0.0, 0.40, 0.80], looking -Z)
#   {F}   = Infeed Staging Stage Center ([0.0, 0.40, 0.04])
#   {T}   = 2x2 Blister Tray Corner Reference ([0.35, 0.20, 0.04])
#   {Q}   = Gravity Remelt Reject Chute Funnel ([-0.35, 0.25, 0.10])
# ---------------------------------------------------------------------------

T_W_R0 = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
])

T_W_C = np.array([
    [1.0,  0.0,  0.0,  0.00],
    [0.0, -1.0,  0.0,  0.40],
    [0.0,  0.0, -1.0,  0.80],
    [0.0,  0.0,  0.0,  1.00]
])

T_W_F = np.array([
    [1.0, 0.0, 0.0, 0.00],
    [0.0, 1.0, 0.0, 0.40],
    [0.0, 0.0, 1.0, 0.04],
    [0.0, 0.0, 0.0, 1.00]
])

T_W_T = np.array([
    [1.0, 0.0, 0.0, 0.35],
    [0.0, 1.0, 0.0, 0.20],
    [0.0, 0.0, 1.0, 0.04],
    [0.0, 0.0, 0.0, 1.00]
])

T_W_Q = np.array([
    [1.0, 0.0, 0.0, -0.35],
    [0.0, 1.0, 0.0,  0.25],
    [0.0, 0.0, 1.0,  0.10],
    [0.0, 0.0, 0.0,  1.00]
])

# Aliases for backwards compatibility
TRANSFORM_WORLD_TO_BASE = T_W_R0
TRANSFORM_WORLD_TO_CAMERA = T_W_C
TRANSFORM_WORLD_TO_INFEED = T_W_F
TRANSFORM_WORLD_TO_TRAY = T_W_T
TRANSFORM_WORLD_TO_REJECT = T_W_Q


# ---------------------------------------------------------------------------
# 2. CALIBRATED UR5e JOINT CONFIGURATIONS (6-DOF)
# Joint order: [shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3]
# All values in radians. Calibrated with sub-millimeter precision in Webots R2025a.
# ---------------------------------------------------------------------------

HOME_POSE_JOINTS    = [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]
STANDBY_POSE_JOINTS = [1.1254, -1.2000, 1.5500, -0.5000, -1.3778, 0.0]

# Infeed inspection stage pick waypoints (Approach -> Plunge Pick -> Lift Retract)
INFEED_APPROACH_JOINTS = [1.1265, -1.3090, 1.8107, -0.5017, -1.1819, 0.0]
INFEED_PICK_JOINTS     = [1.1315, -1.1700, 1.9101, -0.7400, -1.2013, 0.0]
INFEED_RETRACT_JOINTS  = [1.1234, -1.3532, 1.7541, -0.4009, -1.1696, 0.0]

# 2x2 Blister Tray Presentation Slot Joint Poses (4 Cavities)
TRAY_SLOT_JOINTS = {
    1: {  # Slot 1: Dark Chocolate Cube ([0.350, 0.200, Z])
        "approach": [0.1643, -1.2579, 1.7418, -0.4839, -1.5103, 0.0],
        "place":    [0.1662, -1.1279, 1.8429, -0.7150, -1.5174, 0.0],
        "retract":  [0.1627, -1.2985, 1.6838, -0.3853, -1.5042, 0.0]
    },
    2: {  # Slot 2: Milk Chocolate Cylinder ([0.395, 0.200, Z])
        "approach": [0.1395, -1.1717, 1.6192, -0.4474, -1.4792, 0.0],
        "place":    [0.1417, -1.0522, 1.7180, -0.6658, -1.4884, 0.0],
        "retract":  [0.1377, -1.2091, 1.5621, -0.3530, -1.4716, 0.0]
    },
    3: {  # Slot 3: Ruby Chocolate Heart/Prism ([0.350, 0.245, Z])
        "approach": [0.2701, -1.2062, 1.6691, -0.4629, -1.4832, 0.0],
        "place":    [0.2722, -1.0825, 1.7686, -0.6861, -1.4917, 0.0],
        "retract":  [0.2683, -1.2449, 1.6118, -0.3669, -1.4761, 0.0]
    },
    4: {  # Slot 4: White Chocolate Praline ([0.395, 0.245, Z])
        "approach": [0.2378, -1.1239, 1.5478, -0.4240, -1.4600, 0.0],
        "place":    [0.2402, -1.0098, 1.6459, -0.6361, -1.4706, 0.0],
        "retract":  [0.2359, -1.1594, 1.4908, -0.3314, -1.4515, 0.0]
    }
}

# Gravity Remelt Reject Chute (Scrap Remediation)
REJECT_CHUTE_JOINTS = {
    "approach": [2.1763, -1.2402, 1.6054, -0.3652, -1.4556, 0.0],
    "deposit":  [2.1793, -1.1290, 1.7316, -0.6026, -1.4678, 0.0],
    "retract":  [2.1740, -1.2705, 1.5390, -0.2685, -1.4463, 0.0]
}

# Emergency Safe Retract (Vertical clearance before returning home)
SAFE_RETRACT_JOINTS = [1.1234, -1.3532, 1.7541, -0.4009, -1.1696, 0.0]


# ---------------------------------------------------------------------------
# 3. CARTESIAN TARGET COORDINATES (Expressed in World Frame {W})
# ---------------------------------------------------------------------------

INFEED_PICK_TARGET = {
    "approach": [0.0, 0.40, 0.180, 3.14159, 0.0, 0.0],
    "pick":     [0.0, 0.40, 0.055, 3.14159, 0.0, 0.0],
    "retract":  [0.0, 0.40, 0.220, 3.14159, 0.0, 0.0]
}

TRAY_SLOT_TARGETS = {
    1: {"name": "Slot 1 (Dark)",  "variety": "Dark",  "X": 0.350, "Y": 0.200, "Z_place": 0.055, "Z_approach": 0.180},
    2: {"name": "Slot 2 (Milk)",  "variety": "Milk",  "X": 0.395, "Y": 0.200, "Z_place": 0.055, "Z_approach": 0.180},
    3: {"name": "Slot 3 (Ruby)",  "variety": "Ruby",  "X": 0.350, "Y": 0.245, "Z_place": 0.055, "Z_approach": 0.180},
    4: {"name": "Slot 4 (White)", "variety": "White", "X": 0.395, "Y": 0.245, "Z_place": 0.055, "Z_approach": 0.180}
}

REJECT_CHUTE_TARGET = {
    "approach": [-0.350, 0.250, 0.220, 3.14159, 0.0, 0.0],
    "deposit":  [-0.350, 0.250, 0.120, 3.14159, 0.0, 0.0],
    "retract":  [-0.350, 0.250, 0.250, 3.14159, 0.0, 0.0]
}


# ---------------------------------------------------------------------------
# 4. KINEMATICS & REACHABILITY HELPER FUNCTIONS
# ---------------------------------------------------------------------------

def compute_wrist_orientation(orientation_deg):
    """
    Converts vision-detected arrival angle theta (in degrees) to wrist_3_joint yaw angle.
    Normalizes angle to [-pi/2, +pi/2] for optimal cable wrap and joint clearance.
    """
    yaw_rad = np.radians(orientation_deg)
    while yaw_rad > np.pi / 2.0:
        yaw_rad -= np.pi
    while yaw_rad < -np.pi / 2.0:
        yaw_rad += np.pi
    return float(yaw_rad)


def get_tray_slot_joints(slot_id, stage="place"):
    """
    Returns calibrated 6-DOF joint angles for a target blister tray slot.
    stage can be 'approach', 'place', or 'retract'.
    """
    if slot_id not in TRAY_SLOT_JOINTS:
        raise ValueError(f"Invalid slot ID {slot_id}. Must be 1, 2, 3, or 4.")
    if stage not in ("approach", "place", "retract"):
        raise ValueError(f"Invalid stage '{stage}'. Must be approach, place, or retract.")
    return list(TRAY_SLOT_JOINTS[slot_id][stage])


def get_tray_slot_target(slot_id):
    """Returns Cartesian target dictionary for blister tray slot 1..4."""
    if slot_id not in TRAY_SLOT_TARGETS:
        raise ValueError(f"Invalid slot ID {slot_id}. Must be between 1 and 4.")
    return TRAY_SLOT_TARGETS[slot_id]


def get_slot_by_variety(chocolate_variety):
    """Maps chocolate variety string to corresponding tray cavity number."""
    mapping = {
        "Dark": 1,
        "Milk": 2,
        "Ruby": 3,
        "White": 4
    }
    return mapping.get(chocolate_variety, 1)


def forward_kinematics(q, tool_len=0.14):
    """
    Exact forward kinematics of the Webots CellUR5e manipulator with Robotiq 2F-85.
    Returns: (tcp_position [X, Y, Z], rotation_matrix 3x3)
    """
    def rot_axis(axis, angle):
        ux, uy, uz = axis
        c, s = np.cos(angle), np.sin(angle)
        return np.array([
            [c + ux**2*(1-c), ux*uy*(1-c) - uz*s, ux*uz*(1-c) + uy*s, 0],
            [uy*ux*(1-c) + uz*s, c + uy**2*(1-c), uy*uz*(1-c) - ux*s, 0],
            [uz*ux*(1-c) - uy*s, uz*uy*(1-c) + ux*s, c + uz**2*(1-c), 0],
            [0, 0, 0, 1]
        ])

    def trans(x, y, z):
        return np.array([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]])

    q1, q2, q3, q4, q5, q6 = q
    T = np.eye(4)
    T = T @ trans(0, 0, 0.163) @ rot_axis([0, 0, 1], q1)
    T = T @ trans(0, 0.138, 0) @ rot_axis([0, 1, 0], q2) @ rot_axis([0, 1, 0], 1.570796)
    T = T @ trans(0, -0.131, 0.425) @ rot_axis([0, 1, 0], q3)
    T = T @ trans(0, 0, 0.392) @ rot_axis([0, 1, 0], q4) @ rot_axis([0, 1, 0], 1.570796)
    T = T @ trans(0, 0.127, 0) @ rot_axis([0, 0, 1], q5)
    T = T @ trans(0, 0, 0.1) @ rot_axis([0, 1, 0], q6)
    T = T @ trans(0, 0.1, 0)
    T_tip = T @ trans(0, 0, tool_len)
    return T_tip[:3, 3], T[:3, :3]


def verify_reachability():
    """
    Validates reachability and dexterity envelope across all cell targets.
    UR5e maximum reach radius R = 0.850 m.
    """
    r_max = 0.850
    targets = {
        "Infeed Pick": INFEED_PICK_TARGET["pick"][:3],
        "Tray Slot 1": [TRAY_SLOT_TARGETS[1]["X"], TRAY_SLOT_TARGETS[1]["Y"], TRAY_SLOT_TARGETS[1]["Z_place"]],
        "Tray Slot 2": [TRAY_SLOT_TARGETS[2]["X"], TRAY_SLOT_TARGETS[2]["Y"], TRAY_SLOT_TARGETS[2]["Z_place"]],
        "Tray Slot 3": [TRAY_SLOT_TARGETS[3]["X"], TRAY_SLOT_TARGETS[3]["Y"], TRAY_SLOT_TARGETS[3]["Z_place"]],
        "Tray Slot 4": [TRAY_SLOT_TARGETS[4]["X"], TRAY_SLOT_TARGETS[4]["Y"], TRAY_SLOT_TARGETS[4]["Z_place"]],
        "Reject Chute": REJECT_CHUTE_TARGET["deposit"][:3]
    }
    
    report = {}
    for name, pos in targets.items():
        dist = float(np.linalg.norm(pos))
        ratio = (dist / r_max) * 100.0
        reachable = dist <= r_max
        report[name] = {
            "position_m": pos,
            "distance_m": round(dist, 3),
            "reach_ratio_pct": round(ratio, 1),
            "status": "PASS (Optimal Dexterity)" if 35.0 <= ratio <= 75.0 else ("PASS" if reachable else "FAIL")
        }
    return report
