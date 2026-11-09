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


def transform_point(T, point):
    """
    Applies a 4x4 homogeneous transformation matrix T to a 3D Cartesian point [X, Y, Z].
    
    Args:
        T (np.ndarray): 4x4 homogeneous transformation matrix.
        point (list or np.ndarray): 3D coordinates [X, Y, Z].
        
    Returns:
        np.ndarray: Transformed 3D point [X', Y', Z'].
    """
    p_hom = np.append(np.asarray(point, dtype=float)[:3], 1.0)
    p_trans = T @ p_hom
    return p_trans[:3]


def invert_transformation(T):
    """
    Computes the exact analytical inverse of a 4x4 rigid-body homogeneous
    transformation matrix T = [[R, p], [0, 1]] using:
        T^-1 = [[R^T, -R^T @ p], [0, 1]]
        
    Args:
        T (np.ndarray): 4x4 homogeneous transformation matrix.
        
    Returns:
        np.ndarray: 4x4 inverted transformation matrix.
    """
    R = T[:3, :3]
    p = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ p
    return T_inv


def get_tray_slot_transform(slot_id, stage="place"):
    """
    Computes the 4x4 homogeneous transformation matrix T_W_slot mapping
    a target blister tray cavity in the world frame {W} from the tray fixture
    reference frame {T} using the transformation chain:
        T_W_slot = T_W_T @ T_T_slot
        
    Args:
        slot_id (int): Cavity pocket index (1, 2, 3, or 4).
        stage (str): 'approach' (Z = 0.180m), 'place' (Z = 0.081m), or 'retract' (Z = 0.220m).
        
    Returns:
        np.ndarray: 4x4 homogeneous transformation matrix in world coordinates.
    """
    if slot_id not in (1, 2, 3, 4):
        raise ValueError(f"Invalid slot ID {slot_id}. Must be 1, 2, 3, or 4.")
        
    z_map = {"approach": 0.180, "place": 0.081, "retract": 0.220}
    z_height = z_map.get(stage, 0.081)
    
    # 2x2 presentation grid spacing: dx = 0.045m, dy = 0.045m relative to Tray Reference {T}
    dx = 0.045 if slot_id in (2, 4) else 0.000
    dy = 0.045 if slot_id in (3, 4) else 0.000
    
    T_T_slot = np.eye(4)
    T_T_slot[:3, 3] = [dx, dy, z_height - 0.040]
    
    return T_W_T @ T_T_slot


# ---------------------------------------------------------------------------
# 2. CALIBRATED UR5e JOINT CONFIGURATIONS (6-DOF)
# Joint order: [shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3]
# All values in radians. Calibrated with sub-millimeter precision in Webots R2025a.
# ---------------------------------------------------------------------------

HOME_POSE_JOINTS    = [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]
STANDBY_POSE_JOINTS = [1.1254, -1.2000, 1.5500, -0.5000, -1.3778, 0.0]

# Infeed inspection stage pick waypoints (Approach -> Plunge Pick -> Lift Retract)
INFEED_APPROACH_JOINTS = [1.2292, -1.7500, 2.2019, -2.0227, -1.5708, 2.8000]
INFEED_PICK_JOINTS     = [1.2292, -1.5696, 2.3579, -2.3591, -1.5708, 2.8000]
INFEED_RETRACT_JOINTS  = [1.2292, -1.7933, 2.1246, -1.9021, -1.5708, 2.8000]

# 2x2 Blister Tray Presentation Slot Joint Poses (4 Cavities)
TRAY_SLOT_JOINTS = {
    1: {  # Slot 1: Dark Chocolate Cube ([0.350, 0.200, Z])
        "approach": [0.1803, -1.7414, 2.1951, -2.0245, -1.5708, 1.7511],
        "place":    [0.1803, -1.5618, 2.3501, -2.3591, -1.5708, 1.7511],
        "retract":  [0.1803, -1.7847, 2.1182, -1.9043, -1.5708, 1.7511]
    },
    2: {  # Slot 2: Milk Chocolate Cylinder ([0.395, 0.200, Z])
        "approach": [0.1612, -1.6361, 2.1048, -2.0395, -1.5708, 1.7320],
        "place":    [0.1612, -1.4685, 2.2488, -2.3511, -1.5708, 1.7320],
        "retract":  [0.1612, -1.6789, 2.0319, -1.9238, -1.5708, 1.7320]
    },
    3: {  # Slot 3: Ruby Chocolate Heart/Prism ([0.350, 0.245, Z])
        "approach": [0.2917, -1.6764, 2.1409, -2.0354, -1.5708, 1.8625],
        "place":    [0.2917, -1.5040, 2.2889, -2.3557, -1.5708, 1.8625],
        "retract":  [0.2917, -1.7195, 2.0665, -1.9178, -1.5708, 1.8625]
    },
    4: {  # Slot 4: White Chocolate Praline ([0.395, 0.245, Z])
        "approach": [0.2627, -1.5807, 2.0519, -2.0421, -1.5708, 1.8335],
        "place":    [0.2627, -1.4198, 2.1908, -2.3418, -1.5708, 1.8335],
        "retract":  [0.2627, -1.6227, 1.9809, -1.9290, -1.5708, 1.8335]
    }
}

# Gravity Remelt Reject Chute (Scrap Remediation)
REJECT_CHUTE_JOINTS = {
    "approach": [2.2045, -1.7119, 2.0601, -1.9191, -1.5708, -2.5079],
    "deposit":  [2.2045, -1.5949, 2.2153, -2.1911, -1.5708, -2.5079],
    "retract":  [2.2045, -1.7119, 2.0601, -1.9191, -1.5708, -2.5079]
}

# Emergency Safe Retract (Vertical clearance before returning home)
SAFE_RETRACT_JOINTS = [1.2292, -1.8165, 2.0619, -1.8162, -1.5708, 2.8000]


# ---------------------------------------------------------------------------
# 3. CARTESIAN TARGET COORDINATES (Expressed in World Frame {W})
# Positions are for the grasp TCP (centre of the Robotiq finger pads), with the
# gripper pointing straight down. Heights are chosen from the cell geometry:
#   infeed pedestal / tray top surface = 0.040 m, chocolate height = 0.020 m.
#   TCP at 0.080 m positions the finger pads squarely across the chocolate sides
#   while providing clearance from the infeed pedestal during finger closure.
#   Reject chute top = 0.100 m, so the piece is released at 0.130 m.
# ---------------------------------------------------------------------------

Z_PICK      = 0.080   # Grasp height on the infeed pedestal / tray surface
Z_APPROACH  = 0.180   # Clearance before the vertical plunge
Z_RETRACT   = 0.220   # Lift height before any lateral transfer
Z_SAFE      = 0.250   # Emergency vertical retract height

INFEED_PICK_TARGET = {
    "approach": [0.0, 0.40, Z_APPROACH, 3.14159, 0.0, 0.0],
    "pick":     [0.0, 0.40, Z_PICK,     3.14159, 0.0, 0.0],
    "retract":  [0.0, 0.40, Z_RETRACT,  3.14159, 0.0, 0.0]
}

TRAY_SLOT_TARGETS = {
    1: {"name": "Slot 1 (Dark)",  "variety": "Dark",  "X": 0.350, "Y": 0.200, "Z_place": Z_PICK + 0.001, "Z_approach": Z_APPROACH},
    2: {"name": "Slot 2 (Milk)",  "variety": "Milk",  "X": 0.395, "Y": 0.200, "Z_place": Z_PICK + 0.001, "Z_approach": Z_APPROACH},
    3: {"name": "Slot 3 (Ruby)",  "variety": "Ruby",  "X": 0.350, "Y": 0.245, "Z_place": Z_PICK + 0.001, "Z_approach": Z_APPROACH},
    4: {"name": "Slot 4 (White)", "variety": "White", "X": 0.395, "Y": 0.245, "Z_place": Z_PICK + 0.001, "Z_approach": Z_APPROACH}
}

REJECT_CHUTE_TARGET = {
    "approach": [-0.350, 0.250, Z_RETRACT, 3.14159, 0.0, 0.0],
    "deposit":  [-0.350, 0.250, 0.130,     3.14159, 0.0, 0.0],
    "retract":  [-0.350, 0.250, Z_RETRACT, 3.14159, 0.0, 0.0]
}

SAFE_RETRACT_TARGET = [0.0, 0.40, Z_SAFE, 3.14159, 0.0, 0.0]


# ---------------------------------------------------------------------------
# 4. KINEMATICS & REACHABILITY HELPER FUNCTIONS
# ---------------------------------------------------------------------------

def compute_wrist_orientation(orientation_deg, base_yaw=2.8000):
    """
    Converts vision-detected arrival angle theta (in degrees) to wrist_3_joint yaw angle.
    A square chocolate has 4-fold symmetry (90-deg period), so orientation theta is
    folded into [-pi/4, +pi/4]. Offsetting base_yaw keeps joint 6 rotation minimal
    (at most +/- 0.785 rad from base_yaw) and avoids 360-degree reverse cable spins.
    """
    yaw_rad = np.radians(orientation_deg)
    while yaw_rad > np.pi / 4.0:
        yaw_rad -= np.pi / 2.0
    while yaw_rad < -np.pi / 4.0:
        yaw_rad += np.pi / 2.0
    return float(base_yaw + yaw_rad)


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


GRASP_TCP_OFFSET   = 0.130   # Gripper base -> centre of finger pads (along gripper +Z)
TOUCH_SENSOR_OFFSET = 0.125  # Gripper base -> TouchSensor pad (matches the .wbt file)


def forward_kinematics_homogeneous(q, tcp_offset=GRASP_TCP_OFFSET):
    """
    Computes the complete 4x4 homogeneous transformation matrix T_W_TCP mapping
    the robot tool centre point {TCP} relative to the robot base / world frame {W}.
    The kinematic chain follows the joint parameters in protos/CellUR5e.proto,
    the toolSlot offset, the TOOL_MOUNT Pose rotation (1 0 0 -1.5708), and the
    Robotiq finger offset along the gripper approach vector +Z.
    
    Args:
        q (list or np.ndarray): 6-DOF joint angles [q1, q2, q3, q4, q5, q6] in radians.
        tcp_offset (float): Tool centre point distance from gripper base flange.
        
    Returns:
        np.ndarray: 4x4 homogeneous transformation matrix T_W_TCP.
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
    T = T @ trans(0, 0.1, 0)                      # toolSlot
    T = T @ rot_axis([1, 0, 0], -1.5708)          # TOOL_MOUNT
    T_tip = T @ trans(0, 0, tcp_offset)
    return T_tip


def forward_kinematics(q, tcp_offset=GRASP_TCP_OFFSET):
    """
    Forward kinematics of the Webots CellUR5e manipulator with the Robotiq 2F-85.
    Returns: (tcp_position [X, Y, Z], gripper rotation matrix 3x3)
      gripper columns: x, y (finger closing direction), z (approach direction)
    """
    T = forward_kinematics_homogeneous(q, tcp_offset)
    return T[:3, 3], T[:3, :3]


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
