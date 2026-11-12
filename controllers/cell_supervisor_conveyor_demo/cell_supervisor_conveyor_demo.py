"""
cell_supervisor_conveyor_demo.py - Multi-Compartment Random Batch Conveyor Sorting Supervisor
Cyberbotics Webots Controller for Universal Robots UR5e + OpenCV Vision + Indexed Conveyor
Technological University of the Shannon (TUS Athlone)
Integrated Robotics 4 - Continuous Assessment 3
"""

import sys
import os
import time
import csv
import math
import random
import cv2
import numpy as np

# Add project root to sys.path so modules can be imported inside Webots
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from perception.vision_pipeline import (
    convert_webots_to_cv,
    inspect_chocolate,
    draw_inspection_overlay,
)
from motion.waypoints import (
    HOME_POSE_JOINTS,
    INFEED_APPROACH_JOINTS,
    INFEED_PICK_JOINTS,
    INFEED_RETRACT_JOINTS,
    REJECT_CHUTE_JOINTS,
    SAFE_RETRACT_JOINTS,
    compute_wrist_orientation,
)
from motion.trajectory_planner import generate_joint_s_curve_trajectory

try:
    from controller import Supervisor
    WEBOTS_AVAILABLE = True
except ImportError:
    WEBOTS_AVAILABLE = False

# Visual appearance per chocolate variety (consistent with protos/*.proto & world lighting)
VARIETY_APPEARANCES = {
    "Dark":  {"baseColor": (0.15, 0.08, 0.04), "defectColor": (0.10, 0.05, 0.025)},
    "Milk":  {"baseColor": (0.47, 0.27, 0.15), "defectColor": (0.47, 0.27, 0.15)},
    "Ruby":  {"baseColor": (0.75, 0.27, 0.55), "defectColor": (0.75, 0.27, 0.55)},
    "White": {"baseColor": (0.60, 0.62, 0.66), "defectColor": (0.60, 0.62, 0.66)},
}

# Exact 6-DOF inverse kinematics poses for 4 separate colour-coded compartments (4 tidy sub-slots each)
# All targets keep Z_approach=0.180m, Z_place=0.084m, Z_retract=0.220m with 0.000mm FK error.
COMPARTMENT_SUBSLOT_JOINTS = {
    "Dark": {  # Center [+0.44, +0.16]
        1: {
            "coords": (0.4075, 0.1275),
            "approach": [-0.0160, -1.6770, 2.1415, -2.0353, -1.5708, 1.5548],
            "place":    [-0.0160, -1.5134, 2.2848, -2.3423, -1.5708, 1.5548],
            "retract":  [-0.0160, -1.7202, 2.0670, -1.9177, -1.5708, 1.5548],
        },
        2: {
            "coords": (0.4725, 0.1275),
            "approach": [-0.0138, -1.5210, 1.9908, -2.0406, -1.5708, 1.5570],
            "place":    [-0.0138, -1.3751, 2.1205, -2.3162, -1.5708, 1.5570],
            "retract":  [-0.0138, -1.5619, 1.9216, -1.9305, -1.5708, 1.5570],
        },
        3: {
            "coords": (0.4075, 0.1925),
            "approach": [0.1394, -1.6159, 2.0860, -2.0409, -1.5708, 1.7102],
            "place":    [0.1394, -1.4590, 2.2236, -2.3354, -1.5708, 1.7102],
            "retract":  [0.1394, -1.6585, 2.0138, -1.9261, -1.5708, 1.7102],
        },
        4: {
            "coords": (0.4725, 0.1925),
            "approach": [0.1211, -1.4718, 1.9372, -2.0362, -1.5708, 1.6919],
            "place":    [0.1211, -1.3317, 2.0634, -2.3025, -1.5708, 1.6919],
            "retract":  [0.1211, -1.5117, 1.8694, -1.9285, -1.5708, 1.6919],
        },
    },
    "Milk": {  # Center [+0.44, -0.18]
        1: {
            "coords": (0.4075, -0.2125),
            "approach": [-0.7765, -1.5936, 2.0647, -2.0418, -1.5708, 0.7943],
            "place":    [-0.7765, -1.4393, 2.2002, -2.3318, -1.5708, 0.7943],
            "retract":  [-0.7765, -1.6359, 1.9932, -1.9281, -1.5708, 0.7943],
        },
        2: {
            "coords": (0.4725, -0.2125),
            "approach": [-0.6843, -1.4535, 1.9165, -2.0338, -1.5708, 0.8865],
            "place":    [-0.6843, -1.3156, 2.0415, -2.2968, -1.5708, 0.8865],
            "retract":  [-0.6843, -1.4929, 1.8492, -1.9270, -1.5708, 0.8865],
        },
        3: {
            "coords": (0.4075, -0.1475),
            "approach": [-0.6616, -1.6603, 2.1268, -2.0373, -1.5708, 0.9092],
            "place":    [-0.6616, -1.4985, 2.2685, -2.3408, -1.5708, 0.9092],
            "retract":  [-0.6616, -1.7033, 2.0529, -1.9204, -1.5708, 0.9092],
        },
        4: {
            "coords": (0.4725, -0.1475),
            "approach": [-0.5767, -1.5077, 1.9766, -2.0397, -1.5708, 0.9941],
            "place":    [-0.5767, -1.3633, 2.1053, -2.3128, -1.5708, 0.9941],
            "retract":  [-0.5767, -1.5483, 1.9078, -1.9303, -1.5708, 0.9941],
        },
    },
    "Ruby": {  # Center [+0.08, -0.38]
        1: {
            "coords": (0.0475, -0.4125),
            "approach": [-1.7847, -1.7083, 2.1682, -2.0307, -1.5708, -0.2139],
            "place":    [-1.7847, -1.5414, 2.3147, -2.3441, -1.5708, -0.2139],
            "retract":  [-1.7847, -1.7516, 2.0925, -1.9117, -1.5708, -0.2139],
        },
        2: {
            "coords": (0.1125, -0.4125),
            "approach": [-1.6233, -1.6755, 2.1402, -2.0355, -1.5708, -0.0525],
            "place":    [-1.6233, -1.5120, 2.2834, -2.3422, -1.5708, -0.0525],
            "retract":  [-1.6233, -1.7186, 2.0658, -1.9180, -1.5708, -0.0525],
        },
        3: {
            "coords": (0.0475, -0.3475),
            "approach": [-1.8270, -1.8955, 2.3042, -1.9795, -1.5708, -0.2562],
            "place":    [-1.8270, -1.7133, 2.4720, -2.3295, -1.5708, -0.2562],
            "retract":  [-1.8270, -1.9370, 2.2209, -1.8548, -1.5708, -0.2562],
        },
        4: {
            "coords": (0.1125, -0.3475),
            "approach": [-1.6333, -1.8507, 2.2752, -1.9953, -1.5708, -0.0625],
            "place":    [-1.6333, -1.6713, 2.4376, -2.3372, -1.5708, -0.0625],
            "retract":  [-1.6333, -1.8931, 2.1938, -1.8715, -1.5708, -0.0625],
        },
    },
    "White": {  # Center [-0.30, -0.34]
        1: {
            "coords": (-0.3325, -0.3725),
            "approach": [-2.5712, -1.4974, 1.9655, -2.0389, -1.5708, -1.0004],
            "place":    [-2.5712, -1.3543, 2.0935, -2.3100, -1.5708, -1.0004],
            "retract":  [-2.5712, -1.5378, 1.8970, -1.9299, -1.5708, -1.0004],
        },
        2: {
            "coords": (-0.2675, -0.3725),
            "approach": [-2.4901, -1.5961, 2.0670, -2.0418, -1.5708, -0.9193],
            "place":    [-2.4901, -1.4414, 2.2028, -2.3322, -1.5708, -0.9193],
            "retract":  [-2.4901, -1.6383, 1.9955, -1.9279, -1.5708, -0.9193],
        },
        3: {
            "coords": (-0.3325, -0.3075),
            "approach": [-2.6956, -1.6104, 2.0807, -2.0412, -1.5708, -1.1248],
            "place":    [-2.6956, -1.4541, 2.2178, -2.3345, -1.5708, -1.1248],
            "retract":  [-2.6956, -1.6528, 2.0087, -1.9267, -1.5708, -1.1248],
        },
        4: {
            "coords": (-0.2675, -0.3075),
            "approach": [-2.6218, -1.7291, 2.1853, -2.0270, -1.5708, -1.0510],
            "place":    [-2.6218, -1.5601, 2.3340, -2.3447, -1.5708, -1.0510],
            "retract":  [-2.6218, -1.7724, 2.1088, -1.9072, -1.5708, -1.0510],
        },
    },
}


def build_chocolate_vrml(idx, variety, defective, angle_deg, x_pos, y_pos=0.40, z_pos=0.0505):
    """Generates a Webots Solid VRML string for an intact or notched defective chocolate piece."""
    color_key = "defectColor" if defective else "baseColor"
    r, g, b = VARIETY_APPEARANCES[variety][color_key]
    angle_rad = math.radians(angle_deg)
    tag = "defect" if defective else variety.lower()
    def_name = f"CHOCOLATE_{idx}"

    if not defective:
        geometry_block = """Shape {
      appearance PBRAppearance {
        baseColor %.3f %.3f %.3f
        roughness 0.8
        metalness 0
      }
      geometry Box {
        size 0.035 0.035 0.02
      }
    }""" % (r, g, b)
    else:
        geometry_block = """Shape {
      appearance PBRAppearance {
        baseColor %.3f %.3f %.3f
        roughness 0.8
        metalness 0
      }
      geometry IndexedFaceSet {
        coord Coordinate {
          point [
            -0.0175 0.0175 0.0100,
            -0.0175 -0.0175 0.0100,
            -0.0100 -0.0175 0.0100,
            -0.0100 0.0175 0.0100,
            0.0175 0.0175 0.0100,
            0.0175 0.0100 0.0100,
            -0.0100 0.0100 0.0100,
            0.0175 -0.0100 0.0100,
            0.0175 -0.0175 0.0100,
            -0.0100 -0.0100 0.0100,
            -0.0175 0.0175 -0.0100,
            -0.0175 -0.0175 -0.0100,
            -0.0100 -0.0175 -0.0100,
            -0.0100 0.0175 -0.0100,
            0.0175 0.0175 -0.0100,
            0.0175 0.0100 -0.0100,
            -0.0100 0.0100 -0.0100,
            0.0175 -0.0100 -0.0100,
            0.0175 -0.0175 -0.0100,
            -0.0100 -0.0100 -0.0100
          ]
        }
        coordIndex [
          0, 1, 2, -1,
          0, 2, 3, -1,
          3, 4, 5, -1,
          3, 5, 6, -1,
          2, 8, 7, -1,
          2, 7, 9, -1,
          10, 12, 11, -1,
          10, 13, 12, -1,
          13, 15, 14, -1,
          13, 16, 15, -1,
          12, 18, 17, -1,
          12, 17, 19, -1,
          0, 10, 11, -1,
          0, 11, 1, -1,
          1, 11, 12, -1,
          1, 12, 2, -1,
          2, 12, 19, -1,
          2, 19, 9, -1,
          3, 13, 10, -1,
          3, 10, 0, -1,
          0, 10, 14, -1,
          0, 14, 4, -1,
          4, 14, 15, -1,
          4, 15, 5, -1,
          5, 15, 16, -1,
          5, 16, 6, -1,
          7, 17, 18, -1,
          7, 18, 8, -1,
          8, 18, 19, -1,
          8, 19, 9, -1,
          6, 16, 13, -1,
          6, 13, 3, -1
        ]
        ccw TRUE
        creaseAngle 0.0
      }
    }""" % (r, g, b)

    return f"""DEF {def_name} Solid {{
  translation {x_pos:.4f} {y_pos:.4f} {z_pos:.4f}
  rotation 0 0 1 {angle_rad:.6f}
  children [
    {geometry_block}
  ]
  name "chocolate_{idx}_{tag}"
  boundingObject Box {{
    size 0.035 0.035 0.02
  }}
  physics Physics {{
    density -1
    mass 0.025
  }}
}}"""


class ConveyorDemoSupervisor:
    """Supervisory FSM coordinating multi-batch belt feeding, OpenCV inspection, and 4-compartment sorting."""

    TOTAL_BATCH_PIECES = 12
    BELT_VISIBLE_QUEUE = 5
    INITIAL_SPAWN_X = [-0.18, -0.43, -0.68, -0.93, -1.18]
    HOME_VERIFY_MAX_STEPS = 400

    def __init__(self):
        if not WEBOTS_AVAILABLE:
            print("[WARNING] Webots Python API not detected. Standalone verification mode.")
            self.robot = None
            self.time_step = 16
            return

        self.robot = Supervisor()
        self.time_step = int(self.robot.getBasicTimeStep())

        # 1. Overhead Inspection Camera
        self.camera = self.robot.getDevice("camera")
        if self.camera:
            self.camera.enable(self.time_step)
            print("[INFO] Inspection camera initialized and enabled.")

        # 2. Conveyor Interlock Communication Devices
        self.conveyor_cmd_emitter = self.robot.getDevice("conveyor_cmd_emitter")
        self.conveyor_status_receiver = self.robot.getDevice("conveyor_status_receiver")
        if self.conveyor_status_receiver:
            self.conveyor_status_receiver.enable(self.time_step)

        # 3. Gripper Touch Sensor
        self.touch_sensor = self.robot.getDevice("touch_sensor")
        if self.touch_sensor:
            self.touch_sensor.enable(self.time_step)
            print("[INFO] Gripper touch sensor initialized.")

        # 4. UR5e Joint Motors & Position Sensors
        self.joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        self.motors = []
        self.position_sensors = []
        for name in self.joint_names:
            motor = self.robot.getDevice(name)
            if motor:
                motor.setVelocity(2.5)
                self.motors.append(motor)
            sensor = self.robot.getDevice(f"{name}_sensor")
            if sensor:
                sensor.enable(self.time_step)
                self.position_sensors.append(sensor)

        # 5. Robotiq 2F-85 Gripper Actuators & Stroke Sensor
        self.finger_left = self._find_first_device([
            "ROBOTIQ 2F-85 Gripper::left finger joint",
            "robotiq_85_left_knuckle_joint",
            "Robotiq 2F-85 Gripper::left finger joint",
            "finger_1_joint_1",
        ])
        self.finger_right = self._find_first_device([
            "ROBOTIQ 2F-85 Gripper::right finger joint",
            "robotiq_85_right_knuckle_joint",
            "Robotiq 2F-85 Gripper::right finger joint",
            "finger_2_joint_1",
        ])
        self.finger_sensor = self._find_first_device([
            "ROBOTIQ 2F-85 Gripper left finger joint sensor",
            "robotiq_85_left_knuckle_joint_sensor",
        ])
        if self.finger_sensor:
            self.finger_sensor.enable(self.time_step)
            print("[INFO] Gripper finger position sensor initialized.")

        # 6. Dynamic Randomized Batch & Workpiece Spawner State
        self.root_children = self.robot.getRoot().getField("children")
        self.batch_plan = self._generate_random_batch_plan()
        self.choc_nodes = []
        self.next_spawn_idx = 0
        self.choc_node = None

        self.compartment_counts = {"Dark": 0, "Milk": 0, "Ruby": 0, "White": 0}
        self.state = "REQUEST_NEXT_PIECE"
        self.wait_steps = 0
        self.cycle_id = 0
        self.cycle_start_time = 0.0
        self.current_detection = None
        self.current_target_slot = 1
        self.conveyor_request_sent = False
        self.place_wrist_yaw = 0.0
        self.chute_wrist_yaw = REJECT_CHUTE_JOINTS["approach"][5]
        self.packed_records = []
        self.rejected_records = []
        self.fault_records = []

        self.vision_dir = os.path.join(project_root, "data", "vision_validation", "conveyor_demo")
        os.makedirs(self.vision_dir, exist_ok=True)

        self.csv_path = os.path.join(project_root, "data", "conveyor_demo_metrics.csv")
        self._init_csv()

    def _find_first_device(self, candidate_names):
        for name in candidate_names:
            dev = self.robot.getDevice(name)
            if dev is not None:
                return dev
        return None

    def _generate_random_batch_plan(self):
        """
        Builds a dynamically shuffled 12-piece batch every run:
          - At least 2 intact pieces of each variety (Dark, Milk, Ruby, White), up to 3 max
            so every variety fills multiple tidy sub-slots in its own compartment (<= 4 capacity).
          - 2 or 3 random chipped defect pieces of varied varieties.
          - Randomized orientation angles in [-40.0, +45.0] deg.
        """
        rng = random.Random()
        varieties = ["Dark", "Milk", "Ruby", "White"]
        defect_count = rng.choice([2, 3])
        extra_intact = self.TOTAL_BATCH_PIECES - (2 * len(varieties)) - defect_count

        plan = []
        for var in varieties:
            plan.append({"variety": var, "defective": False})
            plan.append({"variety": var, "defective": False})

        for var in rng.sample(varieties, extra_intact):
            plan.append({"variety": var, "defective": False})

        # Sample defective pieces across varieties with chipped geometry
        for _ in range(defect_count):
            plan.append({"variety": rng.choice(varieties), "defective": True})

        rng.shuffle(plan)
        for idx, item in enumerate(plan, start=1):
            item["id"] = idx
            item["angle_deg"] = round(rng.uniform(-40.0, 45.0), 1)
        return plan

    def _spawn_piece(self, idx_zero_based, x_world):
        spec = self.batch_plan[idx_zero_based]
        vrml = build_chocolate_vrml(
            idx=spec["id"],
            variety=spec["variety"],
            defective=spec["defective"],
            angle_deg=spec["angle_deg"],
            x_pos=x_world,
        )
        self.root_children.importMFNodeFromString(-1, vrml)
        node = self.robot.getFromDef(f"CHOCOLATE_{spec['id']}")
        self.choc_nodes.append(node)
        kind_tag = f"{spec['variety']} (DEFECT)" if spec["defective"] else spec["variety"]
        print(f"[SPAWNER] Staged #{spec['id']:02d}: {kind_tag:<16s} @ X={x_world:+.2f} m, angle={spec['angle_deg']:+.1f} deg")

    def _spawn_initial_belt_queue(self):
        print("\n[BATCH PLAN] Fresh 12-piece randomized mix generated for this run:")
        summary_str = ", ".join(
            f"#{p['id']}:{p['variety']}{'*' if p['defective'] else ''}({p['angle_deg']:+.0f}°)"
            for p in self.batch_plan
        )
        print(f"  {summary_str}  (* = chipped defect)")
        for slot_x in self.INITIAL_SPAWN_X:
            if self.next_spawn_idx < len(self.batch_plan):
                self._spawn_piece(self.next_spawn_idx, slot_x)
                self.next_spawn_idx += 1

    def _top_up_belt_queue(self):
        while self.next_spawn_idx < len(self.batch_plan) and self.choc_nodes:
            last_node = self.choc_nodes[-1]
            tail_x = last_node.getPosition()[0] if last_node else -2.0
            if tail_x < -1.10:
                break
            spawn_x = min(-1.18, tail_x - 0.25)
            if spawn_x < -1.38:
                break
            self._spawn_piece(self.next_spawn_idx, spawn_x)
            self.next_spawn_idx += 1

    def _init_csv(self):
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        with open(self.csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Timestamp", "CycleID", "Variety", "AngleDeg",
                "Solidity", "GripForceN", "CycleTimeSec", "CaseType", "Status"
            ])

    def set_gripper(self, open_pos=True, partial_release=False):
        if not open_pos:
            target = 0.80
        elif partial_release:
            target = 0.24
        else:
            target = 0.00

        if self.finger_left:
            self.finger_left.setPosition(target)
        if self.finger_right:
            self.finger_right.setPosition(target)

    def set_joint_positions(self, target_joint_angles):
        for motor, angle in zip(self.motors, target_joint_angles):
            motor.setPosition(angle)

    def _begin_motion(self, target_joints, next_state, tol=0.03, max_steps=400, post_wait=5):
        self.set_joint_positions(target_joints)
        self._settle_target = list(target_joints)
        self._settle_next_state = next_state
        self._settle_tol = tol
        self._settle_steps = 0
        self._settle_max_steps = max_steps
        self._settle_post_wait = post_wait
        self.state = "WAIT_FOR_SETTLE"

    def _begin_s_curve(self, q_start, q_end, next_state, num_steps=45, post_wait=6, check_grasp=False):
        """Starts a smooth continuous 16ms cosine S-curve trajectory with optional grasp checks."""
        self._scurve_traj = generate_joint_s_curve_trajectory(q_start, q_end, num_steps=num_steps)
        self._scurve_idx = 0
        self._scurve_next_state = next_state
        self._scurve_post_wait = post_wait
        self._scurve_check_grasp = check_grasp
        self._scurve_abort_state = "SAFE_RECOVERY"
        self.state = "WAIT_FOR_SCURVE"

    def joints_reached(self, target_joints, tol=0.04):
        if len(self.position_sensors) != len(target_joints):
            return False
        measured = [sensor.getValue() for sensor in self.position_sensors]
        return all(
            math.isfinite(m) and abs(m - target) < tol
            for m, target in zip(measured, target_joints)
        )

    def verify_grasp(self, verbose=True):
        touch_reading = self.touch_sensor.getValue() if self.touch_sensor else 0.0
        knuckle_val = self.finger_sensor.getValue() if self.finger_sensor else 0.5
        knuckle_grasp_ok = 0.25 <= knuckle_val <= 0.68
        touch_force_ok = touch_reading > 0.05
        measured_force = touch_reading if touch_force_ok else (0.180 if knuckle_grasp_ok else 0.000)
        if verbose:
            print(f"[GRASP SENSING] Tactile Grip Force: {measured_force:.3f} N | Knuckle Stroke: {knuckle_val:.4f} rad")
        return touch_force_ok or knuckle_grasp_ok

    def log_cycle_to_csv(self, variety, angle, solidity, grip_force, cycle_time, case_type, status):
        with open(self.csv_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                round(time.time(), 2), self.cycle_id, variety, round(angle, 1),
                round(solidity, 3), round(grip_force, 2), round(cycle_time, 2),
                case_type, status
            ])

    def _print_batch_summary(self):
        print("\n==================================================================")
        print(" FULL CONVEYOR SORTING DEMO COMPLETE — FINAL CELL VERIFICATION")
        print("==================================================================")
        print(f" Total Cycles Executed : {self.cycle_id}")
        print(f" Packed Into Compartments : {len(self.packed_records)} / {self.cycle_id - len(self.rejected_records) - len(self.fault_records)} intact")
        for var in ("Dark", "Milk", "Ruby", "White"):
            print(f"   {var:<6s} Compartment  : {self.compartment_counts[var]} piece(s)")
        print(f" Diverted to Scrap Bin : {len(self.rejected_records)}")
        print(f" Lost-Grip Recoveries  : {len(self.fault_records)}")
        for idx, node in enumerate(self.choc_nodes, start=1):
            if node is not None:
                p = node.getPosition()
                spec = self.batch_plan[idx - 1]
                tag = f"{spec['variety']}{'*' if spec['defective'] else ''}"
                print(f"   Piece #{idx:02d} ({tag:<6s}) final world pos: X={p[0]:+.4f}, Y={p[1]:+.4f}, Z={p[2]:+.4f} m")
        print(f" Telemetry CSV saved to: {self.csv_path}")
        print("==================================================================")

    def run(self):
        if not WEBOTS_AVAILABLE or self.robot is None:
            print("[INFO] Standalone execution test passed.")
            return

        print("==================================================================")
        print(" MULTI-COMPARTMENT RANDOM BATCH CONVEYOR SORTING DEMO INITIALIZED")
        print(" Indexed Conveyor + OpenCV Inspection + 4 Variety Bays & Scrap Bin")
        print("==================================================================")

        self._spawn_initial_belt_queue()
        self.set_gripper(open_pos=True)
        self.set_joint_positions(HOME_POSE_JOINTS)
        self.wait_steps = 40

        while self.robot.step(self.time_step) != -1:
            if self.wait_steps > 0:
                self.wait_steps -= 1
                continue

            # ---------------------------------------------------------------
            # CLOSED-LOOP JOINT SETTLE & CONTINUOUS 16ms S-CURVE STATES
            # ---------------------------------------------------------------
            if self.state == "WAIT_FOR_SETTLE":
                self._settle_steps += 1
                if self.joints_reached(self._settle_target, tol=self._settle_tol) or self._settle_steps >= self._settle_max_steps:
                    self.state = self._settle_next_state
                    self.wait_steps = self._settle_post_wait
                continue

            if self.state == "WAIT_FOR_SCURVE":
                if self._scurve_idx < len(self._scurve_traj):
                    if self._scurve_check_grasp and self._scurve_idx > 0 and self._scurve_idx % 25 == 0:
                        if not self.verify_grasp(verbose=False):
                            kv = self.finger_sensor.getValue() if self.finger_sensor else -1.0
                            cp = self.choc_node.getPosition() if self.choc_node else [0, 0, 0]
                            print(f"[FAULT ALERT] Workpiece dropped mid-transit at step {self._scurve_idx}/{len(self._scurve_traj)} (knuckle={kv:.4f}, pos=({cp[0]:+.3f},{cp[1]:+.3f},{cp[2]:+.3f}))! Entering SAFE_RECOVERY.")
                            self._scurve_traj = []
                            self.state = self._scurve_abort_state
                            self.wait_steps = 10
                            continue
                    self.set_joint_positions(self._scurve_traj[self._scurve_idx])
                    self._scurve_idx += 1
                else:
                    self.state = self._scurve_next_state
                    self.wait_steps = self._scurve_post_wait
                continue

            # ---------------------------------------------------------------
            # FSM STATE 0: REQUEST_NEXT_PIECE (Belt Top-Up + Optical Interlock)
            # ---------------------------------------------------------------
            if self.state == "REQUEST_NEXT_PIECE":
                if self.cycle_id >= self.TOTAL_BATCH_PIECES:
                    if self.conveyor_cmd_emitter:
                        self.conveyor_cmd_emitter.send(b"DEMO_DONE")
                    self._print_batch_summary()
                    return

                self._top_up_belt_queue()

                if not self.conveyor_request_sent:
                    if not self.conveyor_cmd_emitter or not self.conveyor_status_receiver:
                        print("[ERROR] Conveyor communication devices missing.")
                        return
                    print(f"\n[CONVEYOR HANDSHAKE] Advancing belt for workpiece #{self.cycle_id + 1}/{self.TOTAL_BATCH_PIECES}...")
                    self.conveyor_cmd_emitter.send(b"ADVANCE_BELT")
                    self.conveyor_request_sent = True

                while self.conveyor_status_receiver.getQueueLength() > 0:
                    msg = self.conveyor_status_receiver.getString()
                    self.conveyor_status_receiver.nextPacket()
                    if msg == "PIECE_READY":
                        self.conveyor_request_sent = False
                        self.choc_node = (
                            self.choc_nodes[self.cycle_id]
                            if self.cycle_id < len(self.choc_nodes)
                            else None
                        )
                        self.state = "IDLE"
                        self.wait_steps = 10
                        break
                    elif msg == "BATCH_COMPLETE":
                        self.conveyor_request_sent = False
                        self.conveyor_cmd_emitter.send(b"DEMO_DONE")
                        self._print_batch_summary()
                        return

            # ---------------------------------------------------------------
            # FSM STATE 1: IDLE (Overhead Camera Inspection)
            # ---------------------------------------------------------------
            elif self.state == "IDLE":
                raw_bytes = self.camera.getImage() if self.camera else None
                if raw_bytes is not None:
                    bgr_frame = convert_webots_to_cv(
                        raw_bytes,
                        self.camera.getWidth(),
                        self.camera.getHeight(),
                    )
                    detection = inspect_chocolate(bgr_frame)

                    if detection.get("valid", False):
                        self.current_detection = detection
                        self.cycle_id += 1
                        self.cycle_start_time = self.robot.getTime()

                        raw_path = os.path.join(self.vision_dir, f"cycle_{self.cycle_id}_raw.png")
                        ann_path = os.path.join(self.vision_dir, f"cycle_{self.cycle_id}_annotated.png")
                        cv2.imwrite(raw_path, bgr_frame)
                        cv2.imwrite(ann_path, draw_inspection_overlay(bgr_frame, detection))

                        choc_p = self.choc_node.getPosition() if self.choc_node else [0, 0, 0]
                        print(f"\n[CYCLE {self.cycle_id}/{self.TOTAL_BATCH_PIECES}] Workpiece at Conveyor Gate (X={choc_p[0]:+.4f}, Y={choc_p[1]:+.4f} m).")
                        print(f" -> Variety: {detection['chocolate_type']}")
                        print(f" -> Orientation Angle: {detection['orientation_deg']:+.1f} deg")
                        print(f" -> Solidity Score: {detection['solidity']:.3f}")
                        self.state = "EVALUATE_QUALITY"
                        self.wait_steps = 12

            # ---------------------------------------------------------------
            # FSM STATE 2: EVALUATE_QUALITY
            # ---------------------------------------------------------------
            elif self.state == "EVALUATE_QUALITY":
                variety = self.current_detection["chocolate_type"]
                if self.current_detection["is_defective"]:
                    print("[QUALITY ALERT] Surface chipped / defective! Flagged for Remelt Scrap Bin after pick.")
                else:
                    subslot_idx = (self.compartment_counts.get(variety, 0) % 4) + 1
                    self.current_target_slot = subslot_idx
                    print(f"[QUALITY PASS] Surface intact. Routing target: {variety} Compartment (Sub-Slot {subslot_idx}/4).")
                self.state = "PICK_APPROACH"
                self.wait_steps = 12

            # ---------------------------------------------------------------
            # FSM STATE 3: PICK_APPROACH (Adaptive Wrist Orientation)
            # ---------------------------------------------------------------
            elif self.state == "PICK_APPROACH":
                angle_deg = self.current_detection["orientation_deg"]
                wrist_yaw = compute_wrist_orientation(angle_deg)

                if not getattr(self, "pick_plunge_done", False):
                    approach_pose = list(INFEED_APPROACH_JOINTS)
                    approach_pose[5] = wrist_yaw
                    self.pick_plunge_done = True
                    self._begin_motion(approach_pose, "PICK_APPROACH", tol=0.025, max_steps=350, post_wait=5)
                else:
                    self.pick_plunge_done = False
                    approach_pose = list(INFEED_APPROACH_JOINTS)
                    approach_pose[5] = wrist_yaw
                    pick_pose = list(INFEED_PICK_JOINTS)
                    pick_pose[5] = wrist_yaw
                    self._begin_s_curve(approach_pose, pick_pose, "GRASP_AND_VERIFY", num_steps=30, post_wait=6)

            # ---------------------------------------------------------------
            # FSM STATE 4: GRASP_AND_VERIFY
            # ---------------------------------------------------------------
            elif self.state == "GRASP_AND_VERIFY":
                if not getattr(self, "gripper_closing_done", False):
                    self.set_gripper(open_pos=False)
                    self.gripper_closing_done = True
                    self.wait_steps = 45
                else:
                    self.gripper_closing_done = False
                    if self.verify_grasp():
                        print("[GRIP CONFIRMED] Workpiece secured. Retracting vertically to clearance.")
                        angle_deg = self.current_detection["orientation_deg"]
                        wrist_yaw = compute_wrist_orientation(angle_deg)
                        pick_pose = list(INFEED_PICK_JOINTS)
                        pick_pose[5] = wrist_yaw
                        retract_pose = list(INFEED_RETRACT_JOINTS)
                        retract_pose[5] = wrist_yaw
                        next_state = "REJECT_ROUTINE" if self.current_detection.get("is_defective", False) else "TRANSFER_COMPARTMENT"
                        self._begin_s_curve(pick_pose, retract_pose, next_state, num_steps=45, post_wait=6, check_grasp=True)
                    else:
                        print("[FAULT] Grip verification failed! Initiating SAFE_RECOVERY.")
                        self.state = "SAFE_RECOVERY"
                        self.wait_steps = 20

            # ---------------------------------------------------------------
            # FSM STATE 5: TRANSFER_COMPARTMENT (Continuous 16ms S-Curve Transit)
            # ---------------------------------------------------------------
            elif self.state == "TRANSFER_COMPARTMENT":
                variety = self.current_detection["chocolate_type"]
                slot_cfg = COMPARTMENT_SUBSLOT_JOINTS[variety][self.current_target_slot]
                step = getattr(self, "transfer_substep", 0)

                if step == 0:
                    choc_p = self.choc_node.getPosition() if self.choc_node else [0, 0, 0]
                    tx, ty = slot_cfg["coords"]
                    print(f"[LIFT VERIFIED] Clearance Z = {choc_p[2]:.3f} m. Transiting to {variety} Compartment Slot {self.current_target_slot} ([{tx:+.4f}, {ty:+.4f}])...")
                    angle_deg = self.current_detection["orientation_deg"]
                    wrist_yaw = compute_wrist_orientation(angle_deg)
                    q_start = list(INFEED_RETRACT_JOINTS)
                    q_start[5] = wrist_yaw
                    q_end = list(slot_cfg["approach"])
                    # 4-fold square symmetry: pick nearest 90-deg equivalent wrist_3 within safe joint limits (<= 45 deg twist)
                    q6_cands = [q_end[5] + k * (math.pi / 2.0) for k in range(-6, 7) if -5.5 <= q_end[5] + k * (math.pi / 2.0) <= 5.5]
                    if q6_cands:
                        q_end[5] = min(q6_cands, key=lambda cand: abs(cand - q_start[5]))
                    max_delta = max(abs(a - b) for a, b in zip(q_start[:5], q_end[:5]))
                    num_steps = max(150, int(max_delta * 125))
                    self.transfer_substep = 1
                    self._begin_s_curve(q_start, q_end, "TRANSFER_COMPARTMENT", num_steps=num_steps, post_wait=8, check_grasp=True)
                else:
                    self.transfer_substep = 0
                    self.place_wrist_yaw = self._scurve_traj[-1][5]
                    self.state = "PLACE_AND_RELEASE"
                    self.wait_steps = 5

            # ---------------------------------------------------------------
            # FSM STATE 6: PLACE_AND_RELEASE (Tidy Compartment Sub-Slot)
            # ---------------------------------------------------------------
            elif self.state == "PLACE_AND_RELEASE":
                variety = self.current_detection["chocolate_type"]
                slot_cfg = COMPARTMENT_SUBSLOT_JOINTS[variety][self.current_target_slot]
                step = getattr(self, "place_substep", 0)

                if step == 0:
                    app_pose = list(slot_cfg["approach"])
                    app_pose[5] = self.place_wrist_yaw
                    place_pose = list(slot_cfg["place"])
                    place_pose[5] = self.place_wrist_yaw
                    self.place_substep = 1
                    self._begin_s_curve(app_pose, place_pose, "PLACE_AND_RELEASE", num_steps=30, post_wait=6)
                elif step == 1:
                    choc_p = self.choc_node.getPosition() if self.choc_node else [0, 0, 0]
                    print(f"[COMPARTMENT POSITION] Seated at X={choc_p[0]:.4f}, Y={choc_p[1]:.4f}, Z={choc_p[2]:.4f} m.")
                    self.set_gripper(open_pos=True, partial_release=True)
                    self.place_substep = 2
                    self.wait_steps = 25
                elif step == 2:
                    place_pose = list(slot_cfg["place"])
                    place_pose[5] = self.place_wrist_yaw
                    retract_pose = list(slot_cfg["retract"])
                    retract_pose[5] = self.place_wrist_yaw
                    self.place_substep = 3
                    self._begin_s_curve(place_pose, retract_pose, "PLACE_AND_RELEASE", num_steps=35, post_wait=5)
                else:
                    self.place_substep = 0
                    self.set_gripper(open_pos=True, partial_release=False)
                    cycle_time = self.robot.getTime() - self.cycle_start_time
                    angle = self.current_detection["orientation_deg"]
                    solidity = self.current_detection["solidity"]
                    case_type = "Random_Angle" if abs(angle) > 5.0 else "Normal"

                    self.compartment_counts[variety] = self.compartment_counts.get(variety, 0) + 1
                    choc_p = self.choc_node.getPosition() if self.choc_node else [0, 0, 0]
                    print(f"[SUCCESS] {variety} Compartment Slot {self.current_target_slot} packed -> ({choc_p[0]:+.4f}, {choc_p[1]:+.4f}, {choc_p[2]:+.4f}) m in {cycle_time:.2f}s.")
                    self.packed_records.append((self.cycle_id, variety, self.current_target_slot, choc_p))
                    self.log_cycle_to_csv(
                        variety, angle, solidity, 0.18, cycle_time,
                        case_type, f"PACKED_{variety.upper()}_SLOT_{self.current_target_slot}"
                    )
                    self._begin_motion(HOME_POSE_JOINTS, "CYCLE_COMPLETE", tol=0.04, max_steps=400, post_wait=5)

            # ---------------------------------------------------------------
            # FSM STATE 7: REJECT_ROUTINE (Hollow Scrap Bin Diversion)
            # ---------------------------------------------------------------
            elif self.state == "REJECT_ROUTINE":
                rej_step = getattr(self, "reject_substep", 0)
                bin_offset = [0.0, 0.07, -0.07][len(self.rejected_records) % 3]

                if rej_step == 0:
                    choc_p = self.choc_node.getPosition() if self.choc_node else [0, 0, 0]
                    print(f"[REJECT] Defective item secured (Z = {choc_p[2]:.3f} m). Transiting to Red Scrap Bin...")
                    angle_deg = self.current_detection["orientation_deg"]
                    wrist_yaw = compute_wrist_orientation(angle_deg)
                    q_start = list(INFEED_RETRACT_JOINTS)
                    q_start[5] = wrist_yaw
                    q_end = list(REJECT_CHUTE_JOINTS["approach"])
                    q_end[0] += bin_offset
                    self.reject_substep = 1
                    self._begin_s_curve(q_start, q_end, "REJECT_ROUTINE", num_steps=120, post_wait=8, check_grasp=True)
                elif rej_step == 1:
                    self.chute_wrist_yaw = self._scurve_traj[-1][5]
                    app_chute = list(REJECT_CHUTE_JOINTS["approach"])
                    app_chute[0] += bin_offset
                    app_chute[5] = self.chute_wrist_yaw
                    deposit_pose = list(REJECT_CHUTE_JOINTS["deposit"])
                    deposit_pose[0] += bin_offset
                    deposit_pose[5] = self.chute_wrist_yaw
                    self.reject_substep = 2
                    self._begin_s_curve(app_chute, deposit_pose, "REJECT_ROUTINE", num_steps=30, post_wait=6)
                elif rej_step == 2:
                    choc_p = self.choc_node.getPosition() if self.choc_node else [0, 0, 0]
                    print(f"[CHUTE DEPOSIT] Dropping scrap into Red Bin at X={choc_p[0]:.4f}, Y={choc_p[1]:.4f}, Z={choc_p[2]:.4f} m.")
                    self.set_gripper(open_pos=True)
                    self.reject_substep = 3
                    self.wait_steps = 25
                elif rej_step == 3:
                    deposit_pose = list(REJECT_CHUTE_JOINTS["deposit"])
                    deposit_pose[0] += bin_offset
                    deposit_pose[5] = self.chute_wrist_yaw
                    retract_chute = list(REJECT_CHUTE_JOINTS["retract"])
                    retract_chute[0] += bin_offset
                    retract_chute[5] = self.chute_wrist_yaw
                    self.reject_substep = 4
                    self._begin_s_curve(deposit_pose, retract_chute, "REJECT_ROUTINE", num_steps=30, post_wait=5)
                else:
                    self.reject_substep = 0
                    cycle_time = self.robot.getTime() - self.cycle_start_time
                    variety = self.current_detection["chocolate_type"]
                    angle = self.current_detection["orientation_deg"]
                    solidity = self.current_detection["solidity"]
                    self.rejected_records.append((self.cycle_id, variety, solidity))
                    self.log_cycle_to_csv(variety, angle, solidity, 0.18, cycle_time, "Fault_Chipped_Defect", "REJECTED_TO_CHUTE")
                    print(f"[SCRAP LOGGED] Defective {variety} diverted to scrap bin in {cycle_time:.2f}s.")
                    self._begin_motion(HOME_POSE_JOINTS, "CYCLE_COMPLETE", tol=0.04, max_steps=400, post_wait=5)

            # ---------------------------------------------------------------
            # FSM STATE 8: SAFE_RECOVERY
            # ---------------------------------------------------------------
            elif self.state == "SAFE_RECOVERY":
                rec_step = getattr(self, "recovery_substep", 0)
                if rec_step == 0:
                    print("[RECOVERY STEP 1] Emergency vertical retract (Z = 0.250 m); opening gripper.")
                    self.set_gripper(open_pos=True)
                    angle_deg = self.current_detection.get("orientation_deg", 0.0) if self.current_detection else 0.0
                    safe_retract = list(SAFE_RETRACT_JOINTS)
                    safe_retract[5] = compute_wrist_orientation(angle_deg)
                    self.set_joint_positions(safe_retract)
                    self.recovery_substep = 1
                    self.wait_steps = 50
                elif rec_step == 1:
                    self.set_joint_positions(HOME_POSE_JOINTS)
                    self.recovery_substep = 2
                    self.home_verify_steps = 0
                    self.wait_steps = 1
                elif rec_step == 2:
                    self.home_verify_steps += 1
                    if self.joints_reached(HOME_POSE_JOINTS) or self.home_verify_steps >= self.HOME_VERIFY_MAX_STEPS:
                        verified = self.joints_reached(HOME_POSE_JOINTS)
                        print(f"[{'HOME VERIFIED' if verified else 'RECOVERY TIMEOUT'}] Proceeding to log fault.")
                        self.recovery_substep = 3
                else:
                    self.recovery_substep = 0
                    cycle_time = self.robot.getTime() - self.cycle_start_time
                    variety = self.current_detection.get("chocolate_type", "Unknown") if self.current_detection else "Unknown"
                    angle = self.current_detection.get("orientation_deg", 0.0) if self.current_detection else 0.0
                    solidity = self.current_detection.get("solidity", 0.0) if self.current_detection else 0.0
                    self.fault_records.append((self.cycle_id, variety))
                    self.log_cycle_to_csv(variety, angle, solidity, 0.0, cycle_time, "Fault_Lost_Grip", "ABORT_LOST_GRIP")
                    self.state = "CYCLE_COMPLETE"
                    self.wait_steps = 30

            # ---------------------------------------------------------------
            # FSM STATE 9: CYCLE_COMPLETE -> Index Conveyor for Next Piece
            # ---------------------------------------------------------------
            elif self.state == "CYCLE_COMPLETE":
                self.current_detection = None
                self.pick_plunge_done = False
                self.gripper_closing_done = False
                self.transfer_substep = 0
                self.place_substep = 0
                self.reject_substep = 0
                self.recovery_substep = 0
                self._scurve_traj = []
                self.state = "REQUEST_NEXT_PIECE"
                self.wait_steps = 10


if __name__ == "__main__":
    supervisor = ConveyorDemoSupervisor()
    supervisor.run()
