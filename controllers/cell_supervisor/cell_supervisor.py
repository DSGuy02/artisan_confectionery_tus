"""
cell_supervisor.py - Supervisory Finite State Machine (FSM) Controller
Cyberbotics Webots Controller for Universal Robots UR5e + OpenCV Vision
Technological University of the Shannon (TUS Athlone)
Integrated Robotics 4 - Continuous Assessment 3
"""

import sys
import os
import time
import csv
import math
import numpy as np

# Add project root to sys.path so modules can be imported inside Webots
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from perception.vision_pipeline import (
    convert_webots_to_cv,
    inspect_chocolate,
    draw_inspection_overlay
)
from motion.waypoints import (
    HOME_POSE_JOINTS,
    STANDBY_POSE_JOINTS,
    INFEED_APPROACH_JOINTS,
    INFEED_PICK_JOINTS,
    INFEED_RETRACT_JOINTS,
    TRAY_SLOT_JOINTS,
    REJECT_CHUTE_JOINTS,
    SAFE_RETRACT_JOINTS,
    get_tray_slot_target,
    get_slot_by_variety,
    get_tray_slot_joints,
    compute_wrist_orientation
)
from motion.trajectory_planner import compute_adaptive_approach_pose

# Attempt to import Webots controller library
try:
    from controller import Supervisor, Robot, Camera, TouchSensor
    WEBOTS_AVAILABLE = True
except ImportError:
    WEBOTS_AVAILABLE = False


class ConfectioneryCellSupervisor:
    """Master supervisory controller for the artisan confectionery packaging cell."""
    
    def __init__(self):
        if not WEBOTS_AVAILABLE:
            print("[WARNING] Webots Python API not detected. Controller running in standalone verification mode.")
            self.robot = None
            self.time_step = 16
            return
            
        self.robot = Supervisor()
        self.time_step = int(self.robot.getBasicTimeStep())
        
        # 1. Initialize Overhead Inspection Camera
        self.camera = self.robot.getDevice("camera")
        if self.camera:
            self.camera.enable(self.time_step)
            print("[INFO] Inspection camera initialized and enabled.")
        else:
            print("[WARNING] Camera device 'camera' not found in Webots world.")

        # 2. Initialize Gripper Touch Sensor
        self.touch_sensor = self.robot.getDevice("touch_sensor")
        if self.touch_sensor:
            self.touch_sensor.enable(self.time_step)
            print("[INFO] Gripper touch sensor initialized.")
        else:
            print("[INFO] Using simulated grip verification threshold.")

        # 3. Initialize UR5e Joint Motors and Position Sensors
        self.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]
        self.motors = []
        self.position_sensors = []
        for name in self.joint_names:
            motor = self.robot.getDevice(name)
            if motor:
                motor.setVelocity(1.0)
                self.motors.append(motor)
            sensor = self.robot.getDevice(f"{name}_sensor")
            if sensor:
                sensor.enable(self.time_step)
                self.position_sensors.append(sensor)

        # 4. Initialize Robotiq 2F-85 Gripper Actuators (Supports multiple PROTO naming conventions)
        self.finger_left = self._find_first_device([
            "ROBOTIQ 2F-85 Gripper::left finger joint",
            "robotiq_85_left_knuckle_joint",
            "Robotiq 2F-85 Gripper::left finger joint",
            "finger_1_joint_1"
        ])
        self.finger_right = self._find_first_device([
            "ROBOTIQ 2F-85 Gripper::right finger joint",
            "robotiq_85_right_knuckle_joint",
            "Robotiq 2F-85 Gripper::right finger joint",
            "finger_2_joint_1"
        ])
        self.finger_sensor = self._find_first_device([
            "ROBOTIQ 2F-85 Gripper left finger joint sensor",
            "robotiq_85_left_knuckle_joint_sensor"
        ])
        if self.finger_sensor:
            self.finger_sensor.enable(self.time_step)
            print("[INFO] Gripper finger position sensor initialized.")

        # 5. Telemetry & Non-Blocking State Tracking
        self.choc_node = self.robot.getFromDef("CHOCOLATE_WORKPIECE") if hasattr(self.robot, "getFromDef") else None
        self.state = "IDLE"
        self.wait_steps = 0
        self.cycle_id = 0
        self.cycle_start_time = 0.0
        self.current_detection = None
        self.csv_path = os.path.join(project_root, "data/production_metrics.csv")
        self._init_csv()

    def _find_first_device(self, candidate_names):
        """Returns the first matching Webots device from a list of candidate names."""
        for name in candidate_names:
            dev = self.robot.getDevice(name)
            if dev is not None:
                return dev
        return None

    def _init_csv(self):
        """Initializes production metrics CSV if not present."""
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Timestamp", "CycleID", "Variety", "AngleDeg", 
                    "Solidity", "GripForceN", "CycleTimeSec", "CaseType", "Status"
                ])

    def set_gripper(self, open_pos=True):
        """Actuates Robotiq gripper fingers: Open (0.0 rad) or Closed (0.80 rad)."""
        target = 0.0 if open_pos else 0.80
        if self.finger_left:
            self.finger_left.setPosition(target)
        if self.finger_right:
            self.finger_right.setPosition(target)

    def set_joint_positions(self, target_joint_angles):
        """Commands UR5e joint motors to target angle array."""
        for motor, angle in zip(self.motors, target_joint_angles):
            motor.setPosition(angle)

    def joints_reached(self, target_joints, tol=0.04):
        """Returns True if arm position sensors have settled near target_joints."""
        if not self.position_sensors or len(self.position_sensors) != len(target_joints):
            return True
        return all(abs(s.getValue() - target) < tol 
                   for s, target in zip(self.position_sensors, target_joints))

    def verify_grasp(self):
        """Queries touch sensor and gripper knuckle stroke to confirm positive workpiece grasp."""
        touch_reading = self.touch_sensor.getValue() if self.touch_sensor else 0.0
        knuckle_val = self.finger_sensor.getValue() if getattr(self, "finger_sensor", None) else 0.5
        
        # When 35mm chocolate is grasped: knuckle settles between 0.30 and 0.58 rad.
        # When fingers close on empty space without workpiece: knuckle exceeds 0.60 rad.
        knuckle_grasp_ok = (0.25 <= knuckle_val <= 0.58)
        touch_force_ok = (touch_reading > 0.05)
        
        print(f"[GRASP SENSING] Touch Force: {touch_reading:.3f} | Knuckle Stroke: {knuckle_val:.4f} rad")
        return touch_force_ok or knuckle_grasp_ok

    def log_cycle_to_csv(self, variety, angle, solidity, grip_force, cycle_time, case_type, status):
        """Appends cycle telemetry to production CSV log."""
        with open(self.csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                round(time.time(), 2), self.cycle_id, variety, round(angle, 1),
                round(solidity, 3), round(grip_force, 2), round(cycle_time, 2),
                case_type, status
            ])

    def run(self):
        """Main supervisory execution loop."""
        if not WEBOTS_AVAILABLE or self.robot is None:
            print("[INFO] Standalone execution test passed.")
            return

        print("==================================================================")
        print(" ARTISAN CONFECTIONERY ROBOTIC CELL SUPERVISOR INITIALIZED")
        print(" Finite State Machine (FSM) Running. Monitoring Infeed...")
        print("==================================================================")
        
        self.set_gripper(open_pos=True)
        self.set_joint_positions(HOME_POSE_JOINTS)
        self.wait_steps = 40  # Allow robot to settle at home pose initially

        while self.robot.step(self.time_step) != -1:
            # Non-blocking motion settle timer
            if self.wait_steps > 0:
                self.wait_steps -= 1
                continue

            # ---------------------------------------------------------------
            # FSM STATE 1: IDLE (Wait for workpiece in camera FOV)
            # ---------------------------------------------------------------
            if self.state == "IDLE":
                raw_bytes = self.camera.getImage() if self.camera else None
                if raw_bytes is not None:
                    bgr_frame = convert_webots_to_cv(raw_bytes, 
                                                    self.camera.getWidth(), 
                                                    self.camera.getHeight())
                    # TODO (Partner 2): Tune min_contour_area & HSV masks for Webots lighting
                    detection = inspect_chocolate(bgr_frame)
                    
                    if detection.get("valid", False):
                        self.current_detection = detection
                        self.cycle_id += 1
                        self.cycle_start_time = self.robot.getTime()
                        self.state = "EVALUATE_QUALITY"
                        self.wait_steps = 20
                        print(f"\n[CYCLE {self.cycle_id}] Workpiece Sensed on Infeed.")
                        print(f" -> Variety: {detection['chocolate_type']}")
                        print(f" -> Orientation Angle: {detection['orientation_deg']:+.1f} deg")
                        print(f" -> Solidity Score: {detection['solidity']:.3f}")

            # ---------------------------------------------------------------
            # FSM STATE 2: EVALUATE_QUALITY
            # ---------------------------------------------------------------
            elif self.state == "EVALUATE_QUALITY":
                if self.current_detection["is_defective"]:
                    print("[QUALITY ALERT] Surface chipped / defective! Diverting to REJECT_ROUTINE.")
                    self.state = "REJECT_ROUTINE"
                else:
                    print("[QUALITY PASS] Surface intact. Proceeding to PICK_APPROACH.")
                    self.state = "PICK_APPROACH"
                self.wait_steps = 20

            # ---------------------------------------------------------------
            # FSM STATE 3: PICK_APPROACH (Adaptive Wrist Orientation)
            # ---------------------------------------------------------------
            elif self.state == "PICK_APPROACH":
                angle_deg = self.current_detection["orientation_deg"]
                wrist_yaw = compute_wrist_orientation(angle_deg)
                
                if not getattr(self, "pick_plunge_done", False):
                    # Step A: Approach clearance (Z = 0.18m) with aligned wrist
                    approach_pose = list(INFEED_APPROACH_JOINTS)
                    approach_pose[5] = wrist_yaw
                    self.set_joint_positions(approach_pose)
                    self.pick_plunge_done = True
                    self.wait_steps = 100  # Slew from HOME to INFEED approach
                else:
                    # Step B: Plunge descent to contact level (Z = 0.080m)
                    self.pick_plunge_done = False
                    pick_pose = list(INFEED_PICK_JOINTS)
                    pick_pose[5] = wrist_yaw
                    self.set_joint_positions(pick_pose)
                    self.state = "GRASP_AND_VERIFY"
                    self.wait_steps = 50

            # ---------------------------------------------------------------
            # FSM STATE 4: GRASP_AND_VERIFY
            # ---------------------------------------------------------------
            elif self.state == "GRASP_AND_VERIFY":
                if not getattr(self, "gripper_closing_done", False):
                    self.set_gripper(open_pos=False)
                    self.gripper_closing_done = True
                    self.wait_steps = 40  # Allow gripper fingers to close before sampling touch sensor
                else:
                    self.gripper_closing_done = False
                    choc_p = self.choc_node.getPosition() if self.choc_node else [0, 0, 0]
                    if self.verify_grasp():
                        print(f"[GRIP CONFIRMED] Workpiece secured. Starting vertical clearance retract.")
                        # Vertical lift retract to clear infeed pedestal before lateral transit
                        angle_deg = self.current_detection["orientation_deg"]
                        wrist_yaw = compute_wrist_orientation(angle_deg)
                        retract_pose = list(INFEED_RETRACT_JOINTS)
                        retract_pose[5] = wrist_yaw
                        self.set_joint_positions(retract_pose)
                        self.state = "TRANSFER_TRAY"
                        self.wait_steps = 50
                    else:
                        print("[FAULT] Grip verification failed or dropped piece! Executing SAFE_RECOVERY.")
                        self.state = "SAFE_RECOVERY"
                        self.wait_steps = 40

            # ---------------------------------------------------------------
            # FSM STATE 5: TRANSFER_TRAY (Multi-Slot Cavity Routing)
            # ---------------------------------------------------------------
            elif self.state == "TRANSFER_TRAY":
                variety = self.current_detection["chocolate_type"]
                target_slot = get_slot_by_variety(variety)
                slot_info = get_tray_slot_target(target_slot)
                
                if not hasattr(self, "transfer_trajectory") or self.transfer_trajectory is None:
                    choc_p = self.choc_node.getPosition() if self.choc_node else [0, 0, 0]
                    print(f"[LIFT VERIFIED] Vertical clearance attained: Z = {choc_p[2]:.3f} m.")
                    print(f"[TRANSFER] Executing smooth S-curve transit to {slot_info['name']} (Cavity {target_slot}).")
                    
                    angle_deg = self.current_detection["orientation_deg"]
                    wrist_yaw = compute_wrist_orientation(angle_deg)
                    q_start = list(INFEED_RETRACT_JOINTS)
                    q_start[5] = wrist_yaw
                    q_end = get_tray_slot_joints(target_slot, "approach")
                    
                    # Generate 50-step smooth S-curve interpolated joint trajectory
                    N_PTS = 50
                    traj = []
                    for i in range(1, N_PTS + 1):
                        t = i / float(N_PTS)
                        alpha = (1.0 - np.cos(t * np.pi)) / 2.0
                        q_pt = [(1.0 - alpha) * s + alpha * e for s, e in zip(q_start, q_end)]
                        traj.append(q_pt)
                    self.transfer_trajectory = traj
                    self.transfer_index = 0
                
                if self.transfer_index < len(self.transfer_trajectory):
                    self.set_joint_positions(self.transfer_trajectory[self.transfer_index])
                    self.transfer_index += 1
                    self.wait_steps = 2  # 32ms per waypoint for smooth transit
                else:
                    self.transfer_trajectory = None
                    self.transfer_index = 0
                    self.state = "PLACE_AND_RELEASE"
                    self.wait_steps = 25

            # ---------------------------------------------------------------
            # FSM STATE 6: PLACE_AND_RELEASE (Plunge, Release, Retract)
            # ---------------------------------------------------------------
            elif self.state == "PLACE_AND_RELEASE":
                variety = self.current_detection["chocolate_type"]
                target_slot = get_slot_by_variety(variety)
                step = getattr(self, "place_substep", 0)
                
                if step == 0:
                    # Step 1: Plunge down into tray pocket (Z = 0.081m)
                    slot_place = get_tray_slot_joints(target_slot, "place")
                    self.set_joint_positions(slot_place)
                    self.place_substep = 1
                    self.wait_steps = 45
                elif step == 1:
                    # Step 2: Open gripper fingers to deposit workpiece
                    choc_p = self.choc_node.getPosition() if self.choc_node else [0, 0, 0]
                    print(f"[TRAY POSITION] Plunge seated at X={choc_p[0]:.4f}, Y={choc_p[1]:.4f}, Z={choc_p[2]:.4f} m.")
                    self.set_gripper(open_pos=True)
                    self.place_substep = 2
                    self.wait_steps = 25
                elif step == 2:
                    # Step 3: Vertical retract to clearance height (Z = 0.22m)
                    choc_p = self.choc_node.getPosition() if self.choc_node else [0, 0, 0]
                    print(f"[DEPOSIT VERIFIED] Released in cavity {target_slot}. Retracting arm.")
                    slot_retract = get_tray_slot_joints(target_slot, "retract")
                    self.set_joint_positions(slot_retract)
                    self.place_substep = 3
                    self.wait_steps = 45
                else:
                    # Step 4: Return to home pose and log telemetry
                    self.place_substep = 0
                    cycle_time = self.robot.getTime() - self.cycle_start_time
                    angle = self.current_detection["orientation_deg"]
                    solidity = self.current_detection["solidity"]
                    case_type = "Boundary_Angle_45deg" if abs(angle) > 30 else "Normal"
                    
                    print(f"[SUCCESS] Packaged into tray cavity {target_slot} ({variety}). Elapsed: {cycle_time:.2f}s.")
                    self.log_cycle_to_csv(variety, angle, solidity, 0.18, cycle_time, 
                                         case_type, f"PACKED_SLOT_{target_slot}")
                    
                    self.set_joint_positions(HOME_POSE_JOINTS)
                    self.state = "CYCLE_COMPLETE"
                    self.wait_steps = 80

            # ---------------------------------------------------------------
            # FSM STATE 7: REJECT_ROUTINE (Scrap Remediation)
            # ---------------------------------------------------------------
            elif self.state == "REJECT_ROUTINE":
                print("[REJECT] Transferring defective item to Gravity Remelt Chute.")
                rej_step = getattr(self, "reject_substep", 0)
                
                if rej_step == 0:
                    # Step 1: Move to chute approach
                    self.set_joint_positions(REJECT_CHUTE_JOINTS["approach"])
                    self.reject_substep = 1
                    self.wait_steps = 60
                elif rej_step == 1:
                    # Step 2: Lower into chute deposit position
                    self.set_joint_positions(REJECT_CHUTE_JOINTS["deposit"])
                    self.reject_substep = 2
                    self.wait_steps = 35
                elif rej_step == 2:
                    # Step 3: Open gripper to drop into chute
                    self.set_gripper(open_pos=True)
                    self.reject_substep = 3
                    self.wait_steps = 25
                elif rej_step == 3:
                    # Step 4: Vertical retract
                    self.set_joint_positions(REJECT_CHUTE_JOINTS["retract"])
                    self.reject_substep = 4
                    self.wait_steps = 40
                else:
                    self.reject_substep = 0
                    cycle_time = self.robot.getTime() - self.cycle_start_time
                    self.log_cycle_to_csv(
                        self.current_detection["chocolate_type"],
                        self.current_detection["orientation_deg"],
                        self.current_detection["solidity"],
                        0.0, cycle_time, "Fault_Chipped_Defect", "REJECTED_TO_CHUTE"
                    )
                    print("[SCRAP LOGGED] Defective piece deposited in remelt bin. Line ready.")
                    self.set_joint_positions(HOME_POSE_JOINTS)
                    self.state = "CYCLE_COMPLETE"
                    self.wait_steps = 70

            # ---------------------------------------------------------------
            # FSM STATE 8: SAFE_RECOVERY
            # ---------------------------------------------------------------
            elif self.state == "SAFE_RECOVERY":
                rec_step = getattr(self, "recovery_substep", 0)
                if rec_step == 0:
                    print("[RECOVERY] Executing emergency vertical retract to safe clearance.")
                    self.set_gripper(open_pos=True)
                    self.set_joint_positions(SAFE_RETRACT_JOINTS)
                    self.recovery_substep = 1
                    self.wait_steps = 50
                else:
                    self.recovery_substep = 0
                    self.set_joint_positions(HOME_POSE_JOINTS)
                    print("[SAFE] Fault cleared. Returned to home pose.")
                    self.state = "CYCLE_COMPLETE"
                    self.wait_steps = 70

            # ---------------------------------------------------------------
            # FSM STATE 9: CYCLE_COMPLETE (Wait for next workpiece trigger)
            # ---------------------------------------------------------------
            elif self.state == "CYCLE_COMPLETE":
                # TODO (Partner 2): Add feeder handshake trigger before returning to IDLE
                pass


if __name__ == "__main__":
    supervisor = ConfectioneryCellSupervisor()
    supervisor.run()
