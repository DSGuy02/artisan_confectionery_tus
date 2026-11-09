#!/usr/bin/env python3
"""
test_trajectory_planner.py - Unit tests for trajectory generation & S-curve motion profiling.
Technological University of the Shannon (TUS Athlone)
Integrated Robotics 4 - Continuous Assessment 3
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from motion.trajectory_planner import (
    generate_joint_s_curve_trajectory,
    generate_cartesian_trajectory,
    compute_adaptive_approach_pose
)
from motion.waypoints import (
    INFEED_RETRACT_JOINTS,
    TRAY_SLOT_JOINTS,
    TRAY_SLOT_TARGETS,
    REJECT_CHUTE_JOINTS,
    SAFE_RETRACT_JOINTS,
    HOME_POSE_JOINTS,
    T_W_R0,
    T_W_C,
    T_W_F,
    T_W_T,
    T_W_Q,
    transform_point,
    invert_transformation,
    get_tray_slot_transform,
    forward_kinematics,
    forward_kinematics_homogeneous,
    verify_reachability
)


class TestTrajectoryPlanner(unittest.TestCase):
    
    def test_joint_s_curve_endpoint_accuracy(self):
        """Verifies that the S-curve trajectory terminates exactly at the target configuration."""
        q_start = INFEED_RETRACT_JOINTS
        q_end = TRAY_SLOT_JOINTS[1]["approach"]
        num_steps = 50
        
        traj = generate_joint_s_curve_trajectory(q_start, q_end, num_steps=num_steps)
        
        self.assertEqual(len(traj), num_steps)
        # Final point should match q_end to sub-milliradian precision
        for actual, expected in zip(traj[-1], q_end):
            self.assertAlmostEqual(actual, expected, places=4)
            
    def test_joint_s_curve_smoothness(self):
        """
        Verifies that velocity profile starts and ends near zero (bell-shaped cosine profile),
        with peak velocity occurring near the trajectory midpoint (t = 0.5).
        """
        q_start = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        q_end = [1.0, -1.0, 1.5, -0.5, 1.2, 2.0]
        num_steps = 60
        
        traj = np.array(generate_joint_s_curve_trajectory(q_start, q_end, num_steps=num_steps))
        
        # Numerical velocity (finite differences between consecutive waypoints)
        velocities = np.diff(traj, axis=0)
        
        # First delta and last delta should be minimal (near zero velocity at boundaries)
        v_start = np.linalg.norm(velocities[0])
        v_end = np.linalg.norm(velocities[-1])
        v_mid = np.linalg.norm(velocities[len(velocities) // 2])
        
        self.assertLess(v_start, v_mid * 0.15, "Start velocity should be much smaller than peak velocity")
        self.assertLess(v_end, v_mid * 0.15, "End velocity should be much smaller than peak velocity")
        self.assertGreater(v_mid, v_start * 5.0, "Midpoint velocity should peak cleanly")
        
    def test_reject_chute_trajectory_generation(self):
        """Verifies that S-curve trajectory can be generated cleanly from infeed to reject chute with shortest angular path."""
        q_start = list(INFEED_RETRACT_JOINTS)
        q_start[5] = 2.8000
        q_end = REJECT_CHUTE_JOINTS["approach"]
        
        traj = generate_joint_s_curve_trajectory(q_start, q_end, num_steps=50)
        self.assertEqual(len(traj), 50)
        # Joints 0-4 match exact coordinates
        for i in range(5):
            self.assertAlmostEqual(traj[-1][i], q_end[i], places=4)
        # Joint 5 (wrist_3) must match target orientation (difference is 0 modulo 2*pi)
        wrist_diff = (traj[-1][5] - q_end[5] + np.pi) % (2.0 * np.pi) - np.pi
        self.assertAlmostEqual(wrist_diff, 0.0, places=4)
        # Total angular change for wrist 3 must be under 180 degrees (pi radians) due to unwrapping
        self.assertLess(abs(traj[-1][5] - q_start[5]), np.pi)

    def test_cartesian_trajectory(self):
        """Verifies Cartesian linear and S-curve path generation."""
        p_start = [0.0, 0.40, 0.18]
        p_end = [0.35, 0.20, 0.18]
        
        traj_linear = generate_cartesian_trajectory(p_start, p_end, num_steps=20, s_curve=False)
        traj_scurve = generate_cartesian_trajectory(p_start, p_end, num_steps=20, s_curve=True)
        
        self.assertEqual(len(traj_linear), 21)
        self.assertEqual(len(traj_scurve), 21)
        self.assertAlmostEqual(traj_linear[-1][0], 0.35, places=4)
        self.assertAlmostEqual(traj_scurve[-1][0], 0.35, places=4)

    def test_adaptive_approach_pose(self):
        """Verifies adaptive orientation calculation for vision coordinates."""
        poses = compute_adaptive_approach_pose(0.01, 0.39, 45.0)
        self.assertIn("approach_pose", poses)
        self.assertIn("pick_pose", poses)
        self.assertIn("lift_pose", poses)
        self.assertAlmostEqual(poses["approach_pose"][5], np.radians(45.0), places=4)

    def test_all_blister_tray_slots_kinematics_and_trajectories(self):
        """
        Validates forward kinematics positioning accuracy and S-curve joint trajectory
        interpolation across all 4 blister tray pockets (Dark, Milk, Ruby, White).
        """
        q_start = list(INFEED_RETRACT_JOINTS)
        q_start[5] = 2.8000
        
        for slot_id in range(1, 5):
            target = TRAY_SLOT_TARGETS[slot_id]
            q_app = TRAY_SLOT_JOINTS[slot_id]["approach"]
            q_place = TRAY_SLOT_JOINTS[slot_id]["place"]
            q_ret = TRAY_SLOT_JOINTS[slot_id]["retract"]
            
            # 1. Verify S-curve transit trajectory from infeed retract to slot approach
            traj = generate_joint_s_curve_trajectory(q_start, q_app, num_steps=50)
            self.assertEqual(len(traj), 50)
            for i in range(5):
                self.assertAlmostEqual(traj[-1][i], q_app[i], places=4)
                
            # 2. Verify approach pose positioning (Z = 0.180m clearance)
            pos_app, rot_app = forward_kinematics(q_app)
            self.assertAlmostEqual(pos_app[0], target["X"], places=3)
            self.assertAlmostEqual(pos_app[1], target["Y"], places=3)
            self.assertAlmostEqual(pos_app[2], 0.1800, places=3)
            self.assertAlmostEqual(rot_app[2, 2], -1.0, places=3)  # Gripper points straight down
            
            # 3. Verify pocket plunge seating pose (Z = 0.080m contact level)
            pos_place, rot_place = forward_kinematics(q_place)
            self.assertAlmostEqual(pos_place[0], target["X"], places=3)
            self.assertAlmostEqual(pos_place[1], target["Y"], places=3)
            self.assertAlmostEqual(pos_place[2], 0.0800, places=3)
            self.assertAlmostEqual(rot_place[2, 2], -1.0, places=3)
            
            # 4. Verify vertical retract pose (Z = 0.220m clearance height)
            pos_ret, rot_ret = forward_kinematics(q_ret)
            self.assertAlmostEqual(pos_ret[0], target["X"], places=3)
            self.assertAlmostEqual(pos_ret[1], target["Y"], places=3)
            self.assertAlmostEqual(pos_ret[2], 0.2200, places=3)
            self.assertAlmostEqual(rot_ret[2, 2], -1.0, places=3)

    def test_all_cell_targets_reachability(self):
        """Verifies that all cell targets (infeed, 4 tray slots, reject chute) fall within optimal reach."""
        report = verify_reachability()
        self.assertEqual(len(report), 6)
        for name, metrics in report.items():
            self.assertEqual(metrics["status"], "PASS (Optimal Dexterity)", 
                             f"Target {name} failed optimal dexterity check: {metrics}")

    def test_emergency_safe_retract_kinematics(self):
        """
        Verifies that SAFE_RETRACT_JOINTS commands a vertical clearance altitude of Z = 0.250m
        above the infeed pick position with perpendicular downward orientation.
        """
        pos, rot = forward_kinematics(SAFE_RETRACT_JOINTS)
        self.assertAlmostEqual(pos[0], 0.0000, places=3)
        self.assertAlmostEqual(pos[1], 0.4000, places=3)
        self.assertAlmostEqual(pos[2], 0.2500, places=3)
        self.assertAlmostEqual(rot[2, 2], -1.0, places=3)

    def test_homogeneous_coordinate_transformations(self):
        """
        Verifies analytical properties and consistency of 4x4 homogeneous transformation matrices:
        1. Robot base {R0} coincident with World origin {W} (T_W_R0 = Identity).
        2. Analytical invertibility: T @ T^-1 = Identity.
        3. Blister tray cavity positions derived from T_W_T @ T_T_slot match target coordinates.
        4. forward_kinematics_homogeneous matches forward_kinematics.
        5. Point transformation matches manual matrix-vector multiplication.
        """
        # 1. Base frame coincidence
        np.testing.assert_allclose(T_W_R0, np.eye(4), atol=1e-6)
        
        # 2. Invertibility check on all reference frames
        for T_name, T_mat in [("Camera", T_W_C), ("Infeed", T_W_F), ("Tray", T_W_T), ("Reject", T_W_Q)]:
            T_inv = invert_transformation(T_mat)
            identity_prod = T_mat @ T_inv
            np.testing.assert_allclose(identity_prod, np.eye(4), atol=1e-6,
                                       err_msg=f"Inversion failed for {T_name} transformation")

        # 3. Tray slot transformation chain: T_W_slot = T_W_T @ T_T_slot
        for slot_id in range(1, 5):
            T_slot = get_tray_slot_transform(slot_id, stage="place")
            target = TRAY_SLOT_TARGETS[slot_id]
            self.assertAlmostEqual(T_slot[0, 3], target["X"], places=3)
            self.assertAlmostEqual(T_slot[1, 3], target["Y"], places=3)
            self.assertAlmostEqual(T_slot[2, 3], target["Z_place"], places=3)

        # 4. Homogeneous forward kinematics consistency with standard FK
        T_fk = forward_kinematics_homogeneous(HOME_POSE_JOINTS)
        pos_fk, rot_fk = forward_kinematics(HOME_POSE_JOINTS)
        np.testing.assert_allclose(T_fk[:3, 3], pos_fk, atol=1e-5)
        np.testing.assert_allclose(T_fk[:3, :3], rot_fk, atol=1e-5)

        # 5. Point transformation test
        test_pt = [0.045, 0.045, 0.041]
        p_transformed = transform_point(T_W_T, test_pt)
        expected_pt = [0.395, 0.245, 0.081]  # Slot 4 place position
        np.testing.assert_allclose(p_transformed, expected_pt, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
