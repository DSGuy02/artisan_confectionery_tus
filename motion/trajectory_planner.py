"""
trajectory_planner.py - Trajectory Generation & Wrist Orientation Interpolation
Technological University of the Shannon (TUS Athlone)
Integrated Robotics 4 - Continuous Assessment 3
"""

import numpy as np


def compute_adaptive_approach_pose(detected_x, detected_y, orientation_deg, 
                                   approach_z=0.18, pick_z=0.05):
    """
    Computes Cartesian approach and pick targets incorporating vision-detected
    world coordinates (X, Y) and wrist orientation angle theta.
    
    Args:
        detected_x: World X coordinate of chocolate centroid in meters
        detected_y: World Y coordinate of chocolate centroid in meters
        orientation_deg: Detected arrival angle theta in degrees [-90, +90]
        approach_z: Safe clearance altitude before vertical descent
        pick_z: Contact altitude for gripper jaw closure
        
    Returns:
        dict: Target poses for approach, plunge, and lift
    """
    # Convert angle from degrees to radians
    yaw_rad = np.radians(orientation_deg)
    
    # Flange orientation: Downward facing (Roll = pi, Pitch = 0, Yaw = theta)
    return {
        "approach_pose": [detected_x, detected_y, approach_z, 3.14159, 0.0, yaw_rad],
        "pick_pose":     [detected_x, detected_y, pick_z,     3.14159, 0.0, yaw_rad],
        "lift_pose":     [detected_x, detected_y, approach_z, 3.14159, 0.0, yaw_rad]
    }


def generate_cartesian_trajectory(start_pos, end_pos, num_steps=20, s_curve=True):
    """
    Generates a smoothly interpolated Cartesian path between two 3D positions.
    
    Args:
        start_pos: [X, Y, Z] starting position
        end_pos: [X, Y, Z] ending position
        num_steps: Number of intermediate interpolation steps
        s_curve: If True, uses smooth cosine S-curve ramping for gentle handling
        
    Returns:
        list of [X, Y, Z] waypoints
    """
    start = np.array(start_pos)
    end = np.array(end_pos)
    waypoints = []
    
    for i in range(num_steps + 1):
        t = i / float(num_steps)
        if s_curve:
            # S-curve velocity profiling: zero velocity/acceleration at endpoints
            alpha = (1.0 - np.cos(t * np.pi)) / 2.0
        else:
            alpha = t
            
        point = start + alpha * (end - start)
        waypoints.append(point.tolist())
        
    return waypoints
