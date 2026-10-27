# Partner 1 Guide: Simulation Modeling, Kinematics & Mechanics

**Lead Role:** Partner 1 (Oladeji Sanyaolu / Mechanical & Simulation Lead)  
**Primary Focus:** Webots 3D Cell Modeling, UR5e Robot Kinematics, EOAT Gripper & Sensor Integration, Motion Trajectories  
**Associated Code Files:** `worlds/confectionery_cell.wbt`, `motion/waypoints.py`, `motion/trajectory_planner.py`, `protos/`

---

## 1. Immediate Next Steps & Setup

### Step 1: Open the World in Webots
1. Launch **Cyberbotics Webots**.
2. Select **File > Open World...** and navigate to:
   `artisan_confectionery_project/worlds/confectionery_cell.wbt`
3. Inspect the scene tree on the left panel:
   - Check the **CellUR5e** robot node (`protos/CellUR5e.proto`).
   - Check that its `baseSlot` contains the fixed overhead **Camera** node at $[0, 0.4, 0.8]$ looking straight down.
   - Check that its `toolSlot` contains the **Robotiq2f85Gripper** and **TouchSensor**.
   - Locate the **infeed_pedestal**, **blister_tray_fixture**, and **remelt_reject_chute**.

### Step 2: Test Robot Joint Motors & Poses
1. In `controllers/cell_supervisor/cell_supervisor.py`, review how the 6 UR5e joint motors are named:
   - `shoulder_pan_joint`
   - `shoulder_lift_joint`
   - `elbow_joint`
   - `wrist_1_joint`
   - `wrist_2_joint`
   - `wrist_3_joint`
2. Test moving the robot to `HOME_POSE_JOINTS` and verifying that the arm moves smoothly without colliding with the table or camera mount.

### Step 3: Implement & Verify Gripper Contact
1. Verify the Robotiq 2F-85 gripper action (in Webots R2025a `Robotiq2f85Gripper.proto`):
   - Left finger motor: `ROBOTIQ 2F-85 Gripper::left finger joint` (or `robotiq_85_left_knuckle_joint`)
   - Right finger motor: `ROBOTIQ 2F-85 Gripper::right finger joint` (or `robotiq_85_right_knuckle_joint`)
   - Open position: `0.0 rad`
   - Closed position: `0.80 rad`
2. Configure or verify the `TouchSensor` on the inner finger pad:
   - In Webots, ensure the touch sensor returns a non-zero value when clamping onto the chocolate workpiece.
   - If using a simulated distance sensor, calibrate the threshold to detect workpiece presence.

---

## 2. Key Tasks & Milestones for Partner 1

### Milestone 1: Coordinate Frames & Reachability (Due Week 7–8)
- [ ] Define the mathematical transformations between frames in `motion/waypoints.py`:
  - World $\{W\}$ $\to$ Robot Base $\{R_0\}$
  - Robot Base $\{R_0\}$ $\to$ Tool Center Point $\{TCP\}$
  - World $\{W\}$ $\to$ Blister Tray $\{T\}$
  - World $\{W\}$ $\to$ Reject Chute $\{Q\}$
- [ ] Verify that all 4 blister tray pockets are well within the UR5e reach radius ($850\text{ mm}$) and free of kinematic singularities.

### Milestone 2: Pick-and-Place Motion Trajectories (Due Week 9)
- [ ] Implement smooth Cartesian vertical approach and retract vectors:
  - Approach offset: $+60\text{ mm}$ above pick point ($Z_{\text{approach}} = 0.18\text{ m}$)
  - Descent plunge: $v_z \le 0.05\text{ m/s}$ to $Z_{\text{pick}} = 0.05\text{ m}$
  - Lift retract: $+100\text{ mm}$ before starting horizontal transfer
- [ ] Test transferring the chocolate to Tray Slot 1 ($X=0.35\text{ m}, Y=0.20\text{ m}, Z=0.05\text{ m}$) without knocking over neighboring chocolates.

### Milestone 3: Dynamic Orientation Wrist Alignment (Due Week 10)
- [ ] Connect with Partner 2's vision output:
  - Receive detected orientation angle $\theta$ (in degrees).
  - Convert $\theta$ to radians and apply it directly to `wrist_3_joint` (Joint 6 / $RZ$).
  - Demonstrate picking a chocolate that arrives rotated at $45^\circ$ (Boundary Case).

### Milestone 4: Safety Bounds & Emergency Recovery (Due Week 11)
- [ ] Program the safe park/retract waypoint:
  - If grip is lost mid-transit, immediately lift vertically to $Z=0.25\text{ m}$, open gripper, and return to Home without colliding with fixtures.
- [ ] Test the divert trajectory to the gravity remelt chute (Fault Recovery Case).

---

## 3. Deliverables for the CA-3 Final Report (Partner 1 Chapters)
* **Chapter 2 (Cell Design & Layout):** 3D Webots layout screenshot, dimensioned CAD/table drawing, EOAT gripper selection rationale.
* **Chapter 3 (Kinematics & Motion Implementation):** Coordinate frame definitions ($W, R_0, TCP, C, T, Q$), Cartesian waypoints table, singularity analysis.
* **Chapter 4 (Safety Provisions):** Safe operating boundaries, emergency retract vectors, collision mitigation.
