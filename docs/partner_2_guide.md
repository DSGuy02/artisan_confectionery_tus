# Partner 2 Guide: Computer Vision, Supervisory Control & Analytics

**Lead Role:** Partner 2 (Midia Khalaf / Perception & Software Lead)  
**Primary Focus:** OpenCV Image Processing, HSV Classification, Pose/Angle Estimation, Defect Detection, Supervisory FSM, Telemetry Logging  
**Associated Code Files:** `perception/vision_pipeline.py`, `perception/test_vision_offline.py`, `controllers/cell_supervisor/cell_supervisor.py`, `analytics/plot_metrics.py`

---

## 1. Immediate Next Steps & Setup

### Step 1: Run Offline Vision Tests (No Webots Needed)
You can start developing and testing the computer vision pipeline right away on your machine without opening Webots:
1. Activate your Python environment:
   ```bash
   source venv/bin/activate
   ```
2. Run the offline test suite:
   ```bash
   python3 perception/test_vision_offline.py
   ```
3. Inspect the terminal output and view the generated annotated images in:
   `perception/test_results/`
   - `annotated_dark_chocolate_normal.png` (Pass: 0.0° angle, high solidity)
   - `annotated_milk_chocolate_rotated_45deg.png` (Pass: 45.0° angle measured)
   - `annotated_dark_chocolate_chipped_defect.png` (Reject: Solidity < 0.88 detected!)

### Step 2: Understand the OpenCV Vision Pipeline
Review `perception/vision_pipeline.py`:
1. **Color Segmentation:** Converts camera frame to HSV color space. Masks isolate Dark, Milk, Ruby, and White chocolates.
2. **Pose & Centroid:** Calculates image moments ($M_{00}, M_{10}, M_{01}$) for $(cx, cy)$ and projects to world coordinates $(X_w, Y_w)$.
3. **Orientation Angle ($\theta$):** Uses `cv2.minAreaRect` to find the principal grasp axis. Outputs angle in degrees $\theta \in [-90^\circ, +90^\circ]$.
4. **Defect Screening (Solidity Metric):**
   $$\text{Solidity} = \frac{\text{Contour Area}}{\text{Convex Hull Area}}$$
   - Intact chocolates yield $\text{Solidity} \ge 0.92$.
   - Chipped / broken chocolates yield $\text{Solidity} < 0.88 \implies$ Flags `is_defective = True`.

---

## 2. Key Tasks & Milestones for Partner 2

### Milestone 1: Connect Camera to Controller (Due Week 7–8)
- [ ] In `controllers/cell_supervisor/cell_supervisor.py`, enable the overhead camera:
  ```python
  camera = robot.getDevice("camera")
  camera.enable(time_step)
  ```
- [ ] Convert Webots raw image buffer (`camera.getImage()`) into an OpenCV BGR image using `convert_webots_to_cv()`.
- [ ] Display camera frame or verify image shape $(480, 640, 3)$ during simulation steps.

### Milestone 2: HSV Calibration & Threshold Tuning (Due Week 9)
- [ ] Calibrate HSV ranges under the Webots lighting environment:
  - Dark Chocolate: Low value / brightness ($V < 85$)
  - Milk Chocolate: Medium brown ($H \in [10, 25]$)
  - Ruby Chocolate: Pink / magenta ($H \in [145, 175]$)
  - White Chocolate: Low saturation ($S < 40, V > 160$)
- [ ] Test with lighting variations to ensure robust classification.

### Milestone 3: Supervisory State Machine & Handshaking (Due Week 10)
- [ ] Refine the Finite State Machine (FSM) in `controllers/cell_supervisor/cell_supervisor.py`:
  - `IDLE` $\to$ Sensed $\to$ `EVALUATE_QUALITY`
  - Quality Pass $\to$ `PICK_APPROACH` $\to$ `GRASP_AND_VERIFY` $\to$ `TRANSFER_TRAY`
  - Defect Detected $\to$ `REJECT_ROUTINE` (route to remelt chute)
  - Lost Grip $\to$ `SAFE_RECOVERY` (retract vertically and clear fault)
- [ ] Ensure non-blocking execution using `robot.step(time_step)`.

### Milestone 4: Telemetry, CSV Logging & Reporting Plots (Due Week 11)
- [ ] Ensure every completed cycle records a row in `data/production_metrics.csv`:
  - Columns: `Timestamp, CycleID, Variety, AngleDeg, Solidity, GripForceN, CycleTimeSec, CaseType, Status`
- [ ] Run `python3 analytics/plot_metrics.py` to generate `data/performance_plot.png`.
- [ ] Calculate quantitative metrics: Mean cycle time, defect catch rate (100%), false reject rate (0%).

---

## 3. Deliverables for the CA-3 Final Report (Partner 2 Chapters)
* **Chapter 3 (Perception & Supervisory Control):** OpenCV algorithm flow, HSV thresholds, solidity formula, FSM state diagram.
* **Chapter 4 (Fault Detection & Recovery):** Defect detection logic, lost-grip detection, automatic recovery without human intervention.
* **Chapter 5 (Validation & Results):** Experimental results tables, CSV telemetry logs, Matplotlib cycle time and yield charts.
