# Artisan Confectionery Quality Inspection & Packaging Robotic Cell

**Module:** Integrated Robotics 4 (Year 4, Semester 1) — TUS Athlone  
**Assessment:** Continuous Assessment 3 — Integration Demonstration & Technical Report (65% Weighting)  
**Authors:** Oladeji Sanyaolu (A00316054) & Midia Khalaf (A00316264) — Group 2  
**Platform:** Cyberbotics Webots + Python 3 + OpenCV  

---

## 1. Project Overview

This project implements an autonomous robotic cell for high-end artisan confectionery packaging. The cell utilizes a 6-axis **Universal Robots UR5e** manipulator equipped with a **Robotiq 2F-85 parallel gripper** and internal touch sensing. An overhead camera and **OpenCV** pipeline detect incoming chocolates, calculate their arrival orientation angle ($\theta$), inspect surface integrity (via contour solidity), pack compliant chocolates into a 2x2 presentation blister tray, and reject defective/chipped chocolates into a remelt chute.

---

## 2. Repository Structure

```
artisan_confectionery_project/
├── .gitignore                      # Git ignore rules for Python, Webots, and OS caches
├── requirements.txt                # Python dependencies (numpy, opencv-python, matplotlib)
├── setup_env.sh                    # Automated venv creation and setup script
├── setup_webots_flatpak.sh         # Installs numpy & OpenCV into Webots Flatpak sandbox
├── README.md                       # Main project documentation and instructions
│
├── worlds/                         # Webots simulation worlds
│   └── confectionery_cell.wbt      # 3D world with UR5e, Robotiq 2F-85, tables, and fixtures
│
├── controllers/                    # Robot controllers
│   └── cell_supervisor/           # Supervisor controller package
│       └── cell_supervisor.py     # Master Python Finite State Machine (FSM)
│
├── protos/                         # Webots custom PROTO nodes
│   ├── CellUR5e.proto              # UR5e with baseSlot for fixed overhead camera
│   ├── ChocolateDark.proto         # Dark chocolate model
│   ├── ChocolateMilk.proto         # Milk chocolate model
│   └── ChocolateRuby.proto         # Ruby chocolate model
│
├── perception/                     # Computer vision pipeline
│   ├── __init__.py
│   ├── vision_pipeline.py          # OpenCV segmentation, color, angle, and defect screening
│   └── test_vision_offline.py      # Standalone verification script (runs without Webots)
│
├── motion/                         # Kinematics and motion planning
│   ├── __init__.py
│   ├── waypoints.py                # Named targets, coordinate frames {W, R0, TCP, C, T, Q}
│   └── trajectory_planner.py       # Adaptive approach poses and S-curve velocity profiling
│
├── data/                           # Runtime telemetry & logs
│   └── production_metrics.csv      # Logged cycle times, angles, and defect classifications
│
├── analytics/                      # Post-run analytics and visualization
│   └── plot_metrics.py             # Script to generate report figures (cycle times & yield)
│
└── docs/                           # Guidance and technical references
    ├── partner_1_guide.md          # Step-by-step instructions for Partner 1 (Simulation & Motion)
    └── partner_2_guide.md          # Step-by-step instructions for Partner 2 (Vision & Logic)
```

---

## 3. Quickstart Guide

### 3.1 Setup Python Virtual Environment
```bash
./setup_env.sh
source venv/bin/activate
```
*(If you run Webots via **Flatpak**, also run `./setup_webots_flatpak.sh` once so Webots' sandboxed Python has `numpy` and `opencv-python-headless` installed).*

### 3.2 Verify Vision Pipeline Offline (No Webots Needed)
Partner 2 can test the OpenCV pipeline immediately on synthetic images:
```bash
python3 perception/test_vision_offline.py
```
This inspects sample chocolates and saves annotated images to `perception/test_results/`.

### 3.3 Launch Simulation in Webots
1. Open **Cyberbotics Webots**.
2. Open world: `File -> Open World... -> artisan_confectionery_project/worlds/confectionery_cell.wbt`.
3. The robot will automatically bind to the controller at `controllers/cell_supervisor/cell_supervisor.py`.
4. Press **Play** in Webots to run the simulation.

### 3.4 Generate Performance Plots for Report
```bash
python3 analytics/plot_metrics.py
```
This generates `data/performance_plot.png` for inclusion in your CA-3 report.

---

## 4. Work Breakdown (50/50 Division)

* **Partner 1 (Simulation Modeling, Kinematics & Tooling):**
  - Webots 3D cell world layout, lighting, and table arrangement.
  - UR5e manipulator and Robotiq 2F-85 gripper modeling.
  - Coordinate frames ($W, R_0, TCP, C, T, Q$) and Cartesian waypoints.
  - TouchSensor feedback and physical grip verification.
  - Trajectory tuning (approach, plunge, lift, deposit).
  - *Refer to [docs/partner_1_guide.md](docs/partner_1_guide.md)*

* **Partner 2 (Computer Vision, Supervisory Logic & Telemetry):**
  - OpenCV image acquisition from Webots `Camera` node.
  - HSV color classification (Dark, Milk, Ruby, White).
  - Contour centroid and minAreaRect angle ($\theta$) calculation.
  - Defect detection via contour solidity ($\text{Area} / \text{Convex Hull Area} < 0.88$).
  - Python Finite State Machine (FSM) supervisory logic.
  - Fault recovery routines, CSV metrics logging, and Matplotlib graphs.
  - *Refer to [docs/partner_2_guide.md](docs/partner_2_guide.md)*

---

## 5. Demonstration Protocol (20 Marks)

1. **Normal Cycle (6 marks):** Product sensed $\to$ OpenCV classifies type $\to$ UR5e grasps and confirms touch contact $\to$ transfers and places into 2x2 blister tray.
2. **Boundary / Variation Case (3 marks):** Chocolate arrives rotated at $45^\circ \to$ OpenCV measures angle $\to$ robot dynamically aligns wrist Joint 6 during descent $\to$ squares into tray.
3. **Fault & Recovery Case (4 marks):** Chipped chocolate arrives ($\text{Solidity} < 0.88$) $\to$ defect flagged $\to$ diverted to remelt chute $\to$ line resumes automatically without human reset.
4. **Coordination & Metrics (7 marks):** Real-time console status, active sensor handshakes, and quantitative CSV performance metrics.
