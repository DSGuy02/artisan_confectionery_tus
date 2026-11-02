"""
calibrate_waypoints.py - Offline inverse kinematics for the named cell waypoints.

Solves the 6-DOF UR5e joint angles for every Cartesian target in waypoints.py
using the forward kinematics model (checked against Webots to < 0.5 mm).
Constraints applied to every pose:
  - gripper approach axis pointing straight down (world -Z)
  - finger closing direction along world +Y (yaw = 0), so wrist_3 can then be
    offset by the vision angle theta at the infeed
  - elbow-up configuration inside the UR5e joint limits

Run from the project root with the venv active (needs scipy, not used in Webots):
    python3 motion/calibrate_waypoints.py
Paste the printed tables into motion/waypoints.py.
"""

import os
import sys

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from motion.waypoints import (  # noqa: E402
    forward_kinematics, INFEED_PICK_TARGET, TRAY_SLOT_TARGETS,
    REJECT_CHUTE_TARGET, SAFE_RETRACT_TARGET, Z_RETRACT
)

BOUNDS = [(-np.pi, np.pi),   # shoulder_pan
          (-np.pi, 0.0),     # shoulder_lift (elbow-up)
          (0.0, np.pi),      # elbow (hard limit +/- pi in the PROTO)
          (-np.pi, np.pi),   # wrist_1
          (-np.pi, np.pi),   # wrist_2
          (-np.pi, np.pi)]   # wrist_3

DOWN = np.array([0.0, 0.0, -1.0])


def solve(target_xyz, seed, yaw=0.0):
    target = np.asarray(target_xyz, dtype=float)
    close_dir = np.array([-np.sin(yaw), np.cos(yaw), 0.0])

    def cost(q):
        p, R = forward_kinematics(q)
        return (1e4 * np.sum((p - target) ** 2)
                + np.sum((R[:, 2] - DOWN) ** 2)
                + np.sum((R[:, 1] - close_dir) ** 2))

    best = None
    for s in [seed] + [np.array(seed) + np.random.uniform(-0.4, 0.4, 6) for _ in range(6)]:
        res = minimize(cost, s, method="SLSQP", bounds=BOUNDS,
                       options={"maxiter": 500, "ftol": 1e-12})
        if best is None or res.fun < best.fun - 1e-9:
            best = res
    p, R = forward_kinematics(best.x)
    pos_err_mm = np.linalg.norm(p - target) * 1000.0
    tilt_deg = np.degrees(np.arccos(np.clip(-R[2, 2], -1, 1)))
    return [round(float(v), 4) for v in best.x], pos_err_mm, tilt_deg


def fmt(q):
    return "[" + ", ".join(f"{v:.4f}" for v in q) + "]"


def main():
    np.random.seed(0)
    report = []

    def run(name, xyz, seed):
        q, e, t = solve(xyz, seed)
        report.append((name, e, t))
        return q

    # Infeed (seed: arm swung towards +Y)
    seed = [1.57, -1.2, 1.8, -2.2, -1.57, 0.0]
    inf = {k: run(f"infeed_{k}", INFEED_PICK_TARGET[k][:3], seed) for k in ("approach", "pick", "retract")}

    tray = {}
    for sid, t in TRAY_SLOT_TARGETS.items():
        seed = [0.55, -1.2, 1.8, -2.2, -1.57, 0.0]
        tray[sid] = {
            "approach": run(f"slot{sid}_approach", [t["X"], t["Y"], t["Z_approach"]], seed),
            "place":    run(f"slot{sid}_place",    [t["X"], t["Y"], t["Z_place"]], seed),
            "retract":  run(f"slot{sid}_retract",  [t["X"], t["Y"], Z_RETRACT], seed),
        }

    seed = [2.5, -1.2, 1.8, -2.2, -1.57, 0.0]
    rej = {k: run(f"reject_{k}", REJECT_CHUTE_TARGET[k][:3], seed) for k in ("approach", "deposit", "retract")}

    safe = run("safe_retract", SAFE_RETRACT_TARGET[:3], [1.57, -1.2, 1.8, -2.2, -1.57, 0.0])

    print("# ---- paste into motion/waypoints.py ----")
    print(f"INFEED_APPROACH_JOINTS = {fmt(inf['approach'])}")
    print(f"INFEED_PICK_JOINTS     = {fmt(inf['pick'])}")
    print(f"INFEED_RETRACT_JOINTS  = {fmt(inf['retract'])}")
    print("TRAY_SLOT_JOINTS = {")
    for sid, d in tray.items():
        print(f"    {sid}: {{")
        print(f"        \"approach\": {fmt(d['approach'])},")
        print(f"        \"place\":    {fmt(d['place'])},")
        print(f"        \"retract\":  {fmt(d['retract'])}")
        print("    }" + ("," if sid < 4 else ""))
    print("}")
    print("REJECT_CHUTE_JOINTS = {")
    print(f"    \"approach\": {fmt(rej['approach'])},")
    print(f"    \"deposit\":  {fmt(rej['deposit'])},")
    print(f"    \"retract\":  {fmt(rej['retract'])}")
    print("}")
    print(f"SAFE_RETRACT_JOINTS = {fmt(safe)}")
    print("\n# ---- accuracy (model) ----")
    for name, e, t in report:
        print(f"{name:<18} pos err {e:6.3f} mm   tilt {t:5.2f} deg")


if __name__ == "__main__":
    main()
