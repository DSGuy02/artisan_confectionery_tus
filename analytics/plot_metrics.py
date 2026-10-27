"""
plot_metrics.py - Telemetry Analytics & Performance Plotting
Generates high-resolution performance charts for the CA-3 Technical Report.
"""

import os
import csv
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))


def generate_performance_plots(csv_path=None, output_plot_path=None):
    """
    Parses production CSV telemetry and generates performance graphs.
    """
    if csv_path is None:
        csv_path = os.path.join(PROJECT_ROOT, "data/production_metrics.csv")
    if output_plot_path is None:
        output_plot_path = os.path.join(PROJECT_ROOT, "data/performance_plot.png")

    if not os.path.exists(csv_path):
        print(f"[ERROR] CSV metrics file not found at: {csv_path}")
        return False
        
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARNING] Matplotlib not installed in current environment. Install via requirements.txt.")
        return False

    cycles = []
    cycle_times = []
    solidities = []
    case_types = []
    statuses = []

    with open(csv_path, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cycles.append(int(row["CycleID"]))
            cycle_times.append(float(row["CycleTimeSec"]))
            solidities.append(float(row["Solidity"]))
            case_types.append(row["CaseType"])
            statuses.append(row["Status"])

    if not cycles:
        print("[WARNING] No cycle data found in CSV.")
        return False

    # Setup 2-panel figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("Artisan Confectionery Robotic Packaging Cell — Performance Metrics", fontsize=14, fontweight='bold')

    # -----------------------------------------------------------------------
    # SUBPLOT 1: Cycle Time by Trial
    # -----------------------------------------------------------------------
    normal_idx = [i for i, c in enumerate(case_types) if c == "Normal"]
    boundary_idx = [i for i, c in enumerate(case_types) if "Boundary" in c]
    fault_idx = [i for i, c in enumerate(case_types) if "Fault" in c]

    ax1.plot(cycles, cycle_times, color='#34495e', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Scatter points colored by case type
    ax1.scatter([cycles[i] for i in normal_idx], [cycle_times[i] for i in normal_idx],
                color='#27ae60', s=90, label='Normal Cycle', zorder=5)
    ax1.scatter([cycles[i] for i in boundary_idx], [cycle_times[i] for i in boundary_idx],
                color='#f39c12', s=110, marker='^', label='Boundary Case (45° Rotate)', zorder=5)
    ax1.scatter([cycles[i] for i in fault_idx], [cycle_times[i] for i in fault_idx],
                color='#e74c3c', s=120, marker='X', label='Fault Case (Chipped Reject)', zorder=5)

    mean_time = np.mean(cycle_times)
    ax1.axhline(mean_time, color='#2980b9', linestyle=':', linewidth=2, label=f'Mean Time ({mean_time:.2f}s)')

    ax1.set_title("Cycle Execution Times by Trial", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Cycle ID", fontsize=10)
    ax1.set_ylabel("Execution Time (seconds)", fontsize=10)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='lower left', fontsize=9)
    ax1.set_ylim(2.5, 5.5)

    # -----------------------------------------------------------------------
    # SUBPLOT 2: Solidity Distribution & Quality Threshold
    # -----------------------------------------------------------------------
    colors = ['#e74c3c' if s < 0.88 else '#27ae60' for s in solidities]
    ax2.bar(cycles, solidities, color=colors, edgecolor='#2c3e50', width=0.6)

    # Threshold line
    ax2.axhline(0.88, color='#c0392b', linestyle='--', linewidth=2, label='Quality Defect Threshold (0.88)')

    ax2.set_title("Contour Solidity Inspection by Workpiece", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Cycle ID", fontsize=10)
    ax2.set_ylabel("Solidity Ratio (Area / Hull Area)", fontsize=10)
    ax2.set_ylim(0.65, 1.05)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='lower left', fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
    plt.savefig(output_plot_path, dpi=300)
    print(f"[SUCCESS] Performance plots generated and saved to: {output_plot_path}")
    return True


if __name__ == "__main__":
    generate_performance_plots()
