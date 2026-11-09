"""Generate traceable Partner 2 simulation-performance figures."""

import csv
import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = PROJECT_ROOT / "analytics" / "partner_2_analysis" / "partner_2_analysis_metrics.csv"
OUTPUT_DIR = PROJECT_ROOT / "analytics" / "report_figures"
INCONSISTENT_TIMESTAMP = "1793982498.36"
DEFECT_THRESHOLD = 0.88

CATEGORY_LABELS = {
    "normal_packaging_record": "Normal packaging",
    "defect_rejection_record": "Defect rejection",
    "lost_grip_recovery_record": "Lost-grip recovery (fault source unverified)",
    "multi_cycle_controller_sequence": "Multi-cycle controller sequence",
}
CATEGORY_COLORS = {
    "normal_packaging_record": "#287C74",
    "defect_rejection_record": "#C17B25",
    "lost_grip_recovery_record": "#A84A4A",
    "multi_cycle_controller_sequence": "#4C6F9C",
}


def load_records():
    required_columns = {
        "SourceFilename",
        "OriginalRowNumber",
        "Timestamp",
        "CycleID",
        "Variety",
        "AngleDeg",
        "Solidity",
        "GripForceN",
        "CycleTimeSec",
        "CaseType",
        "Status",
        "AnalysisCategory",
        "EvidenceStatus",
        "GraphUse",
        "EvidenceNotes",
    }
    with INPUT_CSV.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        missing = required_columns.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Analysis CSV is missing columns: {sorted(missing)}")
        records = list(reader)
    if not records:
        raise ValueError(f"No records found in {INPUT_CSV}")
    return records


def apply_report_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 15,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.edgecolor": "#59636E",
            "axes.linewidth": 0.8,
            "grid.color": "#DCE1E5",
            "grid.linewidth": 0.7,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def save_cycle_times(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[record["AnalysisCategory"]].append(record)

    categories = [
        "normal_packaging_record",
        "defect_rejection_record",
        "lost_grip_recovery_record",
        "multi_cycle_controller_sequence",
    ]
    present = [category for category in categories if grouped[category]]
    values = [float(record["CycleTimeSec"]) for record in records]
    low, high = min(values), max(values)
    margin = max((high - low) * 0.08, 0.15)

    figure, axis = plt.subplots(figsize=(11.5, 6.2), constrained_layout=True)
    unverified_point = None
    for y_position, category in enumerate(present):
        category_records = grouped[category]
        offsets = [0.0] if len(category_records) == 1 else [
            -0.22 + 0.44 * index / (len(category_records) - 1)
            for index in range(len(category_records))
        ]
        regular_points = []
        for record, offset in zip(category_records, offsets):
            point = (float(record["CycleTimeSec"]), y_position + offset)
            if record["Timestamp"] == INCONSISTENT_TIMESTAMP:
                unverified_point = point
            else:
                regular_points.append(point)

        if regular_points:
            axis.scatter(
                [point[0] for point in regular_points],
                [point[1] for point in regular_points],
                s=43,
                alpha=0.82,
                color=CATEGORY_COLORS[category],
                edgecolors="white",
                linewidths=0.55,
                label=f"{CATEGORY_LABELS[category]} (n={len(category_records)})",
                zorder=3,
            )

    if unverified_point is not None:
        axis.scatter(
            [unverified_point[0]],
            [unverified_point[1]],
            marker="X",
            s=105,
            color="#343A40",
            edgecolors="white",
            linewidths=0.7,
            zorder=5,
        )
        axis.annotate(
            "Unverified historical record",
            xy=unverified_point,
            xytext=(unverified_point[0] - 1.35, unverified_point[1] + 0.28),
            fontsize=8,
            color="#343A40",
            ha="center",
            va="bottom",
            arrowprops={"arrowstyle": "->", "color": "#343A40", "linewidth": 0.9},
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
        )

    axis.set_yticks(
        range(len(present)),
        [f"{CATEGORY_LABELS[key]} (n={len(grouped[key])})" for key in present],
    )
    axis.set_ylim(-0.45, len(present) - 0.55)
    axis.set_xlim(low - margin, high + margin)
    axis.set_xlabel("Recorded cycle time (Webots simulation seconds)")
    axis.set_title("Simulation Cycle Times", loc="left", pad=15, weight="bold")
    axis.text(
        0,
        1.015,
        "One marker per source row; CycleID is not used as a run identifier.",
        transform=axis.transAxes,
        color="#4E5963",
        fontsize=9,
        va="bottom",
    )
    axis.grid(axis="x", zorder=0)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    path = OUTPUT_DIR / "simulation_cycle_times.png"
    figure.savefig(path, dpi=320, bbox_inches="tight")
    plt.close(figure)
    return path


def save_solidity(records):
    normal = [record for record in records if record["AnalysisCategory"] == "normal_packaging_record"]
    defects = [
        record
        for record in records
        if record["AnalysisCategory"] == "defect_rejection_record"
        and record["Timestamp"] != INCONSISTENT_TIMESTAMP
        and float(record["Solidity"]) < DEFECT_THRESHOLD
    ]
    other = [
        record
        for record in records
        if record["AnalysisCategory"] in {
            "lost_grip_recovery_record",
            "multi_cycle_controller_sequence",
        }
    ]
    excluded = [record for record in records if record["Timestamp"] == INCONSISTENT_TIMESTAMP]
    groups = [
        ("Accepted packaging records", normal, CATEGORY_COLORS["normal_packaging_record"], "o"),
        (
            "Defect-rejection records below threshold\n(outcomes not independently verified)",
            defects,
            CATEGORY_COLORS["defect_rejection_record"],
            "D",
        ),
        ("Other controller sequences", other, CATEGORY_COLORS["multi_cycle_controller_sequence"], "s"),
        ("UNVERIFIED reject; excluded\nfrom defect examples", excluded, "#343A40", "X"),
    ]

    all_values = [float(record["Solidity"]) for _, group, _, _ in groups for record in group]
    low, high = min(all_values), max(all_values)
    margin = max((high - low) * 0.09, 0.025)
    x_min = max(0.0, min(low - margin, DEFECT_THRESHOLD - 0.04))
    x_max = min(1.05, max(high + margin, DEFECT_THRESHOLD + 0.04))

    figure, axis = plt.subplots(figsize=(12, 7.0), constrained_layout=True)
    y_ticks = []
    y_labels = []
    for y_position, (label, group, color, marker) in enumerate(groups):
        y_ticks.append(y_position)
        y_labels.append(f"{label} (n={len(group)})")
        offsets = [0.0] if len(group) == 1 else [
            -0.2 + 0.4 * index / (len(group) - 1) for index in range(len(group))
        ]
        if group:
            axis.scatter(
                [float(record["Solidity"]) for record in group],
                [y_position + offset for offset in offsets],
                s=48,
                alpha=0.84,
                color=color if marker != "X" else "none",
                edgecolors=color,
                marker=marker,
                linewidths=1.4 if marker == "X" else 0.7,
                zorder=3,
            )

    axis.axvspan(x_min, DEFECT_THRESHOLD, color="#A84A4A", alpha=0.07, zorder=0)
    axis.axvline(
        DEFECT_THRESHOLD,
        color="#A33F3F",
        linestyle="--",
        linewidth=1.6,
        label="Defect threshold (0.88)",
        zorder=2,
    )
    axis.set_yticks(y_ticks, y_labels)
    axis.set_ylim(-0.48, len(groups) - 0.52)
    axis.set_xlim(x_min, x_max)
    axis.set_xlabel("Recorded vision-system solidity (contour area / convex-hull area)")
    axis.set_title("Chocolate Solidity", loc="left", pad=28, weight="bold")
    axis.text(
        0,
        1.015,
        "Vision-system outputs only; not independently verified detection accuracy.",
        transform=axis.transAxes,
        color="#4E5963",
        fontsize=9,
        va="bottom",
    )
    axis.grid(axis="x", zorder=0)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.text(
        DEFECT_THRESHOLD + 0.006,
        0.06,
        "0.88 threshold",
        color="#A33F3F",
        fontsize=8,
        ha="left",
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.5},
    )
    path = OUTPUT_DIR / "chocolate_solidity.png"
    figure.savefig(path, dpi=320, bbox_inches="tight")
    plt.close(figure)
    return path


def save_outcome_counts(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[record["AnalysisCategory"]].append(record)

    consistent_defects = [
        record
        for record in grouped["defect_rejection_record"]
        if record["Timestamp"] != INCONSISTENT_TIMESTAMP
        and float(record["Solidity"]) < DEFECT_THRESHOLD
    ]
    inconsistent_defects = [record for record in grouped["defect_rejection_record"] if record["Timestamp"] == INCONSISTENT_TIMESTAMP]
    bars = [
        ("Normal packaging records\n(controller-reported; placement not independently sensed)", len(grouped["normal_packaging_record"]), CATEGORY_COLORS["normal_packaging_record"]),
        ("Defect-rejection records\n(solidity < 0.88; outcome unverified)", len(consistent_defects), CATEGORY_COLORS["defect_rejection_record"]),
        ("Defect-rejection record\n(solidity 1.0; UNVERIFIED)", len(inconsistent_defects), "#343A40"),
        ("Lost-grip recovery records\n(fault source unverified)", len(grouped["lost_grip_recovery_record"]), CATEGORY_COLORS["lost_grip_recovery_record"]),
        ("Multi-cycle controller-sequence records\n(not placement verification)", len(grouped["multi_cycle_controller_sequence"]), CATEGORY_COLORS["multi_cycle_controller_sequence"]),
    ]

    labels = [bar[0] for bar in bars]
    counts = [bar[1] for bar in bars]
    colors = [bar[2] for bar in bars]
    figure, axis = plt.subplots(figsize=(12.0, 7.1), constrained_layout=True)
    positions = list(range(len(bars)))
    axis.barh(positions, counts, color=colors, height=0.62, alpha=0.9)
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlim(0, max(counts, default=0) + max(2, max(counts, default=0) * 0.16))
    axis.set_xlabel("Recorded CSV rows (count)")
    axis.set_title("Recorded Controller Outcomes", loc="left", pad=28, weight="bold")
    axis.text(
        0,
        1.015,
        "Counts describe log records only—not manufacturing success, yield, or defect rates.",
        transform=axis.transAxes,
        color="#4E5963",
        fontsize=9,
        va="bottom",
    )
    axis.grid(axis="x", zorder=0)
    axis.set_axisbelow(True)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)
    for position, count in zip(positions, counts):
        axis.text(count + 0.18, position, str(count), va="center", ha="left", weight="bold")
    path = OUTPUT_DIR / "recorded_controller_outcomes.png"
    figure.savefig(path, dpi=320, bbox_inches="tight")
    plt.close(figure)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cycle-times-only",
        action="store_true",
        help="Regenerate only simulation_cycle_times.png.",
    )
    args = parser.parse_args(argv)

    if not INPUT_CSV.is_file():
        raise FileNotFoundError(f"Analysis dataset not found: {INPUT_CSV}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    apply_report_style()
    records = load_records()
    if args.cycle_times_only:
        paths = [save_cycle_times(records)]
    else:
        paths = [
            save_cycle_times(records),
            save_solidity(records),
            save_outcome_counts(records),
        ]
    for path in paths:
        print(f"[CREATED] {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()