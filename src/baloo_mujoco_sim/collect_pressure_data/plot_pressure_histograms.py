"""
plot_pressure_histograms.py

Reads pressure_data_*.jsonl files and plots per-joint max pressure histograms
for all controllers overlaid, with 95th/99th percentile lines marked.

Usage:
    python plot_pressure_histograms.py
    python plot_pressure_histograms.py --files pressure_data_mpc.jsonl pressure_data_rl.jsonl
    python plot_pressure_histograms.py --out pressure_histograms.png
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

JOINT_LABELS = [
    "left_j0", "left_j1", "left_j2",
    "right_j0", "right_j1", "right_j2",
]

JOINT_DISPLAY = [
    "Left J0", "Left J1", "Left J2",
    "Right J0", "Right J1", "Right J2",
]

CONTROLLER_COLORS = {
    "open_loop":  "#FF9800",  # orange
    "Trial_25":   "#4CAF50",  # green
    "Trial_73":   "#9C27B0",  # purple
    "Curtis":     "#2196F3",  # blue
}

CONTROLLER_LABELS = {
    "open_loop":  "Open Loop",
    "Trial_25":   "Trial 25",
    "Trial_73":   "Trial 73",
    "Curtis":     "Curtis",
}

DEFAULT_FILES = {
    "open_loop":  "/home/randonsandall/baloo-gym/pressure/pressure_data_open_loop.jsonl",
    "Trial_25":   "/home/randonsandall/baloo-gym/pressure/pressure_data_Trial_25.jsonl",
    "Trial_73":   "/home/randonsandall/baloo-gym/pressure/pressure_data_Trial_73.jsonl",
    "Curtis":     "/home/randonsandall/baloo-gym/pressure/pressure_data_Curtis.jsonl",
}


def load_jsonl(path):
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def extract_joint_maxes(results, joint, metric="lift_mean"):
    """
    Pull per-trial pressure for a given joint.
    metric: "lift_mean" = mean pressure during lift phase (default)
            "joint_maxes" = max pressure over whole episode
    Filters out trials where all joint_maxes are zero (invalid/early-termination trials).
    """
    filtered = []
    for r in results:
        if metric not in r or joint not in r[metric]:
            continue
        # skip invalid trials — either all zeros or avg max below 200 (box flung immediately)
        jm = r.get("joint_maxes", {})
        if jm:
            avg = sum(jm.values()) / len(jm)
            if avg < 200:
                continue
        filtered.append(r[metric][joint])
    return filtered


def plot_histograms(data, out_file=None, metric="lift_mean"):
    """
    data: dict of {controller_name: list of trial dicts}
    metric: "lift_mean" or "joint_maxes"
    """
    n_joints = len(JOINT_LABELS)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()



    for j_idx, (joint, display) in enumerate(zip(JOINT_LABELS, JOINT_DISPLAY)):
        ax = axes[j_idx]

        # gather all vals across controllers to compute shared bins
        all_vals = []
        for ctrl, results in data.items():
            all_vals += extract_joint_maxes(results, joint, metric=metric)

        if not all_vals:
            continue

        lo = min(all_vals)
        hi = max(all_vals)
        bins = np.linspace(lo, hi, 30)

        for ctrl, results in data.items():
            vals = extract_joint_maxes(results, joint, metric=metric)
            if not vals:
                continue

            color = CONTROLLER_COLORS.get(ctrl, "#999999")
            label = CONTROLLER_LABELS.get(ctrl, ctrl)
            p99   = np.percentile(vals, 99)

            ax.hist(vals, bins=bins, alpha=0.5, color=color, label=label, density=False)
            ax.axvline(p99, color=color, linestyle='--', linewidth=2,
                       label=f"{label} p99={p99:.0f}")

        margin = (hi - lo) * 0.02
        ax.set_title(display, fontweight='bold')
        ax.set_xlabel("Max Pressure (mbar)")
        ax.set_ylabel("Trial Count")
        ax.set_xlim(lo - margin, hi + margin)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if out_file:
        plt.savefig(out_file, dpi=150, bbox_inches='tight')
        print(f"Saved → {out_file}")
    else:
        plt.show()


def print_p95_table(data, metric="lift_mean"):
    print(f"\n{'Joint':<12}", end="")
    for ctrl in data:
        print(f"  {CONTROLLER_LABELS[ctrl]:>12} p95", end="")
    print()
    print("-" * (12 + 18 * len(data)))

    for joint, display in zip(JOINT_LABELS, JOINT_DISPLAY):
        print(f"{display:<12}", end="")
        for ctrl, results in data.items():
            vals = extract_joint_maxes(results, joint)
            p95  = np.percentile(vals, 95) if vals else float('nan')
            print(f"  {p95:>16.1f}", end="")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", default=None,
                        help="JSONL files to plot (default: auto-detect all four)")
    parser.add_argument("--out", type=str, default=None,
                        help="Save plot to file instead of showing it (e.g. plot.png)")
    parser.add_argument("--metric", choices=["lift_mean", "joint_maxes"], default="joint_maxes",
                        help="Which pressure metric to plot (default: joint_maxes)")
    args = parser.parse_args()

    # load whichever files exist
    data = {}
    if args.files:
        for path in args.files:
            # infer controller name from filename
            ctrl = None
            for key in DEFAULT_FILES:
                if key in path:
                    ctrl = key
                    break
            ctrl = ctrl or path.replace(".jsonl", "").replace("pressure_data_", "")
            print(f"Loading {path} ({ctrl})...")
            data[ctrl] = load_jsonl(path)
    else:
        for ctrl, path in DEFAULT_FILES.items():
            try:
                data[ctrl] = load_jsonl(path)
                print(f"Loaded {path}: {len(data[ctrl])} trials")
            except FileNotFoundError:
                print(f"Skipping {path} (not found)")

    if not data:
        print("No data files found. Run collect_pressure_trial_73.py / collect_pressure_rl_models.py first.")
        exit(1)

    print_p95_table(data, metric=args.metric)
    plot_histograms(data, out_file=args.out, metric=args.metric)
