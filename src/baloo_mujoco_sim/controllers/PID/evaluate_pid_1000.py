#!/usr/bin/env python3
'''
Baloo PID Policy — MuJoCo Statistical Analysis
===============================================
Mirrors the histogram outputs of evaluate_rl_1000.py exactly.

Runs NUM_TRIALS (default 1000) episodes under two conditions:
  Condition 1: No perturbations
  Condition 2: 100 N downward perturbations

Each trial ends when:
  SUCCESS : box z-position rises >= 0.5 m above its initial z
  TIP     : box rotation matrix element rot[2,2] < cos(80 deg)
  SLIP    : max_steps elapsed without either of the above

OUTPUT FILES  (written to --output_dir, once per condition)
-----------
Condition suffix is "nopert" or "pert100".

1. pid_pressure_histograms_all<N>_<suffix>.png
       2x3 grid — one histogram per joint, stacked by outcome,
       data = max pressure per trial across all N trials.

2. pid_pressure_histograms_first<D>_<suffix>.png
       Same layout but for the first D (default 10) detail trials only.

3. per_trial_histograms_<suffix>/trial_<N>_<joint>.png
       One histogram per (trial, joint) for the first D trials.
       Shows every pressure sample recorded at every timestep
       for that joint (all 4 chambers), coloured by trial outcome.

4. pid_max_pressures_<suffix>.npy      — (N, 6) max pressures
5. pid_all_pressures_first<D>_<suffix>.npy — list of D arrays,
       each shape (T_i, 6, 4)

6. pid_evaluation_nopert_<N>.csv / pid_evaluation_pert100_<N>.csv
7. table_nopert.csv / table_nopert.png
8. table_pert100.csv / table_pert100.png
9. index.html — gallery

Requirements:
    pip install mujoco stable-baselines3 matplotlib pandas tqdm
'''

import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

sys.path.append('/home/cameronc/baloo_ws/src/baloo-gym/src')

import baloo_gym.policies.PIDHugger as pid_hugger
pid_hugger.OpenLoopHuggerPolicy.print_pressures = lambda self, *args, **kwargs: None
pid_hugger.OpenLoopHuggerPolicy.save_logs = lambda self, *args, **kwargs: None

import mujoco

def setup_plugins():
    plugins = [
        '/home/cameronc/baloo_ws/src/baloo-mujoco-sim/plugin/build/joint_angle_estimator/libjoint_angle_estimator.so',
        '/home/cameronc/baloo_ws/src/baloo-mujoco-sim/plugin/build/ruckig_actuator/libRuckigActuator.so',
        '/home/cameronc/baloo_ws/src/baloo-mujoco-sim/plugin/build/motion_profile_servo/libMotionProfileServo.so'
    ]
    for plugin_path in plugins:
        try:
            mujoco.mj_loadPluginLibrary(plugin_path)
        except mujoco.FatalError as e:
            if "already registered" in str(e):
                pass
            else:
                raise e

# ---------------------------------------------------------------------------
# Shared constants — identical to evaluate_rl_1000.py
# ---------------------------------------------------------------------------
MAX_STEPS_PER_TRIAL  = 480          # PID episodes are shorter (24 s at 20 Hz)
BOX_RISE_Z_DELTA     = 0.5          # metres — success threshold
TIP_ANGLE_COS        = np.cos(np.radians(80.0))

JOINT_LABELS = [
    "Left J0", "Left J1", "Left J2",
    "Right J0", "Right J1", "Right J2",
]
JOINT_SLUGS = [
    "left_j0", "left_j1", "left_j2",
    "right_j0", "right_j1", "right_j2",
]
COLOURS = [
    "#4C72B0", "#DD8452", "#55A868",
    "#C44E52", "#8172B2", "#937860",
]
OUTCOME_COLOURS = {
    "success": "#2ca02c",
    "tip":     "#d62728",
    "timeout": "#9467bd",
}
OUTCOME_LABELS = {
    "success": "Success",
    "tip":     "Tipped",
    "timeout": "Slip (timeout)",
}

# ---------------------------------------------------------------------------
# Pressure extraction helper
# ---------------------------------------------------------------------------
from baloo_mujoco_sim.utils.baloo_mj_api import get_joint_pressures

def _get_step_pressures(model, data):
    """Return (6, 4) array — one row per joint, 4 chambers each."""
    return np.stack([
        get_joint_pressures(model, data, "left",  0),
        get_joint_pressures(model, data, "left",  1),
        get_joint_pressures(model, data, "left",  2),
        get_joint_pressures(model, data, "right", 0),
        get_joint_pressures(model, data, "right", 1),
        get_joint_pressures(model, data, "right", 2),
    ])

# ---------------------------------------------------------------------------
# Single trial
# ---------------------------------------------------------------------------
def run_single_trial(perturbation_magnitude, render_mode=None,
                     record_all_steps=False, pid_config=None,per_joint_pid_params=None):
    """
    Run one PID episode and return:
        result_dict   — outcome metadata (mass, size, outcome, ...)
        joint_max     — (6,) max pressure per joint across the trial
        all_pressures — (T, 6, 4) if record_all_steps else None
    """
    setup_plugins()

    from baloo_gym.envs.baloo_v9 import BalooV9
    from baloo_gym.policies.PIDHugger import OpenLoopHuggerPolicy
    from baloo_gym.wrappers.object_perturbation_wrapper import ObjectPerturbationWrapper

    env = BalooV9(
        render_mode=render_mode,
        ctrl_timestep=0.05,
        randomize_object_size=True,
        randomize_object_mass=True,
        randomize_object_quat=True,
        camera_name="frontcam",
    )
    env = ObjectPerturbationWrapper(env)
    env.perturbation_magnitude = perturbation_magnitude

    if pid_config is None:
        pid_config = {'kp': 80.0, 'ki': 2.0, 'kd': 1.5,
                      'threshold': 0.01, 'correction_max': 70}
    active_policy = OpenLoopHuggerPolicy(N=50, slip_pid_params=pid_config,per_joint_pid_params=per_joint_pid_params)

    model = env.unwrapped.model
    data  = env.unwrapped.data
    box_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box")

    obs, _ = env.reset()
    active_policy.restart()

    xsize, ysize, zsize = env.unwrapped.object_attr[0:3]
    mass     = env.unwrapped.object_attr[3]
    rotation = env.unwrapped.object_zrotation_val

    joint_max   = np.full(6, -np.inf)
    step_buffer = [] if record_all_steps else None
    outcome     = "timeout"
    initial_z   = None
    step_count  = 0
    pid_activated = False

    while step_count < MAX_STEPS_PER_TRIAL:
        current_z = data.xipos[box_body_id][2]
        if initial_z is None:
            initial_z = current_z

        action, _ = active_policy.predict(obs, exact_box_z=current_z)
        obs, reward, _,_, info = env.step(action)

        if active_policy.pid_was_active:
            pid_activated = True

        # Collect pressures AFTER the step (same convention as RL script)
        step_pressures = _get_step_pressures(model, data)  # (6, 4)
        joint_max = np.maximum(joint_max, step_pressures.max(axis=1))
        if record_all_steps:
            step_buffer.append(step_pressures.copy())

        rot_mat = data.xmat[box_body_id].reshape(3, 3)
        if rot_mat[2, 2] < TIP_ANGLE_COS:
            outcome = "tip"
            break
        elif current_z >= (initial_z + BOX_RISE_Z_DELTA):
            outcome = "success"
            break

        step_count += 1

        if render_mode == 'human':
            env.render()

    env.close()

    slipped = outcome == "timeout"
    tipped  = outcome == "tip"
    success = outcome == "success"

    result = {
        "mass_kg":      mass,
        "box_rotation": rotation,
        "box_xsize":    xsize,
        "box_ysize":    ysize,
        "box_zsize":    zsize,
        "pid_activated": pid_activated,
        "slipped":      slipped,
        "tipped":       tipped,
        "success":      success,
        "outcome":      outcome,
    }

    all_pressures = np.stack(step_buffer, axis=0) if record_all_steps else None
    return result, joint_max, all_pressures


# ---------------------------------------------------------------------------
# Batch helpers for parallel headless runs (no pressure detail)
# ---------------------------------------------------------------------------
def _run_chunk_headless(chunk_ids, perturbation_magnitude):
    results = []
    for _ in chunk_ids:
        res, jmax, _ = run_single_trial(perturbation_magnitude,
                                        render_mode=None,
                                        record_all_steps=False)
        results.append((res, jmax))
    return results


# ---------------------------------------------------------------------------
# Plotting helpers — identical to evaluate_rl_1000.py
# ---------------------------------------------------------------------------
def _annotate_ax_stacked(ax, data, outcomes, xlabel, title, joint_colour, n_bins):
    if len(data) == 0:
        ax.set_title(title, fontsize=10, fontweight="bold")
        return

    outcomes = np.asarray(outcomes)
    outcome_keys = ["success", "tip", "timeout"]
    _, bin_edges = np.histogram(data, bins=n_bins)

    bottoms = np.zeros(n_bins)
    for key in outcome_keys:
        mask = outcomes == key
        if not mask.any():
            continue
        counts, _ = np.histogram(data[mask], bins=bin_edges)
        ax.bar(
            bin_edges[:-1], counts, width=np.diff(bin_edges),
            bottom=bottoms, align="edge",
            color=OUTCOME_COLOURS[key], alpha=0.85,
            label=f"{OUTCOME_LABELS[key]} (n={mask.sum()})",
        )
        bottoms += counts

    ax.axvline(data.mean(),     color="black", linestyle="--",
               linewidth=1.3, label=f"Mean: {data.mean():.1f} kPa")
    ax.axvline(np.median(data), color="white", linestyle=":",
               linewidth=1.3, label=f"Median: {np.median(data):.1f} kPa")

    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel("Count", fontsize=8)
    ax.legend(fontsize=6.5, loc="upper right")
    ax.tick_params(labelsize=7)
    n_success = (outcomes == "success").sum()
    ax.text(0.02, 0.95,
            f"N={len(data)}  |  {n_success/len(data)*100:.0f}% success\n"
            f"sigma={data.std():.1f} kPa",
            transform=ax.transAxes, ha="left", va="top", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))


def plot_summary_histograms(max_results, outcomes, title_suffix, filename, n_bins=30):
    """2x3 grid — one subplot per joint, stacked by outcome."""
    from matplotlib.patches import Patch
    outcomes_arr = np.asarray(outcomes)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(f"Max Chamber Pressure per Joint — {title_suffix}",
                 fontsize=13, fontweight="bold")

    for j, (ax, label, colour) in enumerate(
            zip(axes.flat, JOINT_LABELS, COLOURS)):
        _annotate_ax_stacked(
            ax, max_results[:, j], outcomes_arr,
            xlabel="Max Pressure (kPa)",
            title=label,
            joint_colour=colour,
            n_bins=n_bins,
        )

    for row, arm in enumerate(["LEFT ARM", "RIGHT ARM"]):
        fig.text(0.005, 0.75 - row * 0.5, arm,
                 va="center", ha="left", fontsize=9,
                 fontweight="bold", rotation=90, color="#333333")

    legend_patches = [
        Patch(facecolor=OUTCOME_COLOURS[k], alpha=0.85,
              label=OUTCOME_LABELS[k])
        for k in ["success", "tip", "timeout"]
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=3,
               fontsize=9, framealpha=0.8, title="Outcome colour key",
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {filename}")


def plot_per_trial_histograms(detail_data, outcomes, out_dir, suffix, n_bins=30):
    """
    One histogram PNG per (trial, joint) for the detail trials.
    detail_data : list of arrays, each shape (T_i, 6, 4)
    outcomes    : list of strings, one per trial
    """
    os.makedirs(out_dir, exist_ok=True)
    total = len(detail_data) * len(JOINT_LABELS)
    done  = 0

    for trial_idx, (trial_array, outcome) in enumerate(
            zip(detail_data, outcomes)):
        trial_num   = trial_idx + 1
        outcome_col = OUTCOME_COLOURS[outcome]
        outcome_lbl = OUTCOME_LABELS[outcome]

        for j, (label, slug, colour) in enumerate(
                zip(JOINT_LABELS, JOINT_SLUGS, COLOURS)):

            samples = trial_array[:, j, :].ravel()

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(samples, bins=n_bins, color=colour,
                    edgecolor="white", linewidth=0.4, alpha=0.85)
            ax.axvline(samples.mean(),     color="black", linestyle="--",
                       linewidth=1.3, label=f"Mean: {samples.mean():.1f} kPa")
            ax.axvline(np.median(samples), color="gray",  linestyle=":",
                       linewidth=1.3, label=f"Median: {np.median(samples):.1f} kPa")
            ax.set_xlabel("Pressure (kPa)", fontsize=8)
            ax.set_ylabel("Count", fontsize=8)
            ax.legend(fontsize=7)
            ax.tick_params(labelsize=7)
            ax.text(0.97, 0.95,
                    f"sigma={samples.std():.1f} kPa\nN={len(samples)}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
            ax.set_title(
                f"Trial {trial_num:02d}  -  {label}\n"
                f"({len(trial_array)} steps x 4 chambers = {len(samples)} samples)",
                fontsize=10, fontweight="bold",
            )
            ax.text(0.02, 0.95, outcome_lbl,
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=9, fontweight="bold", color="white",
                    bbox=dict(boxstyle="round,pad=0.35",
                              fc=outcome_col, ec="none", alpha=0.9))

            fig.suptitle(f"Per-Timestep Pressure Distribution  |  PID  |  {suffix}",
                         fontsize=9, color="#555555")
            plt.tight_layout()
            plt.savefig(
                os.path.join(out_dir, f"trial_{trial_num:02d}_{slug}.png"),
                dpi=130, bbox_inches="tight")
            plt.close(fig)

            done += 1
            if done % 10 == 0 or done == total:
                print(f"    {done}/{total} per-trial plots written...")

    print(f"  All {total} per-trial histograms saved -> {out_dir}/")


# ---------------------------------------------------------------------------
# Outcome summary table helpers (unchanged from original)
# ---------------------------------------------------------------------------
def compute_outcome_table(df):
    def _row(sub):
        s = int(sub['success'].sum())
        sl = int(sub['slipped'].sum())
        ti = int(sub['tipped'].sum())
        tot = len(sub)
        rate = f"{s/tot*100:.1f}%" if tot > 0 else "0.0%"
        return [s, sl, ti, tot, rate]

    df_no  = df[df['pid_activated'] == False]
    df_pid = df[df['pid_activated'] == True]
    total  = df

    data = [_row(df_no), _row(df_pid), _row(total)]
    return data


def save_table_image(data, columns, rows, title, filename):
    fig, ax = plt.subplots(figsize=(13.5, 2.2))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=data, colLabels=columns,
                     rowLabels=rows, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.4, 2.0)
    num_rows = len(data)
    for (row, col), cell in table.get_celld().items():
        if row == 0 or col == -1:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#2b5c8f')
        elif row == num_rows:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#e8eff7')
        else:
            cell.set_facecolor('#f4f6f9' if row % 2 == 0 else 'white')
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# Per-condition runner
# ---------------------------------------------------------------------------
def run_condition(perturbation_magnitude, suffix, num_trials,
                  num_detail, output_dir):
    """
    Run one full condition (no-pert or pert100).

    Phase 1: num_detail viewed trials with pressure detail.
    Plots validation histograms.
    Phase 2: remaining trials headless in parallel.
    Saves all outputs.

    Returns df (DataFrame of results).
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    label = "No Perturbations" if perturbation_magnitude == 0 else f"{perturbation_magnitude}N Perturbations"
    print(f"\n{'='*60}")
    print(f"CONDITION: {label}  ({num_trials} trials)")
    print(f"{'='*60}")

    max_results = np.zeros((num_trials, 6))
    all_results = []
    detail_data     = []
    detail_outcomes = []

    # ---- Phase 1: viewed + detail trials ----------------------------------
    print(f"\nPhase 1: Running first {num_detail} trials WITH viewer and pressure recording...")
    for i in range(num_detail):
        print(f"  Trial {i+1}/{num_detail} starting...")
        res, jmax, all_pres = run_single_trial(
            perturbation_magnitude,
            render_mode='human',
            record_all_steps=True,
        )
        max_results[i] = jmax
        all_results.append(res)
        detail_data.append(all_pres)
        detail_outcomes.append(res['outcome'])
        print(f"  Trial {i+1}/{num_detail} done. "
              f"Outcome: {res['outcome'].upper()}  "
              f"Steps: {len(all_pres)}  "
              f"Max pressures: {jmax.round(1)}")

    # Switch to non-interactive backend before any plt calls
    matplotlib.use("Agg")

    # Save detail pressure data immediately (crash protection)
    all_pres_file = os.path.join(
        output_dir, f"pid_all_pressures_first{num_detail}_{suffix}.npy")
    np.save(all_pres_file,
            np.array(detail_data, dtype=object), allow_pickle=True)
    print(f"\nPer-timestep data saved -> {all_pres_file}")

    # Validation plots from Phase 1 data
    print(f"\nGenerating validation histograms (first {num_detail} trials)...")
    plot_summary_histograms(
        max_results[:num_detail],
        outcomes=detail_outcomes,
        title_suffix=f"PID  |  {suffix}  --  first {num_detail} trials",
        filename=os.path.join(
            output_dir,
            f"pid_pressure_histograms_first{num_detail}_{suffix}.png"),
        n_bins=5,
    )

    per_trial_dir = os.path.join(output_dir, f"per_trial_histograms_{suffix}")
    print(f"Generating {num_detail * len(JOINT_LABELS)} per-trial per-joint histograms...")
    plot_per_trial_histograms(
        detail_data, detail_outcomes, per_trial_dir, suffix, n_bins=30)

    print(f"\nValidation plots written -- check {output_dir} before continuing.")
    print("Starting headless Phase 2 now...\n")

    # ---- Phase 2: headless parallel trials --------------------------------
    remaining = num_trials - num_detail
    num_cores = os.cpu_count() or 4
    remaining_ids = list(range(num_detail, num_trials))
    chunk_size = max(1, len(remaining_ids) // num_cores)
    job_chunks = [remaining_ids[i:i + chunk_size]
                  for i in range(0, len(remaining_ids), chunk_size)]

    print(f"Phase 2: Running remaining {remaining} trials headlessly on {num_cores} cores...")
    headless_results = []
    headless_jmaxes  = []
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = {
            executor.submit(_run_chunk_headless, chunk, perturbation_magnitude): chunk
            for chunk in job_chunks
        }
        for future in tqdm(as_completed(futures), total=len(job_chunks),
                           desc=f"{label} headless"):
            for res, jmax in future.result():
                headless_results.append(res)
                headless_jmaxes.append(jmax)

    # Fill in Phase 2 rows
    for i, jmax in enumerate(headless_jmaxes):
        max_results[num_detail + i] = jmax
    all_results.extend(headless_results)

    all_outcomes = detail_outcomes + [r['outcome'] for r in headless_results]

    # Save complete max-pressure array
    max_pres_file = os.path.join(
        output_dir, f"pid_max_pressures_{suffix}.npy")
    np.save(max_pres_file, max_results)
    print(f"\nMax-pressure data saved -> {max_pres_file}")

    # All-trials summary histogram
    print(f"\nGenerating summary histograms (all {num_trials} trials)...")
    plot_summary_histograms(
        max_results,
        outcomes=all_outcomes,
        title_suffix=f"PID  |  {suffix}  --  all {num_trials} trials",
        filename=os.path.join(
            output_dir,
            f"pid_pressure_histograms_all{num_trials}_{suffix}.png"),
        n_bins=30,
    )

    df = pd.DataFrame(all_results)
    n_success = sum(o == "success" for o in all_outcomes)
    print(f"\nFinal results ({label}): "
          f"{n_success}/{num_trials} successes "
          f"({n_success/num_trials*100:.1f}%)")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_trials",  type=int, default=1000)
    parser.add_argument("--num_detail",  type=int, default=10,
                        help="Trials to run with viewer + full pressure recording")
    parser.add_argument("--output_dir",  type=str,
                        default="/home/cameronc/evaluation_results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Episode ends on SUCCESS (box rises >= {BOX_RISE_Z_DELTA} m above initial z),")
    print(f"TIP (rot_mat[2,2] < cos(80 deg) = {TIP_ANGLE_COS:.3f}),")
    print(f"or SLIP (timeout after {MAX_STEPS_PER_TRIAL} steps).")

    # ---- Condition 1: No Perturbations ------------------------------------
    df_nopert = run_condition(
        perturbation_magnitude=0,
        suffix="nopert",
        num_trials=args.num_trials,
        num_detail=args.num_detail,
        output_dir=args.output_dir,
    )

    # ---- Condition 2: 100N Perturbations ----------------------------------
    df_pert = run_condition(
        perturbation_magnitude=100,
        suffix="pert100",
        num_trials=args.num_trials,
        num_detail=args.num_detail,
        output_dir=args.output_dir,
    )

    # ---- Save CSVs and outcome tables -------------------------------------
    df_nopert.to_csv(
        os.path.join(args.output_dir,
                     f"pid_evaluation_nopert_{args.num_trials}.csv"),
        index=False)
    df_pert.to_csv(
        os.path.join(args.output_dir,
                     f"pid_evaluation_pert100_{args.num_trials}.csv"),
        index=False)

    columns = ['Successes', 'Slips', 'Tips', 'Total', 'Success Rate (%)']
    rows    = ['PID Not Activated', 'PID Activated', 'Total']

    data_nopert = compute_outcome_table(df_nopert)
    pd.DataFrame(data_nopert, columns=columns, index=rows).to_csv(
        os.path.join(args.output_dir, "table_nopert.csv"))
    save_table_image(
        data_nopert, columns, rows,
        "PID Activation vs. Outcomes (No Perturbations)",
        os.path.join(args.output_dir, "table_nopert.png"))

    data_pert = compute_outcome_table(df_pert)
    pd.DataFrame(data_pert, columns=columns, index=rows).to_csv(
        os.path.join(args.output_dir, "table_pert100.csv"))
    save_table_image(
        data_pert, columns, rows,
        "PID Activation vs. Outcomes (100N Perturbations)",
        os.path.join(args.output_dir, "table_pert100.png"))

    # ---- HTML gallery -----------------------------------------------------
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>PID Evaluation Results</title>
    <style>
        body { font-family: 'Segoe UI', sans-serif; background:#f4f6f9;
               color:#333; margin:0; padding:20px; }
        h1   { text-align:center; color:#1e3d59; margin-bottom:30px; }
        .grid-container { display:grid;
                          grid-template-columns:repeat(auto-fit,minmax(500px,1fr));
                          gap:30px; max-width:1200px; margin:0 auto; }
        .card { background:#fff; border-radius:8px;
                box-shadow:0 4px 6px rgba(0,0,0,0.1);
                overflow:hidden; padding:20px; }
        .card img { width:100%; height:auto; display:block; margin-bottom:15px; }
        .card-body { text-align:center; }
        .card-title { font-weight:bold; font-size:1.2em;
                      margin-bottom:5px; color:#1e3d59; }
        .card-meta  { font-size:0.9em; color:#777; }
    </style>
</head>
<body>
    <h1>PID Controller 1000-Trial Evaluation Outcomes</h1>
    <div class="grid-container">
        <div class="card">
            <img src="table_nopert.png" alt="No Perturbations Table">
            <div class="card-body">
                <div class="card-title">No Perturbations (0N)</div>
                <div class="card-meta">
                    <a href="table_nopert.csv">table_nopert.csv</a>
                </div>
            </div>
        </div>
        <div class="card">
            <img src="table_pert100.png" alt="100N Perturbations Table">
            <div class="card-body">
                <div class="card-title">With 100N Perturbations</div>
                <div class="card-meta">
                    <a href="table_pert100.csv">table_pert100.csv</a>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""
    with open(os.path.join(args.output_dir, "index.html"), "w") as f:
        f.write(html_content)

    print("\n=== ALL EVALUATIONS COMPLETE ===")
    print(f"No Perturbations Success Rate: "
          f"{df_nopert['success'].mean()*100:.2f}%")
    print(f"100N Perturbations Success Rate: "
          f"{df_pert['success'].mean()*100:.2f}%")
    print(f"\nAll data and plots saved to '{args.output_dir}'")
