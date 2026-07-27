# Baloo PID Policy — 6-Joint Individual PID Controllers

An evaluation harness for the Baloo robot's box-lifting task, driven by six
**independent** PID controllers — one per joint, each with its own tuned gains
and correction ceiling — replacing the previous single shared PID. It mirrors
the histogram outputs of `evaluate_rl_1000.py` exactly, so results are directly
comparable against the RL baseline.

## Overview

The script runs a large batch of simulated lifting episodes under two
conditions and reports how often the policy succeeds, how much chamber pressure
each joint uses, and how the PID activation relates to outcomes.

Each trial ends on one of three outcomes:

| Outcome | Condition |
| --- | --- |
| **Success** | Box `z`-position rises ≥ 0.5 m above its initial height |
| **Tip** | Box rotation `rot[2,2]` drops below `cos(80°)` |
| **Slip** | `max_steps` (480) elapse without success or tip |

## Requirements

- The Baloo workspace and MuJoCo simulation checked out at the hardcoded paths
  under `/home/cameronc/baloo_ws/...` (the script appends these to `sys.path`
  and loads three MuJoCo plugin `.so` libraries).
- Python packages: `numpy`, `matplotlib`, `pandas`, `tqdm`, `mujoco`, and the
  project's own `baloo_gym` / `baloo_mujoco_sim`.

Because the sim source, plugin binaries, and default output directory are all
hardcoded, this is designed to run on that specific machine (or one with an
identical layout).

## Usage

```bash
python baloo_pid_policy.py [--num_trials N] [--num_detail D] [--output_dir DIR]
```

### Arguments

| Flag | Default | Description |
| --- | --- | --- |
| `--num_trials` | `1000` | Number of trials **per condition** |
| `--num_detail` | `10` | Trials run with the on-screen viewer + full per-timestep pressure recording |
| `--output_dir` | `/home/cameronc/evaluation_results` | Where all files are written |

### Running fully headless

The first `--num_detail` trials of each condition open the MuJoCo viewer
(`render_mode='human'`). Set it to `0` to skip the viewer entirely and run
every trial headless:

```bash
python baloo_pid_policy.py --num_trials 1000 --num_detail 0
```

On a machine with no display, force the non-interactive plotting backend as a
safeguard:

```bash
MPLBACKEND=Agg python baloo_pid_policy.py --num_trials 1000 --num_detail 0
```

> **Note:** `main` always runs **both** conditions, each with `num_trials`
> episodes. So `--num_trials 1000` means 1000 no-perturbation trials **plus**
> 1000 with 100 N perturbations = 2000 trials total.

## Conditions

Both conditions run on every invocation:

1. **No perturbations** — suffix `nopert`
2. **100 N downward perturbations** — suffix `pert100`

## Execution phases

Each condition runs in two phases:

- **Phase 1 (detail)** — the first `num_detail` trials run with the viewer and
  record pressure at every timestep. Used to generate validation histograms and
  per-trial plots.
- **Phase 2 (headless)** — the remaining trials run in parallel across
  `os.cpu_count()` worker processes, each with its own MuJoCo sim. This will
  saturate all cores; the worker count is not exposed as a flag.

## PID configuration

Each joint carries its own `kp`, `ki`, `kd`, activation `threshold`, and
`correction_max` ceiling, defined in `PER_JOINT_PID_PARAMS`:

| Joint | kp | ki | kd | threshold | correction_max |
| --- | --- | --- | --- | --- | --- |
| `left_j0` | 179.727 | 5.192 | 9.721 | 0.0287 | 150.0 |
| `left_j1` | 32.85 | 9.152 | 7.283 | 0.0996 | 150.0 |
| `left_j2` | 160.805 | 6.513 | 7.776 | 0.0021 | 75.0 |
| `right_j0` | 149.304 | 18.156 | 3.438 | 0.0500 | 150.0 |
| `right_j1` | 151.839 | 19.133 | 1.739 | 0.0339 | 150.0 |
| `right_j2` | 41.285 | 15.548 | 9.606 | 0.002 | 48.0 |

## Output files

Written to `--output_dir`. The `<suffix>` is `nopert` or `pert100`, `<N>` is the
trial count, and `<D>` is the detail count.

| File | Description |
| --- | --- |
| `pid_pressure_histograms_all<N>_<suffix>.png` | 2×3 grid of max chamber pressure per joint, all trials, stacked by outcome |
| `pid_pressure_histograms_first<D>_<suffix>.png` | Same grid for just the detail trials (validation) |
| `per_trial_histograms_<suffix>/trial_<N>_<joint>.png` | One per-timestep pressure histogram per (trial, joint) |
| `pid_max_pressures_<suffix>.npy` | `(num_trials, 6)` array of per-joint max pressures |
| `pid_all_pressures_first<D>_<suffix>.npy` | Per-timestep pressure arrays for detail trials |
| `pid_evaluation_nopert_<N>.csv` / `pid_evaluation_pert100_<N>.csv` | Per-trial outcome metadata |
| `table_nopert.csv` / `table_nopert.png` | PID activation vs. outcomes (no perturbations) |
| `table_pert100.csv` / `table_pert100.png` | PID activation vs. outcomes (100 N perturbations) |
| `index.html` | Gallery linking the outcome tables |

## Console reporting

Alongside the files, each condition prints a **99th-percentile max pressure**
per joint (the value only 1% of trials exceeded) and a final success rate, and
the run ends with a side-by-side success-rate summary for both conditions.
