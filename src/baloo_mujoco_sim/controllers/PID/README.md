# Baloo 6-Joint PID Optimiser

An [Optuna](https://optuna.org/)-driven hyperparameter search that tunes PID slip-correction gains for all six joints of the Baloo hugger robot (left/right J0, J1, J2) independently, using a simulation environment as the evaluation loop.

The optimiser searches over **kp, ki, kd** for each joint (18 parameters total), evaluates each candidate across many simulated pick-up episodes, and maximises the success rate. Results are persisted to a resumable SQLite database and written out as a JSON file of the best per-joint gains.

---

## What this optimiser does

- Tunes 3 gains (`kp`, `ki`, `kd`) per joint × 6 joints = **18 parameters**.
- Holds `correction_max` **fixed** per joint (a safety ceiling, never tuned).
- Runs `n_eval` simulated episodes per candidate and returns the fraction that succeed.
- Uses a TPE sampler and a median pruner to abandon clearly weak candidates early.
- **Persists every completed trial to SQLite**, so a crash or `Ctrl+C` never loses work — you just re-run the same command to resume.
- Writes the best gains found so far to a JSON checkpoint after every trial.

> **Note on thresholds:** this optimiser does *not* tune the per-joint slip `threshold`. Thresholds are treated as hand-set constants that you maintain separately in your controller. If you need thresholds tuned, see [Optional: tuning threshold too](#optional-tuning-threshold-too).

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.8+ | The reference setup uses Python 3.8. |
| A working Baloo simulation environment | Provides `run_single_trial(...)`. See [The evaluation dependency](#the-evaluation-dependency). |
| `optuna` | The optimisation library. |
| `numpy` | Used throughout the sim/eval code. |
| ROS + `baloo-gym` (for the sim) | Only needed because `run_single_trial` drives the simulated robot. Not needed to read results back. |

Install the Python packages:

```bash
pip install optuna numpy
```

---

## The evaluation dependency

This script is only the **search driver**. It does not contain the physics — it imports a function that runs one simulated episode and reports whether the robot succeeded:

```python
from evaluate_pid_1000 import run_single_trial
```

`run_single_trial(...)` must accept these keyword arguments and return a `(result, _, _)` tuple where `result['success']` is truthy on success:

```python
result, _, _ = run_single_trial(
    perturbation_magnitude=perturbation,   # float, external disturbance in Newtons
    render_mode=None,                      # None = headless
    record_all_steps=False,                # skip per-step logging for speed
    per_joint_pid_params=per_joint_config, # nested dict, see below
)
```

The `per_joint_config` it receives looks like:

```python
{
    'left_j0':  {'kp': 97.4, 'ki': 3.1, 'kd': 5.0, 'correction_max': 150.0},
    'left_j1':  {'kp': ...,  'ki': ..., 'kd': ..., 'correction_max': 150.0},
    'left_j2':  {'kp': ...,  'ki': ..., 'kd': ..., 'correction_max': 75.0},
    'right_j0': {'kp': ...,  'ki': ..., 'kd': ..., 'correction_max': 150.0},
    'right_j1': {'kp': ...,  'ki': ..., 'kd': ..., 'correction_max': 150.0},
    'right_j2': {'kp': ...,  'ki': ..., 'kd': ..., 'correction_max': 48.0},
}
```

If your simulation exposes its single-episode runner under a different name or path, adjust the import and the `sys.path.append(...)` lines near the top of the script to match your workspace layout.

---

## The full script

Save this as `optimize_pid_6joint.py`.

```python
#!/usr/bin/env python3
"""
optimize_pid_6joint.py
======================
Uses Optuna to find the best PID gains for all 6 joints of the Baloo
hugger robot independently.

WHAT IS TUNED (18 parameters total)
-------------------------------------
For each of the 6 joints: kp, ki, kd

WHAT IS FIXED (6 parameters — safety ceilings, never tuned)
------------------------------------------------------------
  left_j0  correction_max = 150 kPa
  left_j1  correction_max = 150 kPa
  left_j2  correction_max = 75  kPa
  right_j0 correction_max = 150 kPa
  right_j1 correction_max = 150 kPa
  right_j2 correction_max = 48  kPa

USAGE
-----
    python optimize_pid_6joint.py                        # defaults
    python optimize_pid_6joint.py --n_trials 100 --n_eval 250
    python optimize_pid_6joint.py --perturbation 100     # 100N perturbation
"""

import os
import sys
import json
import argparse
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- Adjust these to match your workspace layout ---
sys.path.append('/home/cameronc/baloo_ws/src/baloo-gym/src')
sys.path.append('/home/cameronc/baloo_ws/src')
from evaluate_pid_1000 import run_single_trial


# Joint names — must match the keys your controller expects
JOINT_NAMES = [
    'left_j0', 'left_j1', 'left_j2',
    'right_j0', 'right_j1', 'right_j2',
]

# Fixed correction_max per joint — Optuna never touches these (kPa)
CORRECTION_MAX = {
    'left_j0':  150.0,
    'left_j1':  150.0,
    'left_j2':  75.0,
    'right_j0': 150.0,
    'right_j1': 150.0,
    'right_j2': 48.0,
}

# Search space — same ranges for every joint
PARAM_RANGES = {
    'kp': (10.0, 200.0),
    'ki': ( 0.0,  20.0),
    'kd': ( 0.0,  10.0),
}

# Baseline gains (a known-good single-PID set broadcast to all joints),
# run as trial 0 so you can see the baseline before Optuna explores.
BASELINE_GAINS = {
    joint: {
        'kp': 105.123335, 'ki': 12.854383, 'kd': 4.886744,
    }
    for joint in JOINT_NAMES
}


def build_per_joint_config(optuna_params: dict) -> dict:
    """Convert Optuna's flat param dict into the nested per-joint dict.
    correction_max is always injected from CORRECTION_MAX, never from Optuna.
    """
    config = {}
    for joint in JOINT_NAMES:
        config[joint] = {
            param: optuna_params[f'{joint}_{param}']
            for param in PARAM_RANGES
        }
        config[joint]['correction_max'] = CORRECTION_MAX[joint]
    return config


def make_objective(n_eval: int, perturbation: float):
    def objective(trial: optuna.Trial) -> float:
        # 1. Ask Optuna for 18 values (3 params x 6 joints)
        flat_params = {}
        for joint in JOINT_NAMES:
            for param, (lo, hi) in PARAM_RANGES.items():
                flat_params[f'{joint}_{param}'] = trial.suggest_float(
                    f'{joint}_{param}', lo, hi
                )

        per_joint_config = build_per_joint_config(flat_params)

        # 2. Run n_eval headless episodes
        successes = 0
        halfway = n_eval // 2

        for episode in range(n_eval):
            try:
                result, _, _ = run_single_trial(
                    perturbation_magnitude=perturbation,
                    render_mode=None,
                    record_all_steps=False,
                    per_joint_pid_params=per_joint_config,
                )
                if result['success']:
                    successes += 1
            except Exception as e:
                print(f"  [warn] episode {episode} raised: {e}")

            # Halfway pruning — abandon clearly bad candidates early
            if episode == halfway - 1:
                interim_rate = successes / halfway
                trial.report(interim_rate, step=episode)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

        # 3. Return success rate
        return successes / n_eval

    return objective


def main():
    parser = argparse.ArgumentParser(
        description="Optimise Baloo 6-joint PID gains with Optuna")
    parser.add_argument('--n_trials', type=int, default=100,
                        help='Number of PID candidates Optuna will try')
    parser.add_argument('--n_eval', type=int, default=20,
                        help='Simulation episodes per candidate')
    parser.add_argument('--perturbation', type=float, default=0.0,
                        help='Perturbation magnitude in Newtons')
    parser.add_argument('--output', type=str, default='best_pid_6joint_gains.json',
                        help='Where to save best gains')
    args = parser.parse_args()

    print("=" * 65)
    print("  Baloo 6-Joint PID Optimiser — powered by Optuna")
    print("=" * 65)
    print(f"  Joints tuned       : {', '.join(JOINT_NAMES)}")
    print(f"  Parameters / joint : kp, ki, kd  (3 x 6 = 18)")
    print(f"  Fixed / joint      : correction_max  (6 values, never tuned)")
    print(f"  PID candidates     : {args.n_trials}")
    print(f"  Episodes/candidate : {args.n_eval}")
    print(f"  Total sim episodes : {args.n_trials * args.n_eval:,}")
    print(f"  Perturbation       : {args.perturbation} N")
    print("=" * 65)
    print()
    print("  Fixed correction_max values (kPa):")
    for joint, val in CORRECTION_MAX.items():
        print(f"    {joint:<12} = {val}")
    print()

    # Create study with resumable SQLite storage
    study = optuna.create_study(
        direction='maximize',
        storage='sqlite:///baloo_pid_study.db',
        study_name='baloo_6joint',
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10,
        ),
    )

    # Seed trial 0 with baseline gains — only on a fresh study.
    # On a resume the baseline is already in the DB.
    if len(study.trials) == 0:
        baseline_flat = {}
        for joint in JOINT_NAMES:
            for param in PARAM_RANGES:
                baseline_flat[f'{joint}_{param}'] = BASELINE_GAINS[joint][param]
        study.enqueue_trial(baseline_flat)

    # Progress log after each trial
    def progress_callback(study, trial):
        best = study.best_value * 100
        status = ("PRUNED" if trial.value is None
                  else f"{trial.value * 100:.1f}%")
        print(f"  Trial {trial.number:>3d}/{args.n_trials}  "
              f"this={status:<8}  best={best:.1f}%")

    # Save best gains to JSON after every trial (crash-safe checkpoint)
    def save_best_callback(study, trial):
        try:
            best_config = build_per_joint_config(study.best_params)
            checkpoint = {
                'best_success_rate_pct': round(study.best_value * 100, 2),
                'completed_trials': trial.number + 1,
                'perturbation_N': args.perturbation,
                'fixed_correction_max': CORRECTION_MAX,
                'best_per_joint_config': best_config,
            }
            with open(args.output, 'w') as f:
                json.dump(checkpoint, f, indent=2)
        except ValueError:
            pass  # no completed trial yet

    # Run — with resume-aware trial counting and crash protection
    objective = make_objective(args.n_eval, args.perturbation)

    already_done = len([t for t in study.trials
                        if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = max(0, args.n_trials - already_done)
    if already_done:
        print(f"  Resuming: {already_done} trials already in DB, "
              f"running {remaining} more.\n")

    try:
        study.optimize(objective, n_trials=remaining,
                       callbacks=[progress_callback, save_best_callback])
    except KeyboardInterrupt:
        print("\n  Interrupted — best gains so far are saved to disk.")
    except Exception as e:
        print(f"\n  Run crashed: {e}")
        print("  Best gains so far are still saved to disk.")

    # Final results
    try:
        best_flat = study.best_params
        best_rate = study.best_value * 100
        best_config = build_per_joint_config(best_flat)
    except ValueError:
        print("  No completed trials to report.")
        return

    print()
    print("=" * 65)
    print("  OPTIMISATION COMPLETE")
    print("=" * 65)
    print(f"  Best success rate : {best_rate:.1f}%")
    print()
    print("  Best gains per joint:")
    for joint in JOINT_NAMES:
        cfg = best_config[joint]
        print(f"    {joint:<12}  kp={cfg['kp']:.3f}  ki={cfg['ki']:.3f}  "
              f"kd={cfg['kd']:.3f}  "
              f"correction_max={cfg['correction_max']}")
    print()

    # Final save (full structured output)
    output_data = {
        'best_success_rate_pct': round(best_rate, 2),
        'perturbation_N':        args.perturbation,
        'n_trials':              args.n_trials,
        'n_eval_per_trial':      args.n_eval,
        'fixed_correction_max':  CORRECTION_MAX,
        'best_per_joint_config': best_config,
        'baseline_per_joint_config': {
            joint: {**BASELINE_GAINS[joint],
                    'correction_max': CORRECTION_MAX[joint]}
            for joint in JOINT_NAMES
        },
    }
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"  Results saved to  : {args.output}")


if __name__ == '__main__':
    main()
```

---

## Running it

From the directory where you want the database and results file to live:

```bash
# Defaults (100 trials, 20 episodes each, no perturbation)
python optimize_pid_6joint.py

# A thorough run — 100 candidates, 250 episodes each
python optimize_pid_6joint.py --n_trials 100 --n_eval 250

# Stress-test the gains against a 100 N external disturbance
python optimize_pid_6joint.py --n_trials 100 --n_eval 250 --perturbation 100
```

### Command-line options

| Flag | Default | Meaning |
|---|---|---|
| `--n_trials` | `100` | How many PID candidates to try (total target, resume-aware). |
| `--n_eval` | `20` | Simulated episodes per candidate. More = less noisy, slower. |
| `--perturbation` | `0.0` | External disturbance force in Newtons applied during each episode. |
| `--output` | `best_pid_6joint_gains.json` | Path for the best-gains JSON. |

> **Directory matters.** The SQLite path `sqlite:///baloo_pid_study.db` is **relative to your current working directory**. Always launch (and resume) from the same directory, or the script won't find the existing database.

### Expected output

```
=================================================================
  Baloo 6-Joint PID Optimiser — powered by Optuna
=================================================================
  ...
  Trial   0/100  this=94.0%     best=94.0%
  Trial   1/100  this=90.4%     best=94.0%
  ...
=================================================================
  OPTIMISATION COMPLETE
=================================================================
  Best success rate : 97.2%
  Best gains per joint:
    left_j0       kp=178.727  ki=5.192  kd=9.721  correction_max=150.0
    ...
  Results saved to  : best_pid_6joint_gains.json
```

---

## Crash safety and resuming

This script is built to survive interruption. Two independent layers protect your work:

1. **SQLite study database** (`baloo_pid_study.db`) — every completed trial is written the instant it finishes. This is the durable, resumable record.
2. **JSON checkpoint** (`best_pid_6joint_gains.json`) — rewritten after every trial with the best gains found so far, for easy human/robot consumption.

### To resume an interrupted run

Just run **the exact same command again, from the same directory**:

```bash
python optimize_pid_6joint.py --n_trials 100 --n_eval 250
```

Because of `load_if_exists=True`, Optuna reattaches to the existing study, counts the completed trials, and runs only the remainder. You'll see:

```
  Resuming: 37 trials already in DB, running 63 more.
```

The TPE sampler reuses all prior trials to guide its search, so you lose neither time nor accumulated search intelligence.

### Inspecting results any time (even mid-run)

From a **separate terminal, in the same directory**, you can read the study without disturbing the running optimiser:

```bash
python -c "
import optuna
s = optuna.load_study(study_name='baloo_6joint', storage='sqlite:///baloo_pid_study.db')
print('completed:', len([t for t in s.trials if t.state == optuna.trial.TrialState.COMPLETE]))
print('best so far:', s.best_value)
print('best params:', s.best_params)
"
```

### Starting fresh

The database persists between runs by design. If you change the search space (e.g. altering `PARAM_RANGES`) the old trials become incomparable — start clean by deleting the DB or using a new `study_name`:

```bash
rm -f baloo_pid_study.db
```

---

## Using the results

The JSON contains a ready-to-use nested config under `best_per_joint_config`. Load it and hand it to your controller:

```python
import json

with open('best_pid_6joint_gains.json') as f:
    data = json.load(f)

per_joint_pid_params = data['best_per_joint_config']
# -> {'left_j0': {'kp': ..., 'ki': ..., 'kd': ..., 'correction_max': 150.0}, ...}
```

> **Remember the thresholds.** This optimiser outputs `kp`, `ki`, `kd`, and `correction_max` only. Your controller also needs a per-joint slip `threshold`, which you set and maintain by hand. When you copy new gains into your controller, update `kp/ki/kd` and leave your hand-tuned thresholds in place.

---

## Optional: tuning threshold too

If you *do* want Optuna to tune the slip threshold, add it to `PARAM_RANGES` with a sensible range for your slip-velocity units:

```python
PARAM_RANGES = {
    'kp':        (10.0, 200.0),
    'ki':        ( 0.0,  20.0),
    'kd':        ( 0.0,  10.0),
    'threshold': ( 0.0,   0.1),   # tune to your slip-velocity scale
}
```

Because `build_per_joint_config` iterates over `PARAM_RANGES`, threshold then flows automatically into the config, the printouts, and the JSON — no other changes needed. Add `threshold` to `BASELINE_GAINS` as well so trial 0 stays valid, and delete the existing `baloo_pid_study.db` before running, since the search space has changed.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'evaluate_pid_1000'` | The sim path isn't on `sys.path`. | Fix the `sys.path.append(...)` lines to point at your workspace. |
| `KeyError: 'threshold'` in the print/save block | Print/save references `threshold` but it isn't tuned. | Remove the `threshold=...` from the print string, or add `threshold` to `PARAM_RANGES`. |
| Resume starts from zero instead of continuing | Launched from a different directory, so the relative DB path missed the file. | Always run from the same directory as `baloo_pid_study.db`. |
| `DuplicatedStudyError` | `load_if_exists` missing/false with an existing study. | Keep `load_if_exists=True`, or use a new `study_name`, or delete the DB. |
| Every resume re-runs the baseline | Baseline enqueue isn't guarded. | Wrap the enqueue in `if len(study.trials) == 0:`. |
| `free variable 'json' referenced before assignment` | A local `import json` inside `main()` shadows the module-level import. | Remove the redundant local `import json`; keep only the top-of-file one. |

---

## License

Add your preferred license here (e.g. MIT).
