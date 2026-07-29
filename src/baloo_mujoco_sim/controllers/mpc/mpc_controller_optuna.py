"""
MPC/MPPI controller for Baloo.

Cost function matches Curtis's guided RL reward exactly (eq. 9.7 in paper):
    r[t] = r_task + r_guide
    r_task  = +10 if box lifted 0.5m above INITIAL height, -2 if tipped
    r_guide = 0.1 * exp(-0.5 * ||a - a_nom||^2)

Box sampling uses same LHS ranges as Curtis (Table 9.1):
    mass:   0.5 - 10 kg
    xsize:  0.2 - 0.6 m
    ysize:  0.2 - 0.6 m
    zsize:  0.5 - 1.25 m
    xpos:   -0.1 - 0.1 m
    rotation: -pi/3 - pi/3 rad
"""

import numpy as np
import mujoco
import time
import json
import os
import ast
from multiprocessing import Pool
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from scipy.stats import qmc

from baloo_gym.envs.baloo_v9 import BalooV9
from baloo_gym.wrappers.three_part_reward_wrapper import ThreePartRewardWrapper
from baloo_gym.policies.open_loop_hugger import OpenLoopHuggerPolicy
from baloo_mujoco_sim.utils.baloo_mj_api import (
    get_box_position, get_box_quat,
    get_link_position,
    set_elevator_cmd, set_joint_pressure_commands
)
import baloo_mujoco_sim

# --- Hyperparameters (matching Curtis's reward weights) ---
N_SAMPLES     = 32
HORIZON       = 8
SIGMA         = 0.15
N_OPEN_LOOP   = 50
W_GUIDE       = 0.1       # matches Curtis eq. 9.6 (was 0.01)
R_LIFT        = 0      # matches Curtis eq. 9.5
R_TIP         = 0      # matches Curtis eq. 9.5
CTRL_TIMESTEP = 0.05
N_WORKERS     = 16
LAMBDA_TEMP   = 1.0
W_HEIGHT      = 0.0       # continuous height reward weight (0 = disabled, matches Curtis)
MAX_STEPS     = 1200      # matches Curtis 60s at 20Hz

_worker_model    = None
_initial_box_z   = None   # set per episode, used for relative lift threshold

# ── p99 pressure caps (mbar), from Trial_73 1000-trial pressure collection ──
JOINT_LABELS = ["left_j0", "left_j1", "left_j2", "right_j0", "right_j1", "right_j2"]
PRESSURE_CAPS = {
    "left_j0":  300.0,
    "left_j1":  300.0,
    "left_j2":  225.0,
    "right_j0": 300.0,
    "right_j1": 300.0,
    "right_j2": 198.0,
}

def cap_action(a):
    """Clip normalized action so resulting pressures stay within PRESSURE_CAPS."""
    a = np.clip(a.copy(), -1, 1)
    for j_idx, label in enumerate(JOINT_LABELS):
        p_max = PRESSURE_CAPS[label]
        max_delta = (p_max - 150) / 150
        idx = 1 + j_idx * 2
        a[idx]     = np.clip(a[idx],     -max_delta, max_delta)
        a[idx + 1] = np.clip(a[idx + 1], -max_delta, max_delta)
    return a


def worker_init(xml_path):
    global _worker_model
    _worker_model = mujoco.MjModel.from_xml_path(xml_path)


def box_tipped(model, data):
    quat  = get_box_quat(model, data)
    rot   = R.from_quat(np.roll(quat, -1))
    box_z = rot.apply([0, 0, 1])
    return np.degrees(np.arccos(np.clip(np.dot(box_z, [0, 0, 1]), -1, 1))) > 80


def box_lifted(model, data, initial_z):
    """Matches Curtis: lifted 0.5m above INITIAL height (not absolute 0.5m)."""
    return get_box_position(model, data)[2] > initial_z + 0.5


def guide_cost(action, nominal_action):
    """Matches Curtis eq. 9.6: r_guide = 0.1 * exp(-0.5 * ||a - a_nom||^2)"""
    diff = action - nominal_action
    return -W_GUIDE * np.exp(-0.5 * np.dot(diff, diff))


def task_reward(model, data, initial_z, w_height=0.0):
    """Matches Curtis eq. 9.5 plus optional continuous height reward."""
    box_z = get_box_position(model, data)[2]
    r = w_height * (box_z - initial_z)
    if box_tipped(model, data):            r += R_TIP
    if box_lifted(model, data, initial_z): r += R_LIFT
    return r


def apply_normalized_action(model, data, a):
    a = np.clip(a, -1, 1)
    pressure_deltas = a[1:] * 150
    set_elevator_cmd(model, data, (a[0] + 1) / 2 * (-900))
    for side_idx, side in enumerate(['left', 'right']):
        for j in range(3):
            idx = side_idx * 6 + j * 2
            set_joint_pressure_commands(model, data, side, j, np.array([
                150 + pressure_deltas[idx],
                150 - pressure_deltas[idx],
                150 + pressure_deltas[idx + 1],
                150 - pressure_deltas[idx + 1],
            ]))


def rollout_worker(args):
    global _worker_model
    data_state, action_seq, nominal_seq, initial_z, w_height = args
    data = mujoco.MjData(_worker_model)
    mujoco.mj_setState(_worker_model, data, data_state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    total_cost = 0.0
    for t in range(len(action_seq)):
        apply_normalized_action(_worker_model, data, action_seq[t])
        mujoco.mj_step(_worker_model, data)
        total_cost += guide_cost(np.clip(action_seq[t], -1, 1), nominal_seq[t])
    total_cost -= task_reward(_worker_model, data, initial_z, w_height)
    return total_cost


# --- LHS box sampling (matches Curtis Table 9.1 / policy_evaluator.py) ---

def sample_lhs(N, seed=42):
    sampler = qmc.LatinHypercube(d=6, seed=seed)
    lhs = sampler.random(n=N)
    x    = np.interp(lhs[:, 0], [0, 1], [0.2, 0.6])
    y    = np.interp(lhs[:, 1], [0, 1], [0.2, 0.6])
    z    = np.interp(lhs[:, 2], [0, 1], [0.5, 1.25])
    mass = np.interp(lhs[:, 3], [0, 1], [0.5, 10.0])
    xpos = np.interp(lhs[:, 4], [0, 1], [-0.1, 0.1])
    rot  = np.interp(lhs[:, 5], [0, 1], [-np.pi / 3, np.pi / 3])
    return [(x[i], y[i], z[i], mass[i], xpos[i], rot[i]) for i in range(N)]


def load_or_generate_lhs_samples(N, seed=42):
    fname = f"{N}_lhs_samples.txt"
    if os.path.exists(fname):
        print(f"Loading LHS samples from {fname}")
        with open(fname) as f:
            return [ast.literal_eval(line.strip()) for line in f]
    print(f"Generating {N} LHS samples (seed={seed})")
    samples = sample_lhs(N, seed)
    with open(fname, "w") as f:
        for s in samples:
            f.write(f"{s}\n")
    return samples


# --- Single episode runner ---

def run_episode(object_size, object_mass, object_xpos=0.0, object_zrotation=0.0,
                render=False, n_samples=N_SAMPLES, horizon=HORIZON,
                sigma=SIGMA, w_guide=W_GUIDE, lambda_temp=LAMBDA_TEMP, w_height=W_HEIGHT):
    """
    Run one MPC episode. Returns dict with success, tip, slip, steps.
    Suitable for Optuna objective or standalone evaluation.
    """
    xml_path = str(baloo_mujoco_sim.XML_PATH)

    env = BalooV9(
        render_mode='human' if render else None,
        camera_name='frontcam',
        ctrl_timestep=CTRL_TIMESTEP,
        render_width=640,
        render_height=480,
        randomize_initial_height=False,
        randomize_object_size=False,
        randomize_object_mass=False,
        object_size=list(object_size),
        object_mass=object_mass,
    )
    env = ThreePartRewardWrapper(env, reward_selection=['dont_drop'])
    obs, _ = env.reset()

    # store initial box height for relative lift threshold (matches Curtis)
    initial_z = get_box_position(env.unwrapped.model, env.unwrapped.data)[2]

    # generate nominal actions from open loop hugger
    nom_policy = OpenLoopHuggerPolicy(N=N_OPEN_LOOP)
    nom_env = BalooV9(
        render_mode=None, camera_name='frontcam', ctrl_timestep=CTRL_TIMESTEP,
        randomize_initial_height=False, randomize_object_size=False,
        randomize_object_mass=False,
        object_size=list(object_size), object_mass=object_mass,
    )
    nom_obs, _ = nom_env.reset()
    nominal_actions = []
    for _ in range(MAX_STEPS):
        a, _ = nom_policy.predict(nom_obs)
        nominal_actions.append(a.copy())
        nom_obs, _, t, tr, _ = nom_env.step(a)
    nom_env.close()
    nominal_actions = np.array(nominal_actions)

    # Run open loop until LIFT state — matches mpc_controller_no_skip.py
    ol_policy    = OpenLoopHuggerPolicy(N=N_OPEN_LOOP)
    skip_to_step = 0
    while ol_policy.state != "LIFT" and skip_to_step < MAX_STEPS:
        a, _ = ol_policy.predict(obs)
        obs, _, terminated, truncated, info = env.step(a)
        skip_to_step += 1
        if terminated or truncated:
            env.close()
            slip = not (info.get("is_success", False) or info.get("box_fell_over", False))
            return {"success": info.get("is_success", False),
                    "tip": info.get("box_fell_over", False),
                    "slip": slip, "steps": skip_to_step}

    done          = False
    step          = skip_to_step
    rng           = np.random.default_rng()
    success       = False
    tipped        = False
    rollout_times = []

    with Pool(N_WORKERS, initializer=worker_init, initargs=(xml_path,)) as pool:
        while not done and step < MAX_STEPS:
            model = env.unwrapped.model
            data  = env.unwrapped.data

            state_size = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
            data_state = np.zeros(state_size)
            mujoco.mj_getState(model, data, data_state, mujoco.mjtState.mjSTATE_FULLPHYSICS)

            t_end      = min(step + horizon, len(nominal_actions))
            nom_window = nominal_actions[step:t_end]

            perturbations = rng.normal(0, sigma, size=(n_samples, len(nom_window), 13))
            candidates    = nom_window[None] + perturbations

            args  = [(data_state, candidates[i], nom_window, initial_z, w_height) for i in range(n_samples)]

            t_rollout_start = time.time()
            costs = np.array(pool.map(rollout_worker, args))
            t_rollout = time.time() - t_rollout_start
            rollout_times.append(t_rollout)

            weights = np.exp(-(costs - costs.min()) / lambda_temp)
            weights /= weights.sum()
            best_action = cap_action(np.sum(weights[:, None] * candidates[:, 0, :], axis=0))

            obs, reward, terminated, truncated, info = env.step(best_action)
            done    = terminated or truncated
            success = info.get("is_success", False)
            tipped  = info.get("box_fell_over", False)
            step   += 1

            if step % 50 == 0:
                elapsed = time.time() - t_rollout_start
                print(f"  step {step:4d} | rollout={t_rollout*1000:.1f}ms | "
                      f"cost={costs.min():.3f} | "
                      f"box_z={get_box_position(model,data)[2]:.3f}")

    env.close()
    slip = not (success or tipped)
    avg_rollout_ms = float(np.mean(rollout_times) * 1000) if rollout_times else 0.0
    return {"success": success, "tip": tipped, "slip": slip, "steps": step,
            "avg_rollout_ms": avg_rollout_ms}


# --- Optuna tuning ---

def run_optuna(n_trials=100, n_eval_boxes=100):
    """
    Run Optuna hyperparameter search over MPC parameters.
    All trials evaluate on the SAME fixed set of n_eval_boxes boxes for fair comparison.
    """
    import optuna
    from tqdm import tqdm
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    boxes = load_or_generate_lhs_samples(1000, seed=42)

    # fixed box indices — same for every trial so comparisons are apples to apples
    fixed_idxs = np.random.default_rng(42).choice(len(boxes), size=n_eval_boxes, replace=False)
    print(f"Using {n_eval_boxes} fixed boxes for all trials (seed=42)")

    pbar  = tqdm(total=n_trials, desc="Optuna trials", unit="trial")

    def objective(trial):
        horizon      = trial.suggest_int  ('horizon',      4,    20)
        n_samples    = trial.suggest_int  ('n_samples',    8,    64)
        sigma        = trial.suggest_float('sigma',        0.05, 3.0)
        w_guide      = trial.suggest_float('w_guide',      0.01, 0.5)
        lambda_temp  = trial.suggest_float('lambda_temp',  0.1,  5.0)
        w_height     = trial.suggest_float('w_height',     0.0,  200.0)

        # same boxes every trial for fair comparison
        idxs = fixed_idxs

        successes    = 0
        trial_results = []
        for i in idxs:
            x, y, z, mass, xpos, rot = boxes[i]
            try:
                result = run_episode(
                    object_size=(x, y, z),
                    object_mass=mass,
                    object_xpos=xpos,
                    object_zrotation=rot,
                    n_samples=n_samples,
                    horizon=horizon,
                    sigma=sigma,
                    w_guide=w_guide,
                    lambda_temp=lambda_temp,
                    w_height=w_height,
                )
                trial_results.append(result)
                if result["success"]:
                    successes += 1
            except Exception as e:
                print(f"  Trial failed on box {i}: {e}")

        rate = successes / n_eval_boxes
        try:
            best_so_far = study.best_value
        except ValueError:
            best_so_far = 0.0
        pbar.set_postfix({
            "success": f"{rate:.2f}",
            "best":    f"{best_so_far:.2f}",
            "horizon": horizon,
            "samples": n_samples,
        })
        pbar.update(1)
        avg_ms = np.mean([r.get("avg_rollout_ms", 0) for r in trial_results]) if trial_results else 0.0
        trial.set_user_attr("avg_rollout_ms", round(avg_ms, 2))
        trial.set_user_attr("n_successes", successes)
        print(f"\nTrial {trial.number}: horizon={horizon} n_samples={n_samples} "
              f"sigma={sigma:.3f} w_guide={w_guide:.3f} "
              f"lambda={lambda_temp:.2f} w_height={w_height:.1f} "
              f"avg_rollout={avg_ms:.1f}ms -> success={rate:.2f}")
        return rate

    study = optuna.create_study(direction='maximize',
                                study_name='mpc_baloo',
                                storage='sqlite:///mpc_optuna.db',
                                load_if_exists=True)
    study.optimize(objective, n_trials=n_trials)
    pbar.close()

    print("\n=== Best hyperparameters ===")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print(f"  best success rate: {study.best_value:.3f}")

    with open("optuna_best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=2)
    print("Saved to optuna_best_params.json")
    return study.best_params


# --- Full 1000-box evaluation (matches Curtis's eval protocol) ---

def run_full_eval(n_boxes=1000, out_file="mpc_eval_results.jsonl"):
    """
    Evaluate MPC on 1000 LHS boxes using same protocol as Curtis.
    Results saved as JSONL, one trial per line.
    """
    boxes = load_or_generate_lhs_samples(n_boxes, seed=42)
    results = []
    for i, (x, y, z, mass, xpos, rot) in enumerate(boxes):
        print(f"[{i+1}/{n_boxes}] size=({x:.2f},{y:.2f},{z:.2f}) mass={mass:.1f}kg")
        result = run_episode(
            object_size=(x, y, z),
            object_mass=mass,
            object_xpos=xpos,
            object_zrotation=rot,
        )
        result.update({"xsize": x, "ysize": y, "zsize": z,
                       "mass": mass, "xpos": xpos, "rotation": rot})
        results.append(result)
        with open(out_file, "a") as f:
            json.dump(result, f)
            f.write("\n")

    total     = len(results)
    successes = sum(r["success"] for r in results)
    tips      = sum(r["tip"]     for r in results)
    slips     = sum(r["slip"]    for r in results)
    print(f"\n=== Results ({total} boxes) ===")
    print(f"  Success: {successes/total*100:.1f}%")
    print(f"  Tip:     {tips/total*100:.1f}%")
    print(f"  Slip:    {slips/total*100:.1f}%")
    return results


# --- Single demo run ---

def run_mpc():
    """Quick single demo with render."""
    result = run_episode(
        object_size=(0.3, 0.3, 0.6),
        object_mass=5.0,
        render=True,
    )
    print(f"\nResult: {result}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['demo', 'optuna', 'eval'], default='demo')
    parser.add_argument('--trials',    type=int, default=100)
    parser.add_argument('--eval_boxes',type=int, default=100)
    parser.add_argument('--n_boxes',   type=int, default=1000)
    args = parser.parse_args()

    if args.mode == 'demo':
        run_mpc()
    elif args.mode == 'optuna':
        run_optuna(n_trials=args.trials, n_eval_boxes=args.eval_boxes)
    elif args.mode == 'eval':
        run_full_eval(n_boxes=args.n_boxes)

