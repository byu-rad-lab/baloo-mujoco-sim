"""
collect_pressure_trial_73.py

Runs Trial_73 over 250 LHS boxes and records max pressure
per joint per trial. Saves to JSONL for histogram plotting.

Same as collect_pressure_rl_models.py but scoped to just Trial_73.

Usage:
    python collect_pressure_trial_73.py
    python collect_pressure_trial_73.py --n_boxes 250
"""

import os, ast, json, argparse
import numpy as np
from scipy.stats import qmc
from stable_baselines3 import PPO
from baloo_gym.envs.baloo_v9 import BalooV9
from baloo_gym.wrappers.three_part_reward_wrapper import ThreePartRewardWrapper

# ── Models ────────────────────────────────────────────────────────────────
MODELS = {
    "Trial_73": "/home/randonsandall/baloo-gym/pressure/Trial_73.zip",
}

JOINT_LABELS  = ["left_j0", "left_j1", "left_j2", "right_j0", "right_j1", "right_j2"]
CTRL_TIMESTEP = 0.05
MAX_STEPS     = 1200


# ── LHS sampling ─────────────────────────────────────────────────────────
def load_or_generate_lhs_samples(N, seed=42):
    fname = f"{N}_lhs_samples.txt"
    if os.path.exists(fname):
        with open(fname) as f:
            return [ast.literal_eval(line.strip()) for line in f]
    sampler = qmc.LatinHypercube(d=6, seed=seed)
    lhs = sampler.random(n=N)
    samples = []
    for i in range(N):
        samples.append((
            np.interp(lhs[i,0],[0,1],[0.2,0.6]),
            np.interp(lhs[i,1],[0,1],[0.2,0.6]),
            np.interp(lhs[i,2],[0,1],[0.5,1.25]),
            np.interp(lhs[i,3],[0,1],[0.5,10.0]),
            np.interp(lhs[i,4],[0,1],[-0.1,0.1]),
            np.interp(lhs[i,5],[0,1],[-np.pi/3,np.pi/3]),
        ))
    with open(fname,"w") as f:
        for s in samples: f.write(f"{s}\n")
    return samples


# ── Pressure extraction ───────────────────────────────────────────────────
def pressures_from_action(action):
    a = np.clip(action, -1, 1)
    deltas = a[1:] * 150
    result = {}
    for j_idx, label in enumerate(JOINT_LABELS):
        idx = j_idx * 2
        d0, d1 = deltas[idx], deltas[idx+1]
        result[label] = [150+d0, 150-d0, 150+d1, 150-d1]
    return result

def max_pressure_per_joint(action):
    p = pressures_from_action(action)
    return {label: float(np.max(vals)) for label, vals in p.items()}

def aggregate_max(step_maxes):
    if not step_maxes:
        return {label: 0.0 for label in JOINT_LABELS}
    return {label: max(d[label] for d in step_maxes) for label in JOINT_LABELS}


# ── Single trial runner ───────────────────────────────────────────────────
def run_trial(policy, object_size, object_mass, object_xpos, object_zrotation):
    env = BalooV9(
        render_mode=None, camera_name='frontcam', ctrl_timestep=CTRL_TIMESTEP,
        randomize_initial_height=False, randomize_object_size=False,
        randomize_object_mass=False,
        object_size=list(object_size), object_mass=object_mass,
        object_xpos=object_xpos, object_zrotation=object_zrotation,
    )
    env = ThreePartRewardWrapper(env, reward_selection=['dont_drop'])
    obs, _ = env.reset()

    step_maxes = []
    done = False
    step = 0
    while not done and step < MAX_STEPS:
        action, _ = policy.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        step_maxes.append(max_pressure_per_joint(action))
        done = terminated or truncated
        step += 1

    env.close()
    return {
        "success":    info.get("is_success", False),
        "tip":        info.get("box_fell_over", False),
        "steps":      step,
        "joint_maxes": aggregate_max(step_maxes),
    }


# ── Main eval loop ────────────────────────────────────────────────────────
def run_eval(n_boxes=250):
    # use same fixed 250 indices as Optuna eval
    all_boxes = load_or_generate_lhs_samples(1000, seed=42)
    idxs = np.random.default_rng(42).choice(len(all_boxes), size=n_boxes, replace=False)
    boxes = [all_boxes[i] for i in idxs]

    print(f"Loading models...")
    policies = {name: PPO.load(path) for name, path in MODELS.items()}
    print(f"  Loaded {len(policies)} models")

    for name, policy in policies.items():
        out_file = f"pressure_data_{name}.jsonl"
        print(f"\n{'='*50}")
        print(f"  Model: {name}  ({n_boxes} boxes)")
        print(f"{'='*50}")

        for i, (x, y, z, mass, xpos, rot) in enumerate(boxes):
            print(f"  [{i+1:4d}/{n_boxes}] size=({x:.2f},{y:.2f},{z:.2f}) mass={mass:.1f}kg", end="", flush=True)

            result = run_trial(policy, (x,y,z), mass, xpos, rot)
            jm = result["joint_maxes"]
            avg = np.mean(list(jm.values()))
            print(f"  success={result['success']}  avg_max_p={avg:.1f}")

            result.update({"box_idx": int(idxs[i]), "xsize": x, "ysize": y,
                           "zsize": z, "mass": mass, "xpos": xpos, "rotation": rot})
            with open(out_file, "a") as f:
                json.dump(result, f)
                f.write("\n")

        # print summary
        results = []
        with open(out_file) as f:
            for line in f:
                if line.strip(): results.append(json.loads(line))

        print(f"\n── {name} summary ({len(results)} trials) ──")
        print(f"  Success: {sum(r['success'] for r in results)/len(results)*100:.1f}%")
        print(f"  {'Joint':<12} {'p50':>8} {'p95':>8} {'p99':>8} {'max':>8}")
        for label in JOINT_LABELS:
            vals = [r["joint_maxes"][label] for r in results]
            print(f"  {label:<12} {np.percentile(vals,50):>8.1f} "
                  f"{np.percentile(vals,95):>8.1f} "
                  f"{np.percentile(vals,99):>8.1f} "
                  f"{np.max(vals):>8.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_boxes", type=int, default=250)
    args = parser.parse_args()
    run_eval(n_boxes=args.n_boxes)
